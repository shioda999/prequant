import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union


class CrossLayerAECompressor(nn.Module):
    """
    GPU-aware cross-layer autoencoder compressor.

    Usage:
        model, stacked = CrossLayerAECompressor.from_weights(
            weights_list,
            bottleneck_dim=128,
            train_steps=1000,
            lr=1e-3,
            verbose=True,
        )
    After from_weights returns, `model` is already trained (trained inside __init__).
    - model: the trained CrossLayerAECompressor instance
    - stacked: the stacked original tensor (L, O, I) on the same device

    Public methods:
    - forward(x) -> reconstruction of x (expects x shape (L, O, I) or (L, O*I) flattened)
    - compress(x, topk=None) -> returns {'bottlenecks': tensor(L, bottleneck_dim)}
    - reconstruct_from_bottlenecks(z) -> recon (L,O,I)
    - export_reconstructed_weights(x) -> recon (same shape as x)
    - save_state(path) / load_state(path)
    """

    def __init__(
        self,
        stacked_weights: torch.Tensor,
        bottleneck_dim: int = 64,
        steps: int = 1000,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: Optional[int] = None,
        l1_reg: float = 0.0,
        verbose: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
    ):
        """
        stacked_weights: torch.Tensor of shape (L, O, I)
        This constructor will:
          - infer device from stacked_weights unless device is provided
          - build autoencoder (encoder: Linear(flat->bottleneck), GELU; decoder: Linear)
          - run an internal training loop for `train_steps`
        """
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        assert stacked_weights.ndim == 3, "stacked_weights must be shape (L, O, I)"
        self.device = torch.device(device) if device is not None else stacked_weights.device
        # move input to device (and keep a detached copy for reference)
        self.original = stacked_weights.to(self.device)
        L, O, I = self.original.shape
        self.L = L
        self.out_dim = O
        self.in_dim = I
        self.flat_dim = O * I
        self.bottleneck_dim = bottleneck_dim

        # Build encoder/decoder
        # Simple non-linear AE: flatten -> bottleneck -> reconstruct
        self.encoder = nn.Sequential(
            nn.Linear(self.flat_dim, bottleneck_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, self.flat_dim),
        )

        # Move model to device
        self.to(self.device)

        # training params
        self.train_steps = int(steps)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = batch_size if batch_size is not None else L  # default: whole-layer batch
        self.l1_reg = float(l1_reg)
        self.verbose = bool(verbose)

        # run training loop inside __init__
        self._train_internal()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (L, O, I) tensor or (N, flat_dim) where N==L
        returns recon: same shape as x
        """
        if x.ndim == 3:
            L, O, I = x.shape
            assert O == self.out_dim and I == self.in_dim
            flat = x.reshape(L, self.flat_dim).to(self.device)
        elif x.ndim == 2:
            flat = x.to(self.device)
            assert flat.shape[1] == self.flat_dim
        else:
            raise ValueError("x must be (L,O,I) or (L, flat_dim)")

        z = self.encoder(flat.float())  # (N, bottleneck)
        recon_flat = self.decoder(z)  # (N, flat_dim)
        recon = recon_flat.view(-1, self.out_dim, self.in_dim)
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return bottleneck representations (L, bottleneck_dim)
        """
        if x.ndim == 3:
            flat = x.reshape(x.shape[0], self.flat_dim).to(self.device)
        else:
            flat = x.to(self.device)
        with torch.no_grad():
            z = self.encoder(flat)
        return z

    def reconstruct_from_bottlenecks(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (L, bottleneck_dim) -> recon (L, O, I)
        """
        if z.ndim != 2 or z.shape[1] != self.bottleneck_dim:
            raise ValueError("z must be shape (L, bottleneck_dim)")
        with torch.no_grad():
            recon_flat = self.decoder(z.to(self.device))
            recon = recon_flat.view(-1, self.out_dim, self.in_dim)
        return recon

    def compress(self, x: Optional[torch.Tensor] = None, topk: Optional[int] = None) -> dict:
        """
        Return compressed representation. By default compresses the original stacked input.
        If topk is provided, performs topk-sparsification on bottleneck by zeroing small values per-row.
        Returns dict: {'bottlenecks': Tensor(L,bn), 'topk': topk}
        """
        if x is None:
            x = self.original
        z = self.encode(x)  # (L, bottleneck_dim)

        if topk is not None:
            # sparsify per-row by keeping topk absolute values
            flat = z.detach().cpu()
            vals, idx = torch.topk(flat.abs(), topk, dim=1)
            mask = torch.zeros_like(flat)
            mask.scatter_(1, idx, 1.0)
            z_sparse = flat * mask
            return {'bottlenecks': z_sparse, 'topk': topk}
        else:
            return {'bottlenecks': z.detach().cpu(), 'topk': None}

    def export_reconstructed_weights(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reconstruct and return weights for x (default original).
        Returned tensor on CPU.
        """
        if x is None:
            x = self.original
        recon = self.forward(x)  # (L,O,I) on device
        return recon.detach().cpu()

    def save_state(self, path: str):
        """Save model state_dict to path"""
        torch.save(self.state_dict(), path)

    def load_state(self, path: str, map_location: Optional[Union[str, torch.device]] = None):
        """Load state_dict"""
        sd = torch.load(path, map_location=map_location)
        self.load_state_dict(sd)

    # ---------------------------
    # Internal training utilities
    # ---------------------------
    def _train_internal(self):
        """
        Train encoder/decoder to reconstruct self.original.
        Uses MSE loss + optional L1 on bottleneck.
        Training runs for self.train_steps steps.
        """
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        dataset = self.original.view(self.L, self.flat_dim)  # (L, flat_dim)
        # If batch_size < L, we do simple random mini-batches
        bs = int(min(self.batch_size, self.L))
        steps = self.train_steps

        # Simple scheduler (optional) - here constant lr
        for step in range(steps):
            if bs == self.L:
                batch = dataset  # full batch
            else:
                idx = torch.randint(0, self.L, (bs,), device=self.device)
                batch = dataset[idx]
            batch = batch.float().detach()
            recon_flat = self.decoder(self.encoder(batch))
            mse = F.mse_loss(recon_flat, batch)
            if self.l1_reg > 0:
                z = self.encoder(batch)
                l1 = z.abs().mean()
                loss = mse + self.l1_reg * l1
            else:
                loss = mse

            opt.zero_grad()
            loss.backward()
            opt.step()

            # optionally normalize weights of encoder/decoder for stability (not strictly needed)
            if (step % 200 == 0) and self.verbose:
                print(f"[Compressor train] step {step}/{steps} mse={mse.item():.6e}")

        if self.verbose:
            # final report
            with torch.no_grad():
                recon_flat = self.decoder(self.encoder(dataset))
                final_mse = F.mse_loss(recon_flat, dataset).item()
            print(f"[Compressor train] finished {steps} steps final_mse={final_mse:.6e}")

        self.eval()

    # ---------------------------
    # Classmethod constructor
    # ---------------------------
    @classmethod
    def from_weights(
        cls,
        stacked_weights: List[torch.Tensor],
        bottleneck_dim: int = 64,
        steps: int = 1000,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: Optional[int] = None,
        l1_reg: float = 0.0,
        verbose: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
    ) -> Tuple["CrossLayerAECompressor", torch.Tensor]:
        """
        Build compressor from a list of weight tensors.
        - Each element of `weights` must be a tensor of identical shape (O, I).
        - Returns: (trained_model, stacked_tensor)
        """
        # verify shapes
        dev = device if device is not None else stacked_weights.device
        # instantiate and train inside __init__
        comp = cls(
            stacked_weights=stacked_weights,
            bottleneck_dim=bottleneck_dim,
            steps=steps,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            l1_reg=l1_reg,
            verbose=verbose,
            device=dev,
            seed=seed,
        )
        return comp, stacked

# ---------------------
# Minimal runnable demo
# ---------------------
if __name__ == "__main__":
    # Demo with random weights
    # Suppose we have 12 layers, each weight is (512, 512)
    L, O, I = 12, 512, 512
    # Put weights on GPU if available to exercise GPU-aware behavior
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = [torch.randn(O, I, device=device) for _ in range(L)]

    # Build and train compressor in one call
    comp, stacked = CrossLayerAECompressor.from_weights(
        weights,
        bottleneck_dim=128,
        train_steps=1000,
        lr=1e-3,
        batch_size=None,  # full-batch
        l1_reg=1e-4,
        verbose=True,
        device=device,
        seed=42,
    )

    # Compress (get bottlenecks)
    compressed = comp.compress(topk=None)  # {'bottlenecks': Tensor(L,bn)}
    print("Compressed bottlenecks shape:", compressed["bottlenecks"].shape)

    # Export reconstructed weights (CPU)
    recon = comp.export_reconstructed_weights()
    print("Reconstructed weights shape:", recon.shape)
