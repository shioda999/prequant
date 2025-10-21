import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


class LoRAStackCompressor(nn.Module):
    def __init__(self, num_matrices: int, d_out: int, d_in: int, r: int = 256, device: Optional[str] = None):
        super().__init__()
        self.num_matrices = num_matrices
        self.d_out = d_out
        self.d_in = d_in
        self.r = r
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Shared base matrix B
        self.B = nn.Parameter(torch.zeros(d_out, d_in, device=self.device))

        # LoRA factors per matrix: A: (num_matrices, d_out, r), C: (num_matrices, r, d_in)
        # We register them as Parameters with shapes (num_matrices, d_out, r) etc.
        self.A = nn.Parameter(torch.randn(num_matrices, d_out, r, device=self.device) * 1e-3)
        self.C = nn.Parameter(torch.randn(num_matrices, r, d_in, device=self.device) * 1e-3)

        # Diagonal for each matrix: length = min(d_out, d_in)
        self.diag_len = min(d_out, d_in)
        self.diag = nn.Parameter(torch.zeros(num_matrices, self.diag_len, device=self.device))

    def forward(self) -> torch.Tensor:
        # Reconstruct full weight stack: shape (num_matrices, d_out, d_in)
        # Compute A @ C for each matrix using batch matmul
        # A: (N, d_out, r), C: (N, r, d_in) -> delta: (N, d_out, d_in)
        delta = torch.matmul(self.A, self.C)
        base = self.B.unsqueeze(0).expand(self.num_matrices, -1, -1)
        W_rec = base + delta

        # Overwrite diagonal entries with diag parameters
        if self.diag_len > 0:
            # create indices
            idx = torch.arange(self.diag_len, device=self.device)
            W_rec[:, idx, idx] = self.diag
        return W_rec

    def mse_to(self, W_target: torch.Tensor) -> torch.Tensor:
        # W_target: (N, d_out, d_in)
        W_rec = self.forward()
        return torch.mean((W_rec - W_target) ** 2)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: Optional[str] = None):
        st = torch.load(path, map_location=map_location)
        self.load_state_dict(st)

    @classmethod
    def from_weights(cls, W_stack: torch.Tensor, r: int = 256, steps: int = 2000, lr: float = 1e-3,
                     weight_decay: float = 0.0, verbose: bool = True, device: Optional[str] = None):
        """
        Create a LoRAStackCompressor and train it (via gradient descent) to approximate W_stack.
        - W_stack: torch.Tensor of shape (N, d_out, d_in)
        - r: LoRA rank (default 256)
        - steps: optimization steps to fit the compressor
        - lr: learning rate for optimizer
        Returns a trained LoRAStackCompressor instance.
        """
        assert W_stack.ndim == 3, f"W_stack must be shape (N, d_out, d_in), but W_stack.shape = {str(W_stack.shape)}"
        N, d_out, d_in = W_stack.shape
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        W = W_stack.to(device)

        # Initialize module
        model = cls(N, d_out, d_in, r=r, device=device)

        # Initialize shared base B as mean of targets (good starting point)
        with torch.no_grad():
            model.B.copy_(W.mean(dim=0))

            # Initialize diag parameters to the diagonal of each weight
            if model.diag_len > 0:
                idx = torch.arange(model.diag_len, device=device)
                model.diag.copy_(W[:, idx, idx])

        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Simple training loop: minimize MSE(W_rec, W)
        for step in range(steps):
            opt.zero_grad()
            loss = model.mse_to(W)
            loss.backward()
            opt.step()
            if verbose and (step % max(1, steps // 10) == 0 or step == steps - 1):
                print(f"step {step+1}/{steps} loss={loss.item():.6e}")
        return model


if __name__ == "__main__":
    # Example usage
    torch.manual_seed(0)
    N = 8
    d_out = 128
    d_in = 64

    # create random weight stack
    W_stack = torch.randn(N, d_out, d_in)

    # Fit compressor with rank r=32 for demo
    r = 32
    steps = 5000
    model = LoRAStackCompressor.from_weights(W_stack, r=r, steps=steps, lr=1e-2, verbose=True)

    # Reconstruct and measure final error
    W_rec = model()
    mse = torch.mean((W_rec.cpu() - W_stack) ** 2).item()
    print(f"final mse: {mse:.6e}")

    # Save compressed representation
    model.save("lora_compressed.pth")

    # To load later:
    # loaded = LoRAStackCompressor(N, d_out, d_in, r=r)
    # loaded.load("lora_compressed.pth")
