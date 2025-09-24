import torch
import torch.nn.functional as F
from torch.autograd import Function


class QuantizationLoss(Function):
    @staticmethod
    def forward(ctx, w, group_sz=32, nbits=4):
        # 元の形状を保存
        original_shape = w.shape
        w_reshaped = w.reshape(-1, group_sz)
        
        # 量子化パラメータ
        Qp, Qn = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
        
        # スケールファクターの計算
        w_max = w_reshaped.max(dim=1, keepdim=True)[0]
        w_min = w_reshaped.min(dim=1, keepdim=True)[0]
        s = torch.maximum(w_max / Qp, w_min / Qn)
        
        # 量子化処理 (元の実装と同じ流れ)
        w_scaled = w_reshaped / s  # div
        w_round = w_scaled.round()  # round (straight-through)
        w_q_clamped = w_round.clamp(Qn, Qp)  # clamp
        w_q = w_q_clamped * s  # mul
        
        # 損失計算
        diff = w_q - w_reshaped
        loss = (diff * diff).sum()
        
        # backwardで必要な値を保存 - 元の実装の計算グラフを模倣
        ctx.save_for_backward(w_reshaped, s, w_scaled, w_round, w_q_clamped)
        ctx.group_sz = group_sz
        ctx.nbits = nbits
        ctx.original_shape = original_shape
        ctx.Qp, ctx.Qn = Qp, Qn
        
        return loss
    
    @staticmethod  
    def backward(ctx, grad_output):
        w_reshaped, s, w_scaled, w_round, w_q_clamped = ctx.saved_tensors
        Qp, Qn = ctx.Qp, ctx.Qn
        original_shape = ctx.original_shape
        
        # 元の実装の計算グラフを逆向きに辿る
        # loss = sum((w_q - w)^2) where w_q = w_q_clamped * s
        # w_q_clamped = clamp(w_round, Qn, Qp)  
        # w_round = round(w_scaled) (straight-through)
        # w_scaled = w / s
        # s = max(w_max/Qp, w_min/Qn)
        
        w_q = w_q_clamped * s
        
        # ∂loss/∂w_q = 2 * (w_q - w)
        grad_loss_wrt_wq = 2 * (w_q - w_reshaped)
        
        # ∂w_q/∂w_q_clamped = s
        grad_wq_wrt_wq_clamped = s
        
        # ∂loss/∂w_q_clamped = grad_loss_wrt_wq * grad_wq_wrt_wq_clamped
        grad_loss_wrt_wq_clamped = grad_loss_wrt_wq * grad_wq_wrt_wq_clamped
        
        # clamp の勾配: クランプされていない場合のみ勾配を通す
        clamp_mask = (w_round >= Qn) & (w_round <= Qp)
        grad_wq_clamped_wrt_wround = clamp_mask.float()
        
        # ∂loss/∂w_round = grad_loss_wrt_wq_clamped * grad_wq_clamped_wrt_wround  
        grad_loss_wrt_wround = grad_loss_wrt_wq_clamped * grad_wq_clamped_wrt_wround
        
        # round の勾配: straight-through estimator (勾配をそのまま通す)
        grad_wround_wrt_wscaled = torch.ones_like(w_scaled)
        
        # ∂loss/∂w_scaled = grad_loss_wrt_wround * grad_wround_wrt_wscaled
        grad_loss_wrt_wscaled = grad_loss_wrt_wround * grad_wround_wrt_wscaled
        
        # w_scaled = w / s の勾配
        # ∂w_scaled/∂w = 1/s, ∂w_scaled/∂s = -w/s^2
        grad_wscaled_wrt_w = 1.0 / s
        grad_wscaled_wrt_s = -w_reshaped / (s * s)
        
        # w_q = w_q_clamped * s の s に関する勾配も考慮
        grad_wq_wrt_s = w_q_clamped
        grad_loss_wrt_s_from_wq = grad_loss_wrt_wq * grad_wq_wrt_s
        
        # s に関する全ての勾配を合計
        grad_loss_wrt_s = (grad_loss_wrt_wscaled * grad_wscaled_wrt_s + 
                          grad_loss_wrt_s_from_wq)
        
        # s = max(w_max/Qp, w_min/Qn) の勾配
        w_max = w_reshaped.max(dim=1, keepdim=True)[0]
        w_min = w_reshaped.min(dim=1, keepdim=True)[0]
        use_max = (w_max / Qp) >= (w_min / Qn)
        
        # ∂s/∂w の計算
        grad_s_wrt_w = torch.zeros_like(w_reshaped)
        
        # max が選ばれた場合
        is_max = (w_reshaped == w_max)
        max_condition = use_max & is_max
        if max_condition.any():
            max_count = is_max.sum(dim=1, keepdim=True).float()
            grad_s_wrt_w = torch.where(max_condition, 1.0 / (Qp * max_count), grad_s_wrt_w)
            
        # min が選ばれた場合  
        is_min = (w_reshaped == w_min)
        min_condition = (~use_max) & is_min
        if min_condition.any():
            min_count = is_min.sum(dim=1, keepdim=True).float() 
            grad_s_wrt_w = torch.where(min_condition, 1.0 / (Qn * min_count), grad_s_wrt_w)
        
        # 最終的な勾配: ∂loss/∂w
        grad_w = (grad_loss_wrt_wscaled * grad_wscaled_wrt_w + 
                 grad_loss_wrt_s.sum(dim=1, keepdim=True) * grad_s_wrt_w)
        
        # 元の形状に戻してgrad_outputを適用
        grad_w = grad_w.reshape(original_shape) * grad_output
        
        return grad_w, None, None


def quantization_loss(w, group_sz=32, nbits=4):
    """
    メモリ効率的な量子化損失関数
    
    Args:
        w: 入力テンソル
        group_sz: グループサイズ（デフォルト: 32）
        nbits: 量子化ビット数（デフォルト: 4）
    
    Returns:
        量子化誤差の損失
    """
    return QuantizationLoss.apply(w, group_sz, nbits)


# 使用例とテスト
if __name__ == "__main__":
    # テスト用のデータ
    torch.manual_seed(42)
    w = torch.randn(64, 128, requires_grad=True)
    
    # 元の実装
    def round_ste(w):
        return w.round() + w - w.detach()
    
    def original_loss_fn(w, group_sz=32, nbits=4):
        w = w.reshape(-1, group_sz)
        Qp, Qn = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
        s = torch.maximum(w.max(dim=1, keepdim=True)[0] / Qp, w.min(dim=1, keepdim=True)[0] / Qn)
        w_q = round_ste(w.div(s)).clamp(Qn, Qp).mul(s)
        return (w_q - w).pow(2).sum()
    
    # 比較テスト
    print("Testing quantization loss implementations...")
    
    # 損失値の比較
    loss_original = original_loss_fn(w.clone())
    loss_efficient = quantization_loss(w.clone())
    
    print(f"Original loss: {loss_original.item():.6f}")
    print(f"Efficient loss: {loss_efficient.item():.6f}")
    print(f"Loss difference: {abs(loss_original.item() - loss_efficient.item()):.8f}")
    
    # 勾配の比較
    w1 = w.clone().detach().requires_grad_(True)
    w2 = w.clone().detach().requires_grad_(True)
    
    loss1 = original_loss_fn(w1)
    loss1.backward()
    grad1 = w1.grad.clone()
    
    loss2 = quantization_loss(w2)
    loss2.backward()
    grad2 = w2.grad.clone()
    
    grad_diff = (grad1 - grad2).abs().max().item()
    grad_rel_diff = ((grad1 - grad2).abs() / (grad1.abs() + 1e-8)).max().item()
    
    print(f"Max gradient difference: {grad_diff:.8f}")
    print(f"Max relative gradient difference: {grad_rel_diff:.8f}")
    
    print("\nMemory usage comparison:")
    print("- Original: Creates multiple intermediate tensors in autograd graph")
    print("- Efficient: Only saves necessary tensors for backward pass")