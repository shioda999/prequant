import torch
from .utils import *
from .get_module import *
import random
import math

def apply_permute_annealing(model, n_trials=1000):
    nbits, group_sz = 4, 32
    w = get_embed(model).weight
    shape, dtype = w.shape, w.dtype
    original_w = w.clone()
    
    # 初期量子化誤差を計算
    def calculate_quantization_error(weight_matrix, permutation=None):
        if permutation is not None:
            weight_matrix = weight_matrix[:, permutation]
        
        w_reshaped = weight_matrix.reshape(-1, group_sz).float()
        Qp, Qn = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
        
        # スケール係数計算
        s = torch.maximum(
            w_reshaped.max(dim=1, keepdim=True)[0] / Qp, 
            w_reshaped.min(dim=1, keepdim=True)[0] / Qn
        )
        
        # 量子化
        w_q = w_reshaped.div(s).round_().clamp_(Qn, Qp).mul_(s)
        
        # 誤差計算
        loss = (w_q - w_reshaped).pow(2).sum()
        return loss.item()
    
    # 初期順列と初期誤差
    n_cols = shape[1]
    current_perm = list(range(n_cols))
    current_loss = calculate_quantization_error(w, current_perm)
    best_perm = current_perm.copy()
    best_loss = current_loss
    
    # SA パラメータ
    initial_temp = current_loss * 0.1  # 初期温度を誤差の10%に設定
    final_temp = current_loss * 0.001  # 最終温度を誤差の0.1%に設定
    alpha = (final_temp / initial_temp) ** (1.0 / n_trials)  # 冷却率
    
    current_temp = initial_temp
    
    print(f"初期量子化誤差: {current_loss:.6f}")
    print(f"初期温度: {initial_temp:.6f}, 最終温度: {final_temp:.6f}, 冷却率: {alpha:.6f}")
    
    # SA メインループ
    for trial in range(n_trials):
        # 新しい順列を生成（隣接する要素をスワップ）
        new_perm = current_perm.copy()
        
        # ランダムに操作を選択
        operation = random.choice(['swap', 'reverse', 'shift'])
        
        if operation == 'swap':
            # 2つの要素をランダムにスワップ
            i, j = random.sample(range(n_cols), 2)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            
        elif operation == 'reverse':
            # 部分配列を逆順にする
            start = random.randint(0, n_cols - 2)
            end = random.randint(start + 1, n_cols)
            new_perm[start:end] = new_perm[start:end][::-1]
            
        elif operation == 'shift':
            # 要素を別の位置に移動
            from_idx = random.randint(0, n_cols - 1)
            to_idx = random.randint(0, n_cols - 1)
            element = new_perm.pop(from_idx)
            new_perm.insert(to_idx, element)
        
        # 新しい順列での誤差を計算
        new_loss = calculate_quantization_error(w, new_perm)
        
        # 受容判定
        delta_loss = new_loss - current_loss
        
        if delta_loss < 0 or random.random() < math.exp(-delta_loss / current_temp):
            # 改善した場合、または確率的に受容
            current_perm = new_perm
            current_loss = new_loss
            
            # ベスト解の更新
            if current_loss < best_loss:
                best_perm = current_perm.copy()
                best_loss = current_loss
                print(f"Trial {trial}: 新しいベスト誤差 = {best_loss:.6f}")
        
        # 温度を下げる
        current_temp *= alpha
        
        # 進捗表示
        if (trial + 1) % 100 == 0:
            print(f"Trial {trial + 1}/{n_trials}: 現在誤差 = {current_loss:.6f}, "
                  f"ベスト誤差 = {best_loss:.6f}, 温度 = {current_temp:.6f}")
    
    print(f"最終結果:")
    print(f"初期誤差: {calculate_quantization_error(w):.6f}")
    print(f"最適化後誤差: {best_loss:.6f}")
    print(f"改善率: {(1 - best_loss / calculate_quantization_error(w)) * 100:.2f}%")
    
    # 最適順列を適用
    optimized_weight = original_w[:, best_perm]
    
    # モデルに適用
    with torch.no_grad():
        get_embed(model).weight.copy_(optimized_weight.to(dtype))
    
    return best_perm, best_loss


def apply_permute_annealing_advanced(model, n_trials=1000, use_parallel_tempering=True):
    """
    より高度なSA実装：パラレルテンパリングと適応的冷却スケジュール
    """
    nbits, group_sz = 4, 32
    w = get_embed(model).weight
    shape, dtype = w.shape, w.dtype
    original_w = w.clone()
    
    def calculate_quantization_error(weight_matrix, permutation=None):
        if permutation is not None:
            weight_matrix = weight_matrix[:, permutation]
        
        w_reshaped = weight_matrix.reshape(-1, group_sz).float()
        Qp, Qn = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
        
        s = torch.maximum(
            w_reshaped.max(dim=1, keepdim=True)[0] / Qp, 
            w_reshaped.min(dim=1, keepdim=True)[0] / Qn
        )
        
        w_q = w_reshaped.div(s).round_().clamp_(Qn, Qp).mul_(s)
        loss = (w_q - w_reshaped).pow(2).sum()
        return loss.item()
    
    n_cols = shape[1]
    
    if use_parallel_tempering:
        # パラレルテンパリング：複数の温度で同時実行
        n_replicas = 4
        replicas = []
        
        for i in range(n_replicas):
            temp_factor = 1.0 + i * 0.5
            initial_temp = calculate_quantization_error(w) * 0.1 * temp_factor
            replicas.append({
                'perm': list(range(n_cols)),
                'loss': calculate_quantization_error(w),
                'temp': initial_temp,
                'alpha': 0.995
            })
        
        best_perm = replicas[0]['perm'].copy()
        best_loss = replicas[0]['loss']
        
        for trial in range(n_trials):
            for replica in replicas:
                # 各レプリカで状態更新
                new_perm = replica['perm'].copy()
                
                # より多様な近傍操作
                operations = ['swap', 'reverse', 'shift', 'block_swap']
                operation = random.choice(operations)
                
                if operation == 'swap':
                    i, j = random.sample(range(n_cols), 2)
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                    
                elif operation == 'reverse':
                    start = random.randint(0, n_cols - 2)
                    end = random.randint(start + 1, min(start + group_sz, n_cols))
                    new_perm[start:end] = new_perm[start:end][::-1]
                    
                elif operation == 'shift':
                    from_idx = random.randint(0, n_cols - 1)
                    to_idx = random.randint(0, n_cols - 1)
                    element = new_perm.pop(from_idx)
                    new_perm.insert(to_idx, element)
                    
                elif operation == 'block_swap':
                    # ブロック単位での交換
                    block_size = min(group_sz, n_cols // 4)
                    start1 = random.randint(0, n_cols - block_size)
                    start2 = random.randint(0, n_cols - block_size)
                    if start1 != start2:
                        block1 = new_perm[start1:start1 + block_size]
                        block2 = new_perm[start2:start2 + block_size]
                        new_perm[start1:start1 + block_size] = block2
                        new_perm[start2:start2 + block_size] = block1
                
                new_loss = calculate_quantization_error(w, new_perm)
                delta_loss = new_loss - replica['loss']
                
                if delta_loss < 0 or random.random() < math.exp(-delta_loss / replica['temp']):
                    replica['perm'] = new_perm
                    replica['loss'] = new_loss
                    
                    if replica['loss'] < best_loss:
                        best_perm = replica['perm'].copy()
                        best_loss = replica['loss']
                
                # 適応的冷却
                if trial > n_trials // 4:  # 後半で冷却を加速
                    replica['temp'] *= 0.99
                else:
                    replica['temp'] *= replica['alpha']
            
            # レプリカ間での情報交換
            if trial % 50 == 0:
                for i in range(len(replicas) - 1):
                    if random.random() < 0.1:  # 10%の確率で交換
                        temp_perm = replicas[i]['perm']
                        temp_loss = replicas[i]['loss']
                        
                        replicas[i]['perm'] = replicas[i + 1]['perm']
                        replicas[i]['loss'] = replicas[i + 1]['loss']
                        
                        replicas[i + 1]['perm'] = temp_perm
                        replicas[i + 1]['loss'] = temp_loss
            
            if (trial + 1) % 100 == 0:
                avg_temp = sum(r['temp'] for r in replicas) / len(replicas)
                print(f"Trial {trial + 1}: ベスト誤差 = {best_loss:.6f}, 平均温度 = {avg_temp:.6f}")
    
    else:
        # 標準のSA
        return apply_permute_annealing(model, n_trials)
    
    print(f"パラレルテンパリング結果:")
    print(f"初期誤差: {calculate_quantization_error(w):.6f}")
    print(f"最適化後誤差: {best_loss:.6f}")
    print(f"改善率: {(1 - best_loss / calculate_quantization_error(w)) * 100:.2f}%")
    
    # 最適順列を適用
    optimized_weight = original_w[:, best_perm]
    
    with torch.no_grad():
        get_embed(model).weight.copy_(optimized_weight.to(dtype))
    
    return best_perm, best_loss

def apply_permute_annealing_swap_only(model, n_trials=1000):
    """
    GPU最適化版：さらなる高速化のためのバッチ処理
    """
    nbits, group_sz = 4, 32
    w = get_embed(model).weight
    shape, dtype = w.shape, w.dtype
    device = w.device
    
    # GPU上で全て処理
    original_w = w.float()
    n_rows, n_cols = shape
    n_blocks = (n_cols + group_sz - 1) // group_sz
    
    print(f"GPU最適化版 - デバイス: {device}")
    print(f"行列サイズ: {shape}, ブロック数: {n_blocks}")
    
    Qp, Qn = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
    
    # GPU上でブロック情報を管理
    current_w = original_w.clone()
    current_perm = torch.arange(n_cols, device=device)
    
    def calculate_total_loss(weight_matrix):
        """GPU上で総誤差を効率的に計算"""
        # パディングして完全なブロック構造にする
        padded_cols = n_blocks * group_sz
        if n_cols < padded_cols:
            padding = torch.zeros(n_rows, padded_cols - n_cols, device=device)
            w_padded = torch.cat([weight_matrix, padding], dim=1)
        else:
            w_padded = weight_matrix
        
        # ブロック単位に reshapeしてバッチ処理
        w_blocks = w_padded.view(n_rows, n_blocks, group_sz)  # (n_rows, n_blocks, group_sz)
        
        # 各ブロックのスケール係数を並列計算
        block_max = w_blocks.max(dim=2, keepdim=True)[0]  # (n_rows, n_blocks, 1)
        block_min = w_blocks.min(dim=2, keepdim=True)[0]  # (n_rows, n_blocks, 1)
        scales = torch.maximum(block_max / Qp, block_min / Qn)
        
        # 全ブロックを並列で量子化
        w_q_blocks = w_blocks.div(scales).round_().clamp_(Qn, Qp).mul_(scales)
        
        # 誤差計算
        loss = (w_q_blocks - w_blocks).pow(2).sum()
        return loss.item()
    
    def calculate_swap_delta(weight_matrix, col1, col2):
        """swap操作による誤差変化を効率的に計算"""
        block1_idx = col1 // group_sz
        block2_idx = col2 // group_sz
        
        if block1_idx == block2_idx:
            # 同一ブロック内のswapは誤差に影響しない
            return 0.0
        
        # 影響を受けるブロックのみ計算
        affected_blocks = [block1_idx, block2_idx]
        
        old_loss = 0.0
        new_loss = 0.0
        
        for block_idx in affected_blocks:
            start_col = block_idx * group_sz
            end_col = min(start_col + group_sz, n_cols)
            
            # 元のブロック
            old_block = weight_matrix[:, start_col:end_col]
            old_max = old_block.max(dim=1, keepdim=True)[0]
            old_min = old_block.min(dim=1, keepdim=True)[0]
            old_scale = torch.maximum(old_max / Qp, old_min / Qn)
            old_q = old_block.div(old_scale).round_().clamp_(Qn, Qp).mul_(old_scale)
            old_loss += (old_q - old_block).pow(2).sum()
            
            # swap後のブロック
            new_w = weight_matrix.clone()
            new_w[:, [col1, col2]] = new_w[:, [col2, col1]]
            new_block = new_w[:, start_col:end_col]
            new_max = new_block.max(dim=1, keepdim=True)[0]
            new_min = new_block.min(dim=1, keepdim=True)[0]
            new_scale = torch.maximum(new_max / Qp, new_min / Qn)
            new_q = new_block.div(new_scale).round_().clamp_(Qn, Qp).mul_(new_scale)
            new_loss += (new_q - new_block).pow(2).sum()
        
        return (new_loss - old_loss).item()
    
    # 初期状態
    current_loss = calculate_total_loss(current_w)
    best_loss = current_loss
    best_w = current_w.clone()
    best_perm = current_perm.clone()
    
    # SA パラメータ
    initial_temp = 0.01
    final_temp = 0.0001
    alpha = (final_temp / initial_temp) ** (1.0 / n_trials)
    current_temp = initial_temp
    
    print(f"初期量子化誤差: {current_loss:.6f}")
    
    accept_count = 0
    improve_count = 0
    
    # SA メインループ
    for trial in range(n_trials):
        # ランダムswap
        col1, col2 = random.sample(range(n_cols), 2)
        
        # 差分計算
        delta_loss = calculate_swap_delta(current_w, col1, col2)
        
        # 受容判定
        accept = False
        if delta_loss < 0:
            accept = True
            improve_count += 1
        elif random.random() < math.exp(-delta_loss / current_temp):
            accept = True
        
        if accept:
            # swap実行
            current_w[:, [col1, col2]] = current_w[:, [col2, col1]]
            current_perm[[col1, col2]] = current_perm[[col2, col1]]
            current_loss += delta_loss
            accept_count += 1
            
            # ベスト更新
            if current_loss < best_loss:
                best_loss = current_loss
                best_w = current_w.clone()
                best_perm = current_perm.clone()
                print(f"Trial {trial}: 新しいベスト誤差 = {best_loss:.6f}")
        
        current_temp *= alpha
        
        if (trial + 1) % 200 == 0:
            accept_rate = accept_count / 200
            improve_rate = improve_count / 200
            print(f"Trial {trial + 1}/{n_trials}: 誤差 = {current_loss:.6f}, "
                  f"ベスト = {best_loss:.6f}, 受容率 = {accept_rate:.3f}, "
                  f"改善率 = {improve_rate:.3f}")
            accept_count = 0
            improve_count = 0
    
    improvement = (1 - best_loss / calculate_total_loss(original_w)) * 100
    print(f"\nGPU最適化版結果:")
    print(f"改善率: {improvement:.2f}%")
    
    # モデル更新
    with torch.no_grad():
        get_embed(model).weight.copy_(best_w.to(dtype))
    
    return best_perm.cpu().numpy().tolist(), best_loss