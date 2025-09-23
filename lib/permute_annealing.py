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

import torch
from .utils import *
from .get_module import *
import random
import math

def apply_permute_annealing_fast(model, n_trials=1000):
    """
    swap操作のみに特化した高速版SA
    差分更新により大幅な高速化を実現
    """
    nbits, group_sz = 4, 32
    w = get_embed(model).weight
    shape, dtype = w.shape, w.dtype
    original_w = w.clone().float()
    
    n_rows, n_cols = shape
    n_blocks = (n_cols + group_sz - 1) // group_sz  # ブロック数（切り上げ）
    
    print(f"行列サイズ: {shape}, ブロック数: {n_blocks}")
    
    # 量子化パラメータ
    Qp, Qn = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
    
    # 初期状態: 各ブロックのスケール係数と誤差を事前計算
    current_perm = list(range(n_cols))
    w_permuted = original_w  # 初期状態では順列なし
    
    # ブロックごとの情報を事前計算・保存
    block_info = {}
    total_loss = 0.0
    
    for block_idx in range(n_blocks):
        start_col = block_idx * group_sz
        end_col = min(start_col + group_sz, n_cols)
        
        # ブロックの重み取得
        block_w = w_permuted[:, start_col:end_col]  # shape: (n_rows, block_size)
        
        # スケール係数計算
        block_max = block_w.max(dim=1, keepdim=True)[0]  # (n_rows, 1)
        block_min = block_w.min(dim=1, keepdim=True)[0]  # (n_rows, 1)
        scale = torch.maximum(block_max / Qp, block_min / Qn)
        
        # 量子化と誤差計算
        block_w_q = block_w.div(scale).round_().clamp_(Qn, Qp).mul_(scale)
        block_loss = (block_w_q - block_w).pow(2).sum().item()
        
        # ブロック情報保存
        block_info[block_idx] = {
            'scale': scale,
            'quantized': block_w_q,
            'loss': block_loss,
            'start_col': start_col,
            'end_col': end_col
        }
        total_loss += block_loss
    
    def get_block_index(col_idx):
        """列インデックスからブロックインデックスを取得"""
        return col_idx // group_sz
    
    def update_block_after_swap(block_idx, w_permuted):
        """スワップ後のブロック情報を更新"""
        start_col = block_idx * group_sz
        end_col = min(start_col + group_sz, n_cols)
        
        # 新しいブロックの重み取得
        block_w = w_permuted[:, start_col:end_col]
        
        # スケール係数再計算
        block_max = block_w.max(dim=1, keepdim=True)[0]
        block_min = block_w.min(dim=1, keepdim=True)[0]
        scale = torch.maximum(block_max / Qp, block_min / Qn)
        
        # 量子化と誤差計算
        block_w_q = block_w.div(scale).round_().clamp_(Qn, Qp).mul_(scale)
        block_loss = (block_w_q - block_w).pow(2).sum().item()
        
        return {
            'scale': scale,
            'quantized': block_w_q,
            'loss': block_loss,
            'start_col': start_col,
            'end_col': end_col
        }
    
    # SA初期化
    current_loss = total_loss
    best_perm = current_perm.copy()
    best_loss = current_loss
    best_w = original_w.clone()
    
    initial_temp = current_loss * 0.1
    final_temp = current_loss * 0.001
    alpha = (final_temp / initial_temp) ** (1.0 / n_trials)
    current_temp = initial_temp
    
    print(f"初期量子化誤差: {current_loss:.6f}")
    print(f"初期温度: {initial_temp:.6f}")
    
    # 現在の重み行列（GPU上で保持）
    current_w = original_w.clone()
    
    accept_count = 0
    improve_count = 0
    
    # SA メインループ
    for trial in range(n_trials):
        # ランダムに2つの列を選択してスワップ
        col1, col2 = random.sample(range(n_cols), 2)
        block1_idx = get_block_index(col1)
        block2_idx = get_block_index(col2)
        
        # 現在の誤差（影響を受けるブロックのみ）
        old_loss_contribution = 0.0
        affected_blocks = set([block1_idx, block2_idx])
        
        for block_idx in affected_blocks:
            old_loss_contribution += block_info[block_idx]['loss']
        
        # スワップ実行（一時的）
        new_w = current_w.clone()
        new_w[:, [col1, col2]] = new_w[:, [col2, col1]]
        
        # 影響を受けるブロックの新しい誤差を計算
        new_loss_contribution = 0.0
        new_block_info = {}
        
        for block_idx in affected_blocks:
            new_info = update_block_after_swap(block_idx, new_w)
            new_block_info[block_idx] = new_info
            new_loss_contribution += new_info['loss']
        
        # 総誤差の変化
        delta_loss = new_loss_contribution - old_loss_contribution
        new_total_loss = current_loss + delta_loss
        
        # 受容判定
        accept = False
        if delta_loss < 0:
            accept = True
            improve_count += 1
        elif random.random() < math.exp(-delta_loss / current_temp):
            accept = True
        
        if accept:
            # 状態更新
            current_w = new_w
            current_loss = new_total_loss
            current_perm[col1], current_perm[col2] = current_perm[col2], current_perm[col1]
            
            # ブロック情報更新
            for block_idx in affected_blocks:
                block_info[block_idx] = new_block_info[block_idx]
            
            accept_count += 1
            
            # ベスト解更新
            if current_loss < best_loss:
                best_loss = current_loss
                best_perm = current_perm.copy()
                best_w = current_w.clone()
                print(f"Trial {trial}: 新しいベスト誤差 = {best_loss:.6f}")
        
        # 温度更新
        current_temp *= alpha
        
        # 進捗表示
        if (trial + 1) % 200 == 0:
            accept_rate = accept_count / 200
            improve_rate = improve_count / 200
            print(f"Trial {trial + 1}/{n_trials}: 現在誤差 = {current_loss:.6f}, "
                  f"ベスト誤差 = {best_loss:.6f}, 受容率 = {accept_rate:.3f}, "
                  f"改善率 = {improve_rate:.3f}, 温度 = {current_temp:.6f}")
            accept_count = 0
            improve_count = 0
    
    improvement = (1 - best_loss / total_loss) * 100
    print(f"\n最終結果:")
    print(f"初期誤差: {total_loss:.6f}")
    print(f"最適化後誤差: {best_loss:.6f}")
    print(f"改善率: {improvement:.2f}%")
    
    # モデルに最適解を適用
    with torch.no_grad():
        get_embed(model).weight.copy_(best_w.to(dtype))
    
    return best_perm, best_loss


def apply_permute_annealing_gpu_optimized(model, n_trials=1000):
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
    
    # SA パラメータ（局所解に陥りにくい問題特性のため低温度）
    initial_temp = 0.01
    final_temp = 0.0
    alpha = (final_temp / initial_temp) ** (1.0 / n_trials) if final_temp > 0 else 0.999
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
    
def apply_permute_annealing_swap_only_fast(model, n_trials=1000):
    """
    超高速版：PyTorchの最適化を最大限活用
    - バッチ化されたswap候補評価
    - JITコンパイル
    - メモリ効率最適化
    """
    nbits, group_sz = 4, 32
    w = get_embed(model).weight
    shape, dtype = w.shape, w.dtype
    device = w.device
    
    original_w = w.float()
    n_rows, n_cols = shape
    n_blocks = (n_cols + group_sz - 1) // group_sz
    
    print(f"超高速版 - デバイス: {device}")
    print(f"行列サイズ: {shape}, ブロック数: {n_blocks}")
    
    Qp, Qn = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
    
    # JITコンパイル用の関数定義
    @torch.jit.script
    def quantize_block(block_w: torch.Tensor, qp: float, qn: float) -> tuple[torch.Tensor, float]:
        """ブロック量子化をJITコンパイル"""
        block_max = block_w.max(dim=1, keepdim=True)[0]
        block_min = block_w.min(dim=1, keepdim=True)[0]
        scale = torch.maximum(block_max / qp, block_min / qn)
        block_w_q = block_w.div(scale).round_().clamp_(qn, qp).mul_(scale)
        loss = (block_w_q - block_w).pow(2).sum()
        return block_w_q, loss
    
    # 事前に計算できるものを準備
    current_w = original_w.clone()
    block_indices = torch.arange(n_cols, device=device) // group_sz
    
    # 初期誤差計算（バッチ処理版）
    def calculate_total_loss_batch(weight_matrix):
        # 完全にバッチ化された計算
        padded_cols = n_blocks * group_sz
        if n_cols < padded_cols:
            padding = torch.zeros(n_rows, padded_cols - n_cols, device=device, dtype=weight_matrix.dtype)
            w_padded = torch.cat([weight_matrix, padding], dim=1)
        else:
            w_padded = weight_matrix
        
        # 全ブロックを一度に処理
        w_blocks = w_padded.view(n_rows, n_blocks, group_sz)
        block_max = w_blocks.max(dim=2, keepdim=True)[0]
        block_min = w_blocks.min(dim=2, keepdim=True)[0]
        scales = torch.maximum(block_max / Qp, block_min / Qn)
        w_q_blocks = w_blocks.div(scales).round_().clamp_(Qn, Qp).mul_(scales)
        return (w_q_blocks - w_blocks).pow(2).sum().item()
    
    # バッチでswap候補を評価（超高速化の鍵）
    def evaluate_multiple_swaps(weight_matrix, swap_pairs, batch_size=32):
        """複数のswap候補を並列評価"""
        deltas = []
        
        for i in range(0, len(swap_pairs), batch_size):
            batch_pairs = swap_pairs[i:i + batch_size]
            batch_deltas = []
            
            for col1, col2 in batch_pairs:
                block1_idx = col1 // group_sz
                block2_idx = col2 // group_sz
                
                if block1_idx == block2_idx:
                    batch_deltas.append(0.0)
                    continue
                
                # 2ブロック分を一括計算
                blocks_to_check = [block1_idx, block2_idx]
                old_loss = torch.tensor(0.0)
                new_loss = torch.tensor(0.0)
                
                for block_idx in blocks_to_check:
                    start_col = block_idx * group_sz
                    end_col = min(start_col + group_sz, n_cols)
                    
                    # 元のブロック
                    old_block = weight_matrix[:, start_col:end_col]
                    _, old_block_loss = quantize_block(old_block, float(Qp), float(Qn))
                    old_loss += old_block_loss
                    
                    # swap後のブロック
                    new_w_temp = weight_matrix.clone()
                    new_w_temp[:, [col1, col2]] = new_w_temp[:, [col2, col1]]
                    new_block = new_w_temp[:, start_col:end_col]
                    _, new_block_loss = quantize_block(new_block, float(Qp), float(Qn))
                    new_loss += new_block_loss
                
                batch_deltas.append((new_loss - old_loss).item())
            
            deltas.extend(batch_deltas)
        
        return deltas
    
    # 事前に大量のswap候補を生成（GPU並列処理に最適）
    def generate_swap_candidates(n_candidates):
        candidates = []
        col1 = random.sample(range(n_cols), 1)[0]
        for _ in range(n_candidates):
            col2 = random.sample(range(n_cols), 1)[0]
            # col1, col2 = random.sample(range(n_cols), 2)
            candidates.append((col1, col2))
        return candidates
    
    current_loss = calculate_total_loss_batch(current_w)
    best_loss = current_loss
    best_w = current_w.clone()
    
    # 温度設定（局所解に陥りにくい特性）
    initial_temp = 0.01
    final_temp = 0.0
    alpha = 0.999  # ゆっくりと0に近づく
    current_temp = initial_temp
    
    print(f"初期量子化誤差: {current_loss:.6f}")
    print(f"温度設定: {initial_temp} → {final_temp}")
    
    # 統計情報
    accept_count = 0
    improve_count = 0
    
    # 高速化：バッチ処理でswap評価
    batch_size = min(64, n_trials // 10)  # GPUメモリに応じて調整
    
    for trial in range(0, n_trials, batch_size):
        current_batch_size = min(batch_size, n_trials - trial)
        
        # バッチでswap候補生成
        swap_candidates = generate_swap_candidates(current_batch_size)
        
        # 並列でswap評価
        with torch.no_grad():  # メモリ効率化
            deltas = evaluate_multiple_swaps(current_w, swap_candidates, batch_size=8)
        
        update = False
        # 各swap候補について受容判定
        for idx, ((col1, col2), delta_loss) in enumerate(zip(swap_candidates, deltas)):
            accept = False
            
            if delta_loss < 0:
                accept = True
                improve_count += 1
            elif current_temp > 0 and random.random() < math.exp(-delta_loss / current_temp):
                accept = True
            
            if accept:
                # swap実行
                current_w[:, [col1, col2]] = current_w[:, [col2, col1]]
                current_loss += delta_loss
                accept_count += 1
                
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_w = current_w.clone()
                    update = True
            
            current_temp *= alpha

        if update:
            print(f"Trial {trial}: ベスト誤差 = {best_loss:.6f}")
        
        # 進捗表示（バッチ単位）
        if trial % 256 == 0:
            recent_trials = min(256, trial + batch_size)
            accept_rate = accept_count / recent_trials if recent_trials > 0 else 0
            improve_rate = improve_count / recent_trials if recent_trials > 0 else 0
            print(f"Trial {trial + batch_size}/{n_trials}: "
                  f"誤差 = {current_loss:.6f}, ベスト = {best_loss:.6f}, "
                  f"受容率 = {accept_rate:.3f}, 改善率 = {improve_rate:.3f}")
    
    improvement = (1 - best_loss / calculate_total_loss_batch(original_w)) * 100
    print(f"\n超高速版結果:")
    print(f"改善率: {improvement:.2f}%")
    
    with torch.no_grad():
        get_embed(model).weight.copy_(best_w.to(dtype))
    
    return None, best_loss  # 順列は追跡しない（メモリ節約）


def apply_permute_annealing_memory_efficient(model, n_trials=1000):
    """
    メモリ効率特化版：極大行列でもOOM回避
    """
    nbits, group_sz = 4, 32
    w = get_embed(model).weight
    shape, dtype = w.shape, w.dtype
    device = w.device
    
    print(f"メモリ効率版 - デバイス: {device}, メモリ使用量最小化")
    
    # インプレース操作でメモリ使用量を最小化
    current_w = w.float()  # 元の重みを直接使用
    n_rows, n_cols = shape
    Qp, Qn = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
    
    # チェックポイント機能付きの量子化関数
    def calculate_block_loss_inplace(weight_matrix, start_col, end_col):
        """メモリ効率を重視したブロック誤差計算"""
        with torch.no_grad():
            block_w = weight_matrix[:, start_col:end_col]
            block_max = block_w.max(dim=1, keepdim=True)[0]
            block_min = block_w.min(dim=1, keepdim=True)[0]
            scale = torch.maximum(block_max / Qp, block_min / Qn)
            
            # 一時的なテンソルを最小限に
            normalized = block_w.div(scale)
            quantized = normalized.round_().clamp_(Qn, Qp)
            reconstructed = quantized.mul_(scale)
            loss = (reconstructed - block_w).pow(2).sum().item()
            
            # メモリ解放
            del normalized, quantized, reconstructed, block_max, block_min, scale
            
            return loss
    
    # ストリーミング処理で初期誤差計算
    initial_loss = 0.0
    for block_idx in range((n_cols + group_sz - 1) // group_sz):
        start_col = block_idx * group_sz
        end_col = min(start_col + group_sz, n_cols)
        initial_loss += calculate_block_loss_inplace(current_w, start_col, end_col)
    
    current_loss = initial_loss
    best_loss = current_loss
    
    # 低温度設定
    current_temp = 0.01
    alpha = 0.999
    
    print(f"初期量子化誤差: {initial_loss:.6f}")
    
    accept_count = improve_count = 0
    
    # メインループ：最小メモリ使用
    for trial in range(n_trials):
        col1, col2 = random.sample(range(n_cols), 2)
        block1_idx, block2_idx = col1 // group_sz, col2 // group_sz
        
        if block1_idx == block2_idx:
            continue  # 同一ブロック内はスキップ
        
        # 影響ブロックの元誤差
        old_loss = 0.0
        for block_idx in [block1_idx, block2_idx]:
            start_col = block_idx * group_sz
            end_col = min(start_col + group_sz, n_cols)
            old_loss += calculate_block_loss_inplace(current_w, start_col, end_col)
        
        # インプレースでswap
        current_w[:, [col1, col2]] = current_w[:, [col2, col1]]
        
        # 新誤差計算
        new_loss = 0.0
        for block_idx in [block1_idx, block2_idx]:
            start_col = block_idx * group_sz
            end_col = min(start_col + group_sz, n_cols)
            new_loss += calculate_block_loss_inplace(current_w, start_col, end_col)
        
        delta_loss = new_loss - old_loss
        
        # 受容判定
        accept = delta_loss < 0 or (current_temp > 0 and random.random() < math.exp(-delta_loss / current_temp))
        
        if accept:
            current_loss += delta_loss
            accept_count += 1
            if delta_loss < 0:
                improve_count += 1
                
            if current_loss < best_loss:
                best_loss = current_loss
                if trial % 100 == 0:
                    print(f"Trial {trial}: ベスト誤差 = {best_loss:.6f}")
        else:
            # 拒否の場合は元に戻す
            current_w[:, [col1, col2]] = current_w[:, [col2, col1]]
        
        current_temp *= alpha
        
        if (trial + 1) % 200 == 0:
            print(f"Trial {trial + 1}: 誤差 = {current_loss:.6f}, "
                  f"受容率 = {accept_count/200:.3f}")
            accept_count = improve_count = 0
    
    improvement = (1 - best_loss / initial_loss) * 100
    print(f"メモリ効率版結果: 改善率 = {improvement:.2f}%")
    
    # 重みはすでにインプレース更新済み
    with torch.no_grad():
        get_embed(model).weight.copy_(current_w.to(dtype))
    
    return None, best_loss