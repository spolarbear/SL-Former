# train.py
"""
训练脚本 (普通回归, 无因果掩码) - 用真实地震动输入预测位移时程

特点:
1. 使用 SLFormer (普通 Transformer, 无掩码), 输入 = 真实地震动加速度 (来自缓存 motions), 不泄漏位移
2. 目标归一化: 位移按全局 std 归一化, 数值稳定
3. NaN 防护: loss 非有限自动跳过, 权重 NaN 自动回退
4. 检查点 + 中断恢复
5. 训练结束立即评估 (独立输出目录)

数据需求: octree_cache 需含 motions 字段 (由 --force_sim + --force_octree 重新生成)

用法:
    python train.py
    python train.py --epochs 200 --batch 64 --lr 3e-4
    python train.py --out ./models --resume
"""
import os
import sys
import time
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from config import Config
from dataset import OctreeDataset
from transformer_model import SLFormer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def build_loaders(cfg, use_db=False, split_file=None, use_voxel_feature=False,
                  voxel_depth=5, voxel_enc_mode=None, db_filter=None):
    """加载数据集并划分 train/val (支持数据库模式)

    split_file: 划分索引缓存文件。若存在则加载同一批划分 (保证 evaluate 与训练一致);
                否则生成并保存。
    use_voxel_feature: True 时用体素化编码特征 (研究用, 替代杆系 frame_features)
    voxel_depth: 体素化八叉树紧缩深度 (4~7)
    voxel_enc_mode: 切杆系编码模式 'token'/'direct'/'cont' (None=按 USE_VOXEL_TOKEN)
    db_filter: 数据子集过滤 dict (仅数据库模式): plane_shape=..., floor_load_kpa=...
               None=不过滤 (全部样本)
    """
    if use_db:
        from dataset import OctreeDataset as DbOctree
        dataset = DbOctree(cfg, use_db=True, use_voxel_feature=use_voxel_feature,
                           voxel_depth=voxel_depth, voxel_enc_mode=voxel_enc_mode,
                           db_filter=db_filter)
    else:
        dataset = OctreeDataset(cfg, force_regen_octree=False,
                                use_voxel_feature=use_voxel_feature,
                                voxel_depth=voxel_depth, voxel_enc_mode=voxel_enc_mode)
    if dataset.motions is None:
        print("  [X] 数据缺少 motions (真实地震动) 字段!")
        if use_db:
            print("      请先运行 db_generate_samples.py 生成样本到数据库")
        else:
            print("      请用 --force_sim --force_octree 重新生成 (从 dzb 读真实波)")
        return None, None, None

    n_total = len(dataset)
    if n_total == 0:
        print("  [X] 数据集为空!")
        return None, None, None
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    # ---------- 划分索引缓存 (保证 train/evaluate 用同一批验证集) ----------
    if split_file is None:
        split_file = os.path.join(cfg.CACHE_DIR, 'dataset_split.pkl')
    if os.path.exists(split_file):
        # 加载已有划分 (校验样本数一致)
        try:
            with open(split_file, 'rb') as f:
                split = pickle.load(f)
            if split.get('n_total') == n_total:
                tr_indices = split['train_indices']
                va_indices = split['val_indices']
                print(f"  🔄 加载已保存的数据集划分 (train {len(tr_indices)}, "
                      f"val {len(va_indices)})")
            else:
                tr_indices, va_indices = None, None
                print(f"  ⚠️ 样本数变化 ({split.get('n_total')}->{n_total}), 重新划分")
        except Exception:
            tr_indices, va_indices = None, None
    else:
        tr_indices, va_indices = None, None

    if tr_indices is None:
        g = torch.Generator().manual_seed(42)
        tr, va = random_split(dataset, [n_train, n_val], generator=g)
        tr_indices = tr.indices
        va_indices = va.indices
        # 保存划分 (供 evaluate 复用)
        try:
            os.makedirs(os.path.dirname(split_file), exist_ok=True)
            with open(split_file, 'wb') as f:
                pickle.dump({'n_total': n_total, 'train_indices': tr_indices,
                             'val_indices': va_indices}, f)
            print(f"  💾 已保存数据集划分: {split_file}")
        except Exception as e:
            print(f"  [W] 划分保存失败: {e}")
    # 统一用 Subset 构造 (加载已有划分 或 新划分都走这里)
    tr = Subset(dataset, tr_indices)
    va = Subset(dataset, va_indices)

    # 自动调整 batch (样本少时不能用大 batch + drop_last)
    batch = cfg.BATCH_SIZE
    if n_train > 0:
        batch = min(batch, n_train)
    n_workers = getattr(cfg, 'NUM_WORKERS', 0)
    tr_loader = DataLoader(tr, batch_size=batch, shuffle=True,
                           num_workers=n_workers, pin_memory=True, drop_last=False,
                           persistent_workers=(n_workers > 0))
    va_loader = DataLoader(va, batch_size=batch, shuffle=False,
                           num_workers=n_workers, pin_memory=True,
                           persistent_workers=(n_workers > 0))
    return tr_loader, va_loader, dataset


# ============================================================
# 位移时程代理任务: 逐样本归一化波形 + 峰值 双目标
# ------------------------------------------------------------
# 问题: 不同样本位移波形差异大(中位相关≈0), 均值≈0, 绝对mm + MAE 下
#       输出常数0是平凡最优解 → 模型退化为输出0.
# 解决: 目标 = 位移/每样本峰值 ([-1,1]波形, 专注形状) + log(峰值) (幅度).
#       预测反缩放: pred_mm = pred_norm * pred_peak.
# ============================================================
def normalize_disp(disp):
    """disp: [B,T] -> (disp_norm, log_peak): 逐样本归一化波形 + log峰值"""
    peak = disp.abs().amax(1, keepdim=True).clamp(min=1e-3)
    disp_norm = disp / peak
    log_peak = torch.log(peak)
    return disp_norm, log_peak


def denormalize_pred(pred, log_peak):
    """pred: [B,T] 归一化波形, log_peak: [B,1] -> 反缩放 mm"""
    return pred * torch.exp(log_peak.clamp(min=-20.0, max=20.0))


def get_loss_thresh(cfg):
    """根据 LOSS_NORM 返回大位移阈值 (relative=峰值比例, absolute=绝对mm)"""
    loss_norm = str(getattr(cfg, 'LOSS_NORM', 'absolute')).lower()
    if loss_norm == 'relative':
        return float(getattr(cfg, 'LOSS_HIGH_THRESH_RATIO', 0.2))
    return float(getattr(cfg, 'LOSS_HIGH_THRESH_MM', 8.0))


def get_loss_norm(cfg):
    """返回归一化方式小写字符串"""
    return str(getattr(cfg, 'LOSS_NORM', 'absolute')).lower()


def weighted_mm_metric(pred, disp, w_peak=1.0, w_high=3.0, thresh_mm=8.0,
                       loss_norm='absolute', use_shape=True, use_peak=True,
                       use_high=True):
    """统一的加权指标 (训练 loss 与验证分数共用同一公式)

    两种归一化模式 (loss_norm):
      'absolute' (mm 量纲):
        1) mae_mm   : 平均位移误差 (pred - disp) 的绝对均值 [mm]
        2) peak_mm  : 峰值位移绝对误差 (预测峰值 vs 真值峰值) [mm]
        3) high_mm  : 大位移惩罚 (真值|disp|>thresh 时刻误差加权 w_high)
        加权总分 = mae_mm + w_peak * peak_mm + high_mm

      'relative' (峰值归一化, 无量纲):
        1) shape_rel : 峰值归一化波形误差 mean(|pred-disp| / peak_true)
        2) peak_rel  : 峰值相对误差 |pred_peak - peak_true| / peak_true
        3) high_rel  : 峰值归一化大位移误差 (|pred-disp|/peak_true, >thresh 加权)
        加权总分 = shape_rel + w_peak * peak_rel + high_rel
        (大位移阈值 thresh_mm 在 relative 模式解释为峰值比例, 如 0.1=10%峰值)

    Args:
        pred: [B,T] 预测位移 (mm)
        disp: [B,T] 真值位移 (mm)
        w_peak: 峰值项权重
        w_high: 大位移区域加权倍数
        thresh_mm: absolute=绝对阈值(mm); relative=峰值比例(0~1)
        loss_norm: 'absolute' 绝对 mm / 'relative' 峰值归一化相对误差
        use_shape: 是否包含波形项 (消融用)
        use_peak: 是否包含峰值项 (消融用)
        use_high: 是否包含大位移惩罚项 (消融用)

    Returns:
        (metric, shape_term, peak_term, high_term)
    """
    # 峰值 (每样本, clamp 防除零)
    peak_true = disp.abs().amax(1, keepdim=True).clamp(min=1e-3)
    is_rel = str(loss_norm).lower() == 'relative'

    if is_rel:
        # ---- 峰值归一化相对误差 (无量纲) ----
        # 1) 波形峰值归一化误差 (逐样本: 每样本误差/该样本峰值, 再平均)
        #    修正: 不能先全局平均再除全局平均峰值 (会被大峰值样本稀释)
        shape_term = ((pred - disp).abs() / peak_true).mean() \
            if use_shape else torch.zeros((), device=pred.device)
        # 2) 峰值相对误差
        if use_peak:
            pred_peak = pred.abs().amax(1, keepdim=True).clamp(min=1e-3)
            peak_term = ((pred_peak - peak_true).abs() / peak_true).mean()
        else:
            peak_term = torch.zeros((), device=pred.device)
        # 3) 峰值归一化大位移惩罚 (thresh_mm 视为峰值比例)
        if use_high:
            high_mask = (disp.abs() > thresh_mm * peak_true).float()
            w = 1.0 + (w_high - 1.0) * high_mask
            high_term = (w * (pred - disp).abs() / peak_true).mean()
        else:
            high_term = torch.zeros((), device=pred.device)
        metric = shape_term + w_peak * peak_term + high_term
    else:
        # ---- 绝对 mm 误差 ----
        # 1) 平均位移 mm
        shape_term = (pred - disp).abs().mean() if use_shape else \
            torch.zeros((), device=pred.device)
        # 2) 峰值位移 mm
        if use_peak:
            pred_peak = pred.abs().amax(1, keepdim=True).clamp(min=1e-3)
            peak_term = (pred_peak - peak_true).abs().mean()
        else:
            peak_term = torch.zeros((), device=pred.device)
        # 3) 大位移惩罚 mm
        if use_high:
            high_mask = (disp.abs() > thresh_mm).float()
            w = 1.0 + (w_high - 1.0) * high_mask
            high_term = (w * (pred - disp).abs()).mean()
        else:
            high_term = torch.zeros((), device=pred.device)
        metric = shape_term + w_peak * peak_term + high_term
    return metric, shape_term, peak_term, high_term


def disp_peak(pred):
    """从预测波形估计峰值 (mm, 需配合 log_peak 反缩放)"""
    return pred.abs().amax(1).clamp(min=0.0)


def train(cfg, device, out_dir, resume=False, epochs=None, lr=None,
                 use_physics=True, use_bypass=True, use_v2=None, use_db=False,
                 model_kwargs=None, loss_kwargs=None, use_voxel_feature=False,
                 voxel_depth=5, voxel_enc_mode=None, db_filter=None):
    """训练模型 (普通回归, 用真实地震动输入; use_db=True 从数据库读数据)

    Args:
        model_kwargs: 额外传给 SLFormer 的构造参数 (消融用, 默认 None)
        loss_kwargs: 损失项开关 dict (消融用, 支持 peak/high/shape 是否启用)
        use_voxel_feature: True 时用体素化编码特征 (研究用)
        voxel_depth: 体素化八叉树紧缩深度 (4~7)
        voxel_enc_mode: 切杆系编码模式 'token'/'direct'/'cont'
                        (None=按 USE_VOXEL_TOKEN: True->token, False->cont)
        db_filter: 数据子集过滤 dict (仅数据库模式): plane_shape=...,
                   floor_load_kpa=... None=不过滤 (全部样本)
    """
    os.makedirs(out_dir, exist_ok=True)
    model_dir = os.path.join(out_dir, 'model')
    plot_dir = os.path.join(out_dir, 'plots')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # ---------- 数据 ----------
    tr_loader, va_loader, dataset = build_loaders(
        cfg, use_db=use_db, use_voxel_feature=use_voxel_feature,
        voxel_depth=voxel_depth, voxel_enc_mode=voxel_enc_mode,
        db_filter=db_filter)
    if tr_loader is None:
        return None

    n_train, n_val = len(tr_loader.dataset), len(va_loader.dataset)
    feat_dim = (cfg.FRAME_FEATURE_DIM if getattr(cfg, 'USE_FRAME_FEATURE', False)
                else cfg.OCTREE_FEATURE_DIM)
    print(f"\n  📊 数据: 训练 {n_train}, 验证 {n_val}, 特征 {feat_dim}")

    # ---------- 目标归一化 (全局 std) ----------
    disp = dataset.displacements
    disp_std = float(disp.std())
    disp_std = max(disp_std, 1e-6)
    print(f"  目标位移 std = {disp_std:.3f} mm (归一化用)")

    # ---------- 模型 (普通 SLFormer + 残差旁路, 无掩码) ----------
    if use_v2 is None:
        use_v2 = bool(getattr(cfg, 'USE_V2', False))
    mk = dict(model_kwargs or {})
    # 默认关闭自注意力 (no_sa — 消融结果显示更好); 显式传 use_sa 可覆盖
    mk.setdefault('use_sa', False)
    model = SLFormer(cfg, use_bypass=use_bypass, use_v2=use_v2,
                     drop_path=getattr(cfg, 'V2_DROP_PATH', 0.1),
                     film=getattr(cfg, 'V2_FILM', True), **mk).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  🧠 模型参数量: {n_params:,} (use_bypass={use_bypass}, use_v2={use_v2}, "
          f"kwargs={mk})")

    # 消融损失开关 (默认: 关 peak 损失 — 消融结果显示 no_peak 更好)
    lk = dict(loss_kwargs or {})
    use_loss_shape = bool(lk.get('shape', True))
    use_loss_peak = bool(lk.get('peak', False))   # 默认关闭峰值惩罚 (no_peak_loss)
    use_loss_high = bool(lk.get('high', True))
    print(f"  🎯 损失项: shape={use_loss_shape}, peak={use_loss_peak}, "
          f"high={use_loss_high}")

    # ---------- 优化器 & 学习率调度 ----------
    # 用户反馈: 初始 1e-3 太大, 前 ~30 个 epoch 几乎不动, 降到 5e-4 才开始学习。
    # 因此默认初始 LR 改为 5e-4 (cfg.LEARNING_RATE)。
    base_lr = lr or cfg.LEARNING_RATE
    total_epochs = epochs or cfg.EPOCHS
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.95))

    # 调度方式 (cfg.LR_SCHEDULER):
    #   'cosine' (推荐): 线性 warmup 升至 base_lr, 再余弦退火至 LR_MIN。
    #       避免初始过高导致"学不动", 后期又能平滑收敛 (transformer 常用)。
    #   'plateau'      : ReduceLROnPlateau, 验证指标平台期自动降 LR (旧行为)。
    scheduler_mode = str(getattr(cfg, 'LR_SCHEDULER', 'cosine')).lower()
    lr_min = float(getattr(cfg, 'LR_MIN', 1e-6))
    if scheduler_mode == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=float(getattr(cfg, 'LR_FACTOR', 0.5)),
            patience=int(getattr(cfg, 'LR_PATIENCE', 10)), min_lr=lr_min)
        scheduler_uses_metric = True
    else:
        # 线性 warmup + 余弦退火 (推荐)
        warmup_epochs = int(getattr(cfg, 'LR_WARMUP_EPOCHS', 5))
        warmup_epochs = max(1, min(warmup_epochs, max(total_epochs // 5, 1)))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=float(getattr(cfg, 'LR_WARMUP_START', 0.1)),
            total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=lr_min)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        scheduler_uses_metric = False

    # 混合精度 (FP16): 目标已归一化(波形[-1,1]+log峰值), 数值稳定, 可安全开启。
    # 仅 CUDA 可用且 cfg.USE_MIXED_PRECISION=True 时启用 (默认开启)。
    use_amp = (torch.cuda.is_available()
               and bool(getattr(cfg, 'USE_MIXED_PRECISION', True)))
    if use_amp:
        try:
            scaler = torch.amp.GradScaler('cuda')   # torch>=2.3 新 API
        except Exception:
            scaler = torch.cuda.amp.GradScaler()     # 旧 API 兼容
        print("  ⚡ 混合精度 (FP16) 已开启")
    else:
        scaler = None
        print("  🅿️ 使用 FP32 训练 (混合精度关闭)")

    # 损失: MAE (绝对误差) 为主
    mae_loss = nn.L1Loss()

    # ---------- 恢复 ----------
    ckpt_path = os.path.join(model_dir, 'checkpoint_last.pth')
    best_path = os.path.join(model_dir, 'best_model.pth')
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = -1
    patience = 0
    history = {'epoch': [], 'train_mae': [], 'val_mae': [], 'val_peak_mae': [],
               'val_high_mae': [], 'lr': [], 'time': []}

    if resume and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            try:
                if 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict']:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception:
                print("  [W] 调度器状态不兼容, 用新调度器")
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            best_epoch = ckpt.get('best_epoch', -1)
            patience = ckpt.get('patience', 0)
            history = ckpt.get('history', history)
            # 兼容旧 checkpoint (无 val_peak_mae / val_high_mae 键)
            if 'val_peak_mae' not in history:
                history['val_peak_mae'] = [float('nan')] * len(history.get('epoch', []))
            if 'val_high_mae' not in history:
                history['val_high_mae'] = [float('nan')] * len(history.get('epoch', []))
            print(f"  [R] 从 epoch {start_epoch} 恢复")
        except Exception as e:
            print(f"  [W] 恢复失败 ({e}), 从头训练")

    # ---------- 训练循环 ----------
    print(f"\n  🚀 开始训练 (epochs={total_epochs}, lr={base_lr}, "
          f"scheduler={scheduler_mode})")
    start_time = time.time()
    last_snapshot = None  # 权重快照用于 NaN 回退

    for epoch in range(start_epoch, total_epochs):
        epoch_t = time.time()
        model.train()
        t_mae = 0.0
        skipped = 0
        pbar = tqdm(tr_loader, desc=f"[Model] E{epoch+1}/{total_epochs} [Tr]")
        for batch in pbar:
            oct = batch['octree_features'].to(device, non_blocking=True)
            motion = batch['motion'].to(device, non_blocking=True)      # 真实地震动 [B,T]
            disp = batch['disp'].to(device, non_blocking=True)          # 原始位移 mm
            params = batch['params'].to(device, non_blocking=True)      # 结构参数 [B,8]
            ffeat = (batch['frame_features'].to(device, non_blocking=True)
                     if batch.get('frame_features') is not None else None)
            # 逐样本归一化目标: 波形 + log峰值 (避开"输出0"平凡解)
            target_norm, log_peak = normalize_disp(disp)
            # 真值峰值 (mm)
            peak_true = torch.exp(log_peak.clamp(min=-20.0, max=20.0))  # [B,1]

            # 低估惩罚超参 (config)
            w_peak = float(getattr(cfg, 'LOSS_PEAK_W', 1.0))
            w_high = float(getattr(cfg, 'LOSS_HIGH_W', 3.0))
            # 损失归一化方式 (relative: 峰值归一化相对误差, 大位移不主导)
            loss_norm = get_loss_norm(cfg)
            thresh_mm = get_loss_thresh(cfg)

            def step():
                optimizer.zero_grad(set_to_none=True)
                pred, _ = model(oct, motion.unsqueeze(-1), cond_params=params,
                                frame_features=ffeat)  # pred: [B,T] 近似 mm
                # 统一的加权 mm 指标 (平均位移 + 峰值位移 + 大位移惩罚), 全部 mm 量纲
                loss, mae_mm, peak_mm, high_mm = weighted_mm_metric(
                    pred, disp, w_peak=w_peak, w_high=w_high,
                    thresh_mm=thresh_mm, loss_norm=loss_norm,
                    use_shape=use_loss_shape, use_peak=use_loss_peak,
                    use_high=use_loss_high)
                return loss, pred, mae_mm, peak_mm, high_mm

            # 前向 + loss (FP16 时用 autocast; FP32 直接跑)
            if scaler is not None:
                with torch.autocast(device_type='cuda'):
                    loss, pred_out, mae_mm, peak_mm, high_mm = step()
            else:
                loss, pred_out, mae_mm, peak_mm, high_mm = step()

            if not torch.isfinite(loss):
                skipped += 1
                pbar.set_postfix({'skip': skipped})
                continue  # 跳过 NaN 步, 不更新

            # 反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                optimizer.step()

            t_mae += loss.item()  # 加权 mm 指标 (平均位移+峰值位移+惩罚), 与验证同口径
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # NaN 权重防护
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                if last_snapshot is not None:
                    model.load_state_dict(last_snapshot)
                print(f"    [W] 权重 NaN, 回退快照")
                break

        t_mae /= max(len(tr_loader) - skipped, 1)

        # 验证: 与训练同一加权指标 (相对: 峰值归一化波形+峰值相对误差+大位移相对误差)
        # 训练和验证都用 weighted_mm_metric, 口径完全统一, 可比
        w_peak_eval = float(getattr(cfg, 'LOSS_PEAK_W', 1.0))
        w_high_eval = float(getattr(cfg, 'LOSS_HIGH_W', 3.0))
        loss_norm_eval = get_loss_norm(cfg)
        thresh_eval = get_loss_thresh(cfg)
        model.eval()
        v_score_sum = 0.0
        v_mae = 0.0
        v_peak_mae = 0.0
        v_high_mae = 0.0
        with torch.no_grad():
            for batch in tqdm(va_loader, desc=f"[Model] E{epoch+1}/{total_epochs} [Va]"):
                oct = batch['octree_features'].to(device, non_blocking=True)
                motion = batch['motion'].to(device, non_blocking=True)
                disp = batch['disp'].to(device, non_blocking=True)      # mm
                params = batch['params'].to(device, non_blocking=True)
                ffeat = (batch['frame_features'].to(device, non_blocking=True)
                         if batch.get('frame_features') is not None else None)
                pred, _ = model(oct, motion.unsqueeze(-1), cond_params=params,
                                frame_features=ffeat)
                if torch.isfinite(pred).all():
                    # 同一加权 mm 指标 (与训练 step 完全一致, 含消融开关)
                    v_score, v_mae_b, v_peak_b, v_high_b = weighted_mm_metric(
                        pred, disp, w_peak=w_peak_eval, w_high=w_high_eval,
                        thresh_mm=thresh_eval, loss_norm=loss_norm_eval,
                        use_shape=use_loss_shape, use_peak=use_loss_peak,
                        use_high=use_loss_high)
                    v_score_sum += v_score.item()
                    v_mae += v_mae_b.item()
                    v_peak_mae += v_peak_b.item()
                    v_high_mae += v_high_b.item()
                else:
                    v_score_sum += float('inf')
                    v_mae += float('inf')
                    v_peak_mae += float('inf')
                    v_high_mae += float('inf')
        n_va = len(va_loader)
        v_score = v_score_sum / n_va
        v_mae /= n_va
        v_peak_mae /= n_va
        v_high_mae /= n_va

        # 学习率调度 (plateau 按验证分数; cosine 按 epoch)
        if scheduler_uses_metric:
            if np.isfinite(v_score):
                scheduler.step(v_score)
        else:
            scheduler.step()

        # 记录 (val_mae 存加权 mm 分数, 与 train_mae 加权 loss 同口径, 可直接对比)
        history['epoch'].append(epoch + 1)
        history['train_mae'].append(t_mae)
        history['val_mae'].append(v_score)
        history['val_peak_mae'].append(v_peak_mae)
        history['val_high_mae'].append(v_high_mae)
        history['lr'].append(scheduler.get_last_lr()[0])
        history['time'].append(time.time() - epoch_t)

        # 保存最佳 (按组合分数: 整体 + 峰值)
        if np.isfinite(v_score) and v_score < best_val_loss:
            best_val_loss = v_score
            best_epoch = epoch
            patience = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss, 'best_epoch': best_epoch,
                        'history': history, 'config': vars(cfg),
                        'model_kwargs': mk, 'disp_std': disp_std}, best_path)
        else:
            patience += 1

        # 快照 (用于 NaN 回退)
        last_snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # 检查点 (恢复)
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss, 'best_epoch': best_epoch,
                    'patience': patience, 'history': history, 'disp_std': disp_std,
                    'model_kwargs': mk},
                   ckpt_path)

        # 量纲标签: relative=百分比(%) 无量纲比值; absolute=绝对 mm
        is_rel = (loss_norm_eval == 'relative')
        if is_rel:
            # 相对模式: 无量纲比值, 显示为百分比 (×100)
            fmt_w = f"{t_mae*100:.2f}%"          # TrWmm
            fmt_vw = f"{v_score*100:.2f}%"       # VaWmm
            fmt_mae = f"{v_mae*100:.2f}%"
            fmt_peak = f"{v_peak_mae*100:.2f}%"
            fmt_high = f"{v_high_mae*100:.2f}%"
            fmt_best = f"{best_val_loss*100:.2f}%"
            unit = ''
        else:
            fmt_w = f"{t_mae:.3f}mm"
            fmt_vw = f"{v_score:.3f}mm"
            fmt_mae = f"{v_mae:.3f}mm"
            fmt_peak = f"{v_peak_mae:.3f}mm"
            fmt_high = f"{v_high_mae:.3f}mm"
            fmt_best = f"{best_val_loss:.3f}mm"
            unit = 'mm'
        print(f"[Model] E{epoch+1}/{total_epochs} TrWmm {fmt_w} "
              f"VaWmm {fmt_vw} VaMAE {fmt_mae} VaPeak {fmt_peak} "
              f"VaHigh {fmt_high} Best {fmt_best}@{best_epoch+1} "
              f"LR {scheduler.get_last_lr()[0]:.2e} Skip{skipped} "
              f"[{unit if unit else 'relative(%)'}]")

        if patience > cfg.EARLY_STOP_PATIENCE:
            print(f"  早停 @epoch {epoch+1}")
            break

    total_time = time.time() - start_time
    print(f"\n  ✅ 训练完成, 用时 {total_time/60:.1f} 分钟, "
          f"best_val_weighted_mm={best_val_loss:.3f}mm "
          f"(平均位移+w_peak*峰值+大位移惩罚)")

    # 保存 history + 曲线
    with open(os.path.join(out_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    plot_history(history, plot_dir)
    with open(os.path.join(out_dir, 'training_info.json'), 'w', encoding='utf-8') as f:
        json.dump({'n_params': n_params, 'best_val_score': float(best_val_loss),
                   'best_epoch': best_epoch, 'disp_std': disp_std}, f, indent=2)

    # 评估 (用 best_model.pth 权重, 与 evaluate.py 一致; 不用最后一个 epoch 的模型)
    if os.path.exists(best_path):
        try:
            best_ck = torch.load(best_path, map_location=device)
            model.load_state_dict(best_ck['model_state_dict'])
            print(f"  📦 已加载 best_model (epoch {best_ck.get('epoch', best_epoch)}) 进行最终评估")
        except Exception as e:
            print(f"  [W] best_model 加载失败 ({e}), 用当前模型评估")
    evaluate_after_train(cfg, model, va_loader, out_dir, device, disp_std)

    return model, history, best_val_loss


def evaluate_after_train(cfg, model, va_loader, out_dir, device, disp_std):
    """在验证集上评估, 输出到独立目录"""
    plot_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    all_pred, all_tgt = [], []
    model.eval()
    with torch.no_grad():
        for batch in va_loader:
            oct = batch['octree_features'].to(device)
            motion = batch['motion'].to(device)
            target = batch['disp'].to(device)
            params = batch['params'].to(device)
            ffeat = (batch['frame_features'].to(device)
                     if batch.get('frame_features') is not None else None)
            pred, _ = model(oct, motion.unsqueeze(-1), cond_params=params,
                            frame_features=ffeat)  # SLFormer 返回 (输出, attn) — 已是 mm
            all_pred.append(pred.cpu().numpy())
            all_tgt.append(target.cpu().numpy())
    pred = np.concatenate(all_pred)
    tgt = np.concatenate(all_tgt)

    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    flat_p, flat_t = pred.flatten(), tgt.flatten()
    r2 = r2_score(flat_t, flat_p)
    rmse = np.sqrt(mean_squared_error(flat_t, flat_p))
    mae = mean_absolute_error(flat_t, flat_p)
    peak_p = np.max(np.abs(pred), axis=1)
    peak_t = np.max(np.abs(tgt), axis=1)
    peak_r2 = r2_score(peak_t, peak_p)
    peak_mae = np.mean(np.abs(peak_p - peak_t))
    eps = 1e-3
    mape = np.mean(np.abs(pred - tgt) / (np.abs(tgt) + eps)) * 100.0

    # ============================================================
    # 相对偏差指标 (消除"大位移样本绝对偏差大"的主导)
    # ------------------------------------------------------------
    # 1) 归一化相对误差: 每样本误差 / 该样本峰值 -> 各样本等权
    norm_err = (pred - tgt) / (np.abs(peak_t)[:, None] + eps)
    rel_rmse = np.sqrt(np.mean(norm_err ** 2))
    rel_mae = np.mean(np.abs(norm_err))
    # 2) 样本级峰值相对误差 (%)
    peak_rel = np.abs(peak_p - peak_t) / (np.abs(peak_t) + eps)
    peak_rel_mae = np.mean(peak_rel) * 100.0
    # 3) 相对偏差 R²: 用归一化时程算 R² (大位移样本不再因幅值大而主导)
    flat_norm_p = (pred / (np.abs(peak_t)[:, None] + eps)).flatten()
    flat_norm_t = (tgt / (np.abs(peak_t)[:, None] + eps)).flatten()
    rel_r2 = r2_score(flat_norm_t, flat_norm_p)
    # 4) 峰值相对偏差 R²
    rel_peak_r2 = r2_score(peak_t, peak_p, )  # 用绝对峰值算 R² 已有 peak_r2
    #    按位移幅值分组: 小位移样本 (<中位峰值) vs 大位移样本 (>=中位峰值)
    med_peak = np.median(peak_t)
    small_idx = peak_t < med_peak
    large_idx = peak_t >= med_peak
    r2_small = r2_score(peak_t[small_idx], peak_p[small_idx]) if small_idx.sum() > 1 else 0.0
    r2_large = r2_score(peak_t[large_idx], peak_p[large_idx]) if large_idx.sum() > 1 else 0.0
    #    小/大位移组的整体时程 R²
    if small_idx.sum() > 1:
        r2_small_all = r2_score(tgt[small_idx].flatten(), pred[small_idx].flatten())
    else:
        r2_small_all = 0.0
    if large_idx.sum() > 1:
        r2_large_all = r2_score(tgt[large_idx].flatten(), pred[large_idx].flatten())
    else:
        r2_large_all = 0.0

    print("\n" + "="*60)
    print("模型评估 (验证集)")
    print("="*60)
    print(f"  R² (整体):    {r2:.4f}")
    print(f"  RMSE:         {rmse:.4f} mm")
    print(f"  MAE:          {mae:.4f} mm")
    print(f"  峰值 R²:      {peak_r2:.4f}")
    print(f"  峰值 MAE:     {peak_mae:.4f} mm")
    print(f"  整体 MAPE:    {mape:.2f} %")
    print("-"*60)
    print("  相对偏差指标 (每样本按峰值归一化, 大位移不主导):")
    print(f"    相对 RMSE:    {rel_rmse:.4f}  (归一化误差均方根)")
    print(f"    相对 MAE:     {rel_mae:.4f}  (归一化误差均值)")
    print(f"    相对偏差 R²:  {rel_r2:.4f}  (归一化时程 R²)")
    print(f"    峰值相对 MAE: {peak_rel_mae:.2f} %")
    print(f"    峰值 R² (绝对): {peak_r2:.4f}")
    print("-"*60)
    print(f"  按峰值位移分组 (中位峰值={med_peak:.2f} mm):")
    print(f"    小位移组 (n={small_idx.sum()}): 整体R²={r2_small_all:.4f}, 峰值R²={r2_small:.4f}")
    print(f"    大位移组 (n={large_idx.sum()}): 整体R²={r2_large_all:.4f}, 峰值R²={r2_large:.4f}")
    print("="*60)

    # 出图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    idx = np.random.choice(len(pred), min(2000, len(pred)), replace=False)
    axes[0, 0].scatter(tgt[idx], pred[idx], s=2, alpha=0.3)
    lims = [min(tgt.min(), pred.min()), max(tgt.max(), pred.max())]
    axes[0, 0].plot(lims, lims, 'k--')
    axes[0, 0].set_xlabel('Target (mm)'); axes[0, 0].set_ylabel('Pred (mm)')
    axes[0, 0].set_title(f'R²={r2:.4f}')
    err = (pred - tgt).flatten()
    axes[0, 1].hist(err, bins=80)
    axes[0, 1].set_title(f'Error (RMSE={rmse:.3f}mm)')
    axes[0, 1].set_xlabel('Error (mm)')
    axes[1, 0].scatter(peak_t, peak_p, alpha=0.5, s=15)
    axes[1, 0].plot(lims, lims, 'k--')
    axes[1, 0].set_xlabel('Target Peak (mm)'); axes[1, 0].set_ylabel('Pred Peak (mm)')
    axes[1, 0].set_title(f'Peak R²={peak_r2:.4f}')
    # 示例时程
    T = tgt.shape[1]
    t = np.arange(T) * cfg.TARGET_DT
    axes[1, 1].plot(t, tgt[0], 'b-', label='Target', alpha=0.8)
    axes[1, 1].plot(t, pred[0], 'r--', label='Pred', alpha=0.8)
    axes[1, 1].set_xlabel('Time (s)'); axes[1, 1].set_ylabel('Disp (mm)')
    axes[1, 1].set_title('Sample 0 Time History')
    axes[1, 1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'eval.png'), dpi=150)
    plt.close()

    result = {'r2': float(r2), 'rmse': float(rmse), 'mae': float(mae),
              'peak_r2': float(peak_r2), 'peak_mae': float(peak_mae),
              'mape_all': float(mape), 'num_samples': len(pred),
              # 相对偏差指标
              'rel_rmse': float(rel_rmse), 'rel_mae': float(rel_mae),
              'rel_r2': float(rel_r2), 'peak_rel_mae_pct': float(peak_rel_mae),
              'r2_small_all': float(r2_small_all), 'r2_large_all': float(r2_large_all),
              'r2_small_peak': float(r2_small), 'r2_large_peak': float(r2_large),
              'median_peak_mm': float(med_peak),
              'n_small': int(small_idx.sum()), 'n_large': int(large_idx.sum())}
    with open(os.path.join(out_dir, 'eval_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  💾 评估结果: {os.path.join(out_dir, 'eval_metrics.json')}")
    return result


def plot_history(history, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    ep = history['epoch']
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    # 1. 加权 mm (训练 vs 验证, 同口径: 平均位移+w_peak*峰值+大位移惩罚)
    ax[0, 0].plot(ep, history['train_mae'], 'b-', label='Train weighted mm')
    ax[0, 0].plot(ep, history['val_mae'], 'r-', label='Val weighted mm')
    ax[0, 0].set_xlabel('Epoch'); ax[0, 0].set_ylabel('Weighted mm')
    ax[0, 0].set_title('Weighted mm (train=val, unified)')
    ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)
    # 2. 平均位移 MAE (mm)
    ax[0, 1].set_xlabel('Epoch'); ax[0, 1].set_ylabel('Peak/High (mm)')
    ax[0, 1].set_title('Val components')
    if 'val_peak_mae' in history and len(history['val_peak_mae']) == len(ep):
        ax[0, 1].plot(ep, history['val_peak_mae'], 'm-', label='Val Peak MAE (mm)')
    if 'val_high_mae' in history and len(history['val_high_mae']) == len(ep):
        ax[0, 1].plot(ep, history['val_high_mae'], 'c-', label='Val High-penalty (mm)')
    ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3)
    # 3. LR
    ax[1, 0].plot(ep, history['lr'], 'g-')
    ax[1, 0].set_xlabel('Epoch'); ax[1, 0].set_ylabel('LR')
    ax[1, 0].set_yscale('log'); ax[1, 0].grid(alpha=0.3)
    # 4. 占位 (说明)
    ax[1, 1].axis('off')
    ax[1, 1].text(0.05, 0.5,
                  'weighted mm = mean_disp_mm\n'
                  '             + w_peak * peak_mm\n'
                  '             + high_penalty_mm\n\n'
                  'train 与 val 用同一公式, 可直接对比',
                  fontsize=10, va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'training_curves.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='模型训练 (普通回归, 真实地震动输入)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--out', type=str, default='./models')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复')
    parser.add_argument('--no_physics', action='store_true', help='关闭物理正则(默认关闭,本版未用)')
    parser.add_argument('--no_bypass', action='store_true',
                        help='关闭残差旁路 (默认开启, 改善收敛)')
    parser.add_argument('--use_v2', action='store_true',
                        help='使用 v2 现代架构 (默认按 config.USE_V2)')
    parser.add_argument('--no_v2', action='store_true',
                        help='强制使用旧架构 (覆盖 config.USE_V2)')
    parser.add_argument('--use_db', action='store_true',
                        help='从 PostgreSQL 数据库读取数据 (替代 pkl 缓存)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='数据库模式: 最多随机取多少个样本训练 (默认 config.DB_MAX_SAMPLES)')
    parser.add_argument('--params', type=str, default=None,
                        help='Optuna 最优超参 JSON 路径 (models_optuna/best_hyperparams.json); '
                             '读取后写回 cfg 再训练')
    args = parser.parse_args()
    args.resume = True
    cfg = Config()
    if args.batch:
        cfg.BATCH_SIZE = args.batch
    if args.max_samples:
        cfg.DB_MAX_SAMPLES = args.max_samples
    # 从 Optuna 结果加载最优超参 (覆盖 config 默认值)
    if args.params:
        import json as _json
        if os.path.exists(args.params):
            with open(args.params, 'r', encoding='utf-8') as _f:
                hp = _json.load(_f)
            best = hp.get('best_params', hp)
            # 映射: Optuna 搜索名 -> cfg 属性
            _map = {
                'lr': 'LEARNING_RATE',
                'weight_decay': 'WEIGHT_DECAY',
                'batch_size': 'BATCH_SIZE',
                'd_model': 'D_MODEL',
                'n_layer': 'N_LAYER',
                'n_head': 'N_HEAD',
                'd_ff': 'D_FF',
                'dropout': 'DROPOUT',
                'v2_drop_path': 'V2_DROP_PATH',
                'v2_conv_kernel': 'V2_CONV_KERNEL',
                'warmup_epochs': 'LR_WARMUP_EPOCHS',
                'loss_peak_w': 'LOSS_PEAK_W',
                'loss_high_w': 'LOSS_HIGH_W',
                'loss_high_thresh': 'LOSS_HIGH_THRESH_MM',
            }
            for k, attr in _map.items():
                if k in best:
                    setattr(cfg, attr, best[k])
            print(f"🎯 已加载 Optuna 最优超参: {args.params}")
            print(f"   lr={best.get('lr')}, d_model={best.get('d_model')}, "
                  f"n_layer={best.get('n_layer')}, n_head={best.get('n_head')}, "
                  f"d_ff={best.get('d_ff')}")
            if 'val_mae_mm' in hp:
                print(f"   寻优验证 MAE={hp.get('val_mae_mm')}, R²={hp.get('val_r2')}")
        else:
            print(f"  ⚠️ 超参文件不存在: {args.params}, 使用 config 默认值")
    use_v2 = None
    if args.use_v2:
        use_v2 = True
    elif args.no_v2:
        use_v2 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train(cfg, device, args.out, resume=args.resume,
                 epochs=args.epochs, lr=args.lr, use_bypass=not args.no_bypass,
                 use_v2=use_v2, use_db=args.use_db)


if __name__ == '__main__':
    main()
