# optuna_search.py
"""
基于 Optuna 的超参数寻优脚本 —— 底层复用 train.py 的训练逻辑

功能:
1. 用 Optuna 对 SLFormer (v2) 的超参数做贝叶斯搜索 (TPE)
2. 搜索范围覆盖:
   - 优化器: 学习率 / 权重衰减 / batch
   - 模型结构: D_MODEL / N_LAYER / N_HEAD / D_FF / DROPOUT / V2_DROP_PATH / V2_CONV_KERNEL
   - 调度: warmup 轮数
   - 损失: LOSS_PEAK_W / LOSS_HIGH_W / LOSS_HIGH_THRESH_MM
3. 每次 trial: 用固定 train/val 划分, 训练固定轮数 (可缩短加速搜索), 返回验证指标
4. 目标指标: 默认最小化 验证 MAE (mm); 可用 --objective r2 改为最大化验证 R²
5. 输出:
   - optuna 数据库 (sqlite)  便于续跑/可视化
   - 最优超参 JSON + 训练好的最优模型 (models_optuna/best)
   - 寻优过程图 (寻优历史 / 参数重要性 / 平行坐标)

用法:
    python optuna_search.py --trials 30 --epochs 30 --objective mae
    python optuna_search.py --trials 30 --epochs 50 --objective r2 --use_db --max_samples 10000
    python optuna_search.py --study optuna.db --trials 20  # 续跑

说明:
    - 数据加载与 train 完全一致 (build_loaders), 支持 --use_db
    - 每次 trial 模型重新构建, 训练结束后释放显存, 避免 OOM
    - 与 train.py 共享 normalize_disp / build_loaders
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from transformer_model import SLFormer
from train import build_loaders, normalize_disp, weighted_mm_metric, \
    get_loss_thresh, get_loss_norm


# ============================================================
# 精简训练函数 (复用 train 的 loss / 调度逻辑)
# ============================================================
def train_with_trial(cfg, device, trial, tr_loader, va_loader, disp_std,
                     out_dir, epochs):
    """按 trial 建议的超参训练模型, 返回 (val_mae_mm, val_r2)"""
    # ---------- 从 trial 读取超参 (注入 cfg) ----------
    cfg.LEARNING_RATE = trial.suggest_float('lr', 5e-5, 5e-3, log=True)
    cfg.WEIGHT_DECAY = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    cfg.BATCH_SIZE = trial.suggest_categorical('batch_size', [16, 32, 64])
    # 模型结构
    cfg.D_MODEL = trial.suggest_categorical('d_model', [128, 256, 384])
    cfg.N_LAYER = trial.suggest_int('n_layer', 2, 6, step=2)
    cfg.N_HEAD = trial.suggest_categorical('n_head', [4, 8])
    cfg.D_FF = trial.suggest_categorical('d_ff', [256, 512, 768])
    cfg.DROPOUT = trial.suggest_float('dropout', 0.0, 0.3)
    cfg.V2_DROP_PATH = trial.suggest_float('v2_drop_path', 0.0, 0.2)
    cfg.V2_CONV_KERNEL = trial.suggest_categorical('v2_conv_kernel', [15, 31, 51])
    # 调度 warmup
    cfg.LR_WARMUP_EPOCHS = trial.suggest_int('warmup_epochs', 0, 10)
    # 损失权重 (默认 no_peak: 不搜峰值权重, 只搜大位移加权)
    cfg.LOSS_HIGH_W = trial.suggest_float('loss_high_w', 2.0, 8.0)
    # 大位移阈值: relative 模式用峰值比例, absolute 用绝对 mm
    if get_loss_norm(cfg) == 'relative':
        cfg.LOSS_HIGH_THRESH_RATIO = trial.suggest_float(
            'loss_high_thresh_ratio', 0.1, 0.5)
    else:
        cfg.LOSS_HIGH_THRESH_MM = trial.suggest_float('loss_high_thresh', 4.0, 12.0)

    # ---------- 构建模型 ----------
    use_v2 = bool(getattr(cfg, 'USE_V2', True))
    # 默认关闭自注意力 (no_sa — 消融结果显示更好)
    model = SLFormer(cfg, use_bypass=True, use_v2=use_v2,
                     use_sa=False,
                     drop_path=cfg.V2_DROP_PATH,
                     film=getattr(cfg, 'V2_FILM', True)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trial.set_user_attr('n_params', n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE,
                                  weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.95))
    # 调度: warmup + cosine
    warmup_epochs = max(0, min(cfg.LR_WARMUP_EPOCHS, max(epochs // 5, 0)))
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)

    # 混合精度
    use_amp = (torch.cuda.is_available()
               and bool(getattr(cfg, 'USE_MIXED_PRECISION', True)))
    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler('cuda')
        except Exception:
            scaler = torch.cuda.amp.GradScaler()

    # ---------- 训练循环 ----------
    model.train()
    for epoch in range(epochs):
        for batch in tr_loader:
            oct = batch['octree_features'].to(device, non_blocking=True)
            motion = batch['motion'].to(device, non_blocking=True)
            disp = batch['disp'].to(device, non_blocking=True)
            params = batch['params'].to(device, non_blocking=True)
            ffeat = (batch['frame_features'].to(device, non_blocking=True)
                     if batch.get('frame_features') is not None else None)

            w_peak = float(cfg.LOSS_PEAK_W)
            w_high = float(cfg.LOSS_HIGH_W)
            thresh = get_loss_thresh(cfg)
            loss_norm = get_loss_norm(cfg)

            def step():
                optimizer.zero_grad(set_to_none=True)
                pred, _ = model(oct, motion.unsqueeze(-1), cond_params=params,
                                frame_features=ffeat)
                # 与 train 完全一致的加权 mm 损失 (默认 no_peak: 关峰值惩罚)
                loss, _, _, _ = weighted_mm_metric(
                    pred, disp, w_peak=w_peak, w_high=w_high,
                    thresh_mm=thresh, loss_norm=loss_norm,
                    use_peak=False, use_shape=True, use_high=True)
                return loss

            if scaler is not None:
                with torch.autocast(device_type='cuda'):
                    loss = step()
            else:
                loss = step()
            if not torch.isfinite(loss):
                continue
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
        scheduler.step()

    # ---------- 验证 (与 train 完全一致的加权 mm 指标) ----------
    model.eval()
    all_pred, all_tgt = [], []
    v_score_sum = 0.0
    v_mae_sum = 0.0
    v_peak_sum = 0.0
    n_batch = 0
    w_peak = float(getattr(cfg, 'LOSS_PEAK_W', 1.0))
    w_high = float(getattr(cfg, 'LOSS_HIGH_W', 3.0))
    thresh = get_loss_thresh(cfg)
    loss_norm = get_loss_norm(cfg)
    with torch.no_grad():
        for batch in va_loader:
            oct = batch['octree_features'].to(device)
            motion = batch['motion'].to(device)
            target = batch['disp'].to(device)
            params = batch['params'].to(device)
            ffeat = (batch['frame_features'].to(device)
                     if batch.get('frame_features') is not None else None)
            pred, _ = model(oct, motion.unsqueeze(-1), cond_params=params,
                            frame_features=ffeat)
            all_pred.append(pred.cpu().numpy())
            all_tgt.append(target.cpu().numpy())
            # 加权 mm 分数 (与 train 相同公式, no_peak: 关峰值惩罚)
            vs, vm, vp, vh = weighted_mm_metric(
                pred, target, w_peak=w_peak, w_high=w_high,
                thresh_mm=thresh, loss_norm=loss_norm,
                use_peak=False, use_shape=True, use_high=True)
            v_score_sum += vs.item()
            v_mae_sum += vm.item()
            v_peak_sum += vp.item()
            n_batch += 1
    pred = np.concatenate(all_pred)
    tgt = np.concatenate(all_tgt)
    flat_p, flat_t = pred.flatten(), tgt.flatten()
    mae_mm = float(np.mean(np.abs(flat_p - flat_t)))
    # 峰值 MAE (mm): 每样本预测峰值 vs 真值峰值
    peak_pred = np.max(np.abs(pred), axis=1)
    peak_true = np.max(np.abs(tgt), axis=1)
    peak_mae_mm = float(np.mean(np.abs(peak_pred - peak_true)))
    # 加权 mm 分数 (与 train best 选择标准一致)
    score = v_score_sum / max(n_batch, 1)
    # R² (参考)
    ss_res = np.sum((flat_t - flat_p) ** 2)
    ss_tot = np.sum((flat_t - flat_t.mean()) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    # 释放显存
    del model, optimizer, scheduler, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mae_mm, peak_mae_mm, score, r2


# ============================================================
# Optuna objective
# ============================================================
def make_objective(cfg, device, tr_loader, va_loader, disp_std, epochs,
                   objective='score', work_dir=None):
    def objective(trial):
        t0 = time.time()
        # 每个 trial 独立输出目录 (便于排查)
        trial_dir = os.path.join(work_dir, f'trial_{trial.number:03d}') if work_dir else None
        if trial_dir:
            os.makedirs(trial_dir, exist_ok=True)
        mae_mm, peak_mae_mm, score, r2 = train_with_trial(
            cfg, device, trial, tr_loader, va_loader, disp_std,
            trial_dir, epochs)
        # 记录到 trial
        trial.set_user_attr('val_mae_mm', mae_mm)
        trial.set_user_attr('val_peak_mae_mm', peak_mae_mm)
        trial.set_user_attr('val_score', score)
        trial.set_user_attr('val_r2', r2)
        trial.set_user_attr('time_s', time.time() - t0)
        print(f"    [Trial {trial.number}] MAE={mae_mm:.4f}mm PeM={peak_mae_mm:.4f}mm "
              f"Score={score:.4f} R²={r2:.4f} time={time.time()-t0:.0f}s")
        # 目标 (与 train best 一致):
        #   score = 整体 MAE + w_peak*峰值 MAE (最小化)
        #   r2     = 整体 R² (最大化)
        if objective == 'r2':
            return r2
        return score
    return objective


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Optuna 超参寻优 (复用 train)')
    parser.add_argument('--trials', type=int, default=20,
                        help='寻优 trial 数 (默认20)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='每个 trial 训练轮数 (默认25, 加速搜索可调小)')
    parser.add_argument('--objective', type=str, default='score',
                        choices=['score', 'mae', 'r2'],
                        help='优化目标: score=加权mm(no_peak,与train一致), '
                             'mae=仅整体MAE, r2=最大化整体R²')
    parser.add_argument('--use_db', action='store_true',
                        help='从 PostgreSQL 读取数据 (与 train --use_db 一致)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='数据库模式样本上限')
    parser.add_argument('--study', type=str, default=None,
                        help='Optuna 数据库文件 (默认 ./optuna.db, 续跑用同文件)')
    parser.add_argument('--out', type=str, default='./models_optuna',
                        help='最优结果输出目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()

    cfg = Config()
    if args.max_samples:
        cfg.DB_MAX_SAMPLES = args.max_samples

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---------- 数据 (只加载一次, 所有 trial 共用同一 train/val 划分) ----------
    print("\n加载数据集 (与 train 一致)...")
    tr_loader, va_loader, dataset = build_loaders(cfg, use_db=args.use_db)
    if tr_loader is None:
        print("  [X] 数据加载失败")
        return
    n_train = len(tr_loader.dataset)
    n_val = len(va_loader.dataset)
    disp_std = float(np.maximum(dataset.displacements.std(), 1e-6))
    print(f"  训练 {n_train}, 验证 {n_val}, 特征 {cfg.OCTREE_FEATURE_DIM}, "
          f"目标 std={disp_std:.3f} mm")

    # ---------- Optuna 数据库 ----------
    study_name = f'optuna_{args.objective}'
    db_file = args.study or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'optuna.db')
    storage = f"sqlite:///{db_file}"
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, 'trials'), exist_ok=True)

    import optuna
    # 用 R² 时最大化: 包一层负号
    if args.objective == 'r2':
        direction = 'maximize'
    else:
        direction = 'minimize'

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    print(f"\nStudy: {study_name} (direction={direction}), 已存在 trials: {len(study.trials)}")
    print(f"开始寻优 {args.trials} 个 trial...")

    # 记录每个 trial 的结果到 CSV
    def objective_wrapper(trial):
        t0 = time.time()
        trial_dir = os.path.join(args.out, 'trials', f'trial_{trial.number:03d}')
        os.makedirs(trial_dir, exist_ok=True)
        mae_mm, peak_mae_mm, score, r2 = train_with_trial(
            cfg, device, trial, tr_loader, va_loader, disp_std,
            trial_dir, args.epochs)
        trial.set_user_attr('val_mae_mm', mae_mm)
        trial.set_user_attr('val_peak_mae_mm', peak_mae_mm)
        trial.set_user_attr('val_score', score)
        trial.set_user_attr('val_r2', r2)
        trial.set_user_attr('time_s', time.time() - t0)
        print(f"    [Trial {trial.number}] MAE={mae_mm:.4f}mm PeM={peak_mae_mm:.4f}mm "
              f"Score={score:.4f} R²={r2:.4f} time={time.time()-t0:.0f}s")
        if args.objective == 'r2':
            return r2
        if args.objective == 'mae':
            return mae_mm
        return score

    study.optimize(objective_wrapper, n_trials=args.trials)

    # ---------- 结果 ----------
    best = study.best_trial
    print("\n" + "=" * 70)
    print(f"寻优完成! 最优 Trial #{best.number}")
    print("=" * 70)
    print("最优超参:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"\n  验证 MAE:   {best.user_attrs.get('val_mae_mm', 'N/A'):.4f} mm")
    print(f"  验证峰值MAE: {best.user_attrs.get('val_peak_mae_mm', 'N/A'):.4f} mm")
    print(f"  验证 Score:  {best.user_attrs.get('val_score', 'N/A'):.4f}")
    print(f"  验证 R²:    {best.user_attrs.get('val_r2', 'N/A'):.4f}")
    print(f"  参数量:     {best.user_attrs.get('n_params', 'N/A'):,}")
    print(f"  用时:       {best.user_attrs.get('time_s', 'N/A'):.0f}s")

    # 保存最优超参
    best_json = os.path.join(args.out, 'best_hyperparams.json')
    with open(best_json, 'w', encoding='utf-8') as f:
        json.dump({
            'best_trial': best.number,
            'best_params': best.params,
            'val_mae_mm': best.user_attrs.get('val_mae_mm'),
            'val_peak_mae_mm': best.user_attrs.get('val_peak_mae_mm'),
            'val_score': best.user_attrs.get('val_score'),
            'val_r2': best.user_attrs.get('val_r2'),
            'n_params': best.user_attrs.get('n_params'),
            'objective': args.objective,
            'epochs_per_trial': args.epochs,
            'db_file': db_file,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n💾 最优超参已保存: {best_json}")

    # ---------- 寻优过程图 ----------
    plot_dir = os.path.join(args.out, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    _plot_study(study, plot_dir, args.objective)
    print(f"📊 寻优过程图: {plot_dir}")
    print(f"\n[OK] 完成! 用以下超参训练最终模型:")
    print(f"     python train.py --use_db --lr {best.params['lr']:.2e} "
          f"--epochs 200")
    return study


def _plot_study(study, plot_dir, objective):
    """绘制寻优历史 / 参数重要性 / 平行坐标 (PNG 需 kaleido; 无则只存 HTML)"""
    import optuna.visualization as vis

    def _save(fig, name):
        # HTML 始终保存 (交互式, 无需额外依赖)
        try:
            fig.write_html(os.path.join(plot_dir, f'{name}.html'))
        except Exception as e:
            print(f"  [W] {name} HTML 失败: {e}")
        # PNG 需要 kaleido, 失败不阻塞
        try:
            fig.write_image(os.path.join(plot_dir, f'{name}.png'),
                            width=1100, height=650, scale=2)
        except Exception:
            pass

    # 1. 寻优历史 (objective 值 vs trial)
    try:
        _save(vis.plot_optimization_history(study), 'optimization_history')
    except Exception as e:
        print(f"  [W] 历史图失败: {e}")

    # 2. 参数重要性
    try:
        _save(vis.plot_param_importances(study), 'param_importance')
    except Exception as e:
        print(f"  [W] 重要性图失败: {e}")

    # 3. 平行坐标
    try:
        _save(vis.plot_parallel_coordinate(study), 'parallel_coordinate')
    except Exception as e:
        print(f"  [W] 平行坐标失败: {e}")


if __name__ == '__main__':
    main()
