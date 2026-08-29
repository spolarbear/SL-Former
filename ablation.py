# ablation.py
"""
消融试验脚本 (基于 train.py 的训练逻辑, 复用现有代码/数据/模型)

设计原则:
1. 不重写训练函数: 直接复用 train.train (通过 model_kwargs / loss_kwargs 控制消融)
2. 复用现有数据: 与 train 完全一致的 build_loaders (支持 --use_db / 分层抽样)
3. 每种消融独立输出目录 (模型 + 训练记录 + 评估图表)
4. 全部完成后输出汇总表 (ablation_summary.csv/json)

消融维度 (基于 v2 SLFormer + 训练损失):
  设计原则: 用户要求基准模型也改用 token 编码。
  (a) 有/无 token 编码对比:
     - full_token   : 完整模型 + token 编码 (微元词表+Embedding) —— 本文基准
     - no_token     : 完整模型 + 无 token (44 维杆系物理特征, 传统 frame_features)
     两者同模型/同数据/同损失, 仅结构编码器不同 (token vs 物理特征)
  (b) token 编码基础上做其他消融 (全部继承 full_token 的 token 基础):
     A. 模型结构消融 (model_kwargs 控制, 均基于 token):
        - no_sa             : 去掉自注意力 (use_sa=False)
        - no_cross_attn     : 去掉结构交叉注意力 (use_cross_attn=False)
        - no_film           : 去掉 FiLM 结构条件注入 (use_struct_fusion=False)
        - no_conv           : 去掉局部卷积 (use_conv=False)
        - no_ffn            : 去掉 SwiGLU FFN (use_ffn=False)
        - no_cond_params    : 去掉显式结构参数条件注入 (use_cond_params=False)
        - no_bypass         : 去掉 motion->位移 残差旁路 (use_bypass=False)
     B. 输入/数据消融 (均基于 token):
        - no_motion         : 不用真实地震动输入 (motion 置零)
        - no_struct_feat    : 不用结构特征 (token 置零)
     C. 损失消融 (loss_kwargs 控制, 均基于 token):
        - no_peak_loss      : 去掉峰值惩罚 (peak=False)
        - no_high_loss      : 去掉大位移加权 (high=False)
        - no_shape_loss     : 去掉波形形状损失 (shape=False)
     D. 结构编码方式消融 (与 full_token 对照, 同模型/同数据/同损失):
        - enc_direct        : 直接 32 位整数编码归一化 + MLP (上一版本简单直接编码)
        - enc_cont          : 6 通道连续物理量特征 + MLP (LLM embedding 启发)

用法:
    python ablation.py                        # 跑全部消融
    python ablation.py --only full_token,no_sa,no_bypass
    python ablation.py --only full_token,no_token          # 只跑有/无 token 对比
    python ablation.py --only enc_token,enc_direct,enc_cont   # 只跑编码消融
    python ablation.py --epochs 60 --batch 64 --use_db --max_samples 20000
    python ablation.py --epochs 200 --use_db --max_samples 20000
    python ablation.py --resume               # 跳过已完成 / 恢复未完成
    python ablation.py --out ./models_ablation
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from train import train, build_loaders


# ============================================================
# 消融配置定义
# ============================================================
def _token_base(**extra):
    """token 编码消融基础: 所有基于 token 的消融共享此设置"""
    base = dict(use_voxel_feature=True, enc_mode='token', input_mode='normal',
                model_kwargs={'use_voxel_token': True, 'vocab_size': 'auto'})
    base.update(extra)
    return base


def get_ablation_configs():
    """返回消融配置列表。

    每个配置 dict:
        name         : 唯一名称 (目录名)
        desc         : 中文描述 (用于汇总表)
        model_kwargs : 传给 SLFormer 的消融开关
        loss_kwargs  : 损失项开关
        input_mode   : 'normal' / 'no_motion' / 'no_struct'
        use_voxel_feature: True 用切杆系编码特征 (token/direct/cont)
        enc_mode     : 'token' / 'direct' / 'cont' / None(无token, 44维物理特征)
    """
    # (a) 有/无 token 编码对比的基准
    full_token = dict(
        name='full_token', desc='完整模型 + Token 编码 (本文基准)',
        model_kwargs={'use_voxel_token': True, 'vocab_size': 'auto'},
        loss_kwargs={}, input_mode='normal', use_voxel_feature=True,
        enc_mode='token',
    )
    no_token = dict(
        name='no_token', desc='完整模型 + 无 Token (44维物理特征)',
        model_kwargs={}, loss_kwargs={}, input_mode='normal',
        use_voxel_feature=False, enc_mode=None,
    )
    # 结构编码方式消融: 同模型/同数据/同损失, 只换结构编码器
    enc_ablations = [
        dict(name='enc_direct', desc='结构编码: 直接 32 位整数归一化 + MLP (上一版)',
             model_kwargs={}, loss_kwargs={}, input_mode='normal',
             use_voxel_feature=True, enc_mode='direct'),
        dict(name='enc_cont', desc='结构编码: 6 通道连续物理量 + MLP (embedding 启发)',
             model_kwargs={}, loss_kwargs={}, input_mode='normal',
             use_voxel_feature=True, enc_mode='cont'),
    ]
    return [
        # ---------- (a) 有/无 token 对比 ----------
        full_token,
        no_token,
        # ---------- (b) token 基础上的 A. 模型结构消融 ----------
        _token_base(name='no_sa', desc='去掉自注意力',
                    model_kwargs={'use_voxel_token': True, 'vocab_size': 'auto',
                                  'use_sa': False}),
        _token_base(name='no_cross_attn', desc='去掉结构交叉注意力',
                    model_kwargs={'use_voxel_token': True, 'vocab_size': 'auto',
                                  'use_cross_attn': False}),
        _token_base(name='no_film', desc='去掉 FiLM 结构条件注入',
                    model_kwargs={'use_voxel_token': True, 'vocab_size': 'auto',
                                  'use_struct_fusion': False}),
        _token_base(name='no_conv', desc='去掉局部卷积',
                    model_kwargs={'use_voxel_token': True, 'vocab_size': 'auto',
                                  'use_conv': False}),
        _token_base(name='no_ffn', desc='去掉 SwiGLU FFN',
                    model_kwargs={'use_voxel_token': True, 'vocab_size': 'auto',
                                  'use_ffn': False}),
        _token_base(name='no_cond_params', desc='去掉结构参数条件注入',
                    model_kwargs={'use_voxel_token': True, 'vocab_size': 'auto',
                                  'use_cond_params': False}),
        _token_base(name='no_bypass', desc='去掉 motion->位移 残差旁路',
                    use_bypass=False,
                    model_kwargs={'use_voxel_token': True, 'vocab_size': 'auto'}),
        # ---------- (b) token 基础上的 B. 输入/数据消融 ----------
        _token_base(name='no_motion', desc='不用真实地震动输入 (置零)',
                    input_mode='no_motion'),
        _token_base(name='no_struct_feat', desc='不用结构特征 (token 置零)',
                    input_mode='no_struct'),
        # ---------- (b) token 基础上的 C. 损失消融 ----------
        _token_base(name='no_peak_loss', desc='去掉峰值惩罚',
                    loss_kwargs=dict(peak=False)),
        _token_base(name='no_high_loss', desc='去掉大位移加权',
                    loss_kwargs=dict(high=False)),
        _token_base(name='no_shape_loss', desc='去掉波形形状损失',
                    loss_kwargs=dict(shape=False)),
        # ---------- D. 结构编码方式消融 (与 full_token 对照) ----------
        *enc_ablations,
    ]


# ============================================================
# 单消融运行 (包装 train, 处理 input_mode / enc_mode)
# ============================================================
def run_one_ablation(cfg, device, ab_cfg, out_dir, resume, epochs, batch, use_db):
    """运行单个消融配置, 返回评估指标 dict"""
    # 应用 batch 覆盖
    if batch:
        cfg.BATCH_SIZE = batch

    # 结构编码消融: 需设置 voxel_enc_mode + use_voxel_feature + model_kwargs
    enc_mode = ab_cfg.get('enc_mode')
    use_voxel_feature = bool(ab_cfg.get('use_voxel_feature', False))

    # 重置特征维度 (防止上个消融运行污染: dataset 会把 cfg.FRAME_FEATURE_DIM 改成
    # 32768/196608, 若不重置, 后续无 token 消融会错误用错维度)
    _cfg_ffd_default = getattr(Config, 'FRAME_FEATURE_DIM', 44)
    cfg.FRAME_FEATURE_DIM = _cfg_ffd_default

    # token 模式: 需构建/加载微元词表, 并把 vocab_size 注入 model_kwargs
    model_kwargs = dict(ab_cfg.get('model_kwargs') or {})
    if enc_mode == 'token':
        if model_kwargs.get('vocab_size') == 'auto' or 'vocab_size' not in model_kwargs:
            from frame_grid_encoder import VoxelVocab
            vf = getattr(cfg, 'VOXEL_VOCAB_FILE', None)
            vocab = VoxelVocab()
            if vf and os.path.exists(vf):
                vocab.load(vf)
                n_tok = len(vocab.id2micro)
            else:
                # 从数据库构建 (与 train_voxel 一致)
                from frame_grid_encoder import build_voxel_vocab_from_db
                from db_manager import SLFDatabase
                db = SLFDatabase()
                n_scan = int(getattr(cfg, 'VOXEL_VOCAB_SCAN_STRUCTS', 2000))
                vocab, n_tok = build_voxel_vocab_from_db(db, n_structs=n_scan)
                db.close()
                if vf:
                    try:
                        vocab.save(vf)
                    except Exception:
                        pass
            cfg.VOXEL_VOCAB_SIZE = n_tok
            model_kwargs['vocab_size'] = n_tok
        model_kwargs['use_voxel_token'] = True

    # 处理输入消融: no_motion / no_struct 需要包装数据集, 这里通过参数传递
    # 简化: 复用 train, 输入消融由下方 custom 逻辑处理 (见 train_ablation)
    import train as tc

    # no_motion / no_struct: 需要对 batch 做 mask, 走定制循环
    input_mode = ab_cfg.get('input_mode', 'normal')
    use_bypass = ab_cfg.get('use_bypass', True)

    if input_mode == 'normal':
        # 直接用 train (最省事, 完全复用), 透传结构编码消融参数
        res = train(
            cfg, device, out_dir, resume=resume, epochs=epochs,
            use_bypass=use_bypass, use_v2=True, use_db=use_db,
            use_voxel_feature=use_voxel_feature,
            voxel_enc_mode=enc_mode,
            model_kwargs=model_kwargs or None,
            loss_kwargs=ab_cfg.get('loss_kwargs'),
        )
        # 补充记录编码消融元信息 (feat_dim / enc_mode / vocab_size)
        # 到 training_info.json (供汇总对比表读取)
        if enc_mode is not None:
            try:
                info_p = os.path.join(out_dir, 'training_info.json')
                info = {}
                if os.path.exists(info_p):
                    with open(info_p, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                info['enc_mode'] = enc_mode
                from config import Config as _Cfg
                _vgrid = int(getattr(_Cfg, 'VOXEL_GRID', 64))
                if enc_mode == 'cont':
                    from frame_grid_encoder import FEAT_C
                    info['feat_dim'] = _vgrid ** 3 * FEAT_C
                else:
                    info['feat_dim'] = _vgrid ** 3
                if 'vocab_size' in model_kwargs:
                    info['vocab_size'] = model_kwargs['vocab_size']
                with open(info_p, 'w', encoding='utf-8') as f:
                    json.dump(info, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
        return res
    else:
        # 输入消融需要定制 train loop (mask motion 或 struct), 复用 train 的组件
        return _train_input_ablation(
            cfg, device, out_dir, ab_cfg, resume, epochs, use_db, input_mode)


def _train_input_ablation(cfg, device, out_dir, ab_cfg, resume, epochs, use_db,
                          input_mode):
    """输入消融 (no_motion / no_struct): 复用 build_loaders + 精简训练循环

    也支持结构编码消融 (enc_token/direct/cont) 与输入消融正交组合:
    需透传 use_voxel_feature + voxel_enc_mode + token vocab_size。
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformer_model import SLFormer
    from train import normalize_disp, weighted_mm_metric, \
        get_loss_thresh, get_loss_norm

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)

    # 结构编码消融参数透传
    enc_mode = ab_cfg.get('enc_mode')
    use_voxel_feature = bool(ab_cfg.get('use_voxel_feature', False))
    model_kwargs = dict(ab_cfg.get('model_kwargs') or {})
    if enc_mode == 'token':
        if 'vocab_size' not in model_kwargs or model_kwargs.get('vocab_size') == 'auto':
            from frame_grid_encoder import VoxelVocab
            vf = getattr(cfg, 'VOXEL_VOCAB_FILE', None)
            vocab = VoxelVocab()
            if vf and os.path.exists(vf):
                vocab.load(vf)
                n_tok = len(vocab.id2micro)
            else:
                from frame_grid_encoder import build_voxel_vocab_from_db
                from db_manager import SLFDatabase
                db = SLFDatabase()
                n_scan = int(getattr(cfg, 'VOXEL_VOCAB_SCAN_STRUCTS', 2000))
                vocab, n_tok = build_voxel_vocab_from_db(db, n_structs=n_scan)
                db.close()
                if vf:
                    try:
                        vocab.save(vf)
                    except Exception:
                        pass
            cfg.VOXEL_VOCAB_SIZE = n_tok
            model_kwargs['vocab_size'] = n_tok
        model_kwargs['use_voxel_token'] = True

    tr_loader, va_loader, dataset = build_loaders(
        cfg, use_db=use_db, use_voxel_feature=use_voxel_feature,
        voxel_enc_mode=enc_mode)
    if tr_loader is None:
        return None

    # 模型 (v2), 透传结构编码消融
    model = SLFormer(cfg, use_bypass=True, use_v2=True,
                     drop_path=getattr(cfg, 'V2_DROP_PATH', 0.1),
                     film=getattr(cfg, 'V2_FILM', True),
                     **model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE,
                                  weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.95))
    total_epochs = epochs or cfg.EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # 常量 mask 张量
    no_motion = input_mode == 'no_motion'
    no_struct = input_mode == 'no_struct'
    zeros_B1 = torch.zeros((1,), device=device)
    w_peak = float(getattr(cfg, 'LOSS_PEAK_W', 1.0))
    w_high = float(getattr(cfg, 'LOSS_HIGH_W', 3.0))
    thresh = get_loss_thresh(cfg)
    loss_norm = get_loss_norm(cfg)

    model.train()
    for epoch in range(total_epochs):
        for batch in tr_loader:
            oct = batch['octree_features'].to(device)
            motion = batch['motion'].to(device)
            disp = batch['disp'].to(device)
            params = batch['params'].to(device)
            ffeat = (batch['frame_features'].to(device)
                     if batch.get('frame_features') is not None else None)
            # 输入消融
            if no_motion:
                motion = torch.zeros_like(motion)
            if no_struct:
                oct = torch.zeros_like(oct)
                ffeat = None

            def step():
                optimizer.zero_grad(set_to_none=True)
                pred, _ = model(oct, motion.unsqueeze(-1), cond_params=params,
                                frame_features=ffeat)
                # 与 train 一致的加权 mm 损失
                loss, _, _, _ = weighted_mm_metric(
                    pred, disp, w_peak=w_peak, w_high=w_high,
                    thresh_mm=thresh, loss_norm=loss_norm)
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

    # 验证 (mm MAE + R² + 峰值 + 相对指标, 与 train.evaluate_after_train 一致口径)
    model.eval()
    all_pred, all_tgt = [], []
    with torch.no_grad():
        for batch in va_loader:
            oct = batch['octree_features'].to(device)
            motion = batch['motion'].to(device)
            target = batch['disp'].to(device)
            params = batch['params'].to(device)
            ffeat = (batch['frame_features'].to(device)
                     if batch.get('frame_features') is not None else None)
            if no_motion:
                motion = torch.zeros_like(motion)
            if no_struct:
                oct = torch.zeros_like(oct)
                if ffeat is not None:
                    ffeat = torch.zeros_like(ffeat)
            pred, _ = model(oct, motion.unsqueeze(-1), cond_params=params,
                            frame_features=ffeat)
            all_pred.append(pred.cpu().numpy())
            all_tgt.append(target.cpu().numpy())
    pred = np.concatenate(all_pred)
    tgt = np.concatenate(all_tgt)

    # ---- 与 train.evaluate_after_train 一致的多指标 ----
    def _r2(a, b):
        ss_res = np.sum((a - b) ** 2)
        ss_tot = np.sum((a - a.mean()) ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-12))

    mae_mm = float(np.mean(np.abs(pred - tgt)))
    rmse = float(np.sqrt(np.mean((pred - tgt) ** 2)))
    r2 = _r2(tgt, pred)
    # 峰值
    peak_t = np.abs(tgt).max(axis=1)
    peak_p = np.abs(pred).max(axis=1)
    peak_mae = float(np.mean(np.abs(peak_p - peak_t)))
    peak_r2 = _r2(peak_t, peak_p)
    # 相对误差 (每样本按峰值归一化)
    denom = np.maximum(peak_t, 1e-3)
    rel_mae = float(np.mean(np.mean(np.abs(pred - tgt) / denom[:, None], axis=1)))
    peak_rel_mae_pct = float(np.mean(np.abs(peak_p - peak_t) / denom) * 100.0)
    # 大/小位移分组
    med_peak = float(np.median(peak_t))
    small = peak_t < med_peak
    large = ~small
    r2_small_all = _r2(tgt[small], pred[small]) if small.sum() > 1 else float('nan')
    r2_large_all = _r2(tgt[large], pred[large]) if large.sum() > 1 else float('nan')

    # 加权 mm 分数 (与 train 一致)
    import torch as _t
    v_score, _, _, _ = weighted_mm_metric(
        _t.from_numpy(pred), _t.from_numpy(tgt),
        w_peak=w_peak, w_high=w_high, thresh_mm=thresh, loss_norm=loss_norm)
    wmm = float(v_score.item())

    # 保存模型与指标
    torch.save({'model_state_dict': model.state_dict()},
               os.path.join(out_dir, 'model', 'best_model.pth'))
    metrics = {
        'r2': r2, 'rmse': rmse, 'mae': mae_mm, 'peak_r2': peak_r2,
        'peak_mae': peak_mae, 'weighted_mm': wmm,
        'rel_mae': rel_mae, 'peak_rel_mae_pct': peak_rel_mae_pct,
        'r2_small_all': r2_small_all, 'r2_large_all': r2_large_all,
        'median_peak_mm': med_peak,
        'n_small': int(small.sum()), 'n_large': int(large.sum()),
    }
    with open(os.path.join(out_dir, 'eval_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


# ============================================================
# 汇总 / 结果读取
# ============================================================
def read_metrics(out_dir, name):
    """读取单消融的评估指标 (从 eval_metrics.json)"""
    p = os.path.join(out_dir, name, 'eval_metrics.json')
    if os.path.exists(p):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return None


def read_params(out_dir, name):
    """读取单消融的模型参数量 / 特征维度 (从 training_info.json)"""
    p = os.path.join(out_dir, name, 'training_info.json')
    if os.path.exists(p):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                info = json.load(f)
            return info.get('n_params'), info.get('feat_dim'), info.get('vocab_size')
        except Exception:
            pass
    return None, None, None


def _fmt_metrics(m):
    """把指标 dict 规整成汇总行 (含参数量/最佳epoch)"""
    if m is None:
        return {}
    out = {}
    for k, v in m.items():
        if isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, float):
            out[k] = round(float(v), 4)
        else:
            out[k] = v
    return out


def make_enc_ablation_summary(summary, out_root=None):
    """专门输出结构编码消融 (D 组) 对比表: full_token(tok) vs enc_direct vs enc_cont"""
    want = {'full_token', 'enc_direct', 'enc_cont'}
    rows = [s for s in summary if s.get('name') in want]
    if len(rows) < 2:
        # 至少要有 token 基准 + 一个对照
        token = [s for s in summary if s.get('name') == 'full_token']
        others = [s for s in summary if s.get('name') in ('enc_direct', 'enc_cont')]
        rows = token + others
        if len(rows) < 2:
            return None
    enc_names = {'full_token': 'token(本文)', 'enc_direct': 'direct(上一版)',
                 'enc_cont': 'cont(embedding启发)'}
    print("\n" + "=" * 70)
    print("结构编码消融对比 (D 组) — 同模型/同数据/同损失, 只换结构编码器")
    print("=" * 70)
    header = (f"{'模式':<16}{'参数量':>10}{'维度':>8}{'R²':>9}{'MAE(mm)':>10}"
              f"{'峰值R²':>9}{'峰值MAE':>10}{'相对MAE':>10}")
    print(header)
    print("-" * 70)
    for s in rows:
        nm = enc_names.get(s.get('name', ''), s.get('name', ''))
        params = s.get('n_params', '')
        dim = s.get('feat_dim', '')
        r2 = s.get('r2', '')
        mae = s.get('mae', s.get('mae_mm', ''))   # train 用 'mae'
        pr2 = s.get('peak_r2', '')
        pmae = s.get('peak_mae', '')
        rmae = s.get('rel_mae', '')
        print(f"{nm:<16}{str(params):>10}{str(dim):>8}{str(r2):>9}{str(mae):>10}"
              f"{str(pr2):>9}{str(pmae):>10}{str(rmae):>10}")
    # 保存专门对比 CSV (输出到 out_root, 默认 ./models_ablation)
    if out_root is None:
        out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'models_ablation')
    csv_path = os.path.join(out_root, 'ablation_enc_summary.csv')
    try:
        import pandas as _pd
        df = _pd.DataFrame(rows)
        if not df.empty:
            os.makedirs(out_root, exist_ok=True)
            _pd.DataFrame(rows).to_csv(csv_path, index=False,
                                       encoding='utf-8-sig')
            print(f"💾 编码消融对比 CSV: {csv_path}")
    except Exception:
        pass
    return True


def main():
    parser = argparse.ArgumentParser(description='消融试验 (复用 train)')
    parser.add_argument('--only', type=str, default=None,
                        help='只跑指定消融 (逗号分隔, 默认全部)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--use_db', action='store_true',
                        help='从 PostgreSQL 读取数据')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='数据库模式样本上限')
    parser.add_argument('--out', type=str, default='./models_ablation',
                        help='输出根目录')
    parser.add_argument('--resume', action='store_true',
                        help='跳过已完成 / 恢复未完成')
    args = parser.parse_args()

    cfg = Config()
    if args.max_samples:
        cfg.DB_MAX_SAMPLES = args.max_samples
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    configs = get_ablation_configs()
    if args.only:
        wanted = {x.strip() for x in args.only.split(',')}
        configs = [c for c in configs if c['name'] in wanted]

    os.makedirs(args.out, exist_ok=True)
    summary = []

    for ab in configs:
        name = ab['name']
        out_dir = os.path.join(args.out, name)
        print("\n" + "=" * 70)
        print(f"消融: {name} — {ab['desc']}")
        print("=" * 70)

        # 已完成跳过
        if args.resume and os.path.exists(os.path.join(out_dir, 'eval_metrics.json')):
            m = read_metrics(args.out, name)
            row = {'name': name, 'desc': ab['desc'], **_fmt_metrics(m)}
            row['enc_mode'] = ab.get('enc_mode', '')
            np_, fd, vs = read_params(args.out, name)
            row['n_params'] = np_
            row['feat_dim'] = fd
            row['vocab_size'] = vs
            print(f"  ⏭ 已完成 (MAE={m.get('mae', m.get('mae_mm', 'N/A'))}), 跳过")
            summary.append(row)
            continue

        try:
            res = run_one_ablation(cfg, device, ab, out_dir, args.resume,
                                   args.epochs, args.batch, args.use_db)
            m = read_metrics(args.out, name)
            if res is not None and m is None and isinstance(res, dict):
                m = res
            row = {'name': name, 'desc': ab['desc'], **_fmt_metrics(m)}
            row['enc_mode'] = ab.get('enc_mode', '')
            np_, fd, vs = read_params(args.out, name)
            row['n_params'] = np_
            row['feat_dim'] = fd
            row['vocab_size'] = vs
            print(f"  ✓ {name} 完成: {m if m else '已保存模型'}")
            summary.append(row)
        except Exception as e:
            import traceback
            print(f"  [X] {name} 失败: {e}")
            traceback.print_exc()
            summary.append({'name': name, 'desc': ab['desc'], 'error': str(e)})

    # ---------- 汇总 ----------
    df = pd.DataFrame(summary)
    if not df.empty:
        csv_path = os.path.join(args.out, 'ablation_summary.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        # 汇总展示: 精选关键列
        cols = [c for c in ['name', 'desc', 'n_params', 'r2', 'mae', 'rmse',
                            'peak_r2', 'peak_mae', 'peak_rel_mae_pct',
                            'rel_mae', 'weighted_mm', 'error'] if c in df.columns]
        print("\n" + "=" * 70)
        print("消融汇总表")
        print("=" * 70)
        with pd.option_context('display.max_columns', None,
                               'display.width', 200):
            print(df[cols].to_string(index=False) if cols else df.to_string(index=False))
        print(f"\n💾 汇总 CSV: {csv_path}")
        # 编码消融专门对比 (token vs direct vs cont)
        make_enc_ablation_summary(summary, out_root=args.out)
    else:
        print("  无结果")

    print("\n[OK] 消融试验完成!")


if __name__ == '__main__':
    main()
