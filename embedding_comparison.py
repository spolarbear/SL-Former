# embedding_comparison.py
"""
体素 token embedding 初始化方式对比 (三种策略消融)

对比三种 embedding 初始化方式 (同一数据/同一模型/同一损失, 仅初始化不同):
  1. random : 纯随机初始化 nn.Embedding (无物理先验)
  2. rich8  : rich 8 维物理向量初始化 (类型/柱EI/梁EI/面积/偏位/填充)
  3. hexa9  : 六面体刚度 9 维初始化 (3 对对面 剪切GA + 抗弯EI + 类型/填充/偏位)
  (可选 basic5: 精简 5 维物理向量初始化)

原理: 物理向量初始化使"刚度/截面相似"的微元 token 在 embedding 空间初始即邻近,
训练中继续微调。本脚本对比不同初始化方式对最终预测精度的影响。

用法:
    python embedding_comparison.py --use_db                    # 跑全部三种
    python embedding_comparison.py --use_db --only random,hexa9
    python embedding_comparison.py --use_db --epochs 60 --batch 64 --max_samples 20000
    python embedding_comparison.py --use_db --resume           # 跳过已完成
    python embedding_comparison.py --use_db --out ./models_embed_cmp
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from train import train


# ============================================================
# 三种 embedding 方式定义
# ============================================================
def get_embedding_modes():
    """返回对比方式列表.

    每个 mode dict:
        name    : 唯一名称 (目录名)
        desc    : 中文描述
        phy     : 传给 cfg.VOXEL_TOKEN_INIT_PHYSICS 的值
    """
    return [
        dict(name='random', desc='随机初始化 (无物理先验)',
             phy='random'),
        dict(name='rich8', desc='rich 8 维物理向量初始化 (类型/柱EI/梁EI/面积/偏位/填充)',
             phy='rich8'),
        dict(name='hexa9', desc='六面体刚度 9 维初始化 (3对对面 剪切GA+抗弯EI)',
             phy='hexa9'),
    ]


# ============================================================
# 单方式运行 (包装 train, 复用 token vocab 解析)
# ============================================================
def run_one_mode(cfg, device, mode, out_dir, resume, epochs, batch, use_db):
    """运行单个 embedding 方式, 返回评估指标 dict."""
    # 应用 batch 覆盖
    if batch:
        cfg.BATCH_SIZE = batch
    # 重置特征维度 (防止上次运行污染)
    _cfg_ffd_default = getattr(Config, 'FRAME_FEATURE_DIM', 44)
    cfg.FRAME_FEATURE_DIM = _cfg_ffd_default
    # 关键: 切换 embedding 初始化方式
    cfg.VOXEL_TOKEN_INIT_PHYSICS = mode['phy']

    # token 模式: 构建/加载微元词表, 注入实际 vocab_size
    model_kwargs = {'use_voxel_token': True, 'vocab_size': 'auto'}
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

    res = train(
        cfg, device, out_dir, resume=resume, epochs=epochs,
        use_bypass=True, use_v2=True, use_db=use_db,
        use_voxel_feature=True, voxel_enc_mode='token',
        model_kwargs=model_kwargs,
        loss_kwargs=None,
        db_filter=None,   # 同一数据: 全部样本
    )
    return res


# ============================================================
# 汇总 / 结果读取
# ============================================================
def read_metrics(out_dir, name):
    p = os.path.join(out_dir, name, 'eval_metrics.json')
    if os.path.exists(p):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _fmt_metrics(m):
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


def main():
    parser = argparse.ArgumentParser(
        description='体素 token embedding 初始化方式对比 (random / rich8 / hexa9)')
    parser.add_argument('--only', type=str, default=None,
                        help='只跑指定方式 (逗号分隔: random,rich8,hexa9,basic5; '
                             '默认全部)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--use_db', action='store_true',
                        help='从 PostgreSQL 读取数据 (必须)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='数据库模式样本上限')
    parser.add_argument('--out', type=str, default='./models_embed_cmp',
                        help='输出根目录')
    parser.add_argument('--resume', action='store_true',
                        help='跳过已完成 / 恢复未完成')
    args = parser.parse_args()

    if not args.use_db:
        print("[X] 此对比基于数据库样本集, 请加 --use_db")
        return

    import torch
    cfg = Config()
    if args.max_samples:
        cfg.DB_MAX_SAMPLES = args.max_samples
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    modes = get_embedding_modes()
    if args.only:
        wanted = {x.strip() for x in args.only.split(',')}
        modes = [m for m in modes if m['name'] in wanted]

    os.makedirs(args.out, exist_ok=True)
    summary = []

    for mode in modes:
        name = mode['name']
        out_dir = os.path.join(args.out, name)
        print("\n" + "=" * 70)
        print(f"方式: {name} — {mode['desc']}  (VOXEL_TOKEN_INIT_PHYSICS={mode['phy']})")
        print("=" * 70)

        if args.resume and os.path.exists(os.path.join(out_dir, 'eval_metrics.json')):
            m = read_metrics(args.out, name)
            row = {'name': name, 'desc': mode['desc'],
                   'phy': mode['phy'], **_fmt_metrics(m)}
            print(f"  ⏭ 已完成 (MAE={m.get('mae', 'N/A')}), 跳过")
            summary.append(row)
            continue

        try:
            res = run_one_mode(cfg, device, mode, out_dir, args.resume,
                               args.epochs, args.batch, args.use_db)
            m = read_metrics(args.out, name)
            if res is not None and m is None and isinstance(res, dict):
                m = res
            row = {'name': name, 'desc': mode['desc'],
                   'phy': mode['phy'], **_fmt_metrics(m)}
            print(f"  ✓ {name} 完成: {m if m else '已保存模型'}")
            summary.append(row)
        except Exception as e:
            import traceback
            print(f"  [X] {name} 失败: {e}")
            traceback.print_exc()
            summary.append({'name': name, 'desc': mode['desc'],
                            'phy': mode['phy'], 'error': str(e)})

    # ---------- 汇总 ----------
    df = pd.DataFrame(summary)
    if not df.empty:
        csv_path = os.path.join(args.out, 'embedding_comparison_summary.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        cols = [c for c in ['name', 'desc', 'n_train', 'n_val', 'r2', 'mae',
                            'rmse', 'peak_r2', 'peak_mae', 'rel_mae',
                            'peak_rel_mae_pct', 'weighted_mm', 'error']
                if c in df.columns]
        print("\n" + "=" * 70)
        print("Embedding 初始化方式对比汇总表")
        print("=" * 70)
        with pd.option_context('display.max_columns', None,
                               'display.width', 250):
            print(df[cols].to_string(index=False) if cols else df.to_string(index=False))
        print(f"\n💾 汇总 CSV: {csv_path}")
    else:
        print("  无结果")

    print("\n[OK] Embedding 对比试验完成!")


if __name__ == '__main__':
    main()
