# dataset_comparison.py
"""
样本集三情况对比试验 (复用 train, 按数据子集过滤)

设计原则:
1. 复用 train.train (通过 db_filter 控制数据子集, 无需改训练逻辑)
2. 三种情况 (同一模型/同一损失/同一训练配置, 仅数据子集不同):
   (1) all           : 所有结构 (不过滤)
   (2) rect_only     : 仅矩形结构 (plane_shape='rect', 楼层荷载不限)
   (3) rect_load15   : 仅矩形结构 且 楼层荷载每层均为 15 kPa
3. 每种情况独立输出目录 (模型 + 训练记录 + 评估图表)
4. 全部完成后输出汇总对比表 (dataset_comparison_summary.csv/json)

用法:
    python dataset_comparison.py --use_db            # 跑全部三种情况
    python dataset_comparison.py --use_db --only all,rect_only
    python dataset_comparison.py --use_db --epochs 60 --batch 64 --max_samples 20000
    python dataset_comparison.py --use_db --resume   # 跳过已完成
    python dataset_comparison.py --use_db --out ./models_dataset_cmp
    python dataset_comparison.py --use_db --epochs 80 --batch 64 --max_samples 20000
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
# 三情况定义
# ============================================================
def get_comparison_cases():
    """返回对比情况列表.

    每个 case dict:
        name        : 唯一名称 (目录名)
        desc        : 中文描述 (用于汇总表)
        db_filter   : 传给 train 的 db_filter (数据子集过滤)
                      {} = 不过滤 (全部样本)
    """
    return [
        dict(name='all',
             desc='所有结构 (全部样本)',
             db_filter={}),
        dict(name='rect_only',
             desc='仅矩形结构 (楼层荷载不限)',
             db_filter={'plane_shape': 'rect'}),
        dict(name='rect_load15',
             desc='仅矩形结构 且 楼层荷载每层均为15kPa',
             db_filter={'plane_shape': 'rect', 'floor_load_kpa': 15.0}),
    ]


# ============================================================
# 单情况运行 (包装 train)
# ============================================================
def run_one_case(cfg, device, case, out_dir, resume, epochs, batch, use_db):
    """运行单个情况, 返回评估指标 dict (同 train 的返回/写入)."""
    # 应用 batch 覆盖
    if batch:
        cfg.BATCH_SIZE = batch

    # 重置特征维度 (防止上次运行污染, 与 ablation 一致)
    _cfg_ffd_default = getattr(Config, 'FRAME_FEATURE_DIM', 44)
    cfg.FRAME_FEATURE_DIM = _cfg_ffd_default

    # token 模式: 需构建/加载微元词表, 并把实际 vocab_size 注入 model_kwargs
    # (与 ablation.run_one_ablation 一致; train 本身不解析 'auto')
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
        db_filter=case.get('db_filter') or None,
    )
    return res


# ============================================================
# 汇总 / 结果读取
# ============================================================
def read_metrics(out_dir, name):
    """读取单情况的评估指标 (从 eval_metrics.json)"""
    p = os.path.join(out_dir, name, 'eval_metrics.json')
    if os.path.exists(p):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _fmt_metrics(m):
    """把指标 dict 规整成汇总行"""
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
        description='样本集三情况对比 (所有 / 仅矩形 / 矩形+荷载15kPa)')
    parser.add_argument('--only', type=str, default=None,
                        help='只跑指定情况 (逗号分隔: all,rect_only,rect_load15; '
                             '默认全部)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--use_db', action='store_true',
                        help='从 PostgreSQL 读取数据 (必须)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='数据库模式样本上限')
    parser.add_argument('--out', type=str, default='./models_dataset_cmp',
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

    cases = get_comparison_cases()
    if args.only:
        wanted = {x.strip() for x in args.only.split(',')}
        cases = [c for c in cases if c['name'] in wanted]

    os.makedirs(args.out, exist_ok=True)
    summary = []

    for case in cases:
        name = case['name']
        out_dir = os.path.join(args.out, name)
        print("\n" + "=" * 70)
        print(f"情况: {name} — {case['desc']}  (db_filter={case['db_filter']})")
        print("=" * 70)

        # 已完成跳过
        if args.resume and os.path.exists(os.path.join(out_dir, 'eval_metrics.json')):
            m = read_metrics(args.out, name)
            row = {'name': name, 'desc': case['desc'],
                   'db_filter': json.dumps(case['db_filter'], ensure_ascii=False),
                   **_fmt_metrics(m)}
            print(f"  ⏭ 已完成 (MAE={m.get('mae', 'N/A')}), 跳过")
            summary.append(row)
            continue

        try:
            res = run_one_case(cfg, device, case, out_dir, args.resume,
                               args.epochs, args.batch, args.use_db)
            m = read_metrics(args.out, name)
            if res is not None and m is None and isinstance(res, dict):
                m = res
            row = {'name': name, 'desc': case['desc'],
                   'db_filter': json.dumps(case['db_filter'], ensure_ascii=False),
                   **_fmt_metrics(m)}
            print(f"  ✓ {name} 完成: {m if m else '已保存模型'}")
            summary.append(row)
        except Exception as e:
            import traceback
            print(f"  [X] {name} 失败: {e}")
            traceback.print_exc()
            summary.append({'name': name, 'desc': case['desc'], 'error': str(e)})

    # ---------- 汇总 ----------
    df = pd.DataFrame(summary)
    if not df.empty:
        csv_path = os.path.join(args.out, 'dataset_comparison_summary.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        # 汇总展示: 精选关键列
        cols = [c for c in ['name', 'desc', 'n_train', 'n_val', 'r2', 'mae',
                            'rmse', 'peak_r2', 'peak_mae', 'rel_mae',
                            'peak_rel_mae_pct', 'weighted_mm', 'error']
                if c in df.columns]
        print("\n" + "=" * 70)
        print("样本集三情况对比汇总表")
        print("=" * 70)
        with pd.option_context('display.max_columns', None,
                               'display.width', 250):
            print(df[cols].to_string(index=False) if cols else df.to_string(index=False))
        print(f"\n💾 汇总 CSV: {csv_path}")
    else:
        print("  无结果")

    print("\n[OK] 样本集对比试验完成!")


if __name__ == '__main__':
    main()
