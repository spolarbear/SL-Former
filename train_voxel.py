# train_voxel.py
"""
切杆系连续特征训练脚本 (研究用, 对比杆系编码)

与 train.py 的训练逻辑完全一致, 唯一区别:
- 结构编码: 直接切杆系模型 (frame_model) -> 固定 64×64×64 网格, 每格真实 1m
  体素不缩放 (64m 空间, 原点对齐, 结构占一部分格); 每格输出 6 维连续物理量
  (类型梯度/柱EI/梁EI/密度/节点偏位), 相似格子空间距离近 (LLM embedding 启发);
  不再体素化, 楼板忽略; depth 仅兼容参数 (网格固定 64³) -> 262144维 (64³×6)
- 杆系编码 (frame_feature_encoder 44维) 保留在 train.py 原样不动

用法:
    python train_voxel.py --use_db --epochs 200
    python train_voxel.py --use_db --epochs 200 --max_samples 20000
    python train_voxel.py --use_db --max_samples 10000 --epochs 50
    python train_voxel.py --use_db --max_samples 80000 --epochs 300 --out ./models_voxel_token
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from train import train


def main():
    parser = argparse.ArgumentParser(description='切杆系编码训练 (研究用, 对比杆系)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--depth', type=int, default=5, choices=[4, 5, 6, 7],
                        help='切格深度 (兼容参数, 网格固定 64³ = 262144维)')
    parser.add_argument('--enc', type=str, default=None, choices=['token', 'direct', 'cont'],
                        help='切杆系编码模式: token=微元词表+Embedding (默认), '
                             'direct=直接128位整数编码归一化 (上一版本), '
                             'cont=6通道连续物理量特征 (LLM embedding 启发)')
    parser.add_argument('--out', type=str, default='./models_voxel')
    parser.add_argument('--use_db', action='store_true',
                        help='从 PostgreSQL 读取数据 (与 train.py 一致)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='数据库模式样本上限 (默认 config.DB_MAX_SAMPLES)')
    parser.add_argument('--params', type=str, default=None,
                        help='Optuna 最优超参 JSON (可选, 复用寻优结果)')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复')
    args = parser.parse_args()

    cfg = Config()
    if args.batch:
        cfg.BATCH_SIZE = args.batch
    if args.max_samples:
        cfg.DB_MAX_SAMPLES = args.max_samples
    if args.params and os.path.exists(args.params):
        import json
        with open(args.params, 'r', encoding='utf-8') as f:
            hp = json.load(f)
        best = hp.get('best_params', hp)
        _map = {'lr': 'LEARNING_RATE', 'weight_decay': 'WEIGHT_DECAY',
                'batch_size': 'BATCH_SIZE', 'd_model': 'D_MODEL',
                'n_layer': 'N_LAYER', 'n_head': 'N_HEAD', 'd_ff': 'D_FF',
                'dropout': 'DROPOUT', 'v2_drop_path': 'V2_DROP_PATH',
                'v2_conv_kernel': 'V2_CONV_KERNEL', 'warmup_epochs': 'LR_WARMUP_EPOCHS',
                'loss_peak_w': 'LOSS_PEAK_W', 'loss_high_w': 'LOSS_HIGH_W',
                'loss_high_thresh': 'LOSS_HIGH_THRESH_MM'}
        for k, attr in _map.items():
            if k in best:
                setattr(cfg, attr, best[k])
        print(f"🎯 已加载 Optuna 最优超参: {args.params}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # ---------- 切杆系编码模式 ----------
    # token: 微元词表 + nn.Embedding (默认)
    # direct: 直接 32 位整数编码归一化 (上一版本简单直接编码, MLP)
    # cont: 6 通道连续物理量特征 (LLM embedding 启发, MLP)
    enc_mode = args.enc or ('token' if getattr(cfg, 'USE_VOXEL_TOKEN', True) else 'cont')

    # ---------- 构建/加载微元词表 (LLM tokenizer, 仅 token 模式需要) ----------
    if enc_mode == 'token':
        from frame_grid_encoder import VoxelVocab, build_voxel_vocab_from_db
        vf = getattr(cfg, 'VOXEL_VOCAB_FILE', None)
        vocab = VoxelVocab()
        if vf and os.path.exists(vf):
            vocab.load(vf)
            n_tok = len(vocab.id2micro)
            print(f"📖 加载微元词表: {vf} ({n_tok} token)")
        else:
            print("🔍 未找到词表, 从数据库构建 (可稍后 --save-vocab 保存)...")
            from db_manager import SLFDatabase
            db = SLFDatabase()
            n_scan = int(getattr(cfg, 'VOXEL_VOCAB_SCAN_STRUCTS', 2000))
            vocab, n_tok = build_voxel_vocab_from_db(db, n_structs=n_scan)
            db.close()
            print(f"📖 从 {n_scan} 结构构建词表: {n_tok} token")
            if vf:
                try:
                    vocab.save(vf)
                    print(f"💾 词表已保存: {vf}")
                except Exception as e:
                    print(f"  ⚠️ 词表保存失败: {e}")
        cfg.VOXEL_VOCAB_SIZE = n_tok
        model_kwargs = {'use_voxel_token': True, 'vocab_size': n_tok}
    else:
        # direct / cont: 用 PrecomputedOctreeEncoder (连续 MLP) 编码
        model_kwargs = None
        print(f"🧱 {enc_mode} 编码模式: 用 MLP 连续编码器 (非 token)")

    print(f"🧱 体素训练 ({enc_mode}, "
          f"32³格, 每格2m, 词表{getattr(cfg, 'VOXEL_VOCAB_SIZE', 0) if enc_mode == 'token' else '-'})")

    train(cfg, device, args.out, resume=args.resume,
          epochs=args.epochs, lr=args.lr,
          use_bypass=True, use_v2=True, use_db=args.use_db,
          use_voxel_feature=True, voxel_depth=args.depth,
          voxel_enc_mode=enc_mode, model_kwargs=model_kwargs)


if __name__ == '__main__':
    main()
