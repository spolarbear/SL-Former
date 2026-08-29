# -*- coding: utf-8 -*-
"""并行扫描数据库全部结构, 统计微元计数, 构建全量词表.
用法: python scan_vocab_full.py [--workers N] [--min_freq 1]
"""
import sys, os, time, pickle, argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _worker(chunk_ids):
    """子进程: 扫描一段 struct_id, 返回 (微元计数, 结构数)."""
    import numpy as np
    from db_manager import SLFDatabase
    from frame_model import build_frame_model
    from frame_grid_encoder import (encode_frame_grid, decode_cell,
                                    VoxelVocab)
    db = SLFDatabase()
    counts = Counter()
    n_ok = 0
    for sid in chunk_ids:
        struct = db.get_structure(sid)
        if struct is None:
            continue
        try:
            model = build_frame_model(struct=struct)
            codes, _ = encode_frame_grid(model)
        except Exception:
            continue
        n_ok += 1
        nz = codes != 0
        # 非空格逐个解码为微元键
        for ix, iy, k in zip(*np.argwhere(nz).T):
            d = decode_cell(codes[ix, iy, k])
            key = VoxelVocab._micro_key_from_cell(d)
            counts[key] += 1
    db.close()
    return counts, n_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--min_freq', type=int, default=1)
    ap.add_argument('--out', default='cache/voxel_counts_full.pkl')
    args = ap.parse_args()

    import numpy as np
    from frame_grid_encoder import VoxelVocab

    ids_file = 'cache/_all_struct_ids.pkl'
    with open(ids_file, 'rb') as f:
        all_ids = pickle.load(f)
    print(f'扫描结构数: {len(all_ids)}')

    # 分块
    chunks = [all_ids[i::args.workers] for i in range(args.workers)]
    chunks = [c for c in chunks if c]

    t0 = time.time()
    if len(chunks) == 1:
        results = [_worker(chunks[0])]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_worker, chunks))

    total_counts = Counter()
    n_ok_total = 0
    for counts, n_ok in results:
        total_counts.update(counts)
        n_ok_total += n_ok
    print(f'扫描完成: {n_ok_total} 结构, {time.time()-t0:.0f}s')

    # 构建词表
    vocab = VoxelVocab()
    vocab.counts = dict(total_counts)
    n_tok = vocab.build(min_freq=args.min_freq)
    print(f'全量词表 token 数 (min_freq={args.min_freq}): {n_tok}')
    print(f'出现过的微元种类 (含 freq>=1): {len(total_counts)}')

    # 分布
    from collections import Counter as C
    combos = C(k[0] for k in total_counts.keys())
    print('combo 分布:', dict(sorted(combos.items())))
    # 频率分布
    freqs = sorted(total_counts.values())
    print(f'频率: min={freqs[0]}, median={freqs[len(freqs)//2]}, max={freqs[-1]}')
    print(f'频率==1 的微元数: {sum(1 for v in total_counts.values() if v==1)}')
    print(f'频率<3 的微元数: {sum(1 for v in total_counts.values() if v<3)}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump({'id2micro': vocab.id2micro, 'counts': dict(total_counts)}, f)
    print(f'已保存: {args.out}')


if __name__ == '__main__':
    main()
