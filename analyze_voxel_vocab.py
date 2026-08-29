# -*- coding: utf-8 -*-
"""
微元词表临近性分析 (增强版, 全库数据, SCI 配图)

基于微元 Token 词表 (VoxelVocab) 做"语义临近性"多维分析。
与旧版只画一张粗糙 PCA 散点不同, 本脚本:

  - 数据源: 优先用全库扫描缓存 cache/voxel_counts_full.pkl (39365 结构,
    284 微元), 否则现场扫描 --n 个结构。
  - 物理向量升级: 8 维 (类型梯度 / 柱EI / 梁EI / 柱截面面积 / 梁截面面积 /
    偏位X / 偏位Y / 填充强度), 比旧版 5 维更完整地表达"微元语义"。
  - 输出 4 张 SCI 图 (PDF+PNG 300dpi):
    1) voxel_vocab_pca_2d   : PCA 2D 散点, 点大小=微元频率, 颜色=构件类型
    2) voxel_vocab_stiffness: 同一投影按柱/梁截面连续着色, 验证"刚度谱单调"
    3) voxel_vocab_combo_map: 类型×截面档位热力图 + 微元频率长尾分布
    4) voxel_vocab_physics_modes_compare: 三种物理向量 (rich8/hexa9/basic5)
       并排 PCA 对比 (--modes 开启), 量化类间分离度/同类聚集度

用法:
    python analyze_voxel_vocab.py                 # 全库缓存 (最快)
    python analyze_voxel_vocab.py --modes         # 额外输出三物理向量 PCA 对比
    python analyze_voxel_vocab.py --modes --augment   # 临时扩充偏位变体样本 (PCA 更丰富)
    python analyze_voxel_vocab.py --n 5000        # 现场扫描 5000 结构
    python analyze_voxel_vocab.py --out ./plots/sci
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import font_manager
from collections import Counter, defaultdict
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frame_grid_encoder import (VoxelVocab, COMBO_COL, COMBO_BX, COMBO_BY,
                                TYPE_SCALAR, SECTION_MOD, _level_section,
                                E_CONCRETE, _log1p_norm,
                                micro_physics_vector_rich,
                                hexa_stiffness_vector)

# ============================================================
# SCI 样式
# ============================================================
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['legend.frameon'] = False
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['mathtext.fontset'] = 'stix'

SCI_DPI = 300
SCI_FORMATS = ['pdf', 'png']

# 类型分类与颜色
def combo_class(combo):
    """把微元 combo 归类: 空/梁/柱/柱+梁节点/交叉梁."""
    if combo == 0:
        return 'Empty'
    if combo & COMBO_COL:
        if combo & (COMBO_BX | COMBO_BY):
            return 'Column+Beam'
        return 'Column'
    if combo & COMBO_BX and combo & COMBO_BY:
        return 'X+Y Beam'
    return 'Beam'

CLASS_ORDER = ['Column', 'Beam', 'Column+Beam', 'X+Y Beam', 'Empty']
CLASS_COLORS = {'Empty': '#cccccc', 'Column': '#B85C4A', 'Beam': '#4A7DB4',
                'Column+Beam': '#8E5BB8', 'X+Y Beam': '#3DB8B0'}


def load_vocab_counts(n_structs=None, use_cache=True):
    """加载微元计数: 优先全库缓存, 否则现场扫描."""
    cache_file = './cache/voxel_counts_full.pkl'
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = __import__('pickle').load(f)
        vocab = VoxelVocab()
        vocab.id2micro = data['id2micro']
        vocab.counts = data['counts']
        vocab.micro2id = {v: k for k, v in vocab.id2micro.items()}
        vocab.built = True
        print(f"[load] use full-db cache {cache_file} "
              f"({len(vocab.id2micro)} tokens)")
        return vocab
    from db_manager import SLFDatabase
    from frame_grid_encoder import build_voxel_vocab_from_db
    db = SLFDatabase()
    vocab, n_tok = build_voxel_vocab_from_db(db, n_structs=n_structs)
    db.close()
    print(f"[scan] {n_structs} structs -> {n_tok} tokens")
    return vocab


# ============================================================
# 临时样本扩充 (仅可视化用, 不改数据库/生成器)
#   目标: 真实样本结构坐标全为整数 (span 3~8m), 1m 网格下偏位
#         只有 -4(格中心左半格) 和 -8(未设置方向) 两个取值,
#         词表仅 44 个微元, PCA 点太少。
#   方案: 对每个真实微元类型, 生成若干"偏位变体" —— 在格内
#         (off_x, off_y) 遍历更多档位, 模拟构件在格子内不同位置,
#         扩充词表让 PCA 更丰富, 展示"偏位→embedding"的影响。
# ============================================================
def build_augmented_vocab(vocab, n_off_per_key=4, seed=42, max_tokens=300):
    """基于现有词表生成偏位扩充词表 (仅内存, 不落库).

    Args:
        vocab: 现有 VoxelVocab
        n_off_per_key: 每个真实微元生成多少个偏位变体
        seed: 随机种子 (可复现)
        max_tokens: 扩充后词表上限 (防止过大)

    Returns:
        新 VoxelVocab (id2micro/counts 扩充)
    """
    from frame_grid_encoder import COMBO_COL, COMBO_BX, COMBO_BY
    rng = np.random.default_rng(seed)
    new_vocab = VoxelVocab()
    # 先复制现有 token
    new_vocab.id2micro = dict(vocab.id2micro)
    new_vocab.micro2id = dict(vocab.micro2id)
    new_vocab.counts = dict(vocab.counts)
    new_vocab.built = True

    # 偏位档位: -7..+7 (排除 0 保留多样性; 用合理偏移范围)
    off_pool = [-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7]

    # 对每个真实微元 (非空), 生成偏位变体
    base_keys = [k for k in new_vocab.id2micro.values() if k[0] != 0]
    for key in base_keys:
        combo, cl, bw, bh, ox, oy = key
        # 已有变体数
        existing = set()
        for kk in new_vocab.id2micro.values():
            if kk[:4] == (combo, cl, bw, bh):
                existing.add((kk[4], kk[5]))
        made = 0
        tries = 0
        while made < n_off_per_key and tries < 200:
            tries += 1
            nox = int(rng.choice(off_pool))
            noy = int(rng.choice(off_pool))
            # 有柱的微元偏位通常对称, 保持 x/y 同号更接近物理 (可选, 不强求)
            if (combo & COMBO_COL) and rng.random() < 0.5:
                noy = nox
            if (nox, noy) in existing:
                continue
            # 新 key: 保持组合/截面, 换偏位
            nk = (combo, cl, bw, bh, nox, noy)
            if nk in new_vocab.micro2id:
                continue
            tid = len(new_vocab.id2micro)
            new_vocab.id2micro[tid] = nk
            new_vocab.micro2id[nk] = tid
            new_vocab.counts[nk] = 1   # 合成样本计数=1
            existing.add((nox, noy))
            made += 1
            if len(new_vocab.id2micro) >= max_tokens:
                break
        if len(new_vocab.id2micro) >= max_tokens:
            break
    return new_vocab


def _micro_physics_vector_basic(key):
    """精简 5 维物理向量 (类型/柱EI/梁EI/偏位X/偏位Y)."""
    combo, col_lvl, bw_lvl, bh_lvl, off_x, off_y = key
    col = _level_section(col_lvl) if (combo & COMBO_COL) else 0.0
    bw = _level_section(bw_lvl) if (combo & (COMBO_BX | COMBO_BY)) else 0.0
    bh = _level_section(bh_lvl) if (combo & (COMBO_BX | COMBO_BY)) else 0.0
    ei_col = E_CONCRETE * col**4 / 12.0 if col > 0 else 0.0
    ei_beam = E_CONCRETE * bw * bh**3 / 12.0 if bw > 0 and bh > 0 else 0.0
    lo_ei = np.log1p(1e5); hi_ei = np.log1p(1e10)
    return np.array([
        TYPE_SCALAR.get(combo, 0.0),
        _log1p_norm(ei_col, lo_ei, hi_ei),
        _log1p_norm(ei_beam, lo_ei, hi_ei),
        float(np.clip(off_x / 8.0, -1, 1)),
        float(np.clip(off_y / 8.0, -1, 1)),
    ], dtype=np.float32)


def _micro_physics_vector_rich_offw(key, off_w=0.3):
    """rich8 的偏位降权变体 (仅用于可视化 PCA 分析).

    rich8 原始 8 维里 off_x/off_y 幅度为 1, 在偏位扩充样本下方差过大,
    主导 PC1 使同类型点被偏位拉散 (图显乱)。本变体把偏位维度乘以
    系数 off_w (默认 0.3), 让 PCA 主要由刚度/截面/类型主导,
    物理分类重新显现。不影响训练侧 embedding (frame_grid_encoder 未改动).
    """
    combo, col_lvl, bw_lvl, bh_lvl, off_x, off_y = key
    col = _level_section(col_lvl) if (combo & COMBO_COL) else 0.0
    bw = _level_section(bw_lvl) if (combo & (COMBO_BX | COMBO_BY)) else 0.0
    bh = _level_section(bh_lvl) if (combo & (COMBO_BX | COMBO_BY)) else 0.0
    ei_col = E_CONCRETE * col**4 / 12.0 if col > 0 else 0.0
    ei_beam = E_CONCRETE * bw * bh**3 / 12.0 if bw > 0 and bh > 0 else 0.0
    lo_ei = np.log1p(1e5); hi_ei = np.log1p(1e10)
    n_comp = (1 if combo & COMBO_COL else 0) + \
             (1 if combo & COMBO_BX else 0) + (1 if combo & COMBO_BY else 0)
    return np.array([
        TYPE_SCALAR.get(combo, 0.0) / 3.0,             # 类型梯度
        _log1p_norm(ei_col, lo_ei, hi_ei),             # 柱刚度
        _log1p_norm(ei_beam, lo_ei, hi_ei),            # 梁刚度
        (col / 1.4)**2,                                # 柱截面面积
        (bw * bh) / (0.5 * 0.8),                       # 梁截面面积
        float(np.clip(off_x / 8.0, -1, 1)) * off_w,   # 偏位X (降权)
        float(np.clip(off_y / 8.0, -1, 1)) * off_w,   # 偏位Y (降权)
        n_comp / 3.0,                                  # 填充强度
    ], dtype=np.float32)


def build_feature_table(vocab, mode='rich8', rich8_offw=None):
    """构建微元特征表: keys / 物理向量 / 类型 / 截面 / 频率.

    Args:
        mode: 物理向量模式 'rich8' / 'hexa9' / 'basic5'
            rich8 : 8 维 (类型/柱EI/梁EI/柱面积/梁面积/偏位X/偏位Y/填充强度)
            hexa9 : 9 维 (3对对面 剪切GA+抗弯EI + 类型/填充/偏位)
            basic5: 5 维 (类型/柱EI/梁EI/偏位X/偏位Y)
        rich8_offw: rich8 偏位降权系数 (None=不降权用原始 rich8;
                    0.0~1.0 之间用降权变体, 偏位×系数, 避免偏位主导 PCA)
    """
    if mode == 'rich8' and rich8_offw is not None:
        vec_fn = lambda k: _micro_physics_vector_rich_offw(k, off_w=rich8_offw)
    else:
        vec_fn = {
            'rich8': micro_physics_vector_rich,
            'hexa9': hexa_stiffness_vector,
            'basic5': _micro_physics_vector_basic,
        }.get(mode)
    if vec_fn is None:
        raise ValueError(f"未知物理向量模式: {mode} (可选 rich8/hexa9/basic5)")
    keys = list(vocab.id2micro.values())
    ids = list(vocab.id2micro.keys())
    X = np.stack([vec_fn(k) for k in keys])
    labels = np.array([combo_class(k[0]) for k in keys])
    # 截面信息 (供连续着色)
    col_sizes = np.array([_level_section(k[1]) if (k[0] & COMBO_COL) else 0.0
                          for k in keys])
    beam_sizes = np.array([
        _level_section(k[2]) * _level_section(k[3])
        if (k[0] & (COMBO_BX | COMBO_BY)) else 0.0 for k in keys])
    # 频率 (词频)
    freqs = np.array([vocab.counts.get(k, 0) for k in keys], dtype=np.float64)
    # 仅非空微元 (token 0 = 空不参与临近性主分析)
    nz = np.array([k[0] != 0 for k in keys])
    return {
        'keys': keys, 'ids': np.array(ids), 'X': X,
        'labels': labels, 'col_sizes': col_sizes,
        'beam_sizes': beam_sizes, 'freqs': freqs, 'nz': nz,
    }


def pca_project(X, n_comp=2):
    """PCA 投影 (去零方差列 + 中心化 + SVD)."""
    mask = X.var(axis=0) > 1e-12
    Xv = X[:, mask]
    Xc = Xv - Xv.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:n_comp].T
    evr = S[:n_comp]**2 / (S**2).sum()
    return Z, evr


def save_fig(fig, path):
    for fmt in SCI_FORMATS:
        fig.savefig(f"{path}.{fmt}", dpi=SCI_DPI, bbox_inches='tight',
                    facecolor='white', format=fmt)
    plt.close(fig)
    print(f"  [saved] {path}.pdf/.png")


# ============================================================
# 图 1: PCA 2D 散点 (点大小=频率, 颜色=类型)
# ============================================================
def plot_pca_2d(feats, Z, evr, out_dir, title_suffix=''):
    labels = np.array(feats['labels'])
    freqs = feats['freqs']
    nz = feats['nz']
    # 点大小 = 频率 (log 缩放, 便于显示长尾)
    s = 30 + 120 * (np.log1p(freqs[nz]) / np.log1p(freqs[nz].max()))
    # 归一化坐标 (便于展示)
    z = Z[nz]

    fig, ax = plt.subplots(figsize=(9.5, 8))
    for cls in CLASS_ORDER:
        m = (labels[nz] == cls)
        if not m.any():
            continue
        ax.scatter(z[m, 0], z[m, 1], c=CLASS_COLORS[cls], s=s[m],
                   alpha=0.75, edgecolors='white', linewidth=0.5,
                   label=f'{cls} ({m.sum()})')
    ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}% var)')
    ax.set_title(f'Micro-element token proximity (PCA 2D){title_suffix}\n'
                 f'{nz.sum()} non-empty tokens, dot size = frequency',
                 fontsize=11)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'voxel_vocab_pca_2d'))

    # 打印类间质心距离
    by = defaultdict(list)
    for zz, l in zip(z, labels[nz]):
        by[l].append(zz)
    print("\n  类间质心距离 (PC1,PC2):")
    cls_list = [c for c in CLASS_ORDER if c in by and c != 'Empty']
    header = ' ' * 14 + ''.join(f'{c:>12}' for c in cls_list)
    print('  ' + header)
    for l1 in cls_list:
        row = f'  {l1:12s}'
        c1 = np.array(by[l1]).mean(axis=0)
        for l2 in cls_list:
            c2 = np.array(by[l2]).mean(axis=0)
            row += f'{np.linalg.norm(c1-c2):>12.3f}'
        print(row)


# ============================================================
# 图 2: 同一投影, 按柱/梁截面连续着色 (刚度谱)
# ============================================================
def plot_stiffness_spectrum(feats, Z, evr, out_dir, title_suffix=''):
    """按柱/梁截面尺寸连续着色, 验证截面越大距离越远 (刚度谱单调)."""
    labels = np.array(feats['labels'])
    col_sizes = feats['col_sizes']
    beam_sizes = feats['beam_sizes']
    nz = feats['nz']
    z = Z[nz]
    col_sizes_nz = col_sizes[nz]
    beam_sizes_nz = beam_sizes[nz]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # 左: 柱型微元按柱截面着色
    ax = axes[0]
    m = col_sizes_nz > 0
    sc = ax.scatter(z[m, 0], z[m, 1], c=col_sizes_nz[m], cmap='viridis',
                    s=40, alpha=0.8, edgecolors='white', linewidth=0.4)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title(f'Column-bearing tokens, color = column size (m){title_suffix}',
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Column section (m)')

    # 右: 梁型微元按梁截面面积着色
    ax = axes[1]
    m = beam_sizes_nz > 0
    sc = ax.scatter(z[m, 0], z[m, 1], c=beam_sizes_nz[m], cmap='plasma',
                    s=40, alpha=0.8, edgecolors='white', linewidth=0.4)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('Beam-bearing tokens, color = beam section area (m²)',
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Beam section area (m²)')

    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'voxel_vocab_stiffness'))

    # 打印截面档位距离单调性验证 (柱: 相邻档距离)
    print("\n  柱截面档位 → 物理向量距离 (验证单调):")
    col_vals = sorted({round(c, 2) for c in col_sizes_nz if c > 0})
    # 用物理向量距离 (全 8 维)
    Xnz = feats['X'][nz]
    for ci, cv in enumerate(col_vals):
        m = (col_sizes_nz == cv) & (labels[nz] == 'Column')
        if not m.any():
            continue
        # 该截面档的代表向量
        rep = Xnz[m].mean(axis=0)
        if ci > 0:
            # 与上一档距离
            m0 = (col_sizes_nz == col_vals[ci-1]) & (labels[nz] == 'Column')
            if m0.any():
                rep0 = Xnz[m0].mean(axis=0)
                d = np.linalg.norm(rep - rep0)
                print(f"    {col_vals[ci-1]:.1f}->{cv:.1f}m: vec dist {d:.3f}")


# ============================================================
# 图 3: 类型×截面档位热力图 + 频率长尾
# ============================================================
def plot_combo_map(feats, out_dir, title_suffix=''):
    """左: 类型×柱截面档位微元数热力图; 右: 微元频率长尾分布."""
    keys = np.array(feats['keys'], dtype=object)
    labels = np.array(feats['labels'])
    col_sizes = feats['col_sizes']
    freqs = feats['freqs']
    nz = feats['nz']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # ---- 左: 类型 × 柱截面档 热力图 (非空微元) ----
    ax = axes[0]
    col_vals = sorted({round(c, 2) for c in col_sizes if c > 0})
    cls_present = [c for c in CLASS_ORDER if c != 'Empty'
                   and (labels[nz] == c).any()]
    mat = np.zeros((len(cls_present), len(col_vals)), dtype=int)
    for i, c in enumerate(cls_present):
        for j, cv in enumerate(col_vals):
            mat[i, j] = int(((labels == c) & (np.abs(col_sizes - cv) < 1e-9)
                             & nz).sum())
    im = ax.imshow(mat, aspect='auto', cmap='YlGnBu')
    ax.set_xticks(range(len(col_vals)))
    ax.set_xticklabels([f'{v:.1f}' for v in col_vals], fontsize=8)
    ax.set_yticks(range(len(cls_present)))
    ax.set_yticklabels(cls_present, fontsize=9)
    ax.set_xlabel('Column section size (m)')
    ax.set_ylabel('Micro-element type')
    ax.set_title(f'Micro-element count by type × column section{title_suffix}',
                 fontsize=11)
    for i in range(len(cls_present)):
        for j in range(len(col_vals)):
            if mat[i, j] > 0:
                ax.text(j, i, str(mat[i, j]), ha='center', va='center',
                        fontsize=8, color='black')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # ---- 右: 微元频率长尾分布 (log-log) ----
    ax = axes[1]
    fr = np.sort(freqs[nz])[::-1]
    rank = np.arange(1, len(fr) + 1)
    ax.loglog(rank, fr, 'o-', color='#4A7DB4', ms=5, alpha=0.8)
    ax.set_xlabel('Micro-element rank (by frequency)')
    ax.set_ylabel('Frequency (occurrences)')
    ax.set_title(f'Micro-element frequency Zipf curve ({len(fr)} tokens)',
                 fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    # 注释 top5
    top5_idx = np.argsort(freqs)[::-1][:5]
    for t in top5_idx:
        if not nz[t]:
            continue
        ax.annotate(f'#{feats["ids"][t]} ({labels[t]})',
                    (rank[np.where(fr == freqs[t])[0][0]], freqs[t]),
                    textcoords='offset points', xytext=(4, 4), fontsize=7)

    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'voxel_vocab_combo_map'))


# ============================================================
# 图 4: 三种物理向量 PCA 对比 (rich8 / hexa9 / basic5 并排)
# ============================================================
PHYSICS_MODES = {
    'rich8':  'rich 8D (type/colEI/beamEI/area/offset/fill)',
    'hexa9':  'hexa 9D (3-pair shear GA + bending EI)',
    'basic5': 'basic 5D (type/colEI/beamEI/offset)',
}

def plot_physics_modes_compare(vocab, out_dir, title_suffix='',
                               rich8_offw=0.3):
    """三物理向量并排 PCA 2D 散点 (颜色=构件类型, 点大小=频率).

    对比 rich8 / hexa9 / basic5 三种物理向量: 哪种更能把"刚度/截面相似"
    的微元聚拢 (同类型成簇), 不同类型 (柱/梁/节点) 分开.

    Args:
        rich8_offw: rich8 偏位降权系数 (默认 0.3, 避免扩充样本的偏位
                    主导 PCA 把同类型点拉散; None=不降权用原始 rich8)
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax, (mode, label) in zip(axes, PHYSICS_MODES.items()):
        feats = build_feature_table(
            vocab, mode=mode,
            rich8_offw=(rich8_offw if mode == 'rich8' else None))
        Z, evr = pca_project(feats['X'])
        nz = feats['nz']
        labels = np.array(feats['labels'])
        freqs = feats['freqs']
        z = Z[nz]
        s = 30 + 120 * (np.log1p(freqs[nz]) / np.log1p(freqs[nz].max()))

        # ---- 类型散点 ----
        for cls in CLASS_ORDER:
            m = (labels[nz] == cls)
            if not m.any():
                continue
            ax.scatter(z[m, 0], z[m, 1], c=CLASS_COLORS[cls], s=s[m],
                       alpha=0.85, edgecolors='white', linewidth=0.5,
                       label=f'{cls} ({m.sum()})')
        ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}% var)')
        ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}% var)')
        ax.set_title(f'{mode} — {label}\n'
                     f'cum var {evr.sum()*100:.1f}%{title_suffix}',
                     fontsize=10)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.axvline(0, color='gray', lw=0.5, ls='--')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        # 量化: 类间分离度 (柱 vs 梁 质心距离) + 同类聚集度
        by = defaultdict(list)
        for zz, l in zip(z, labels[nz]):
            by[l].append(zz)
        cls_list = [c for c in CLASS_ORDER if c in by and c != 'Empty']
        sep = 0.0
        cnt_sep = 0
        for i in range(len(cls_list)):
            c1 = np.array(by[cls_list[i]]).mean(axis=0)
            for j in range(i+1, len(cls_list)):
                c2 = np.array(by[cls_list[j]]).mean(axis=0)
                sep += np.linalg.norm(c1-c2)
                cnt_sep += 1
        # 同类聚集度: 类内平均半径 (越小越聚拢)
        agg = 0.0
        cnt_agg = 0
        for c in cls_list:
            pts = np.array(by[c])
            cen = pts.mean(axis=0)
            agg += np.linalg.norm(pts - cen, axis=1).mean()
            cnt_agg += 1
        sep_m = sep / max(1, cnt_sep)
        agg_m = agg / max(1, cnt_agg)
        ax.text(0.02, 0.02,
                f'separation={sep_m:.3f}\ncluster radius={agg_m:.3f}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', fc='white', alpha=0.75))
        print(f"  [{mode}] 类间分离度(质心距离均值)={sep_m:.3f}, "
              f"同类聚集(类内半径均值)={agg_m:.3f}")

    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'voxel_vocab_physics_modes_compare'))


# ============================================================
# 主入口
# ============================================================
def main():
    ap = argparse.ArgumentParser(description='微元词表临近性分析 (增强版)')
    ap.add_argument('--n', type=int, default=None, help='扫描结构数 (默认用全库缓存)')
    ap.add_argument('--no-cache', action='store_true', help='不用全库缓存, 强制现场扫描')
    ap.add_argument('--out', type=str, default='./plots/sci/vocab',
                    help='输出目录')
    ap.add_argument('--modes', action='store_true',
                    help='输出三种物理向量 (rich8/hexa9/basic5) PCA 对比图')
    ap.add_argument('--vocab-file', type=str, default=None,
                    help='直接用已有词表文件 (如 cache/voxel_vocab.pkl), '
                         '跳过全库扫描 (快速分析固定词表)')
    ap.add_argument('--augment', action='store_true',
                    help='临时扩充样本: 为每个微元生成偏位变体 (仅可视化, '
                         '不改数据库), 让 PCA 图更丰富')
    ap.add_argument('--aug_off', type=int, default=4,
                    help='--augment 时每个微元生成多少偏位变体 (默认 4)')
    ap.add_argument('--aug_seed', type=int, default=42,
                    help='--augment 随机种子 (默认 42)')
    ap.add_argument('--rich8_offw', type=float, default=None,
                    help='rich8 偏位降权系数 (0~1, 默认 None=不降权; '
                         '建议 0.3: 避免偏位主导 PCA 使同类型点拉散)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 直接用已有词表文件 (快速, 词表固定)
    if args.vocab_file:
        if not os.path.exists(args.vocab_file):
            print(f"[X] 词表文件不存在: {args.vocab_file}")
            return
        with open(args.vocab_file, 'rb') as f:
            data = __import__('pickle').load(f)
        vocab = VoxelVocab()
        vocab.id2micro = data['id2micro']
        vocab.counts = data.get('counts', {})
        vocab.micro2id = {v: k for k, v in vocab.id2micro.items()}
        vocab.built = True
        use_cache = True
        print(f"[load] vocab file {args.vocab_file} "
              f"({len(vocab.id2micro)} tokens)")
    else:
        use_cache = (args.n is None) and (not args.no_cache)
        vocab = load_vocab_counts(args.n, use_cache=use_cache)

    # 临时扩充样本 (考虑偏位): 为每个微元生成偏位变体
    if args.augment:
        n_before = len(vocab.id2micro)
        vocab = build_augmented_vocab(
            vocab, n_off_per_key=args.aug_off, seed=args.aug_seed)
        print(f"[augment] 词表扩充: {n_before} -> {len(vocab.id2micro)} "
              f"微元 (偏位变体)")

    feats = build_feature_table(vocab, mode='rich8',
                                rich8_offw=args.rich8_offw)
    Z, evr = pca_project(feats['X'])
    print(f"PCA: PC1={evr[0]*100:.1f}% PC2={evr[1]*100:.1f}% "
          f"(累计 {evr.sum()*100:.1f}%)")
    cnt = Counter(feats['labels'][feats['nz']])
    print("非空微元类型分布:", dict(cnt))
    print(f"非空微元数: {feats['nz'].sum()} (含空 {len(feats['keys'])})")

    suf = '' if use_cache else f' (scan {args.n})'

    # 三物理向量 PCA 对比 (rich8 / hexa9 / basic5 并排)
    if args.modes:
        print("\n三种物理向量 PCA 对比:")
        plot_physics_modes_compare(vocab, args.out, title_suffix=suf,
                                   rich8_offw=args.rich8_offw)

    plot_pca_2d(feats, Z, evr, args.out, title_suffix=suf)
    plot_stiffness_spectrum(feats, Z, evr, args.out, title_suffix=suf)
    plot_combo_map(feats, args.out, title_suffix=suf)

    print(f"\n[done] output dir: {args.out}")


if __name__ == '__main__':
    main()
