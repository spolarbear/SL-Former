# sample_diversity.py
"""
样本集空间离散性 / 最近邻区分度 定量分析

用途:
    评估样本集向量化后在特征空间中的分布情况, 判断是否需要加密样本:
    1. 结构参数空间 (frame_features 44维): 结构多样性
    2. 输入空间 (结构 + 地震动特征): 输入组合多样性
    3. 响应空间 (位移时程特征): 输出多样性

定量指标 (每类特征空间):
    - 最近邻距离 (k=1 NN distance) 分布: 中位/均值/P5/P95
    - 特征空间尺度 (对角线长度/各维 std): 判断离散程度
    - 覆盖率 (PCA 主成分占比 / 有效维数)
    - 相似样本占比 (NN 距离 < 阈值 的比例, 表示冗余)
    - 是否需要加密: 若 NN 距离小(样本密集)或大(局部稀疏), 给出建议

用法:
    python sample_diversity.py                     # 全部分析
    python sample_diversity.py --num 20000          # 抽样分析
    python sample_diversity.py --out ./plots/sci/diversity
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_manager import SLFDatabase, ST_TABLE, SP_TABLE
from config import Config

SCI_DPI = 300
SCI_FORMATS = ['pdf', 'png']
COLOR_BLUE = '#4A7DB4'
COLOR_GREEN = '#6B8E6B'
COLOR_RED = '#B85C4A'
COLOR_ORANGE = '#D4A574'

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


def save_sci_figure(fig, path, dpi=None, formats=None):
    dpi = dpi or SCI_DPI
    formats = formats or SCI_FORMATS
    for fmt in formats:
        fig.savefig(f"{path}.{fmt}", dpi=dpi, format=fmt, bbox_inches='tight',
                    facecolor='white')


# ============================================================
# 数据加载: 结构 frame_features + 地震动 + 响应
# (抽样逻辑与训练完全一致: 按楼层均匀分层抽样)
# ============================================================
def load_features(db, num_samples=None, seed=42):
    """加载样本特征: 返回 (struct_feats, input_feats, resp_feats, params)

    抽样: 与 train 训练一致的多维分层均匀抽样
          (复用 dataset._stratified_sample_uniform: 楼层均匀 + 响应量级 +
           结构形态), 保证离散性分析的样本分布与训练数据分布相同。
    """
    from dataset import _stratified_sample_uniform, filter_rows_for_training
    rows = db.query_samples()
    # 与训练一致: 先 PGA 筛选 + 剔除顶点位移超标样本
    rows = filter_rows_for_training(rows, Config())
    n_rows = len(rows)
    if num_samples and num_samples < n_rows:
        rng = np.random.default_rng(seed)
        cfg = Config()
        n_peak = int(getattr(cfg, 'DB_STRATIFY_RESPONSE_BINS', 3) or 0)
        use_struct = bool(getattr(cfg, 'DB_STRATIFY_STRUCT', True))
        peak_bins = None if n_peak < 2 else []
        struct_bins = {} if use_struct else None
        rows = _stratified_sample_uniform(rows, num_samples, rng,
                                          peak_bins=peak_bins,
                                          struct_bins=struct_bins)
        # 统计每层数量 (与训练打印一致)
        from collections import Counter
        dist = Counter(int(r['num_stories']) for r in rows)
        dist_str = ", ".join(f"{k}层:{v}" for k, v in sorted(dist.items()))
        print(f"  🎲 按楼层分层抽 {len(rows)} 个 (seed={seed}) [{dist_str}]")

    struct_list, motion_list, resp_list = [], [], []
    params_list = []
    for r in rows:
        resp = db.get_sample(r['sample_id'])
        if resp is None or resp.get('roof_disp') is None:
            continue
        struct = db.get_structure(resp['struct_id'])
        gm = db.get_ground_motion(resp['gm_id'])
        if struct is None or gm is None or gm.get('motion') is None:
            continue
        params_list.append(SLFDbParams(struct))
        struct_list.append(struct)
        motion_list.append(np.asarray(gm['motion'], dtype=np.float64))
        resp_list.append(np.asarray(resp['roof_disp'], dtype=np.float64))
    return struct_list, motion_list, resp_list, params_list


def SLFDbParams(st):
    """从结构字段重建 8 维 params (与 dataset_db 一致)"""
    masses = st.get('floor_masses') or []
    return np.array([st['num_stories'], st['num_bays_x'], st['num_bays_y'],
                     st['span_x'], st['span_y'], st['story_height'],
                     float(np.mean(masses)) if masses else 0.0, 0.05], dtype=np.float32)


def build_struct_feature_matrix(struct_list, params_list):
    """构建结构特征矩阵 (frame_features 44维 + 关键标量)"""
    from frame_feature_encoder import encode_frame_batch
    feats, infos = encode_frame_batch(params_list)
    return np.asarray(feats, dtype=np.float64)


def build_motion_feature_matrix(motion_list, dt=0.02):
    """地震动特征: PGA/主频/频带能量/时长 (与 dataset_analysis 一致)"""
    from earthquake_simulator_3d import EarthquakeLoader3D
    feats = []
    for m in motion_list:
        m = np.asarray(m, dtype=np.float64)
        pga = np.max(np.abs(m))
        n = len(m)
        fs = 1.0 / dt
        win = m * np.hanning(n)
        spec = np.abs(np.fft.rfft(win))
        freqs = np.fft.rfftfreq(n, d=dt)
        mask = (freqs > 0.1) & (freqs <= 20.0)
        if mask.any():
            f_main = freqs[mask][np.argmax(spec[mask])]
            total = spec[mask].sum() + 1e-12
            f_low = spec[(freqs >= 0.1) & (freqs < 2.0)].sum() / total
            f_mid = spec[(freqs >= 2.0) & (freqs < 8.0)].sum() / total
            f_high = spec[(freqs >= 8.0) & (freqs <= 20.0)].sum() / total
        else:
            f_main, f_low, f_mid, f_high = 1.0, 1.0, 0.0, 0.0
        feats.append([pga, f_main, f_low, f_mid, f_high])
    return np.asarray(feats, dtype=np.float64)


def build_response_feature_matrix(resp_list):
    """响应特征: 峰值/标准差/RMS/最终值/绝对积分"""
    feats = []
    for d in resp_list:
        d = np.asarray(d, dtype=np.float64)
        feats.append([np.max(np.abs(d)), np.std(d), np.sqrt(np.mean(d**2)),
                      d[-1], np.sum(np.abs(d))])
    return np.asarray(feats, dtype=np.float64)


# ============================================================
# 最近邻 / 离散性分析
# ============================================================
def nearest_neighbor_stats(X, sample_rate=1.0, k=1):
    """用 KD-Tree 算最近邻距离 (抽样加速)

    Returns:
        dict: nn_dist 分布统计
    """
    n = len(X)
    if n < 2:
        return None
    # 抽样 (KD-Tree 查询 O(n log n), 抽样到 ~8000 足够)
    max_q = 8000
    if n > max_q:
        idx = np.random.choice(n, max_q, replace=False)
        Xq = X[idx]
    else:
        Xq = X
    tree = cKDTree(X)
    # 找每个查询点的最近邻 (排除自身, k=2 取第2近)
    dists, _ = tree.query(Xq, k=min(2, n))
    if dists.ndim == 1:
        nn = dists
    else:
        nn = dists[:, 1] if dists.shape[1] >= 2 else dists[:, 0]
    # 只过滤非有限值; 保留 NN=0 的完全重复样本 (这正是多样性分析要报告的冗余)
    nn = nn[np.isfinite(nn)]
    if len(nn) == 0:
        return None
    return {
        'n_queried': len(nn),
        'mean': float(nn.mean()),
        'median': float(np.median(nn)),
        'std': float(nn.std()),
        'p5': float(np.percentile(nn, 5)),
        'p95': float(np.percentile(nn, 95)),
        'min': float(nn.min()),
        'max': float(nn.max()),
        'nn_dist': nn,
    }


def feature_scale_stats(X):
    """特征空间尺度 / 有效维数"""
    X = np.asarray(X, dtype=np.float64)
    stds = X.std(axis=0)
    # 特征空间对角线长度 (归一化后, 保持原维度, 常数列分母用1)
    denom = np.where(stds > 1e-12, stds, 1.0)
    Xn = X / denom
    diag = np.sqrt((Xn.max(axis=0) - Xn.min(axis=0)) ** 2).sum()
    # PCA 有效维数 (累计 90% 方差), 只用非零方差列
    nz = stds > 1e-12
    if nz.sum() == 0:
        return {'n_dim': X.shape[1], 'diag_length': float(diag),
                'eff_dim_90pct': 1, 'std_range': [0.0, 0.0]}
    Xc = Xn - Xn.mean(axis=0)
    try:
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
        var = s ** 2
        cum = np.cumsum(var) / (var.sum() + 1e-12)
        eff_dim = int(np.searchsorted(cum, 0.9) + 1)
    except Exception:
        eff_dim = int(nz.sum())
    return {
        'n_dim': X.shape[1],
        'diag_length': float(diag),
        'eff_dim_90pct': eff_dim,
        'std_range': [float(stds.min()), float(stds.max())],
    }


def analyze_space(name, X, out_dir, threshold_frac=0.05):
    """分析一个特征空间的离散性"""
    print(f"\n{'='*60}")
    print(f"[{name}] 特征空间离散性分析")
    print(f"{'='*60}")
    n = len(X)
    if n < 2:
        print("  样本不足")
        return None

    # 归一化 (标准化, 常数列分母用1避免除零)
    stds = X.std(axis=0)
    Xn = X / np.where(stds > 1e-12, stds, 1.0)
    scale = feature_scale_stats(X)
    print(f"  样本数: {n}, 特征维度: {scale['n_dim']}, "
          f"有效维数(90%方差): {scale['eff_dim_90pct']}")
    print(f"  特征空间对角线长度(归一化): {scale['diag_length']:.1f}")

    # 最近邻
    nn = nearest_neighbor_stats(Xn)
    if nn is None:
        return None
    print(f"  最近邻距离 (归一化特征):")
    print(f"    均值 {nn['mean']:.4f} | 中位 {nn['median']:.4f} | "
          f"P5 {nn['p5']:.4f} | P95 {nn['p95']:.4f} | 最大 {nn['max']:.4f}")

    # 相似样本占比 (NN 距离 < 特征空间对角线*阈值 视为"过于相似")
    # 用相对阈值: 中位最近邻距离的 1/10 视为"几乎重复"
    med = nn['median']
    near_thresh = med * 0.1 if med > 0 else 1e-6
    near_frac = float((nn['nn_dist'] < near_thresh).mean())
    print(f"  几乎重复样本占比 (NN距离<中位/10): {near_frac*100:.2f}%")

    # 相对离散度: 中位NN距离 / 特征空间对角线长度
    # (归一化空间中, 最近邻占特征空间跨度的比例; 越小=样本越密, 越大=越稀疏)
    diag_len = scale['diag_length']
    rel_disp = med / (diag_len + 1e-12)
    # 每有效维的平均 NN 距离 (近似均匀采样的网格间距)
    nn_per_dim = med / max(np.sqrt(scale['eff_dim_90pct']), 1.0)
    print(f"  相对离散度 (中位NN/特征空间跨度): {rel_disp:.4f}")
    print(f"  每有效维平均NN距离: {nn_per_dim:.4f}")

    # 是否需要加密
    advice = []
    if rel_disp < 0.02:
        advice.append("特征空间样本过密(相对离散度<0.02), 存在冗余, 可去重而非加密")
    elif rel_disp < 0.05:
        advice.append("特征空间离散度适中, 采样较充分")
    elif rel_disp < 0.1:
        advice.append("特征空间离散度偏大, 局部可能稀疏, 建议加密(尤其稀疏区域)")
    else:
        advice.append("特征空间离散度大, 采样稀疏, 强烈建议加密样本")
    if near_frac > 0.05:
        advice.append(f"注意: {near_frac*100:.1f}% 样本几乎重复(冗余), 可去重")
    print(f"  建议: {'; '.join(advice)}")

    return {
        'name': name, 'n': n, 'n_dim': scale['n_dim'],
        'eff_dim': scale['eff_dim_90pct'], 'diag': scale['diag_length'],
        'nn_mean': nn['mean'], 'nn_median': med, 'nn_p5': nn['p5'],
        'nn_p95': nn['p95'], 'nn_max': nn['max'],
        'near_frac': near_frac, 'rel_disp': rel_disp,
        'nn_per_dim': float(nn_per_dim),
        'advice': '; '.join(advice),
        'nn_dist': nn['nn_dist'],
    }


# ============================================================
# 绘图
# ============================================================
def plot_diversity(results, out_dir):
    """绘制各空间最近邻距离分布 (自适应分箱, 减少空隙)

    分箱策略:
      - 偏度小 (接近对称): 用等宽 bin (适量减少 bin 数)
      - 偏度大 (右偏重尾): 用对数分箱, 让尾部也覆盖到
      - bin 数按样本量自适应 (n/40 ~ 60, 但不超过 50), 减少空 bin
    """
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    valid = [r for r in results if r is not None]
    if not valid:
        return
    fig, axes = plt.subplots(1, len(valid), figsize=(5.5 * len(valid), 4.5))
    if len(valid) == 1:
        axes = [axes]
    for ax, r in zip(axes, valid):
        d = np.asarray(r['nn_dist'], dtype=np.float64)
        d = d[np.isfinite(d)]
        n = len(d)
        if n == 0:
            continue
        # 自适应 bin 数 (样本越多 bin 越多, 但封顶 50 减少空隙)
        n_bins = max(20, min(50, n // 40))
        # 偏度判断: 右偏重尾 -> 对数分箱; 否则等宽
        std = d.std()
        skew = float(((d - d.mean()) ** 3).mean() / (std ** 3 + 1e-12)) if std > 0 else 0.0
        if skew > 1.5 and d.min() > 0 and d.max() / max(d.min(), 1e-12) > 10:
            bins = np.logspace(np.log10(d.min()), np.log10(d.max()), n_bins + 1)
        else:
            bins = n_bins
        ax.hist(d, bins=bins, color=COLOR_BLUE, edgecolor='black', alpha=0.75)
        ax.axvline(r['nn_median'], color='red', linestyle='--', linewidth=1.5,
                   label=f"median={r['nn_median']:.3f}")
        ax.axvline(r['nn_p95'], color='orange', linestyle='-.', linewidth=1.5,
                   label=f"P95={r['nn_p95']:.3f}")
        ax.set_xlabel('Nearest-neighbor distance (normalized)')
        ax.set_ylabel('Count')
        ax.set_title(f"{r['name']}\n(n={r['n']}, eff_dim={r['eff_dim']})")
        ax.legend(fontsize=8)
        ax.tick_params(axis='both', labelsize=8, direction='in')
    plt.tight_layout()
    save_sci_figure(fig, os.path.join(out_dir, 'nn_distance_distribution'))
    plt.close()


# ============================================================
# 均匀抽样分布图 (正常训练时的抽样分布)
# ------------------------------------------------------------
# 展示按"多维分层均匀抽样"原则抽取样本时, 各分层维度的分布情况,
# 验证抽样均匀性 (楼层均匀 / 每层内大-小变形均匀 / 结构形态均匀)。
# 与 dataset_analysis 的 sampling_distribution 一致, 保证离散性分析
# 所用的抽样分布可复现可视化。
# ============================================================
def plot_sampling_distribution(db, out_dir, num_samples=None, seed=42):
    """绘制均匀抽样原则下的样本分布图 (与训练抽样逻辑完全一致)

    抽样: 复用 dataset._stratified_sample_uniform (楼层+响应量级+结构形态),
          config 开关控制各维度启用 (与 train 训练一致)。
    图:   sampling_distribution.pdf/png
          - 楼层分布柱状图 (均匀参考线)
          - 每层内响应量级分档 (低/中/高变形 分组柱状图, 验证 1:1:1)
          - 结构形态分档分布 (跨度大小×层高×跨数)
          - 楼层 × 响应量级 热力图 (验证每层内大小变形均匀)
    """
    from dataset import _stratified_sample_uniform, filter_rows_for_training

    os.makedirs(out_dir, exist_ok=True)
    cfg = Config()
    n_peak = int(getattr(cfg, 'DB_STRATIFY_RESPONSE_BINS', 3) or 0)
    use_struct = bool(getattr(cfg, 'DB_STRATIFY_STRUCT', True))

    # ---- 抽样 (与 load_features / train 训练一致) ----
    rows = db.query_samples()
    # 与训练一致: 先 PGA 筛选 + 剔除顶点位移超标样本
    rows = filter_rows_for_training(rows, cfg)
    n_rows = len(rows)
    if num_samples and num_samples < n_rows:
        rng = np.random.default_rng(seed)
        peak_bins = None if n_peak < 2 else []
        struct_bins = {} if use_struct else None
        rows = _stratified_sample_uniform(rows, num_samples, rng,
                                          peak_bins=peak_bins,
                                          struct_bins=struct_bins)
    n = len(rows)
    print(f"  ✓ 均匀抽样分布图: 抽样 {n} / {n_rows} (seed={seed})")

    # ---- 提取分层维度 ----
    stories = np.array([int(r['num_stories']) for r in rows])
    peaks = np.array([float(r['disp_peak']) for r in rows])
    spans = np.array([float(r['span_x']) for r in rows])
    heights = np.array([float(r['story_height']) for r in rows])
    bays = np.array([int(r['num_bays_x']) for r in rows])

    story_vals = sorted(set(stories.tolist()))
    peak_labels = ['Low', 'Mid', 'High'] if n_peak >= 2 else ['All']

    # 每层内响应量级分档 (与 _stratified_sample_uniform 相同的组内分位)
    peak_bin_of = np.zeros(n, dtype=int)
    if n_peak >= 2:
        for ns in story_vals:
            mask = stories == ns
            vals = peaks[mask]
            if vals.std() > 1e-12:
                lo, hi = np.percentile(vals, [33.3, 66.7])
                peak_bin_of[mask] = np.clip(
                    np.searchsorted([lo, hi], peaks[mask]), 0, n_peak - 1)
    # 结构形态分档 (与 _stratified_sample_uniform 一致)
    span_split = float(np.median(spans))
    height_split = float(np.median(heights))
    span_bin = (spans > span_split).astype(int)
    height_bin = (heights > height_split).astype(int)
    bay_bin = (bays > 2).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ---- (0,0) 楼层分布 ----
    ax = axes[0, 0]
    counts = Counter(stories.tolist())
    xs = list(story_vals)
    ys = [counts.get(x, 0) for x in xs]
    ax.bar(xs, ys, color=COLOR_BLUE, edgecolor='black', alpha=0.8, width=0.7)
    ax.axhline(n / len(story_vals), color=COLOR_RED, linestyle='--', linewidth=1.3,
               label=f'uniform={n/len(story_vals):.0f}')
    ax.set_xlabel('Number of stories')
    ax.set_ylabel('Count')
    ax.set_title(f'Stories distribution (n={n})')
    ax.set_xticks(xs)
    ax.legend(fontsize=8)
    ax.tick_params(axis='both', labelsize=9, direction='in')

    # ---- (0,1) 每层内响应量级分组柱状图 ----
    ax = axes[0, 1]
    width = 0.28
    x_pos = np.arange(len(story_vals))
    for bi, lab in enumerate(peak_labels):
        vals = [np.sum((stories == ns) & (peak_bin_of == bi)) for ns in story_vals]
        ax.bar(x_pos + (bi - (len(peak_labels) - 1) / 2) * width, vals, width,
               label=lab, edgecolor='black', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(story_vals)
    ax.set_xlabel('Stories')
    ax.set_ylabel('Count')
    ax.set_title('Response magnitude per story (Low/Mid/High)')
    ax.legend(fontsize=8, ncol=len(peak_labels))
    ax.tick_params(axis='both', labelsize=9, direction='in')

    # ---- (1,0) 结构形态分档分布 ----
    ax = axes[1, 0]
    struct_labels = ['Sx- · Hy- · B-', 'Sx+ · Hy- · B-', 'Sx- · Hy+ · B-',
                     'Sx+ · Hy+ · B-', 'Sx- · Hy- · B+', 'Sx+ · Hy- · B+',
                     'Sx- · Hy+ · B+', 'Sx+ · Hy+ · B+']
    comb = span_bin * 4 + height_bin * 2 + bay_bin
    comb_counts = Counter(comb.tolist())
    ys = [comb_counts.get(i, 0) for i in range(8)]
    colors = [COLOR_BLUE, COLOR_GREEN, COLOR_ORANGE, COLOR_RED]
    ax.bar(range(8), ys, color=[colors[c // 2] for c in range(8)],
           edgecolor='black', alpha=0.85)
    ax.set_xticks(range(8))
    ax.set_xticklabels(struct_labels, rotation=30, ha='right', fontsize=7.5)
    ax.set_ylabel('Count')
    ax.set_title(f'Structure form bins (span×height×bays, split '
                 f'{span_split:.1f}m/{height_split:.1f}m)')
    ax.tick_params(axis='both', labelsize=8, direction='in')

    # ---- (1,1) 楼层 × 响应量级热力图 ----
    ax = axes[1, 1]
    mat = np.zeros((len(story_vals), max(len(peak_labels), 1)), dtype=int)
    for i, ns in enumerate(story_vals):
        for bi in range(max(len(peak_labels), 1)):
            mat[i, bi] = np.sum((stories == ns) & (peak_bin_of == bi))
    im = ax.imshow(mat, aspect='auto', cmap='YlGnBu')
    ax.set_xticks(range(len(peak_labels)))
    ax.set_xticklabels(peak_labels)
    ax.set_yticks(range(len(story_vals)))
    ax.set_yticklabels(story_vals)
    ax.set_xlabel('Response magnitude')
    ax.set_ylabel('Stories')
    ax.set_title('Stories × response magnitude')
    for i in range(len(story_vals)):
        for j in range(len(peak_labels)):
            ax.text(j, i, int(mat[i, j]), ha='center', va='center',
                    fontsize=8, color='black')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    save_sci_figure(fig, os.path.join(out_dir, 'sampling_distribution'))
    plt.close()
    print(f"  ✓ 均匀抽样分布图: {out_dir}/sampling_distribution.pdf/.png")


# ============================================================
# 参数组合覆盖度缺口分析 (找出缺失/稀疏的样本组合)
# ============================================================
def _load_combo_space(db):
    """从数据库实际取值构建离散参数组合空间 (避免生成规则的窄假设)。

    Returns:
        dict: 各维度的离散取值列表 + floor_load 分档
    """
    from collections import Counter
    db.cur.execute(
        f"""SELECT num_stories, num_bays_x, num_bays_y, span_x, span_y,
                  story_height FROM {ST_TABLE}""")
    rows = db.cur.fetchall()
    c = {k: Counter() for k in
         ['stories', 'baysx', 'baysy', 'spanx', 'spany', 'height']}
    for r in rows:
        c['stories'][int(r['num_stories'])] += 1
        c['baysx'][int(r['num_bays_x'])] += 1
        c['baysy'][int(r['num_bays_y'])] += 1
        c['spanx'][round(float(r['span_x']), 2)] += 1
        c['spany'][round(float(r['span_y']), 2)] += 1
        c['height'][round(float(r['story_height']), 2)] += 1
    # PGA 从 done 样本实际取值
    db.cur.execute(f"SELECT target_pga FROM {SP_TABLE} WHERE sim_status='done'")
    pga = Counter(round(r['target_pga'], 3) for r in db.cur.fetchall())
    space = {
        'stories': sorted(c['stories']),
        'bays': sorted(set(c['baysx']) | set(c['baysy'])),
        'spans': sorted(set(c['spanx']) | set(c['spany'])),
        'heights': sorted(c['height']),
        'pga': sorted(pga),
    }
    return space


def _load_exists_combs(db):
    """统计数据库全量 done 样本的参数组合出现次数.

    floor_load 连续值 -> 按楼层均值分 3 档 (低/中/高), 不算独立离散维度。
    """
    from collections import Counter
    rows = db.query_samples()
    db.cur.execute(f"SELECT struct_id, floor_loads FROM {ST_TABLE}")
    load_cache = {r['struct_id']: r['floor_loads'] for r in db.cur.fetchall()}
    exist = Counter()
    load_hist = []
    for r in rows:
        loads = load_cache.get(r['struct_id'])
        if loads:
            vals = [float(x) for x in loads.split(',')]
            mean_load = float(np.mean(vals))
        else:
            mean_load = 20.0
        load_hist.append(mean_load)
        key = (int(r['num_stories']), int(r['num_bays_x']),
               int(r['num_bays_y']), round(float(r['span_x']), 2),
               round(float(r['span_y']), 2), round(float(r['story_height']), 2),
               round(float(r['target_pga']), 3))
        exist[key] += 1
    load_hist = np.array(load_hist)
    lo, hi = np.percentile(load_hist, [33, 67])
    return exist, (float(lo), float(hi))


def coverage_gap_analysis(db, out_dir, sparse_thresh=5):
    """分析离散参数组合的覆盖度, 找出缺失/稀疏组合并输出补全清单。

    组合空间从数据库实际离散取值构建 (层数 × 跨数² × 跨度² × 层高 × PGA),
    不把 floor_load 当独立维度 (连续值, 仅做统计参考)。

    输出:
      - coverage_gap.csv   : 缺失/稀疏组合清单 (供补采样)
      - coverage_heatmap   : 层数×跨度 / 层数×PGA 覆盖热力图
    """
    import itertools
    from collections import Counter

    # 1. 构建组合空间
    space = _load_combo_space(db)
    stories = space['stories']; bays = space['bays']
    spans = space['spans']; heights = space['heights']; pga_opts = space['pga']
    all_combs = list(itertools.product(stories, bays, bays, spans, spans,
                                       heights, pga_opts))
    total_combs = len(all_combs)
    print(f"\n{'='*60}")
    print("[参数组合覆盖度分析] 组合空间 (从数据库实际取值)")
    print(f"{'='*60}")
    print(f"  层数{stories} × 跨数{bays}² × 跨度{spans}² × "
          f"层高{heights} × PGA{pga_opts}")
    print(f"  理论组合数: {total_combs}")

    # 2. 统计已有组合
    exist_combs, (lo, hi) = _load_exists_combs(db)
    n_done = sum(exist_combs.values())
    print(f"  数据库 done 样本: {n_done}")
    print(f"  floor_load 分档: 低<{lo:.1f} ≤中<{hi:.1f} ≤高")

    # 3. 找出缺失/稀疏组合
    missing = []
    sparse = []
    covered = 0
    for comb in all_combs:
        c = exist_combs.get(comb, 0)
        if c == 0:
            missing.append(comb)
        elif c < sparse_thresh:
            sparse.append((comb, c))
        else:
            covered += 1
    print(f"  已覆盖组合: {covered} / {total_combs} "
          f"({covered/total_combs*100:.1f}%)")
    print(f"  缺失组合: {len(missing)} ({len(missing)/total_combs*100:.1f}%)")
    print(f"  稀疏组合 (1~{sparse_thresh-1}条): {len(sparse)}")

    # 4. 输出 CSV
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for comb in missing:
        records.append({'status': 'missing', 'count': 0,
                        'num_stories': comb[0], 'num_bays_x': comb[1],
                        'num_bays_y': comb[2], 'span_x': comb[3],
                        'span_y': comb[4], 'story_height': comb[5],
                        'target_pga': comb[6]})
    for comb, c in sparse:
        records.append({'status': 'sparse', 'count': c,
                        'num_stories': comb[0], 'num_bays_x': comb[1],
                        'num_bays_y': comb[2], 'span_x': comb[3],
                        'span_y': comb[4], 'story_height': comb[5],
                        'target_pga': comb[6]})
    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, 'coverage_gap.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  ✅ 缺失/稀疏组合清单: {csv_path} ({len(df)} 条)")

    # 5. 覆盖热力图 (层数×跨度 / 层数×PGA)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    comb_counts = Counter()
    for comb, c in exist_combs.items():
        comb_counts[(comb[0], comb[3])] += c
    grid = np.zeros((len(stories), len(spans)))
    for i, ns in enumerate(stories):
        for j, sp in enumerate(spans):
            grid[i, j] = comb_counts.get((ns, sp), 0)
    im = axes[0].imshow(grid, aspect='auto', cmap='YlGnBu',
                        interpolation='nearest')
    axes[0].set_xticks(range(len(spans))); axes[0].set_xticklabels(spans)
    axes[0].set_yticks(range(len(stories))); axes[0].set_yticklabels(stories)
    axes[0].set_xlabel('Span (m)'); axes[0].set_ylabel('Stories')
    axes[0].set_title('Coverage: Stories × Span')
    for i in range(len(stories)):
        for j in range(len(spans)):
            axes[0].text(j, i, int(grid[i, j]), ha='center', va='center',
                         fontsize=8, color='black')
    plt.colorbar(im, ax=axes[0])
    comb_counts2 = Counter()
    for comb, c in exist_combs.items():
        comb_counts2[(comb[0], comb[6])] += c
    grid2 = np.zeros((len(stories), len(pga_opts)))
    for i, ns in enumerate(stories):
        for j, pg in enumerate(pga_opts):
            grid2[i, j] = comb_counts2.get((ns, pg), 0)
    im2 = axes[1].imshow(grid2, aspect='auto', cmap='YlGnBu',
                         interpolation='nearest')
    axes[1].set_xticks(range(len(pga_opts))); axes[1].set_xticklabels(pga_opts)
    axes[1].set_yticks(range(len(stories))); axes[1].set_yticklabels(stories)
    axes[1].set_xlabel('Target PGA (g)'); axes[1].set_ylabel('Stories')
    axes[1].set_title('Coverage: Stories × PGA')
    for i in range(len(stories)):
        for j in range(len(pga_opts)):
            axes[1].text(j, i, int(grid2[i, j]), ha='center', va='center',
                         fontsize=8, color='black')
    plt.colorbar(im2, ax=axes[1])
    plt.tight_layout()
    save_sci_figure(fig, os.path.join(out_dir, 'coverage_heatmap'))
    plt.close()
    print(f"  ✅ 覆盖热力图: {out_dir}/coverage_heatmap.pdf/.png")

    return {'total_combs': total_combs, 'covered': covered,
            'missing': len(missing), 'sparse': len(sparse),
            'load_range': (lo, hi), 'df': df, 'space': space,
            'exist_combs': exist_combs}


# ============================================================
# 补全缺失/稀疏组合的样本 (针对方形规则平面)
# ============================================================
def fill_missing_samples(db, out_dir, max_samples=500, workers=4,
                         only_square=True, sparse_thresh=5, wave_pool=50,
                         stories=None):
    """补全覆盖缺口: 生成缺失/稀疏组合的结构样本并入数据库仿真。

    策略:
      - 从 coverage_gap.csv 读取缺失/稀疏组合 (只取方形平面 bays_x==bays_y 且
        span_x==span_y, 过滤怪异矩形, 更贴近工程实际)
      - stories: 可选, 只补指定楼层 (如 [7,8,9,10,11])
      - 构造 frame -> get_or_create_structure -> get_or_create_sample
        (pending) -> 复用 db_generate_samples._run_one_sample 并行仿真写库
      - 地震动用数据库波库前 wave_pool 条 (与 generate_samples 一致)

    注意: 仿真较慢 (每样本数秒~数十秒), 建议先小批量 (--fill 100) 试跑。
    """
    from db_generate_samples import build_frame_from_params, _run_one_sample
    from simulation_cache import generate_floor_loads, compute_floor_node_masses
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x

    cfg = Config()
    dt = cfg.TARGET_DT
    seq_len = cfg.get_seq_len()

    # 1. 读取缺口清单
    csv_path = os.path.join(out_dir, 'coverage_gap.csv')
    if not os.path.exists(csv_path):
        print("[ERR] 请先运行 --coverage 生成 coverage_gap.csv")
        return 0
    df = pd.read_csv(csv_path)

    # 2. 只取方形规则平面 (补齐覆盖, 排除怪异矩形)
    square = df[(df['num_bays_x'] == df['num_bays_y']) &
                (df['span_x'] == df['span_y'])]
    if only_square:
        df = square
    # 按楼层过滤 (可选)
    if stories:
        df = df[df['num_stories'].isin(stories)]
    print(f"\n[补全] 方形平面组合: {len(square)} / 总缺口 {len(pd.read_csv(csv_path))}"
          f"{' , 过滤楼层' + str(stories) if stories else ''}")
    print(f"  本次候选缺口: {len(df)}")
    if len(df) == 0:
        print("  无可补全组合")
        return 0

    # 3. 地震动池 (数据库波库前 wave_pool 条)
    waves = db.get_all_ground_motions()
    if len(waves) == 0:
        print("[ERR] 数据库波库为空")
        return 0
    n_w = min(wave_pool, len(waves))
    motion_pool = [w['motion'] for w in waves[:n_w]]
    wave_ids = [w['gm_id'] for w in waves[:n_w]]
    print(f"  使用波库前 {n_w} 条 (gm_id {wave_ids[0]}~{wave_ids[-1]})")

    # 4. 构造结构 + 查重 + 建 pending 样本
    todo = []
    made = skipped = 0
    for i, row in df.iterrows():
        if len(todo) >= max_samples:
            break
        ns = int(row['num_stories']); nx = int(row['num_bays_x'])
        ny = int(row['num_bays_y']); sx = float(row['span_x'])
        sy = float(row['span_y']); sh = float(row['story_height'])
        pga = float(row['target_pga'])

        # 构造结构参数向量 (与 build_frame_from_params 兼容, 21 维: 8+形状+每层荷载)
        from config import Config as _C
        from generate_frames import shape_to_id
        _pdim = int(getattr(_C, 'PARAMS_DIM', 21))
        _poff = int(getattr(_C, 'PARAMS_FLOOR_LOAD_OFFSET', 9))
        _psidx = int(getattr(_C, 'PARAMS_SHAPE_IDX', 8))
        _pmax = int(getattr(_C, 'PARAMS_MAX_FLOORS', 12))
        ms = max(sx, sy)
        bh = max(0.4, min(ms/12, 0.8)); bh = round(bh / 0.2) * 0.2   # 200mm
        bw = max(0.2, min(bh/2.5, 0.5)); bw = round(bw / 0.2) * 0.2  # 200mm
        loads = generate_floor_loads(ns, [15.0, 20.0])
        masses = compute_floor_node_masses(ns, nx, ny, sx, sy, loads)
        params = np.zeros(_pdim, dtype=np.float32)
        params[0] = ns; params[1] = nx; params[2] = ny
        params[3] = sx; params[4] = sy; params[5] = sh
        params[6] = float(np.mean(masses)); params[7] = 0.05
        params[_psidx] = shape_to_id(row.get('plane_shape') or 'rect')
        for _k, _v in enumerate(np.asarray(loads, dtype=np.float32)[:_pmax]):
            params[_poff + _k] = float(_v)
        # 随机柱/梁截面 (与 db_generate_samples 一致)
        from db_generate_samples import _random_col_sections, _random_beam_sections
        col_secs = _random_col_sections(ns, nx, ny, sx, sy, sh)
        beam_secs = _random_beam_sections(ns)
        frame = build_frame_from_params(params, col_sections=col_secs,
                                        beam_sections=beam_secs)
        masses_rounded = np.round(masses / 100.0) * 100.0
        struct_id = db.get_or_create_structure(
            frame, loads, masses_rounded.astype(np.float32))

        # 建样本 (每结构配多条不同波, 用当前缺口的波循环)
        wid = wave_ids[i % len(wave_ids)]
        st = db.get_sample_status(struct_id, wid, pga)
        if st is None:
            sample_id, _ = db.get_or_create_sample(struct_id, wid, pga)
            todo.append((sample_id, params, motion_pool[i % len(motion_pool)],
                         pga, masses_rounded.astype(np.float32),
                         col_secs, beam_secs))
            made += 1
        elif st['sim_status'] != 'done':
            todo.append((st['sample_id'], params,
                         motion_pool[i % len(motion_pool)], pga,
                         masses_rounded.astype(np.float32),
                         col_secs, beam_secs))
            made += 1
        else:
            skipped += 1
    print(f"  建样本: 新增待仿真 {made}, 已存在done跳过 {skipped}")

    if made == 0:
        print("[DONE] 无新增样本")
        return 0

    # 5. 并行仿真写库
    tasks = [(sid, p, m, seq_len, dt, pga, fm, cs, bs)
             for (sid, p, m, pga, fm, cs, bs) in todo]
    todo_by_sid = {t[0]: t for t in todo}
    print(f"[SIM] 启动 {workers} 进程仿真 {made} 个样本...")
    t0 = time.time()
    done = failed = 0
    from db_generate_samples import _flush_results_to_db
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one_sample, a): a[0] for a in tasks}
        results = {}
        for fu in as_completed(futs):
            r = fu.result()
            results[r['sample_id']] = r
            if len(results) >= 50:
                n = _flush_results_to_db(db, results, todo_by_sid, cfg,
                                         commit=True)
                done += n[0]; failed += n[1]
                results.clear()
                print(f"  💾 已保存 (累计 done={done}, failed={failed})")
    if results:
        n = _flush_results_to_db(db, results, todo_by_sid, cfg, commit=True)
        done += n[0]; failed += n[1]
    elapsed = time.time() - t0
    print(f"[SIM] 完成: done={done}, failed={failed}, 用时 {elapsed:.1f}s")
    print("[DONE] 数据库统计:", db.stats())
    return done


# ============================================================
# 恢复 pending 样本 (已有结构, 只差仿真)
# ============================================================
def recover_pending_samples(db, workers=4, stories=None, max_samples=None,
                            wave_pool=50):
    """恢复数据库中 sim_status='pending' 的样本 (结构已存在, 只差仿真)。

    背景: 批量生成中断/超时会产生大量 pending; 高层(7~11) pending 尤其多
          (如 8层 4533 个), 恢复它们比新建结构更省算力。

    流程:
      1. 查询 pending 样本 + 对应结构几何参数 (一次 JOIN)
      2. 按 stories 过滤 (可选)
      3. 重建 params/floor_masses (与 db_generate_samples 确定性一致)
      4. 并行仿真写库

    Returns:
        完成的样本数
    """
    from db_generate_samples import _run_one_sample, _flush_results_to_db, \
        build_frame_from_params
    from simulation_cache import generate_floor_loads, compute_floor_node_masses
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time

    cfg = Config()
    dt = cfg.TARGET_DT
    seq_len = cfg.get_seq_len()

    # 1. 查询 pending 样本 (含结构几何 + 截面)
    db.cur.execute(f"""
        SELECT s.sample_id, s.struct_id, s.gm_id, s.target_pga,
               st.num_stories, st.num_bays_x, st.num_bays_y, st.span_x,
               st.span_y, st.story_height, st.floor_loads, st.floor_masses,
               st.col_sections, st.beam_sections
        FROM {SP_TABLE} s
        JOIN {ST_TABLE} st ON st.struct_id = s.struct_id
        WHERE s.sim_status='pending'
        ORDER BY s.sample_id
    """)
    pend_rows = db.cur.fetchall()
    print(f"\n[恢复] 数据库 pending 样本: {len(pend_rows)}")

    # 按楼层过滤
    if stories:
        pend_rows = [r for r in pend_rows if int(r['num_stories']) in stories]
        print(f"  -> 过滤楼层 {stories}: {len(pend_rows)} 个")
    if max_samples:
        pend_rows = pend_rows[:max_samples]
        print(f"  -> 限制 {max_samples} 个")

    if not pend_rows:
        print("  无 pending 样本可恢复")
        return 0

    # 2. 地震动池
    waves = db.get_all_ground_motions()
    if len(waves) == 0:
        print("[ERR] 数据库波库为空")
        return 0
    n_w = min(wave_pool, len(waves))
    motion_pool = [w['motion'] for w in waves[:n_w]]
    wave_ids = [w['gm_id'] for w in waves[:n_w]]
    gm_map = {w['gm_id']: w['motion'] for w in waves}
    print(f"  使用波库前 {n_w} 条 (gm_id {wave_ids[0]}~{wave_ids[-1]})")

    # 3. 重建 params / floor_masses (确定性, 与生成时一致)
    todo = []
    n_skip = 0
    for r in pend_rows:
        sid = int(r['sample_id'])
        ns = int(r['num_stories']); nx = int(r['num_bays_x'])
        ny = int(r['num_bays_y']); sx = float(r['span_x'])
        sy = float(r['span_y']); sh = float(r['story_height'])
        pga = float(r['target_pga'])
        motion = gm_map.get(int(r['gm_id']))
        if motion is None:
            n_skip += 1
            continue
        # floor_masses 优先用数据库存的 (若存在), 否则重建
        if r['floor_masses']:
            masses = np.array([float(x) for x in r['floor_masses'].split(',')],
                              dtype=np.float32)
            loads_arr = [float(x) for x in r['floor_loads'].split(',')] \
                if r['floor_loads'] else None
        else:
            loads = [float(x) for x in r['floor_loads'].split(',')] \
                if r['floor_loads'] else generate_floor_loads(ns, [15.0, 20.0])
            loads_arr = loads
            masses = np.round(compute_floor_node_masses(
                ns, nx, ny, sx, sy, loads) / 100.0) * 100.0
            masses = masses.astype(np.float32)
        ms = max(sx, sy)
        bh = max(0.4, min(ms/12, 0.8)); bh = round(bh / 0.2) * 0.2   # 200mm
        bw = max(0.2, min(bh/2.5, 0.5)); bw = round(bw / 0.2) * 0.2  # 200mm
        # 21 维 params: 前 8 维 + 形状 ID + 每层荷载 (kPa, 最多 12 层, 不足补 0)
        from config import Config as _C
        from generate_frames import shape_to_id
        _pdim = int(getattr(_C, 'PARAMS_DIM', 21))
        _poff = int(getattr(_C, 'PARAMS_FLOOR_LOAD_OFFSET', 9))
        _psidx = int(getattr(_C, 'PARAMS_SHAPE_IDX', 8))
        _pmax = int(getattr(_C, 'PARAMS_MAX_FLOORS', 12))
        params = np.zeros(_pdim, dtype=np.float32)
        params[0] = ns; params[1] = nx; params[2] = ny
        params[3] = sx; params[4] = sy; params[5] = sh
        params[6] = float(np.mean(masses)); params[7] = 0.05
        params[_psidx] = shape_to_id(r.get('plane_shape') or 'rect')
        if loads_arr:
            for _k, _v in enumerate(np.asarray(loads_arr, dtype=np.float32)[:_pmax]):
                params[_poff + _k] = float(_v)
        # 用数据库存的截面 (柱/梁逐层)
        col_secs = [float(x) for x in r['col_sections'].split(',')] \
            if r['col_sections'] else None
        beam_secs = None
        if r['beam_sections']:
            beam_secs = [tuple(float(y) for y in x.split(','))
                         for x in r['beam_sections'].split(';')]
        todo.append((sid, params, motion, pga, masses, col_secs, beam_secs))
    print(f"  可恢复: {len(todo)} (跳过波缺失 {n_skip})")
    if not todo:
        return 0

    # 4. 并行仿真写库
    tasks = [(sid, p, m, seq_len, dt, pga, fm, cs, bs)
             for (sid, p, m, pga, fm, cs, bs) in todo]
    todo_by_sid = {t[0]: t for t in todo}
    print(f"[SIM] 启动 {workers} 进程恢复仿真 {len(todo)} 个样本...")
    t0 = time.time()
    done = failed = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one_sample, a): a[0] for a in tasks}
        results = {}
        for fu in as_completed(futs):
            r = fu.result()
            results[r['sample_id']] = r
            if len(results) >= 50:
                n = _flush_results_to_db(db, results, todo_by_sid, cfg,
                                         commit=True)
                done += n[0]; failed += n[1]
                results.clear()
                print(f"  💾 已保存 (累计 done={done}, failed={failed})")
    if results:
        n = _flush_results_to_db(db, results, todo_by_sid, cfg, commit=True)
        done += n[0]; failed += n[1]
    elapsed = time.time() - t0
    print(f"[SIM] 完成: done={done}, failed={failed}, 用时 {elapsed:.1f}s")
    print("[DONE] 数据库统计:", db.stats())
    return done


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='样本集空间离散性/最近邻分析')
    parser.add_argument('--num', type=int, default=5000, help='抽样样本数')
    parser.add_argument('--out', type=str, default='./plots/sci/diversity', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='抽样种子')
    parser.add_argument('--coverage', action='store_true',
                        help='额外做参数组合覆盖度缺口分析 (全量数据库样本)')
    parser.add_argument('--sparse-thresh', type=int, default=5,
                        help='稀疏组合阈值 (count < 该值视为稀疏), 默认5')
    parser.add_argument('--fill', type=int, default=0, metavar='N',
                        help='补全操作: 生成缺失/稀疏方形组合的样本 (N=最多新样本数)')
    parser.add_argument('--stories', type=str, default=None,
                        help='补全/恢复时只处理指定楼层, 逗号分隔如 7,8,9,10,11')
    parser.add_argument('--recover', action='store_true',
                        help='恢复 pending 样本仿真 (已有结构只差仿真, 比新建省算力)')
    parser.add_argument('--workers', type=int, default=4, help='仿真并行进程数')
    parser.add_argument('--wave-pool', type=int, default=50,
                        help='补全时使用的波库波形数 (默认50)')
    args = parser.parse_args()

    # 解析楼层过滤
    stories = None
    if args.stories:
        stories = [int(x) for x in args.stories.split(',') if x.strip()]

    os.makedirs(args.out, exist_ok=True)
    cfg = Config()
    db = SLFDatabase()

    print("=" * 60)
    print("样本集空间离散性 / 最近邻区分度分析")
    print("=" * 60)

    print("\n加载样本特征...")
    struct_list, motion_list, resp_list, params_list = load_features(
        db, num_samples=args.num, seed=args.seed)
    n = len(struct_list)
    print(f"  加载 {n} 个样本")

    # 结构特征矩阵
    print("\n构建结构特征矩阵...")
    S = build_struct_feature_matrix(struct_list, params_list)
    # 地震动特征矩阵
    print("构建地震动特征矩阵...")
    M = build_motion_feature_matrix(motion_list)
    # 响应特征矩阵
    print("构建响应特征矩阵...")
    R = build_response_feature_matrix(resp_list)
    # 组合输入 (结构 + 地震动)
    print("构建组合输入特征矩阵...")
    I = np.hstack([S, M])

    results = []
    results.append(analyze_space('Structure (frame_feat)', S, args.out))
    results.append(analyze_space('Ground Motion', M, args.out))
    results.append(analyze_space('Response (disp)', R, args.out))
    results.append(analyze_space('Input (struct+motion)', I, args.out))

    # 绘图
    plot_diversity(results, args.out)

    # 均匀抽样分布图 (正常训练时的抽样分布)
    plot_sampling_distribution(db, args.out, num_samples=args.num, seed=args.seed)

    # CSV
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'nn_dist'} for r in results
                       if r is not None])
    csv_path = os.path.join(args.out, 'diversity_metrics.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 完成! 指标 CSV: {csv_path}")

    # 参数组合覆盖度缺口分析 (可选)
    if args.coverage:
        coverage_gap_analysis(db, args.out, sparse_thresh=args.sparse_thresh)

    # 恢复 pending 样本仿真 (可选): 已有结构只差仿真, 比新建省算力
    if args.recover:
        recover_pending_samples(db, workers=args.workers, stories=stories,
                                wave_pool=args.wave_pool)

    # 补全操作 (可选): 生成缺失/稀疏组合样本
    if args.fill > 0:
        fill_missing_samples(db, args.out, max_samples=args.fill,
                             workers=args.workers, wave_pool=args.wave_pool,
                             sparse_thresh=args.sparse_thresh, stories=stories)

    db.close()


if __name__ == '__main__':
    main()
