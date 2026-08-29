# dataset_analysis.py
"""
样本集 (PostgreSQL) 主要参数分布分析与 SCI 图表输出

功能:
1. 从数据库统计样本集三大部分参数分布:
   - 地震波 (ground_motions) : PGA / 时长 / 步数 / 频带能量
   - 结构   (structures)     : 层数 / 跨数 / 跨度 / 层高 / 总高 / 总质量 / 柱梁数 / 高宽比
   - 计算结果 (samples)      : 位移峰值 / 位移std / 位移终点 / 层间位移角(估算)
2. 输出 SCI 论文配图 (PDF + 300dpi PNG):
   - distribution_pga.png    : 地震波 PGA 分布
   - distribution_motion.png : 地震波时长/步数/主频分布
   - distribution_struct.png : 结构参数分布 (层数/跨数/跨度/层高/总高/质量)
   - distribution_response.png: 计算结果分布 (峰值位移/位移std/层间位移角)
   - distribution_overview.png: 全部关键参数分布总览 (多子图)
3. 输出 CSV:
   - dataset_params.csv       : 每个样本的参数明细 (波/结构/结果)
   - dataset_summary.csv      : 各参数描述性统计 (均值/标准差/中位/P5/P95)
   - dataset_histogram.csv    : 各参数分布直方图 (用于论文表)

用法:
    python dataset_analysis.py                 # 全部输出
    python dataset_analysis.py --out ./plots/sci/dataset
    python dataset_analysis.py --csv ./data
    python dataset_analysis.py --num 5000      # 只统计随机抽取 N 个样本
"""

import os
import io
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from config import Config
from db_manager import SLFDatabase

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

SCI_DPI = 300
SCI_FORMATS = ['pdf', 'png']
COLOR_BLUE = '#4A7DB4'
COLOR_GREEN = '#6B8E6B'
COLOR_RED = '#B85C4A'
COLOR_ORANGE = '#D4A574'


def save_sci_figure(fig, path, dpi=None, formats=None):
    dpi = dpi or SCI_DPI
    formats = formats or SCI_FORMATS
    for fmt in formats:
        fig.savefig(f"{path}.{fmt}", dpi=dpi, format=fmt, bbox_inches='tight',
                    facecolor='white')


# ============================================================
# 数据加载与特征提取
# ============================================================
def _filter_excess_drift(rows):
    """剔除顶点位移超标的样本 (与训练 dataset._load_from_db 完全一致)

    判定: disp_peak(mm)/1000 > total_height(m) × DB_MAX_DRIFT_RATIO
    开关: config.DB_FILTER_MAX_DRIFT (True 时启用)

    Args:
        rows: query_samples() 返回的行列表

    Returns:
        筛选后的行列表
    """
    cfg = Config()
    if not getattr(cfg, 'DB_FILTER_MAX_DRIFT', True):
        return rows
    ratio_th = float(getattr(cfg, 'DB_MAX_DRIFT_RATIO', 0.005))
    kept = []
    n_drop = 0
    for r in rows:
        peak_mm = float(r.get('disp_peak') or 0.0)
        h_m = float(r.get('total_height') or 0.0)
        # 无位移/高度数据时保守保留
        if peak_mm <= 0 or h_m <= 0:
            kept.append(r)
            continue
        drift = (peak_mm / 1000.0) / h_m
        if drift > ratio_th:
            n_drop += 1
            continue
        kept.append(r)
    if n_drop > 0:
        print(f"  🚫 剔除顶点位移>总高{ratio_th*100:.1f}% 的样本 {n_drop} 个 "
              f"({len(rows)} -> {len(kept)})")
    return kept


def load_dataset(db, num_samples=None, seed=42):
    """从数据库加载样本集参数 (波/结构/结果)

    抽样: 与训练一致的多维分层均匀抽样 (楼层+响应量级+结构形态),
          保证分布分析与训练样本集同分布 (最新原则)。
    筛选: 抽样前先剔除顶点位移>总高阈值的样本 (与训练一致)。
    """
    from dataset import _stratified_sample_uniform
    rows = db.query_samples()
    if not rows:
        return None, None, None

    # 剔除顶点位移超标的样本 (与训练 dataset._load_from_db 一致)
    rows = _filter_excess_drift(rows)
    if not rows:
        print("  [X] 剔除超标样本后无可用样本")
        return None, None, None

    # 多维分层均匀抽样 (与 train 训练一致)
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
        print(f"  多维分层均匀抽样 {len(rows)} / {n_rows} 样本 "
              f"(seed={seed}, 楼层+响应量级+结构形态)")

    motion_rows = []
    struct_rows = []
    resp_rows = []
    for r in rows:
        sid = r['sample_id']
        resp = db.get_sample(sid)
        if resp is None or resp.get('roof_disp') is None:
            continue
        struct = db.get_structure(resp['struct_id'])
        gm = db.get_ground_motion(resp['gm_id'])
        if struct is None or gm is None or gm.get('motion') is None:
            continue
        motion_rows.append(gm)
        struct_rows.append(struct)
        resp_rows.append(resp)

    return motion_rows, struct_rows, resp_rows


def motion_features(motion, dt):
    """从地震波时程提取特征: PGA / 时长 / 步数 / 主频 / 频带能量占比"""
    motion = np.asarray(motion, dtype=np.float64)
    if motion is None or len(motion) == 0:
        return None
    pga = np.max(np.abs(motion))
    dur = len(motion) * dt
    # 主频 (FFT 峰值频率, 0.1~20 Hz 范围)
    fs = 1.0 / dt
    n = len(motion)
    if n > 8:
        win = motion * np.hanning(n)
        spec = np.abs(np.fft.rfft(win))
        freqs = np.fft.rfftfreq(n, d=dt)
        m = (freqs > 0.1) & (freqs <= 20.0)
        if m.any():
            f_main = freqs[m][np.argmax(spec[m])]
        else:
            f_main = 1.0
        # 频带能量占比 (0.1~2 Hz, 2~8 Hz, 8~20 Hz)
        bands = [(0.1, 2.0), (2.0, 8.0), (8.0, 20.0)]
        total = spec[m].sum() + 1e-12
        band_frac = []
        for lo, hi in bands:
            bm = (freqs >= lo) & (freqs < hi)
            band_frac.append(spec[bm].sum() / total)
    else:
        f_main, band_frac = 1.0, [1.0, 0.0, 0.0]
    return {
        'pga': pga,
        'duration': dur,
        'n_steps': len(motion),
        'f_main': f_main,
        'f_low': band_frac[0],   # 0.1-2 Hz
        'f_mid': band_frac[1],   # 2-8 Hz
        'f_high': band_frac[2],  # 8-20 Hz
    }


def struct_features(st):
    """从结构字段提取特征 (含自振周期估算)"""
    ns = int(st['num_stories'])
    nx = int(st['num_bays_x'])
    ny = int(st['num_bays_y'])
    sx = float(st['span_x'])
    sy = float(st['span_y'])
    sh = float(st['story_height'])
    th = float(st['total_height'])
    mass = float(st.get('total_mass_kg', 0.0))
    n_col = int(st.get('n_columns', 0))
    n_beam = int(st.get('n_beams', 0))
    floor_loads = st.get('floor_loads') or []
    loads = [float(x) for x in floor_loads] if floor_loads else []
    # 自振周期估算: 用瑞利商基频 (freq -> T=1/f)
    fund_freq = 0.0
    natural_period = 0.0
    try:
        from frame_feature_encoder import estimate_fundamental_frequency
        col_sections = st.get('col_sections') or []
        floor_masses = st.get('floor_masses') or []
        if col_sections and floor_masses and ns > 0:
            omega, f_hz = estimate_fundamental_frequency(
                ns, sh, col_sections,
                [float(m) for m in floor_masses])
            fund_freq = float(f_hz)
            natural_period = 1.0 / fund_freq if fund_freq > 1e-9 else 0.0
    except Exception:
        pass
    return {
        'num_stories': ns,
        'num_bays_x': nx,
        'num_bays_y': ny,
        'span_x': sx,
        'span_y': sy,
        'story_height': sh,
        'total_height': th,
        'total_mass_t': mass / 1000.0,       # kg -> t
        'n_columns': n_col,
        'n_beams': n_beam,
        'aspect_ratio': th / max(sx * nx, sy * ny, 1e-6),  # 高宽比
        'floor_load_avg': float(np.mean(loads)) if loads else 0.0,
        'fund_freq_hz': fund_freq,           # 基频 (Hz)
        'natural_period': natural_period,    # 自振周期 (s)
    }


def response_features(resp, total_height):
    """从计算结果提取特征 (位移 mm)"""
    disp = resp.get('roof_disp')
    if disp is None or len(disp) == 0:
        return None
    disp = np.asarray(disp, dtype=np.float64)
    peak = float(np.max(np.abs(disp)))
    # 层间位移角估算: 峰值位移 / 总高 (简化, 用于分布展示)
    drift = peak / max(total_height * 1000.0, 1e-6)
    return {
        'disp_peak_mm': peak,
        'disp_std_mm': float(np.std(disp)),
        'disp_final_mm': float(disp[-1]),
        'disp_rms_mm': float(np.sqrt(np.mean(disp ** 2))),
        'drift_ratio_est': drift,
        'target_pga': float(resp.get('target_pga', 0.0)),
        'applied_pga': float(resp.get('applied_pga', 0.0)),
    }


# ============================================================
# 分布绘图
# ============================================================
def _hist_ax(ax, data, title, xlabel, bins=30, color=COLOR_BLUE, unit=''):
    """单个直方图子图"""
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    ax.hist(data, bins=bins, color=color, edgecolor='black', alpha=0.75)
    ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=1.2,
               label=f"mean={np.mean(data):.3g}{unit}")
    ax.axvline(np.median(data), color='blue', linestyle='-.', linewidth=1.2,
               label=f"med={np.median(data):.3g}{unit}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(f"{title} (n={len(data)})", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(axis='both', labelsize=8, direction='in')


def plot_distributions(mot_feats, st_feats, resp_feats, out_dir):
    """绘制三类分布图"""
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. 地震波分布 ----
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    _hist_ax(axes[0, 0], [m['pga'] for m in mot_feats], 'Ground motion PGA',
             'PGA (g)', bins=30, color=COLOR_RED)
    _hist_ax(axes[0, 1], [m['duration'] for m in mot_feats], 'Duration',
             'Duration (s)', bins=30, color=COLOR_RED)
    _hist_ax(axes[0, 2], [m['n_steps'] for m in mot_feats], 'Steps',
             'Number of steps', bins=30, color=COLOR_RED)
    _hist_ax(axes[1, 0], [m['f_main'] for m in mot_feats], 'Dominant frequency',
             'Frequency (Hz)', bins=30, color=COLOR_RED)
    _hist_ax(axes[1, 1], [m['f_low'] for m in mot_feats], 'Energy 0.1-2 Hz',
             'Energy fraction', bins=30, color=COLOR_ORANGE)
    _hist_ax(axes[1, 2], [m['f_mid'] for m in mot_feats], 'Energy 2-8 Hz',
             'Energy fraction', bins=30, color=COLOR_ORANGE)
    plt.tight_layout()
    save_sci_figure(fig, os.path.join(out_dir, 'distribution_motion'))
    plt.close(fig)

    # ---- 2. 结构分布 ----
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    _hist_ax(axes[0, 0], [s['num_stories'] for s in st_feats], 'Number of stories',
             'Stories', bins=12, color=COLOR_BLUE)
    _hist_ax(axes[0, 1], [s['num_bays_x'] for s in st_feats], 'Bays in X',
             'Bays', bins=12, color=COLOR_BLUE)
    _hist_ax(axes[0, 2], [s['num_bays_y'] for s in st_feats], 'Bays in Y',
             'Bays', bins=12, color=COLOR_BLUE)
    _hist_ax(axes[0, 3], [s['span_x'] for s in st_feats], 'Span in X',
             'Span (m)', bins=30, color=COLOR_BLUE)
    _hist_ax(axes[1, 0], [s['span_y'] for s in st_feats], 'Span in Y',
             'Span (m)', bins=30, color=COLOR_BLUE)
    _hist_ax(axes[1, 1], [s['story_height'] for s in st_feats], 'Story height',
             'Height (m)', bins=30, color=COLOR_BLUE)
    _hist_ax(axes[1, 2], [s['total_height'] for s in st_feats], 'Total height',
             'Height (m)', bins=30, color=COLOR_GREEN)
    _hist_ax(axes[1, 3], [s['total_mass_t'] for s in st_feats], 'Total mass',
             'Mass (t)', bins=30, color=COLOR_GREEN)
    plt.tight_layout()
    save_sci_figure(fig, os.path.join(out_dir, 'distribution_struct'))
    plt.close(fig)

    # 结构补充: 高宽比 / 柱梁数
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    _hist_ax(axes[0], [s['aspect_ratio'] for s in st_feats], 'Aspect ratio (H/W)',
             'Ratio', bins=30, color=COLOR_GREEN)
    _hist_ax(axes[1], [s['n_columns'] for s in st_feats], 'Number of columns',
             'Count', bins=30, color=COLOR_GREEN)
    _hist_ax(axes[2], [s['n_beams'] for s in st_feats], 'Number of beams',
             'Count', bins=30, color=COLOR_GREEN)
    plt.tight_layout()
    save_sci_figure(fig, os.path.join(out_dir, 'distribution_struct_extra'))
    plt.close(fig)

    # ---- 3. 计算结果分布 ----
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    _hist_ax(axes[0, 0], [r['disp_peak_mm'] for r in resp_feats], 'Peak displacement',
             'Displacement (mm)', bins=30, color=COLOR_RED)
    _hist_ax(axes[0, 1], [r['disp_rms_mm'] for r in resp_feats], 'RMS displacement',
             'Displacement (mm)', bins=30, color=COLOR_RED)
    _hist_ax(axes[0, 2], [r['disp_std_mm'] for r in resp_feats], 'Std displacement',
             'Displacement (mm)', bins=30, color=COLOR_RED)
    _hist_ax(axes[1, 0], [r['drift_ratio_est'] * 100 for r in resp_feats],
             'Drift ratio (est.)', 'Drift ratio (%)', bins=30, color=COLOR_ORANGE)
    _hist_ax(axes[1, 1], [r['applied_pga'] for r in resp_feats], 'Applied PGA',
             'PGA (g)', bins=30, color=COLOR_ORANGE)
    _hist_ax(axes[1, 2], [r['disp_final_mm'] for r in resp_feats], 'Final displacement',
             'Displacement (mm)', bins=30, color=COLOR_ORANGE)
    plt.tight_layout()
    save_sci_figure(fig, os.path.join(out_dir, 'distribution_response'))
    plt.close(fig)

    # ---- 4. 总览 (精选 12 个关键参数) ----
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    all_plots = [
        ('Motion PGA (g)', [m['pga'] for m in mot_feats], COLOR_RED),
        ('Duration (s)', [m['duration'] for m in mot_feats], COLOR_RED),
        ('Dominant freq. (Hz)', [m['f_main'] for m in mot_feats], COLOR_RED),
        ('Stories', [s['num_stories'] for s in st_feats], COLOR_BLUE),
        ('Bays (X)', [s['num_bays_x'] for s in st_feats], COLOR_BLUE),
        ('Span (m)', [s['span_x'] for s in st_feats], COLOR_BLUE),
        ('Story height (m)', [s['story_height'] for s in st_feats], COLOR_BLUE),
        ('Total height (m)', [s['total_height'] for s in st_feats], COLOR_GREEN),
        ('Total mass (t)', [s['total_mass_t'] for s in st_feats], COLOR_GREEN),
        ('Peak disp. (mm)', [r['disp_peak_mm'] for r in resp_feats], COLOR_RED),
        ('Drift ratio (%)', [r['drift_ratio_est'] * 100 for r in resp_feats], COLOR_ORANGE),
        ('Applied PGA (g)', [r['applied_pga'] for r in resp_feats], COLOR_ORANGE),
    ]
    for ax, (label, data, color) in zip(axes.flat, all_plots):
        _hist_ax(ax, data, '', label, bins=30, color=color)
    plt.tight_layout()
    save_sci_figure(fig, os.path.join(out_dir, 'distribution_overview'))
    plt.close(fig)

    print(f"  ✓ 分布图已保存至: {out_dir}")


# ============================================================
# 均匀抽样分布图 (正常训练时的抽样分布)
# ------------------------------------------------------------
# 展示按"多维分层均匀抽样"原则抽取样本时, 各分层维度的分布情况,
# 验证抽样均匀性 (楼层均匀 / 每层内大-小变形均匀 / 结构形态均匀)。
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
    from collections import Counter
    from dataset import _stratified_sample_uniform

    os.makedirs(out_dir, exist_ok=True)
    cfg = Config()
    n_peak = int(getattr(cfg, 'DB_STRATIFY_RESPONSE_BINS', 3) or 0)
    use_struct = bool(getattr(cfg, 'DB_STRATIFY_STRUCT', True))
    use_stories = bool(getattr(cfg, 'DB_STRATIFY_STORIES', True))

    # ---- 抽样 (与训练一致) ----
    rows = db.query_samples()
    # 剔除顶点位移超标的样本 (与训练 dataset._load_from_db 一致)
    rows = _filter_excess_drift(rows)
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
    plt.close(fig)
    print(f"  ✓ 均匀抽样分布图: {out_dir}/sampling_distribution.pdf/.png")


# ============================================================
# 描述性统计 + CSV
# ============================================================
def summary_stats(name, data, unit=''):
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return {'parameter': name, 'n': 0}
    q = np.percentile(data, [5, 25, 50, 75, 95])
    return {
        'parameter': f"{name}{unit}",
        'n': len(data),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'p5': float(q[0]),
        'p25': float(q[1]),
        'median': float(q[2]),
        'p75': float(q[3]),
        'p95': float(q[4]),
        'max': float(np.max(data)),
    }


def build_tables(mot_feats, st_feats, resp_feats, out_dir, num_samples):
    """构建明细 CSV + 汇总 CSV + 直方图 CSV"""
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. 样本明细 CSV (波/结构/结果) ----
    rows = []
    n = min(len(mot_feats), len(st_feats), len(resp_feats))
    for i in range(n):
        m, s, r = mot_feats[i], st_feats[i], resp_feats[i]
        rows.append({
            'sample_id': i,
            **{f'm_{k}': v for k, v in m.items()},
            **{f's_{k}': v for k, v in s.items()},
            **{f'r_{k}': v for k, v in r.items()},
        })
    df_params = pd.DataFrame(rows)
    params_csv = os.path.join(out_dir, 'dataset_params.csv')
    df_params.to_csv(params_csv, index=False, encoding='utf-8-sig')
    print(f"  ✓ 参数明细 CSV: {params_csv} ({len(df_params)} 行)")

    # ---- 2. 汇总统计 CSV ----
    summary = []
    # 地震波
    for k, label, unit in [
        ('pga', 'Ground motion PGA', ' (g)'),
        ('duration', 'Duration', ' (s)'),
        ('n_steps', 'Number of steps', ''),
        ('f_main', 'Dominant frequency', ' (Hz)'),
        ('f_low', 'Energy fraction 0.1-2 Hz', ''),
        ('f_mid', 'Energy fraction 2-8 Hz', ''),
        ('f_high', 'Energy fraction 8-20 Hz', ''),
    ]:
        summary.append(summary_stats(label, [m[k] for m in mot_feats], unit))
    # 结构
    for k, label, unit in [
        ('num_stories', 'Number of stories', ''),
        ('num_bays_x', 'Number of bays (X)', ''),
        ('num_bays_y', 'Number of bays (Y)', ''),
        ('span_x', 'Span (X)', ' (m)'),
        ('span_y', 'Span (Y)', ' (m)'),
        ('story_height', 'Story height', ' (m)'),
        ('total_height', 'Total height', ' (m)'),
        ('total_mass_t', 'Total mass', ' (t)'),
        ('n_columns', 'Number of columns', ''),
        ('n_beams', 'Number of beams', ''),
        ('aspect_ratio', 'Aspect ratio (H/W)', ''),
        ('floor_load_avg', 'Floor load (avg)', ' (kPa)'),
    ]:
        summary.append(summary_stats(label, [s[k] for s in st_feats], unit))
    # 结果
    for k, label, unit in [
        ('disp_peak_mm', 'Peak displacement', ' (mm)'),
        ('disp_std_mm', 'Std displacement', ' (mm)'),
        ('disp_rms_mm', 'RMS displacement', ' (mm)'),
        ('disp_final_mm', 'Final displacement', ' (mm)'),
        ('drift_ratio_est', 'Drift ratio (est.)', ''),
        ('target_pga', 'Target PGA', ' (g)'),
        ('applied_pga', 'Applied PGA', ' (g)'),
    ]:
        summary.append(summary_stats(label, [r[k] for r in resp_feats], unit))

    df_summary = pd.DataFrame(summary)
    # 重排列: parameter, n, mean, std, min, p5, p25, median, p75, p95, max
    col_order = ['parameter', 'n', 'mean', 'std', 'min', 'p5', 'p25',
                 'median', 'p75', 'p95', 'max']
    df_summary = df_summary[col_order]
    sum_csv = os.path.join(out_dir, 'dataset_summary.csv')
    df_summary.to_csv(sum_csv, index=False, encoding='utf-8-sig')
    print(f"  ✓ 汇总统计 CSV: {sum_csv}")

    # ---- 3. 直方图 CSV (用于论文表) ----
    hist_records = []
    for label, data, bins in [
        ('PGA (g)', [m['pga'] for m in mot_feats], 20),
        ('Stories', [s['num_stories'] for s in st_feats], 12),
        ('Total height (m)', [s['total_height'] for s in st_feats], 20),
        ('Peak disp. (mm)', [r['disp_peak_mm'] for r in resp_feats], 20),
        ('Drift ratio (%)', [r['drift_ratio_est'] * 100 for r in resp_feats], 20),
    ]:
        data = np.asarray(data, dtype=np.float64)
        counts, edges = np.histogram(data, bins=bins)
        for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
            hist_records.append({'parameter': label, 'bin_low': lo, 'bin_high': hi,
                                 'count': int(c)})
    df_hist = pd.DataFrame(hist_records)
    hist_csv = os.path.join(out_dir, 'dataset_histogram.csv')
    df_hist.to_csv(hist_csv, index=False, encoding='utf-8-sig')
    print(f"  ✓ 直方图 CSV: {hist_csv}")

    # ---- 打印汇总表 ----
    print("\n" + "=" * 72)
    print("样本集参数描述性统计")
    print("=" * 72)
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    return df_params, df_summary, df_hist


# ============================================================
# 关键参数区段覆盖检查 (总层数 / 总质量 / 自振周期)
# ------------------------------------------------------------
# 目标: 找出样本集中这些参数是否有"区段缺失"——即某个数值区间内没有样本,
#       影响机器学习外推能力 (缺失区段模型无法学习)。
# 输出: 分箱覆盖表 CSV + 缺失区段报告 + 覆盖度图 (含缺失高亮)。
# ============================================================
def coverage_analysis(st_feats, out_dir, bins_map=None):
    """
    对总层数/总质量/自振周期做分箱覆盖检查。

    Args:
        st_feats: struct_features 列表 (含 num_stories/total_mass_t/natural_period)
        out_dir: 图表输出目录
        bins_map: {参数名: 手动分箱边界列表} 可选; 未提供时自动等宽分箱

    Returns:
        dict: {参数名: 覆盖分析结果} 并写 CSV / 图
    """
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    bins_map = bins_map or {}
    default_bins = {
        # 总层数: 每 1 层一档, 覆盖 0~8 层
        'num_stories': np.arange(0, 9, 1),
        # 总质量 (t): 不等宽, 覆盖 0~350t
        'total_mass_t': [0, 20, 40, 60, 80, 100, 120, 150, 200, 300, 400],
        # 自振周期 (s): 每 0.1s 一档, 覆盖 0~3s
        'natural_period': np.arange(0, 3.1, 0.1),
    }
    fields = [
        ('num_stories', 'Number of stories', ''),
        ('total_mass_t', 'Total mass', 't'),
        ('natural_period', 'Natural period', 's'),
    ]

    all_records = []
    results = {}
    for key, label, unit in fields:
        data = np.asarray([s[key] for s in st_feats], dtype=np.float64)
        data = data[np.isfinite(data)]
        n = len(data)
        bins = bins_map.get(key, default_bins[key])
        counts, edges = np.histogram(data, bins=bins)
        # 每个区段: 计数 + 是否覆盖
        records = []
        covered = []
        for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
            records.append({
                'parameter': label,
                'bin_low': lo, 'bin_high': hi,
                'count': int(c),
                'covered': c > 0,
            })
            covered.append(c > 0)
        # 缺失区段: 只统计落在数据 [min,max] 范围内的空箱 (中间空洞, 影响外推);
        # 范围外的空箱 (min 以下 / max 以上) 属正常, 不计入缺失。
        dmin, dmax = float(data.min()), float(data.max())
        missing = []
        for rec in records:
            if rec['covered']:
                continue
            lo, hi = rec['bin_low'], rec['bin_high']
            # 与数据范围有重叠且完全在数据范围内才视为"中间缺失"
            if hi > dmin and lo < dmax:
                missing.append((lo, hi))
        results[key] = {
            'label': label, 'unit': unit, 'n': n,
            'min': dmin, 'max': dmax,
            'total_bins': len(records),
            'covered_bins': int(sum(covered)),
            'missing_bins': len(missing),
            'missing_ranges': missing,
            'records': records,
        }
        all_records.extend(records)
        print(f"\n  [{label}] n={n}, 范围 [{dmin:.3g}, {dmax:.3g}] {unit}, "
              f"分箱 {len(records)}, 覆盖 {sum(covered)}, 中间缺失 {len(missing)}")
        if missing:
            print(f"      数据范围内的缺失区段:")
            for lo, hi in missing:
                print(f"        {lo:.3g} ~ {hi:.3g} {unit}")

    # ---- 写 CSV ----
    df_cov = pd.DataFrame(all_records)
    cov_csv = os.path.join(out_dir, 'coverage_check.csv')
    df_cov.to_csv(cov_csv, index=False, encoding='utf-8-sig')
    print(f"\n  ✓ 覆盖检查 CSV: {cov_csv}")

    # ---- 覆盖度图 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, label, unit) in zip(axes, fields):
        res = results[key]
        recs = res['records']
        lows = [r['bin_low'] for r in recs]
        highs = [r['bin_high'] for r in recs]
        counts = [r['count'] for r in recs]
        x = [(lo + hi) / 2 for lo, hi in zip(lows, highs)]
        width = [(hi - lo) * 0.85 for lo, hi in zip(lows, highs)]
        colors = [COLOR_GREEN if r['covered'] else COLOR_RED for r in recs]
        ax.bar(x, counts, width=width, color=colors, edgecolor='black', alpha=0.85)
        # 数据范围高亮 (用该参数的 min/max, 而非循环外残留变量)
        ax.axvspan(res['min'], res['max'], color='gray', alpha=0.08)
        ax.set_xlabel(f'{label} ({unit})' if unit else label)
        ax.set_ylabel('Count')
        ax.set_title(f'{label} coverage\n'
                     f'({res["covered_bins"]}/{res["total_bins"]} bins, '
                     f'{res["missing_bins"]} gaps)')
        ax.tick_params(axis='both', labelsize=8, direction='in')
        # 图例
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=COLOR_GREEN, label='covered'),
                           Patch(color=COLOR_RED, label='MISSING')],
                  fontsize=8, loc='upper right')
    plt.tight_layout()
    save_sci_figure(fig, os.path.join(out_dir, 'coverage_check'))
    plt.close(fig)
    print(f"  ✓ 覆盖度图: {out_dir}/coverage_check.pdf/.png")

    return results


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='样本集参数分布分析 (PostgreSQL, SCI 配图 + CSV)')
    parser.add_argument('--out', type=str, default='./plots/sci/dataset',
                        help='图表输出目录')
    parser.add_argument('--csv', type=str, default='./data',
                        help='CSV 输出目录')
    parser.add_argument('--num', type=int, default=None,
                        help='随机抽取样本数 (None=全部)')
    parser.add_argument('--formats', type=str, default='pdf,png',
                        help='输出格式 (默认 pdf,png)')
    args = parser.parse_args()

    global SCI_FORMATS
    SCI_FORMATS = [f.strip() for f in args.formats.split(',') if f.strip()]

    print("=" * 60)
    print("样本集参数分布分析 (PostgreSQL)")
    print("=" * 60)

    db = SLFDatabase()
    stats = db.stats()
    n_done = db.count_done_samples()
    n_pending = db.count_pending_samples()
    print(f"  数据库: 波 {stats['ground_motions']}, 结构 {stats['structures']}, "
          f"样本总数 {stats['samples']}")
    print(f"         有计算结果(done): {n_done} | 未完成(pending): {n_pending} "
          f"| 仅统计 done, 不含 pending")

    print("\n加载样本集参数...")
    motion_rows, struct_rows, resp_rows = load_dataset(db, num_samples=args.num)
    if not resp_rows:
        print("  [X] 无数据")
        return
    n = len(resp_rows)
    print(f"  ✓ 分析样本数: {n} (全部为 done 样本)")

    print("\n提取特征...")
    dt = Config.TARGET_DT
    mot_feats = [motion_features(m['motion'], dt) for m in motion_rows]
    st_feats = [struct_features(s) for s in struct_rows]
    resp_feats = []
    for r, s in zip(resp_rows, struct_rows):
        rf = response_features(r, float(s['total_height']))
        if rf:
            resp_feats.append(rf)
    # 过滤无效
    valid = [(m, s, r) for m, s, r in zip(mot_feats, st_feats, resp_feats)
             if m is not None and r is not None]
    mot_feats = [v[0] for v in valid]
    st_feats = [v[1] for v in valid]
    resp_feats = [v[2] for v in valid]
    print(f"  ✓ 有效样本: {len(mot_feats)}")

    print("\n生成分布图 (SCI)...")
    plot_distributions(mot_feats, st_feats, resp_feats, args.out)

    print("\n生成均匀抽样分布图 (正常训练时的抽样分布)...")
    plot_sampling_distribution(db, args.out, num_samples=args.num, seed=42)

    print("\n生成 CSV 表格...")
    build_tables(mot_feats, st_feats, resp_feats, args.csv, len(mot_feats))

    print("\n关键参数区段覆盖检查 (总层数/总质量/自振周期)...")
    print("  [*] 说明: 红色区间 = 数据范围内缺失的区段 (影响外推)")
    cov = coverage_analysis(st_feats, args.out)

    print("\n[OK] 分析完成!")


if __name__ == '__main__':
    main()
