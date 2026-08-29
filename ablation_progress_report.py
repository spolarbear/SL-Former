# ablation_progress_report.py
"""消融过程版数据整理与图表 (跑完后可反复运行, 自动跳过未完成项)

用法:
    python ablation_progress_report.py [--out ./models_ablation] [--save ./plots/ablation_progress]
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 消融展示顺序 (与 ablation.py 一致) + 分组 + 短标签
ORDER = ['full_token', 'no_token', 'no_sa', 'no_cross_attn', 'no_film',
         'no_conv', 'no_ffn', 'no_cond_params', 'no_bypass', 'no_motion',
         'no_struct_feat', 'no_peak_loss', 'no_high_loss', 'no_shape_loss',
         'enc_direct', 'enc_cont']
SHORT = {
    'full_token': 'full_token\n(基准)', 'no_token': 'no_token',
    'no_sa': 'no_sa', 'no_cross_attn': 'no_cross\n_attn', 'no_film': 'no_film',
    'no_conv': 'no_conv', 'no_ffn': 'no_ffn', 'no_cond_params': 'no_cond\n_params',
    'no_bypass': 'no_bypass', 'no_motion': 'no_motion',
    'no_struct_feat': 'no_struct\n_feat', 'no_peak_loss': 'no_peak\n_loss',
    'no_high_loss': 'no_high\n_loss', 'no_shape_loss': 'no_shape\n_loss',
    'enc_direct': 'enc\ndirect', 'enc_cont': 'enc\ncont',
}
GROUP = {
    'full_token': 'Baseline', 'no_token': 'Encoding',
    'no_sa': 'Architecture', 'no_cross_attn': 'Architecture', 'no_film': 'Architecture',
    'no_conv': 'Architecture', 'no_ffn': 'Architecture', 'no_cond_params': 'Architecture',
    'no_bypass': 'Architecture',
    'no_motion': 'Input Data', 'no_struct_feat': 'Input Data',
    'no_peak_loss': 'Loss', 'no_high_loss': 'Loss', 'no_shape_loss': 'Loss',
    'enc_direct': 'Encoder Scheme', 'enc_cont': 'Encoder Scheme',
}
GROUP_COLOR = {'Baseline': '#C00000', 'Encoding': '#ED7D31', 'Architecture': '#2E75B6',
               'Input Data': '#70AD47', 'Loss': '#7030A0', 'Encoder Scheme': '#A6A6A6'}


def collect(out_root):
    """读取每个消融目录的指标, 返回 DataFrame"""
    rows = []
    for name in ORDER:
        d = os.path.join(out_root, name)
        ep = os.path.join(d, 'eval_metrics.json')
        tp = os.path.join(d, 'training_info.json')
        row = {'name': name, 'short': SHORT.get(name, name),
               'group': GROUP.get(name, ''), 'status': '未运行'}
        if not os.path.isdir(d):
            row['status'] = '未运行'
            rows.append(row)
            continue
        if os.path.exists(ep):
            with open(ep, 'r', encoding='utf-8') as f:
                m = json.load(f)
            row['status'] = '完成'
            row['r2'] = m.get('r2')
            row['mae'] = m.get('mae', m.get('mae_mm'))
            row['rmse'] = m.get('rmse')
            row['peak_r2'] = m.get('peak_r2')
            row['peak_mae'] = m.get('peak_mae')
            row['rel_mae'] = m.get('rel_mae')
            row['peak_rel_mae_pct'] = m.get('peak_rel_mae_pct')
            row['weighted_mm'] = m.get('weighted_mm')
            row['r2_small_all'] = m.get('r2_small_all')
            row['r2_large_all'] = m.get('r2_large_all')
            row['median_peak_mm'] = m.get('median_peak_mm')
        else:
            row['status'] = '运行中' if os.path.exists(os.path.join(d, 'model')) else '未运行'
        # 参数量 / 特征维度 / vocab
        if os.path.exists(tp):
            with open(tp, 'r', encoding='utf-8') as f:
                info = json.load(f)
            row['n_params'] = info.get('n_params')
            row['feat_dim'] = info.get('feat_dim')
            row['vocab_size'] = info.get('vocab_size')
            row['enc_mode'] = info.get('enc_mode', '')
        rows.append(row)
    df = pd.DataFrame(rows)
    # 数值列排序列
    return df


def fmt_row(r):
    def f(v, nd=3):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return '-'
        return f"{v:.{nd}f}"
    return (f"{r['name']:<16}{r['group']:<6}{r['status']:<5}"
            f"{f(r.get('mae')):>10}{f(r.get('r2')):>9}"
            f"{f(r.get('peak_mae')):>10}{f(r.get('peak_r2')):>9}"
            f"{f(r.get('rel_mae')):>10}{f(r.get('peak_rel_mae_pct')):>9}")


def plot_progress(df, save_path):
    """生成 2x2 指标对比图 (MAE / R2 / 峰值MAE / 峰值R2), 按分组着色"""
    done = df[df['status'] == '完成'].copy()
    if done.empty:
        print("  暂无完成项, 跳过绘图")
        return
    metrics = [
        ('mae', 'MAE (mm) ↓', True),
        ('r2', 'R² ↑', False),
        ('peak_mae', '峰值 MAE (mm) ↓', True),
        ('peak_r2', '峰值 R² ↑', False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    fig.suptitle('SLF Ablation Study (in progress) - Completed Runs',
                 fontsize=15, fontweight='bold')
    colors = [GROUP_COLOR[g] for g in done['group']]
    for ax, (col, title, lower_better) in zip(axes.flat, metrics):
        vals = done[col].astype(float).values
        xs = np.arange(len(done))
        ax.bar(xs, vals, color=colors, edgecolor='black', linewidth=0.4, width=0.72)
        ax.set_xticks(xs)
        ax.set_xticklabels(done['short'], fontsize=8)
        ax.set_title(title, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.set_axisbelow(True)
        # 基准线
        base = done[done['name'] == 'full_token']
        if not base.empty:
            bv = float(base[col].iloc[0])
            ax.axhline(bv, color='#C00000', linestyle='--', linewidth=1.0)
        # 值标注
        for x, v in zip(xs, vals):
            ax.text(x, v, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
        if lower_better:
            # lower is better: mark the best bar
            best = vals.min()
            if not base.empty and bv != best:
                ax.text(xs[np.argmin(vals)], best * 0.98, '\u2193', ha='center',
                        va='top', fontsize=9, color='green')
    # 图例 (分组)
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLOR.items()
               if g in set(done['group'])]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [图] 已保存: {save_path}")


def plot_group_bar(df, save_path):
    """按消融组输出平均退化幅度 (相对基准 full_token)"""
    done = df[df['status'] == '完成'].copy()
    base = done[done['name'] == 'full_token']
    if base.empty or len(done) < 2:
        return
    bv_mae = float(base['mae'].iloc[0])
    bv_r2 = float(base['r2'].iloc[0])
    # 相对变化: MAE 相对增长% / R2 相对下降%
    delta = []
    for _, r in done.iterrows():
        if r['name'] == 'full_token':
            continue
        d_mae = (float(r['mae']) - bv_mae) / bv_mae * 100.0 if bv_mae else np.nan
        d_r2 = (bv_r2 - float(r['r2'])) / bv_r2 * 100.0 if bv_r2 else np.nan
        delta.append({'name': r['name'], 'short': r['short'], 'group': r['group'],
                      'd_mae_pct': d_mae, 'd_r2_pct': d_r2})
    ddf = pd.DataFrame(delta)
    if ddf.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.suptitle('Degradation Relative to full_token Baseline (in progress)',
                 fontsize=14, fontweight='bold')
    colors = [GROUP_COLOR[g] for g in ddf['group']]
    xs = np.arange(len(ddf))
    for ax, col, title, bcolor in [
            (axes[0], 'd_mae_pct', 'MAE Relative Change (%)  \u2191 worse', '#C00000'),
            (axes[1], 'd_r2_pct', 'R\u00b2 Relative Drop (%)  \u2191 worse', '#C00000')]:
        vals = ddf[col].astype(float).values
        bars = ax.bar(xs, vals, color=colors, edgecolor='black', linewidth=0.4, width=0.72)
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(ddf['short'], fontsize=8, rotation=20, ha='right')
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.set_axisbelow(True)
        for x, v in zip(xs, vals):
            ax.text(x, v + (0.4 if v >= 0 else -0.8), f'{v:.1f}',
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=7)
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLOR.items()
               if g in set(ddf['group'])]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [图] 已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='./models_ablation')
    parser.add_argument('--save', type=str, default='./plots/ablation_progress')
    args = parser.parse_args()

    df = collect(args.out)
    done = df[df['status'] == '完成']
    n_done = len(done)
    n_run = len(df[df['status'] == '运行中'])
    print("=" * 92)
    print(f"SLF 消融 过程版报告 — 已完成 {n_done}/{len(df)}, 运行中 {n_run}, "
          f"未运行 {len(df) - n_done - n_run}")
    print("=" * 92)
    print(f"{'name':<16}{'group':<6}{'st':<5}{'MAE(mm)':>10}{'R²':>9}"
          f"{'峰值MAE':>10}{'峰值R²':>9}{'相对MAE':>10}{'峰值相对%':>9}")
    print("-" * 92)
    for _, r in df.iterrows():
        print(fmt_row(r))
    print("-" * 92)

    # 基准参考行
    base = done[done['name'] == 'full_token']
    if not base.empty:
        print(f"\n  [基准] full_token: MAE={float(base['mae'].iloc[0]):.3f} mm, "
              f"R²={float(base['r2'].iloc[0]):.4f}")
        # 相对退化排序 (MAE 增长)
        print("\n  相对基准的退化幅度 (MAE 增长%) :")
        rows = []
        bv = float(base['mae'].iloc[0])
        for _, r in done.iterrows():
            if r['name'] == 'full_token':
                continue
            if r['mae'] is None or np.isnan(float(r['mae'])):
                continue
            rows.append(((float(r['mae']) / bv - 1.0) * 100.0, r['name']))
        for pct, nm in sorted(rows, reverse=True):
            print(f"    {nm:<16} MAE 相对基准 {pct:+7.1f}%")

    # 保存 CSV
    csv_path = os.path.join(args.out, 'ablation_progress_summary.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n  [CSV] 已保存: {csv_path}")

    # 绘图
    save = os.path.splitext(args.save)[0]
    plot_progress(df, save + '.png')
    plot_group_bar(df, save + '_degradation.png')

    print("\n[OK] 过程版报告完成")


if __name__ == '__main__':
    main()
