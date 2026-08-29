# -*- coding: utf-8 -*-
"""绘制 SL-Former v2 (token 模式) 论文结构图.
输出: plots/slformer_token_architecture.{pdf,png} (300dpi, SCI 风格)
布局: 三列
  左列: 结构编码 (token)  中列: 时序编码 + V2Block×4 + 输出  右列: 结构参数条件
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches
import os

# ---------- SCI 样式 ----------
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9
plt.rcParams['mathtext.fontset'] = 'stix'

# 颜色
C_INPUT = '#5B8DB8'
C_STRUCT = '#8E6B9E'
C_TEMP = '#4C9A6E'
C_ATTN = '#D0813C'
C_OUT = '#C0504D'
C_TOKEN = '#6B8E6B'
C_BOX = '#F2F2F2'
C_LINE = '#444444'
C_DIM = '#888888'


def draw_box(ax, x, y, w, h, text, fc=C_BOX, ec=C_LINE, fs=8.5, bold=False,
             lw=1.2, rounded=True, text_color='#222222', dim=None):
    if rounded:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                             linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    else:
        box = Rectangle((x, y), w, h, linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, fontweight='bold' if bold else 'normal',
            color=text_color, zorder=3)
    if dim:
        ax.text(x + w / 2, y + h + 0.02, dim, ha='center', va='bottom',
                fontsize=6.8, color=C_DIM, style='italic', zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, color=C_LINE, lw=1.4, style='-|>',
               ms=11, ls='-', connectionstyle=None):
    if connectionstyle:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                mutation_scale=ms, linewidth=lw, color=color,
                                linestyle=ls, connectionstyle=connectionstyle, zorder=4)
    else:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                mutation_scale=ms, linewidth=lw, color=color,
                                linestyle=ls, zorder=4)
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(16, 14.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 15)
    ax.axis('off')

    # ============================================================
    # 布局坐标
    # ============================================================
    # 左列 (结构编码) x: 0.4 ~ 4.2, 中心 x=2.3
    LX0, LXW = 0.4, 3.8
    LXC = LX0 + LXW / 2
    # 中列 (时序/Transformer) x: 5.6 ~ 10.4, 中心 x=8.0
    MX0, MXW = 5.6, 4.8
    MXC = MX0 + MXW / 2
    # 右列 (条件) x: 11.6 ~ 15.4, 中心 x=13.5
    RX0, RXW = 11.6, 3.8
    RXC = RX0 + RXW / 2

    # ---------- 左列: token 结构编码 ----------
    draw_box(ax, LX0, 0.3, LXW, 1.0,
             'Frame Micro-Element\nToken Features',
             fc='#E8F0F7', ec=C_INPUT, fs=8.2, bold=True,
             dim='$\\mathbb{Z}^{B \\times 32768}$  (32$^3$ tokens)')
    draw_box(ax, LX0, 1.75, LXW, 0.95,
             'nn.Embedding\n(V=283, $d_e$=32, pad=0)',
             fc='#F0EAEF', ec=C_TOKEN, fs=8.0, bold=True,
             dim='$\\mathbb{R}^{B \\times 32768 \\times 32}$')
    draw_arrow(ax, LXC, 1.3, LXC, 1.75, color=C_STRUCT, lw=1.5)
    draw_box(ax, LX0, 3.15, LXW, 0.95,
             'Masked Mean-Pool\n+ Fill Ratio $r$',
             fc='#F0EAEF', ec=C_TOKEN, fs=8.0, bold=True,
             dim='$\\mathbb{R}^{B \\times 32}$ + $\\mathbb{R}^{B}$')
    draw_arrow(ax, LXC, 2.7, LXC, 3.15, color=C_STRUCT, lw=1.5)
    draw_box(ax, LX0, 4.55, LXW, 0.95,
             'Structure MLP\n(33 → 128 → 64 → 128)',
             fc='#E8E2EE', ec=C_STRUCT, fs=8.0, bold=True,
             dim='$\\mathbb{R}^{B \\times 128}$  ($d_{CNN}$)')
    draw_arrow(ax, LXC, 4.1, LXC, 4.55, color=C_STRUCT, lw=1.5)
    draw_box(ax, LX0, 5.95, LXW, 0.9,
             'struct_proj + RMSNorm\n(128 → 256)',
             fc='#E8E2EE', ec=C_STRUCT, fs=8.0, bold=True,
             dim='$\\mathbf{z}_{struct} \\in \\mathbb{R}^{B \\times 256}$')
    draw_arrow(ax, LXC, 5.5, LXC, 5.95, color=C_STRUCT, lw=1.5)
    draw_box(ax, LX0, 7.3, LXW, 1.0,
             'Learnable Struct Tokens\n+ $\\mathbf{z}_{struct}$ + $\\mathbf{c}_{param}$',
             fc='#E8E2EE', ec=C_STRUCT, fs=7.6, bold=True,
             dim='$\\mathbf{S} \\in \\mathbb{R}^{B \\times 8 \\times 256}$')
    draw_arrow(ax, LXC, 6.85, LXC, 7.3, color=C_STRUCT, lw=1.5)

    # ---------- 中列: 时序 + Transformer + 输出 ----------
    draw_box(ax, MX0, 0.3, MXW, 1.0,
             'Ground Motion\nAcceleration $a(t)$',
             fc='#E8F0F7', ec=C_INPUT, fs=8.2, bold=True,
             dim='$\\mathbb{R}^{B \\times 500 \\times 1}$')
    draw_box(ax, MX0, 1.55, MXW, 0.9,
             'InputNorm + $s \\cdot a(t)$\n+ temporal_proj + PosEnc',
             fc='#E8F3EC', ec=C_TEMP, fs=7.8, bold=True,
             dim='$\\mathbb{R}^{B \\times 500 \\times 256}$')
    draw_arrow(ax, MXC, 1.3, MXC, 1.55, color=C_TEMP, lw=1.5)

    # V2Block × 4 (垂直堆叠)
    n_block = 4
    block_h = 1.2
    block_gap = 0.3
    block_top = 8.3
    block_ys = [block_top - i * (block_h + block_gap) for i in range(n_block)]
    # 校验: 最底 block 底部 = block_top - 3*(1.5) = 8.3-4.5 = 3.8 > 时序顶部 2.45 ✓
    for i in range(n_block):
        y = block_ys[i]
        fc = '#F7F7F7' if i % 2 == 0 else '#EFEFEF'
        text = (f'V2Block #{i+1}\n'
                'Self-Attn (RoPE + QK-Norm)\n'
                'Cross-Attn (Struct Tokens)\n'
                'FiLM + Conv1D + SwiGLU')
        draw_box(ax, MX0, y, MXW, block_h, text,
                 fc=fc, ec=C_TEMP if i == 0 else C_LINE,
                 fs=7.0, bold=(i == 0),
                 dim=f'$\\mathbb{{R}}^{{B \\times 500 \\times 256}}$' if i == 0 else None)
        if i > 0:
            draw_arrow(ax, MXC, block_ys[i-1], MXC, y + block_h, color=C_LINE, lw=1.1)
    # 时序 → 最底 block
    draw_arrow(ax, MXC, 2.45, MXC, block_ys[-1], color=C_TEMP, lw=1.5)

    # 输出
    draw_box(ax, MX0, 9.95, MXW, 0.9,
             'HeadNorm + Output MLP\n(256 → 128 → 64 → 1)',
             fc='#F8E8E8', ec=C_OUT, fs=8.0, bold=True,
             dim='$\\mathbb{R}^{B \\times 500 \\times 1}$')
    draw_arrow(ax, MXC, block_top + block_h, MXC, 9.95, color=C_OUT, lw=1.7)

    # bypass
    draw_box(ax, MX0, 11.2, MXW, 0.85,
             'Bypass Conv1D ($s \\cdot a(t) \\to \\hat{u}$)',
             fc='#E8F3EC', ec=C_TEMP, fs=7.8, bold=True,
             dim='$\\mathbb{R}^{B \\times 500 \\times 1}$')
    draw_arrow(ax, MXC, 10.85, MXC, 11.2, color=C_TEMP, lw=1.1, ls='--')
    # bypass 从地震动输入绕 (画在右侧外)
    draw_arrow(ax, MX0 + MXW, 0.8, RX0 - 0.2, 0.8, color=C_TEMP, lw=1.0, ls='--')
    draw_arrow(ax, RX0 - 0.2, 0.8, RX0 - 0.2, 11.65, color=C_TEMP, lw=1.0, ls='--')
    draw_arrow(ax, RX0 - 0.2, 11.65, MX0 + MXW, 11.65, color=C_TEMP, lw=1.0, ls='--')
    draw_arrow(ax, MX0 + MXW, 11.65, MXC, 11.65, color=C_TEMP, lw=1.0, ls='--')

    # 预测输出
    draw_box(ax, MX0, 12.55, MXW, 0.8,
             'Predicted Roof Displacement $\\hat{u}(t)$',
             fc='#F8E8E8', ec=C_OUT, fs=9.5, bold=True,
             dim='$\\mathbb{R}^{B \\times 500}$')
    draw_arrow(ax, MXC, 12.05, MXC, 12.55, color=C_OUT, lw=1.8)

    # ---------- 右列: 结构参数条件 ----------
    draw_box(ax, RX0, 0.3, RXW, 1.0,
             'Structural\nParameters $\\mathbf{p}$',
             fc='#E8F0F7', ec=C_INPUT, fs=8.2, bold=True,
             dim='$\\mathbb{R}^{B \\times 8}$')
    draw_box(ax, RX0, 1.75, RXW, 1.0,
             'Param Cond Projection\n(scaled + MLP)',
             fc='#FDE9DD', ec=C_ATTN, fs=8.0, bold=True,
             dim='$\\mathbb{R}^{B \\times 256}$')
    draw_arrow(ax, RXC, 1.3, RXC, 1.75, color=C_ATTN, lw=1.5)
    draw_box(ax, RX0, 3.15, RXW, 1.0,
             'Struct Cond Vector\n$\\mathbf{c}_{cond}$ = cond + param',
             fc='#FDE9DD', ec=C_ATTN, fs=7.8, bold=True,
             dim='$\\mathbb{R}^{B \\times 256}$')
    draw_arrow(ax, RXC, 2.75, RXC, 3.15, color=C_ATTN, lw=1.5)

    # ============================================================
    # 跨列连接箭头
    # ============================================================
    # 结构 tokens (左) → V2Block 交叉注意力 (中)
    for i in range(n_block):
        y_mid = block_ys[i] + block_h / 2
        draw_arrow(ax, LX0 + LXW, 7.8, MX0, y_mid, color=C_STRUCT, lw=1.0, ls='--')
    # 条件向量 (右) → V2Block FiLM (中)
    for i in range(n_block):
        y_mid = block_ys[i] + block_h / 2
        draw_arrow(ax, RX0, y_mid, MX0 + MXW, y_mid, color=C_ATTN, lw=1.0, ls=':')

    # ============================================================
    # 图例
    # ============================================================
    legend_items = [
        mpatches.Patch(fc='#E8F0F7', ec=C_INPUT, label='Input'),
        mpatches.Patch(fc='#E8E2EE', ec=C_STRUCT, label='Structure encoding (token)'),
        mpatches.Patch(fc='#E8F3EC', ec=C_TEMP, label='Temporal (ground motion)'),
        mpatches.Patch(fc='#FDE9DD', ec=C_ATTN, label='Structural conditioning (FiLM)'),
        mpatches.Patch(fc='#F8E8E8', ec=C_OUT, label='Output (displacement)'),
        mpatches.Patch(fc='#F0EAEF', ec=C_TOKEN, label='Token embedding'),
    ]
    ax.legend(handles=legend_items, loc='lower center', bbox_to_anchor=(0.5, -0.02),
              fontsize=8.5, frameon=False, ncol=3)

    # ============================================================
    # 标题
    # ============================================================
    ax.text(8.0, 14.5, 'SL-Former v2 with Micro-Element Token Encoding',
            ha='center', fontsize=15, fontweight='bold', color='#222222')
    ax.text(8.0, 14.15, 'Structural Seismic Response Surrogate Model',
            ha='center', fontsize=11, color='#555555')

    # ============================================================
    # 保存
    # ============================================================
    out_dir = 'plots'
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, 'slformer_token_architecture')
    for fmt, dpi in [('pdf', 300), ('png', 300)]:
        fig.savefig(f"{base}.{fmt}", dpi=dpi, bbox_inches='tight',
                    facecolor='white', format=fmt)
    print("saved", base)
    plt.close(fig)


if __name__ == '__main__':
    main()
