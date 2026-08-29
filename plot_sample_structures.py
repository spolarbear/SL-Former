# -*- coding: utf-8 -*-
"""批量可视化真实结构: 原始杆系模型 + 体素立方体堆砌模型 (淡色, 按体素填色).

每个非空 2m 体素画一个半透明立方体, 颜色按体素位置填色 (淡色版本).
视图缩放: 布满坐标轴 (按结构真实尺寸比例).

每个样本输出:
  - plots/samples/sample_{i:02d}_frame_model.{png,pdf}   原始杆系模型
  - plots/samples/sample_{i:02d}_voxel_stack.{png,pdf}   体素立方体堆砌 (淡色按体素)
  - plots/samples/sample_{i:02d}_combined.{png,pdf}      左右拼接 (左杆系 | 右体素堆砌)

用法:
    python plot_sample_structures.py [--num 10] [--out ./plots/samples] [--seed 0]
    python plot_sample_structures.py --num 2
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import hsv_to_rgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frame_grid_encoder import (encode_frame_grid, COMBO_COL, COMBO_BX, COMBO_BY,
                                GRID_X, GRID_Y, MAX_FLOORS, grid_cells,
                                cell_geometry, frame_dims)
from frame_model import build_frame_model, iter_columns, iter_beams, iter_slabs

# 颜色
C_COL = '#B85C4A'    # 柱 (红棕)
C_BX = '#4A7DB4'     # X 梁 (蓝)
C_BY = '#3a6ea5'     # Y 梁 (深蓝)
C_SLAB = '#A8C8E8'   # 板 (浅蓝)
# 体素 3 类颜色 (梁/柱/节点复合)
V_COL = "#F7F4F1"    # 柱 (柔和红)
V_BEAM = "#E3E8EB"   # 梁 (柔和蓝)
V_NODE = "#EBE5EE"   # 节点复合 (柱+梁, 紫)

SCI_DPI = 300
SCI_FORMATS = ['pdf', 'png']


def _voxel_type_color(code):
    """根据格子编码的构件组合分类: 返回 (颜色, 类别名).

    三类:
      - 含柱 + 梁  -> 节点复合 (紫)
      - 只含柱      -> 柱 (红)
      - 只含梁/其他 -> 梁 (蓝)
    """
    from frame_grid_encoder import decode_cell
    combo = decode_cell(code)['combo']
    has_col = bool(combo & COMBO_COL)
    has_beam = bool(combo & (COMBO_BX | COMBO_BY))
    if has_col and has_beam:
        return V_NODE, 'Node (column+beam)'
    if has_col:
        return V_COL, 'Column'
    return V_BEAM, 'Beam'


def _frame_dims_3d(model):
    return frame_dims(model)


def plot_frame_model_3d(model, out_base, struct_id=None):
    """原始杆系模型 3D 视图 (柱/梁线), 输出 png+pdf."""
    xmax, ymax, zmax = _frame_dims_3d(model)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    for col in iter_columns(model):
        ax.plot([col['x'], col['x']], [col['y'], col['y']],
                [col['z_bottom'], col['z_top']],
                color=C_COL, linewidth=3, alpha=0.95, zorder=3,
                solid_capstyle='round')
    for b in iter_beams(model):
        c = C_BX if b['direction'] == 'x' else C_BY
        ax.plot([b['x1'], b['x2']], [b['y1'], b['y2']], [b['z'], b['z']],
                color=c, linewidth=2, alpha=0.85, zorder=3)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _P3D
    polys = []
    for s in iter_slabs(model):
        polys.append([[s['x0'], s['y0'], s['z']], [s['x1'], s['y0'], s['z']],
                      [s['x1'], s['y1'], s['z']], [s['x0'], s['y1'], s['z']]])
    if polys:
        ax.add_collection3d(_P3D(polys, alpha=0.06, facecolors=C_SLAB,
                                 edgecolors='none'))

    _set_axes_3d(ax, xmax, ymax, zmax)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(f'Frame model (raw){f"  struct #{struct_id}" if struct_id else ""}',
                 fontsize=11)
    ax.view_init(elev=25, azim=-45)
    handles = [Patch(facecolor=C_COL, label='Column'),
               Patch(facecolor=C_BX, label='Beam X'),
               Patch(facecolor=C_BY, label='Beam Y'),
               Patch(facecolor=C_SLAB, alpha=0.3, label='Slab')]
    ax.legend(handles=handles, fontsize=8, loc='upper right')

    for fmt in SCI_FORMATS:
        fig.savefig(f"{out_base}.{fmt}", dpi=SCI_DPI, bbox_inches='tight',
                    facecolor='white', format=fmt)
    plt.close(fig)
    print(f"  [saved] {out_base}.png/.pdf")


def _set_axes_3d(ax, xmax, ymax, zmax, margin=0.02):
    """设置 3D 轴: 范围布满坐标轴 + 等比例缩放."""
    # 加一点边距避免贴边
    pad_x, pad_y, pad_z = xmax * margin, ymax * margin, zmax * margin
    ax.set_xlim(-pad_x, xmax + pad_x)
    ax.set_ylim(-pad_y, ymax + pad_y)
    ax.set_zlim(-pad_z, zmax + pad_z)
    try:
        ax.set_box_aspect((xmax, ymax, zmax))
    except Exception:
        pass


def plot_voxel_stack_solid(model, codes, out_base, struct_id=None,
                           grid_x=GRID_X, grid_y=GRID_Y, max_floors=MAX_FLOORS):
    """体素立方体堆砌: 每个非空 2m 体素一个半透明淡色立方体 (按体素位置填色).

    用 ax.voxels 一次渲染, 网格裁剪到结构占用范围 (布满坐标轴),
    颜色淡 (低饱和高亮度), 布满坐标轴.
    """
    xmax, ymax, zmax = _frame_dims_3d(model)
    dx, dy, dz = grid_cells(model, grid_x, grid_y, max_floors)

    # 裁剪到结构占用范围 (非空格的最大索引), 让体素布满坐标轴
    nz = np.argwhere(codes != 0)
    if len(nz) == 0:
        return
    gx = int(nz[:, 0].max()) + 1
    gy = int(nz[:, 1].max()) + 1
    gz = int(nz[:, 2].max()) + 1
    sub = codes[:gx, :gy, :gz]

    filled = (sub != 0).astype(bool)
    colors = np.empty(filled.shape, dtype=object)
    colors.fill('none')
    n_col = n_beam = n_node = 0
    for (ix, iy, k) in np.argwhere(sub != 0):
        c, cls = _voxel_type_color(sub[ix, iy, k])
        colors[ix, iy, k] = c
        if cls.startswith('Node'):
            n_node += 1
        elif cls.startswith('Column'):
            n_col += 1
        else:
            n_beam += 1

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 坐标: 2m 格的真实尺寸 (裁剪后), 用 meshgrid 3D 坐标 (非立方也可用)
    xc = np.arange(0, gx + 1) * dx
    yc = np.arange(0, gy + 1) * dy
    zc = np.arange(0, gz + 1) * dz
    Xg, Yg, Zg = np.meshgrid(xc, yc, zc, indexing='ij')
    ax.voxels(Xg, Yg, Zg, filled, facecolors=colors,
              edgecolor='gray', linewidth=0.15, alpha=0.75)

    _set_axes_3d(ax, gx * dx, gy * dy, gz * dz)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(f'Voxel stack (3-type color){f"  struct #{struct_id}" if struct_id else ""}\n'
                 f'{len(nz)} occupied 2m voxels: '
                 f'col {n_col}, beam {n_beam}, node {n_node}', fontsize=11)
    ax.view_init(elev=25, azim=-45)
    ax.grid(True, alpha=0.1, linewidth=0.2)

    # 图例: 3 类
    handles = [Patch(facecolor=V_COL, edgecolor='gray', label='Column'),
               Patch(facecolor=V_BEAM, edgecolor='gray', label='Beam'),
               Patch(facecolor=V_NODE, edgecolor='gray',
                     label='Node (column+beam)')]
    ax.legend(handles=handles, fontsize=8, loc='upper right')

    for fmt in SCI_FORMATS:
        fig.savefig(f"{out_base}.{fmt}", dpi=SCI_DPI, bbox_inches='tight',
                    facecolor='white', format=fmt)
    plt.close(fig)
    print(f"  [saved] {out_base}.png/.pdf")


def plot_combined(model, codes, out_base, struct_id=None):
    """左右拼接: 左 = 原始杆系, 右 = 体素立方体堆砌 (淡色)."""
    xmax, ymax, zmax = _frame_dims_3d(model)
    dx, dy, dz = grid_cells(model, codes.shape[0], codes.shape[1], codes.shape[2])

    fig = plt.figure(figsize=(15, 7.5))
    # 左: 杆系模型
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    for col in iter_columns(model):
        ax1.plot([col['x'], col['x']], [col['y'], col['y']],
                 [col['z_bottom'], col['z_top']],
                 color=C_COL, linewidth=3, alpha=0.95, solid_capstyle='round')
    for b in iter_beams(model):
        c = C_BX if b['direction'] == 'x' else C_BY
        ax1.plot([b['x1'], b['x2']], [b['y1'], b['y2']], [b['z'], b['z']],
                 color=c, linewidth=2, alpha=0.85)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _P3D
    polys = []
    for s in iter_slabs(model):
        polys.append([[s['x0'], s['y0'], s['z']], [s['x1'], s['y0'], s['z']],
                      [s['x1'], s['y1'], s['z']], [s['x0'], s['y1'], s['z']]])
    if polys:
        ax1.add_collection3d(_P3D(polys, alpha=0.06, facecolors=C_SLAB,
                                  edgecolors='none'))
    _set_axes_3d(ax1, xmax, ymax, zmax)
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
    ax1.set_title('(a) Frame model (raw)', fontsize=12)
    ax1.view_init(elev=25, azim=-45)
    ax1.legend(handles=[Patch(facecolor=C_COL, label='Column'),
                        Patch(facecolor=C_BX, label='Beam X'),
                        Patch(facecolor=C_BY, label='Beam Y')],
               fontsize=8, loc='upper right')

    # 右: 体素立方体堆砌 (3 类颜色: 梁/柱/节点复合, 裁剪到占用范围布满坐标轴)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    nz2 = np.argwhere(codes != 0)
    if len(nz2) > 0:
        gx2 = int(nz2[:, 0].max()) + 1
        gy2 = int(nz2[:, 1].max()) + 1
        gz2 = int(nz2[:, 2].max()) + 1
        sub = codes[:gx2, :gy2, :gz2]
        filled = (sub != 0).astype(bool)
        colors = np.empty(filled.shape, dtype=object)
        colors.fill('none')
        for (ix, iy, k) in np.argwhere(sub != 0):
            colors[ix, iy, k], _ = _voxel_type_color(sub[ix, iy, k])
        xc = np.arange(0, gx2 + 1) * dx
        yc = np.arange(0, gy2 + 1) * dy
        zc = np.arange(0, gz2 + 1) * dz
        Xg, Yg, Zg = np.meshgrid(xc, yc, zc, indexing='ij')
        ax2.voxels(Xg, Yg, Zg, filled, facecolors=colors,
                   edgecolor='gray', linewidth=0.15, alpha=0.75)
        _set_axes_3d(ax2, gx2 * dx, gy2 * dy, gz2 * dz)
    else:
        _set_axes_3d(ax2, xmax, ymax, zmax)
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)'); ax2.set_zlabel('Z (m)')
    ax2.set_title('(b) Voxel stack (column / beam / node)', fontsize=12)
    ax2.view_init(elev=25, azim=-45)
    ax2.grid(True, alpha=0.1, linewidth=0.2)
    ax2.legend(handles=[Patch(facecolor=V_COL, edgecolor='gray', label='Column'),
                        Patch(facecolor=V_BEAM, edgecolor='gray', label='Beam'),
                        Patch(facecolor=V_NODE, edgecolor='gray',
                              label='Node (col+beam)')],
               fontsize=8, loc='upper right')

    fig.suptitle(f'Structural sample{f"  #{struct_id}" if struct_id else ""}',
                 fontsize=14)
    for fmt in SCI_FORMATS:
        fig.savefig(f"{out_base}.{fmt}", dpi=SCI_DPI, bbox_inches='tight',
                    facecolor='white', format=fmt)
    plt.close(fig)
    print(f"  [saved] {out_base}.png/.pdf")


def main():
    ap = argparse.ArgumentParser(description='批量可视化真实结构 (杆系+体素堆砌)')
    ap.add_argument('--num', type=int, default=10, help='样本数')
    ap.add_argument('--out', type=str, default='./plots/samples', help='输出目录')
    ap.add_argument('--seed', type=int, default=0, help='随机种子')
    ap.add_argument('--ids', type=str, default=None,
                    help='指定结构 id 列表 (逗号分隔, 覆盖随机)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from db_manager import SLFDatabase, ST_TABLE
    db = SLFDatabase()
    if args.ids:
        ids = [int(x.strip()) for x in args.ids.split(',')]
    else:
        db.cur.execute(
            f"SELECT struct_id FROM {ST_TABLE} WHERE num_stories>=3 ORDER BY random() LIMIT %s",
            (args.num,))
        rows = db.cur.fetchall()
        ids = [r['struct_id'] for r in rows]
    print(f"处理 {len(ids)} 个结构: {ids}")

    for i, sid in enumerate(ids):
        struct = db.get_structure(sid)
        if struct is None:
            print(f"  !! struct #{sid} 不存在, 跳过")
            continue
        model = build_frame_model(struct=struct)
        codes, _ = encode_frame_grid(model)
        base = os.path.join(args.out, f"sample_{i:02d}")
        print(f"--- [{i+1}/{len(ids)}] struct #{sid} ---")
        plot_frame_model_3d(model, f"{base}_frame_model", struct_id=sid)
        plot_voxel_stack_solid(model, codes, f"{base}_voxel_stack", struct_id=sid)
        plot_combined(model, codes, f"{base}_combined", struct_id=sid)

    db.close()
    print("done")


if __name__ == '__main__':
    main()
