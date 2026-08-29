# visualize_frame_cell.py
"""
Voxel-grid cell visualization: frame model + voxel-sliced members + coded voxels.

Core idea: cut the 3D frame model with a fixed 64x64x64 grid (1 m/cell, 64 m space,
origin-aligned). Three side-by-side views:
  (a) Original frame model (member line model)
  (b) Voxel-sliced members (real cross-sections as solid boxes, direct grid clipping)
  (c) Coded voxels (white boxes, 128-bit/cell encoding)

Usage:
    python visualize_frame_cell.py                       # synthetic example (triptych)
    python visualize_frame_cell.py --struct 3350        # real structure (triptych)
    python visualize_frame_cell.py --struct 42751       # real structure (triptych)
    python visualize_frame_cell.py --struct 3351 --cell 3,5 --floor 1
    python visualize_frame_cell.py --struct 3351 --auto  # auto pick typical cells
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

from frame_grid_encoder import (encode_frame_grid, decode_cell,
                                COMBO_COL, COMBO_BX, COMBO_BY,
                                COMBO_NAMES, GRID_X, GRID_Y, MAX_FLOORS,
                                grid_cells, cell_geometry, frame_dims)
from frame_model import build_frame_model, iter_columns, iter_beams, iter_slabs

# ============================================================
# SCI figure style (English, publication-ready)
# ============================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# ============================================================
# Colors
# ============================================================
C_COL = '#B85C4A'    # column (red-brown)
C_BX = '#4A7DB4'     # X beam (blue)
C_BY = '#3a6ea5'     # Y beam (dark blue)
C_NODE = '#E8B54A'   # node (gold)
C_GRID = '#888888'   # grid wireframe

# voxel-type colors (middle/right views)
C_TYPE_NODE = '#C44E4E'   # node cell (column+beam) — red
C_TYPE_COL  = '#4A7DB4'   # pure column cell — blue
C_TYPE_BEAM = '#6BAE6B'   # pure beam cell — green
C_TYPE_EMPTY = '#F0F0F0'  # empty cell (white box in right view)

# darker edge colors for solid boxes (SCI: visible member outlines)
C_TYPE_NODE_EDGE = '#8E2E2E'
C_TYPE_COL_EDGE  = '#2E5580'
C_TYPE_BEAM_EDGE = '#3F7A3F'


def _voxel_type_code(code):
    """Return (color, type name) by voxel type: node(col+beam)/column/beam/empty."""
    from frame_grid_encoder import decode_cell, COMBO_COL, COMBO_BX, COMBO_BY
    if code == 0:
        return C_TYPE_EMPTY, 'Empty'
    combo = decode_cell(code)['combo']
    has_col = bool(combo & COMBO_COL)
    has_beam = bool(combo & (COMBO_BX | COMBO_BY))
    if has_col and has_beam:
        return C_TYPE_NODE, 'Node (column+beam)'
    if has_col:
        return C_TYPE_COL, 'Column'
    return C_TYPE_BEAM, 'Beam'


def _voxel_color(ix, iy, k):
    """Unique color per voxel position (HSL, golden-angle mixing).

    Adjacent voxels differ strongly in hue (golden angle ~137.5 deg);
    lightness/saturation tuned by position to avoid color clusters.
    """
    h = ((ix * 137.508 + iy * 89.236 + k * 47.312) % 360) / 360.0
    s = 0.62 + 0.22 * ((ix + iy + k) % 3) / 2.0
    l = 0.42 + 0.20 * ((ix * 2 + iy * 3 + k * 5) % 3) / 2.0
    r, g, b = hsv_to_rgb((h, s, l))
    return (r, g, b)


def _voxel_code_color(t, max_t=1.0):
    """Map a 0..1 rank (t) to a white->blue gradient color.

    Larger t (larger code value) -> deeper blue; smaller -> whiter.
    Returns an RGBA-ish hex color (opaque).
    """
    import matplotlib.colors as mcolors
    v = float(np.clip(t / max_t if max_t > 0 else 0.0, 0.0, 1.0))
    # white -> blue: interpolate in RGB (white 1,1,1 -> blue 0,0,1)
    r = 1.0 - 0.9 * v
    g = 1.0 - 0.9 * v
    b = 1.0
    return mcolors.to_hex((r, g, b))


def _cell_geo(model, ix, iy, k, grid_x=GRID_X, grid_y=GRID_Y,
              max_floors=MAX_FLOORS):
    """Return cell geometry: (x0,x1,y0,y1,z0,z1, dx, dy, dz)."""
    dx, dy, dz = grid_cells(model, grid_x, grid_y, max_floors)
    x0, x1, y0, y1, z0, z1 = cell_geometry(dx, dy, dz, ix, iy, k)
    return x0, x1, y0, y1, z0, z1, dx, dy, dz


def _draw_transparent_box(ax, x0, x1, y0, y1, z0, z1,
                          face_alpha=0.06, face_color=None,
                          edge_color='#888888', edge_lw=0.8):
    """Draw a translucent cube (voxel): very low face alpha, gray edges.

    Faces are almost transparent; gray edges mark the voxel extent so the
    members inside remain clearly visible.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    verts = [
        [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]],  # bottom
        [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],  # top
        [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]],  # -X
        [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],  # +X
        [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],  # -Y
        [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],  # +Y
    ]
    fc = face_color if face_color is not None else '#dddddd'
    poly = Poly3DCollection(verts, facecolor=fc, alpha=face_alpha,
                            edgecolor=edge_color, linewidth=edge_lw,
                            zorder=1)
    ax.add_collection3d(poly)


def _solid_box_verts(x0, x1, y0, y1, z0, z1):
    """Return the 6 face vertex lists of a box (for batched Poly3DCollection)."""
    return [
        [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]],  # bottom
        [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],  # top
        [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]],  # -X
        [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],  # +X
        [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],  # -Y
        [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],  # +Y
    ]


def _draw_solid_boxes_batch(ax, boxes, zorder=3):
    """Draw many solid boxes as ONE Poly3DCollection (correct 3D occlusion).

    matplotlib's 3D renderer does painter's-algorithm sorting per collection;
    separate collections can be mis-ordered. Batching all faces into a single
    Poly3DCollection lets matplotlib sort all faces by camera depth correctly.

    Args:
        boxes: iterable of dicts with x0,x1,y0,y1,z0,z1,color,alpha[,edge]
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.colors as mcolors
    verts = []
    facecolors = []   # RGBA with per-face alpha
    edgecolors = []
    for b in boxes:
        v = _solid_box_verts(b['x0'], b['x1'], b['y0'], b['y1'],
                             b['z0'], b['z1'])
        verts.extend(v)
        alpha = b.get('alpha', 0.8)
        rgba = mcolors.to_rgba(b['color'], alpha)
        edge = b.get('edge') or _darken(b['color'], 0.5)
        facecolors.extend([rgba] * 6)
        edgecolors.extend([edge] * 6)
    poly = Poly3DCollection(verts, facecolors=facecolors, edgecolors=edgecolors,
                            linewidths=0.5, zorder=zorder)
    ax.add_collection3d(poly)


def _draw_solid_box(ax, x0, x1, y0, y1, z0, z1, color, alpha=0.75,
                    edge_color=None, edge_lw=0.5, zorder=3):
    """Draw a solid box (member rectangular cross-section block).

    Used for the "voxel-sliced members" view: columns/beams rendered with their
    real cross-sections as solid boxes. If edge_color is None, a darkened
    version of `color` is used as the edge so member outlines are visible.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    verts = _solid_box_verts(x0, x1, y0, y1, z0, z1)
    if edge_color is None:
        edge_color = _darken(color, factor=0.55)
    poly = Poly3DCollection(verts, facecolor=color, alpha=alpha,
                            edgecolor=edge_color,
                            linewidth=edge_lw, zorder=zorder)
    ax.add_collection3d(poly)


def _darken(hex_color, factor=0.6):
    """Darken a hex color by multiplying RGB channels by `factor`."""
    import matplotlib.colors as mcolors
    try:
        r, g, b = mcolors.to_rgb(hex_color)
        return (r * factor, g * factor, b * factor)
    except Exception:
        return hex_color


def _draw_cell_members(ax, model, codes, ix, iy, k, draw_box=False,
                       grid_x=GRID_X, grid_y=GRID_Y, color_mode='type', **kw):
    """Draw members inside one cell on 3D axis ax (column/beam lines).

    Args:
        draw_box: True draws cell wireframe (light gray translucent).
        color_mode: 'type' color by member type (col red / beam blue);
                    'voxel' color by voxel position (unique per voxel).
        Returns the cell's decode dict (for caller statistics).
    """
    x0, x1, y0, y1, z0, z1, dx, dy, dz = _cell_geo(model, ix, iy, k,
                                                grid_x, grid_y)
    code = codes[ix, iy, k]
    d = decode_cell(code)

    if draw_box:
        corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for a, b in edges:
            ax.plot(*zip(corners[a], corners[b]), color=C_GRID,
                    linewidth=0.6, alpha=0.35, zorder=1)

    if code == 0:
        return d

    # color: by member type (type) or by voxel position (voxel)
    c_col = _voxel_color(ix, iy, k) if color_mode == 'voxel' else C_COL
    c_bx = _voxel_color(ix, iy, k) if color_mode == 'voxel' else C_BX
    c_by = _voxel_color(ix, iy, k) if color_mode == 'voxel' else C_BY
    c_node = _voxel_color(ix, iy, k) if color_mode == 'voxel' else C_NODE

    cx = (ix + 0.5) * dx
    cy = (iy + 0.5) * dy
    px = cx + d['off_x'] * dx / 8.0
    py = cy + d['off_y'] * dy / 8.0

    # ---- column line (vertical, at offset px,py) ----
    if d['combo'] & COMBO_COL:
        ax.plot([px, px], [py, py], [z0, z1], color=c_col,
                linewidth=4, alpha=0.95, zorder=3,
                solid_capstyle='round')

    # ---- X beam line (along X through cell, fixed y=py) ----
    if d['combo'] & COMBO_BX:
        z_beam = (z0 + z1) / 2.0
        ax.plot([x0, x1], [py, py], [z_beam, z_beam], color=c_bx,
                linewidth=3, alpha=0.9, zorder=3, solid_capstyle='round')

    # ---- Y beam line (along Y through cell, fixed x=px) ----
    if d['combo'] & COMBO_BY:
        z_beam = (z0 + z1) / 2.0
        ax.plot([px, px], [y0, y1], [z_beam, z_beam], color=c_by,
                linewidth=3, alpha=0.9, zorder=3, solid_capstyle='round')

    # ---- column-beam node (offset intersection) ----
    if d['combo'] & COMBO_COL and d['combo'] & (COMBO_BX | COMBO_BY):
        z_node = (z0 + z1) / 2.0
        ax.scatter([px], [py], [z_node], color=c_node, s=40, zorder=5,
                   edgecolors='black', linewidth=0.5)

    return d


def plot_voxel_stack(model, codes, out_png, struct_id=None, grid_x=GRID_X,
                     grid_y=GRID_Y, max_floors=MAX_FLOORS):
    """Decoded voxel assembly: translucent cubes (gray edges) + cell members,
    colored by voxel position.

    Grid fits structure dimensions (model fills grid), origin-aligned.
    Each non-empty cell: translucent cube (very low face alpha with voxel
    color, gray edges) + member lines. xyz equal scale.
    """
    xmax, ymax, zmax = frame_dims(model)
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')

    nz = np.argwhere(codes != 0)
    n_empty = codes.size - len(nz)
    n_col = n_bx = n_by = 0
    for (ix, iy, k) in nz:
        x0, x1, y0, y1, z0, z1, dx, dy, dz = _cell_geo(model, ix, iy, k,
                                                        grid_x, grid_y)
        vc = _voxel_color(ix, iy, k)
        # translucent cube: very low alpha with voxel color, gray edges
        _draw_transparent_box(ax, x0, x1, y0, y1, z0, z1,
                              face_alpha=0.07, face_color=vc,
                              edge_color='#777777', edge_lw=0.7)
        # cell members (voxel color)
        d = _draw_cell_members(ax, model, codes, ix, iy, k,
                               draw_box=False, grid_x=grid_x, grid_y=grid_y,
                               color_mode='voxel')
        if d['combo'] & COMBO_COL:
            n_col += 1
        if d['combo'] & COMBO_BX:
            n_bx += 1
        if d['combo'] & COMBO_BY:
            n_by += 1

    # equal-scale axes (by structure real size)
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.set_zlim(0, zmax)
    try:
        ax.set_box_aspect((xmax, ymax, zmax))
    except Exception:
        pass
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(f'Voxel assembly (decoded) — translucent boxes + members, '
                 f'by voxel color'
                 f'{f"  struct #{struct_id}" if struct_id else ""}\n'
                 f'Non-empty {len(nz)} | col {n_col} | bx {n_bx} | by {n_by}',
                 fontsize=11)
    ax.view_init(elev=25, azim=-45)

    # legend: show a few voxel color examples
    samples = [(ix, iy, k) for ix, iy, k in nz[:6]]
    handles = []
    for ix, iy, k in samples:
        c = _voxel_color(ix, iy, k)
        handles.append(Patch(facecolor=c, edgecolor='gray',
                             label=f'cell({ix},{iy},{k})'))
    ax.legend(handles=handles, fontsize=7, loc='upper right',
              title='Voxel colors')

    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out_png}')



def _clip_members_to_cell(model, ix, iy, k, grid_x=GRID_X, grid_y=GRID_Y,
                          max_floors=MAX_FLOORS):
    """Clip the 3D member model with a grid box (cell); return in-cell segments
    with real cross-sections.

    Whatever the grid clips is what you get: iterate real columns/beams, keep
    only segments intersecting this cell, with real cross-section sizes
    (no 128-bit quantization).

    Returns:
        col_boxes: [(x0,x1,y0,y1,z0,z1, size)] column boxes (vertical, size x size)
        beam_boxes: [(x0,x1,y0,y1,z0,z1, w, h, direction)] beam boxes
    """
    x0, x1, y0, y1, z0, z1, dx, dy, dz = _cell_geo(model, ix, iy, k,
                                                    grid_x, grid_y)
    col_boxes = []
    beam_boxes = []

    # ---- columns: vertical members, section size x size, clip z segment ----
    # 同一 (x,y) 位置若有上下两段柱 (截面不同), 合并取最大截面并延伸到该格,
    # 使下端大柱在节点区保持到梁面标高 (视觉连续).
    col_by_pos = {}   # (x,y) -> (section, zb, zt)
    for col in iter_columns(model):
        cx, cy = col['x'], col['y']
        cs = col['section']
        # column plan position inside this cell (incl. boundary)
        if not (x0 - 1e-9 <= cx <= x1 + 1e-9 and y0 - 1e-9 <= cy <= y1 + 1e-9):
            continue
        # clip z: intersection of column z_bottom..z_top with cell z0..z1
        zb = max(col['z_bottom'], z0)
        zt = min(col['z_top'], z1)
        if zt - zb < 1e-6:
            continue
        key = (round(cx, 4), round(cy, 4))
        if key not in col_by_pos:
            col_by_pos[key] = [cs, zb, zt]
        else:
            # 同位置多段柱: 取最大截面, z 范围取并集 (下段延伸到上段)
            prev = col_by_pos[key]
            if cs > prev[0]:
                prev[0] = cs
            prev[1] = min(prev[1], zb)
            prev[2] = max(prev[2], zt)
    for (cx, cy), (cs, zb, zt) in col_by_pos.items():
        col_boxes.append((cx - cs/2, cx + cs/2,
                          cy - cs/2, cy + cs/2,
                          zb, zt, cs))

    # ---- beams: along X or Y, clip axial segment ----
    for b in iter_beams(model):
        w, h = b['width'], b['height']
        z_beam = b['z']
        # beam height range intersects cell z?
        if not (z0 - h/2 - 1e-9 <= z_beam <= z1 + h/2 + 1e-9):
            continue
        if b['direction'] == 'x':
            # X beam: y fixed (y1==y2), inside cell y range
            by = b['y1']
            if not (y0 - w/2 - 1e-9 <= by <= y1 + w/2 + 1e-9):
                continue
            # clip x range
            bx0 = max(b['x1'], x0)
            bx1 = min(b['x2'], x1)
            if bx1 - bx0 < 1e-6:
                continue
            beam_boxes.append((bx0, bx1,
                               by - w/2, by + w/2,
                               z_beam - h/2, z_beam + h/2,
                               w, h, 'x'))
        else:
            # Y beam: x fixed, inside cell x range
            bx = b['x1']
            if not (x0 - w/2 - 1e-9 <= bx <= x1 + w/2 + 1e-9):
                continue
            by0 = max(b['y1'], y0)
            by1 = min(b['y2'], y1)
            if by1 - by0 < 1e-6:
                continue
            beam_boxes.append((bx - w/2, bx + w/2,
                               by0, by1,
                               z_beam - h/2, z_beam + h/2,
                               w, h, 'y'))

    return col_boxes, beam_boxes


def plot_voxel_split_3d(model, codes, out_png, struct_id=None, grid_x=GRID_X,
                        grid_y=GRID_Y, max_floors=MAX_FLOORS):
    """Voxel-sliced members view (middle panel): clip 3D member model with grid boxes.

    Directly clip the real column/beam model with grid boxes; whatever is
    clipped is what you get:
      - each cell keeps the column/beam segments crossing it, drawn as solid
        boxes with real cross-sections
      - no 128-bit encode/decode (no quantization loss)
    Draw columns first (opaque), then beams (translucent).

    Colors by cell type: node (col+beam)=red, column=blue, beam=green.
    """
    from frame_grid_encoder import (decode_cell, COMBO_COL, COMBO_BX, COMBO_BY)
    xmax, ymax, zmax = frame_dims(model)
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')

    nz = np.argwhere(codes != 0)

    # ---- clip real members per cell, collect ALL boxes (col+beam) ----
    all_box_list = []    # dicts for batched draw (ONE collection -> correct depth)
    for (ix, iy, k) in nz:
        code = codes[ix, iy, k]
        if code == 0:
            continue
        d = decode_cell(code)
        combo = d['combo']
        has_col = bool(combo & COMBO_COL)
        has_beam = bool(combo & (COMBO_BX | COMBO_BY))
        if has_col and has_beam:
            cell_color = C_TYPE_NODE
        elif has_col:
            cell_color = C_TYPE_COL
        else:
            cell_color = C_TYPE_BEAM
        c_boxes, b_boxes = _clip_members_to_cell(model, ix, iy, k,
                                                 grid_x, grid_y)
        for (x0, x1, y0, y1, z0, z1, cs) in c_boxes:
            all_box_list.append(dict(x0=x0, x1=x1, y0=y0, y1=y1,
                                     z0=z0, z1=z1, color=cell_color,
                                     alpha=0.95))
        for (x0, x1, y0, y1, z0, z1, w, h, direction) in b_boxes:
            all_box_list.append(dict(x0=x0, x1=x1, y0=y0, y1=y1,
                                     z0=z0, z1=z1, color=cell_color,
                                     alpha=0.95))

    # ---- ONE Poly3DCollection: matplotlib sorts all faces by camera depth ----
    _draw_solid_boxes_batch(ax, all_box_list, zorder=3)

    # equal-scale axes
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.set_zlim(0, zmax)
    try:
        ax.set_box_aspect((xmax, ymax, zmax))
    except Exception:
        pass
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(f'Voxel-sliced members (direct grid clipping)'
                 f'{f"  struct #{struct_id}" if struct_id else ""}\n'
                 f'Node (col+beam) / Column / Beam',
                 fontsize=11)
    ax.view_init(elev=25, azim=-45)

    handles = [Patch(facecolor=C_TYPE_NODE, edgecolor=C_TYPE_NODE_EDGE,
                     label='Node (column+beam)'),
               Patch(facecolor=C_TYPE_COL, edgecolor=C_TYPE_COL_EDGE,
                     label='Column'),
               Patch(facecolor=C_TYPE_BEAM, edgecolor=C_TYPE_BEAM_EDGE,
                     label='Beam')]
    ax.legend(handles=handles, fontsize=8, loc='upper right')

    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out_png}')


def plot_voxel_coded_3d(model, codes, out_png, struct_id=None, grid_x=GRID_X,
                        grid_y=GRID_Y, max_floors=MAX_FLOORS):
    """Coded voxel view (right panel): one solid white box per non-empty cell.

    White boxes with dark-gray edges represent coded units (128-bit/cell).
    z-limit is extended by +2 m so the topmost box is not clipped.
    """
    from frame_grid_encoder import decode_cell, COMBO_COL, COMBO_BX, COMBO_BY
    xmax, ymax, zmax = frame_dims(model)
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')

    nz = np.argwhere(codes != 0)
    n_col = n_bx = n_by = 0
    for (ix, iy, k) in nz:
        x0, x1, y0, y1, z0, z1, dx, dy, dz = _cell_geo(model, ix, iy, k,
                                                        grid_x, grid_y)
        code = codes[ix, iy, k]
        d = decode_cell(code)
        combo = d['combo']
        # white opaque box + dark-gray edge
        _draw_solid_box(ax, x0, x1, y0, y1, z0, z1,
                        '#FFFFFF', alpha=1.0,
                        edge_color='#444444', edge_lw=0.8)
        if combo & COMBO_COL:
            n_col += 1
        if combo & COMBO_BX:
            n_bx += 1
        if combo & COMBO_BY:
            n_by += 1

    # extend z by +2 m so the topmost voxel box is fully visible
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.set_zlim(0, zmax + 2.0)
    try:
        ax.set_box_aspect((xmax, ymax, zmax + 2.0))
    except Exception:
        pass
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(f'Coded voxels — 128 bit/cell (white boxes)'
                 f'{f"  struct #{struct_id}" if struct_id else ""}\n'
                 f'Non-empty {len(nz)} | col {n_col} | bx {n_bx} | by {n_by}',
                 fontsize=11)
    ax.view_init(elev=25, azim=-45)

    handles = [Patch(facecolor='#FFFFFF', edgecolor='#444444',
                     label='Voxel (coded unit)')]
    ax.legend(handles=handles, fontsize=8, loc='upper right')

    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out_png}')



def plot_frame_model_3d(model, out_png, struct_id=None):
    """Original frame model 3D view (member lines, origin-aligned).

    Columns: red-brown thick lines; X-beams: blue; Y-beams: dark blue;
    slabs: translucent (floor loads are already lumped to nodes).
    """
    xmax, ymax, zmax = frame_dims(model)
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')

    # columns
    for col in iter_columns(model):
        ax.plot([col['x'], col['x']],
                [col['y'], col['y']],
                [col['z_bottom'], col['z_top']],
                color=C_COL, linewidth=3, alpha=0.95, zorder=3,
                solid_capstyle='round')
    # beams
    for b in iter_beams(model):
        c = C_BX if b['direction'] == 'x' else C_BY
        ax.plot([b['x1'], b['x2']],
                [b['y1'], b['y2']],
                [b['z'], b['z']],
                color=c, linewidth=2, alpha=0.85, zorder=3)

    # slabs (translucent, schematic)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _P3D
    polys = []
    for s in iter_slabs(model):
        polys.append([[s['x0'], s['y0'], s['z']],
                      [s['x1'], s['y0'], s['z']],
                      [s['x1'], s['y1'], s['z']],
                      [s['x0'], s['y1'], s['z']]])
    if polys:
        ax.add_collection3d(_P3D(polys, alpha=0.06, facecolors='#A8C8E8',
                                 edgecolors='none'))

    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.set_zlim(0, zmax)
    try:
        ax.set_box_aspect((xmax, ymax, zmax))
    except Exception:
        pass
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(f'Original frame model — origin-aligned'
                 f'{f"  struct #{struct_id}" if struct_id else ""}',
                 fontsize=11)
    ax.view_init(elev=25, azim=-45)
    handles = [Patch(facecolor=C_COL, label='Column'),
               Patch(facecolor=C_BX, label='Beam X'),
               Patch(facecolor=C_BY, label='Beam Y'),
               Patch(facecolor='#A8C8E8', alpha=0.3, label='Slab')]
    ax.legend(handles=handles, fontsize=8, loc='upper right')

    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out_png}')


def draw_cell_ax(ax, model, codes, ix, iy, k, title=None):
    """Draw a single cell's members on 3D axis ax (cell wireframe + members)."""
    x0, x1, y0, y1, z0, z1, dx, dy, dz = _cell_geo(model, ix, iy, k)
    code = codes[ix, iy, k]

    # cell wireframe (12 edges) + members
    d = _draw_cell_members(ax, model, codes, ix, iy, k, draw_box=True)

    if code == 0:
        ax.set_title((title or f'Cell ({ix},{iy},k={k})') + '\nEmpty (no member)')
        return

    # title
    info = (f"{d['combo_name']}  col={d['col']:.2f} "
            f"b={d['bw']:.2f}x{d['bh']:.2f}")
    faces = ','.join(d['face_names'])
    title_full = (title or f'Cell ({ix},{iy},k={k})') + f'\n{info}\nfaces: {faces}'
    ax.set_title(title_full, fontsize=9)

    # view range (margin around cell)
    mx = max(dx, dy, z1 - z0)
    ax.set_xlim(x0 - 0.3 * mx, x1 + 0.3 * mx)
    ax.set_ylim(y0 - 0.3 * mx, y1 + 0.3 * mx)
    ax.set_zlim(z0 - 0.3 * mx, z1 + 0.6 * mx)
    # xyz equal scale (voxel is rectangular: dx x dy x floor height)
    try:
        ax.set_box_aspect((dx, dy, z1 - z0))
    except Exception:
        pass


def auto_pick_cells(model, codes, max_cells=9):
    """Auto-pick representative cells covering all 8 combos (empty/col/bx/by/...).

    One representative cell per combo (prefer bottom floors). Fill with
    non-empty cells if fewer than max_cells.
    """
    nz = np.argwhere(codes != 0)
    by_combo = {}
    for ix, iy, k in nz:
        c = codes[ix, iy, k]
        combo = decode_cell(c)['combo']
        by_combo.setdefault(combo, []).append((int(ix), int(iy), int(k)))
    picked = []
    # one per combo (0..7)
    for combo in range(8):
        if combo in by_combo:
            picked.append(by_combo[combo][0])
    # fill with non-empty cells if fewer than max_cells (spread across floors)
    if len(picked) < max_cells:
        seen = set(picked)
        for ix, iy, k in nz:
            if len(picked) >= max_cells:
                break
            cell = (int(ix), int(iy), int(k))
            if cell not in seen:
                seen.add(cell)
                picked.append(cell)
    return picked[:max_cells]


def plot_cells(model, codes, cells, out_png, struct_id=None):
    """Plot selected cells (one 3D subplot per cell)."""
    n = len(cells)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(ncols * 5.0, nrows * 5.2))
    for i, (ix, iy, k) in enumerate(cells):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        title = f'Cell ({ix},{iy},k={k})'
        draw_cell_ax(ax, model, codes, ix, iy, k, title=title)

    # legend
    handles = [Patch(facecolor=C_COL, label='Column'),
               Patch(facecolor=C_BX, label='Beam X'),
               Patch(facecolor=C_BY, label='Beam Y'),
               Patch(facecolor=C_NODE, label='Column-beam node')]
    fig.legend(handles=handles, fontsize=9, loc='lower center',
               ncol=4, framealpha=0.9)

    tag = f' (struct #{struct_id})' if struct_id else ' (synthetic)'
    fig.suptitle(f'Single voxel cells — decoded members (no merging){tag}',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out_png}')


def plot_triptych(model, codes, out_png, struct_id=None, grid_x=GRID_X,
                  grid_y=GRID_Y, max_floors=MAX_FLOORS,
                  show_frame_overlay=False):
    """Three side-by-side views (1x3): (a) frame, (b) sliced members, (c) coded voxels.

    (a) Original frame model (member lines) + node circles
    (b) Voxel-sliced members (solid boxes with real cross-sections, direct grid
        clipping; colors by type: node red / column blue / beam green)
    (c) Coded voxel boxes (128-bit/cell, white->blue gradient by code value,
        semi-transparent; optional member line overlay)
    X axis extended by +2 m on all panels.
    """
    from frame_grid_encoder import (decode_cell, COMBO_COL, COMBO_BX, COMBO_BY,
                                    FACE_NX, FACE_PX, FACE_NY, FACE_PY,
                                    FACE_NZ, FACE_PZ)
    xmax, ymax, zmax = frame_dims(model)
    # X axis display range: max +2 m
    xlim_max = xmax + 2.0
    fig = plt.figure(figsize=(27, 8))

    # ============ Left: original frame model + node circles ============
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    for col in iter_columns(model):
        ax1.plot([col['x'], col['x']], [col['y'], col['y']],
                 [col['z_bottom'], col['z_top']],
                 color=C_COL, linewidth=3, alpha=0.95, zorder=3,
                 solid_capstyle='round')
    for b in iter_beams(model):
        c = C_BX if b['direction'] == 'x' else C_BY
        ax1.plot([b['x1'], b['x2']], [b['y1'], b['y2']], [b['z'], b['z']],
                 color=c, linewidth=2, alpha=0.85, zorder=3)
    # node circles: column top (beam level) positions
    for col in iter_columns(model):
        ax1.scatter([col['x']], [col['y']], [col['z_top']],
                    color='none', edgecolors=C_NODE, s=30, linewidths=1.5,
                    zorder=5)
    ax1.set_xlim(0, xlim_max); ax1.set_ylim(0, ymax); ax1.set_zlim(0, zmax)
    try:
        ax1.set_box_aspect((xlim_max, ymax, zmax))
    except Exception:
        pass
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
    ax1.set_title('(a) Original frame model', fontsize=12)
    ax1.view_init(elev=25, azim=-45)

    # ============ Middle: voxel-sliced members (grid clipping, by-type color) ============
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    nz = np.argwhere(codes != 0)
    all_boxes = []    # ONE batched draw (columns + beams -> correct depth)
    for (ix, iy, k) in nz:
        code = codes[ix, iy, k]
        if code == 0:
            continue
        d = decode_cell(code)
        combo = d['combo']
        has_col = bool(combo & COMBO_COL)
        has_beam = bool(combo & (COMBO_BX | COMBO_BY))
        if has_col and has_beam:
            cell_color = C_TYPE_NODE
        elif has_col:
            cell_color = C_TYPE_COL
        else:
            cell_color = C_TYPE_BEAM
        c_boxes, b_boxes = _clip_members_to_cell(model, ix, iy, k,
                                                 grid_x, grid_y)
        for (x0, x1, y0, y1, z0, z1, cs) in c_boxes:
            all_boxes.append(dict(x0=x0, x1=x1, y0=y0, y1=y1,
                                  z0=z0, z1=z1, color=cell_color, alpha=0.95))
        for (x0, x1, y0, y1, z0, z1, w, h, direction) in b_boxes:
            all_boxes.append(dict(x0=x0, x1=x1, y0=y0, y1=y1,
                                  z0=z0, z1=z1, color=cell_color, alpha=0.95))
    _draw_solid_boxes_batch(ax2, all_boxes, zorder=3)
    ax2.set_xlim(0, xlim_max); ax2.set_ylim(0, ymax); ax2.set_zlim(0, zmax)
    try:
        ax2.set_box_aspect((xlim_max, ymax, zmax))
    except Exception:
        pass
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)'); ax2.set_zlabel('Z (m)')
    ax2.set_title('(b) Voxel-sliced members (clipped)', fontsize=12)
    ax2.view_init(elev=25, azim=-45)

    # ============ Right: coded voxel boxes (white->blue gradient by code value) ============
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    # rank-normalize code values: 200+ distinct codes mapped 0..1 so the
    # white->blue gradient spreads across the whole value range
    nz_codes = sorted(int(codes[ix, iy, k]) for (ix, iy, k) in nz)
    n_codes = len(nz_codes)
    if n_codes:
        rank_of = {code: r / max(1.0, n_codes - 1.0)
                   for r, code in enumerate(nz_codes)}
    else:
        rank_of = {}
    # semi-transparent voxel boxes (single batch for best 3D sorting)
    boxes3 = []
    for (ix, iy, k) in nz:
        x0, x1, y0, y1, z0, z1, dx, dy, dz = _cell_geo(model, ix, iy, k,
                                                        grid_x, grid_y)
        fill = _voxel_code_color(rank_of[int(codes[ix, iy, k])])
        boxes3.append(dict(x0=x0, x1=x1, y0=y0, y1=y1,
                           z0=z0, z1=z1, color=fill, alpha=0.55,
                           edge='#555555'))
    _draw_solid_boxes_batch(ax3, boxes3, zorder=2)
    if show_frame_overlay:
        # overlay the frame model lines, snapped to each voxel's cell center so
        # members run exactly through the middle of the coded boxes
        dxc, dyc, dzc = grid_cells(model, grid_x, grid_y, max_floors)
        for col in iter_columns(model):
            ix = min(max(int(col['x'] // dxc), 0), grid_x - 1)
            iy = min(max(int(col['y'] // dyc), 0), grid_y - 1)
            cx = (ix + 0.5) * dxc
            cy = (iy + 0.5) * dyc
            ax3.plot([cx, cx], [cy, cy],
                     [col['z_bottom'], col['z_top']],
                     color=C_COL, linewidth=2.2, alpha=1.0, zorder=6,
                     solid_capstyle='round')
        for b in iter_beams(model):
            c = C_BX if b['direction'] == 'x' else C_BY
            k = min(max_floors - 1, max(0, int(b['z'] // dzc)))
            cz = (k + 0.5) * dzc
            if b['direction'] == 'x':
                iy = min(max(int(b['y1'] // dyc), 0), grid_y - 1)
                cy = (iy + 0.5) * dyc
                ax3.plot([b['x1'], b['x2']], [cy, cy], [cz, cz],
                         color=c, linewidth=1.6, alpha=1.0, zorder=6)
            else:
                ix = min(max(int(b['x1'] // dxc), 0), grid_x - 1)
                cx = (ix + 0.5) * dxc
                ax3.plot([cx, cx], [b['y1'], b['y2']], [cz, cz],
                         color=c, linewidth=1.6, alpha=1.0, zorder=6)
    # extend z by +2 m so the topmost voxel box is fully visible
    ax3.set_xlim(0, xlim_max); ax3.set_ylim(0, ymax); ax3.set_zlim(0, zmax + 2.0)
    try:
        ax3.set_box_aspect((xlim_max, ymax, zmax + 2.0))
    except Exception:
        pass
    ax3.set_xlabel('X (m)'); ax3.set_ylabel('Y (m)'); ax3.set_zlabel('Z (m)')
    ax3.set_title('(c) Coded voxels (white->blue by value)', fontsize=12)
    ax3.view_init(elev=25, azim=-45)

    # overall title
    tag = f'  struct #{struct_id}' if struct_id else '  (synthetic)'
    fig.suptitle(f'Voxel encoding visualization — three views{tag}', fontsize=14)

    # explicit layout: three 3D subplots equal width, right margin kept
    fig.subplots_adjust(left=0.02, right=0.985, top=0.90, bottom=0.05,
                        wspace=0.05)
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out_png}')


def _draw_sliced_members_ax(ax, model, codes, grid_x=GRID_X, grid_y=GRID_Y,
                            max_floors=MAX_FLOORS, lims=None):
    """Draw the voxel-sliced members (middle view) on an existing 3D axis.

    Used by both plot_triptych (single) and plot_batch_sliced_3d (grid).
    ALL member faces (columns AND beams) are merged into ONE Poly3DCollection
    so matplotlib sorts every face by camera depth — front members occlude
    rear ones correctly (columns do not wrongly cover beams).

    Args:
        lims: optional (xmax, ymax, zmax) to force a shared axis scale
              (used for batch figures so all subplots share one scale).
    """
    from frame_grid_encoder import (decode_cell, COMBO_COL, COMBO_BX, COMBO_BY)
    xmax, ymax, zmax = frame_dims(model)
    if lims is not None:
        xmax, ymax, zmax = lims
    nz = np.argwhere(codes != 0)
    all_boxes = []   # columns + beams in ONE batch (correct occlusion)
    for (ix, iy, k) in nz:
        code = codes[ix, iy, k]
        if code == 0:
            continue
        d = decode_cell(code)
        combo = d['combo']
        has_col = bool(combo & COMBO_COL)
        has_beam = bool(combo & (COMBO_BX | COMBO_BY))
        if has_col and has_beam:
            cell_color = C_TYPE_NODE
        elif has_col:
            cell_color = C_TYPE_COL
        else:
            cell_color = C_TYPE_BEAM
        c_boxes, b_boxes = _clip_members_to_cell(model, ix, iy, k,
                                                 grid_x, grid_y)
        for (x0, x1, y0, y1, z0, z1, cs) in c_boxes:
            all_boxes.append(dict(x0=x0, x1=x1, y0=y0, y1=y1,
                                  z0=z0, z1=z1, color=cell_color, alpha=0.95))
        for (x0, x1, y0, y1, z0, z1, w, h, direction) in b_boxes:
            all_boxes.append(dict(x0=x0, x1=x1, y0=y0, y1=y1,
                                  z0=z0, z1=z1, color=cell_color, alpha=0.95))
    _draw_solid_boxes_batch(ax, all_boxes, zorder=3)
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.set_zlim(0, zmax)
    try:
        ax.set_box_aspect((xmax, ymax, zmax))
    except Exception:
        pass
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.view_init(elev=25, azim=-45)
    return xmax, ymax, zmax


def plot_batch_sliced_3d(struct_ids, out_png, ncols=5, seed=None,
                         grid_x=GRID_X, grid_y=GRID_Y, max_floors=MAX_FLOORS):
    """Batch render voxel-sliced members (middle view) for many structures.

    Randomly pick/use the given structure ids and lay them out in a grid
    (default 5 columns x 2 rows = 10 samples) for a publication figure.

    Args:
        struct_ids: list of structure ids to render (taken in order).
        ncols: number of columns (default 5 -> 10 samples = 2 rows).
        seed: optional random seed (currently ids are given explicitly).
    """
    from db_manager import SLFDatabase
    from frame_grid_encoder import encode_frame_grid, frame_dims as _fd
    n = len(struct_ids)
    if n == 0:
        print('No structure ids given')
        return
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(ncols * 4.5, nrows * 4.5))

    db = SLFDatabase()
    # ---- load all models first, compute ONE shared axis range ----
    models = []
    gxmax = gymax = gzmax = 0.0
    for sid in struct_ids:
        struct = db.get_structure(sid)
        if struct is None:
            models.append(None)
            continue
        model = build_frame_model(struct=struct)
        xm, ym, zm = _fd(model)
        gxmax = max(gxmax, xm)
        gymax = max(gymax, ym)
        gzmax = max(gzmax, zm)
        models.append((sid, model))
    # 统一比例尺: 所有子图用同一坐标范围 (z 加 2m 余量避免顶部被裁)
    shared_lims = (gxmax, gymax, gzmax + 2.0)

    for i, sid in enumerate(struct_ids):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        entry = models[i]
        if entry is None:
            ax.set_title(f'#{sid} (missing)', fontsize=10)
            ax.set_axis_off()
            continue
        _, model = entry
        codes, _ = encode_frame_grid(model)
        _draw_sliced_members_ax(ax, model, codes, grid_x, grid_y, max_floors,
                                lims=shared_lims)
        ax.set_title(f'#{sid}', fontsize=10)
    db.close()

    fig.suptitle('Voxel-sliced members — shared scale (batch of structures)',
                 fontsize=14)
    fig.subplots_adjust(left=0.02, right=0.985, top=0.90, bottom=0.02,
                        wspace=0.05, hspace=0.2)
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out_png} ({n} structures, shared scale {gxmax:.0f}x{gymax:.0f}x{gzmax:.0f})')


def synthetic_model():
    """Build a small synthetic frame model (3 stories, 2x1 bays) for examples."""
    from generate_frames import generate_fixed_frame
    frame = generate_fixed_frame(
        num_stories=3, num_spans_x=2, num_spans_y=1,
        span_x=6.0, span_y=6.0, story_height=3.0,
        beam_width=0.3, beam_height=0.6,
        col_sections=[0.6, 0.4, 0.2],
        beam_sections=[(0.3, 0.6), (0.3, 0.6), (0.3, 0.6)],
    )
    return build_frame_model(frame=frame)


def main():
    ap = argparse.ArgumentParser(
        description='Frame / voxel-sliced / coded-voxel visualization (64 m space)')
    ap.add_argument('--struct', type=int, default=None,
                    help='database structure id (default: synthetic example)')
    ap.add_argument('--cell', type=str, default=None,
                    help='specific cells "ix,iy,k" (separate by ; e.g. 3,5,0;8,2,1)')
    ap.add_argument('--floor', type=int, default=None,
                    help='specific floor (1-based, conflicts with --cell)')
    ap.add_argument('--auto', action='store_true',
                    help='auto-pick typical cells (one per combo)')
    ap.add_argument('--stack', action='store_true',
                    help='output only the voxel assembly view')
    ap.add_argument('--frame', action='store_true',
                    help='output only the original frame view')
    ap.add_argument('--single', action='store_true',
                    help='output only the single-cell view')
    ap.add_argument('--batch', type=int, default=0,
                    help='random batch render N structures (middle sliced view)')
    ap.add_argument('--n_batch', type=int, default=10,
                    help='number of structures in batch (default 10)')
    ap.add_argument('--seed', type=int, default=None,
                    help='random seed for batch selection')
    ap.add_argument('--frame_overlay', action='store_true',
                    help='panel (c): draw member line overlay on coded voxels (default: off)')
    ap.add_argument('--out', type=str, default='./plots/frame_cell_lines.png')
    args = ap.parse_args()

    # SCI style: Arial/sans-serif font family already set at module import

    # ---- batch mode: sample structures by plane type (rect/T/L/C/U), big
    #      enough (stories>3, bays>=4), voxel-sliced middle view ----
    if args.batch > 0:
        from db_manager import SLFDatabase, ST_TABLE
        db = SLFDatabase()
        # 生成器最大跨数=4 (make_structure_params [1,2,3,4]), 故用 >=4 表示"大跨数"
        db.cur.execute(f"""
            SELECT struct_id, plane_shape FROM {ST_TABLE}
            WHERE num_stories > 3 AND num_bays_x >= 4 AND num_bays_y >= 4
            ORDER BY random()
        """)
        rows = db.cur.fetchall()
        db.close()
        if not rows:
            print('No structures satisfy filter '
                  '(stories>3, bays>=4) — try loosening criteria')
            return
        # 按平面形状分组, 每种形状优先抽 1 个, 再循环补充凑够 n_batch
        n_want = max(1, int(args.n_batch))
        by_shape = {}
        for r in rows:
            by_shape.setdefault(r['plane_shape'], []).append(r['struct_id'])
        # 形状顺序: rect/T/L/C/U, 保证图里能看到多种平面类型
        shape_order = [s for s in ('rect', 't', 'l', 'c', 'u')
                       if s in by_shape]
        ids = []
        i = 0
        while len(ids) < n_want and any(by_shape.values()):
            s = shape_order[i % len(shape_order)]
            pool = by_shape[s]
            if pool:
                ids.append(pool.pop(0))
            i += 1
        ids = ids[:n_want]
        out_batch = args.out if args.out != './plots/frame_cell_lines.png' \
            else './plots/frame_voxel_sliced_batch.png'
        # 每种形状可用的数量 (供提示)
        avail = {s: len(v) for s, v in by_shape.items()}
        print(f'Batch: {len(ids)} structures, shapes available: {avail}')
        plot_batch_sliced_3d(ids, out_batch, ncols=5)
        return

    if args.struct:
        from db_manager import SLFDatabase
        db = SLFDatabase()
        struct = db.get_structure(args.struct)
        db.close()
        model = build_frame_model(struct=struct)
        struct_id = args.struct
    else:
        model = synthetic_model()
        struct_id = None

    codes, _ = encode_frame_grid(model)
    tag = f'_{struct_id}' if struct_id else ''

    # decide which views to output
    only = [m for m, flag in [('frame', args.frame), ('single', args.single),
                              ('stack', args.stack)] if flag]
    if only:
        want = set(only)
    elif args.cell or args.floor:
        want = {'single'}      # specific cell -> single-cell view only
    else:
        want = {'triptych'}    # default: three views side by side

    # 0) triptych (default): left frame / middle sliced / right coded
    if 'triptych' in want:
        plot_triptych(model, codes,
                      f'./plots/frame_voxel_triptych{tag}.png',
                      struct_id=struct_id,
                      show_frame_overlay=args.frame_overlay)

    # 1) original frame model
    if 'frame' in want:
        plot_frame_model_3d(model, f'./plots/frame_model{tag}.png',
                            struct_id=struct_id)

    # 2) voxel assembly
    if 'stack' in want:
        plot_voxel_stack(model, codes,
                         f'./plots/frame_voxel_stack{tag}.png',
                         struct_id=struct_id)

    # 3) single-cell view
    if 'single' in want:
        # pick cells
        if args.cell:
            cells = []
            for part in args.cell.split(';'):
                ix, iy, k = (int(v) for v in part.split(','))
                cells.append((ix, iy, k))
        elif args.floor is not None:
            k = args.floor - 1
            nz = np.argwhere(codes[:, :, k] != 0)
            if len(nz) == 0:
                print(f'No members on floor {args.floor}')
                cells = []
            else:
                cells = [(int(v[0]), int(v[1]), k) for v in nz[:6]]
        else:
            cells = auto_pick_cells(model, codes)

        if not cells:
            print('No cells available')
        else:
            print(f'Selected {len(cells)} cells: {cells}')
            for (ix, iy, k) in cells:
                d = decode_cell(codes[ix, iy, k])
                print(f"  ({ix},{iy},k={k}): code={codes[ix,iy,k]} "
                      f"combo={d['combo_name']} faces={d['face_names']}")
            out_single = args.out if args.out != './plots/frame_cell_lines.png' \
                else f'./plots/frame_cell_lines{tag}.png'
            plot_cells(model, codes, cells, out_single, struct_id=struct_id)


if __name__ == '__main__':
    main()
