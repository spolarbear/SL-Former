# frame_model.py
"""
统一杆系结构模型 (Single Source of Truth)

目标: 把"参数化生成杆系结构计算模型"的逻辑集中到一套代码里。
     - 节点坐标 (与 OpenSees run_analysis 完全一致的网格规则)
     - 杆单元 (柱/梁: 两端节点、截面、方向)
     - 板 (楼板)
     - 逐层截面 (柱 col_sections / 梁 beam_sections)

任何地方要渲染/计算杆系模型, 都应从 generate_fixed_frame 得到的 frame
构建 frame_model, 再统一遍历节点/单元/截面。不要各自重新解码。

节点坐标规则 (与 earthquake_simulator_3d.OpenSeesSimulator3D 一致):
    x = ix * span_x,  y = iy * span_y,  z = floor * story_height
    ix = 0..num_bays_x, iy = 0..num_bays_y, floor = 0..num_stories
柱单元: 楼层 floor 1..num_stories, 节点 (floor-1, ix, iy)->(floor, ix, iy)
X梁   : 楼层 floor 1..num_stories, 节点 (floor, ix, iy)->(floor, ix+1, iy)
Y梁   : 楼层 floor 1..num_stories, 节点 (floor, ix, iy)->(floor, ix, iy+1)
"""
import numpy as np
from generate_frames import generate_fixed_frame


# ============================================================
# 从结构字段 / params 重建 frame (与各调用方统一入口)
# ============================================================
def rebuild_frame_from_struct(struct):
    """从数据库 structures 字段重建框架 (用真实截面/板厚).

    与 visualize_comprehensive.rebuild_frame_from_struct 逻辑一致,
    但统一放这里作为唯一实现。
    """
    num_stories = int(struct['num_stories'])
    num_bays_x = int(struct['num_bays_x'])
    num_bays_y = int(struct['num_bays_y'])
    span_x = float(struct['span_x'])
    span_y = float(struct['span_y'])
    story_height = float(struct['story_height'])
    beam_width = float(struct.get('beam_width', 0.3))
    beam_height = float(struct.get('beam_height', 0.6))
    slab_thickness = float(struct.get('slab_thickness', 0.2))
    plane_shape = str(struct.get('plane_shape') or 'rect').lower()

    col_sections = struct.get('col_sections') or []
    if not col_sections:
        col_sections = [float(beam_width)] * num_stories
    beam_sections = struct.get('beam_sections') or None
    if isinstance(beam_sections, list) and len(beam_sections) == num_stories:
        beam_sections = [tuple(float(x) for x in b) if not isinstance(b, tuple) else b
                         for b in beam_sections]
    else:
        beam_sections = None

    frame = generate_fixed_frame(
        num_stories=num_stories,
        num_spans_x=num_bays_x,
        num_spans_y=num_bays_y,
        span_x=span_x,
        span_y=span_y,
        story_height=story_height,
        axis_ratio=0.6,
        beam_width=beam_width,
        beam_height=beam_height,
        col_sections=[float(c) for c in col_sections] if col_sections else None,
        beam_sections=beam_sections,
        plane_shape=plane_shape,
    )
    frame['slab_thickness'] = slab_thickness
    frame['col_sections'] = [float(c) for c in col_sections]
    return frame


def rebuild_frame(params):
    """从 8/9 维 params 重建框架 (无真实截面时用估算).

    params: [ns, nx, ny, sx, sy, sh, mass, damping] 或
            [ns, nx, ny, sx, sy, sh, mass, damping, shape] (9维含形状)
    """
    num_stories = int(params[0])
    num_bays_x = int(params[1])
    num_bays_y = int(params[2])
    bay_width_x = float(params[3])
    bay_width_y = float(params[4])
    story_height = float(params[5])
    # 第 8 位为形状 ID (可选, 兼容旧 8 维参数)
    from generate_frames import id_to_shape
    plane_shape = id_to_shape(params[8]) if len(params) > 8 else 'rect'
    max_span = max(bay_width_x, bay_width_y)
    beam_height = max(0.4, min(max_span / 12, 0.8))
    beam_height = round(beam_height / 0.2) * 0.2
    beam_width = max(0.2, min(beam_height / 2.5, 0.5))
    beam_width = round(beam_width / 0.2) * 0.2
    return generate_fixed_frame(
        num_stories=num_stories, num_spans_x=num_bays_x,
        num_spans_y=num_bays_y, span_x=bay_width_x, span_y=bay_width_y,
        story_height=story_height, axis_ratio=0.6,
        beam_width=beam_width, beam_height=beam_height,
        plane_shape=plane_shape,
    )


# ============================================================
# 节点生成 (与 OpenSees 完全一致)
# ============================================================
def build_nodes(frame):
    """生成节点 dict: {(floor, ix, iy): (x, y, z)}.

    节点从 frame['columns'] 的 (x, y) 平面坐标反推 (ix, iy),
    只生成形状内存在的节点 (T/L/C 形状不含开口区节点)。
    x = ix*span_x, y = iy*span_y, z = floor*story_height
    """
    ns = int(frame['num_stories'])
    sx = float(frame['span_x'])
    sy = float(frame['span_y'])
    sh = float(frame['story_height'])

    # 从柱列表收集该结构用到的全部平面节点 (x,y)
    plane = set()
    for (x, y, zb, zt, cs) in frame.get('columns', []):
        plane.add((float(x), float(y)))
    if not plane:
        # 兼容: 无柱列表时退回完整矩形 (旧数据)
        nx = int(frame['num_spans_x']); ny = int(frame['num_spans_y'])
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                plane.add((ix * sx, iy * sy))

    # (x,y) -> (ix,iy) 映射 (坐标应为 span 整数倍)
    nodes = {}
    for floor in range(ns + 1):
        for (x, y) in sorted(plane):
            ix = int(round(x / sx)) if sx > 0 else 0
            iy = int(round(y / sy)) if sy > 0 else 0
            nodes[(floor, ix, iy)] = (x, y, floor * sh)
    return nodes


# ============================================================
# 杆单元生成 (与 OpenSees element 完全一致)
# ============================================================
def build_elements(frame, nodes=None):
    """生成单元 dict: {elem_id: {...}}.

    单元类型:
      - column: 垂直杆 (连接同平面上下节点), 截面=col_sections[floor-1]
      - beam_x : 沿 X 方向梁, 截面=beam_sections[floor-1]
      - beam_y : 沿 Y 方向梁, 截面=beam_sections[floor-1]

    每个单元含: type, floor, (i1, j1)-(i2, j2) 节点, 两端坐标,
               section (宽度/高度或边长), direction
    """
    nodes = nodes or build_nodes(frame)
    ns = int(frame['num_stories'])
    col_sections = frame.get('col_sections') or [0.5] * ns
    beam_sections = frame.get('beam_sections') or [(0.3, 0.6)] * ns
    sx = float(frame['span_x'])
    sy = float(frame['span_y'])
    sh = float(frame['story_height'])

    # (x, y) -> (ix, iy) 映射辅助
    def _ij(x, y):
        return (int(round(x / sx)) if sx > 0 else 0,
                int(round(y / sy)) if sy > 0 else 0)

    elems = []
    eid = 0

    # 柱: 从 frame['columns'] 列表 (形状内, 每柱上下两节点)
    col_sections = list(col_sections) + [col_sections[-1]] * (ns - len(col_sections))
    for (x, y, zb, zt, cs) in frame.get('columns', []):
        # 找所在楼层: zb = floor*sh
        floor = int(round(zb / sh)) + 1 if sh > 0 else 1
        floor = max(1, min(floor, ns))
        ix, iy = _ij(x, y)
        n1 = nodes[(floor - 1, ix, iy)]
        n2 = nodes[(floor, ix, iy)]
        cs_eff = float(cs or col_sections[floor - 1])
        elems.append({
            'id': eid, 'type': 'column', 'floor': floor,
            'node': ((floor - 1, ix, iy), (floor, ix, iy)),
            'coord': (n1, n2),
            'section': (cs_eff, cs_eff),   # 方形柱 (w, h)
            'direction': 'z',
        })
        eid += 1

    # 梁: 从 frame['beams'] 列表 (形状内, 含方向)
    # 注意: generate_fixed_frame 的 beams 每格 4 条边不去重 (相邻格子共享边重复),
    # 这里按 (floor, 两端节点, 方向) 去重, 得到形状实际唯一的梁。
    seen = set()
    for (x1, x2, y1, y2, z, w, h, direction) in frame.get('beams', []):
        z_eff = float(z)
        floor = int(round((z_eff - 0.05) / sh)) if sh > 0 else 1
        floor = max(1, min(floor, ns))
        i1, j1 = _ij(x1, y1)
        i2, j2 = _ij(x2, y2)
        # 归一化: 无向边 (保证两端顺序一致)
        key = (floor, min(i1, i2), min(j1, j2), max(i1, i2), max(j1, j2), direction)
        if key in seen:
            continue
        seen.add(key)
        n1 = nodes[(floor, i1, j1)]
        n2 = nodes[(floor, i2, j2)]
        etype = 'beam_x' if direction == 'x' else 'beam_y'
        elems.append({
            'id': eid, 'type': etype, 'floor': floor,
            'node': ((floor, i1, j1), (floor, i2, j2)),
            'coord': (n1, n2),
            'section': (float(w), float(h)),
            'direction': direction,
        })
        eid += 1
    return elems


# ============================================================
# 板
# ============================================================
def build_slabs(frame):
    """生成板 dict: {slab_id: {floor, x0,x1,y0,y1,z_center,thickness}}.

    板从 frame['slabs'] 列表 (形状内每格子一块, T/L/C 贴合形状)。
    """
    ns = int(frame['num_stories'])
    sh = float(frame['story_height'])
    th = float(frame.get('slab_thickness', 0.2))
    slabs = []
    frame_slabs = frame.get('slabs', [])
    if not frame_slabs:
        # 兼容旧数据: 整层一块矩形
        nx = int(frame['num_spans_x']); ny = int(frame['num_spans_y'])
        sx = float(frame['span_x']); sy = float(frame['span_y'])
        for floor in range(1, ns + 1):
            z_bottom = floor * sh + 0.05
            z_top = z_bottom + th
            slabs.append({
                'floor': floor,
                'x0': 0.0, 'x1': nx * sx,
                'y0': 0.0, 'y1': ny * sy,
                'z': (z_bottom + z_top) / 2.0,
                'thickness': th,
            })
        return slabs
    # 按楼层分组: 每 (x0,x1,y0,y1) 对应一块板
    by_floor = {}
    for (x0, x1, y0, y1, zc, th_slab) in frame_slabs:
        z_bottom = float(zc) - th / 2.0
        floor = int(round((z_bottom - 0.05) / sh)) if sh > 0 else 1
        floor = max(1, min(floor, ns))
        by_floor.setdefault(floor, []).append((float(x0), float(x1), float(y0), float(y1), float(zc)))
    for floor in range(1, ns + 1):
        for (x0, x1, y0, y1, zc) in by_floor.get(floor, []):
            slabs.append({
                'floor': floor,
                'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1,
                'z': zc, 'thickness': th,
            })
    return slabs


# ============================================================
# 统一模型构建
# ============================================================
def build_frame_model(frame=None, struct=None, params=None):
    """构建统一杆系模型.

    Args:
        frame : generate_fixed_frame 返回的 frame (优先)
        struct: 数据库结构字段 (用真实截面)
        params: 8维参数 (无真实截面时)

    Returns:
        model dict: {
            'frame': frame, 'nodes': nodes, 'elements': elems,
            'slabs': slabs, 'floor_masses': [每层总质量 kg] (struct 提供时),
            'floor_loads': [每层荷载 kPa] (struct 提供时), 'meta': {...}
        }
    """
    if frame is None:
        if struct is not None:
            frame = rebuild_frame_from_struct(struct)
        elif params is not None:
            frame = rebuild_frame(params)
        else:
            raise ValueError("需提供 frame/struct/params 之一")
    nodes = build_nodes(frame)
    elems = build_elements(frame, nodes)
    slabs = build_slabs(frame)
    # 楼层荷载/质量 (struct 提供时, 供编码器编码楼层荷载通道)
    floor_masses = None
    floor_loads = None
    if struct is not None:
        if isinstance(struct.get('floor_masses'), (list, tuple)):
            floor_masses = [float(x) for x in struct['floor_masses']]
        if isinstance(struct.get('floor_loads'), (list, tuple)):
            floor_loads = [float(x) for x in struct['floor_loads']]
    return {
        'frame': frame,
        'nodes': nodes,
        'elements': elems,
        'slabs': slabs,
        'floor_masses': floor_masses,
        'floor_loads': floor_loads,
    }


# ============================================================
# 统一遍历接口 (供渲染/计算复用)
# ============================================================
def iter_columns(model):
    """迭代柱单元: yield dict {x,y,z_bottom,z_top,section}."""
    for e in model['elements']:
        if e['type'] == 'column':
            (x1, y1, z1), (x2, y2, z2) = e['coord']
            w, h = e['section']
            yield {'x': x1, 'y': y1, 'z_bottom': min(z1, z2),
                   'z_top': max(z1, z2), 'section': w}


def iter_beams(model):
    """迭代梁单元: yield dict {x1,x2,y1,y2,z,width,height,direction}."""
    for e in model['elements']:
        if e['type'] in ('beam_x', 'beam_y'):
            (x1, y1, z1), (x2, y2, z2) = e['coord']
            w, h = e['section']
            yield {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'z': z1,
                   'width': w, 'height': h,
                   'direction': 'x' if e['type'] == 'beam_x' else 'y'}


def iter_slabs(model):
    """迭代板: yield dict {x0,x1,y0,y1,z,thickness}."""
    for s in model['slabs']:
        yield {'x0': s['x0'], 'x1': s['x1'], 'y0': s['y0'], 'y1': s['y1'],
               'z': s['z'], 'thickness': s['thickness']}


if __name__ == '__main__':
    # 自测: 从 params 重建并验证节点/单元数 (rect 5层 2×2 跨)
    m = build_frame_model(params=[5, 2, 2, 6.0, 6.0, 3.5, 35000.0, 0.05])
    ns = int(m['frame']['num_stories'])
    nx = int(m['frame']['num_spans_x'])
    ny = int(m['frame']['num_spans_y'])
    n_nodes = len(m['nodes'])
    n_col = sum(1 for e in m['elements'] if e['type'] == 'column')
    n_bx = sum(1 for e in m['elements'] if e['type'] == 'beam_x')
    n_by = sum(1 for e in m['elements'] if e['type'] == 'beam_y')
    # rect: 节点 (ns+1)*(nx+1)*(ny+1), 柱 ns*(nx+1)*(ny+1)
    # 梁去重后: X = ns*(ny+1)*nx, Y = ns*ny*(nx+1)
    # 板: 形状内格子每层一块 -> ns * (nx*ny)
    print(f"nodes={n_nodes} (expect {(ns+1)*(nx+1)*(ny+1)})")
    print(f"columns={n_col} (expect {ns*(nx+1)*(ny+1)})")
    print(f"beam_x={n_bx} beam_y={n_by} (expect {ns*(ny+1)*nx} {ns*ny*(nx+1)})")
    print(f"slabs={len(m['slabs'])} (expect {ns*nx*ny})")

    # 形状自测: T/L/C 形状
    from generate_frames import plane_mask, shape_cell_count
    for shape in ['T', 'L', 'C', 'U']:
        m2 = build_frame_model(frame=__import__('generate_frames').generate_fixed_frame(
            num_stories=4, num_spans_x=3, num_spans_y=3,
            span_x=6.0, span_y=6.0, story_height=3.5, plane_shape=shape))
        n_cells, n_nodes = shape_cell_count(shape, 3, 3)
        n_col2 = sum(1 for e in m2['elements'] if e['type'] == 'column')
        n_bx2 = sum(1 for e in m2['elements'] if e['type'] == 'beam_x')
        n_by2 = sum(1 for e in m2['elements'] if e['type'] == 'beam_y')
        n_slab2 = len(m2['slabs'])
        print(f"[{shape}] cells={n_cells} nodes/floor={n_nodes} -> "
              f"columns={n_col2}(expect {n_nodes*4}) beams_x={n_bx2} beams_y={n_by2} "
              f"slabs={n_slab2}(expect {4*n_cells})")
