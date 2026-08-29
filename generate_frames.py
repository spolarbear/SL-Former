# generate_frames.py (简洁版 - 无debug输出)
"""
参数化框架结构生成器
核心思路：把框架看成 nx × ny 个格子的棋盘
每个格子的四条边都生成梁，不去重（因为每条边都需要）

支持平面形状 (plane_shape):
    'rect' : 完整矩形 (nx×ny)
    'T'    : T 形平面 (顶部横条全宽 + 中间竖条向下)
    'L'    : L 形平面 (左下角 L 形)
    'C'    : C 形平面 (左竖条 + 右竖条 + 底部横条, 中上部开口)
    'U'    : U 形平面 (左竖条 + 右竖条 + 顶部横条, 中下部开口)
形状由格子占位掩码 (occupancy mask) 表示, 柱/梁/板只生成掩码内部分。
"""

import numpy as np
import random
from config import Config


# ============================================================
# 平面形状掩码 (格子级 occupancy, 形状 = 掩码为 True 的格子)
# ============================================================
def plane_mask(plane_shape='rect', num_spans_x=2, num_spans_y=2):
    """返回 (num_spans_y, num_spans_x) 的 bool 掩码, True=有楼板/构件.

    mask[iy, ix]: 第 iy 行 (y 方向), 第 ix 列 (x 方向) 的格子是否有楼板。
    约定: y 方向 iy=0 为底部, iy=ny-1 为顶部 (与坐标 y=iy*span_y 一致)。
    """
    nx = max(1, int(num_spans_x))
    ny = max(1, int(num_spans_y))
    shape = (plane_shape or 'rect').lower()
    mask = np.zeros((ny, nx), dtype=bool)

    if shape == 'rect':
        mask[:] = True
    elif shape == 't':
        # T 形: 顶部一行全宽 (iy=ny-1), 中间一列 (ix=nx//2) 向下贯穿
        mask[ny - 1, :] = True
        mask[:, nx // 2] = True
    elif shape == 'l':
        # L 形: 底部一行全宽 (iy=0), 左列 (ix=0) 向上贯穿
        mask[0, :] = True
        mask[:, 0] = True
    elif shape == 'c':
        # C 形: 底部一行 + 左右两列, 顶部中间开口
        mask[0, :] = True
        mask[:, 0] = True
        mask[:, nx - 1] = True
    elif shape == 'u':
        # U 形: 顶部一行 + 左右两列, 底部中间开口
        mask[ny - 1, :] = True
        mask[:, 0] = True
        mask[:, nx - 1] = True
    else:
        raise ValueError(f"未知 plane_shape: {plane_shape} (可选 rect/T/L/C/U)")
    return mask


def shape_cell_count(plane_shape='rect', num_spans_x=2, num_spans_y=2):
    """形状内格子数 (楼板数) 与 形状内节点数."""
    mask = plane_mask(plane_shape, num_spans_x, num_spans_y)
    n_cells = int(mask.sum())
    # 节点: 每个 mask 格子的 4 角, 相邻共享; 用 mask 格子的角点集合数
    corners = set()
    ny, nx = mask.shape
    for iy in range(ny):
        for ix in range(nx):
            if mask[iy, ix]:
                corners.add((ix, iy))
                corners.add((ix + 1, iy))
                corners.add((ix, iy + 1))
                corners.add((ix + 1, iy + 1))
    return n_cells, len(corners)


# ============================================================
# 平面形状 <-> ID (params 第 8 位用 ID, 便于数值存储)
# ============================================================
SHAPE_NAMES = ['rect', 't', 'l', 'c', 'u']

def shape_to_id(plane_shape):
    """形状名 -> ID (rect=0, T=1, L=2, C=3, U=4)."""
    name = str(plane_shape or 'rect').lower()
    return SHAPE_NAMES.index(name) if name in SHAPE_NAMES else 0

def id_to_shape(shape_id):
    """ID -> 形状名 (0=rect, 1=T, 2=L, 3=C, 4=U; 越界默认 rect)."""
    i = int(shape_id) if shape_id is not None else 0
    return SHAPE_NAMES[i] if 0 <= i < len(SHAPE_NAMES) else 'rect'


def calculate_column_section(num_stories, num_spans_x, num_spans_y, 
                             span_x, span_y, story_height,
                             load_per_area=20.0,
                             axis_ratio_min=0.4,
                             axis_ratio_max=0.8,
                             concrete_fc=14.3):
    """
    根据轴压比估算柱截面尺寸 (正方形柱, 200mm 模数)

    新规则 (2026-08-14):
      - 轴压比范围 0.3~0.9 (由调用方传 axis_ratio_min/max)
      - 截面尺寸 200×200 ~ 1400×1400 mm
      - 底部大顶部小: 按楼层渐变 + 随机扰动
    """
    import random as _random
    tributary_area = span_x * span_y
    total_load = load_per_area * tributary_area * num_stories
    axis_ratio = _random.uniform(axis_ratio_min, axis_ratio_max)
    
    N_kn = total_load
    fc_kpa = concrete_fc * 1000
    
    def _snap200(v):
        """吸附到 200mm 模数并限幅 [0.2, 1.4]"""
        v = max(0.2, min(v, 1.4))
        return round(round(v / 0.2) * 0.2, 2)
    
    Ac_required = N_kn / (fc_kpa * axis_ratio)
    col_size = np.sqrt(Ac_required)
    col_size = _snap200(col_size)
    
    if num_stories == 1:
        return [col_size]
    
    bottom_size = np.sqrt(N_kn / (fc_kpa * axis_ratio_min))
    bottom_size = _snap200(bottom_size)
    
    top_load = load_per_area * tributary_area * 1
    top_size = np.sqrt(top_load / (fc_kpa * axis_ratio_max))
    top_size = _snap200(top_size)
    
    if top_size > bottom_size:
        top_size = max(0.2, bottom_size * 0.6)
    
    col_sections = np.linspace(bottom_size, top_size, num_stories).tolist()
    for i in range(1, len(col_sections)):
        if col_sections[i] > col_sections[i-1]:
            col_sections[i] = col_sections[i-1] * 0.95
    
    # 随机扰动 (每层 ±1 档 200mm, 保持渐变趋势) + 200mm 模数
    col_sections = [_snap200(max(0.2, cs * _random.uniform(0.9, 1.1)))
                    for cs in col_sections]
    # 再次单调化 (保证底部>=顶部)
    for i in range(1, len(col_sections)):
        if col_sections[i] > col_sections[i-1]:
            col_sections[i] = col_sections[i-1]
    
    return col_sections


def generate_fixed_frame(num_stories=5, num_spans_x=2, num_spans_y=2,
                         span_x=6.0, span_y=6.0, story_height=3.5,
                         axis_ratio=0.6,
                         beam_width=0.3, beam_height=0.6,
                         col_sections=None, beam_sections=None,
                         plane_shape='rect'):
    """
    生成固定参数的框架结构 (基于格子生成，不去重)

    Args:
        plane_shape: 平面形状 'rect'/'T'/'L'/'C'/'U'
            柱/梁/板只生成形状掩码内 (非矩形平面) 的部分。

    截面:
        - col_sections: 逐层柱截面列表 [num_stories] (None=内部计算)
        - beam_sections: 逐层梁截面列表 [(w,h), ...] 长度 num_stories (None=单值 beam_width/beam_height)
           同一层所有梁用同一截面, 不同层可不同
    梁数据格式: (x1, x2, y1, y2, z, width, height, direction)
        direction: 'x' 表示X方向梁 (沿X轴延伸)
                   'y' 表示Y方向梁 (沿Y轴延伸)
    板数据格式: (x0, x1, y0, y1, z_center, thickness) — 每形状格子一块
    """
    plane_shape = (plane_shape or 'rect').lower()
    mask = plane_mask(plane_shape, num_spans_x, num_spans_y)  # [ny, nx]

    if col_sections is None:
        col_sections = calculate_column_section(
            num_stories=num_stories,
            num_spans_x=num_spans_x,
            num_spans_y=num_spans_y,
            span_x=span_x,
            span_y=span_y,
            story_height=story_height,
            axis_ratio_min=axis_ratio - 0.1,
            axis_ratio_max=axis_ratio + 0.1
        )

    slab_thickness = 0.2

    # 柱子用set去重
    columns_set = set()
    # 梁不去重，直接用列表
    beams_list = []
    slabs_list = []

    # ============================================================
    # 遍历每一层 (包括屋顶层)
    # 楼层范围: 0 到 num_stories (屋顶)
    # ============================================================
    for floor in range(num_stories + 1):  # +1 包含屋顶层
        # 柱子只在 0 到 num_stories-1 层有 (屋顶没有柱子)
        if floor < num_stories:
            col_size = col_sections[floor]
            z_bottom = floor * story_height
            z_top = (floor + 1) * story_height

            # 柱子: 只生成形状掩码内格子 (i,j) 的 4 个角点
            for iy in range(num_spans_y):
                for ix in range(num_spans_x):
                    if not mask[iy, ix]:
                        continue
                    for (cx, cy) in [(ix, iy), (ix + 1, iy),
                                     (ix, iy + 1), (ix + 1, iy + 1)]:
                        x = cx * span_x
                        y = cy * span_y
                        columns_set.add((x, y, z_bottom, z_top, col_size))

        # ---- 梁: 每层都有 (包括屋顶) ----
        # 地面层没有梁
        if floor == 0:
            continue

        # 该层梁截面 (逐层或单值)
        if beam_sections is not None and len(beam_sections) >= floor:
            bw_f, bh_f = beam_sections[floor - 1]
        else:
            bw_f, bh_f = beam_width, beam_height

        z_beam = floor * story_height + 0.05

        # 遍历形状掩码内所有格子
        for iy in range(num_spans_y):
            for ix in range(num_spans_x):
                if not mask[iy, ix]:
                    continue
                x0 = ix * span_x
                x1 = (ix + 1) * span_x
                y0 = iy * span_y
                y1 = (iy + 1) * span_y

                # 下边: y = y0, x从x0到x1 (X方向梁)
                beams_list.append((x0, x1, y0, y0, z_beam, bw_f, bh_f, 'x'))

                # 右边: x = x1, y从y0到y1 (Y方向梁)
                beams_list.append((x1, x1, y0, y1, z_beam, bw_f, bh_f, 'y'))

                # 上边: y = y1, x从x0到x1 (X方向梁)
                beams_list.append((x0, x1, y1, y1, z_beam, bw_f, bh_f, 'x'))

                # 左边: x = x0, y从y0到y1 (Y方向梁)
                beams_list.append((x0, x0, y0, y1, z_beam, bw_f, bh_f, 'y'))

    # ---- 楼板 (每形状格子一块, 贴合形状) ----
    for floor in range(1, num_stories + 1):
        z_bottom = floor * story_height + 0.05
        z_top = z_bottom + slab_thickness
        z_center = (z_bottom + z_top) / 2
        for iy in range(num_spans_y):
            for ix in range(num_spans_x):
                if not mask[iy, ix]:
                    continue
                x0 = ix * span_x
                x1 = (ix + 1) * span_x
                y0 = iy * span_y
                y1 = (iy + 1) * span_y
                slabs_list.append((x0, x1, y0, y1, z_center, slab_thickness))

    # ============================================================
    # 转换为列表
    # ============================================================
    columns = list(columns_set)
    beams = beams_list

    # ============================================================
    # 统计 (按形状掩码内实际构件)
    # ============================================================
    x_count = sum(1 for b in beams if b[7] == 'x')
    y_count = sum(1 for b in beams if b[7] == 'y')
    n_cells, n_nodes = shape_cell_count(plane_shape, num_spans_x, num_spans_y)

    # 理论值 (掩码内)
    expected_cols = n_nodes * num_stories
    expected_beams_total = 4 * n_cells * num_stories
    expected_beams_x = 2 * n_cells * num_stories
    expected_beams_y = 2 * n_cells * num_stories

    # 形状面积 (掩码内格子面积) 与 节点数
    shape_area = n_cells * span_x * span_y
    shape_nodes_per_floor = n_nodes

    return {
        'columns': columns,
        'beams': beams,
        'slabs': slabs_list,
        'num_stories': num_stories,
        'story_height': story_height,
        'span_x': span_x,
        'span_y': span_y,
        'total_height': num_stories * story_height,
        'col_sections': col_sections,
        'beam_width': beam_width,
        'beam_height': beam_height,
        'beam_sections': (list(beam_sections) if beam_sections is not None
                          else [(beam_width, beam_height)] * num_stories),
        'slab_thickness': slab_thickness,
        'num_spans_x': num_spans_x,
        'num_spans_y': num_spans_y,
        'plane_shape': plane_shape,
        'shape_mask': mask,
        'shape_cells': n_cells,
        'shape_nodes_per_floor': n_nodes,
        'shape_area': shape_area,
        'axis_ratio_min': axis_ratio - 0.1,
        'axis_ratio_max': axis_ratio + 0.1,
        'tributary_area': span_x * span_y,
        'total_load': 20.0 * shape_area * num_stories,
        'stats': {
            'columns': len(columns),
            'beams': len(beams),
            'beams_x': x_count,
            'beams_y': y_count,
            'slabs': len(slabs_list),
            'expected_beams': expected_beams_total,
            'expected_beams_x': expected_beams_x,
            'expected_beams_y': expected_beams_y,
            'expected_cols': expected_cols,
            'beams_verified': (len(columns) == expected_cols and
                              len(beams) == expected_beams_total and
                              x_count == expected_beams_x and
                              y_count == expected_beams_y)
        }
    }


def generate_random_frame():
    """生成随机框架结构"""
    cfg = Config()
    
    num_stories = random.randint(cfg.NUM_STORIES_RANGE[0], cfg.NUM_STORIES_RANGE[1])
    story_height = random.choice(cfg.STORY_HEIGHTS)
    span_x = random.choice(cfg.SPAN_WIDTHS)
    span_y = random.choice(cfg.SPAN_WIDTHS)
    
    num_spans_x = max(1, int(cfg.SPACE_X // span_x) - 1)
    num_spans_y = max(1, int(cfg.SPACE_Y // span_y) - 1)
    num_spans_x = min(num_spans_x, 4)
    num_spans_y = min(num_spans_y, 4)
    # 避免单跨薄弱结构 (至少 2 跨)
    num_spans_x = max(2, num_spans_x)
    num_spans_y = max(2, num_spans_y)

    # 随机平面形状 (避免薄弱连接):
    #   - C/U 形: 横条(连接左右翼)中间段 = nx-2 需 >=2 跨 -> nx>=4
    #   - T/L 形: 至少 2×2
    #   - rect : 任意
    if num_spans_x >= 4 and num_spans_y >= 2:
        plane_shape = random.choice(['rect', 'T', 'L', 'C', 'U'])
    elif num_spans_x >= 2 and num_spans_y >= 2:
        plane_shape = random.choice(['rect', 'T', 'L'])
    else:
        plane_shape = 'rect'
    
    axis_ratio_min = random.uniform(0.35, 0.45)
    axis_ratio_max = random.uniform(0.70, 0.85)
    
    max_span = max(span_x, span_y)
    beam_height = random.uniform(max_span / 15, max_span / 10)
    beam_height = max(0.4, min(beam_height, 0.9))
    beam_height = round(beam_height / 0.2) * 0.2   # 200mm 模数
    
    beam_width = random.uniform(beam_height / 3, beam_height / 2)
    beam_width = max(0.2, min(beam_width, 0.5))
    beam_width = round(beam_width / 0.2) * 0.2   # 200mm 模数
    
    frame = generate_fixed_frame(
        num_stories=num_stories,
        num_spans_x=num_spans_x,
        num_spans_y=num_spans_y,
        span_x=span_x,
        span_y=span_y,
        story_height=story_height,
        axis_ratio=(axis_ratio_min + axis_ratio_max) / 2,
        beam_width=beam_width,
        beam_height=beam_height,
        plane_shape=plane_shape
    )
    
    frame['axis_ratio_min'] = axis_ratio_min
    frame['axis_ratio_max'] = axis_ratio_max
    
    return frame


def print_frame_info(frame):
    """打印框架结构信息"""
    s = frame['stats']
    verified_status = "✅ 通过" if s.get('beams_verified', False) else "❌ 失败"
    
    print("="*60)
    print("框架结构信息")
    print("="*60)
    print(f"  层数: {frame['num_stories']}")
    print(f"  层高: {frame['story_height']:.2f}m")
    print(f"  总高: {frame['total_height']:.2f}m")
    print(f"  跨度: {frame['span_x']:.2f}m × {frame['span_y']:.2f}m")
    print(f"  跨数: {frame['num_spans_x']} × {frame['num_spans_y']}")
    print(f"  平面形状: {frame.get('plane_shape', 'rect').upper()}"
          f" (格子 {frame.get('shape_cells', frame['num_spans_x']*frame['num_spans_y'])}, "
          f"节点 {frame.get('shape_nodes_per_floor', (frame['num_spans_x']+1)*(frame['num_spans_y']+1))})")
    print(f"  梁截面: {frame['beam_width']*1000:.0f}×{frame['beam_height']*1000:.0f}mm")
    print(f"  构件数量:")
    print(f"    柱: {s['columns']} 根")
    print(f"    梁: {s['beams']} 根 (X方向: {s['beams_x']}, Y方向: {s['beams_y']})")
    print(f"    板: {s['slabs']} 块")
    print(f"  验证: 期望 {s['expected_beams']} (X: {s['expected_beams_x']}, Y: {s['expected_beams_y']})")
    print(f"  状态: {verified_status}")
    print("="*60)


if __name__ == '__main__':
    print("="*60)
    print("框架生成测试")
    print("="*60)
    
    # 测试不同配置 + 形状
    test_configs = [
        (3, 2, 2, 6.0, 6.0, 3.5, 'rect'),
        (5, 3, 2, 6.0, 5.0, 3.5, 'rect'),
        (2, 1, 1, 8.0, 8.0, 4.0, 'rect'),
        (4, 3, 3, 6.0, 6.0, 3.5, 'T'),
        (4, 3, 3, 6.0, 6.0, 3.5, 'L'),
        (4, 3, 3, 6.0, 6.0, 3.5, 'C'),
        (4, 3, 3, 6.0, 6.0, 3.5, 'U'),
    ]
    
    for i, (ns, nx, ny, sx, sy, sh, shape) in enumerate(test_configs):
        print(f"\n{'='*60}")
        print(f"测试 {i+1}: {ns}层, {nx}×{ny}跨, {sx}m×{sy}m, 层高{sh}m, 形状 {shape}")
        print(f"{'='*60}")
        
        frame = generate_fixed_frame(
            num_stories=ns,
            num_spans_x=nx,
            num_spans_y=ny,
            span_x=sx,
            span_y=sy,
            story_height=sh,
            axis_ratio=0.6,
            beam_width=0.3,
            beam_height=0.6,
            plane_shape=shape
        )
        
        print_frame_info(frame)
        # 打印掩码
        mask = frame.get('shape_mask')
        if mask is not None:
            print("平面掩码 (X=有板):")
            for iy in range(mask.shape[0] - 1, -1, -1):
                print("   " + ''.join('X' if mask[iy, ix] else '.' for ix in range(mask.shape[1])))