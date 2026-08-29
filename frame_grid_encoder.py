# frame_grid_encoder.py
"""
直接切杆系模型编码 (FrameGridEncoder) — 128bit 体素编码

核心思路 (用户要求 2026-08-19):
    在杆系模型上切 64×64×64 网格 (固定 1m 体素, 原点对齐):
      - 坐标直接 ÷1m 定位 (ix = x/1, iy = y/1, k = z/1)
      - 楼板忽略 (楼面荷载已在节点荷载体现)
    每格一个 128 位编码, 记录:
      1) 6 个面是否贯通 (-X/+X/-Y/+Y/-Z/+Z)
      2) 每个贯通面对应的构件截面尺寸 (宽×高等级)
      3) 每个贯通面构件材料编码
      4) 节点 X/Y/Z 三向偏位
    由六面贯通可推断构件类型 (无独立 combo 字段):
      - -Z+Z 贯通 = 柱 (竖向杆)
      - -X+X 贯通 = X 方向梁
      - -Y+Y 贯通 = Y 方向梁
      组合可同时出现 (柱+梁 节点等)

编码位段 (bit0 = LSB, 共 128 位):
    bit0-5    : 6 面贯通标记  (-X=1, +X=2, -Y=4, +Y=8, -Z=16, +Z=32)
    bit6-53   : 6 面截面尺寸 6×8=48 位
                  每面 8 位 = 4位宽等级(bit0-3) + 4位高等级(bit4-7), 0~15
                  顺序同 6 面 (-X,+X,-Y,+Y,-Z,+Z), 每面占 8 位:
                    face i (i=0..5) -> 偏移 bit6 + i*8
    bit54-77  : 6 面材料编码 6×4=24 位, 每面 4 位 (0~15, 16 种材料)
                  顺序同 6 面, 每面占 4 位: bit54 + i*4
    bit78-81  : 节点 X 偏位 (4 位有符号, -8~+7, 单位=格宽/8)
    bit82-85  : 节点 Y 偏位 (4 位有符号, -8~+7)
    bit86-89  : 节点 Z 偏位 (4 位有符号, -8~+7)
    bit90-127 : 预留 38 位 (未来扩展)

解码恢复:
    - 六面贯通 + 每面截面/材料 -> 每面杆件 (柱/梁) 及其截面
    - 节点偏位 -> 杆件精确坐标 (格中心 + 偏位·格宽/8)
    - 材料 -> 弹性模量 (默认 0=混凝土)
"""
import os
import numpy as np

# 材料 (默认混凝土; 16 种材料, 0=混凝土)
E_CONCRETE = 3.25e10          # 弹性模量 Pa
MATERIAL_NAMES = {
    0: 'concrete', 1: 'steel', 2: 'composite',
}
MATERIAL_E = {                 # 弹性模量 (Pa), 未知材料回退混凝土
    0: 3.25e10, 1: 2.0e11, 2: 1.2e11,
}

GRID = 64                     # 每方向 64 格 (1m/格, 覆盖 64m 空间)
GRID_X = GRID
GRID_Y = GRID
MAX_FLOORS = GRID             # 竖向 64 格
SPACE_SIZE = 64.0             # 64 格 × 1m = 64m 空间
CELL_SIZE = 1.0               # 每格真实尺寸 1m (用户要求 2026-08-19)
SECTION_MOD = 0.2             # 200mm 模数


# ============================================================
# 128bit 编码位段布局
# ============================================================
N_BITS = 128

# 6 面顺序
FACE_NX, FACE_PX, FACE_NY, FACE_PY, FACE_NZ, FACE_PZ = range(6)
FACE_NAMES = ['-X', '+X', '-Y', '+Y', '-Z', '+Z']
FACE_BITS = [1, 2, 4, 8, 16, 32]          # bit0-5 六面贯通标记

# 每面截面 8 位 (4位宽等级 + 4位高等级), 偏移 = 6 + face*8
FACE_SECTION_OFFSET = 6
FACE_SECTION_BITS = 8                     # 每面截面占 8 位
# 每面材料 4 位, 偏移 = 54 + face*4
FACE_MATERIAL_OFFSET = FACE_SECTION_OFFSET + 6 * FACE_SECTION_BITS   # 54
FACE_MATERIAL_BITS = 4

# 节点三向偏位 (4 位有符号, -8~+7)
OFF_X_OFFSET = FACE_MATERIAL_OFFSET + 6 * FACE_MATERIAL_BITS          # 78
OFF_Y_OFFSET = OFF_X_OFFSET + 4                                       # 82
OFF_Z_OFFSET = OFF_Y_OFFSET + 4                                       # 86

# 截面等级: 4 位 -> 0~15 (0.2m 模数 -> 0.2~3.2m)
SECTION_LEVEL_MAX = 15


# 材料默认值
DEFAULT_MATERIAL = 0          # 混凝土


# ============================================================
# 截面/材料等级工具
# ============================================================
def _section_level(size, max_lvl=SECTION_LEVEL_MAX):
    """截面尺寸 -> 0.2m 模数等级 (0~max_lvl), 四舍五入到最近档, 越界钳制"""
    lvl = int(round(size / SECTION_MOD)) - 1
    return int(np.clip(lvl, 0, max_lvl))


def _level_section(lvl):
    """等级 -> 截面尺寸 (m)"""
    return SECTION_MOD * (int(lvl) + 1)


def _material_e(mat_id):
    """材料编码 -> 弹性模量 (Pa)."""
    return MATERIAL_E.get(int(mat_id), E_CONCRETE)


# ============================================================
# 128bit 编码构建 / 解码辅助
# ============================================================
def _pack_faces(faces):
    """6 面贯通标记 -> 6 bit (bit0-5)."""
    return int(faces) & 0x3F


def _pack_face_section(face, w_lvl, h_lvl):
    """把某面截面 (宽/高等级 0~15) 编码进 8 位字段."""
    return ((int(h_lvl) & 0xF) << 4) | (int(w_lvl) & 0xF)


def _pack_face_material(face, mat):
    """把某面材料 (0~15) 编码进 4 位字段."""
    return int(mat) & 0xF


def _set_face_section(code, face, w_lvl, h_lvl):
    """在 128bit code 的 face 截面字段写入宽/高等级 (返回新 code)."""
    shift = FACE_SECTION_OFFSET + face * FACE_SECTION_BITS
    val = _pack_face_section(face, w_lvl, h_lvl)
    return (code & ~(0xFF << shift)) | (val << shift)


def _set_face_material(code, face, mat):
    """在 128bit code 的 face 材料字段写入材料编码 (返回新 code)."""
    shift = FACE_MATERIAL_OFFSET + face * FACE_MATERIAL_BITS
    val = _pack_face_material(face, mat)
    return (code & ~(0xF << shift)) | (val << shift)


def _set_offset(code, which, val):
    """写入节点偏位 (which: 'x'/'y'/'z', val: -8~+7)."""
    off = {'x': OFF_X_OFFSET, 'y': OFF_Y_OFFSET, 'z': OFF_Z_OFFSET}[which]
    v = (int(val) + 8) & 0xF   # 有符号 -> 无符号 (0~15)
    return (code & ~(0xF << off)) | (v << off)


def _get_faces(code):
    return int(code) & 0x3F


def _get_face_section(code, face):
    """读某面截面 -> (w_lvl, h_lvl)."""
    shift = FACE_SECTION_OFFSET + face * FACE_SECTION_BITS
    val = (int(code) >> shift) & 0xFF
    return (val & 0xF), ((val >> 4) & 0xF)


def _get_face_material(code, face):
    """读某面材料编码 (0~15)."""
    shift = FACE_MATERIAL_OFFSET + face * FACE_MATERIAL_BITS
    return (int(code) >> shift) & 0xF


def _get_offset(code, which):
    """读节点偏位 (有符号 -8~+7)."""
    off = {'x': OFF_X_OFFSET, 'y': OFF_Y_OFFSET, 'z': OFF_Z_OFFSET}[which]
    return ((int(code) >> off) & 0xF) - 8


def _face_has_member(code, face):
    """某面是否贯通 (有构件穿过)."""
    return bool(_get_faces(code) & FACE_BITS[face])


# ============================================================
# 六面贯通 -> 构件组合 (兼容旧 combo 字段, 供下游复用)
# ============================================================
# 组合编码 (3 个独立 bit: 柱=1, X梁=2, Y梁=4, 可组合)
COMBO_COL = 1
COMBO_BX = 2
COMBO_BY = 4
COMBO_NAMES = {
    0: '空', 1: '柱C', 2: 'X梁', 3: '柱+X梁',
    4: 'Y梁', 5: '柱+Y梁', 6: 'X+Y梁', 7: '柱+X+Y梁',
}


def faces_to_combo(faces):
    """六面贯通标记 -> combo (柱=X? 由 -Z/+Z; X梁 由 -X/+X; Y梁 由 -Y/+Y)."""
    faces = int(faces)
    combo = 0
    if faces & FACE_BITS[FACE_NZ] and faces & FACE_BITS[FACE_PZ]:
        combo |= COMBO_COL
    if faces & FACE_BITS[FACE_NX] and faces & FACE_BITS[FACE_PX]:
        combo |= COMBO_BX
    if faces & FACE_BITS[FACE_NY] and faces & FACE_BITS[FACE_PY]:
        combo |= COMBO_BY
    return combo


def combo_from_faces(faces):
    """faces_to_combo 的别名 (语义清晰)."""
    return faces_to_combo(faces)


# ============================================================
# 结构尺寸 / 网格几何 (网格贴合结构, 原点对齐)
# ============================================================
def frame_dims(model):
    """返回结构尺寸: (xmax, ymax, zmax)."""
    from frame_model import iter_columns, iter_beams
    xs, ys, zs = [], [], []
    for c in iter_columns(model):
        xs += [c['x'], c['x']]
        ys += [c['y'], c['y']]
        zs += [c['z_bottom'], c['z_top']]
    for b in iter_beams(model):
        xs += [b['x1'], b['x2']]
        ys += [b['y1'], b['y2']]
        zs += [b['z'], b['z']]
    if not xs:
        return (1.0, 1.0, 1.0)
    return (max(xs), max(ys), max(zs))


def grid_cells(model, grid_x=GRID_X, grid_y=GRID_Y, max_floors=MAX_FLOORS):
    """返回每格尺寸 (dx, dy, dz).

    体素固定真实尺寸 1m (不可缩放): dx = dy = dz = CELL_SIZE = 1.0.
    网格 64×64×64 对应 64m 空间, 结构占其中一部分格, 其余为空.
    """
    c = CELL_SIZE
    return c, c, c


def cell_geometry(dx, dy, dz, ix, iy, k):
    """返回格子 (ix,iy,k) 的几何: (x0,x1,y0,y1,z0,z1)."""
    return (ix * dx, (ix + 1) * dx, iy * dy, (iy + 1) * dy,
            k * dz, (k + 1) * dz)


# ============================================================
# 主编码: frame_model -> 32×32×32 编码数组 (128bit/格, 固定 2m 体素)
# ============================================================
def encode_frame_grid(model, grid_x=GRID_X, grid_y=GRID_Y,
                      max_floors=MAX_FLOORS):
    """从杆系模型直接切格编码 (楼板忽略, 固定 2m 体素, 原点对齐).

    每格 128bit 编码, 记录六面贯通 + 每面截面/材料 + 节点三向偏位。
    返回 (codes, mass_grid):
        codes: object 数组 [grid_x, grid_y, max_floors], 每元素 Python int (128bit)
               0 = 空 (无任何构件)
        mass_grid: np.float32 [grid_x, grid_y, max_floors], 每格质量 kg
    """
    from frame_model import iter_columns, iter_beams
    dx, dy, dz = grid_cells(model, grid_x, grid_y, max_floors)

    codes = np.zeros((grid_x, grid_y, max_floors), dtype=object)
    mass_grid = np.zeros((grid_x, grid_y, max_floors), dtype=np.float32)

    def _clip_ij(ix, iy):
        return min(max(ix, 0), grid_x - 1), min(max(iy, 0), grid_y - 1)

    # ---- 柱: 原点对齐, 竖向穿过 z_bottom..z_top 覆盖的所有格子层 ----
    # 柱贯通 -Z + +Z 两面, 截面 = 每面 (w=col_size, h=col_size)
    for col in iter_columns(model):
        x, y = col['x'], col['y']
        cs = col['section']
        zb = col['z_bottom']
        zt = col['z_top']
        k0 = max(0, int(zb // dz))
        # 柱顶含边界: zt 正好在格边界(如 3.0/6.0/9.0)时归入该格,
        # 使柱与梁面(楼面标高)同格 -> 节点(柱+梁)正确识别
        k1 = min(max_floors - 1, max(k0, int(np.ceil(zt / dz - 1e-9))))
        ix, iy = _clip_ij(int(x // dx), int(y // dy))
        cx = (ix + 0.5) * dx
        cy = (iy + 0.5) * dy
        off_x = int(np.clip(round((x - cx) / dx * 8), -8, 7))
        off_y = int(np.clip(round((y - cy) / dy * 8), -8, 7))
        col_lvl = _section_level(cs)   # 0~15

        for k in range(k0, k1 + 1):
            code = codes[ix, iy, k]
            if code is None or code == 0:
                code = 0
            # 柱: -Z +Z 贯通; 每面截面 (col, col)
            code = code | FACE_BITS[FACE_NZ] | FACE_BITS[FACE_PZ]
            code = _set_face_section(code, FACE_NZ, col_lvl, col_lvl)
            code = _set_face_section(code, FACE_PZ, col_lvl, col_lvl)
            code = _set_face_material(code, FACE_NZ, DEFAULT_MATERIAL)
            code = _set_face_material(code, FACE_PZ, DEFAULT_MATERIAL)
            code = _set_offset(code, 'x', off_x)
            code = _set_offset(code, 'y', off_y)
            code = _set_offset(code, 'z', 0)   # 柱沿 Z 贯通格子中心
            codes[ix, iy, k] = code

    # ---- 梁: 楼层 z 处, 沿 X/Y 穿过沿线所有格子 ----
    # X 梁贯通 -X + +X 两面, 截面 (w, h); Y 梁贯通 -Y + +Y
    for b in iter_beams(model):
        z = b['z']
        k = min(max_floors - 1, max(0, int(z // dz)))
        w, h = b['width'], b['height']
        bw_lvl = _section_level(w)
        bh_lvl = _section_level(h)
        if b['direction'] == 'x':
            y = b['y1']
            iy = min(max(int(y // dy), 0), grid_y - 1)
            cy = (iy + 0.5) * dy
            off_y = int(np.clip(round((y - cy) / dy * 8), -8, 7))
            x0, x1 = b['x1'], b['x2']
            i0 = int(x0 // dx)
            # 梁端点含边界: x1 在格边界(如 12.0)时归入该格 -> 角柱节点识别
            i1 = min(grid_x - 1, max(0, int(np.ceil(x1 / dx - 1e-9))))
            for ix in range(max(0, i0), max(0, i1) + 1):
                code = codes[ix, iy, k]
                if code is None or code == 0:
                    code = 0
                # X 梁: -X +X 贯通; 截面 (w, h) 写入 -X/+X 面
                code = code | FACE_BITS[FACE_NX] | FACE_BITS[FACE_PX]
                code = _set_face_section(code, FACE_NX, bw_lvl, bh_lvl)
                code = _set_face_section(code, FACE_PX, bw_lvl, bh_lvl)
                code = _set_face_material(code, FACE_NX, DEFAULT_MATERIAL)
                code = _set_face_material(code, FACE_PX, DEFAULT_MATERIAL)
                code = _set_offset(code, 'y', off_y)
                # Z 偏位: 梁楼面位置相对格子中心 (负=偏下, 楼面常偏下)
                code = _set_offset(code, 'z',
                                   int(np.clip(round((z - (k + 0.5) * dz) / dz * 8),
                                               -8, 7)))
                # 若已有柱偏位, 保留柱的 off_x
                codes[ix, iy, k] = code
        else:
            x = b['x1']
            ix = min(max(int(x // dx), 0), grid_x - 1)
            cx = (ix + 0.5) * dx
            off_x = int(np.clip(round((x - cx) / dx * 8), -8, 7))
            y0, y1 = b['y1'], b['y2']
            j0 = int(y0 // dy)
            # 梁端点含边界
            j1 = min(grid_y - 1, max(0, int(np.ceil(y1 / dy - 1e-9))))
            for iy in range(max(0, j0), max(0, j1) + 1):
                code = codes[ix, iy, k]
                if code is None or code == 0:
                    code = 0
                # Y 梁: -Y +Y 贯通; 截面 (w, h) 写入 -Y/+Y 面
                code = code | FACE_BITS[FACE_NY] | FACE_BITS[FACE_PY]
                code = _set_face_section(code, FACE_NY, bw_lvl, bh_lvl)
                code = _set_face_section(code, FACE_PY, bw_lvl, bh_lvl)
                code = _set_face_material(code, FACE_NY, DEFAULT_MATERIAL)
                code = _set_face_material(code, FACE_PY, DEFAULT_MATERIAL)
                code = _set_offset(code, 'x', off_x)
                # Z 偏位: 梁楼面位置相对格子中心
                code = _set_offset(code, 'z',
                                   int(np.clip(round((z - (k + 0.5) * dz) / dz * 8),
                                               -8, 7)))
                codes[ix, iy, k] = code

    # ---- 楼层荷载质量通道: 每层 floor_masses[fl] 均摊到该层楼板投影格 ----
    floor_masses = model.get('floor_masses') or []
    frame = model['frame']
    ns = int(frame['num_stories'])
    sh = float(frame['story_height'])
    xmax, ymax, _ = frame_dims(model)
    n_ix = min(grid_x, int(np.ceil(xmax / dx)) + 1)
    n_iy = min(grid_y, int(np.ceil(ymax / dy)) + 1)
    for fl in range(min(ns, max_floors)):
        z_floor = (fl + 1) * sh
        k = min(max_floors - 1, max(0, int(z_floor // dz)))
        m_floor = float(floor_masses[fl]) if fl < len(floor_masses) else 0.0
        if m_floor <= 0:
            continue
        # 均摊到该层楼板投影区域格
        n_cells = max(1, n_ix * n_iy)
        per_cell = m_floor / n_cells
        for ix in range(n_ix):
            for iy in range(n_iy):
                mass_grid[ix, iy, k] += per_cell

    return codes, mass_grid


# ============================================================
# 连续物理量特征 (LLM embedding 启发): 相似格子向量距离近
# ============================================================
# 类型连续标量 (不是独热, 而是"柔性->刚性"的有序梯度, 类似 token 语义序)
#   空=0, 板=0.5, X梁=1.0, Y梁=1.2, X+Y梁=1.6, 柱=2.0, 柱+梁=2.5
#   单调: 越接近 2.0 越"柱型"(刚度集中), 越接近 0 越"空/板"
TYPE_SCALAR = {
    0: 0.0,            # 空
    2: 1.0,            # X梁
    4: 1.2,            # Y梁
    6: 1.6,            # X+Y梁
    1: 2.0,            # 柱
    3: 2.5,            # 柱+X梁
    5: 2.6,            # 柱+Y梁
    7: 3.0,            # 柱+X+Y梁
}

# 每格连续特征通道 (C 维, 相似格子欧氏距离近):
#   ch0: 类型标量 (柔性->刚性梯度)
#   ch1: 柱刚度 log10(EI_col) 归一化
#   ch2: 梁刚度 log10(EI_beam) 归一化
#   ch3: 密度 (质量/格体积) 归一化
#   ch4: 节点偏位 X (真实米, 归一化到格宽)
#   ch5: 节点偏位 Y
FEAT_C = 6


def _log1p_norm(v, lo, hi):
    """log1p(v) 线性映射到 [0,1] (lo/hi 为 log1p 后的范围)"""
    lv = np.log1p(max(float(v), 0.0))
    return float(np.clip((lv - lo) / (hi - lo), 0.0, 1.0))


def encode_frame_grid_features(model, grid_x=GRID_X, grid_y=GRID_Y,
                               max_floors=MAX_FLOORS):
    """输出每格连续物理量向量 [32,32,32,C] (LLM embedding 启发).

    相似格子 (类型/刚度/密度接近) 在 C 维空间中欧氏距离近:
      - 类型用连续梯度 (空=0 → 梁=1 → 柱=2), 非独热
      - 刚度用 log10(EI) 单调映射 (小刚度→小值, 大刚度→大值)
      - 密度 = 质量/格体积
      - 节点偏位用真实米数

    Returns:
        feats: np.float32 [grid_x, grid_y, max_floors, C]
    """
    from frame_model import iter_columns, iter_beams
    codes, mass_grid = encode_frame_grid(model, grid_x, grid_y, max_floors)
    dx = dy = dz = CELL_SIZE
    cell_vol = dx * dy * dz

    feats = np.zeros((grid_x, grid_y, max_floors, FEAT_C), dtype=np.float32)

    # 刚度归一化范围 (log1p(EI), EI 单位 N·m²):
    #   EI_col ~ E*0.2^4/12 = 4.3e6  ~ E*1.4^4/12 = 7.4e9
    #   EI_beam ~ E*0.2*0.2^3/12 = 4.3e5 ~ E*0.5*0.8^3/12 = 6.9e8
    lo_ei = np.log1p(1e5)
    hi_ei = np.log1p(1e10)

    for k in range(max_floors):
        for iy in range(grid_y):
            for ix in range(grid_x):
                code = codes[ix, iy, k]
                if code == 0:
                    continue
                d = decode_cell(code)
                combo = d['combo']
                # ch0: 类型梯度
                feats[ix, iy, k, 0] = TYPE_SCALAR.get(combo, 0.0)
                # ch1: 柱刚度
                if d['ei_col'] > 0:
                    feats[ix, iy, k, 1] = _log1p_norm(d['ei_col'], lo_ei, hi_ei)
                # ch2: 梁刚度
                if d['ei_beam'] > 0:
                    feats[ix, iy, k, 2] = _log1p_norm(d['ei_beam'], lo_ei, hi_ei)
                # ch3: 密度 (质量/体积)
                m = mass_grid[ix, iy, k]
                if m > 0:
                    rho = m / cell_vol
                    # 密度归一化: 混凝土 ~2400 kg/m³, 楼面荷载 ~ 几百~几千
                    feats[ix, iy, k, 3] = float(np.clip(rho / 5000.0, 0.0, 1.0))
                # ch4/5: 节点偏位 (格宽/2 归一化 → [-1,1])
                feats[ix, iy, k, 4] = float(np.clip(d['off_x'] / 8.0, -1, 1))
                feats[ix, iy, k, 5] = float(np.clip(d['off_y'] / 8.0, -1, 1))

    return feats


# ============================================================
# 微元词表 (VoxelVocab) — LLM tokenizer 思想
# ============================================================
# 微元类型 = (combo, col_lvl, bw_lvl, bh_lvl, off_x, off_y) 全离散组合。
# 词表从数据库扫描构建 (不依赖具体样本的连续物理量), 每个微元一个 token ID。
#   0 = 空 (无构件)
#   1..V-1 = 各微元类型 (按出现频率降序编号, 频率高 ID 小)
# 推理时: 每个格子按离散截面/偏位档位查表得到 token ID, 与具体样本解耦。
MICRO_KEY_NAMES = ['combo', 'col_lvl', 'bw_lvl', 'bh_lvl', 'off_x', 'off_y']


class VoxelVocab:
    """微元词表: 数据库扫描 → 固定 token 映射.

    Attributes:
        id2micro: dict {token_id: micro_key}, micro_key = (combo,col_lvl,bw_lvl,bh_lvl,off_x,off_y)
        micro2id: dict {micro_key: token_id}
        counts  : dict {micro_key: 出现次数}
    """

    def __init__(self):
        self.id2micro = {0: (0, 0, 0, 0, 0, 0)}   # token 0 = 空
        self.micro2id = {(0, 0, 0, 0, 0, 0): 0}
        self.counts = {}
        self.built = False

    # ---------- 构建词表 ----------
    @staticmethod
    def _micro_key_from_cell(d):
        """从 decode_cell 结果提取离散微元键 (不依赖样本).

        键 = (combo, col_lvl, bw_lvl, bh_lvl, off_x, off_y), 保持 6 元组兼容:
            col_lvl: 柱截面等级 (取 -Z 面)
            bw_lvl/bh_lvl: 梁截面宽/高等级 (取 -X 或 -Y 贯通面)
        """
        combo = d['combo']
        secs = d.get('face_sections', {})
        # 柱截面等级 (-Z 或 +Z 面)
        col_lvl = 0
        if combo & COMBO_COL:
            f = FACE_NZ if FACE_NZ in secs else FACE_PZ
            col_lvl = int(np.clip(secs.get(f, (0, 0))[0], 0, SECTION_LEVEL_MAX))
        # 梁截面等级 (-X 或 -Y 贯通面)
        bw_lvl = 0
        bh_lvl = 0
        if combo & (COMBO_BX | COMBO_BY):
            f = FACE_NX if combo & COMBO_BX else FACE_NY
            w_lvl, h_lvl = secs.get(f, (0, 0))
            bw_lvl = int(np.clip(w_lvl, 0, SECTION_LEVEL_MAX))
            bh_lvl = int(np.clip(h_lvl, 0, SECTION_LEVEL_MAX))
        return (combo, col_lvl, bw_lvl, bh_lvl, d['off_x'], d['off_y'])

    def add_codes(self, codes):
        """从编码数组统计微元出现 (增量累积 counts)."""
        nz = np.argwhere(codes != 0)
        for ix, iy, k in nz:
            d = decode_cell(codes[ix, iy, k])
            key = self._micro_key_from_cell(d)
            self.counts[key] = self.counts.get(key, 0) + 1

    def build(self, min_freq=1):
        """根据统计次数构建 token 映射 (频率降序)."""
        # 频率降序排序 (频率相同按组合稳定)
        ranked = sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))
        for key, n in ranked:
            if n < min_freq:
                continue
            if key in self.micro2id:
                continue
            tid = len(self.id2micro)
            self.id2micro[tid] = key
            self.micro2id[key] = tid
        self.built = True
        return len(self.id2micro)

    # ---------- 编码 ----------
    def encode_codes(self, codes):
        """把编码数组映射为 token ID 数组 [32,32,32]."""
        tok = np.zeros(codes.shape, dtype=np.int64)
        nz = np.argwhere(codes != 0)
        for ix, iy, k in nz:
            d = decode_cell(codes[ix, iy, k])
            key = self._micro_key_from_cell(d)
            tok[ix, iy, k] = self.micro2id.get(key, 0)
        return tok

    # ---------- 物理量向量 (用于临近性分析, 不参与训练) ----------
    def micro_physics_vector(self, key):
        """微元的物理量向量 (类型/刚度/密度代理), 供 PCA 临近性分析."""
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

    # ---------- 物理向量初始化 embedding (供 token encoder 用) ----------
    def physics_embeddings(self, embed_dim=32, rich=True, mode=None):
        """为每个 token 生成物理量向量并 PCA 降维到 embed_dim (初始化 embedding 用).

        目标: 让"刚度/截面相似"的微元 token 在 embedding 空间中也邻近.
        返回 np.float32 [vocab_size, embed_dim]:
            token 0 (空) 向量 = 0; 其余 token 用物理向量 PCA 投影.

        Args:
            mode: 物理向量模式
                'rich8'  : rich 8 维 (类型/柱EI/梁EI/面积/偏位/填充)
                'hexa9'  : 六面体刚度 9 维 (3 对对面 剪切GA+抗弯EI + 类型/填充/偏位)
                'basic5' : 精简 5 维 (类型/柱EI/梁EI/偏位X/偏位Y)
            兼容旧参数: rich=True -> 'rich8', rich=False -> 'basic5'
        """
        import numpy as _np
        if mode is None:
            mode = 'rich8' if rich else 'basic5'
        vec_fn = {
            'rich8': micro_physics_vector_rich,
            'hexa9': hexa_stiffness_vector,
            'basic5': VoxelVocab.micro_physics_vector,
        }.get(mode)
        if vec_fn is None:
            raise ValueError(f"未知物理向量模式: {mode} (可选 rich8/hexa9/basic5)")
        dims = {'rich8': 8, 'hexa9': 9, 'basic5': 5}[mode]
        n_tok = len(self.id2micro)
        # 收集所有 token 的物理向量 (token 0 空置零)
        phys = []
        for tid in range(n_tok):
            key = self.id2micro[tid]
            if key[0] == 0:   # 空
                phys.append(_np.zeros(dims, dtype=_np.float32))
            else:
                phys.append(vec_fn(key))
        X = _np.stack(phys)                      # [V, D]
        # 去掉全零列 (避免 PCA 除零)
        col_std = X.std(axis=0)
        keep = col_std > 1e-9
        Xr = X[:, keep]
        if Xr.shape[1] == 0:
            return _np.zeros((n_tok, embed_dim), dtype=_np.float32)
        # 中心化
        Xc = Xr - Xr.mean(axis=0, keepdims=True)
        # PCA: 用 SVD 求前 embed_dim 主成分 (稳健, 不依赖 sklearn)
        U, S, Vt = _np.linalg.svd(Xc, full_matrices=False)
        k = min(int(embed_dim), Xc.shape[1])
        proj = Xc @ Vt[:k].T                     # [V, k]
        # 归一化每行到单位方差 (避免初始尺度过大), 空向量保持 0
        row_std = proj.std(axis=0, keepdims=True) + 1e-8
        proj = proj / row_std
        out = _np.zeros((n_tok, embed_dim), dtype=_np.float32)
        out[:, :k] = proj.astype(_np.float32)
        return out

    # ---------- 保存 / 加载 ----------
    def save(self, path):
        import pickle as _pkl
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            _pkl.dump({'id2micro': self.id2micro, 'counts': self.counts,
                       'built': self.built}, f, protocol=_pkl.HIGHEST_PROTOCOL)

    def load(self, path):
        import pickle as _pkl
        with open(path, 'rb') as f:
            data = _pkl.load(f)
        self.id2micro = data['id2micro']
        self.counts = data.get('counts', {})
        self.built = data.get('built', True)
        self.micro2id = {v: k for k, v in self.id2micro.items()}
        return True


def micro_physics_vector_rich(key):
    """8 维物理量向量: 类型/柱EI/梁EI/柱面积/梁面积/偏位X/偏位Y/填充强度.

    微元键 = (combo, col_lvl, bw_lvl, bh_lvl, off_x, off_y).
    用于 embedding 初始化和词表临近性分析: 刚度/截面相似的微元,
    在此 8 维空间中欧氏距离近 (同类构件的截面档位单调变化).
    """
    combo, col_lvl, bw_lvl, bh_lvl, off_x, off_y = key
    col = _level_section(col_lvl) if (combo & COMBO_COL) else 0.0
    bw = _level_section(bw_lvl) if (combo & (COMBO_BX | COMBO_BY)) else 0.0
    bh = _level_section(bh_lvl) if (combo & (COMBO_BX | COMBO_BY)) else 0.0
    ei_col = E_CONCRETE * col**4 / 12.0 if col > 0 else 0.0
    ei_beam = E_CONCRETE * bw * bh**3 / 12.0 if bw > 0 and bh > 0 else 0.0
    lo_ei = np.log1p(1e5); hi_ei = np.log1p(1e10)
    # 填充强度: 组合中含构件数量 (0/1/2/3) 归一化
    n_comp = (1 if combo & COMBO_COL else 0) + \
             (1 if combo & COMBO_BX else 0) + (1 if combo & COMBO_BY else 0)
    return np.array([
        TYPE_SCALAR.get(combo, 0.0) / 3.0,            # 类型梯度 0~1
        _log1p_norm(ei_col, lo_ei, hi_ei),            # 柱刚度
        _log1p_norm(ei_beam, lo_ei, hi_ei),           # 梁刚度
        (col / 1.4)**2,                                # 柱截面面积 (归一化 0.2²/1.4²)
        (bw * bh) / (0.5 * 0.8),                       # 梁截面面积 (归一化)
        float(np.clip(off_x / 8.0, -1, 1)),           # 偏位X
        float(np.clip(off_y / 8.0, -1, 1)),           # 偏位Y
        n_comp / 3.0,                                  # 填充强度
    ], dtype=np.float32)


def hexa_stiffness_vector(key):
    """六面体微元刚度向量 (9 维): 3 对对面的剪切/抗弯刚度简化计算.

    概念: 一个格子 = 六面体微元, 有 3 对对面 (X: -X/+X, Y: -Y/+Y, Z: -Z/+Z).
    每对对面若有杆件贯通 (该方向有构件), 则该方向同时贡献:
      - 抗弯刚度 EI (杆件截面沿正交轴的弯曲刚度)
      - 剪切刚度 GA (截面剪切刚度, 简化取 G·A, G≈0.4E)

    各方向杆件截面:
      - Z 方向: 柱 (截面 col×col, 竖杆)
          EI_z = E·col⁴/12 (绕水平轴), GA_z = G·col²
      - X 方向: X 梁 (截面 bw×bh)
          EI_x = E·bw·bh³/12 (竖向弯曲), GA_x = G·bw·bh
      - Y 方向: Y 梁 (截面 bw×bh)
          EI_y = E·bw³·bh/12 (竖向弯曲), GA_y = G·bw·bh

    输出 9 维 (每方向 3 维): [EI_x, GA_x, EI_y, GA_y, EI_z, GA_z, 类型梯度, 填充强度, 偏位幅]
    全部 log/线性归一化到约 [0,1], 使"刚度/截面相似"的微元欧氏距离近.
    """
    combo, col_lvl, bw_lvl, bh_lvl, off_x, off_y = key
    E = E_CONCRETE
    G = 0.4 * E   # 剪切模量 (简化)
    col = _level_section(col_lvl) if (combo & COMBO_COL) else 0.0
    bw = _level_section(bw_lvl) if (combo & (COMBO_BX | COMBO_BY)) else 0.0
    bh = _level_section(bh_lvl) if (combo & (COMBO_BX | COMBO_BY)) else 0.0

    # 各方向抗弯/剪切刚度 (未贯通方向 = 0)
    ei_x = E * bw * bh**3 / 12.0 if (combo & COMBO_BX) else 0.0   # X梁 竖向弯曲
    ga_x = G * bw * bh if (combo & COMBO_BX) else 0.0             # X梁 剪切
    ei_y = E * bw**3 * bh / 12.0 if (combo & COMBO_BY) else 0.0   # Y梁 竖向弯曲
    ga_y = G * bw * bh if (combo & COMBO_BY) else 0.0             # Y梁 剪切
    ei_z = E * col**4 / 12.0 if (combo & COMBO_COL) else 0.0      # 柱 抗弯
    ga_z = G * col**2 if (combo & COMBO_COL) else 0.0             # 柱 剪切

    # log1p 归一化 (EI: 1e5~1e10, GA: 1e7~1e10)
    lo_ei = np.log1p(1e5); hi_ei = np.log1p(1e10)
    lo_ga = np.log1p(1e6); hi_ga = np.log1p(1e10)
    n_comp = (1 if combo & COMBO_COL else 0) + \
             (1 if combo & COMBO_BX else 0) + (1 if combo & COMBO_BY else 0)
    off_mag = float(np.hypot(off_x, off_y)) / 8.0   # 偏位幅 [0,1]
    return np.array([
        _log1p_norm(ei_x, lo_ei, hi_ei),   # 0: X 抗弯
        _log1p_norm(ga_x, lo_ga, hi_ga),   # 1: X 剪切
        _log1p_norm(ei_y, lo_ei, hi_ei),   # 2: Y 抗弯
        _log1p_norm(ga_y, lo_ga, hi_ga),   # 3: Y 剪切
        _log1p_norm(ei_z, lo_ei, hi_ei),   # 4: Z 抗弯 (柱)
        _log1p_norm(ga_z, lo_ga, hi_ga),   # 5: Z 剪切 (柱)
        TYPE_SCALAR.get(combo, 0.0) / 3.0,  # 6: 类型梯度
        n_comp / 3.0,                       # 7: 填充强度
        off_mag,                            # 8: 偏位幅
    ], dtype=np.float32)


def build_voxel_vocab_from_db(db, n_structs=None, min_freq=1):
    """扫描数据库构建微元词表.

    Args:
        db: SLFDatabase 实例 (已连接)
        n_structs: 扫描结构数上限 (None=全部, 建议抽样加快)
        min_freq: 最小出现次数 (过滤罕见微元)

    Returns:
        VoxelVocab
    """
    from frame_model import build_frame_model
    from db_manager import ST_TABLE  # v3 表名 (避免顶部循环导入)
    if n_structs:
        db.cur.execute(
            f"SELECT struct_id FROM {ST_TABLE} ORDER BY random() LIMIT %s",
            (int(n_structs),))
        ids = [r['struct_id'] for r in db.cur.fetchall()]
    else:
        db.cur.execute(f"SELECT struct_id FROM {ST_TABLE}")
        ids = [r['struct_id'] for r in db.cur.fetchall()]
    vocab = VoxelVocab()
    for sid in ids:
        struct = db.get_structure(sid)
        if struct is None:
            continue
        model = build_frame_model(struct=struct)
        codes, _ = encode_frame_grid(model)
        vocab.add_codes(codes)
    n_tok = vocab.build(min_freq=min_freq)
    return vocab, n_tok


# ============================================================
# 解码: 编码数组 -> 恢复杆系信息
# ============================================================
def decode_cell(code):
    """解码单个格子 128bit 编码 -> dict (构件/截面/偏位/刚度/材料).

    兼容旧字段 (供下游复用):
        combo / combo_name : 由六面贯通推断
        col / bw / bh      : 柱截面 / 梁截面 (取贯通面对应截面)
        off_x / off_y      : 节点偏位
        ei_col / ei_beam   : 刚度
    新增:
        faces / face_names : 六面贯通
        sections           : {face: (w_lvl, h_lvl, w_m, h_m)}
        materials          : {face: mat_id}
        off_z              : Z 向偏位
    """
    code = int(code) or 0
    faces = code & 0x3F
    combo = faces_to_combo(faces)

    off_x = _get_offset(code, 'x')
    off_y = _get_offset(code, 'y')
    off_z = _get_offset(code, 'z')

    # 每面截面/材料
    face_sections = {}   # face -> (w_lvl, h_lvl)
    face_materials = {}  # face -> mat_id
    face_w = {}
    face_h = {}
    for f in range(6):
        if faces & FACE_BITS[f]:
            w_lvl, h_lvl = _get_face_section(code, f)
            mat = _get_face_material(code, f)
            face_sections[f] = (w_lvl, h_lvl)
            face_materials[f] = mat
            face_w[f] = _level_section(w_lvl)
            face_h[f] = _level_section(h_lvl)

    # 兼容字段: 柱截面 = -Z/+Z 面 (竖向杆); 梁截面 = -X/+X 或 -Y/+Y 面
    col = 0.0
    if combo & COMBO_COL:
        f = FACE_NZ if FACE_NZ in face_w else FACE_PZ
        col = face_w.get(f, 0.0)
    bw = 0.0
    bh = 0.0
    if combo & (COMBO_BX | COMBO_BY):
        f = FACE_NX if combo & COMBO_BX else FACE_NY
        bw = face_w.get(f, 0.0)
        bh = face_h.get(f, 0.0)

    # 刚度 (EI, N·m²): 用每面实际材料弹模
    e_col = _material_e(face_materials.get(FACE_NZ, DEFAULT_MATERIAL))
    e_beam = _material_e(face_materials.get(FACE_NX, DEFAULT_MATERIAL))
    ei_col = e_col * col**4 / 12.0 if col > 0 else 0.0
    ei_beam = e_beam * bw * bh**3 / 12.0 if bw > 0 and bh > 0 else 0.0

    face_names = []
    for i, nm in enumerate(FACE_NAMES):
        if faces & FACE_BITS[i]:
            face_names.append(nm)

    return {
        'combo': combo, 'combo_name': COMBO_NAMES.get(combo, '?'),
        'faces': faces, 'face_names': face_names,
        'col': col, 'bw': bw, 'bh': bh,
        'off_x': off_x, 'off_y': off_y, 'off_z': off_z,
        'ei_col': ei_col, 'ei_beam': ei_beam,
        'face_sections': face_sections,   # {face: (w_lvl, h_lvl)}
        'face_materials': face_materials, # {face: mat_id}
        'face_w': face_w,                 # {face: 宽 m}
        'face_h': face_h,                 # {face: 高 m}
    }


def decode_frame_grid(codes, frame=None, grid_x=GRID_X, grid_y=GRID_Y,
                      max_floors=MAX_FLOORS):
    """解码整个编码数组, 恢复结构描述 (固定 2m 体素, 原点对齐).

    Args:
        codes: [grid_x, grid_y, max_floors] 编码数组
        frame: 原 frame (用于恢复楼层数), 可选
        grid_x/grid_y/max_floors: 网格数 (默认 32×32×32)

    Returns:
        dict: {'columns': [...], 'beam_x': [...], 'beam_y': [...],
               'n_floors': 实际楼层数}
    """
    codes = np.asarray(codes, dtype=object)
    if frame is not None:
        ns = int(frame['num_stories'])
    else:
        ns = 1
    dx = dy = dz = CELL_SIZE   # 固定 1m 体素
    xmax = grid_x * CELL_SIZE
    ymax = grid_y * CELL_SIZE

    # 柱: 收集"每格一段", 再按 (x, y, section) 合并连续 k 段为一整根柱
    col_segs = []   # (round(px), round(py), section, k, z_bot, z_top)
    # 梁先收集"每格一段", 再按 (楼层格, 固定轴坐标, 截面) 合并连续段
    bx_segs = []   # (k, y, bw, bh, ix)
    by_segs = []   # (k, x, bw, bh, iy)
    for k in range(max_floors):
        for iy in range(grid_y):
            for ix in range(grid_x):
                code = codes[ix, iy, k]
                if code == 0:
                    continue
                d = decode_cell(code)
                cx = (ix + 0.5) * dx
                cy = (iy + 0.5) * dy
                px = cx + d['off_x'] * dx / 8.0
                py = cy + d['off_y'] * dy / 8.0
                z_bot = k * dz
                z_top = (k + 1) * dz
                if d['combo'] & COMBO_COL:
                    col_segs.append((round(px, 4), round(py, 4),
                                     d['col'], k, z_bot, z_top))
                if d['combo'] & COMBO_BX:
                    bx_segs.append((k, round(py, 4), d['bw'], d['bh'], ix))
                if d['combo'] & COMBO_BY:
                    by_segs.append((k, round(px, 4), d['bw'], d['bh'], iy))

    # 合并柱: 同 (x, y, section) 的连续 k 段合成一根整柱
    col_segs.sort()
    columns = []
    for seg in col_segs:
        px, py, sec, k, z_bot, z_top = seg
        if (columns and abs(columns[-1]['x'] - px) < 1e-6
                and abs(columns[-1]['y'] - py) < 1e-6
                and abs(columns[-1]['section'] - sec) < 1e-6
                and columns[-1]['_k_end'] + 1 == k):
            columns[-1]['_k_end'] = k
            columns[-1]['z_top'] = z_top
        else:
            columns.append({
                'x': px, 'y': py,
                'z_bottom': z_bot, 'z_top': z_top,
                'section': sec, 'floor': k + 1,
                '_k_end': k,
            })
    for c in columns:
        c.pop('_k_end', None)

    # 合并 X 梁: 同 (k, y, bw, bh) 的连续 ix 段合成一根
    bx_segs.sort()
    beam_x = []
    for seg in bx_segs:
        k, y, bw, bh, ix = seg
        _iy = min(max(int(round(y / dy)), 0), grid_y - 1)
        has_col = bool(decode_cell(codes[ix, _iy, k])['combo'] & COMBO_COL) \
            if codes.ndim == 3 else False
        if (beam_x and beam_x[-1]['_k'] == k
                and abs(beam_x[-1]['y'] - y) < 1e-6
                and beam_x[-1]['width'] == bw and beam_x[-1]['height'] == bh
                and beam_x[-1]['_ix_end'] + 1 == ix
                and not has_col):
            beam_x[-1]['_ix_end'] = ix
            beam_x[-1]['x1'] = (ix + 1) * dx if ix < grid_x - 1 else xmax
        else:
            beam_x.append({
                '_k': k, 'y': y,
                'x0': ix * dx, 'x1': (ix + 1) * dx if ix < grid_x - 1 else xmax,
                'width': bw, 'height': bh,
                'z': (k + 0.5) * dz,
                '_ix_end': ix,
            })
    for b in beam_x:
        b.pop('_ix_end', None)
        b.pop('_k', None)

    # 合并 Y 梁
    by_segs.sort()
    beam_y = []
    for seg in by_segs:
        k, x, bw, bh, iy = seg
        _ix = min(max(int(round(x / dx)), 0), grid_x - 1)
        has_col = bool(decode_cell(codes[_ix, iy, k])['combo'] & COMBO_COL) \
            if codes.ndim == 3 else False
        if (beam_y and beam_y[-1]['_k'] == k
                and abs(beam_y[-1]['x'] - x) < 1e-6
                and beam_y[-1]['width'] == bw and beam_y[-1]['height'] == bh
                and beam_y[-1]['_iy_end'] + 1 == iy
                and not has_col):
            beam_y[-1]['_iy_end'] = iy
            beam_y[-1]['y1'] = (iy + 1) * dy if iy < grid_y - 1 else ymax
        else:
            beam_y.append({
                '_k': k, 'x': x,
                'y0': iy * dy, 'y1': (iy + 1) * dy if iy < grid_y - 1 else ymax,
                'width': bw, 'height': bh,
                'z': (k + 0.5) * dz,
                '_iy_end': iy,
            })
    for b in beam_y:
        b.pop('_iy_end', None)
        b.pop('_k', None)

    return {
        'columns': columns,
        'beam_x': beam_x,
        'beam_y': beam_y,
        'n_floors': min(ns, max_floors),
    }



def _model_elements(frame):
    """从 frame 生成 elements (供 decode_frame_grid 用 frame 重建 model)."""
    from frame_model import build_nodes, build_elements
    nodes = build_nodes(frame)
    return build_elements(frame, nodes)


def _model_slabs(frame):
    """从 frame 生成 slabs (供 decode_frame_grid 用)."""
    from frame_model import build_slabs
    return build_slabs(frame)
