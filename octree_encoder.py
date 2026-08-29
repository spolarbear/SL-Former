# octree_encoder.py
"""
3D 体素 + 八叉树紧缩编码器 (200mm 分辨率)

设计目标 (对应"梁板柱截面按 200mm 模数 / 300×300×500 @200mm / 紧缩深度 4~7"):
1. frame_to_voxel: 从杆系框架 (generate_fixed_frame) 生成 3D 体素矩阵
   - 空间 60×60×100 m @ 200mm 分辨率 = 300×300×500 网格
   - 梁/板/柱截面均为 200mm 整数倍, 恰好占据整数个立方体
   - 同时输出 质量体素 / 刚度体素 (构件材料属性)
2. OctreeBuilder: 八叉树紧缩 (深度 4~7 可调)
   - 深度 d → 2^d 网格; 深度5 = 32³
   - 每个格子统计它覆盖的原始构件空间的"体积代表值":
       质量 / 刚度 / 三向偏置(X,Y,Z质心) / 三向占比百分比
   - 7 通道: mass, stiffness, cx, cy, cz, fill_x, fill_y
   - 按 Z 分层聚合输出 (保留楼层语义, 维度可控)
3. PrecomputedOctreeEncoder: 模型结构编码器 (MLP, 保留, 模型依赖)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import zoom

# ============================================================
# 常量: 200mm 模数 / 材料
# ============================================================
MODULE = 0.2                      # 200mm 模数 (m)
CONCRETE_DENSITY = 2400.0         # 混凝土密度 (kg/m³)
E_CONCRETE = 3.25e10              # 弹性模量 (Pa)

# 空间范围 (与 config 一致)
SPACE_X = 60.0
SPACE_Y = 60.0
SPACE_Z = 100.0

# 体素标记值 (与 config 一致)
MARKER_COLUMN = 35.0
MARKER_BEAM = 32.0
MARKER_SLAB = 28.0


def snap_to_module(x, module=MODULE):
    """把尺寸吸附到 module 的整数倍 (200mm 模数), 并消除浮点误差"""
    n = round(float(x) / module)
    v = n * module
    # 消除 0.6000000000000001 类误差 (吸附到 0.01 精度)
    return round(v, 6)


# ============================================================
# 体素生成: frame -> 300×300×500 体素 (+ 质量/刚度体素)
# ============================================================
def _grid_dims(resolution=MODULE):
    """网格维度 300×300×500"""
    nx = int(round(SPACE_X / resolution))
    ny = int(round(SPACE_Y / resolution))
    nz = int(round(SPACE_Z / resolution))
    return nx, ny, nz


def _voxel_range(center, size, resolution):
    """按"中心 + 尺寸"计算体素覆盖范围 (半开区间 [lo, hi), 尺寸至少 1 格).

    用四舍五入 (floor(x+0.5) 而非 round()) 避免 Python 银行家舍入,
    且以中心对称展开, 从根本上消除贴边 (x=0 或 x=max) 柱因 round/浮点误差
    导致"最大边宽 1 格 / 最小边窄 1 格"的不对称问题。
    """
    inv = 1.0 / resolution
    c = int(center * inv + 0.5)     # 中心体素索引 (四舍五入)
    n = int(size * inv + 0.5)       # 覆盖格数 (四舍五入)
    if n < 1:
        n = 1
    lo = c - n // 2
    return lo, lo + n


def _box_indexes(x0, x1, y0, y1, z0, z1, shape, resolution):
    """把 (x0,x1,y0,y1,z0,z1) 盒体转为体素索引 (半开区间).

    越界处理: 若盒子部分超出网格 (如贴边柱心在 x=0, 半跨伸出域外),
    整体平移使其完整落回域内 (保持截面宽), 而不是截断成半个截面。
    """
    nx, ny, nz = shape
    i0, i1 = _voxel_range((x0 + x1) / 2.0, x1 - x0, resolution)
    j0, j1 = _voxel_range((y0 + y1) / 2.0, y1 - y0, resolution)
    k0, k1 = _voxel_range((z0 + z1) / 2.0, z1 - z0, resolution)
    # 整体平移回域内 (保持宽度), 而非截断
    if i0 < 0: i1 -= i0; i0 = 0
    if i1 > nx: i0 -= (i1 - nx); i1 = nx
    if j0 < 0: j1 -= j0; j0 = 0
    if j1 > ny: j0 -= (j1 - ny); j1 = ny
    if k0 < 0: k1 -= k0; k0 = 0
    if k1 > nz: k0 -= (k1 - nz); k1 = nz
    return i0, i1, j0, j1, k0, k1


def _fill_box(voxel, x0, x1, y0, y1, z0, z1, value, resolution):
    """在体素中填充一个轴对齐长方体 (索引裁剪到边界).

    用中心+尺寸对称算法计算索引, 保证: 任何位置/任何截面尺寸的构件
    宽度都恰好 = round(尺寸/体素) 格, 且以构件中心对称 (不再因 round()
    银行家舍入/浮点误差在网格边缘宽 1 格或偏移 1 格)。
    """
    i0, i1, j0, j1, k0, k1 = _box_indexes(x0, x1, y0, y1, z0, z1, voxel.shape, resolution)
    if i0 < i1 and j0 < j1 and k0 < k1:
        voxel[i0:i1, j0:j1, k0:k1] = value


def _fill_box_accum(arr, x0, x1, y0, y1, z0, z1, value, resolution):
    """向数组累加填充 (用于质量/刚度, 重叠区域求和). 与 _fill_box 同索引算法."""
    i0, i1, j0, j1, k0, k1 = _box_indexes(x0, x1, y0, y1, z0, z1, arr.shape, resolution)
    if i0 < i1 and j0 < j1 and k0 < k1:
        arr[i0:i1, j0:j1, k0:k1] += value
    return arr


def _fill_box_if_air(voxel, x0, x1, y0, y1, z0, z1, value, resolution):
    """仅当体素当前为空气 (<=0.1) 时写入标记值. (用于梁/板, 保证柱优先级)

    柱 > 梁 > 板: 后填充的构件不覆盖已存在的构件标记。
    返回被写入的体素数量 (供统计)。
    """
    i0, i1, j0, j1, k0, k1 = _box_indexes(x0, x1, y0, y1, z0, z1, voxel.shape, resolution)
    if i0 >= i1 or j0 >= j1 or k0 >= k1:
        return 0
    block = voxel[i0:i1, j0:j1, k0:k1]
    air_mask = block <= 0.1
    n_written = int(air_mask.sum())
    if n_written > 0:
        block[air_mask] = value
    return n_written


def frame_to_voxel(frame_params, floor_node_masses=None, resolution=MODULE):
    """从杆系框架生成 3D 体素 (标记值) + 质量体素 + 刚度体素。

    Args:
        frame_params: generate_fixed_frame 返回的 dict
            columns: (x, y, z_bottom, z_top, col_size)
            beams:   (x1, x2, y1, y2, z, width, height, direction)
            slabs:   (x0, xmax, y0, ymax, z_center, thickness)
        floor_node_masses: [num_stories] 每层节点质量 (kg), 可选。
            提供时, 把每层总质量 (每节点×节点数) 均摊到该层楼板体素上
            (楼面荷载对应的惯性质量), 叠加到混凝土自重的质量体素。
        resolution: 体素尺寸 (默认 0.2m = 200mm)

    Returns:
        (voxel, mass_voxel, stiff_voxel):
            voxel       : [nx,ny,nz] 构件标记 (柱/梁/板)
            mass_voxel  : [nx,ny,nz] 质量密度 (kg/m³)
            stiff_voxel : [nx,ny,nz] 刚度指数 (E·I 归一)
    """
    nx, ny, nz = _grid_dims(resolution)
    voxel = np.full((nx, ny, nz), 0.001, dtype=np.float32)   # 空气
    mass_voxel = np.zeros((nx, ny, nz), dtype=np.float32)
    # 三向抗弯刚度 E·I: stiff_voxel[0]=绕X轴, [1]=绕Y轴, [2]=绕Z轴
    stiff_voxel = np.zeros((3, nx, ny, nz), dtype=np.float32)

    columns = frame_params.get('columns', [])
    beams = frame_params.get('beams', [])
    slabs = frame_params.get('slabs', [])

    # ---- 柱: 立方体 col_size×col_size ----
    # 柱竖向杆件: 绕X轴与绕Y轴抗弯 I=cs^4/12; 绕Z轴(自身轴)不抗弯 I_z=0
    for (x, y, z_b, z_t, cs) in columns:
        cs = snap_to_module(cs)
        half = cs / 2.0
        _fill_box(voxel, x - half, x + half, y - half, y + half,
                  z_b, z_t, MARKER_COLUMN, resolution)
        mass_voxel = _fill_box_accum(
            mass_voxel, x - half, x + half, y - half, y + half,
            z_b, z_t, CONCRETE_DENSITY, resolution)
        I = cs ** 4 / 12.0
        stiff = E_CONCRETE * I
        stiff_voxel[0] = _fill_box_accum(
            stiff_voxel[0], x - half, x + half, y - half, y + half,
            z_b, z_t, stiff, resolution)
        stiff_voxel[1] = _fill_box_accum(
            stiff_voxel[1], x - half, x + half, y - half, y + half,
            z_b, z_t, stiff, resolution)
        # stiff_voxel[2] (绕Z轴) 柱不贡献抗弯

    # ---- 梁: width×height 截面 (宽度垂直于梁走向展开) ----
    # 梁沿X向: 竖向抗弯绕Y轴 I_y=w*h^3/12, 水平抗弯绕Z轴 I_z=h*w^3/12, I_x=0
    # 梁沿Y向: 竖向抗弯绕X轴 I_x=w*h^3/12, 水平抗弯绕Z轴 I_z=h*w^3/12, I_y=0
    for (x1, x2, y1, y2, z, w, h, direction) in beams:
        w = snap_to_module(w)
        h = snap_to_module(h)
        z_b = z - h / 2.0
        z_t = z + h / 2.0
        I_vert = w * h ** 3 / 12.0    # 竖向抗弯 (绕垂直于走向的水平轴)
        I_horiz = h * w ** 3 / 12.0   # 水平抗弯 (绕Z轴)
        if direction == 'x':
            # 沿X方向: 截面宽度在 Y 方向展开
            box_y0 = min(y1, y2) - w / 2.0
            box_y1 = max(y1, y2) + w / 2.0
            _fill_box_if_air(voxel, min(x1, x2), max(x1, x2),
                             box_y0, box_y1, z_b, z_t, MARKER_BEAM, resolution)
            mass_voxel = _fill_box_accum(
                mass_voxel, min(x1, x2), max(x1, x2),
                box_y0, box_y1, z_b, z_t, CONCRETE_DENSITY, resolution)
            # 绕Y轴 (竖向) + 绕Z轴 (水平)
            stiff_voxel[1] = _fill_box_accum(
                stiff_voxel[1], min(x1, x2), max(x1, x2),
                box_y0, box_y1, z_b, z_t, E_CONCRETE * I_vert, resolution)
            stiff_voxel[2] = _fill_box_accum(
                stiff_voxel[2], min(x1, x2), max(x1, x2),
                box_y0, box_y1, z_b, z_t, E_CONCRETE * I_horiz, resolution)
        else:
            # 沿Y方向: 截面宽度在 X 方向展开
            box_x0 = min(x1, x2) - w / 2.0
            box_x1 = max(x1, x2) + w / 2.0
            _fill_box_if_air(voxel, box_x0, box_x1,
                             min(y1, y2), max(y1, y2), z_b, z_t, MARKER_BEAM, resolution)
            mass_voxel = _fill_box_accum(
                mass_voxel, box_x0, box_x1,
                min(y1, y2), max(y1, y2), z_b, z_t, CONCRETE_DENSITY, resolution)
            # 绕X轴 (竖向) + 绕Z轴 (水平)
            stiff_voxel[0] = _fill_box_accum(
                stiff_voxel[0], box_x0, box_x1,
                min(y1, y2), max(y1, y2), z_b, z_t, E_CONCRETE * I_vert, resolution)
            stiff_voxel[2] = _fill_box_accum(
                stiff_voxel[2], box_x0, box_x1,
                min(y1, y2), max(y1, y2), z_b, z_t, E_CONCRETE * I_horiz, resolution)

    # ---- 板: 厚度 slab_thickness (只在空气处填, 不覆盖柱/梁) ----
    # 板平面构件: 绕X/Y/Z轴抗弯均 I=th^3/12
    for (x0, xm, y0, ym, zc, th) in slabs:
        th = snap_to_module(th)
        z_b = zc - th / 2.0
        z_t = zc + th / 2.0
        _fill_box_if_air(voxel, x0, xm, y0, ym, z_b, z_t, MARKER_SLAB, resolution)
        mass_voxel = _fill_box_accum(
            mass_voxel, x0, xm, y0, ym, z_b, z_t, CONCRETE_DENSITY, resolution)
        I_slab = 1.0 * th ** 3 / 12.0
        for d in range(3):
            stiff_voxel[d] = _fill_box_accum(
                stiff_voxel[d], x0, xm, y0, ym, z_b, z_t,
                E_CONCRETE * I_slab, resolution)

    # ---- 楼层质量均摊到楼板体素 (楼面荷载对应的惯性质量) ----
    # 若提供 floor_node_masses [每层每节点质量 kg], 计算每层总质量并均摊到
    # 该层楼板体素上 (质量密度叠加到混凝土自重之上)。
    # 语义: floor_node_masses[fl] × 节点数 = 该层楼面荷载总质量 (惯性质量)
    if floor_node_masses is not None and len(floor_node_masses) > 0:
        num_stories = frame_params.get('num_stories', len(floor_node_masses))
        num_spans_x = frame_params.get('num_spans_x', 1)
        num_spans_y = frame_params.get('num_spans_y', 1)
        # 形状内每层节点数 (T/L/C 比矩形少); 旧 frame 无此字段时退回矩形
        nodes_per_floor = int(frame_params.get(
            'shape_nodes_per_floor', (num_spans_x + 1) * (num_spans_y + 1)))
        story_height = frame_params.get('story_height', 3.5)
        # 收集每层楼板体素索引 (板标记)
        slab_mask = (voxel == MARKER_SLAB)
        for fl in range(min(num_stories, len(floor_node_masses))):
            if fl == 0:
                continue   # 首层 (底部) 通常无楼面荷载质量 (地面)
            floor_mass_total = float(floor_node_masses[fl]) * nodes_per_floor  # kg
            if floor_mass_total <= 0:
                continue
            # 该楼层对应的 Z 范围 (板 z_center ≈ (fl+0.5)*story_height 附近)
            z_mid = (fl + 0.5) * story_height
            z0 = int((z_mid - story_height / 2.0) / resolution)
            z1 = int((z_mid + story_height / 2.0) / resolution)
            z0 = max(0, z0); z1 = min(nz, z1)
            # 该层楼板体素数
            n_slab_vox = int(slab_mask[:, :, z0:z1].sum())
            if n_slab_vox <= 0:
                continue
            # 每体素叠加质量 = 楼层总质量 / 楼板体素数 (kg/体素)
            # 转换为密度: kg / (体素体积 m³) = kg / (resolution³)
            mass_per_voxel = floor_mass_total / n_slab_vox  # kg per voxel
            density_add = mass_per_voxel / (resolution ** 3)  # kg/m³
            # 均摊到该层楼板体素
            layer_slab = slab_mask[:, :, z0:z1]
            mass_voxel[:, :, z0:z1] += layer_slab * density_add

    return voxel, mass_voxel, stiff_voxel


# ============================================================
# 八叉树紧缩编码器
# ============================================================
class OctreeBuilder:
    """
    八叉树构建器 - 将 3D 体素紧缩为 2^depth 网格, 每格含体积代表值。

    深度 d → 2^d 网格; 深度5 = 32³; 深度4=16³; 深度6=64³; 深度7=128³.

    每个格子 (体素单元) 代表原始构件空间的体积统计, 10 通道 (按Z分层聚合):
        0 mass        : 质量 (归一)
        1 stiffness   : 刚度 E·I (归一)
        2 cx          : X方向质心偏置 ([-1,1], 质量加权)
        3 cy          : Y方向质心偏置 ([-1,1])
        4 cz          : Z方向质心偏置 ([-1,1])
        5 scx         : X方向刚心偏置 ([-1,1], 刚度加权)
        6 scy         : Y方向刚心偏置 ([-1,1])
        7 ecc         : 质心-刚心偏心距 (扭转指标)
        8 fill        : 该层填充占比
        9 aniso       : 方向各向异性 (X向刚度占比)
    """

    def __init__(self, max_depth=5):
        """
        max_depth: 最大紧缩深度 (5 = 32x32x32)
        """
        self.max_depth = max_depth
        self.target_size = 2 ** max_depth

    # ------------------------------------------------------------
    # 主入口: 体素 + 质量 + 刚度 -> 紧缩特征
    # ------------------------------------------------------------
    def build_features_v2(self, voxel, mass_voxel=None, stiff_voxel=None,
                          depth=None, agg_mode='z_layers'):
        """3D 八叉树紧缩编码。

        Args:
            voxel: [nx,ny,nz] 构件标记体素 (frame_to_voxel 输出)
            mass_voxel: [nx,ny,nz] 质量体素 (可选, 否则从标记推导)
            stiff_voxel: [nx,ny,nz] 刚度体素 (可选)
            depth: 紧缩深度 4~7 (默认 self.max_depth)
            agg_mode: 聚合方式
                - 'z_layers': 按Z分层聚合 (默认, 保留楼层语义)
                - 'flat'    : 全量展平 (维度巨大)

        Returns:
            np.ndarray 特征向量
        """
        depth = depth or self.max_depth
        if not (4 <= depth <= 7):
            raise ValueError(f"紧缩深度需在 4~7 之间, 收到 {depth}")
        target = 2 ** depth

        voxel = np.asarray(voxel, dtype=np.float32)
        voxel_bin = (voxel > 0.1).astype(np.float32)

        if mass_voxel is None:
            mass_voxel = np.where(voxel_bin > 0, CONCRETE_DENSITY, 0.0).astype(np.float32)
        if stiff_voxel is None:
            stiff_voxel = np.where(voxel_bin > 0, E_CONCRETE, 0.0).astype(np.float32)

        # ---- 降采样到 target³ ----
        # 复用与 downsample_for_viz 完全相同的逻辑 (分块聚合 + block_label +
        # 柱生长 + 按比例缩放): 不再用 zoom(order=0) 最近邻, 因为 zoom 会
        # 漏采样薄梁/板、劈开柱, 且与可视化不一致。
        # 同时计算每格柱/梁/板比例 (return_ratios), 供特征含构件比例。
        res = self._downsample_core(voxel, mass_voxel, stiff_voxel, target,
                                    return_ratios=True)
        voxel_d, mass_d, stiff_d, (z0, z1), ratio_d, code_d = res
        occ = voxel_d > 0.1

        if agg_mode == 'flat':
            return self._flat_features(voxel_d, mass_d, stiff_d, occ)
        if agg_mode == 'floor_code':
            return self._floor_code_features(code_d, voxel_d, target)
        return self._z_layer_features(voxel_d, mass_d, stiff_d, occ,
                                      ratio_d=ratio_d, code_d=code_d)

    # ------------------------------------------------------------
    # 楼层平面编码特征: 32×32×10, 每格一个16位编码数字
    # ------------------------------------------------------------
    def _floor_code_features(self, code_d, voxel_d, target, max_floors=10):
        """提取每个楼层的楼面平面 (含梁+板的层), 每格一个 16 位编码数字.

        输出: 展平向量, 维度 = target × target × 10 (每层楼面 32×32 个编码)。
        - 每个格子 = 一个 16 位编码 (0~65535), 不再分通道
        - 楼层平面从底到顶取"含梁+板"的层 (code & 7 == 7, 即 CBS)
        - 结构层数 < 10 用 0 填充; > 10 取底部 10 层
        """
        # 从 code_d 的 bit0-2 判断构件组合, 楼面候选 = 含梁(B=2)或板(S=4)的层
        # 再按楼层聚类: 相邻楼面层距 <=2 的合并为一层楼, 每层取构件最多的代表层
        candidate = []   # (z_layer, 构件数)
        for k in range(code_d.shape[2]):
            layer_code = code_d[:, :, k]
            combo = layer_code & 0x7
            has_floor = ((combo & 6) != 0).any()
            if has_floor:
                n_comp = int((combo & 6).sum())   # 梁+板构件格数
                candidate.append((k, n_comp))
        # 按 Z 聚类: 相邻候选层距 <= 2 视为同一楼层
        floor_codes = []
        for (k, n_comp) in candidate:
            if floor_codes and k - floor_codes[-1][0] <= 2:
                # 同一楼层: 保留构件数更多的层
                if n_comp > floor_codes[-1][1]:
                    floor_codes[-1] = (k, n_comp)
            else:
                floor_codes.append((k, n_comp))
        # 取楼面层编码平面 (底到顶), 每层楼 1 个, 最多 max_floors 层
        floor_layers = [k for (k, _) in floor_codes[:max_floors]]
        out = np.zeros((target, target, max_floors), dtype=np.int32)
        for f, k in enumerate(floor_layers):
            out[:, :, f] = code_d[:, :, k]
        return out.reshape(-1).astype(np.float32)

    # ------------------------------------------------------------
    # 可视化辅助: 复用 build_features_v2 的 Z裁剪+降采样逻辑, 返回降采样构件体素
    # ------------------------------------------------------------
    def downsample_for_viz(self, voxel, mass_voxel=None, stiff_voxel=None,
                           depth=None):
        """用与 build_features_v2 完全相同的 Z 裁剪 + 降采样逻辑,
        返回降采样后的 构件标记体素 / 质量体素 / 刚度体素 (target³)。

        供可视化直接使用 (不重新实现降采样逻辑)。

        Returns:
            (voxel_d, mass_d, stiff_d): [target,target,target] float32
                voxel_d 保留构件标记值 (柱35/梁32/板28/空气0)
        """
        depth = depth or self.max_depth
        if not (4 <= depth <= 7):
            raise ValueError(f"紧缩深度需在 4~7 之间, 收到 {depth}")
        target = 2 ** depth

        voxel = np.asarray(voxel, dtype=np.float32)
        voxel_bin = (voxel > 0.1).astype(np.float32)
        if mass_voxel is None:
            mass_voxel = np.where(voxel_bin > 0, CONCRETE_DENSITY, 0.0).astype(np.float32)
        if stiff_voxel is None:
            stiff_voxel = np.where(voxel_bin > 0, E_CONCRETE, 0.0).astype(np.float32)

        voxel_d, mass_d, stiff_d, (z0, z1) = self._downsample_core(
            voxel, mass_voxel, stiff_voxel, target)

        return voxel_d, mass_d, stiff_d, (z0, z1)

    # ------------------------------------------------------------
    # 降采样核心 (可视化与 ML 特征共用): Z裁剪 + 分块聚合 + 柱生长
    # ------------------------------------------------------------
    def _downsample_core(self, voxel, mass_voxel, stiff_voxel, target,
                         return_ratios=False):
        """降采样到 target³ 的核心逻辑 (供 build_features_v2 / downsample_for_viz 共用).

        1) Z 方向裁剪到结构实际高度 (含余量), 让 target 层全部用于表示结构
        2) 分块聚合: 构件标记用 block_label (柱绝对占优才判柱, 否则有梁判梁、
           有板判板 — 避免柱+梁混合块被 max 判成柱而把梁吞掉, 导致梁断开);
           质量取格内 mean; 刚度与几何同逻辑 (取该格判定构件的刚度均值)
        3) 柱标记单格化: 每层柱连通域合并到质心格, 刚度/质量同步转移
        4) 柱从底层向上生长: 每根柱用中心坐标 ÷ 每格大小 (round) 定位唯一
           目标格, 从底到顶连续生长 (穿过梁/板层), 位置按比例正确缩小

        Args:
            return_ratios: 若 True, 额外返回 ratio_d [target,target,target,3]
                (每格 柱/梁/板 体素占比) — 供 ML 特征含构件比例

        Returns:
            (voxel_d, mass_d, stiff_d, (z0, z1)): target³ 数组 + Z裁剪范围
            (return_ratios=True 时返回 5 元: + ratio_d)
        """
        voxel_bin = (voxel > 0.1).astype(np.float32)
        # ---- Z 方向裁剪到结构实际高度 (含余量) ----
        z_occ = voxel_bin.sum(axis=(0, 1))
        nz = voxel_bin.shape[2]
        zs = np.nonzero(z_occ > 0)[0]
        if len(zs) > 0:
            z0 = max(0, int(zs[0]) - 2)          # 底部留 2 格余量
            z1 = min(nz, int(zs[-1]) + 3)        # 顶部留 3 格余量
        else:
            z0, z1 = 0, nz

        voxel_c = voxel[:, :, z0:z1]
        mass_c = mass_voxel[:, :, z0:z1]
        # 三向刚度: stiff_voxel 可能为 4D [3,nx,ny,nz] (三向 E·I) 或 3D (旧单一刚度)
        stiff_is_3dir = (stiff_voxel.ndim == 4)
        if stiff_is_3dir:
            stiff_c = [stiff_voxel[d][:, :, z0:z1] for d in range(3)]
        else:
            stiff_c = [stiff_voxel[:, :, z0:z1]]

        # ---- 均匀网格分块 (每格固定覆盖 n_orig/target 个原始体素) ----
        # 用"坐标/每格大小"定位, 使柱/梁位置随分辨率按比例正确缩小
        # (每格代表原来 n_orig/target 分之一)。不用 _block_edges 的 round
        # 不均匀边界 (每格19,19,18格), 否则柱/梁位置会偏移 1 格。
        n_orig_x, n_orig_y, n_orig_z = voxel_c.shape
        cell_x = n_orig_x / float(target)
        cell_y = n_orig_y / float(target)
        cell_z = n_orig_z / float(target)

        def _coord_to_cell(coord, cell_size):
            """坐标(原始体素索引) -> 目标格 (均匀缩放, round)."""
            return min(max(int(round(coord / cell_size)), 0), target - 1)

        def _edges(n_orig, n_tgt, cell_size):
            """均匀网格边界: 每个目标格覆盖 [round(g*cell), round((g+1)*cell))"""
            edges = []
            prev = 0
            for g in range(n_tgt):
                hi = int(round((g + 1) * cell_size))
                if g == n_tgt - 1:
                    hi = n_orig
                hi = max(hi, prev + 1)
                edges.append((prev, hi))
                prev = hi
            return edges

        ex = _edges(n_orig_x, target, cell_x)
        ey = _edges(n_orig_y, target, cell_y)
        ez = _edges(n_orig_z, target, cell_z)

        def _block_label(bv):
            """块内构件标记 (柱>梁>板优先级, 但柱需绝对占优)."""
            n_col = int((bv == MARKER_COLUMN).sum())
            n_beam = int((bv == MARKER_BEAM).sum())
            n_slab = int((bv == MARKER_SLAB).sum())
            if n_col > n_beam + n_slab and n_col >= 3:
                return MARKER_COLUMN
            if n_beam > 0:
                return MARKER_BEAM
            if n_slab > 0:
                return MARKER_SLAB
            return 0.001

        # ---- 分块聚合 (向量化 bincount, 均匀网格) ----
        idx_x = np.zeros(n_orig_x, dtype=np.int64)
        for g, (lo_e, hi_e) in enumerate(ex):
            idx_x[lo_e:hi_e] = g
        idx_y = np.zeros(n_orig_y, dtype=np.int64)
        for g, (lo_e, hi_e) in enumerate(ey):
            idx_y[lo_e:hi_e] = g
        idx_z = np.zeros(n_orig_z, dtype=np.int64)
        for g, (lo_e, hi_e) in enumerate(ez):
            idx_z[lo_e:hi_e] = g
        ix, iy, iz = np.meshgrid(idx_x, idx_y, idx_z, indexing='ij')
        flat_idx = (ix * target * target + iy * target + iz).reshape(-1)
        N = target ** 3

        vc_f = voxel_c.reshape(-1)
        mc_f = mass_c.reshape(-1)
        fcol = (vc_f == MARKER_COLUMN).astype(np.float64)
        fbeam = (vc_f == MARKER_BEAM).astype(np.float64)
        fslab = (vc_f == MARKER_SLAB).astype(np.float64)
        n_col = np.bincount(flat_idx, weights=fcol, minlength=N)
        n_beam = np.bincount(flat_idx, weights=fbeam, minlength=N)
        n_slab = np.bincount(flat_idx, weights=fslab, minlength=N)
        m_sum = np.bincount(flat_idx, weights=mc_f.astype(np.float64), minlength=N)
        cnt = np.bincount(flat_idx, minlength=N)

        # ---- 每格质量加权质心 (X/Y/Z) 与质心偏位 ----
        # 原始体素坐标 (x,y,z) 与目标格索引 (ix,iy,iz)
        xx, yy, zz = np.meshgrid(
            np.arange(n_orig_x), np.arange(n_orig_y), np.arange(n_orig_z),
            indexing='ij')
        mc_w = mc_f.astype(np.float64)
        m_x = np.bincount(flat_idx, weights=(mc_w * xx.reshape(-1)), minlength=N)
        m_y = np.bincount(flat_idx, weights=(mc_w * yy.reshape(-1)), minlength=N)
        m_z = np.bincount(flat_idx, weights=(mc_w * zz.reshape(-1)), minlength=N)
        # 目标格中心坐标 (原始体素坐标)
        cell_cx = np.array([(lo_e + hi_e) / 2.0 for lo_e, hi_e in ex])
        cell_cy = np.array([(lo_e + hi_e) / 2.0 for lo_e, hi_e in ey])
        cell_cz = np.array([(lo_e + hi_e) / 2.0 for lo_e, hi_e in ez])
        # 每格质心 (质量加权) -> 偏位 (相对格中心, 归一化到格尺寸)
        denom_m = np.maximum(m_sum, 1e-12)
        # 质心偏位 (各方向, 单位=格)
        off_x = np.zeros(N, dtype=np.float64)
        off_y = np.zeros(N, dtype=np.float64)
        off_z = np.zeros(N, dtype=np.float64)
        for g in range(N):
            if m_sum[g] <= 0:
                continue
            i, j, k = np.unravel_index(g, (target, target, target))
            cxm = m_x[g] / denom_m[g]
            cym = m_y[g] / denom_m[g]
            czm = m_z[g] / denom_m[g]
            off_x[g] = (cxm - cell_cx[i]) / cell_x
            off_y[g] = (cym - cell_cy[j]) / cell_y
            off_z[g] = (czm - cell_cz[k]) / cell_z

        # ---- 16位格子编码 (0~65535) ----
        # bit0-2 : 构件组合 (C=1, B=2, S=4)
        # bit3-7 : 填充量级 (该格构件体素数, 5位 0-31)
        # bit8-10: 质量量级 (该格总质量, 3位 0-7)
        # bit11-13: 质心偏位量级 (|偏移|最大分量, 3位 0-7)
        # bit14-15: 预留
        # 空气格 (无任何构件): 编码 = 0
        n_fill = n_col + n_beam + n_slab   # 构件体素数 (不含空气)
        def _level5(x):
            # 填充量级: 0~31 分段 (对数)
            return np.minimum(np.log2(np.maximum(x, 1) + 1).astype(np.int64), 31)
        def _level3(x, scale):
            # 通用 3位量级: x>0 时按对数分段到 0-7
            return np.minimum((np.log2(np.maximum(x, 1) + 1) / scale).astype(np.int64), 7)
        fill_lvl = _level5(n_fill)
        mass_lvl = _level3(m_sum, 10.0)
        off_mag = np.maximum(np.maximum(np.abs(off_x), np.abs(off_y)), np.abs(off_z))
        off_lvl = np.minimum((off_mag * 8).astype(np.int64), 7)
        code = (n_col > 0).astype(np.int64) * 1 \
             + (n_beam > 0).astype(np.int64) * 2 \
             + (n_slab > 0).astype(np.int64) * 4
        code = code | (fill_lvl << 3) | (mass_lvl << 8) | (off_lvl << 11)
        # 空气格 (无构件) 编码归零
        empty = (n_fill <= 0)
        code[empty] = 0

        label = np.full(N, 0.001, dtype=np.float32)
        is_col = (n_col > n_beam + n_slab) & (n_col >= 3)
        is_beam = (~is_col) & (n_beam > 0)
        is_slab = (~is_col) & (~is_beam) & (n_slab > 0)
        label[is_col] = MARKER_COLUMN
        label[is_beam] = MARKER_BEAM
        label[is_slab] = MARKER_SLAB

        m_mean = np.zeros(N, dtype=np.float32)
        np.divide(m_sum, np.maximum(cnt, 1), out=m_mean, where=cnt > 0)
        # 刚度: 每个方向分别聚合 (只对该格判定构件的体素求均值)
        n_stiff = len(stiff_c)
        s_mean = np.zeros((n_stiff, N), dtype=np.float32)
        for d in range(n_stiff):
            sc_f = stiff_c[d].reshape(-1)
            for lab_val in (MARKER_COLUMN, MARKER_BEAM, MARKER_SLAB):
                m = (vc_f == lab_val).astype(np.float64)
                s_lab = np.bincount(flat_idx, weights=sc_f.astype(np.float64) * m, minlength=N)
                c_lab = np.bincount(flat_idx, weights=m, minlength=N)
                np.divide(s_lab, np.maximum(c_lab, 1), out=s_mean[d],
                          where=(c_lab > 0) & (label == lab_val))

        voxel_d = label.reshape(target, target, target)
        mass_d = m_mean.reshape(target, target, target)
        if n_stiff == 3:
            stiff_d = s_mean.reshape(3, target, target, target)   # [3,t,t,t]
        else:
            stiff_d = s_mean.reshape(target, target, target)       # [t,t,t]

        # ---- 每格 柱/梁/板 比例 (可选返回, 供 ML 特征含构件比例) ----
        ratio_d = None
        code_d = None
        if return_ratios:
            denom = np.maximum(cnt, 1)
            ratio_d = np.stack([
                (n_col / denom).astype(np.float32),
                (n_beam / denom).astype(np.float32),
                (n_slab / denom).astype(np.float32),
            ], axis=-1).reshape(target, target, target, 3)
            code_d = code.reshape(target, target, target).astype(np.int32)

        # ---- 柱标记单格化 (分块劈开 → 质心格) ----
        # stiff_d 可能为 [3,t,t,t] (三向) 或 [t,t,t] (单一)
        stiff_ndim = stiff_d.ndim
        n_stiff = stiff_d.shape[0] if stiff_ndim == 4 else 1
        from scipy import ndimage as _ndi
        for k in range(target):
            sl = voxel_d[:, :, k]
            mask = (sl == MARKER_COLUMN)
            if not mask.any():
                continue
            lab, ncomp = _ndi.label(mask.astype(np.int32))
            if ncomp < 1:
                continue
            col_s = np.zeros((n_stiff, ncomp + 1), dtype=np.float32)
            col_m = np.zeros(ncomp + 1, dtype=np.float32)
            col_n = np.zeros(ncomp + 1, dtype=np.int32)
            for (ii, jj) in np.argwhere(mask):
                c = lab[ii, jj]
                for d in range(n_stiff):
                    if stiff_ndim == 4:
                        col_s[d, c] += float(stiff_d[d, ii, jj, k])
                    else:
                        col_s[d, c] += float(stiff_d[ii, jj, k])
                col_m[c] += float(mass_d[ii, jj, k])
                col_n[c] += 1
                voxel_d[ii, jj, k] = 0.001
                for d in range(n_stiff):
                    if stiff_ndim == 4:
                        stiff_d[d, ii, jj, k] = 0.0
                    else:
                        stiff_d[ii, jj, k] = 0.0
                mass_d[ii, jj, k] = 0.0
            for c in range(1, ncomp + 1):
                coords = np.argwhere(lab == c)
                cy = int(round(coords[:, 0].mean()))
                cx = int(round(coords[:, 1].mean()))
                cy = min(max(cy, 0), target - 1)
                cx = min(max(cx, 0), target - 1)
                voxel_d[cy, cx, k] = MARKER_COLUMN
                if col_n[c] > 0:
                    for d in range(n_stiff):
                        if stiff_ndim == 4:
                            stiff_d[d, cy, cx, k] = col_s[d, c] / col_n[c]
                        else:
                            stiff_d[cy, cx, k] = col_s[d, c] / col_n[c]
                    mass_d[cy, cx, k] = col_m[c] / col_n[c]

        # ---- 柱从底层向上生长 (穿过梁/板层不打断 + 按比例缩放定位) ----
        # 需求: 柱子应连续从底层生长到顶, 中间楼层(梁+板层)即使柱体素不占优
        #       也不应被梁/板标记打断; 顶层细柱完整。
        # 实现: 用原始体素 voxel_c 的柱体素做 3D 连通域, 每根柱用其中心坐标
        #       (体素质心) ÷ 每格大小 (round) 确定唯一目标格 — 柱位置随分辨率
        #       按比例正确缩小 (每格代表原来 n_orig/target 分之一)。
        #       然后从底部到该柱顶部连续生长 (竖直贯穿, 穿过梁/板层)。
        col_orig = (voxel_c == MARKER_COLUMN)
        if col_orig.any():
            from scipy import ndimage as _ndi
            lab3d, ncol = _ndi.label(col_orig.astype(np.int32))
            # 清除现有柱标记 (用生长结果完全替代, 避免劈开残留)
            voxel_d[voxel_d == MARKER_COLUMN] = 0.001
            if stiff_ndim == 4:
                for d in range(3):
                    stiff_d[d][voxel_d == 0.001] = 0.0
            else:
                stiff_d[voxel_d == 0.001] = 0.0
            mass_d[voxel_d == 0.001] = 0.0
            for c in range(1, ncol + 1):
                coords = np.argwhere(lab3d == c)   # [N,3] (x,y,z)
                if len(coords) == 0:
                    continue
                # 柱中心 (体素质心) -> 目标格 (均匀缩放 round)
                cx_v = float(coords[:, 0].mean())
                cy_v = float(coords[:, 1].mean())
                gi = _coord_to_cell(cx_v, cell_x)
                gj = _coord_to_cell(cy_v, cell_y)
                # 顶部原始 z 层 -> 目标层 (均匀缩放)
                z_top_v = int(coords[:, 2].max())
                k_top = _coord_to_cell(z_top_v, cell_z)
                # 从底部到 k_top 连续生长 (穿过梁/板层)
                for k in range(0, k_top + 1):
                    voxel_d[gi, gj, k] = MARKER_COLUMN
                    if stiff_ndim == 4:
                        for d in range(3):
                            if stiff_d[d, gi, gj, k] <= 0:
                                stiff_d[d, gi, gj, k] = float(E_CONCRETE * 0.4 ** 4 / 12.0)
                    else:
                        if stiff_d[gi, gj, k] <= 0:
                            stiff_d[gi, gj, k] = float(E_CONCRETE * 0.4 ** 4 / 12.0)
                    if mass_d[gi, gj, k] <= 0:
                        mass_d[gi, gj, k] = float(CONCRETE_DENSITY)

        if return_ratios:
            return voxel_d, mass_d, stiff_d, (z0, z1), ratio_d, code_d
        return voxel_d, mass_d, stiff_d, (z0, z1)


    # ------------------------------------------------------------
    # 按 Z 分层聚合 (保留楼层语义, 维度可控)
    # ------------------------------------------------------------
    def _z_layer_features(self, voxel_d, mass_d, stiff_d, occ, ratio_d=None,
                          code_d=None):
        """对 target³ 网格按 Z 分层聚合, 每层输出统计特征。

        通道 (随输入动态增减):
            0 mass          : 本层总质量 (归一)
            1..n_stiff      : 本层各方向刚度 E·I (归一) (1 或 3 个方向)
            +2 cx           : 质量加权质心 X 偏置 [-1,1]
            +3 cy           : 质量加权质心 Y 偏置 [-1,1]
            +4 cz           : 质量加权质心 Z 偏置 [-1,1]
            +5 scx          : 刚度加权中心 X 偏置 [-1,1] (刚心)
            +6 scy          : 刚度加权中心 Y 偏置 [-1,1] (刚心)
            +7 ecc          : 质心-刚心偏心距 (扭转指标)
            +8 fill         : 本层被占用格子占比
            +9 aniso        : 方向各向异性 (X向刚度占比)
            +10 col_ratio   : 本层柱体素占比 (若 ratio_d)
            +11 beam_ratio  : 本层梁体素占比 (若 ratio_d)
            +12 slab_ratio  : 本层板体素占比 (若 ratio_d)
            +13..20         : 8 种格子组合类型频率 (空/C/B/S/CB/CS/BS/CBS) (若 code_d)

        深度5: 三向刚度+比例+编码 → 32×(1+3+9+8) = 32×21 = 672维
        """
        target = voxel_d.shape[0]
        stiff_4d = (stiff_d.ndim == 4)
        n_stiff = stiff_d.shape[0] if stiff_4d else 1
        n_feat = 1 + n_stiff + 9
        if ratio_d is not None:
            n_feat += 3
        if code_d is not None:
            n_feat += 8
        feats = np.zeros((target, n_feat), dtype=np.float32)

        m_max = float(mass_d.max()) if mass_d.max() > 0 else 1.0
        s_max = float(stiff_d.max()) if stiff_d.max() > 0 else 1.0
        coords = np.arange(target) / target * 2.0 - 1.0  # [-1,1]
        Xc, Yc = np.meshgrid(coords, coords, indexing='ij')
        area = float(target * target)

        for k in range(target):
            layer_m = mass_d[:, :, k]
            if stiff_4d:
                layer_s_list = [stiff_d[d][:, :, k] for d in range(n_stiff)]
                layer_s = layer_s_list[0]
            else:
                layer_s = stiff_d[:, :, k]
            layer_o = occ[:, :, k]

            m_total = float(layer_m.sum())
            s_total = float(layer_s.sum())
            n_occ = float(layer_o.sum())

            c = 0
            feats[k, c] = m_total / (m_max * area + 1e-12)   # 0 mass
            c += 1
            # 刚度 (1 或 3 个方向)
            for d in range(n_stiff):
                s_l = layer_s_list[d] if stiff_4d else layer_s
                feats[k, c + d] = float(s_l.sum()) / (s_max * area + 1e-12)
            c += n_stiff
            # 质量加权质心
            if m_total > 1e-9:
                feats[k, c] = float((layer_m * Xc).sum()) / m_total     # cx
                feats[k, c+1] = float((layer_m * Yc).sum()) / m_total   # cy
                feats[k, c+2] = coords[k]                               # cz
            c += 3
            # 刚度加权中心 (刚心) + 偏心距
            if s_total > 1e-9:
                scx = float((layer_s * Xc).sum()) / s_total
                scy = float((layer_s * Yc).sum()) / s_total
                feats[k, c] = scx
                feats[k, c+1] = scy
                feats[k, c+2] = np.hypot(feats[k, c-3] - scx, feats[k, c-2] - scy) / 2.0
            c += 3
            # 填充占比
            feats[k, c] = n_occ / area
            c += 1
            # 方向各向异性: X向刚度占比
            if s_total > 1e-9:
                x_stiff = float(layer_s.sum(axis=1).max())
                feats[k, c] = x_stiff / (s_total + 1e-12)
            c += 1
            # 柱/梁/板 比例
            if ratio_d is not None:
                rl = ratio_d[:, :, k, :]
                n_vox = float(rl.sum())
                if n_vox > 0:
                    feats[k, c] = float(rl[:, :, 0].sum()) / n_vox   # col_ratio
                    feats[k, c+1] = float(rl[:, :, 1].sum()) / n_vox  # beam_ratio
                    feats[k, c+2] = float(rl[:, :, 2].sum()) / n_vox  # slab_ratio
                c += 3
            # 8 种格子组合类型频率 (空/C/B/S/CB/CS/BS/CBS)
            if code_d is not None:
                layer_code = code_d[:, :, k]
                combo_bits = layer_code & 0x7   # bit0-2: C/B/S
                for combo in range(8):
                    freq = float((combo_bits == combo).sum()) / (target * target)
                    feats[k, c + combo] = freq
                c += 8

        return feats.astype(np.float32)

    # ------------------------------------------------------------
    # 平铺模式 (可选, 维度巨大: target³×7)
    # ------------------------------------------------------------
    def _flat_features(self, voxel_d, mass_d, stiff_d, occ):
        target = voxel_d.shape[0]
        m_max = float(mass_d.max()) if mass_d.max() > 0 else 1.0
        s_max = float(stiff_d.max()) if stiff_d.max() > 0 else 1.0
        coords = np.arange(target) / target * 2.0 - 1.0

        mass_n = mass_d / (m_max + 1e-12)
        stiff_n = stiff_d / (s_max + 1e-12)
        cx = np.zeros_like(mass_d)
        cy = np.zeros_like(mass_d)
        for k in range(target):
            w = mass_d[:, :, k]
            tot = w.sum() + 1e-12
            cx[:, :, k] = (w * coords.reshape(-1, 1)).sum(axis=0) / tot
            cy[:, :, k] = (w * coords.reshape(1, -1)).sum(axis=1) / tot
        cz = np.broadcast_to(coords.reshape(1, 1, -1), mass_d.shape)

        stack = np.stack([mass_n, stiff_n, cx, cy, cz, occ, occ], axis=-1)
        return stack.reshape(-1).astype(np.float32)

    # ------------------------------------------------------------
    # 兼容旧接口 (build_features: 旧 Z 投影, 保留供 pkl 模式)
    # ------------------------------------------------------------
    def build_features(self, voxel, threshold=0.1, floor_node_masses=None):
        """旧接口: 从体素提取 4*2^depth 维 (兼容历史调用)"""
        target = self.target_size
        if voxel.ndim == 4:
            return np.array([
                self._extract_features(
                    voxel[i].numpy() if torch.is_tensor(voxel) else voxel[i],
                    threshold)
                for i in range(voxel.shape[0])], dtype=np.float32)
        return self._extract_features(
            voxel.numpy() if torch.is_tensor(voxel) else voxel, threshold)

    def _extract_features(self, voxel, threshold=0.1):
        """旧实现: Z方向分层投影 (密度+X偏+Y偏, 兼容保留)"""
        target = self.target_size
        zf = np.array([target / s for s in voxel.shape])
        voxel_down = zoom(voxel, zf, order=0)
        voxel_binary = (voxel_down > threshold).astype(np.float32)

        layer_density = voxel_binary.mean(axis=(0, 1))
        layer_count = voxel_binary.sum(axis=(0, 1)) + 1e-6

        x_coords = (np.arange(target) / target * 2 - 1).reshape(-1, 1, 1)
        x_com = (voxel_binary * x_coords).sum(axis=(0, 1)) / layer_count
        x_com = np.nan_to_num(x_com, nan=0.0)

        y_coords = (np.arange(target) / target * 2 - 1).reshape(1, -1, 1)
        y_com = (voxel_binary * y_coords).sum(axis=(0, 1)) / layer_count
        y_com = np.nan_to_num(y_com, nan=0.0)

        return np.concatenate([layer_density, x_com, y_com], axis=0).astype(np.float32)


# ============================================================
# 预计算版本: 直接从缓存读取八叉树特征，无需实时构建体素
# (模型使用本版本)
# ============================================================

class PrecomputedOctreeEncoder(nn.Module):
    """
    预计算八叉树特征 - 直接从缓存读取，无需实时构建
    适合在dataset中预计算并存储
    """
    
    def __init__(self, output_dim=64, max_depth=5, input_dim=None):
        super().__init__()
        self.max_depth = max_depth
        self.target_size = 2 ** max_depth
        # 特征维度: 7 * 2^max_depth (质量+刚度+三向偏置+占比) 或由调用方指定
        if input_dim is None:
            input_dim = 7 * self.target_size
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, octree_features):
        """
        octree_features: [batch, 10*2^depth] 预计算的八叉树特征
        返回: [batch, output_dim]
        """
        return self.encoder(octree_features)


class VoxelTokenEncoder(nn.Module):
    """
    体素 token 编码器 (LLM embedding 思想).

    输入: [batch, 32³] 每格一个离散微元 token ID (0=空, 1..V-1=微元类型)
    流程:
      1. nn.Embedding(V, embed_dim, padding_idx=0): 每格 token → 向量 (可学习)
      2. 平均池化所有非空格子的 embedding (mask 掉空)
      3. 附加统计: 非空格数 / 总格数 (结构体量信息)
      4. MLP 投影到 output_dim
    与 PrecomputedOctreeEncoder 输出维度一致 (可无缝替换 struct_encoder).

    物理初始化 (init_with_physics=True / physics_mode, 传入 vocab):
      - 用每个 token 的物理向量 (rich8 或 hexa9 或 basic5) PCA 降到 embed_dim,
        作为 nn.Embedding 的初始权重。
      - 目的: "刚度/截面相似"的微元 token 在 embedding 空间中初始即邻近,
        让模型在训练早期就利用物理相似性 (训练中继续微调).
      - physics_mode 可选:
          'random' : 纯随机初始化 (无物理先验)
          'rich8'  : rich 8 维物理向量 (类型/柱EI/梁EI/面积/偏位/填充)
          'hexa9'  : 六面体刚度 9 维 (3 对对面 剪切GA+抗弯EI + 类型/填充/偏位)
          'basic5' : 精简 5 维 (类型/柱EI/梁EI/偏位)
    """

    def __init__(self, vocab_size, output_dim=64, embed_dim=32, grid=32,
                 init_with_physics=False, vocab=None, physics_mode=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.grid = grid
        self.n_tokens = grid ** 3
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 物理向量初始化: 刚度/截面相似的 token 向量初始邻近
        if physics_mode is None:
            physics_mode = 'rich8' if init_with_physics else 'random'
        if physics_mode != 'random':
            try:
                if vocab is not None and hasattr(vocab, 'physics_embeddings'):
                    pe = vocab.physics_embeddings(embed_dim=embed_dim,
                                                  mode=physics_mode)
                    if pe is not None and pe.shape[0] >= self.vocab_size:
                        self.embedding.weight.data.copy_(
                            torch.from_numpy(pe[:self.vocab_size]))
                        print(f"  🧲 体素 token embedding 用物理向量初始化 "
                              f"({self.vocab_size}×{embed_dim}, mode={physics_mode}: "
                              f"刚度/截面相似 -> 邻近)")
                else:
                    print("  [W] 物理向量初始化: 未提供 vocab, 保持随机初始化")
            except Exception as e:
                print(f"  [W] 物理向量初始化跳过: {e} (保持随机初始化)")
        self.agg = nn.Sequential(
            nn.Linear(embed_dim + 1, 128),
            nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, output_dim), nn.LayerNorm(output_dim),
        )

    def forward(self, token_ids):
        """
        token_ids: [batch, 32³] token ID (float 或 int, 内部转 long)
        返回: [batch, output_dim]
        """
        token_ids = token_ids.long()          # Embedding 需要 Long/Int
        # 防御: 非法 token (>=vocab_size 或 <0) 归为 0 (padding), 避免 CUDA 越界
        token_ids = token_ids.clamp(min=0)
        token_ids = torch.where(token_ids < self.vocab_size, token_ids,
                                torch.zeros_like(token_ids))
        # [B, N] -> [B, N, d]
        emb = self.embedding(token_ids)
        mask = (token_ids > 0).float().unsqueeze(-1)   # [B, N, 1]
        nz = mask.sum(dim=1).clamp(min=1.0)            # [B, 1]
        pooled = (emb * mask).sum(dim=1) / nz          # [B, d] 平均池化
        # 附加: 非空格占比 (结构体量)
        ratio = (mask.sum(dim=1) / float(self.n_tokens))  # [B, 1]
        x = torch.cat([pooled, ratio], dim=-1)         # [B, d+1]
        return self.agg(x)


# ============================================================
# 测试
# ============================================================

if __name__ == '__main__':
    from generate_frames import generate_fixed_frame

    print("=" * 60)
    print("3D 体素 + 八叉树紧缩编码器测试")
    print("=" * 60)

    # ---- 生成框架 ----
    frame = generate_fixed_frame(
        num_stories=5, num_spans_x=2, num_spans_y=2,
        span_x=6.0, span_y=6.0, story_height=3.5,
        axis_ratio=0.6, beam_width=0.3, beam_height=0.6)
    print("框架: 5层 2×2跨 6m")
    print("柱截面:", frame['col_sections'])

    # ---- 体素生成 ----
    voxel, mass_v, stiff_v = frame_to_voxel(frame)
    print(f"\n体素: {voxel.shape} (300×300×500 @ 200mm)")
    print(f"构件非空比例: {(voxel>0.1).mean():.4f}")
    print(f"质量体素非零: {(mass_v>0).mean():.4f}, 刚度体素非零: {(stiff_v>0).mean():.4f}")

    # ---- 各深度紧缩 ----
    print("\n各紧缩深度特征维度:")
    for depth in [4, 5, 6, 7]:
        builder = OctreeBuilder(max_depth=depth)
        feat = builder.build_features_v2(voxel, mass_v, stiff_v)
        print(f"  深度{depth} (2^{depth}={2**depth}³): {feat.shape} "
              f"(每层{feat.shape[1]}特征×{2**depth}层)")

    # ---- 平铺模式 (仅深度4测试) ----
    builder = OctreeBuilder(max_depth=4)
    flat = builder.build_features_v2(voxel, mass_v, stiff_v, agg_mode='flat')
    print(f"\n平铺模式 (深度4): {flat.shape} (16³×7)")

    # ---- 预计算编码器 ----
    enc = PrecomputedOctreeEncoder(output_dim=64, max_depth=5)
    dummy = torch.randn(4, 7 * 32)
    out = enc(dummy)
    print(f"\nPrecomputedOctreeEncoder: {dummy.shape} -> {out.shape}")
    print(f"参数量: {sum(p.numel() for p in enc.parameters()):,}")