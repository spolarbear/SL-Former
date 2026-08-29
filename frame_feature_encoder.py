# frame_feature_encoder.py
"""
杆系结构特征编码器 (替代体素化/八叉树)

任务背景:
    预测屋顶顶点平均位移时程, 输入 = 结构 + 地震动加速度。
    这是明确的单自由度/多自由度动力响应问题, 结构整体响应由:
        刚度分布(柱截面/梁截面/层高/跨数), 质量分布(每层质量), 阻尼
    决定。体素化把精确的杆系构件信息粗粒度化(网格化), 丢失截面/质量细节。

本编码器直接从杆系框架模型 (frame_params) 提取物理量:
    1. 几何/拓扑: 层数, 跨数X/Y, 跨度X/Y, 层高, 总高, 节点数, 构件数
    2. 截面: 各层柱截面积A/I (从 col_sections), 梁截面
    3. 质量: 各层节点质量 -> 楼层质量分布 (插值到固定层数)
    4. 刚度: 各层柱抗侧刚度估计 (12EI/h^3), 层刚度分布
    5. 动力: 估算基频 (Ritz/悬臂近似), 阻尼比
    6. 平面: 楼面面积, 质量密度

输出: 固定长度物理特征向量 (供 MLP 编码), 或逐层序列 (供 Transformer 编码)。
"""
import numpy as np


def compute_story_stiffness(col_section, story_height, E=3.25e10, n_cols=1):
    """单柱抗侧刚度 k = 12EI/h^3 (两端刚接悬臂)"""
    I = col_section ** 4 / 12.0
    return 12.0 * E * I / (story_height ** 3) * n_cols


def estimate_fundamental_frequency(num_stories, story_height, col_sections,
                                   floor_masses, E=3.25e10):
    """
    用集中质量悬臂/框架的瑞利商估计基频。
    floor_masses: [num_stories] 每层总质量 (kg)
    返回: omega (rad/s), f (Hz)
    """
    n = num_stories
    if n == 0:
        return 0.0, 0.0
    # 简化: 假定剪切型框架, 各层刚度取平均柱抗侧刚度
    h = story_height
    k_floors = []
    for fl in range(n):
        col = col_sections[fl] if fl < len(col_sections) else col_sections[-1]
        k_floors.append(compute_story_stiffness(col, h))
    # 层间刚度串联 -> 顶层等效刚度 (近似剪切梁)
    # 采用集中质量模态法: 假设一阶振型为线性 (phi_i = i/n)
    phi = np.arange(1, n + 1) / n
    # 层间位移差
    delta = np.diff(np.concatenate([[0.0], phi]))
    K_eff = np.sum(1.0 / (np.array(k_floors) + 1e-9)) ** -1  # 串联刚度
    # 瑞利商: omega^2 = (phi^T K phi) / (phi^T M phi), K 用层间剪切
    m = np.array([floor_masses[fl] if fl < len(floor_masses) else 0.0
                  for fl in range(n)])
    # 剪切框架层间刚度串联后的广义刚度
    # omega^2 ~ (sum k_i) / (sum m_i) * (1/n^2)  (剪切梁一阶近似)
    k_sum = float(np.sum(k_floors))
    m_sum = max(float(np.sum(m)), 1e-9)
    omega2 = (k_sum / m_sum) * (1.0 / (n ** 2))
    omega = float(np.sqrt(max(omega2, 0.0)))
    return omega, omega / (2 * np.pi)


def extract_frame_features(frame_params, floor_node_masses=None, max_stories=12,
                           E=3.25e10):
    """
    从杆系框架模型提取结构化物理特征。

    Args:
        frame_params: generate_fixed_frame 返回的 dict (columns/beams/col_sections等)
        floor_node_masses: [num_stories] 每层节点质量 (kg)
        max_stories: 特征向量的固定楼层数 (不足补0, 超出截断)
        E: 混凝土弹性模量 (Pa)

    Returns:
        feat: np.float32 特征向量, 维度 = 定长
        info: dict 附带的物理量 (供解释/可视化)
    """
    columns = frame_params.get('columns', [])
    beams = frame_params.get('beams', [])
    col_sections = frame_params.get('col_sections', [])
    beam_w = frame_params.get('beam_width', 0.3)
    beam_h = frame_params.get('beam_height', 0.6)
    num_stories = int(frame_params.get('num_stories', 0))
    num_bays_x = int(frame_params.get('num_spans_x', 0))
    num_bays_y = int(frame_params.get('num_spans_y', 0))
    span_x = float(frame_params.get('span_x', 0.0))
    span_y = float(frame_params.get('span_y', 0.0))
    story_height = float(frame_params.get('story_height', 0.0))
    total_height = float(frame_params.get('total_height', 0.0))

    # 每层节点质量 (优先用传入的, 否则从构件质量估算)
    if floor_node_masses is not None and len(floor_node_masses) > 0:
        masses = np.asarray(floor_node_masses, dtype=np.float64)
    else:
        # 用总荷载/面积估算每层质量 (简化) — 用形状面积/节点数
        n_nodes_f = int(frame_params.get('shape_nodes_per_floor', (num_bays_x+1)*(num_bays_y+1)))
        area = float(frame_params.get('shape_area',
                      span_x * span_y * num_bays_x * num_bays_y))
        masses = np.full(num_stories, 20.0 * area * 1000 / 9.81 / max(1, n_nodes_f))
    if len(masses) < num_stories:
        masses = np.pad(masses, (0, num_stories - len(masses)))
    elif len(masses) > num_stories:
        masses = masses[:num_stories]

    # 每层柱截面积/惯性矩 (柱沿高度逐层)
    col_A = np.zeros(max_stories)
    col_I = np.zeros(max_stories)
    col_k = np.zeros(max_stories)   # 抗侧刚度
    n_cols_per_floor = max(1, int(frame_params.get('shape_nodes_per_floor',
                          (num_bays_x + 1) * (num_bays_y + 1))))
    for fl in range(num_stories):
        cs = col_sections[fl] if fl < len(col_sections) else (col_sections[-1] if col_sections else 0.4)
        A = cs ** 2
        I = cs ** 4 / 12.0
        k = compute_story_stiffness(cs, story_height, E, n_cols_per_floor)
        if fl < max_stories:
            col_A[fl] = A
            col_I[fl] = I
            col_k[fl] = k

    # 楼层质量分布 (插值/填充到 max_stories)
    mass_floor = np.zeros(max_stories)
    for fl in range(min(num_stories, max_stories)):
        mass_floor[fl] = masses[fl]

    # 每层总质量 (质量 x 节点数) — 按形状内节点数
    n_nodes_per_floor = n_cols_per_floor
    floor_total_mass = mass_floor * n_nodes_per_floor

    # 梁截面
    beam_A = beam_w * beam_h
    beam_I = beam_w * beam_h ** 3 / 12.0

    # 平面/体积 (按形状面积)
    floor_area = float(frame_params.get('shape_area',
                      span_x * span_y * num_bays_x * num_bays_y))
    total_mass = float(np.sum(floor_total_mass))

    # 基频估计
    omega, f_hz = estimate_fundamental_frequency(
        num_stories, story_height, col_sections, floor_total_mass, E)

    # ================================================================
    # 组装特征向量 (定长)
    # 分组: 全局标量(8) + 逐层柱刚度(12) + 逐层质量(12) + 逐层柱截面(12) + 拓扑(若干)
    # ================================================================
    feat = np.zeros(8 + 3 * max_stories, dtype=np.float32)

    # --- 全局标量 ---
    feat[0] = num_stories / 12.0                      # 层数 (归一)
    feat[1] = num_bays_x / 8.0                        # 跨数X
    feat[2] = num_bays_y / 8.0                        # 跨数Y
    feat[3] = span_x / 8.0                            # 跨度X
    feat[4] = span_y / 8.0                            # 跨度Y
    feat[5] = story_height / 5.0                      # 层高
    feat[6] = np.log10(total_mass + 1e-3) / 6.0       # 总质量 (对数)
    feat[7] = f_hz / 10.0                             # 基频

    # --- 逐层柱抗侧刚度 (对数) ---
    off = 8
    col_k_log = np.log10(col_k + 1e-6) / 12.0
    feat[off:off+max_stories] = col_k_log

    # --- 逐层总质量 (对数) ---
    off += max_stories
    mass_log = np.log10(floor_total_mass + 1e-3) / 7.0
    feat[off:off+max_stories] = mass_log

    # --- 逐层柱截面 (惯性矩对数) ---
    off += max_stories
    I_log = np.log10(col_I + 1e-12) / 12.0
    feat[off:off+max_stories] = I_log

    info = {
        'num_stories': num_stories,
        'num_bays_x': num_bays_x, 'num_bays_y': num_bays_y,
        'span_x': span_x, 'span_y': span_y, 'story_height': story_height,
        'total_height': total_height,
        'total_mass': total_mass, 'floor_area': floor_area,
        'fund_freq_hz': f_hz, 'omega': omega,
        'beam_A': beam_A, 'beam_I': beam_I,
        'n_columns': len(columns), 'n_beams': len(beams),
        'col_sections': list(col_sections),
        'floor_masses': floor_total_mass.tolist(),
    }
    return feat.astype(np.float32), info


def encode_frame_batch(params_list, floor_masses_list=None, max_stories=12,
                       E=3.25e10):
    """批量编码 (供 dataset 多进程/单进程使用)"""
    from generate_frames import generate_fixed_frame

    feats = []
    infos = []
    for i, p in enumerate(params_list):
        num_stories = int(p[0]); num_bays_x = int(p[1]); num_bays_y = int(p[2])
        bay_width_x = float(p[3]); bay_width_y = float(p[4])
        story_height = float(p[5])
        from generate_frames import id_to_shape
        plane_shape = id_to_shape(p[8]) if len(p) > 8 else 'rect'
        max_span = max(bay_width_x, bay_width_y)
        beam_height = max(0.4, min(max_span / 12, 0.8)); beam_height = round(beam_height / 0.2) * 0.2   # 200mm
        beam_width = max(0.2, min(beam_height / 2.5, 0.5)); beam_width = round(beam_width / 0.2) * 0.2  # 200mm
        frame = generate_fixed_frame(
            num_stories=num_stories, num_spans_x=num_bays_x, num_spans_y=num_bays_y,
            span_x=bay_width_x, span_y=bay_width_y, story_height=story_height,
            axis_ratio=0.6, beam_width=beam_width, beam_height=beam_height,
            plane_shape=plane_shape)
        fm = floor_masses_list[i] if floor_masses_list is not None else None
        feat, info = extract_frame_features(frame, fm, max_stories, E)
        feats.append(feat)
        infos.append(info)
    return np.array(feats, dtype=np.float32), infos


if __name__ == '__main__':
    from generate_frames import generate_fixed_frame
    frame = generate_fixed_frame(
        num_stories=3, num_spans_x=2, num_spans_y=2,
        span_x=6.0, span_y=6.0, story_height=3.5,
        axis_ratio=0.6, beam_width=0.3, beam_height=0.6)
    fm = np.array([10873.0] * 3)
    feat, info = extract_frame_features(frame, fm)
    print("特征维度:", feat.shape)
    print("基频: %.2f Hz, 总质量: %.1f t" % (info['fund_freq_hz'], info['total_mass']/1000))
    print("逐层刚度(log10):", np.round(feat[8:11], 3))
    print("逐层质量(log10):", np.round(feat[11:14], 3))
