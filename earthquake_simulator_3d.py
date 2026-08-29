# earthquake_simulator_3d.py (完整修复版)
import numpy as np
import os
import glob
from scipy.interpolate import interp1d
import warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings('ignore')

try:
    import openseespy.opensees as ops
except ImportError:
    print("警告: openseespy未安装，请安装: pip install openseespy")
    ops = None

# ================================================================
# 配置参数
# ================================================================

class SimConfig3D:
    """3D仿真配置"""
    
    WINDOW_BEFORE = 2.5
    WINDOW_AFTER = 2.5
    TARGET_DT = 0.01
    TARGET_PGA = 0.035
    
    E = 3.25e10
    NU = 0.2
    RHO = 2400
    
    COL_SECTIONS = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.38, 0.35, 0.3]  # 12层逐层递减
    BEAM_SECTION = (0.3, 0.6)
    SLAB_THICKNESS = 0.12
    
    PARAM_RANGES = {
        'num_stories': [3, 12],
        'num_bays_x': [1, 8],
        'num_bays_y': [1, 8],
        'bay_width_x': [4.0, 8.0],
        'bay_width_y': [4.0, 8.0],
        'story_height': [3.0, 4.5],
        'mass_per_node': [10000, 60000],
        'damping_ratio': [0.03, 0.06]
    }
    
    @classmethod
    def get_param_dim(cls):
        return len(cls.PARAM_RANGES)
    
    @classmethod
    def get_seq_len(cls):
        return int((cls.WINDOW_BEFORE + cls.WINDOW_AFTER) / cls.TARGET_DT)


# ================================================================
# 地震波加载器
# ================================================================

class EarthquakeLoader3D:
    def __init__(self, config=None):
        self.config = config or SimConfig3D()
        self.seq_len = self.config.get_seq_len()
    
    def extract_peak_window(self, motion, dt=0.01, window_before=2.5, window_after=2.5):
        n_steps = int((window_before + window_after) / dt)
        peak_idx = np.argmax(np.abs(motion))
        
        start_idx = max(0, peak_idx - int(window_before / dt))
        end_idx = min(len(motion), peak_idx + int(window_after / dt))
        
        windowed = motion[start_idx:end_idx]
        
        if len(windowed) < n_steps:
            pad_left = max(0, int(window_before / dt) - (peak_idx - start_idx))
            pad_right = n_steps - len(windowed) - pad_left
            windowed = np.pad(windowed, (pad_left, pad_right), 'constant')
        
        if len(windowed) > n_steps:
            excess = len(windowed) - n_steps
            crop_left = excess // 2
            crop_right = excess - crop_left
            windowed = windowed[crop_left:len(windowed)-crop_right]
        
        return windowed
    
    def load_earthquake_files(self, folder_path, target_pga=0.035, dt=0.01,
                              window_before=2.5, window_after=2.5, max_files=200):
        if not os.path.exists(folder_path):
            return np.array([]), 0
        
        patterns = ['*.txt', '*.dat', '*.csv', '*.at2']
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(folder_path, pattern)))
        files = list(set(files))
        files = [f for f in files if os.path.isfile(f)]
        
        if len(files) == 0:
            return np.array([]), 0
        
        all_motions = []
        all_names = []
        target_steps = int((window_before + window_after) / dt)
        
        for file_path in files[:max_files]:
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                values = []
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    for part in line.split():
                        try:
                            values.append(float(part))
                        except:
                            continue
                
                if len(values) < 10:
                    continue
                
                motion_orig = np.array(values)
                t_orig = np.arange(len(motion_orig)) * 0.01
                t_target = np.arange(0, t_orig[-1], dt)
                f = interp1d(t_orig, motion_orig, kind='linear', fill_value='extrapolate')
                motion_resampled = f(t_target)
                
                pga = np.max(np.abs(motion_resampled))
                if pga > 1e-10:
                    motion_scaled = motion_resampled / pga * target_pga
                else:
                    motion_scaled = motion_resampled
                
                windowed = self.extract_peak_window(motion_scaled, dt, window_before, window_after)
                
                if len(windowed) == target_steps:
                    all_motions.append(windowed)
                    # 用源文件名 (去扩展名) 作为波名, 便于数据库识别
                    base = os.path.basename(file_path)
                    name = os.path.splitext(base)[0]
                    all_names.append(name)
            except:
                continue
        
        if len(all_motions) == 0:
            return np.array([]), [], 0
        
        return np.array(all_motions), all_names, len(all_motions)
    
    def generate_synthetic_ground_motion(self, duration=10.0, dt=0.01, target_pga=0.035,
                                          window_before=2.5, window_after=2.5):
        n = int(duration / dt)
        t = np.linspace(0, duration, n)
        
        envelope = np.exp(-0.3 * t / duration * 5) * (1 - np.exp(-3 * t / duration))
        envelope = envelope / (np.max(envelope) + 1e-10)
        peak_shift = np.exp(-2 * ((t - duration*0.4) / (duration*0.3))**2)
        envelope = envelope * peak_shift
        envelope = envelope / (np.max(envelope) + 1e-10)
        
        n_freqs = np.random.randint(10, 20)
        freqs = np.random.uniform(0.5, 15, n_freqs)
        amps = np.random.uniform(0.3, 1.0, n_freqs)
        phases = np.random.uniform(0, 2*np.pi, n_freqs)
        
        motion = np.zeros(n)
        for f, a, p in zip(freqs, amps, phases):
            motion += a * np.sin(2*np.pi*f*t + p)
        
        motion = motion * envelope
        pga = np.max(np.abs(motion))
        if pga > 1e-10:
            motion = motion / pga * target_pga
        
        return self.extract_peak_window(motion, dt, window_before, window_after)
    
    def get_earthquake_pool(self, folder_path=None, num_waves=60):
        all_motions = []
        all_names = []
        
        if folder_path and os.path.exists(folder_path):
            natural_motions, natural_names, n_natural = self.load_earthquake_files(
                folder_path,
                self.config.TARGET_PGA,
                self.config.TARGET_DT,
                self.config.WINDOW_BEFORE,
                self.config.WINDOW_AFTER,
                max_files=num_waves * 2
            )
            if len(natural_motions) > 0:
                all_motions.extend(natural_motions)
                all_names.extend(natural_names)
        
        n_needed = max(num_waves - len(all_motions), 0)
        if n_needed > 0:
            synthetic = []
            for _ in range(n_needed * 2):
                motion = self.generate_synthetic_ground_motion(
                    10.0, self.config.TARGET_DT, self.config.TARGET_PGA,
                    self.config.WINDOW_BEFORE, self.config.WINDOW_AFTER
                )
                synthetic.append(motion)
            synthetic = np.array(synthetic)
            if len(synthetic) > n_needed:
                indices = np.random.choice(len(synthetic), n_needed, replace=False)
                synthetic = synthetic[indices]
            all_motions.extend(synthetic)
        
        return np.array(all_motions)


# ================================================================
# 3D OpenSees 仿真器 (修复版)
# ================================================================

class OpenSeesSimulator3D:
    """3D OpenSees 框架时程分析 (修复坐标变换问题)"""
    
    def __init__(self, config=None):
        self.config = config or SimConfig3D()
        if ops is None:
            raise ImportError("openseespy 未安装，请运行: pip install openseespy")
    
    def run_analysis(self, ground_motion_x, ground_motion_y, dt, 
                     num_stories, num_bays_x, num_bays_y,
                     bay_width_x, bay_width_y, story_height, 
                     mass_per_node, damping_ratio, floor_node_masses=None,
                     col_sections=None, beam_sections=None,
                     plane_shape='rect'):
        """
        3D框架结构时程分析 (修复版)

        Args:
            floor_node_masses: [num_stories] 每层节点质量 (kg)。
                若提供则按楼层逐层赋质量 (每层节点质量 = floor_node_masses[floor-1])，
                否则用统一的 mass_per_node。
            col_sections: [num_stories] 逐层柱截面 (正方形边长 m)。
                若提供则覆盖 config.COL_SECTIONS。
            beam_sections: [num_stories] 逐层梁截面 [(w,h), ...] m。
                若提供则每层用对应截面; 否则用 config.BEAM_SECTION 单值。
            plane_shape: 平面形状 'rect'/'T'/'L'/'C'/'U'。
                只生成形状掩码内的节点/柱/梁/质量 (与 generate_fixed_frame 一致)。
        """
        if ops is None:
            return np.zeros((len(ground_motion_x), 3))
        
        n_steps = len(ground_motion_x)
        
        # 参数限制 (放宽: 层数<=12, 跨数<=8, 匹配体素空间 60m 平面 / 100m 高度)
        num_stories = int(max(2, min(num_stories, 12)))
        num_bays_x = int(max(1, min(num_bays_x, 8)))
        num_bays_y = int(max(1, min(num_bays_y, 8)))
        bay_width_x = float(max(3.0, min(bay_width_x, 15.0)))
        bay_width_y = float(max(3.0, min(bay_width_y, 15.0)))
        story_height = float(max(2.5, min(story_height, 6.0)))
        mass_per_node = float(max(10000, min(mass_per_node, 60000)))
        damping_ratio = float(max(0.01, min(damping_ratio, 0.10)))
        
        # 平面形状掩码 [ny, nx] (与 generate_fixed_frame 一致)
        from generate_frames import plane_mask
        plane_shape = str(plane_shape or 'rect').lower()
        mask = plane_mask(plane_shape, num_bays_x, num_bays_y)
        # 形状内节点 (每层): 掩码格子 4 角的并集
        _corners = set()
        for iy in range(num_bays_y):
            for ix in range(num_bays_x):
                if mask[iy, ix]:
                    _corners |= {(ix, iy), (ix + 1, iy), (ix, iy + 1), (ix + 1, iy + 1)}
        corners = sorted(_corners)   # [(ix, iy), ...]
        n_plane_nodes = len(corners)

        # 每层节点质量 (默认用 mass_per_node)
        if floor_node_masses is not None and len(floor_node_masses) >= num_stories:
            floor_mass_arr = np.asarray(floor_node_masses, dtype=np.float64)[:num_stories]
            floor_mass_arr = np.clip(floor_mass_arr, 10000, 60000)  # 合理范围
            m_ref = float(np.mean(floor_mass_arr))
        else:
            floor_mass_arr = None
            m_ref = mass_per_node
        
        # 获取柱截面 (若传入逐层截面则覆盖 config)
        if col_sections is None:
            col_sections = self.config.COL_SECTIONS[:num_stories]
        col_sections = list(col_sections)
        if len(col_sections) < num_stories:
            col_sections = col_sections + [col_sections[-1]] * (num_stories - len(col_sections))
        col_sections = [max(0.2, min(float(c), 1.4)) for c in col_sections]
        
        # 梁截面: 逐层 (beam_sections) 或单值 (BEAM_SECTION)
        if beam_sections is not None and len(beam_sections) >= num_stories:
            beam_secs = [(max(0.1, float(w)), max(0.2, float(h)))
                         for w, h in beam_sections[:num_stories]]
        else:
            beam_b0, beam_h0 = self.config.BEAM_SECTION
            beam_secs = [(float(beam_b0), float(beam_h0))] * num_stories
        E = self.config.E
        nu = self.config.NU
        G = E / (2 * (1 + nu))
        
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        
        # ============================================================
        # 1. 创建节点 (仅形状掩码内)
        # ============================================================
        node_tags = {}
        node_id = 0
        
        for floor in range(num_stories + 1):
            for (ix, iy) in corners:
                node_id += 1
                x = ix * bay_width_x
                y = iy * bay_width_y
                z = floor * story_height
                ops.node(node_id, x, y, z)
                node_tags[(floor, ix, iy)] = node_id
        
        # 约束底部节点
        for (ix, iy) in corners:
            ops.fix(node_tags[(0, ix, iy)], 1, 1, 1, 1, 1, 1)
        
        # ============================================================
        # 2. 定义材料和坐标变换 (修复关键)
        # ============================================================
        ops.uniaxialMaterial('Elastic', 1, E)
        ops.uniaxialMaterial('Elastic', 2, G)
        
        # 修复: 使用多个坐标变换，避免参考向量平行问题
        # 变换1: 用于垂直柱 (参考向量为全局X轴)
        ops.geomTransf('Linear', 1, 1, 0, 0)
        # 变换2: 用于X方向梁 (参考向量为全局Y轴)
        ops.geomTransf('Linear', 2, 0, 1, 0)
        # 变换3: 用于Y方向梁 (参考向量为全局X轴)
        ops.geomTransf('Linear', 3, 1, 0, 0)
        
        # ============================================================
        # 3. 柱单元 (仅形状掩码内的角点节点, 使用变换1)
        # ============================================================
        elem_id = 0
        for floor in range(1, num_stories + 1):
            col_idx = floor - 1
            col_size = col_sections[col_idx]
            A_col = col_size ** 2
            I_col = col_size ** 4 / 12
            
            for (ix, iy) in corners:
                elem_id += 1
                node_bottom = node_tags[(floor - 1, ix, iy)]
                node_top = node_tags[(floor, ix, iy)]
                ops.element('elasticBeamColumn', elem_id, node_bottom, node_top,
                           A_col, E, G, I_col, I_col, I_col, 1)
        
        # ============================================================
        # 4. X方向梁 (形状内相邻格子, 使用变换2, 每层截面可不同)
        # ============================================================
        def _x_beams():
            """形状内所有 X 方向梁 ((ix,iy)->(ix+1,iy)), 去重."""
            s = set()
            for iy in range(num_bays_y):
                for ix in range(num_bays_x):
                    if mask[iy, ix]:
                        s.add((ix, iy, ix + 1, iy))
            return sorted(s)
        
        for floor in range(1, num_stories + 1):
            beam_b, beam_h = beam_secs[floor - 1]
            A_beam = beam_b * beam_h
            I_beam = beam_b * beam_h ** 3 / 12
            for (ix, iy, jx, jy) in _x_beams():
                elem_id += 1
                node_left = node_tags[(floor, ix, iy)]
                node_right = node_tags[(floor, jx, jy)]
                ops.element('elasticBeamColumn', elem_id, node_left, node_right,
                           A_beam, E, G, I_beam, I_beam, I_beam, 2)
        
        # ============================================================
        # 5. Y方向梁 (形状内相邻格子, 使用变换3, 每层截面可不同)
        # ============================================================
        def _y_beams():
            """形状内所有 Y 方向梁 ((ix,iy)->(ix,iy+1)), 去重."""
            s = set()
            for iy in range(num_bays_y):
                for ix in range(num_bays_x):
                    if mask[iy, ix]:
                        s.add((ix, iy, ix, iy + 1))
            return sorted(s)
        
        for floor in range(1, num_stories + 1):
            beam_b, beam_h = beam_secs[floor - 1]
            A_beam = beam_b * beam_h
            I_beam = beam_b * beam_h ** 3 / 12
            for (ix, iy, jx, jy) in _y_beams():
                elem_id += 1
                node_left = node_tags[(floor, ix, iy)]
                node_right = node_tags[(floor, jx, jy)]
                ops.element('elasticBeamColumn', elem_id, node_left, node_right,
                           A_beam, E, G, I_beam, I_beam, I_beam, 3)
        
        # ============================================================
        # 6. 质量 (仅形状掩码内节点, 按楼层赋质量)
        # ============================================================
        for floor in range(1, num_stories + 1):
            if floor_mass_arr is not None:
                m_node = floor_mass_arr[floor - 1]
            else:
                m_node = mass_per_node
            for (ix, iy) in corners:
                ops.mass(node_tags[(floor, ix, iy)], 
                        m_node, m_node, m_node, 0, 0, 0)
        
        # ============================================================
        # 7. 阻尼 (Rayleigh) - 用平均节点质量估算基频 (按形状内节点数)
        # ============================================================
        col_size_0 = col_sections[0]
        I_col_0 = col_size_0 ** 4 / 12
        k_col_est = 12 * E * I_col_0 / (story_height ** 3)
        k_eff = n_plane_nodes * k_col_est
        m_eff = n_plane_nodes * m_ref
        omega1 = np.sqrt(k_eff / m_eff) if m_eff > 0 else 1.0
        omega2 = omega1 * 2.5
        
        if omega1 < 0.1:
            omega1 = 1.0
            omega2 = 2.5
        
        alpha_m = 2 * damping_ratio * omega1 * omega2 / (omega1 + omega2)
        beta_k = 2 * damping_ratio / (omega1 + omega2)
        ops.rayleigh(alpha_m, beta_k, 0.0, 0.0)
        
        # ============================================================
        # 8. 地面运动
        # ============================================================
        accel_values_x = list(-1.0 * np.array(ground_motion_x) * 9.81)
        ops.timeSeries('Path', 1, '-dt', dt, '-values', *accel_values_x)
        ops.pattern('UniformExcitation', 1, 1, '-accel', 1, '-dir', 1)
        
        if len(ground_motion_y) > 0:
            accel_values_y = list(-1.0 * np.array(ground_motion_y) * 9.81)
            ops.timeSeries('Path', 2, '-dt', dt, '-values', *accel_values_y)
            ops.pattern('UniformExcitation', 2, 2, '-accel', 1, '-dir', 2)
        
        # ============================================================
        # 9. 分析设置
        # ============================================================
        ops.wipeAnalysis()
        ops.constraints('Transformation')
        ops.numberer('RCM')
        ops.system('BandGeneral')
        ops.test('NormDispIncr', 1e-6, 100)
        ops.algorithm('Newton')
        ops.integrator('Newmark', 0.5, 0.25)
        ops.analysis('Transient')
        
        # ============================================================
        # 10. 时程分析 + 记录 (顶部位移 + 峰值帧全节点位移 + 单元峰值内力)
        # ============================================================
        # 顶部参考节点: 取形状内 (0,0) 角 (如无则用第一个角点)
        top_ref = (num_stories, 0, 0)
        if top_ref not in node_tags:
            top_ref = (num_stories,) + corners[0]
        top_node = node_tags[top_ref]

        # 节点坐标映射 (OpenSees node_id -> (x,y,z) 物理坐标, 供可视化几何匹配)
        node_coords = {}
        for floor in range(num_stories + 1):
            for (ix, iy) in corners:
                nid = node_tags[(floor, ix, iy)]
                node_coords[nid] = (ix * bay_width_x, iy * bay_width_y,
                                    floor * story_height)

        # 单元几何映射 (OpenSees elem_id -> 类型/楼层/两端节点, 供可视化内力图分组)
        elem_geom = {}
        _eid = 0
        for floor in range(1, num_stories + 1):
            for (ix, iy) in corners:
                _eid += 1
                elem_geom[_eid] = {
                    'type': 'column', 'floor': floor,
                    'n1': node_tags[(floor - 1, ix, iy)],
                    'n2': node_tags[(floor, ix, iy)],
                }
        for floor in range(1, num_stories + 1):
            for (ix, iy, jx, jy) in _x_beams():
                _eid += 1
                elem_geom[_eid] = {
                    'type': 'beam_x', 'floor': floor,
                    'n1': node_tags[(floor, ix, iy)],
                    'n2': node_tags[(floor, jx, jy)],
                }
        for floor in range(1, num_stories + 1):
            for (ix, iy, jx, jy) in _y_beams():
                _eid += 1
                elem_geom[_eid] = {
                    'type': 'beam_y', 'floor': floor,
                    'n1': node_tags[(floor, ix, iy)],
                    'n2': node_tags[(floor, jx, jy)],
                }

        disp_x = []
        disp_y = []
        disp_z = []
        # 全节点位移时程 (m) 与全单元内力时程 (N, N*m), 仅用于提取峰值帧
        node_disp_hist = {nid: [] for nid in node_coords}
        elem_force_hist = {eid: [] for eid in elem_geom}

        for step in range(n_steps):
            ok = ops.analyze(1, dt)
            if ok != 0:
                break

            d = ops.nodeDisp(top_node)
            disp_x.append(d[0] if len(d) > 0 else 0)
            disp_y.append(d[1] if len(d) > 1 else 0)
            disp_z.append(d[2] if len(d) > 2 else 0)

            for nid in node_coords:
                dn = ops.nodeDisp(nid)
                node_disp_hist[nid].append(
                    [dn[0], dn[1], dn[2]] if len(dn) >= 3 else [0.0, 0.0, 0.0])
            for eid in elem_geom:
                f = ops.eleForce(eid)
                elem_force_hist[eid].append(list(f[:6]) if len(f) >= 6 else [0.0] * 6)

        top_disp = np.column_stack([disp_x, disp_y, disp_z]) * 1000.0  # mm
        # 补齐长度 (分析提前中断时用 edge 填充)
        if len(top_disp) < n_steps:
            top_disp = np.pad(top_disp,
                              ((0, n_steps - len(top_disp)), (0, 0)), 'edge')

        # 峰值帧: 取顶部位移 X 向绝对值最大的时刻
        peak_t = int(np.argmax(np.abs(top_disp[:, 0]))) if len(top_disp) else 0

        # 峰值帧全节点位移 (m) - 云图用 (只保留 1 帧, 缓存紧凑)
        node_peak_disp = {}
        for nid in node_coords:
            h = node_disp_hist[nid]
            if len(h) > peak_t:
                node_peak_disp[nid] = np.asarray(h[peak_t], dtype=np.float32)
            else:
                node_peak_disp[nid] = np.zeros(3, dtype=np.float32)

        # 单元峰值内力 (各分量绝对值峰值, N / N*m) - 内力图用
        elem_peak_force = {}
        for eid in elem_geom:
            h = elem_force_hist[eid]
            if len(h) > 0:
                arr = np.abs(np.asarray(h, dtype=np.float64))  # [T,6]
                elem_peak_force[eid] = arr.max(axis=0).astype(np.float32)
            else:
                elem_peak_force[eid] = np.zeros(6, dtype=np.float32)

        return {
            'top_displacement': top_disp,           # [T,3] mm
            'node_peak_disp': node_peak_disp,       # {nid: [3]} m
            'elem_peak_force': elem_peak_force,     # {eid: [6]} N / N*m
            'node_coords': node_coords,             # {nid: (x,y,z)} m
            'elem_geom': elem_geom,                 # {eid: {type,floor,n1,n2}}
        }


# ================================================================
# 测试
# ================================================================

if __name__ == "__main__":
    print("="*60)
    print("3D OpenSees 仿真测试 (修复版)")
    print("="*60)
    
    config = SimConfig3D()
    simulator = OpenSeesSimulator3D(config)
    loader = EarthquakeLoader3D(config)
    
    motion = loader.generate_synthetic_ground_motion()
    
    print("\n运行单次3D分析...")
    resp = simulator.run_analysis(
        motion, motion * 0.8,
        config.TARGET_DT,
        num_stories=3,
        num_bays_x=2,
        num_bays_y=2,
        bay_width_x=6.0,
        bay_width_y=5.0,
        story_height=3.5,
        mass_per_node=35000,
        damping_ratio=0.05
    )
    
    print(f"  顶部位移 X: 最大 {np.max(np.abs(resp[:,0])):.2f} mm")
    print(f"  顶部位移 Y: 最大 {np.max(np.abs(resp[:,1])):.2f} mm")