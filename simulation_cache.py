# simulation_cache.py
"""
仿真结果缓存模块 - 多进程并行版
功能：
1. 使用 ProcessPoolExecutor 并行运行多个独立仿真
2. 缓存原始结构参数和仿真位移结果
3. 支持增量追加（新增样本不覆盖旧样本）
4. 支持按需加载，不占内存
5. 完整的错误处理和状态判断
6. 进度条显示并行任务状态
"""
import os
import pickle
import numpy as np
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from config import Config
from earthquake_simulator_3d import OpenSeesSimulator3D, SimConfig3D, EarthquakeLoader3D
from generate_frames import generate_fixed_frame, calculate_column_section


# ============================================================
# 每层荷载 / 节点质量生成
# ============================================================
G = 9.81  # 重力加速度 (m/s^2)


def compute_floor_node_masses(num_stories, num_spans_x, num_spans_y,
                              span_x, span_y, floor_loads,
                              plane_shape='rect'):
    """
    计算每层每个节点的质量 (kg)。

    规则:
    - 楼层面积 = 形状内格子数 × 单格面积 (m^2)  (T/L/C 比矩形小)
    - 楼层总质量 = 面积 * 荷载(kPa) * 1000 / g  (kg)
    - 每层节点数 = 形状内角点数
    - 每节点质量 = 楼层总质量 / 每层节点数

    Args:
        floor_loads: [num_stories] 每层楼面荷载 (kPa)
        plane_shape: 平面形状 'rect'/'T'/'L'/'C'/'U' (默认 rect 兼容旧)

    Returns:
        floor_node_masses: [num_stories] 每层节点质量 (kg)，每层均匀分摊
    """
    from generate_frames import shape_cell_count
    plane_shape = str(plane_shape or 'rect').lower()
    n_cells, n_nodes = shape_cell_count(plane_shape, num_spans_x, num_spans_y)
    floor_area = n_cells * span_x * span_y
    nodes_per_floor = max(1, n_nodes)

    floor_node_masses = []
    for fl in range(int(num_stories)):
        load_kpa = float(floor_loads[fl])
        floor_mass = floor_area * load_kpa * 1000.0 / G  # kg
        floor_node_masses.append(floor_mass / nodes_per_floor)
    return np.array(floor_node_masses, dtype=np.float32)


def generate_floor_loads(num_stories, options=None):
    """为每层随机生成楼面荷载 (kPa)，不同楼层可能不同"""
    if options is None:
        cfg = Config()
        options = getattr(cfg, 'FLOOR_LOAD_OPTIONS', [10, 15, 20, 25])
    return np.random.choice(options, size=int(num_stories)).astype(np.float32)


# ============================================================
# 全局变量：用于多进程共享配置
# ============================================================


def run_single_simulation(args):
    """
    单个仿真任务（在子进程中执行）
    
    Args:
        args: (params, motion, seq_len, target_dt, col_sections, beam_section, E,
               floor_node_masses, pga_target)
    
    Returns:
        dict: 包含仿真结果或错误信息
    """
    (params, motion, seq_len, target_dt, col_sections, beam_section, E,
     floor_node_masses, pga_target) = args
    
    # 在子进程中创建仿真器
    sim_config = SimConfig3D()
    sim_config.COL_SECTIONS = col_sections
    sim_config.BEAM_SECTION = beam_section
    sim_config.E = E
    sim_config.TARGET_DT = target_dt
    
    simulator = OpenSeesSimulator3D(sim_config)    
    # 提取参数
    num_stories = int(params[0])
    num_bays_x = int(params[1])
    num_bays_y = int(params[2])
    bay_width_x = float(params[3])
    bay_width_y = float(params[4])
    story_height = float(params[5])
    damping_ratio = float(params[7])
    # 平面形状 (第 8 位是形状 ID, 用 id_to_shape 转名称; 旧 8 维默认 rect)
    from generate_frames import id_to_shape
    plane_shape = id_to_shape(params[8]) if len(params) > 8 else 'rect'
    
    # 确保地震动长度
    if len(motion) < seq_len:
        motion = np.pad(motion, (0, seq_len - len(motion)), 'constant')
    elif len(motion) > seq_len:
        motion = motion[:seq_len]
    
    # 每样本 PGA 缩放: motion 池基准为 TARGET_PGA, 缩放到目标 pga (g)
    motion_peak = np.max(np.abs(motion))
    if motion_peak > 1e-10:
        motion = motion.astype(np.float64) * (pga_target / motion_peak)
    # Y方向地震动（缩放，PGA 约为 X 向的 0.5~1.0）
    motion_y = motion * np.random.uniform(0.5, 1.0)
    
    try:
        # 运行仿真 (使用每层节点质量 + 逐层柱/梁截面)
        result = simulator.run_analysis(
            ground_motion_x=motion,
            ground_motion_y=motion_y,
            dt=target_dt,
            num_stories=num_stories,
            num_bays_x=num_bays_x,
            num_bays_y=num_bays_y,
            bay_width_x=bay_width_x,
            bay_width_y=bay_width_y,
            story_height=story_height,
            mass_per_node=0.0,  # 用 floor_node_masses 逐层赋值
            damping_ratio=damping_ratio,
            floor_node_masses=floor_node_masses,
            col_sections=col_sections,
            beam_sections=beam_section,
            plane_shape=plane_shape,
        )
        # 兼容: run_analysis 现在返回 dict (含 top_displacement / node_peak_disp / elem_peak_force)
        if isinstance(result, dict):
            top_disp = result['top_displacement']           # [T,3] mm
            node_peak_disp = result.get('node_peak_disp', {})
            elem_peak_force = result.get('elem_peak_force', {})
            node_coords = result.get('node_coords', {})
            elem_geom = result.get('elem_geom', {})
        else:
            # 旧版返回 [T,3] 数组
            top_disp = result
            node_peak_disp = {}
            elem_peak_force = {}
            node_coords = {}
            elem_geom = {}
        displacement = top_disp[:, 0] if top_disp.shape[1] >= 1 else np.zeros(seq_len)
        
        # 检查有效性
        if np.isnan(displacement).any() or np.isinf(displacement).any():
            displacement = np.zeros(seq_len)
            failed = True
        else:
            failed = False
            
    except Exception as e:
        displacement = np.zeros(seq_len)
        node_peak_disp = {}
        elem_peak_force = {}
        node_coords = {}
        elem_geom = {}
        failed = True
    
    # 确保长度一致
    if len(displacement) < seq_len:
        displacement = np.pad(displacement, (0, seq_len - len(displacement)), 'constant')
    elif len(displacement) > seq_len:
        displacement = displacement[:seq_len]
    
    # 记录实际输入地震动 (X向, 单位 g, 与位移时间轴一致)
    motion_input = np.asarray(motion, dtype=np.float32)
    if len(motion_input) < seq_len:
        motion_input = np.pad(motion_input, (0, seq_len - len(motion_input)), 'constant')
    elif len(motion_input) > seq_len:
        motion_input = motion_input[:seq_len]
    
    # 实际应用的目标 PGA (g) - 供报告/统计使用
    pga_applied = float(np.max(np.abs(motion_input))) if len(motion_input) else float(pga_target)
    
    return {
        'displacement': displacement.astype(np.float32),
        'motion': motion_input.astype(np.float32),
        'height': np.float32(num_stories * story_height),
        'E_avg': np.float32(30.0),
        'pga': pga_applied,
        'failed': failed,
        'params': params,
        'node_peak_disp': node_peak_disp,
        'elem_peak_force': elem_peak_force,
        'node_coords': node_coords,
        'elem_geom': elem_geom
    }


class SimulationCache:
    """
    仿真结果缓存管理器（多进程版）
    
    缓存内容:
        - params: 结构参数 [N, 8]
        - displacements: 位移时程 [N, T]
        - heights: 建筑高度 [N]
        - E_avg: 平均弹性模量 [N]
        - motion_indices: 使用的地震动索引 [N]
    
    特点:
        - 只缓存仿真结果，不缓存体素
        - 支持增量追加
        - 使用多进程并行加速
        - 完整的状态判断
    """
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.cache_file = self.config.SIM_CACHE_FILE
        self.seq_len = self.config.get_seq_len()
        
        # 数据存储
        self.params = None          # [N, 8]
        self.displacements = None   # [N, T]
        self.motions = None         # [N, T] 每样本实际输入地震动加速度 (g)
        self.heights = None         # [N]
        self.E_avg = None           # [N]
        self.motion_indices = None  # [N]
        self.floor_loads = None     # [N] 每层荷载 (kPa) list of arrays
        self.floor_node_masses = None  # [N] 每层节点质量 (kg) list of arrays
        self.pgas = None            # [N] 每样本实际应用 PGA (g)
        self.node_peak_disp = None  # [N] 峰值帧全节点位移 (m) list of dicts
        self.elem_peak_force = None # [N] 单元峰值内力 (N,N*m) list of dicts
        self.node_coords = None     # [N] 节点坐标 dicts (可视化几何匹配用)
        self.elem_geom = None       # [N] 单元几何 dicts (type/floor/节点)
        
        # 状态
        self.is_loaded = False
        self.num_samples = 0
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
    
    def load(self):
        """从文件加载缓存"""
        if not os.path.exists(self.cache_file):
            print(f"  ⚠️ 缓存文件不存在: {self.cache_file}")
            return False
        
        try:
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)
            
            required_keys = ['params', 'displacements', 'heights', 'E_avg', 'motion_indices']
            for key in required_keys:
                if key not in data:
                    print(f"  ⚠️ 缓存缺少字段: {key}")
                    return False
            
            self.params = data['params']
            self.displacements = data['displacements']
            self.heights = data['heights']
            self.E_avg = data['E_avg']
            self.motion_indices = data['motion_indices']
            # 每层荷载/质量/实际地震动 (旧缓存可能没有，回退为 None)
            self.floor_loads = data.get('floor_loads', None)
            self.floor_node_masses = data.get('floor_node_masses', None)
            self.motions = data.get('motions', None)
            self.pgas = data.get('pgas', None)
            self.node_peak_disp = data.get('node_peak_disp', None)
            self.elem_peak_force = data.get('elem_peak_force', None)
            self.node_coords = data.get('node_coords', None)
            self.elem_geom = data.get('elem_geom', None)
            self.num_samples = len(self.params)
            self.is_loaded = True
            
            print(f"  ✅ 加载仿真缓存: {self.num_samples} 个样本")
            print(f"     位移范围: [{self.displacements.min():.2f}, {self.displacements.max():.2f}] mm")
            if self.motions is not None:
                print(f"     含实际输入地震动: {self.motions.shape}")
            if self.pgas is not None:
                print(f"     目标PGA范围: [{self.pgas.min():.4f}, {self.pgas.max():.4f}] g")
            if self.node_peak_disp is not None:
                print(f"     含节点位移/单元内力 (云图+内力图数据): ✓")
            return True
            
        except Exception as e:
            print(f"  ⚠️ 缓存加载失败: {e}")
            return False
    
    def save(self):
        """保存缓存到文件"""
        if not self.is_loaded or self.num_samples == 0:
            print("  ⚠️ 无数据可保存")
            return False
        
        try:
            data = {
                'params': self.params,
                'displacements': self.displacements,
                'motions': self.motions,
                'heights': self.heights,
                'E_avg': self.E_avg,
                'motion_indices': self.motion_indices,
                'floor_loads': self.floor_loads,
                'floor_node_masses': self.floor_node_masses,
                'pgas': self.pgas,
                'node_peak_disp': self.node_peak_disp,
                'elem_peak_force': self.elem_peak_force,
                'node_coords': self.node_coords,
                'elem_geom': self.elem_geom,
                'num_samples': self.num_samples,
                'seq_len': self.seq_len,
                'version': '1.4'
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size = os.path.getsize(self.cache_file) / (1024**2)
            print(f"  ✅ 仿真缓存已保存: {self.cache_file} ({file_size:.1f} MB)")
            return True
            
        except Exception as e:
            print(f"  ⚠️ 缓存保存失败: {e}")
            return False
    
    def append(self, new_params, new_displacements, new_heights, new_E_avg, new_motion_indices,
               floor_loads=None, floor_node_masses=None, new_motions=None, new_pgas=None,
               new_node_peak_disp=None, new_elem_peak_force=None,
               new_node_coords=None, new_elem_geom=None):
        """追加新数据到缓存"""
        n_new = len(new_params)
        if n_new == 0:
            return
        
        # 统一转为 np.ndarray (list of arrays -> 用 object 数组或直接存 list)
        floor_loads = list(floor_loads) if floor_loads is not None else None
        floor_node_masses = list(floor_node_masses) if floor_node_masses is not None else None
        
        if not self.is_loaded:
            self.params = np.array(new_params, dtype=np.float32)
            self.displacements = np.array(new_displacements, dtype=np.float32)
            self.heights = np.array(new_heights, dtype=np.float32)
            self.E_avg = np.array(new_E_avg, dtype=np.float32)
            self.motion_indices = np.array(new_motion_indices, dtype=np.int32)
            self.floor_loads = floor_loads
            self.floor_node_masses = floor_node_masses
            self.motions = np.array(new_motions, dtype=np.float32) if new_motions is not None else None
            self.pgas = np.array(new_pgas, dtype=np.float32) if new_pgas is not None else None
            self.node_peak_disp = list(new_node_peak_disp) if new_node_peak_disp is not None else None
            self.elem_peak_force = list(new_elem_peak_force) if new_elem_peak_force is not None else None
            self.node_coords = list(new_node_coords) if new_node_coords is not None else None
            self.elem_geom = list(new_elem_geom) if new_elem_geom is not None else None
            self.num_samples = n_new
            self.is_loaded = True
        else:
            self.params = np.concatenate([self.params, np.array(new_params, dtype=np.float32)], axis=0)
            self.displacements = np.concatenate([self.displacements, np.array(new_displacements, dtype=np.float32)], axis=0)
            self.heights = np.concatenate([self.heights, np.array(new_heights, dtype=np.float32)], axis=0)
            self.E_avg = np.concatenate([self.E_avg, np.array(new_E_avg, dtype=np.float32)], axis=0)
            self.motion_indices = np.concatenate([self.motion_indices, np.array(new_motion_indices, dtype=np.int32)], axis=0)
            if new_motions is not None:
                new_motions = np.array(new_motions, dtype=np.float32)
                self.motions = new_motions if self.motions is None else np.concatenate([self.motions, new_motions], axis=0)
            if floor_loads is not None:
                self.floor_loads = (self.floor_loads or []) + floor_loads
            if floor_node_masses is not None:
                self.floor_node_masses = (self.floor_node_masses or []) + floor_node_masses
            if new_pgas is not None:
                new_pgas = np.array(new_pgas, dtype=np.float32)
                self.pgas = new_pgas if self.pgas is None else np.concatenate([self.pgas, new_pgas], axis=0)
            if new_node_peak_disp is not None:
                self.node_peak_disp = (self.node_peak_disp or []) + list(new_node_peak_disp)
            if new_elem_peak_force is not None:
                self.elem_peak_force = (self.elem_peak_force or []) + list(new_elem_peak_force)
            if new_node_coords is not None:
                self.node_coords = (self.node_coords or []) + list(new_node_coords)
            if new_elem_geom is not None:
                self.elem_geom = (self.elem_geom or []) + list(new_elem_geom)
            self.num_samples = len(self.params)
        
        print(f"  ✅ 追加 {n_new} 个样本，总计 {self.num_samples} 个")
    
    def get_all(self):
        """获取所有数据"""
        if not self.is_loaded:
            return None
        return {
            'params': self.params,
            'displacements': self.displacements,
            'motions': self.motions,
            'heights': self.heights,
            'E_avg': self.E_avg,
            'motion_indices': self.motion_indices,
            'floor_loads': self.floor_loads,
            'floor_node_masses': self.floor_node_masses,
            'pgas': self.pgas,
            'node_peak_disp': self.node_peak_disp,
            'elem_peak_force': self.elem_peak_force,
            'node_coords': self.node_coords,
            'elem_geom': self.elem_geom
        }
    
    def get_stats(self):
        """获取统计信息"""
        if not self.is_loaded or self.num_samples == 0:
            return None
        return {
            'num_samples': self.num_samples,
            'seq_len': self.seq_len,
            'param_mean': self.params.mean(axis=0),
            'param_std': self.params.std(axis=0),
            'disp_mean': self.displacements.mean(),
            'disp_std': self.displacements.std(),
            'disp_min': self.displacements.min(),
            'disp_max': self.displacements.max(),
            'height_mean': self.heights.mean(),
            'height_range': [self.heights.min(), self.heights.max()]
        }


class SimulationGenerator:
    """
    仿真数据生成器（多进程并行版）
    功能：
    1. 生成大量结构参数
    2. 加载地震动
    3. 使用多进程并行运行OpenSees仿真
    4. 自动缓存结果
    """
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.cache = SimulationCache(config)
        
        # 初始化地震动加载器 (使用与仿真一致的 dt/window, 保证波长度=seq_len)
        sim_config = SimConfig3D()
        sim_config.TARGET_DT = self.config.TARGET_DT
        sim_config.WINDOW_BEFORE = self.config.WINDOW_BEFORE
        sim_config.WINDOW_AFTER = self.config.WINDOW_AFTER
        sim_config.TARGET_PGA = self.config.TARGET_PGA
        self.loader = EarthquakeLoader3D(sim_config)
        
        # 地震动池
        self.motion_pool = None
        self.num_waves = 0
        
        # 多进程参数
        self.max_workers = min(mp.cpu_count(), getattr(self.config, 'MAX_SIM_WORKERS', 16))
        print(f"  🖥️ 检测到 {mp.cpu_count()} 个CPU核心，使用 {self.max_workers} 个工作进程")
    
    def load_motions(self, earthquake_folder=None):
        """
        加载地震动池: 优先从 dzb 真实波文件夹读取，缓存到 motion_pool.pkl 以便复现
        """
        print(f"\n  📊 加载地震动...")
        motion_pkl = os.path.join(self.config.CACHE_DIR, 'motion_pool.pkl')
        
        # 若已缓存地震动池且不强制重读，直接加载 (保证与仿真时一致的波形)
        if (not getattr(self.config, 'FORCE_MOTION_REGEN', False)
                and os.path.exists(motion_pkl)):
            try:
                with open(motion_pkl, 'rb') as f:
                    self.motion_pool = pickle.load(f)
                self.num_waves = len(self.motion_pool)
                print(f"  ✅ 加载地震动池缓存: {self.num_waves} 条")
                return self.motion_pool
            except Exception as e:
                print(f"  ⚠️ 地震动池缓存加载失败: {e}")
        
        # 从 dzb 真实波读取 (若无则合成)
        folder = earthquake_folder or getattr(self.config, 'EARTHQUAKE_FOLDER', None)
        self.motion_pool = self.loader.get_earthquake_pool(
            folder, num_waves=self.config.NUM_WAVES)
        self.num_waves = len(self.motion_pool)
        
        if self.num_waves == 0:
            print("  ⚠️ 无地震动，生成合成波...")
            synthetic = self.loader.generate_synthetic_ground_motion()
            self.motion_pool = np.array([synthetic])
            self.num_waves = 1
        
        # 落盘地震动池 (可复现)
        try:
            with open(motion_pkl, 'wb') as f:
                pickle.dump(self.motion_pool, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  ✅ 地震动池已缓存: {motion_pkl}")
        except Exception as e:
            print(f"  ⚠️ 地震动池缓存失败: {e}")
        
        print(f"  ✅ 地震动池: {self.num_waves} 条")
        return self.motion_pool
    
    def generate_structure_params(self, num_samples):
        """
        生成随机结构参数 (使用 generate_fixed_frame 生成完整框架)
        
        注意: 参数范围必须与 OpenSees 仿真器的支持范围一致，
        否则体素特征描述的结构与实际仿真的结构不一致 (训练数据不匹配)。
        仿真器限制: 层数<=6, 跨数<=2 (见 earthquake_simulator_3d.run_analysis)
        
        多样性增强:
        - 每层随机选择楼面荷载 (10/15/20/25 kPa)，不同楼层可能不同
        - 由楼层面积计算楼层质量，平均分摊到本层节点
        
        返回:
            params: [N, 9] 结构参数
                列: [num_stories, num_bays_x, num_bays_y, 
                     bay_width_x, bay_width_y, story_height,
                     mass_per_node, damping_ratio, plane_shape_id]
                (plane_shape_id: rect=0, T=1, L=2, C=3, U=4; 兼容旧 8 维)
            floor_loads_list: [N] 每样本每层荷载数组 (kPa) [num_stories]
            floor_node_masses_list: [N] 每样本每层节点质量数组 (kg) [num_stories]
        """
        cfg = self.config
        
        # 与仿真器一致的最大值 (earthquake_simulator_3d.run_analysis 的 clamp)
        # 放宽: 层数<=12, 跨数<=8 (跨数×跨度自动 <= 60m 平面)
        MAX_STORIES = 12
        MAX_SPANS = 8
        
        params = []
        floor_loads_list = []
        floor_node_masses_list = []
        
        for _ in range(num_samples):
            # 随机选择参数
            num_stories = np.random.randint(cfg.NUM_STORIES_RANGE[0],
                                            min(cfg.NUM_STORIES_RANGE[1], MAX_STORIES) + 1)
            story_height = np.random.choice(cfg.STORY_HEIGHTS)
            span_x = np.random.choice(cfg.SPAN_WIDTHS)
            span_y = np.random.choice(cfg.SPAN_WIDTHS)
            
            # 跨数: 独立随机 (2~MAX_SPANS)，避免单跨薄弱结构，同时保证跨数×跨度 <= 60m
            max_spans_by_plane_x = max(1, int(cfg.SPACE_X // span_x))
            max_spans_by_plane_y = max(1, int(cfg.SPACE_Y // span_y))
            num_spans_x = np.random.randint(2, min(max_spans_by_plane_x, MAX_SPANS) + 1)
            num_spans_y = np.random.randint(2, min(max_spans_by_plane_y, MAX_SPANS) + 1)
            
            # 随机平面形状 (避免薄弱连接):
            #   - C/U 形: 横条(连接左右翼)中间段 = nx-2 需 >=2 跨 -> nx>=4
            #   - T/L 形: 至少 2×2
            #   - rect : 任意
            if num_spans_x >= 4 and num_spans_y >= 2:
                plane_shape = np.random.choice(['rect', 'T', 'L', 'C', 'U'])
            elif num_spans_x >= 2 and num_spans_y >= 2:
                plane_shape = np.random.choice(['rect', 'T', 'L'])
            else:
                plane_shape = 'rect'
            plane_shape_id = {'rect': 0, 't': 1, 'l': 2, 'c': 3, 'u': 4}[plane_shape.lower()]
            
            # 每层随机楼面荷载 (10/15/20/25 kPa)，不同楼层可能不同
            floor_loads = generate_floor_loads(num_stories, cfg.FLOOR_LOAD_OPTIONS)
            # 由楼层面积计算每层节点质量 (kg) — 按形状算面积/节点数
            floor_node_masses = compute_floor_node_masses(
                num_stories, num_spans_x, num_spans_y,
                span_x, span_y, floor_loads, plane_shape=plane_shape)
            # 平均节点质量 (兼容旧字段，供 params 使用)
            mass_per_node = float(np.mean(floor_node_masses))
            
            # 阻尼比 (0.03~0.06)
            damping_ratio = np.random.uniform(0.03, 0.06)
            
            params.append([
                float(num_stories),
                float(num_spans_x),
                float(num_spans_y),
                float(span_x),
                float(span_y),
                float(story_height),
                float(mass_per_node),
                float(damping_ratio),
                float(plane_shape_id)
            ])
            floor_loads_list.append(np.asarray(floor_loads, dtype=np.float32))
            floor_node_masses_list.append(np.asarray(floor_node_masses, dtype=np.float32))
        
        return (np.array(params, dtype=np.float32),
                floor_loads_list,
                floor_node_masses_list)
    
    def run_simulation_parallel(self, params, motion_indices, floor_node_masses_list=None):
        """
        使用多进程并行运行仿真
        
        Args:
            params: [N, 8] 结构参数
            motion_indices: [N] 地震动索引
            floor_node_masses_list: [N] 每样本每层节点质量 (kg)
        
        Returns:
            dict: 包含所有仿真结果
        """
        n_samples = len(params)
        seq_len = self.config.get_seq_len()
        
        # 每样本目标 PGA (g): 在 PGA_RANGE 内随机, 增强样本多样性
        pga_range = getattr(self.config, 'PGA_RANGE', (0.1, 0.2))
        pga_targets = np.random.uniform(pga_range[0], pga_range[1], n_samples)
        
        # 准备任务参数
        sim_config = SimConfig3D()
        task_args = []
        for i in range(n_samples):
            motion = self.motion_pool[motion_indices[i]]
            f_masses = floor_node_masses_list[i] if floor_node_masses_list is not None else None
            task_args.append((
                params[i],
                motion,
                seq_len,
                self.config.TARGET_DT,
                sim_config.COL_SECTIONS,
                sim_config.BEAM_SECTION,
                sim_config.E,
                f_masses,
                pga_targets[i]
            ))
        
        # 并行执行
        print(f"\n  ⚙️ 启动 {self.max_workers} 个进程并行仿真...")
        print(f"     总任务数: {n_samples}")
        
        all_displacements = []
        all_heights = []
        all_E_avg = []
        all_params = []
        all_motions = []
        all_pgas = []
        all_node_peak_disp = []
        all_elem_peak_force = []
        all_node_coords = []
        all_elem_geom = []
        failed_count = 0
        
        start_time = time.time()
        
        # 使用 ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = {executor.submit(run_single_simulation, args): i for i, args in enumerate(task_args)}
            
            # 使用 tqdm 显示进度
            with tqdm(total=n_samples, desc="  仿真进度") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=300)  # 5分钟超时
                        all_displacements.append(result['displacement'])
                        all_motions.append(result['motion'])
                        all_heights.append(result['height'])
                        all_E_avg.append(result['E_avg'])
                        all_params.append(result['params'])
                        all_pgas.append(result['pga'])
                        all_node_peak_disp.append(result.get('node_peak_disp', {}))
                        all_elem_peak_force.append(result.get('elem_peak_force', {}))
                        all_node_coords.append(result.get('node_coords', {}))
                        all_elem_geom.append(result.get('elem_geom', {}))
                        if result['failed']:
                            failed_count += 1
                    except Exception as e:
                        # 超时或异常，使用零向量 (motion 用该任务实际分配的波)
                        idx = futures[future]
                        motion_fallback = task_args[idx][1]
                        all_displacements.append(np.zeros(seq_len, dtype=np.float32))
                        all_motions.append(np.asarray(motion_fallback, dtype=np.float32))
                        all_heights.append(0.0)
                        all_E_avg.append(30.0)
                        all_params.append(np.zeros(8, dtype=np.float32))
                        all_pgas.append(pga_targets[idx])
                        all_node_peak_disp.append({})
                        all_elem_peak_force.append({})
                        all_node_coords.append({})
                        all_elem_geom.append({})
                        failed_count += 1
                    pbar.update(1)
        
        elapsed = time.time() - start_time
        
        print(f"  ✅ 仿真完成，用时 {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
        print(f"     成功: {n_samples - failed_count}, 失败: {failed_count}")
        pga_arr = np.array(all_pgas, dtype=np.float32)
        print(f"     目标PGA范围: [{pga_arr.min():.4f}, {pga_arr.max():.4f}] g")
        
        return {
            'displacements': np.array(all_displacements, dtype=np.float32),
            'motions': np.array(all_motions, dtype=np.float32),
            'heights': np.array(all_heights, dtype=np.float32),
            'E_avg': np.array(all_E_avg, dtype=np.float32),
            'params': np.array(all_params, dtype=np.float32),
            'pga': pga_arr,
            'failed': failed_count,
            'elapsed': elapsed,
            'node_peak_disp': all_node_peak_disp,
            'elem_peak_force': all_elem_peak_force,
            'node_coords': all_node_coords,
            'elem_geom': all_elem_geom
        }
    
    def generate_or_load(self, earthquake_folder=None, force_regen=False):
        """
        生成或加载仿真数据（主入口）
        
        Args:
            earthquake_folder: 地震动文件夹路径
            force_regen: 是否强制重新生成
        
        Returns:
            cache: SimulationCache 对象
        """
        print("\n" + "="*70)
        print("仿真数据生成/加载 (多进程并行)")
        print("="*70)
        
        # 1. 尝试加载缓存 (优先复用已有仿真数据, 避免浪费已算好的结果)
        #    只有 force_regen=True 或缓存不存在时才重新生成。
        #    样本数不一致时由调用方 (run_pipeline) 显式决定是否 force_regen。
        if not force_regen and self.cache.load():
            print("  ✅ 使用已有仿真缓存")
            return self.cache
        
        # 2. 需要生成新数据
        print("  🚀 开始生成仿真数据...")
        
        # 2.1 加载地震动
        self.load_motions(earthquake_folder)
        
        # 2.2 生成结构参数 (含每层荷载/节点质量)
        print(f"\n  📐 生成 {self.config.NUM_SIMULATIONS} 组结构参数 (含每层荷载多样性)...")
        params, floor_loads_list, floor_node_masses_list = self.generate_structure_params(
            self.config.NUM_SIMULATIONS)
        
        # 2.3 分配地震动: 有限波形池循环复用 (关键)
        #     目的: 让"同一条波"配多个结构, 使模型能学到"波形->结构响应"的泛化,
        #     而不是每个样本一条随机波(验证集全是未见波形, 无法外推)。
        #     策略: 只用前 N_WAVE_SLOTS 条波, 循环重复覆盖所有样本。
        n_wave_slots = int(getattr(self.config, 'N_WAVE_SLOTS', 0)) or min(
            self.num_waves, max(8, self.config.NUM_SIMULATIONS // 4))
        if n_wave_slots <= 0:
            n_wave_slots = 1
        motion_indices = np.arange(self.config.NUM_SIMULATIONS) % n_wave_slots
        # 打乱顺序 (避免所有同波样本聚集)
        np.random.shuffle(motion_indices)
        print(f"  📡 波形分配: {n_wave_slots} 条波形循环复用 "
              f"(每波平均 {self.config.NUM_SIMULATIONS/n_wave_slots:.1f} 个结构)")
        
        # 2.4 并行运行仿真
        print(f"\n  ⚙️ 运行 OpenSees 仿真 (并行)...")
        results = self.run_simulation_parallel(params, motion_indices, floor_node_masses_list)
        
        # 2.5 存入缓存
        self.cache.append(
            results['params'],
            results['displacements'],
            results['heights'],
            results['E_avg'],
            motion_indices,
            floor_loads=floor_loads_list,
            floor_node_masses=floor_node_masses_list,
            new_motions=results.get('motions', None),
            new_pgas=results.get('pga', None),
            new_node_peak_disp=results.get('node_peak_disp', None),
            new_elem_peak_force=results.get('elem_peak_force', None),
            new_node_coords=results.get('node_coords', None),
            new_elem_geom=results.get('elem_geom', None)
        )
        
        # 2.6 保存到文件
        self.cache.save()
        
        # 2.7 打印统计
        stats = self.cache.get_stats()
        if stats:
            print(f"\n  📊 仿真统计:")
            print(f"     成功: {stats['num_samples'] - results['failed']}")
            print(f"     失败: {results['failed']}")
            print(f"     位移范围: [{stats['disp_min']:.2f}, {stats['disp_max']:.2f}] mm")
            print(f"     高度范围: {stats['height_range']}")
            print(f"     总用时: {results['elapsed']:.1f} 秒")
        
        return self.cache


# ============================================================
# 测试
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("仿真缓存测试 (多进程并行)")
    print("="*60)
    
    import sys
    sys.path.append('.')
    
    cfg = Config()
    cfg.NUM_SIMULATIONS = 20  # 测试用
    
    generator = SimulationGenerator(cfg)
    cache = generator.generate_or_load(force_regen=True)
    
    print(f"\n缓存样本数: {cache.num_samples}")
    stats = cache.get_stats()
    if stats:
        print(f"位移均值: {stats['disp_mean']:.2f} mm")