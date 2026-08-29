# dataset.py - 完整修复版（适配新梁格式）

"""
数据集模块
功能：
1. 从仿真缓存读取原始数据
2. 动态生成体素矩阵 → 八叉树特征
3. 缓存八叉树特征（不缓存体素）
4. 完整的数据验证和错误处理
"""
import torch
import numpy as np
import pickle
import os
from torch.utils.data import Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from config import Config
from generate_frames import generate_fixed_frame  # 直接使用
# 注: voxel_converter / octree_encoder / simulation_cache 已改为延迟导入
# (数据库模式用 frame_features, 不再依赖体素/八叉树构建/pkl缓存)


# ============================================================
# 按楼层数分层抽样 (保证抽取样本的楼层数分布均匀)
# ============================================================
def _stratified_sample_by_stories(rows, max_n, rng=None):
    """
    从样本行列表中按 num_stories 分层均匀抽样 max_n 条。

    策略:
      - 按 num_stories 分组
      - 每组均匀分配配额 (max_n / 组数, 余数分给样本多的组)
      - 若某组样本数不足配额, 全取该组, 剩余配额重新平均分给其它组 (自适应)
      - 组内用 rng 随机抽样, 保证可复现

    Args:
        rows: query_samples() 返回的行列表 (每行含 num_stories 字段)
        max_n: 需要抽取的样本数
        rng: np.random.Generator (固定种子)

    Returns:
        抽取后的行列表 (长度约为 max_n, 若总样本不足则全部)
    """
    rng = rng or np.random.default_rng(42)
    # 按楼层分组
    groups = {}
    for r in rows:
        ns = int(r.get('num_stories', 0))
        groups.setdefault(ns, []).append(r)
    # 楼层排序
    story_keys = sorted(groups.keys())
    # 自适应配额分配: 先均匀分配, 不足的组全取, 剩余再分
    selected = []
    quota_remaining = max_n
    remaining_groups = dict(groups)
    # 先计算每层应有配额 (均匀)
    while quota_remaining > 0 and remaining_groups:
        n_groups = len(remaining_groups)
        # 本轮每层配额 (下限1)
        per = max(1, quota_remaining // n_groups)
        for ns in list(remaining_groups.keys()):
            pool = remaining_groups[ns]
            take = min(per, len(pool))
            chosen = list(rng.choice(len(pool), size=take, replace=False))
            selected.extend(pool[i] for i in chosen)
            # 从池中移除已选
            remaining_groups[ns] = [x for i, x in enumerate(pool) if i not in chosen]
            quota_remaining -= take
            if len(remaining_groups[ns]) == 0:
                del remaining_groups[ns]
        # 若所有组都取完但仍不够 max_n (总样本不足), 结束
        if len(selected) >= max_n or (not remaining_groups and quota_remaining > 0):
            break
    # 精确裁剪到 max_n (若超出)
    if len(selected) > max_n:
        selected = list(rng.choice(selected, size=max_n, replace=False))
    return selected


# ============================================================
# 多维分层均匀抽样 (在按楼层均匀基础上, 叠加响应量级 + 结构形态)
# ============================================================
def _discretize(value, bins):
    """把连续值分到 [0, len(bins)-1] 档 (bins 为升序边界, 首尾为 -inf/+inf)"""
    for i, b in enumerate(bins):
        if value < b:
            return i
    return len(bins)


def _sample_key(r, peak_bins=None, struct_bins=None):
    """构造多维分层 key: (num_stories, 响应量级档, 结构形态档)

    - 响应量级: disp_peak 按分位数档 (peak_bins=[q33,q67] 或
      peak_bins 为 dict {num_stories: [q33,q67]} 按每层组内分位,
      传None则只按楼层)
    - 结构形态: 跨度/层高/跨数 合并分档 (struct_bins 为可选 dict)
    """
    ns = int(r.get('num_stories', 0))
    key = (ns,)
    if peak_bins is not None:
        peak = float(r.get('disp_peak', 0.0))
        if isinstance(peak_bins, dict):
            bins = peak_bins.get(ns)
            if bins is not None:
                key += (_discretize(peak, bins),)
        else:
            key += (_discretize(peak, peak_bins),)
    if struct_bins is not None:
        # 结构形态: 用小跨度/大跨度 × 低层高/高层高 × 单跨/多跨 的组合
        sx = float(r.get('span_x', 0.0))
        sh = float(r.get('story_height', 0.0))
        nx = int(r.get('num_bays_x', 0))
        span_bin = 0 if sx <= struct_bins.get('span_split', 5.0) else 1
        height_bin = 0 if sh <= struct_bins.get('height_split', 3.5) else 1
        bay_bin = 0 if nx <= 2 else 1   # 单/双跨 vs 多跨
        key += (span_bin, height_bin, bay_bin)
    return key


def _stratified_sample_uniform(rows, max_n, rng=None,
                               peak_bins=None, struct_bins=None):
    """多维分层均匀抽样 (自适应配额)

    在按 num_stories 均匀的基础上, 可选叠加:
      - 响应量级分档 (peak_bins): 大/小变形均匀
      - 结构形态分档 (struct_bins): 跨度/层高/跨数组合均匀

    分层 key = (num_stories, [peak_bin], [span_bin, height_bin, bay_bin])
    每层自适应均匀配额 (不足的组全取, 剩余配额均分给其它组)。

    Args:
        rows: query_samples() 返回的行列表
        max_n: 目标样本数
        rng: np.random.Generator (固定种子)
        peak_bins: 响应量级分位数边界列表 (如 [q33, q67]); None=不按响应分层
        struct_bins: dict {'span_split','height_split'} 或 None=不按结构分层

    Returns:
        抽取后的行列表
    """
    rng = rng or np.random.default_rng(42)
    if peak_bins is not None:
        # 自动计算 disp_peak 分位数边界 (若未提供)
        if not peak_bins:
            # 按楼层组内分位: 保证每层内部低/中/高变形都均匀
            # (全局分位会让低层结构全部落在低变形档, 牺牲楼层均匀)
            peak_map = {}
            for r in rows:
                ns = int(r.get('num_stories', 0))
                peak_map.setdefault(ns, []).append(float(r.get('disp_peak', 0.0)))
            peak_bins = {}
            for ns, vals in peak_map.items():
                vals = np.array(vals)
                if vals.std() > 1e-12:
                    peak_bins[ns] = [float(x) for x in np.percentile(vals, [33.3, 66.7])]
                else:
                    peak_bins[ns] = None   # 该层全等 -> 不细分
            # 若所有层都无法分档, 退回不按响应分层
            if all(v is None for v in peak_bins.values()):
                peak_bins = None
    if struct_bins is not None:
        # 自动取跨度/层高中位作为分界
        if 'span_split' not in struct_bins:
            spans = np.array([float(r.get('span_x', 0.0)) for r in rows])
            struct_bins['span_split'] = float(np.median(spans))
        if 'height_split' not in struct_bins:
            hs = np.array([float(r.get('story_height', 0.0)) for r in rows])
            struct_bins['height_split'] = float(np.median(hs))

    # 按 key 分组
    groups = {}
    for r in rows:
        k = _sample_key(r, peak_bins, struct_bins)
        groups.setdefault(k, []).append(r)

    # 自适应配额 (与 _stratified_sample_by_stories 相同的 while 循环逻辑)
    selected = []
    quota_remaining = max_n
    remaining_groups = dict(groups)
    while quota_remaining > 0 and remaining_groups:
        n_groups = len(remaining_groups)
        per = max(1, quota_remaining // n_groups)
        for k in list(remaining_groups.keys()):
            pool = remaining_groups[k]
            take = min(per, len(pool))
            chosen = list(rng.choice(len(pool), size=take, replace=False))
            selected.extend(pool[i] for i in chosen)
            remaining_groups[k] = [x for i, x in enumerate(pool) if i not in chosen]
            quota_remaining -= take
            if len(remaining_groups[k]) == 0:
                del remaining_groups[k]
        if len(selected) >= max_n or (not remaining_groups and quota_remaining > 0):
            break
    if len(selected) > max_n:
        selected = list(rng.choice(selected, size=max_n, replace=False))
    return selected


# ============================================================
# 均匀抽样前的样本筛选 (训练/可视化共用同一套筛选逻辑)
# 1) PGA 筛选: 只保留 target_pga == DB_FILTER_PGA 的样本 (None=不过滤)
# 2) 位移超标剔除: disp_peak(mm)/1000 > total_height(m) × DB_MAX_DRIFT_RATIO
# ============================================================
def filter_rows_for_training(rows, config=None):
    """按训练规则筛选样本行 (PGA 筛选 + 顶点位移超标剔除)

    Args:
        rows: query_samples() 返回的行列表
        config: Config 实例 (默认 Config())

    Returns:
        筛选后的行列表 (可能为空)
    """
    if config is None:
        config = Config()

    # ---- 1) PGA 筛选 ----
    pga_f = getattr(config, 'DB_FILTER_PGA', None)
    if pga_f is not None:
        kept_rows = []
        n_drop = 0
        for r in rows:
            tp = float(r.get('target_pga') or 0.0)
            if abs(tp - float(pga_f)) > 1e-9:
                n_drop += 1
                continue
            kept_rows.append(r)
        if n_drop > 0:
            print(f"  🎚 只保留 target_pga={float(pga_f):.2f}g 的样本, "
                  f"剔除 {n_drop} 个 ({len(rows)} -> {len(kept_rows)})")
        rows = kept_rows
        if not rows:
            print(f"  ❌ 无 target_pga={float(pga_f):.2f}g 的可用样本")
            return rows

    # ---- 2) 顶点位移超标剔除 ----
    if getattr(config, 'DB_FILTER_MAX_DRIFT', True):
        ratio_th = float(getattr(config, 'DB_MAX_DRIFT_RATIO', 0.005))
        kept_rows = []
        n_drop = 0
        for r in rows:
            peak_mm = float(r.get('disp_peak') or 0.0)
            h_m = float(r.get('total_height') or 0.0)
            # 顶点最大位移换算成 与总高度的比值 (无位移/高度数据时保守保留)
            if peak_mm <= 0 or h_m <= 0:
                kept_rows.append(r)
                continue
            drift = (peak_mm / 1000.0) / h_m
            if drift > ratio_th:
                n_drop += 1
                continue
            kept_rows.append(r)
        if n_drop > 0:
            print(f"  🚫 剔除顶点位移>总高{ratio_th*100:.1f}% 的样本 {n_drop} 个 "
                  f"({len(rows)} -> {len(kept_rows)})")
        rows = kept_rows

    return rows


# ============================================================
# 体素化特征: 结构 -> 3D体素 -> 八叉树紧缩 (研究用, 替代杆系)
# ============================================================
def _voxel_features_from_struct(struct, depth=5, vocab=None,
                                enc_mode=None):
    """从结构字段重建杆系模型 -> 微元特征 (三种编码模式).

    与 frame_model.build_frame_model 一致 (用数据库真实 col_sections/beam_sections),
    不再体素化 (300×300×500)。固定 32×32×32 网格, 每格真实 2m (原点对齐)。

    三种编码模式 (enc_mode):
      'token'  : 微元 token ID (查词表, 不依赖样本连续量) — 默认, nn.Embedding 学习
                 token 0 = 空, 1..V-1 = 微元类型 (combo+柱截面档+梁截面档+偏位档)
                 返回展平 [32³] = [32768] int
      'direct' : 直接 32 位整数编码归一化 (上一版本"简单直接编码")
                 返回展平 [32³] = [32768] float (÷2^32 归一化到 [0,1])
      'cont'   : 每格 6 通道连续物理量 (类型梯度/柱EI/梁EI/密度/节点偏位)
                 返回展平 [32³×6] = [196608] float (LLM embedding 启发)

    Args:
        vocab: VoxelVocab 实例 (None 时从 config.VOXEL_VOCAB_FILE 加载; token 模式用)
        enc_mode: 'token' / 'direct' / 'cont'; None 时按 USE_VOXEL_TOKEN 决定
    """
    from frame_model import build_frame_model
    from frame_grid_encoder import encode_frame_grid, VoxelVocab, \
        encode_frame_grid_features
    from config import Config
    if enc_mode is None:
        enc_mode = 'token' if getattr(Config, 'USE_VOXEL_TOKEN', True) else 'cont'

    model = build_frame_model(struct=struct)

    if enc_mode == 'cont':
        # 连续物理量特征 (LLM embedding 启发): 每格 6 通道, 相似格子距离近
        feats = encode_frame_grid_features(model)   # [32,32,32,6]
        return feats.reshape(-1).astype(np.float32)

    codes, _ = encode_frame_grid(model)

    if enc_mode == 'direct':
        # 直接编码: 128 位整数归一化到 [0,1] (MLP 学习; object 数组逐元素转)
        codes_flat = [int(c) for c in codes.reshape(-1)]
        return (np.array(codes_flat, dtype=np.float64) / float(2 ** 128)) \
            .astype(np.float32)

    # ---- token 模式 ----
    if vocab is None:
        vf = getattr(Config, 'VOXEL_VOCAB_FILE', None)
        vocab = VoxelVocab()
        if vf and os.path.exists(vf):
            vocab.load(vf)
        else:
            # 无词表: 直接用 codes 的整数值作为 token (回退)
            codes_flat = [int(c) for c in codes.reshape(-1)]
            return (np.array(codes_flat, dtype=np.float64) / float(2 ** 128)) \
                .astype(np.float32)
    tok = vocab.encode_codes(codes)
    return tok.reshape(-1).astype(np.float32)


# ============================================================
# 全局工作函数：多进程生成单个样本的体素化特征 (数据库模式)
# struct 为普通 dict, 可被 multiprocessing 序列化
# ============================================================
def _compute_voxel_feature_single(args):
    """
    单个样本的体素化特征计算（在子进程中执行）

    Args:
        args: (struct, depth, enc_mode)
            struct: db_manager.get_structure() 返回的字典
            depth : 八叉树紧缩深度
            enc_mode: 'token' / 'direct' / 'cont'

    Returns:
        np.ndarray: 特征向量 (失败时返回 None)
    """
    struct, depth, enc_mode = args
    try:
        return _voxel_features_from_struct(struct, depth, enc_mode=enc_mode)
    except Exception:
        return None


def _compute_voxel_features_parallel(structs, depth=5, max_workers=None,
                                     enc_mode=None):
    """
    多进程并行计算体素化特征, 带进度条

    Args:
        structs: 结构字典列表
        depth: 八叉树紧缩深度 (4~7)
        max_workers: 并行进程数 (默认 config.MAX_OCTREE_WORKERS)
        enc_mode: 'token' / 'direct' / 'cont'

    Returns:
        (features, failed):
            features: list, 与 structs 顺序一一对应 (失败为 None)
            failed  : 失败样本数
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    from config import Config

    n = len(structs)
    if n == 0:
        return [], 0

    if max_workers is None:
        max_workers = min(mp.cpu_count(),
                          getattr(Config, 'MAX_OCTREE_WORKERS', 8))
    max_workers = max(1, max_workers)

    # 样本量很少时退化为串行 (进程池开销反而更大)
    if n <= max_workers:
        feats = []
        failed = 0
        for s in tqdm(structs, desc="  切杆系编码", unit="样本"):
            f = _compute_voxel_feature_single((s, depth, enc_mode))
            if f is None:
                failed += 1
            feats.append(f)
        return feats, failed

    print(f"  🖥️ 使用 {max_workers} 个进程并行切杆系编码 ({n} 个样本, "
          f"深度 {depth} = {2**depth}³ 格, 64m空间)")

    results = [None] * n
    failed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_compute_voxel_feature_single, (s, depth, enc_mode)): i
                   for i, s in enumerate(structs)}
        with tqdm(total=n, desc="  切杆系编码", unit="样本") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = None
                if results[idx] is None:
                    failed += 1
                pbar.update(1)

    return results, failed


# ============================================================
# 全局工作函数：多进程生成单个样本的八叉树特征
# ============================================================
def _compute_octree_single(args):
    """
    单个样本的八叉树特征计算（在子进程中执行）

    Args:
        args: (p, displacement, octree_depth, floor_node_masses)

    Returns:
        dict: octree_feat / displacement / height / E_avg / params
    """
    p, displacement, octree_depth, floor_node_masses = args

    # 延迟导入 (仅 pkl 旧模式需要; 数据库模式不触发此函数)
    from octree_encoder import OctreeBuilder, frame_to_voxel
    builder = OctreeBuilder(max_depth=octree_depth)

    num_stories = int(p[0])
    num_bays_x = int(p[1])
    num_bays_y = int(p[2])
    bay_width_x = float(p[3])
    bay_width_y = float(p[4])
    story_height = float(p[5])
    # 平面形状 (第 8 位 ID, 兼容旧 8 维)
    from generate_frames import id_to_shape
    plane_shape = id_to_shape(p[8]) if len(p) > 8 else 'rect'

    # 根据跨度估算梁截面 (200mm 模数)
    max_span = max(bay_width_x, bay_width_y)
    beam_height = max(0.4, min(max_span / 12, 0.8))
    beam_height = round(beam_height / 0.2) * 0.2
    beam_width = max(0.2, min(beam_height / 2.5, 0.5))
    beam_width = round(beam_width / 0.2) * 0.2

    try:
        frame_params = generate_fixed_frame(
            num_stories=num_stories,
            num_spans_x=num_bays_x,
            num_spans_y=num_bays_y,
            span_x=bay_width_x,
            span_y=bay_width_y,
            story_height=story_height,
            axis_ratio=0.6,
            beam_width=beam_width,
            beam_height=beam_height,
            plane_shape=plane_shape
        )
        voxel, mass_voxel, stiff_voxel = frame_to_voxel(
            frame_params, floor_node_masses=floor_node_masses)
    except Exception:
        # 体素生成失败 -> 空特征
        cfg = Config()
        voxel = np.zeros(cfg.get_grid_dims(), dtype=np.float32)
        mass_voxel = None
        stiff_voxel = None

    # 八叉树紧缩特征 (质量/刚度/质心/刚心/偏心/占比/各向异性 10通道, 按Z分层聚合)
    octree_feat = builder.build_features_v2(
        voxel, mass_voxel, stiff_voxel)

    return {
        'octree_feat': octree_feat,
        'displacement': displacement,
        'height': np.float32(num_stories * story_height),
        'E_avg': np.float32(30.0),
        'params': p,
    }


class OctreeDataset(Dataset):
    """
    八叉树数据集
    
    数据流:
        仿真缓存 (原始参数+位移) 
        → 重建结构 → 生成体素矩阵 [300,300,500] 
        → 八叉树特征 [96]
        → 缓存八叉树特征 (不缓存体素)
    """
    
    def __init__(self, config=None, force_regen_octree=False, use_db=False,
                 use_voxel_feature=False, voxel_depth=5, voxel_enc_mode=None,
                 db_filter=None):
        self.config = config or Config()
        # 数据子集过滤 (仅数据库模式): dict(plane_shape=..., floor_load_kpa=...)
        # 传给 db.query_samples (None=不过滤, 用全部样本)
        self.db_filter = db_filter or {}
        self.use_voxel_feature = use_voxel_feature
        self.voxel_depth = voxel_depth
        # 切杆系编码模式: 'token' / 'direct' / 'cont'
        #   None -> 由 USE_VOXEL_TOKEN 决定 (token: USE_VOXEL_TOKEN=True; cont: False)
        self.voxel_enc_mode = voxel_enc_mode
        if use_voxel_feature:
            if self.voxel_enc_mode is None:
                self.voxel_enc_mode = ('token' if getattr(self.config, 'USE_VOXEL_TOKEN', False)
                                       else 'cont')
            if self.voxel_enc_mode == 'token':
                # 体素 token 模式: 每格一个离散微元 token ID (查词表)
                # 维度 = 64³ = 262144 (1m/格, 64m空间; LLM embedding 层学习映射)
                grid = int(getattr(self.config, 'VOXEL_GRID', 64))
                self.config.FRAME_FEATURE_DIM = grid ** 3
                print(f"  🧱 微元 token 特征模式: 网格{grid}³ -> 特征维度 "
                      f"{self.config.FRAME_FEATURE_DIM} "
                      f"({grid}³ 微元 token, nn.Embedding 学习)")
            elif self.voxel_enc_mode == 'direct':
                # 直接 128 位整数编码归一化 (MLP 学习)
                grid = int(getattr(self.config, 'VOXEL_GRID', 64))
                self.config.FRAME_FEATURE_DIM = grid ** 3
                print(f"  🧱 直接编码特征模式: 网格{grid}³ -> 特征维度 "
                      f"{self.config.FRAME_FEATURE_DIM} "
                      f"({grid}³ 位整数编码归一化, 简单直接编码)")
            else:
                # 体素特征维度 = grid³ × C: 64×64×64 连续物理量特征
                # 每格 C=6 通道 (类型梯度/柱EI/梁EI/密度/节点偏位), 相似格子距离近
                from frame_grid_encoder import FEAT_C
                grid = int(getattr(self.config, 'VOXEL_GRID', 64))
                self.config.FRAME_FEATURE_DIM = grid ** 3 * FEAT_C
                print(f"  🧱 切杆系连续特征模式: 网格{grid}³ -> 特征维度 "
                      f"{self.config.FRAME_FEATURE_DIM} "
                      f"(32×32×32×{FEAT_C} 连续物理量, 相似格子距离近)")
        # 八叉树构建器仅在 pkl 模式需要; 延迟创建 (数据库模式不依赖)
        self.octree_builder = None
        self.seq_len = self.config.get_seq_len()
        
        self.grid_x, self.grid_y, self.grid_z = self.config.get_grid_dims()
        
        self.octree_features = None
        self.frame_features = None   # 杆系结构化物理特征 (替代体素八叉树, 默认使用)
        self.displacements = None
        self.motions = None         # 每样本实际输入地震动加速度 (g) [N, T]
        self.heights = None
        self.E_avg = None
        self.params = None
        self.floor_node_masses = None   # 每层节点质量 (kg) list of arrays
        self.meta = None                # [N] 每样本附加信息 dict (shape_type/pga/截面等)
        
        self.num_samples = 0
        self.is_loaded = False
        
        os.makedirs(os.path.dirname(self.config.OCTREE_CACHE_FILE), exist_ok=True)

        # 数据库模式: 直接从 PostgreSQL 读取 (替代 pkl 缓存)
        if use_db:
            if self._load_from_db():
                print(f"  ✅ 从数据库加载数据集: {self.num_samples} 个样本")
            else:
                print("  ❌ 数据库无数据, 请先运行 db_generate_samples.py 生成样本")
            return

        # pkl 缓存模式: 加载八叉树缓存; 若缓存不可用则自动回退数据库
        loaded = False
        if not force_regen_octree and self._load_octree_cache():
            print(f"  ✅ 加载八叉树缓存: {self.num_samples} 个样本")
            loaded = True
        else:
            # 尝试从 pkl 仿真缓存生成; 失败 (缓存不存在/为空) 时回退数据库
            if self._generate_octree_cache():
                loaded = True
            elif self._load_from_db():
                print(f"  ✅ 八叉树缓存不可用, 已回退从数据库加载: {self.num_samples} 个样本")
                loaded = True
            else:
                print("  ❌ 无可用数据: pkl 缓存与数据库均为空")

        if not loaded:
            self.num_samples = 0
            self.is_loaded = False

    def _load_from_db(self):
        """从 PostgreSQL 数据库加载样本/结构/波/响应 (数据库模式)"""
        try:
            from db_manager import SLFDatabase
            from dataset_db import SLFDbDataset
        except ImportError as e:
            print(f"  ⚠️ 数据库模块导入失败: {e}")
            return False
        # 复用 SLFDbDataset 的特征重建逻辑 (静态方法)
        _params_from_struct = SLFDbDataset._params_from_struct
        _frame_features = SLFDbDataset._frame_features

        db = SLFDatabase()
        rows = db.query_samples(
            plane_shape=self.db_filter.get('plane_shape'),
            floor_load_kpa=self.db_filter.get('floor_load_kpa'))
        if not rows:
            return False

        # ------------------------------------------------------------
        # 均匀抽样前的样本筛选: PGA 筛选 + 顶点位移超标剔除
        # (共用模块级 filter_rows_for_training, 与可视化/分析口径一致)
        # ------------------------------------------------------------
        rows = filter_rows_for_training(rows, self.config)
        if not rows:
            return False

        # 随机抽取子集: 数据库样本太多时只取一部分 (config.DB_MAX_SAMPLES, 固定种子可复现)
        # 默认按楼层数分层抽样 (num_stories 均匀分散), 避免某楼层过多/过少
        n_rows = len(rows)
        max_n = int(getattr(self.config, 'DB_MAX_SAMPLES', 0) or 0)
        if max_n > 0 and n_rows > max_n:
            seed = int(getattr(self.config, 'DB_SAMPLE_SEED', 42))
            rng = np.random.default_rng(seed)
            if getattr(self.config, 'DB_STRATIFY_STORIES', True):
                # 多维分层均匀抽样: 楼层均匀 + (可选)响应量级 + (可选)结构形态
                n_peak = int(getattr(self.config, 'DB_STRATIFY_RESPONSE_BINS', 3) or 0)
                use_struct = bool(getattr(self.config, 'DB_STRATIFY_STRUCT', True))
                peak_bins = None if n_peak < 2 else []
                struct_bins = {} if use_struct else None
                rows = _stratified_sample_uniform(rows, max_n, rng,
                                                  peak_bins=peak_bins,
                                                  struct_bins=struct_bins)
                # 统计每层数量 + 响应量级分档分布
                nstories = {}
                for r in rows:
                    nstories[r['num_stories']] = nstories.get(r['num_stories'], 0) + 1
                dist = ", ".join(f"{k}层:{v}" for k, v in sorted(nstories.items()))
                label = "楼层分层"
                if n_peak >= 2:
                    label += "+响应量级"
                if use_struct:
                    label += "+结构形态"
                print(f"  🎲 数据库样本过多 ({n_rows}), 按{label}抽 {len(rows)} 个 "
                      f"(seed={seed}) [{dist}]")
            else:
                keep = set(rng.choice(n_rows, size=max_n, replace=False).tolist())
                rows = [r for i, r in enumerate(rows) if i in keep]
                print(f"  🎲 数据库样本过多 ({n_rows}), 随机抽取 {len(rows)} 个训练 (seed={seed})")

        # 逐样本组装 (与 pkl 模式相同字段)
        params_list, disp_list, motion_list = [], [], []
        height_list, E_list = [], []
        ff_list = []
        fm_list = []
        meta_list = []
        voxel_struct_ids = []      # 体素化样本的 struct_id (缓存键用)
        voxel_structs = []         # 体素化样本的结构字典 (并行编码用)
        for r in rows:
            sid = r['sample_id']
            resp = db.get_sample(sid)
            if resp is None or resp.get('roof_disp') is None:
                continue
            struct = db.get_structure(resp['struct_id'])
            if struct is None:
                continue
            gm = db.get_ground_motion(resp['gm_id'])
            if gm is None or gm.get('motion') is None:
                continue

            params_list.append(_params_from_struct(struct))
            disp_list.append(resp['roof_disp'])
            motion_list.append(gm['motion'])
            height_list.append(float(struct['total_height']))
            E_list.append(30.0)
            if self.use_voxel_feature:
                # 体素化编码 (研究用): 3D体素 -> 八叉树紧缩 (并行计算)
                voxel_structs.append(struct)
                voxel_struct_ids.append(int(resp['struct_id']))
                ff_list.append(None)  # 占位, 并行完成后回填
            else:
                ff_list.append(_frame_features(struct))
            fm_list.append(struct.get('floor_masses') or [])
            # 组装 meta (供报告/可视化显示真实 PGA 与截面参数)
            loads = struct.get('floor_loads') or []
            meta_list.append({
                'shape_type': 'rect',
                'load_per_area': float(np.mean(loads)) if loads else 20.0,
                'pga': float(resp.get('applied_pga') or resp.get('target_pga') or 0.0),
                'axis_ratio': 0.6,
                'beam_width': float(struct.get('beam_width', 0.3)),
                'beam_height': float(struct.get('beam_height', 0.6)),
                'slab_thickness': float(struct.get('slab_thickness', 0.2)),
                'num_stories': int(struct['num_stories']),
                'span_x': float(struct['span_x']),
                'span_y': float(struct['span_y']),
                'story_height': float(struct['story_height']),
                'num_cells': int(struct['num_bays_x'] * struct['num_bays_y']),
                'total_load': float(struct.get('total_mass_kg', 0.0) * 9.81 / 1000.0),
                'unique_key': str(sid),
            })

        # ---- 体素化模式: 优先读文件缓存(按struct), 缺的才并行算并写回 ----
        if self.use_voxel_feature and voxel_structs:
            depth = self.voxel_depth
            enc_mode = self.voxel_enc_mode or (
                'token' if getattr(self.config, 'USE_VOXEL_TOKEN', False) else 'cont')
            voxel_feats = self._load_voxel_cache(voxel_struct_ids, depth, enc_mode)
            if voxel_feats is None:
                # 无缓存/整体失效 -> 全量并行体素化
                voxel_feats, _ = _compute_voxel_features_parallel(
                    voxel_structs, depth=depth, enc_mode=enc_mode)
                voxel_feats = list(voxel_feats)
            else:
                # 部分命中: 只对缺失的结构并行补算
                missing_idx = [i for i, f in enumerate(voxel_feats) if f is None]
                if missing_idx:
                    missing_structs = [voxel_structs[i] for i in missing_idx]
                    miss_feats, miss_failed = _compute_voxel_features_parallel(
                        missing_structs, depth=depth, enc_mode=enc_mode)
                    for pos, i in enumerate(missing_idx):
                        voxel_feats[i] = miss_feats[pos]
                    if miss_failed > 0:
                        print(f"  ⚠️ 补算体素化失败 {miss_failed}/{len(missing_idx)} "
                              f"个 (已置空)")
            # 失败样本置零, 写回缓存 (增量合并, 保留旧结构)
            _vgrid = int(getattr(self.config, 'VOXEL_GRID', 64))
            if self.voxel_enc_mode == 'cont':
                from frame_grid_encoder import FEAT_C
                feat_dim = _vgrid ** 3 * FEAT_C
            else:   # token / direct: 64³ = 262144
                feat_dim = _vgrid ** 3
            feats_final = [
                f if f is not None
                else np.zeros(feat_dim, dtype=np.float32)
                for f in voxel_feats
            ]
            self._save_voxel_cache(voxel_struct_ids, depth, feats_final,
                                   enc_mode=enc_mode)
            # 按顺序回填到 ff_list 对应位置
            vi = 0
            for i in range(len(ff_list)):
                if ff_list[i] is None:
                    ff_list[i] = feats_final[vi]
                    vi += 1

        if not disp_list:
            return False

        self.params = np.asarray(params_list, dtype=np.float32)
        self.displacements = np.asarray(disp_list, dtype=np.float32)
        self.motions = np.asarray(motion_list, dtype=np.float32)
        self.heights = np.asarray(height_list, dtype=np.float32)
        self.E_avg = np.asarray(E_list, dtype=np.float32)
        self.frame_features = np.asarray(ff_list, dtype=np.float32)
        self.floor_node_masses = fm_list
        self.meta = meta_list
        # octree_features 兼容占位 (数据库模式用 frame_features)
        if getattr(self.config, 'USE_FRAME_FEATURE', False):
            self.octree_features = np.zeros((len(disp_list), 1), dtype=np.float32)
        else:
            self.octree_features = self.frame_features
        self.seq_len = self.displacements.shape[1]
        self.num_samples = len(disp_list)
        self.is_loaded = True
        return True
    
    def _load_octree_cache(self):
        cache_file = self.config.OCTREE_CACHE_FILE
        if not os.path.exists(cache_file):
            return False
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            required_keys = ['octree_features', 'displacements', 'heights', 'E_avg', 'params']
            for key in required_keys:
                if key not in data:
                    print(f"  ⚠️ 八叉树缓存缺少字段: {key}")
                    return False
            
            # 与仿真缓存一致性校验: 若样本数不匹配 (仿真已重新生成但八叉树未更新),
            # 视为缓存陈旧, 返回 False 触发重新生成
            try:
                from simulation_cache import SimulationCache  # 延迟导入 (pkl 模式)
                sim_cache = SimulationCache(self.config)
                if sim_cache.load():
                    sim_n = sim_cache.num_samples
                    oct_n = len(data['octree_features'])
                    if sim_n != oct_n:
                        print(f"  ⚠️ 八叉树缓存样本数({oct_n}) 与仿真缓存({sim_n}) 不一致, 重新生成")
                        return False
            except Exception as e:
                print(f"  ⚠️ 仿真缓存校验跳过: {e}")
            
            self.octree_features = data['octree_features']
            self.frame_features = data.get('frame_features', None)
            self.displacements = data['displacements']
            self.heights = data['heights']
            self.E_avg = data['E_avg']
            self.params = data['params']
            self.floor_node_masses = data.get('floor_node_masses', None)
            self.motions = data.get('motions', None)
            self.meta = data.get('meta', None)
            self.num_samples = len(self.octree_features)
            self.is_loaded = True
            
            expected_feat_dim = 4 * (2 ** self.config.OCTREE_DEPTH)
            if self.octree_features.shape[1] != expected_feat_dim:
                print(f"  ⚠️ 特征维度不匹配: {self.octree_features.shape[1]} vs {expected_feat_dim}")
                return False
            # 若启用杆系特征但缓存没有, 需要重新生成
            if getattr(self.config, 'USE_FRAME_FEATURE', False) and self.frame_features is None:
                print("  ⚠️ 缓存缺少 frame_features, 重新生成")
                return False
            
            return True
            
        except Exception as e:
            print(f"  ⚠️ 八叉树缓存加载失败: {e}")
            return False
    
    def _save_octree_cache(self):
        if not self.is_loaded or self.num_samples == 0:
            print("  ⚠️ 无八叉树数据可保存")
            return False
        
        try:
            data = {
                'octree_features': self.octree_features,
                'frame_features': self.frame_features,
                'heights': self.heights,
                'E_avg': self.E_avg,
                'params': self.params,
                'floor_node_masses': self.floor_node_masses,
                'motions': self.motions,
                'meta': self.meta,
                'num_samples': self.num_samples,
                'seq_len': self.seq_len,
                'octree_depth': self.config.OCTREE_DEPTH,
                'version': '1.4'
            }
            
            # 轻量模式: 不保存位移全时程 (训练时从仿真缓存按需读取)
            if not getattr(self.config, 'LEAN_OCTREE_CACHE', False):
                data['displacements'] = self.displacements
            else:
                data['displacements'] = None
            
            with open(self.config.OCTREE_CACHE_FILE, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size = os.path.getsize(self.config.OCTREE_CACHE_FILE) / (1024**2)
            print(f"  ✅ 八叉树缓存已保存: {self.config.OCTREE_CACHE_FILE} ({file_size:.2f} MB)")
            return True
            
        except Exception as e:
            print(f"  ⚠️ 八叉树缓存保存失败: {e}")
            return False
    
    # ============================================================
    # 体素化特征文件缓存 (数据库模式, train_voxel --use_db)
    # 体素化编码最耗时, 缓存到文件避免每次训练重复体素化。
    # 缓存按 struct 存 (键 = struct_id + 特征定义版本 + 深度):
    #   - 体素化特征只依赖结构 (struct), 与波/PGA/样本组合无关
    #   - 同一样本集多次抽样时, 相同结构直接复用缓存, 避免全量失效
    #   - 只有 特征定义版本 或 深度 变化才全量失效
    # ============================================================
    @staticmethod
    def _voxel_cache_key(struct_ids, depth, config=None):
        """构造体素化特征缓存键 (确定性字符串)"""
        cfg = config or Config()
        version = getattr(cfg, 'VOXEL_CACHE_VERSION', 1)
        return {
            'version': version,
            'depth': int(depth),
            'struct_ids': [int(s) for s in struct_ids],
        }

    @staticmethod
    def _voxel_cache_file(enc_mode):
        """按编码方式返回独立缓存文件路径 (避免 token/direct/cont 互相覆盖)"""
        from config import Config as _C
        base = getattr(_C, 'VOXEL_CACHE_FILE', './cache/voxel_features_cache.pkl')
        base = base.replace('.pkl', '')
        return f"{base}_{enc_mode}.pkl"

    def _load_voxel_cache(self, struct_ids, depth, enc_mode=None):
        """
        从文件加载体素化特征缓存 (按 struct 键校验)

        Args:
            struct_ids: 本次抽样对应的结构 id 列表 (顺序与待回填一致)
            depth: 八叉树紧缩深度
            enc_mode: 'token' / 'direct' / 'cont' (各编码方式独立缓存文件)

        Returns:
            list: 与 struct_ids 顺序一致的 [特征维度] 数组,
                  命中但缺某结构的位为 None; 整体不匹配/失败返回 None
        """
        cfg = self.config
        if enc_mode is None:
            enc_mode = ('token' if getattr(cfg, 'USE_VOXEL_TOKEN', False) else 'cont')
        cache_file = self._voxel_cache_file(enc_mode)
        if not cache_file or not os.path.exists(cache_file):
            return None
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            stored = data.get('cache')
            if stored is None:
                return None
            stored_version = data.get('version')
            stored_depth = data.get('depth')
            stored_enc = data.get('enc_mode')
            cur_version = getattr(cfg, 'VOXEL_CACHE_VERSION', 1)
            if (stored_version != cur_version or stored_depth != int(depth)
                    or stored_enc != enc_mode):
                print(f"  ⚠️ 体素化缓存版本/深度/编码方式变化 "
                      f"(v{stored_version}/d{stored_depth}/e{stored_enc} "
                      f"-> v{cur_version}/d{int(depth)}/e{enc_mode}), 重新体素化")
                return None
            if enc_mode == 'cont':
                from frame_grid_encoder import FEAT_C
                feat_dim = int(getattr(cfg, 'VOXEL_GRID', 64)) ** 3 * FEAT_C
            else:   # token / direct: 64³ = 262144
                feat_dim = int(getattr(cfg, 'VOXEL_GRID', 64)) ** 3
            result = []
            miss = 0
            for sid in struct_ids:
                f = stored.get(int(sid))
                if f is not None:
                    f = np.asarray(f, dtype=np.float32)
                    if f.shape[0] == feat_dim:
                        result.append(f)
                        continue
                result.append(None)
                miss += 1
            if miss:
                print(f"  💿 体素化缓存命中 {len(struct_ids)-miss}/{len(struct_ids)} "
                      f"(缺 {miss} 个结构, 仅补算缺失)")
            else:
                print(f"  💿 命中体素化特征缓存: {len(struct_ids)} 个样本 "
                      f"(深度 {depth}, 编码 {enc_mode}, 无需重新体素化)")
            return result
        except Exception as e:
            print(f"  ⚠️ 体素化缓存加载失败: {e}")
            return None

    def _save_voxel_cache(self, struct_ids, depth, feats, enc_mode=None):
        """把体素化特征写入文件缓存 (按 struct_id 增量合并, 保留旧结构)

        每种编码方式 (token/direct/cont) 用独立缓存文件, 避免互相覆盖。
        """
        cfg = self.config
        if enc_mode is None:
            enc_mode = ('token' if getattr(cfg, 'USE_VOXEL_TOKEN', False) else 'cont')
        cache_file = self._voxel_cache_file(enc_mode)
        if not cache_file:
            return
        try:
            cur_version = getattr(cfg, 'VOXEL_CACHE_VERSION', 1)
            cache = {}
            # 合并已存在的缓存 (同一版本+深度+编码方式, 增量追加新结构)
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        old = pickle.load(f)
                    if (old.get('version') == cur_version
                            and old.get('depth') == int(depth)
                            and old.get('enc_mode') == enc_mode
                            and isinstance(old.get('cache'), dict)):
                        cache = old['cache']
                except Exception:
                    cache = {}
            for sid, f in zip(struct_ids, feats):
                if f is not None:
                    cache[int(sid)] = np.asarray(f, dtype=np.float32)
            data = {
                'version': cur_version,
                'depth': int(depth),
                'enc_mode': enc_mode,
                'cache': cache,
            }
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            size = os.path.getsize(cache_file) / (1024 ** 2)
            print(f"  💾 体素化特征已缓存: {cache_file} "
                  f"({len(cache)} 个结构, {size:.2f} MB, enc={enc_mode})")
        except Exception as e:
            print(f"  ⚠️ 体素化缓存保存失败: {e}")

    def _generate_octree_cache(self):
        """从仿真缓存生成八叉树特征 (多进程并行); 成功返回 True"""
        print("\n  [..] 生成八叉树特征...")

        from simulation_cache import SimulationCache  # 延迟导入 (pkl 模式)
        sim_cache = SimulationCache(self.config)
        if not sim_cache.load():
            print("  ⚠️ 仿真缓存不存在 (将尝试从数据库读取)")
            return False

        sim_data = sim_cache.get_all()
        if sim_data is None:
            print("  ⚠️ 仿真数据为空 (将尝试从数据库读取)")
            return False

        n_samples = len(sim_data['params'])
        print(f"  📊 从仿真缓存读取 {n_samples} 个样本")

        # 每层节点质量 (旧缓存可能没有)
        floor_masses_all = sim_data.get('floor_node_masses', None)

        # 准备任务参数
        task_args = []
        for i in range(n_samples):
            fm = floor_masses_all[i] if floor_masses_all is not None else None
            task_args.append((
                sim_data['params'][i],
                sim_data['displacements'][i],
                self.config.OCTREE_DEPTH,
                fm
            ))

        octree_features_list = []
        displacements_list = []
        heights_list = []
        E_avg_list = []
        params_list = []
        failed = 0

        max_workers = min(mp.cpu_count(), getattr(self.config, 'MAX_OCTREE_WORKERS', 8))
        print(f"  🖥️ 使用 {max_workers} 个进程并行生成八叉树特征")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_compute_octree_single, args): i
                       for i, args in enumerate(task_args)}
            with tqdm(total=n_samples, desc="  生成八叉树特征") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        octree_features_list.append(result['octree_feat'])
                        displacements_list.append(result['displacement'])
                        heights_list.append(result['height'])
                        E_avg_list.append(result['E_avg'])
                        params_list.append(result['params'])
                    except Exception:
                        failed += 1
                        # 失败样本使用空特征
                        octree_features_list.append(
                            np.zeros(4 * (2 ** self.config.OCTREE_DEPTH), dtype=np.float32))
                        displacements_list.append(sim_data['displacements'][idx])
                        heights_list.append(sim_data['heights'][idx])
                        E_avg_list.append(sim_data['E_avg'][idx])
                        params_list.append(sim_data['params'][idx])
                    pbar.update(1)

        if failed > 0:
            print(f"  ⚠️ {failed} 个样本八叉树生成失败 (已置空)")

        # ---- 统一 params 维度: 补成 PARAMS_DIM (8 + 形状 + 每层荷载), 兼容旧缓存 ----
        from config import Config as _C
        _pdim = int(getattr(_C, 'PARAMS_DIM', 21))
        _poff = int(getattr(_C, 'PARAMS_FLOOR_LOAD_OFFSET', 9))
        _pmax = int(getattr(_C, 'PARAMS_MAX_FLOORS', 12))
        loads_all = sim_data.get('floor_loads', None)
        new_params = []
        for i, p in enumerate(params_list):
            p = np.asarray(p, dtype=np.float32)
            if p.ndim == 0:
                p = p.reshape(1)
            if p.shape[0] < _pdim:
                p2 = np.zeros(_pdim, dtype=np.float32)
                k = min(len(p), 8)
                p2[:k] = p[:k]
                # 每层荷载 (若缓存有 floor_loads)
                if loads_all is not None and i < len(loads_all):
                    ld = np.asarray(loads_all[i], dtype=np.float32)[:_pmax]
                    p2[_poff:_poff + len(ld)] = ld
                p = p2
            new_params.append(p)
        params_list = new_params

        self.octree_features = np.array(octree_features_list, dtype=np.float32)
        self.displacements = np.array(displacements_list, dtype=np.float32)
        self.heights = np.array(heights_list, dtype=np.float32)
        self.E_avg = np.array(E_avg_list, dtype=np.float32)
        self.params = np.array(params_list, dtype=np.float32)
        self.floor_node_masses = floor_masses_all
        # 实际输入地震动加速度 (若仿真缓存有 motions 字段)
        motions_all = sim_data.get('motions', None)
        self.motions = np.array(motions_all, dtype=np.float32) if motions_all is not None else None
        self.num_samples = len(self.octree_features)
        self.is_loaded = True

        # 杆系结构化特征 (直接从 params + floor_node_masses 提取, 更快更准)
        if getattr(self.config, 'USE_FRAME_FEATURE', False):
            from frame_feature_encoder import encode_frame_batch
            frame_feats, _ = encode_frame_batch(
                self.params, floor_masses_list=floor_masses_all,
                max_stories=12, E=3.25e10)
            self.frame_features = frame_feats
            print(f"  ✅ 杆系结构化特征: {self.frame_features.shape}")
        else:
            self.frame_features = None

        self._save_octree_cache()

        print(f"     特征维度: {self.octree_features.shape[1]}")
        return True
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if idx >= self.num_samples:
            raise IndexError(f"索引 {idx} 超出范围 {self.num_samples}")
        
        sample = {
            'octree_features': torch.FloatTensor(self.octree_features[idx]),
            'frame_features': (torch.FloatTensor(self.frame_features[idx])
                               if self.frame_features is not None else None),
            'disp': torch.FloatTensor(self.displacements[idx]),
            'height': torch.FloatTensor([self.heights[idx]]),
            'E_avg': torch.FloatTensor([self.E_avg[idx]]),
            'params': torch.FloatTensor(self.params[idx]),   # [8] 结构参数 (结构条件注入)
        }
        # 真实地震动输入 (因果解码器输入信号)
        if self.motions is not None:
            sample['motion'] = torch.FloatTensor(self.motions[idx])
        return sample
    
    def get_stats(self):
        if not self.is_loaded:
            return None
        return {
            'num_samples': self.num_samples,
            'seq_len': self.seq_len,
            'octree_feature_dim': self.octree_features.shape[1],
            'disp_mean': self.displacements.mean(),
            'disp_std': self.displacements.std(),
            'disp_min': self.displacements.min(),
            'disp_max': self.displacements.max(),
            'height_mean': self.heights.mean(),
            'height_range': [self.heights.min(), self.heights.max()]
        }


# ============================================================
# 测试
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("八叉树数据集测试")
    print("="*60)
    
    cfg = Config()
    dataset = OctreeDataset(cfg, force_regen_octree=True)
    
    print(f"\n数据集大小: {len(dataset)}")
    sample = dataset[0]
    print(f"  octree_features: {sample['octree_features'].shape}")
    print(f"  disp: {sample['disp'].shape}")
    
    stats = dataset.get_stats()
    if stats:
        print(f"\n统计信息:")
        print(f"  位移均值: {stats['disp_mean']:.2f} mm")
        print(f"  位移范围: [{stats['disp_min']:.2f}, {stats['disp_max']:.2f}] mm")