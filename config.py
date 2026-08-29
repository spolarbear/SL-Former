# config.py
"""
全局配置类 - 所有参数集中管理
"""

import sys
import os

# Windows 控制台默认 GBK 编码，无法打印 emoji 等非 BMP 字符，
# 会导致 print 抛出 UnicodeEncodeError。这里统一重配置为标准输出编码。
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import torch


class Config:
    """
    全局配置类
    所有参数集中管理，修改此处即可调整整个系统
    """
    
    # ============================================================
    # 0. 空间参数 (用于体素生成)
    # ============================================================
    SPACE_X = 60.0                    # X方向空间范围 (m)
    SPACE_Y = 60.0                    # Y方向空间范围 (m)
    SPACE_Z = 100.0                   # Z方向空间范围 (m)
    VOXEL_SIZE = 0.2                  # 体素尺寸 (m)
    
    # ============================================================
    # 1. 体素标记值 (用于区分构件类型)
    # ============================================================
    MARKER_AIR = 0.001                # 空气
    MARKER_COLUMN = 35.0              # 柱 (蓝色系)
    MARKER_BEAM = 32.0                # 梁 (绿色系)
    MARKER_SLAB = 28.0                # 板 (红色系)
    
    # ============================================================
    # 2. 材料参数 (用于OpenSees仿真)
    # ============================================================
    E_CONCRETE = 3.25e10              # C40混凝土弹性模量 (Pa)
    E_STEEL = 2.1e11                  # 钢筋弹性模量 (Pa)
    CONCRETE_FC = 14.3                # C30混凝土抗压强度 (MPa)
    COL_SECTIONS = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.38, 0.35, 0.3]  # 12层逐层递减
    BEAM_SECTION = (0.3, 0.6)         # 梁截面 (宽, 高) (m) - 备用
    SLAB_THICKNESS = 0.2              # 楼板厚度 (m)
    
    # ============================================================
    # 3. 仿真参数
    # ============================================================
    NUM_SIMULATIONS = 400             # 仿真样本数
    NUM_WAVES = 200                   # 地震动波数 (dzb 真实波数量)
    WINDOW_BEFORE = 5                 # 峰值前时间 (秒)
    WINDOW_AFTER = 5                  # 峰值后时间 (秒)
    TARGET_DT = 0.02                  # 时间步长 (秒)
    TARGET_PGA = 0.035                # 地震动池基准峰值加速度 (g) (加载真实波时统一归一到该值)
    PGA_RANGE = (0.1, 0.2)            # 每样本目标峰值加速度范围 (g)：仿真时在此范围内随机缩放
    PGA_OPTIONS = [0.10]  # 离散 PGA 取值 (固定步长, 保证组合可重复/查重生效)
    EARTHQUAKE_FOLDER = './dzb'       # 真实地震动文件夹 (dzb 内记录)
    FORCE_MOTION_REGEN = False        # 强制重新生成/读取地震动池
    
    # ============================================================
    # 4. 结构参数范围
    # ============================================================
    NUM_STORIES_RANGE = [2, 6]       # 层数范围
    STORY_HEIGHTS = [3.0, 3.3, 3.6, 4.0, 4.5]  # 层高选项
    SPAN_WIDTHS = [4.0, 5.0, 6.0, 7.0, 8.0]    # 跨度选项
    
    # 面荷载和轴压比参数
    FLOOR_LOAD = 20.0                 # 楼面荷载默认 (kPa) - 兼容旧参数
    FLOOR_LOAD_OPTIONS = [15, 15, 15, 15]  # 每层随机面荷载选项 (kPa)，增强样本多样性
    AXIS_RATIO_MIN = 0.35             # 最小轴压比
    AXIS_RATIO_MAX = 0.80             # 最大轴压比
    
    # ============================================================
    # 5. 八叉树参数
    # ============================================================
    OCTREE_DEPTH = 5                  # 八叉树深度 (5 = 32x32x32)
    OCTREE_FEATURE_DIM = 128          # 八叉树原始特征维度 (4 * 2^depth: 密度+质心X+质心Y+质量)
    # 杆系结构化物理特征 (frame_feature_encoder): 直接从杆系模型提取构件信息
    # (层数/跨数/跨度/层高/每层柱截面/每层抗侧刚度/每层质量/基频/总质量),
    # 保留原始构件信息, 比体素化更精确。默认使用此特征替代体素八叉树。
    FRAME_FEATURE_DIM = 44
    USE_FRAME_FEATURE = True          # True: 用杆系结构化特征替代体素八叉树 (更快更准)
    
    # ============================================================
    # 结构参数向量 p 维度 (显式条件注入 cond_params)
    #   p[0:8]  : 原 8 维 (层数/跨数X/Y/跨度X/Y/层高/平均质量/阻尼)
    #   p[8]    : 平面形状 ID (0=rect, 1=T, 2=L, 3=C, 4=U)
    #   p[9:21] : 每层楼面平均荷载 (kPa), 每层一个值, 最多 12 层, 不足补 0
    # ============================================================
    PARAMS_DIM = 21                   # 8 + 1(形状) + 12(每层荷载)
    PARAMS_SHAPE_IDX = 8              # 形状 ID 索引
    PARAMS_FLOOR_LOAD_OFFSET = 9      # 每层荷载起始索引
    PARAMS_MAX_FLOORS = 12            # 每层荷载最大层数 (与仿真器一致)
    
    # ============================================================
    # 6. 模型参数 (增强版)
    # ============================================================
    CNN_FEATURE_DIM = 128
    D_MODEL = 256
    N_HEAD = 8
    N_LAYER = 4
    D_FF = 512
    DROPOUT = 0.2
    # v2 现代架构参数
    USE_V2 = True             # 默认使用 v2 现代架构 (train 训练主路径)
    V2_DROP_PATH = 0.2        # v2 随机深度概率 (增大正则, 缓解过拟合)
    V2_FILM = True            # v2 用 FiLM 条件注入
    V2_CONV_KERNEL = 31       # v2 局部卷积核大小
    # 切杆系 128bit 体素编码网格 (frame_grid_encoder): 每方向格数
    # 1m/格, 64 格覆盖 64m 空间 (用户要求 2026-08-19)
    VOXEL_GRID = 64
    # 体素 token embedding 初始化方式 (三种策略, 可消融对比):
    #   True / 'rich8' : rich 8 维物理向量初始化 (类型/柱EI/梁EI/面积/偏位/填充)
    #   'hexa9'        : 六面体刚度 9 维初始化 (3 对对面剪切GA+抗弯EI + 类型/填充/偏位)
    #   'basic5'       : 精简 5 维物理向量初始化
    #   False/'random' : 纯随机初始化 (无物理先验)
    # 物理初始化使"刚度/截面相似"的微元 token 向量初始即邻近 (训练中继续微调)
    VOXEL_TOKEN_INIT_PHYSICS = 'rich8'
    
    # ============================================================
    # 7. 训练参数
    # ============================================================
    BATCH_SIZE = 32
    EPOCHS = 150
    # 初始学习率: 实测 1e-3 过高, 前期 ~30 epoch 几乎不动, 降到 5e-4 才开始学习 → 改为 5e-4
    LEARNING_RATE = 0.0009
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    EARLY_STOP_PATIENCE = 20
    # 学习率调度方式:
    #   'cosine' (推荐): 线性 warmup 升至 base_lr, 再余弦退火至 LR_MIN。
    #       避免初始过大"学不动", 后期平滑收敛 (transformer 常用)
    #   'plateau'      : ReduceLROnPlateau, 验证指标平台期自动降 LR (旧行为)
    LR_SCHEDULER = 'cosine'
    LR_WARMUP_EPOCHS = 5            # cosine: 线性 warmup 轮数
    LR_WARMUP_START = 0.1           # cosine: warmup 起始 LR 比例 (base_lr * 0.1)
    LR_MIN = 1e-6                   # 最小学习率 (cosine 下限 / plateau 的 min_lr)
    LR_FACTOR = 0.5                 # plateau: 每次降低比例
    LR_PATIENCE = 10                # plateau: 连续多少个 epoch 无改善则降 LR
    # DataLoader 数据加载 worker 数 (Windows 下建议 0~8):
    #   数据已全部在内存 (numpy 数组), __getitem__ 只是切片+转 tensor, 很轻量,
    #   所以 num_workers=0 也能跑; 设 4~8 可让 GPU 传输与 CPU 组装重叠、略快一些。
    #   若多进程在 Windows 上报错, 改回 0 最稳。
    NUM_WORKERS = 4
    # 混合精度训练 (FP16): 目标已归一化(波形[-1,1] + log峰值)且损失为 L1(MAE),
    #   数值量级小、无平方溢出风险, 可安全开启 (仅 CUDA 可用时生效)。
    USE_MIXED_PRECISION = True
    # 位移峰值/低估惩罚 (解决模型整体低估、峰值被截断问题):
    #   原损失用 log(峰值)+0.3 权重, log 压缩梯度 → 模型无动力放大峰值 → 输出偏 0。
    # 1) LOSS_PEAK_W: 峰值项权重 (峰值相对误差 / 峰值绝对误差)
    # 2) LOSS_HIGH_W: 大位移区域(>阈值)误差加权系数, 直接对抗"大位移杆件被截断"
    # 3) LOSS_HIGH_THRESH_MM: 大位移判定阈值 (mm)
    LOSS_PEAK_W = 1.0
    LOSS_HIGH_W = 3.0
    LOSS_HIGH_THRESH_MM = 8.0
    # 损失归一化方式:
    #   'absolute' (默认): 用绝对 mm 误差训练 (大位移样本主导损失)
    #   'relative'       : 峰值归一化相对误差训练 (波形误差/峰值 + 峰值相对误差
    #                       + 大位移相对误差; 大位移不主导, 改善小位移相对精度)
    LOSS_NORM = 'relative'
    # 相对模式: 大位移判定阈值 = 该比例 × 样本峰值 (如 0.2 = 峰值的20%)
    # (absolute 模式用 LOSS_HIGH_THRESH_MM 的绝对 mm 阈值)
    LOSS_HIGH_THRESH_RATIO = 0.2
    
    # ============================================================
    # 8. 物理正则化
    # ============================================================
    PHYSICS_LAMBDA = 0.01
    PHYSICS_DELTA = 0.1
    
    # ============================================================
    # 9. 缓存设置
    # ============================================================
    CACHE_DIR = './cache'
    SIM_CACHE_FILE = os.path.join(CACHE_DIR, 'simulation_cache.pkl')
    OCTREE_CACHE_FILE = os.path.join(CACHE_DIR, 'octree_cache.pkl')
    USE_CACHE = True
    MAX_CACHE_SIZE = 5000
    # 体素化特征缓存 (数据库模式, train_voxel.py --use_db)
    # 体素化编码最耗时, 缓存到文件避免每次训练重复体素化
    VOXEL_CACHE_FILE = os.path.join(CACHE_DIR, 'voxel_features_cache.pkl')
    VOXEL_CACHE_VERSION = 14  # 特征定义/编码逻辑变化时 +1 (v14: 128bit 编码 + 1m/64³ 网格)
    # 微元词表 (LLM tokenizer 思想): 数据库扫描构建的固定词表
    VOXEL_VOCAB_FILE = os.path.join(CACHE_DIR, 'voxel_vocab.pkl')
    USE_VOXEL_TOKEN = True            # True: 体素用 token + nn.Embedding (替代连续特征)
    VOXEL_VOCAB_SIZE = 300            # 词表大小 (含空; 实际按扫描结果, 自动扩展)
    VOXEL_VOCAB_SCAN_STRUCTS = 2000   # 词表缺失时从数据库扫描多少结构来构建
    VOXEL_TOKEN_EMBED_DIM = 32        # 每格 token 的 embedding 维度
    # 数据库模式训练: 每次最多随机抽取多少样本 (数据库可能有 8000+ 条, 全取太慢;
    # 设为 0 或 None 表示全部使用)
    DB_MAX_SAMPLES = 10000
    DB_SAMPLE_SEED = 42             # 随机抽样种子 (保证可复现)
    # 均匀抽样前的样本筛选: 顶点最大位移大于 总建筑高度×阈值 的样本先剔除
    # (位移过大说明结构处于严重非线性/近倒塌, 训练时可能污染回归目标)
    # 判定: disp_peak(mm)/1000 > total_height(m) × DB_MAX_DRIFT_RATIO
    DB_FILTER_MAX_DRIFT = True      # 开关
    DB_MAX_DRIFT_RATIO = 0.005      # 阈值 = 总建筑高度的 0.5%
    # 均匀抽样前的 PGA 筛选: 只保留指定目标 PGA 的样本 (None=不筛选)
    # 例: 0.10 表示只要 target_pga=0.1g 的样本 (数据库 PGA_OPTIONS 离散值)
    DB_FILTER_PGA = 0.10            # 只保留该 PGA (g) 的样本; None=不过滤
    # 按楼层数分层抽样: True 时抽取样本会按 num_stories 均匀分散, 避免某楼层过多/过少
    DB_STRATIFY_STORIES = True
    # 响应量级均匀化: 在按楼层均匀的基础上, 再按 disp_peak(峰值位移) 分位数分档,
    # 保证大变形/小变形样本均匀 (True=按三分位低/中/高, 0/False=不启用)
    DB_STRATIFY_RESPONSE_BINS = 3
    # 结构形态均匀化: 在楼层+响应基础上, 再按跨度/层高/跨数合并分档均匀
    # (True=按跨度大小×层高×跨数 2×2×2 分档, False=不启用)
    DB_STRATIFY_STRUCT = True
    # 轻量缓存: 只保存八叉树特征和必要标量，不重复保存位移时程等大数据
    # (训练只需 octree_features/displacements/heights/E_avg/params，全部保留，
    #  但去掉 meta/unique_keys 等冗余时程，需用时从仿真缓存按需读取)
    LEAN_OCTREE_CACHE = False   # 若为 True，八叉树缓存不再保存 displacements 全时程
    # 多进程并行配置
    MAX_SIM_WORKERS = 16         # 仿真并行进程数上限
    MAX_OCTREE_WORKERS = 16       # 八叉树特征并行进程数上限 (单个体素约180MB，控制内存)
    # 增量生成时的批大小 (避免一次性申请过多内存)
    GENERATION_CHUNK = 256
    
    # ============================================================
    # 10. 设备和路径
    # ============================================================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PLOT_DIR = './plots'
    MODEL_DIR = './models'
    REPORT_DIR = './reports'
    
    @classmethod
    def get_seq_len(cls):
        """获取时程序列长度"""
        return int((cls.WINDOW_BEFORE + cls.WINDOW_AFTER) / cls.TARGET_DT)
    
    @classmethod
    def get_param_dim(cls):
        """获取结构参数维度"""
        return 8  # 层数, 跨数X, 跨数Y, 跨宽X, 跨宽Y, 层高, 质量, 阻尼
    
    @classmethod
    def get_grid_dims(cls):
        """获取体素网格维度"""
        grid_x = int(cls.SPACE_X / cls.VOXEL_SIZE)
        grid_y = int(cls.SPACE_Y / cls.VOXEL_SIZE)
        grid_z = int(cls.SPACE_Z / cls.VOXEL_SIZE)
        return grid_x, grid_y, grid_z
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        grid_x, grid_y, grid_z = cls.get_grid_dims()
        print("="*70)
        print("配置信息")
        print("="*70)
        print(f"  【空间】")
        print(f"    空间范围: {cls.SPACE_X}x{cls.SPACE_Y}x{cls.SPACE_Z} m")
        print(f"    体素网格: {grid_x}x{grid_y}x{grid_z}")
        print(f"  【材料标记】")
        print(f"    柱: {cls.MARKER_COLUMN}, 梁: {cls.MARKER_BEAM}, 板: {cls.MARKER_SLAB}")
        print(f"  【仿真】")
        print(f"    仿真样本数: {cls.NUM_SIMULATIONS}")
        print(f"    地震动波数: {cls.NUM_WAVES}")
        print(f"    序列长度: {cls.get_seq_len()} 步 ({cls.WINDOW_BEFORE+cls.WINDOW_AFTER}s)")
        print(f"  【结构参数】")
        print(f"    层数范围: {cls.NUM_STORIES_RANGE}")
        print(f"    面荷载: {cls.FLOOR_LOAD} kPa")
        print(f"    轴压比范围: {cls.AXIS_RATIO_MIN} ~ {cls.AXIS_RATIO_MAX}")
        print(f"  【八叉树】")
        print(f"    深度: {cls.OCTREE_DEPTH} (分辨率: {2**cls.OCTREE_DEPTH}³)")
        print(f"  【模型】")
        print(f"    D_MODEL: {cls.D_MODEL}, N_LAYER: {cls.N_LAYER}, N_HEAD: {cls.N_HEAD}")
        print(f"  【训练】")
        print(f"    Epochs: {cls.EPOCHS}, Batch: {cls.BATCH_SIZE}")
        print(f"    初始LR: {cls.LEARNING_RATE}, 调度: {cls.LR_SCHEDULER}")
        print(f"  【设备】")
        print(f"    {cls.DEVICE}")
        print("="*70)
    
    @classmethod
    def ensure_dirs(cls):
        """确保所有目录存在"""
        for d in [cls.CACHE_DIR, cls.PLOT_DIR, cls.MODEL_DIR, cls.REPORT_DIR]:
            os.makedirs(d, exist_ok=True)