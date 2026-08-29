# transformer_model.py
"""
Transformer模型模块 - 增强版
功能：
1. 更深更宽的Transformer结构
2. 支持八叉树特征输入
3. 双向交叉注意力
4. 完整的残差连接和层归一化
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from octree_encoder import PrecomputedOctreeEncoder


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class CrossAttentionBlock(nn.Module):
    """
    双向交叉注意力块
    路径A: 时序查询结构 (Q=时序, K/V=结构)
    路径B: 结构查询时序 (Q=结构, K/V=时序)
    """
    def __init__(self, d_model, n_head, d_ff, dropout=0.1, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        
        # 路径A: 时序 → 结构
        self.cross_attn_A = nn.MultiheadAttention(
            d_model, n_head, batch_first=True, dropout=dropout
        )
        # 路径B: 结构 → 时序
        self.cross_attn_B = nn.MultiheadAttention(
            d_model, n_head, batch_first=True, dropout=dropout
        )
        
        # 前馈网络 (使用GELU激活)
        self.ffn_A = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_B = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1_A = nn.LayerNorm(d_model)
        self.norm1_B = nn.LayerNorm(d_model)
        self.norm2_A = nn.LayerNorm(d_model)
        self.norm2_B = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, temporal_feat, struct_feat):
        """
        temporal_feat: [batch, T, d_model]
        struct_feat: [batch, L, d_model]
        """
        # 路径A: 时序查询结构
        attn_out_A, attn_weights_A = self.cross_attn_A(
            query=temporal_feat,
            key=struct_feat,
            value=struct_feat
        )
        temporal_feat = self.norm1_A(temporal_feat + self.dropout(attn_out_A))
        temporal_feat = self.norm2_A(temporal_feat + self.ffn_A(temporal_feat))
        
        if self.bidirectional:
            # 路径B: 结构查询时序
            attn_out_B, attn_weights_B = self.cross_attn_B(
                query=struct_feat,
                key=temporal_feat,
                value=temporal_feat
            )
            struct_feat = self.norm1_B(struct_feat + self.dropout(attn_out_B))
            struct_feat = self.norm2_B(struct_feat + self.ffn_B(struct_feat))
            return temporal_feat, struct_feat, (attn_weights_A, attn_weights_B)
        else:
            # 仅单向: 时序查询结构 (消融用)
            return temporal_feat, struct_feat, (attn_weights_A, None)


# ================================================================
# 现代组件 (v2 架构) - 融合前沿 Transformer 思路
#   - RMSNorm: 比 LayerNorm 更稳更省 (LLaMA 系列)
#   - DropPath: 随机深度正则, 小样本防过拟合
#   - RotaryEmbedding (RoPE): 相对位置编码, 外推与泛化更好 (LLaMA/PaLM)
#   - QK-Norm: query/key 归一化稳定注意力 (Meta LLaMA 2/3)
#   - SwiGLU FFN: 门控前馈 (LLaMA/PaLM)
#   - ConvBlock: Conformer 风格局部波形建模 (1D 深度卷积+GLU)
#   - FiLM 条件注入: 结构特征以 scale+shift 方式调制时序 (条件 Transformer)
# ================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class DropPath(nn.Module):
    """随机深度: 训练时以概率 p 丢弃残差分支"""
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = torch.empty(x.shape[0], 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep


class RotaryEmbedding(nn.Module):
    """RoPE 旋转位置编码 (应用于 Q/K)"""
    def __init__(self, dim, max_len=1000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)          # [T, dim/2]
        # 每个频率复制 2 次 -> [T, dim] (与 head_dim 逐元素对齐)
        self.register_buffer('cos_cached', freqs.cos().repeat_interleave(2, dim=-1))
        self.register_buffer('sin_cached', freqs.sin().repeat_interleave(2, dim=-1))

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        """x: [B, T, n_head, head_dim] -> 施加 RoPE"""
        T = x.size(1)
        cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(2)   # [1, T, 1, dim]
        sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(2)
        return x * cos + self._rotate_half(x) * sin


class SwiGLU(nn.Module):
    """SwiGLU 门控前馈"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class QKNormAttention(nn.Module):
    """多头注意力 + QK-Norm + 可选 RoPE (自注意力/交叉注意力通用)"""
    def __init__(self, d_model, n_head, dropout=0.1, qk_norm=True,
                 rotary=None, causal=False):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm
        self.rotary = rotary
        self.causal = causal

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self._causal_mask = None

    def _split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,H,T,D]

    def _get_causal_mask(self, T, device):
        if self._causal_mask is None or self._causal_mask.shape[0] < T:
            self._causal_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool), diagonal=1).to(device)
        return self._causal_mask[:T, :T]

    def forward(self, x, kv=None):
        """x: [B,T,d]; kv: [B,L,d] (None=自注意力)"""
        kv = x if kv is None else kv
        B, T, _ = x.shape
        q = self._split(self.wq(x))                    # [B,H,T,D]
        k = self._split(self.wk(kv))                   # [B,H,L,D]
        v = self._split(self.wv(kv))
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.rotary is not None and kv is x:
            q = self.rotary(q)
            k = self.rotary(k)
        attn = (q * self.scale) @ k.transpose(-2, -1)  # [B,H,T,L]
        if self.causal and kv is x:
            mask = self._get_causal_mask(T, attn.device)
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v                                  # [B,H,T,D]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.dropout(self.wo(out)), attn


class ConvBlock(nn.Module):
    """Conformer 风格 1D 卷积: 局部波形建模 (GLU + 深度卷积)"""
    def __init__(self, d_model, kernel_size=31, dropout=0.1, causal=False):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.pw1 = nn.Linear(d_model, d_model)         # 先降维/投影
        self.pw2 = nn.Linear(d_model, d_model)         # GLU 分支
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size,
                                   padding=0, groups=d_model)
        self.pw3 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        if causal:
            self.pad = (kernel_size - 1, 0)            # 只向左看
        else:
            p = (kernel_size - 1) // 2
            self.pad = (p, kernel_size - 1 - p)

    def forward(self, x):
        """x: [B,T,d]"""
        h = self.norm(x)
        a = self.pw1(h)
        b = self.pw2(h)
        h = a * torch.sigmoid(b)                       # GLU 门控
        h = h.transpose(1, 2)                          # [B,d,T]
        h = F.pad(h, self.pad)
        h = self.depthwise(h)                          # [B,d,T]
        h = h.transpose(1, 2)
        h = self.act(h)
        h = self.pw3(h)
        return self.dropout(h)


class V2Block(nn.Module):
    """现代 Transformer 块 (Pre-LN):
        自注意力(RoPE+QK-Norm) -> 结构交叉注意力 -> FiLM条件注入 -> 卷积 -> SwiGLU FFN
    """
    def __init__(self, d_model, n_head, d_ff, dropout=0.1, drop_path=0.0,
                 rotary=None, use_cross_attn=True, use_struct_fusion=True,
                 conv_kernel=31, use_sa=True, use_conv=True, use_ffn=True):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_struct_fusion = use_struct_fusion
        self.use_sa = use_sa
        self.use_conv = use_conv
        self.use_ffn = use_ffn
        self.drop_path = DropPath(drop_path)

        # 1. 自注意力 (局部/全局波形依赖)
        if use_sa:
            self.norm_sa = RMSNorm(d_model)
            self.sa = QKNormAttention(d_model, n_head, dropout, qk_norm=True, rotary=rotary)

        # 2. 结构交叉注意力 (时序 <- 结构)
        if use_cross_attn:
            self.norm_ca = RMSNorm(d_model)
            self.ca = QKNormAttention(d_model, n_head, dropout, qk_norm=True, rotary=None)

        # 3. FiLM 条件注入 (结构 scale+shift 调制)
        if use_struct_fusion:
            self.norm_film = RMSNorm(d_model)
            self.film = nn.Linear(d_model, 2 * d_model)

        # 4. 局部卷积 (波形整形)
        if use_conv:
            self.norm_conv = RMSNorm(d_model)
            self.conv = ConvBlock(d_model, kernel_size=conv_kernel, dropout=dropout)

        # 5. SwiGLU FFN
        if use_ffn:
            self.norm_ff = RMSNorm(d_model)
            self.ffn = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x, struct_tokens, struct_cond):
        """x: [B,T,d]; struct_tokens: [B,L,d]; struct_cond: [B,d]"""
        # 1. 自注意力 (Pre-LN)
        if self.use_sa:
            h = self.norm_sa(x)
            x = x + self.drop_path(self.sa(h)[0])
        # 2. 结构交叉注意力
        if self.use_cross_attn:
            h = self.norm_ca(x)
            x = x + self.drop_path(self.ca(h, kv=struct_tokens)[0])
        # 3. FiLM 结构条件注入
        if self.use_struct_fusion:
            gamma, beta = self.film(struct_cond).chunk(2, dim=-1)  # [B,d]
            h = self.norm_film(x)
            x = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        # 4. 局部卷积
        if self.use_conv:
            h = self.norm_conv(x)
            x = x + self.drop_path(self.conv(h))
        # 5. SwiGLU FFN
        if self.use_ffn:
            h = self.norm_ff(x)
            x = x + self.drop_path(self.ffn(h))
        return x


class SLFormer(nn.Module):
    """
    SL-Former 增强版 (双架构)

    - 旧架构 (use_v2=False, 默认): 双向交叉注意力 + 位置编码, 与已训练基准/消融模型完全兼容
    - v2 架构 (use_v2=True): 融合现代 Transformer 思路
        * Pre-LN + DropPath + RMSNorm (训练稳定, 小样本友好)
        * RoPE 旋转位置编码 + QK-Norm 自注意力
        * 结构交叉注意力 + FiLM 结构条件注入 (scale+shift)
        * Conformer 风格 1D 卷积局部波形建模
        * SwiGLU FFN
        * motion -> 位移 卷积残差旁路 (强梯度路径, 改善收敛)
    """

    def __init__(self, config, use_cross_attn=True, use_struct_fusion=True,
                 use_pos_enc=True, bidirectional=True, use_bypass=False,
                 use_v2=False, drop_path=0.0, film=True,
                 use_sa=True, use_conv=True, use_ffn=True, use_cond_params=True,
                 use_voxel_token=False, vocab_size=None):
        """
        SLFormer 主模型

        Args:
            use_cross_attn: 是否使用交叉注意力
            use_struct_fusion: 是否将结构特征注入每个时间步
            use_pos_enc: 是否使用位置编码
            bidirectional: 双向交叉注意力 (仅旧架构)
            use_bypass: 是否使用 motion->位移 残差旁路 + 输入缩放
            use_v2: 使用 v2 现代架构 (默认 False, 保持旧模型/消融兼容)
            drop_path: v2 随机深度概率 (0~1)
            film: v2 是否用 FiLM 条件注入 (False 则退化为 bias 注入)
            use_sa: v2 是否用自注意力 (消融用, 默认 True)
            use_conv: v2 是否用局部卷积 (消融用, 默认 True)
            use_ffn: v2 是否用 SwiGLU FFN (消融用, 默认 True)
            use_cond_params: v2 是否用显式结构参数条件注入 (消融用, 默认 True)
            use_voxel_token: True 时用 VoxelTokenEncoder (每格 token + Embedding),
                             替代 PrecomputedOctreeEncoder (连续特征 MLP)
            vocab_size: 微元词表大小 (use_voxel_token=True 时必填)
        """
        super().__init__()
        self.config = config
        self.use_cross_attn = use_cross_attn
        self.use_struct_fusion = use_struct_fusion
        self.use_pos_enc = use_pos_enc
        self.bidirectional = bidirectional
        self.use_bypass = use_bypass
        self.use_v2 = use_v2
        self.film = film
        self.use_sa = use_sa
        self.use_conv = use_conv
        self.use_ffn = use_ffn
        self.use_cond_params = use_cond_params

        # ============================================================
        # 1. 结构编码器 (八叉树 → 结构特征; 或杆系结构化物理特征)
        # ============================================================
        if use_voxel_token:
            # 体素 token 编码: 每格离散 token + nn.Embedding (LLM 思想)
            from octree_encoder import VoxelTokenEncoder
            if vocab_size is None:
                vocab_size = int(getattr(config, 'VOXEL_VOCAB_SIZE', 300))
            _grid = int(getattr(config, 'VOXEL_GRID', 64))   # 64 (1m/格, 64m空间)
            # 物理向量初始化 embedding: 从词表文件加载 id2micro (刚度/截面相似
            # 的 token 初始即邻近), 由 VOXEL_TOKEN_INIT_PHYSICS 控制
            #   True / 'rich8' -> rich 8 维; 'hexa9' -> 六面体刚度 9 维;
            #   'basic5' -> 精简 5 维; False / 'random' -> 随机初始化
            _pmode = getattr(config, 'VOXEL_TOKEN_INIT_PHYSICS', True)
            if _pmode is True:
                _pmode = 'rich8'
            elif _pmode is False or _pmode is None:
                _pmode = 'random'
            _vocab = None
            if _pmode != 'random':
                try:
                    from frame_grid_encoder import VoxelVocab
                    _vf = getattr(config, 'VOXEL_VOCAB_FILE', None)
                    if _vf and os.path.exists(_vf):
                        _vocab = VoxelVocab()
                        _vocab.load(_vf)
                    elif os.path.exists('./cache/voxel_vocab.pkl'):
                        _vocab = VoxelVocab()
                        _vocab.load('./cache/voxel_vocab.pkl')
                except Exception:
                    _vocab = None
            self.struct_encoder = VoxelTokenEncoder(
                vocab_size=vocab_size,
                output_dim=config.CNN_FEATURE_DIM,
                embed_dim=int(getattr(config, 'VOXEL_TOKEN_EMBED_DIM', 32)),
                grid=_grid,
                physics_mode=_pmode,
                vocab=_vocab,
            )
            self.struct_input_dim = self.struct_encoder.n_tokens
        elif getattr(config, 'USE_FRAME_FEATURE', False):
            # 杆系结构化特征 (44维, 含层数/跨数/跨度/层高/截面/刚度/质量/基频)
            struct_input_dim = getattr(config, 'FRAME_FEATURE_DIM', 44)
            self.struct_encoder = PrecomputedOctreeEncoder(
                output_dim=config.CNN_FEATURE_DIM,
                max_depth=config.OCTREE_DEPTH,
                input_dim=struct_input_dim,
            )
            self.struct_input_dim = struct_input_dim
        else:
            struct_input_dim = config.OCTREE_FEATURE_DIM
            self.struct_encoder = PrecomputedOctreeEncoder(
                output_dim=config.CNN_FEATURE_DIM,
                max_depth=config.OCTREE_DEPTH,
                input_dim=struct_input_dim,
            )
            self.struct_input_dim = struct_input_dim
        
        # ============================================================
        # 2-8. 时序编码 / 注意力 / 融合 / 输出 / 旁路
        # ============================================================
        if use_v2:
            self._build_v2(config, use_cross_attn, use_struct_fusion, film,
                           drop_path, use_bypass, use_sa, use_conv, use_ffn,
                           use_cond_params)
        else:
            self._build_v1(config, use_cross_attn, use_struct_fusion,
                           use_pos_enc, bidirectional, use_bypass)

        # 初始化权重
        self._init_weights()

    # ------------------------------------------------------------
    # v1 旧架构 (兼容已训练模型/消融)
    # ------------------------------------------------------------
    def _build_v1(self, config, use_cross_attn, use_struct_fusion,
                  use_pos_enc, bidirectional, use_bypass):
        # 2. 时序编码
        self.temporal_proj = nn.Linear(1, config.D_MODEL)
        if use_pos_enc:
            self.pos_encoder = PositionalEncoding(
                config.D_MODEL,
                max_len=config.get_seq_len()
            )
        else:
            self.pos_encoder = nn.Identity()

        # 3. 结构特征投影
        self.struct_proj = nn.Linear(config.CNN_FEATURE_DIM, config.D_MODEL)

        # 4. 竖向分区数 (用于结构特征扩展)
        self.L = 10

        # 5. 交叉注意力层
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(
                config.D_MODEL,
                config.N_HEAD,
                config.D_FF,
                dropout=config.DROPOUT,
                bidirectional=bidirectional,
            )
            for _ in range(config.N_LAYER if use_cross_attn else 0)
        ])

        # 6. 结构融合层
        self.struct_fusion = nn.Sequential(
            nn.Linear(config.D_MODEL, config.D_MODEL),
            nn.LayerNorm(config.D_MODEL),
            nn.GELU()
        )

        # 7. 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(config.D_MODEL, config.D_MODEL // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.D_MODEL // 2, config.D_MODEL // 4),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.D_MODEL // 4, 1)
        )

        # 8. 残差旁路 + 输入缩放
        if use_bypass:
            self.motion_scale = nn.Parameter(torch.tensor(50.0))
            self.bypass_proj = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=31, padding=15),
                nn.GELU(),
                nn.Conv1d(32, 32, kernel_size=31, padding=15),
                nn.GELU(),
                nn.Conv1d(32, 1, kernel_size=31, padding=15),
            )

    # ------------------------------------------------------------
    # v2 现代架构 (注意力 Transformer + 前沿思路)
    # ------------------------------------------------------------
    def _build_v2(self, config, use_cross_attn, use_struct_fusion, film,
                  drop_path, use_bypass, use_sa=True, use_conv=True,
                  use_ffn=True, use_cond_params=True):
        d = config.D_MODEL
        self.L = 8  # 结构 token 数

        # 2. 时序编码: 输入归一化 + 线性投影 + 正弦位置编码
        self.input_norm = RMSNorm(1)
        self.temporal_proj = nn.Linear(1, d)
        self.pos_encoder = PositionalEncoding(d, max_len=config.get_seq_len())

        # 3. 结构编码: 投影 + LayerNorm + 结构 token (可学习, 让交叉注意力聚焦)
        self.struct_proj = nn.Linear(config.CNN_FEATURE_DIM, d)
        self.struct_norm = RMSNorm(d)
        self.struct_tokens = nn.Parameter(torch.randn(self.L, d) * (d ** -0.5))
        # 结构条件向量 (FiLM/bias 注入用)
        self.struct_cond_proj = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        # 显式结构参数条件注入 (层数/跨数/跨度/层高/质量/阻尼 + 每层楼面荷载)
        # 参数量纲差异大, 输入前标准化 (config.normalize_params)
        self.use_cond_params = use_cond_params
        from config import Config as _C
        params_dim = int(getattr(_C, 'PARAMS_DIM', 20))
        self.cond_params_proj = nn.Sequential(
            nn.Linear(params_dim, d), nn.GELU(), nn.Linear(d, d))

        # 4. RoPE
        self.rotary = RotaryEmbedding(d // config.N_HEAD, max_len=config.get_seq_len())

        # 5. 现代 Transformer 块 (Pre-LN + DropPath + RoPE + QK-Norm + FiLM + Conv + SwiGLU)
        dpr = [drop_path * (i / max(1, config.N_LAYER - 1)) for i in range(config.N_LAYER)]
        self.blocks = nn.ModuleList([
            V2Block(d, config.N_HEAD, config.D_FF, config.DROPOUT, drop_path=dpr[i],
                    rotary=self.rotary, use_cross_attn=use_cross_attn,
                    use_struct_fusion=use_struct_fusion,
                    conv_kernel=getattr(config, 'V2_CONV_KERNEL', 31),
                    use_sa=use_sa, use_conv=use_conv, use_ffn=use_ffn)
            for i in range(config.N_LAYER)
        ])

        # 6. 输出层
        self.head_norm = RMSNorm(d)
        self.output_layer = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(config.DROPOUT),
            nn.Linear(d // 2, d // 4), nn.GELU(), nn.Dropout(config.DROPOUT),
            nn.Linear(d // 4, 1),
        )

        # 7. 残差旁路: motion -> 位移 (强梯度路径, 改善收敛)
        if use_bypass:
            self.motion_scale = nn.Parameter(torch.tensor(50.0))
            self.bypass_proj = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=31, padding=15),
                nn.GELU(),
                nn.Conv1d(32, 32, kernel_size=31, padding=15),
                nn.GELU(),
                nn.Conv1d(32, 1, kernel_size=31, padding=15),
            )

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, octree_features, acceleration, cond_params=None, frame_features=None):
        """
        octree_features: [batch, F_octree] 预计算的八叉树特征 (USE_FRAME_FEATURE=False 时用)
        frame_features:  [batch, F_frame]  杆系结构化物理特征 (USE_FRAME_FEATURE=True 时用)
        acceleration: [batch, T, 1] 加速度时程
        cond_params: [batch, PARAMS_DIM=20] 结构参数 (可选, v2 显式条件注入)
                     p[0:8]  层数/跨数/跨度/层高/平均质量/阻尼
                     p[8:20] 每层楼面平均荷载 (kPa), 每层一个值, 最多 12 层, 不足补 0
        返回: [batch, T] 预测位移时程
        """
        if getattr(self, 'use_v2', False):
            return self._forward_v2(octree_features, acceleration, cond_params, frame_features)
        return self._forward_v1(octree_features, acceleration, frame_features)

    def _forward_v1(self, octree_features, acceleration, frame_features=None):
        batch_size, T = acceleration.shape[0], acceleration.shape[1]
        struct_input = frame_features if frame_features is not None else octree_features
        
        # ============================================================
        # 1. 提取结构特征
        # ============================================================
        struct_feat = self.struct_encoder(struct_input)  # [batch, cnn_dim]
        
        # ============================================================
        # 2. 投影到 D_MODEL
        # ============================================================
        struct_feat = self.struct_proj(struct_feat)  # [batch, d_model]
        
        # ============================================================
        # 3. 时序编码 (use_bypass 时对 motion 输入缩放, 量级 O(1))
        # ============================================================
        if getattr(self, 'use_bypass', False):
            acc_scale = getattr(self, 'motion_scale', torch.tensor(1.0))
            temporal_feat = self.temporal_proj(acceleration * acc_scale)
        else:
            temporal_feat = self.temporal_proj(acceleration)
        if self.use_pos_enc:
            temporal_feat = self.pos_encoder(temporal_feat)   # [batch, T, d_model]
        
        # ============================================================
        # 4. 注入结构特征到每个时间步
        # ============================================================
        if self.use_struct_fusion:
            struct_bias = self.struct_fusion(struct_feat).unsqueeze(1)  # [batch, 1, d_model]
            temporal_feat = temporal_feat + struct_bias
        
        # ============================================================
        # 5. 扩展结构特征为序列 (用于交叉注意力)
        # ============================================================
        struct_feat_expanded = struct_feat.unsqueeze(1).expand(-1, self.L, -1)
        
        # ============================================================
        # 6. 多层交叉注意力
        # ============================================================
        attn_weights_list = []
        if self.use_cross_attn:
            for layer in self.cross_attn_layers:
                temporal_feat, struct_feat_expanded, attn_weights = layer(
                    temporal_feat, struct_feat_expanded
                )
                attn_weights_list.append(attn_weights)
        
        # ============================================================
        # 7. 输出层
        # ============================================================
        output = self.output_layer(temporal_feat)  # [batch, T, 1]
        
        # ============================================================
        # 8. 残差旁路: motion -> 位移 (1D 卷积波形整形, 强梯度路径)
        # ============================================================
        if getattr(self, 'use_bypass', False):
            acc_scale = getattr(self, 'motion_scale', torch.tensor(1.0))
            bypass = self.bypass_proj((acceleration * acc_scale).transpose(1, 2))
            bypass = bypass.transpose(1, 2)  # [B, T, 1]
            output = output + bypass
        
        return output.squeeze(-1), attn_weights_list

    def _forward_v2(self, octree_features, acceleration, cond_params=None, frame_features=None):
        """v2 现代架构前向"""
        B, T = acceleration.shape[0], acceleration.shape[1]
        struct_input = frame_features if frame_features is not None else octree_features

        # 1. 结构特征 + 结构 token + 条件向量
        struct_feat = self.struct_encoder(struct_input)          # [B, cnn_dim]
        struct_feat = self.struct_norm(self.struct_proj(struct_feat))  # [B, d]
        struct_tokens = self.struct_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, L, d]
        struct_tokens = struct_tokens + struct_feat.unsqueeze(1)   # 结构条件融入 token
        struct_cond = self.struct_cond_proj(struct_feat)            # [B, d] (FiLM/bias 用)

        # 1b. 显式结构参数条件注入 (可选, 消融: use_cond_params=False 时跳过)
        if cond_params is not None and getattr(self, 'use_cond_params', True):
            cp = cond_params.float()
            # 轻量标准化 (量纲对齐): 前 8 维结构参数 + 形状 ID + 每层荷载 (kPa)
            from config import Config as _C
            pdim = int(getattr(_C, 'PARAMS_DIM', 21))
            sidx = int(getattr(_C, 'PARAMS_SHAPE_IDX', 8))
            scales = [0.5, 0.3, 0.3, 0.25, 0.25, 0.5, 1e-4, 20.0]
            scales = scales[:sidx] + [0.2]  # 形状 ID (0~4) -> 0.2
            if pdim > sidx + 1:
                scales += [0.1] * (pdim - (sidx + 1))   # 每层荷载 kPa -> 0.1
            cp = cp * torch.tensor(scales, device=cp.device, dtype=cp.dtype)
            param_cond = self.cond_params_proj(cp)                 # [B, d]
            struct_cond = struct_cond + param_cond
            struct_tokens = struct_tokens + param_cond.unsqueeze(1)

        # 2. 时序编码: 输入归一化 + 缩放 + 投影 + 位置编码
        acc = acceleration * getattr(self, 'motion_scale', torch.tensor(1.0))
        temporal = self.temporal_proj(self.input_norm(acc))         # [B, T, d]
        temporal = self.pos_encoder(temporal)

        # 3. 现代 Transformer 块
        attn_weights_list = []
        for block in self.blocks:
            temporal = block(temporal, struct_tokens, struct_cond)

        # 4. 输出
        output = self.output_layer(self.head_norm(temporal))        # [B, T, 1]

        # 5. 残差旁路: motion -> 位移
        if getattr(self, 'use_bypass', False):
            bypass = self.bypass_proj((acceleration * getattr(self, 'motion_scale', torch.tensor(1.0))).transpose(1, 2))
            bypass = bypass.transpose(1, 2)
            output = output + bypass

        return output.squeeze(-1), attn_weights_list


# ============================================================
# 因果解码器 (CausalSLDecoder)
# 给定结构特征 + 真实地震动加速度, 逐时刻预测屋顶位移时程。
# 不提前知道未来:
#   - 输入只有真实地震动 motion (不含位移) -> 不泄漏位移
#   - 自注意力用因果下三角掩码 -> t 时刻只用 <=t 的地震动, 不泄漏未来加速度
# ============================================================
class CausalBlock(nn.Module):
    """因果解码块: 因果自注意力 + 结构交叉注意力 + FFN"""

    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True,
                                               dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True,
                                                dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, temporal, struct_seq, causal_mask):
        """
        temporal: [B, T, d_model]  (地震动时序)
        struct_seq: [B, L, d_model] (结构特征序列, 无时间泄漏)
        causal_mask: [T, T] 因果掩码 (True=屏蔽未来)
        """
        # 因果自注意力 (只能看 <=t)
        attn1, _ = self.self_attn(temporal, temporal, temporal, attn_mask=causal_mask)
        temporal = self.norm1(temporal + self.dropout(attn1))
        # 结构交叉注意力 (结构无时间维, 不泄漏未来)
        attn2, _ = self.cross_attn(temporal, struct_seq, struct_seq)
        temporal = self.norm2(temporal + self.dropout(attn2))
        # FFN
        temporal = self.norm3(temporal + self.ffn(temporal))
        return temporal


class CausalSLDecoder(nn.Module):
    """
    因果解码器: 结构特征 + 真实地震动 -> 屋顶位移时程

    输入:
        octree_features: [B, F] 八叉树结构特征
        motion: [B, T] 真实地震动加速度时程 (g)

    输出:
        displacement: [B, T] 预测位移时程 (mm)
    """

    def __init__(self, config, use_pos_enc=True):
        super().__init__()
        self.config = config
        self.seq_len = config.get_seq_len()
        d = config.D_MODEL

        # 结构编码
        self.struct_encoder = PrecomputedOctreeEncoder(
            output_dim=config.CNN_FEATURE_DIM,
            max_depth=config.OCTREE_DEPTH,
            input_dim=config.OCTREE_FEATURE_DIM,
        )
        self.struct_proj = nn.Linear(config.CNN_FEATURE_DIM, d)

        # 地震动输入投影 (motion 单位 g, 量级小, 加固定缩放稳定)
        self.motion_proj = nn.Linear(1, d)
        self.motion_scale = nn.Parameter(torch.tensor(10.0))
        if use_pos_enc:
            self.pos_encoder = PositionalEncoding(d, max_len=self.seq_len)
        else:
            self.pos_encoder = nn.Identity()

        # 结构融合 (bias 注入每个时间步)
        self.struct_fusion = nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU()
        )
        self.L = 10  # 结构序列长度

        # 因果解码块
        self.decoder_layers = nn.ModuleList([
            CausalBlock(d, config.N_HEAD, config.D_FF, config.DROPOUT)
            for _ in range(config.N_LAYER)
        ])

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(config.DROPOUT),
            nn.Linear(d // 2, d // 4), nn.GELU(), nn.Dropout(config.DROPOUT),
            nn.Linear(d // 4, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_causal_mask(self, T, device):
        """下三角因果掩码: True=屏蔽 (上三角被屏蔽, 只能看 <=t)"""
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        return mask.to(device)

    def forward(self, octree_features, motion):
        """
        octree_features: [B, F]
        motion: [B, T] 真实地震动 (g)
        返回: displacement [B, T]
        """
        B, T = motion.shape
        device = motion.device

        # 结构特征
        struct_feat = self.struct_encoder(octree_features)      # [B, cnn_dim]
        struct_feat = self.struct_proj(struct_feat)             # [B, d]

        # 地震动输入
        motion = motion.unsqueeze(-1)                            # [B, T, 1]
        temporal = self.motion_proj(motion * self.motion_scale)  # [B, T, d]
        temporal = self.pos_encoder(temporal)

        # 结构融合 bias
        struct_bias = self.struct_fusion(struct_feat).unsqueeze(1)  # [B, 1, d]
        temporal = temporal + struct_bias

        # 结构序列 (交叉注意力用)
        struct_seq = struct_feat.unsqueeze(1).expand(-1, self.L, -1)  # [B, L, d]

        # 因果掩码
        causal_mask = self._make_causal_mask(T, device)

        # 因果解码
        for layer in self.decoder_layers:
            temporal = layer(temporal, struct_seq, causal_mask)

        output = self.output_layer(temporal)  # [B, T, 1]
        return output.squeeze(-1)


# ============================================================
# 测试
# ============================================================

if __name__ == '__main__':
    from config import Config
    cfg = Config()
    model = SLFormer(cfg)
    
    dummy_octree = torch.randn(4, 96)
    dummy_acc = torch.randn(4, cfg.get_seq_len(), 1)
    
    out, attn = model(dummy_octree, dummy_acc)
    print(f"输出形状: {out.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")