# visualize_comprehensive.py
"""
综合可视化模块 - 读取现有样本集 (simulation_cache.pkl) 绘制结构图

功能:
1. 杆系模型轴测图 (3D 等轴测)
2. 杆系模型平面图 (俯视图 X-Y)
3. 杆系模型立面图 (X-Z 与 Y-Z 两个立面)
4. 顶点位移时程曲线
5. 加速度加载时程曲线 (输入地震动)
6. 批量: 50 个样本，每 10 个样本轴测图拼成一张大图

数据来源:
- simulation_cache.pkl: params [N,8], displacements [N,T], motion_indices [N]
- motion_pool.pkl: 地震动池 (若存在，用于还原输入加速度)
- 框架由 params 通过 generate_fixed_frame 重建 (与数据集/仿真一致的确定性重建)

说明:
- 不修改现有模型结构与缓存数据结构
- 框架重建使用 tuple 格式 (与 generate_fixed_frame 返回一致)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec
import os
import pickle
import warnings
from config import Config
from generate_frames import generate_fixed_frame
from simulation_cache import SimulationCache
warnings.filterwarnings('ignore')

# ============================================================
# SCI 论文配图样式
# ------------------------------------------------------------
# 期刊配图要求: 矢量格式 (PDF) / 高分辨率 (>=300 dpi)、
# 白底、Times New Roman 字体、线宽/字号统一、无边框干扰、
# 坐标轴刻度向内、图例放图内空白处。
# ============================================================
# 中文字体（标题用中文时保持可读；SCI 输出用英文标签避免方框）
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['axes.labelpad'] = 6
plt.rcParams['legend.frameon'] = False
plt.rcParams['savefig.bbox'] = 'tight'

# 默认构架颜色
COLOR_COLUMN = '#4A7DB4'
COLOR_BEAM_X = '#6B8E6B'
COLOR_BEAM_Y = '#D4A574'
COLOR_SLAB = '#D4A574'

# SCI 输出格式: 优先矢量 PDF (期刊投稿), 同时出高分辨率 PNG
SCI_FORMATS = ['pdf', 'png']
SCI_DPI = 300
SCI_FONTSIZE_TICK = 9
SCI_FONTSIZE_LABEL = 11


def save_figure(fig, path, dpi=None, formats=None):
    """按 SCI 要求保存图形 (矢量 PDF + 高分辨率 PNG)

    Args:
        fig: matplotlib figure
        path: 保存路径 (不含扩展名)
        dpi: 位图分辨率 (默认 SCI_DPI=300)
        formats: 保存格式列表 (默认 ['pdf','png'])
    """
    dpi = dpi or SCI_DPI
    formats = formats or SCI_FORMATS
    saved = []
    for fmt in formats:
        if fmt == 'pdf':
            p = f"{path}.pdf"
            fig.savefig(p, dpi=dpi, format='pdf', bbox_inches='tight',
                        facecolor='white')
        else:
            p = f"{path}.{fmt}"
            fig.savefig(p, dpi=dpi, format=fmt, bbox_inches='tight',
                        facecolor='white')
        saved.append(p)
    return saved


def _apply_sci_axes(ax, fs_tick=None, fs_label=None):
    """统一 SCI 坐标轴样式: 刻度朝内、统一字号"""
    fs_tick = fs_tick or SCI_FONTSIZE_TICK
    fs_label = fs_label or SCI_FONTSIZE_LABEL
    ax.tick_params(axis='both', which='major', labelsize=fs_tick,
                   direction='in')
    ax.xaxis.label.set_fontsize(fs_label)
    ax.yaxis.label.set_fontsize(fs_label)
    if hasattr(ax, 'zaxis'):
        try:
            ax.zaxis.label.set_fontsize(fs_label)
        except Exception:
            pass


# ============================================================
# 框架重建
# ============================================================
def rebuild_frame(params):
    """
    从 8 维结构参数重建框架 (tuple 格式，与 generate_fixed_frame 一致)

    params: [num_stories, num_bays_x, num_bays_y,
             bay_width_x, bay_width_y, story_height,
             mass_per_node, damping_ratio]
    """
    num_stories = int(params[0])
    num_bays_x = int(params[1])
    num_bays_y = int(params[2])
    bay_width_x = float(params[3])
    bay_width_y = float(params[4])
    story_height = float(params[5])

    # 梁截面估算 (与 dataset._compute_octree_single 一致, 200mm 模数)
    max_span = max(bay_width_x, bay_width_y)
    beam_height = max(0.4, min(max_span / 12, 0.8))
    beam_height = round(beam_height / 0.2) * 0.2
    beam_width = max(0.2, min(beam_height / 2.5, 0.5))
    beam_width = round(beam_width / 0.2) * 0.2

    frame = generate_fixed_frame(
        num_stories=num_stories,
        num_spans_x=num_bays_x,
        num_spans_y=num_bays_y,
        span_x=bay_width_x,
        span_y=bay_width_y,
        story_height=story_height,
        axis_ratio=0.6,
        beam_width=beam_width,
        beam_height=beam_height,
    )
    return frame


def rebuild_frame_from_struct(struct):
    """从数据库 structures 字段重建框架 (与仿真时完全一致的参数)"""
    num_stories = int(struct['num_stories'])
    num_bays_x = int(struct['num_bays_x'])
    num_bays_y = int(struct['num_bays_y'])
    span_x = float(struct['span_x'])
    span_y = float(struct['span_y'])
    story_height = float(struct['story_height'])
    beam_width = float(struct.get('beam_width', 0.3))
    beam_height = float(struct.get('beam_height', 0.6))
    slab_thickness = float(struct.get('slab_thickness', 0.2))

    col_sections = struct.get('col_sections') or []
    if not col_sections:
        col_sections = [float(beam_width)] * num_stories

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
    )
    # 覆盖为数据库记录的真实截面/板厚 (与仿真一致)
    frame['slab_thickness'] = slab_thickness
    frame['col_sections'] = list(col_sections)
    return frame


def get_frame_limits(frame):
    """获取框架几何范围"""
    num_spans_x = frame.get('num_spans_x', 1)
    num_spans_y = frame.get('num_spans_y', 1)
    span_x = frame.get('span_x', 6.0)
    span_y = frame.get('span_y', 6.0)
    total_height = frame.get('total_height', 10.0)
    return {
        'x_max': num_spans_x * span_x,
        'y_max': num_spans_y * span_y,
        'z_max': total_height,
    }


def compute_unified_scale(frames, margin=0.5):
    """
    计算所有样本共用的统一缩放尺度。

    需求:
    - X、Y 从 0 开始 (结构坐标均为正，无负数)
    - 所有样本共用同一套缩放比例
    - XYZ 三个轴取同一比例 (单位长度相同)

    返回 dict: {x_max, y_max, z_max, unit, margin}
      - x_max/y_max/z_max: 全局最大值 (取所有样本的最大)
      - unit = max(x_max, y_max, z_max): 三轴统一长度 (XYZ 同比例)
      - 各轴绘图范围均为 [0, unit] (或 [0, axis_max] 当 axis_max 接近 unit)
    """
    gx = gy = gz = 0.0
    for frame in frames:
        lim = get_frame_limits(frame)
        gx = max(gx, lim['x_max'])
        gy = max(gy, lim['y_max'])
        gz = max(gz, lim['z_max'])
    unit = max(gx, gy, gz)
    return {
        'x_max': gx,
        'y_max': gy,
        'z_max': gz,
        'unit': unit,
        'margin': margin,
    }


def get_single_scale(frame, margin=0.5):
    """单样本时的缩放 (无全局 scale 时回退用自身范围)"""
    lim = get_frame_limits(frame)
    unit = max(lim['x_max'], lim['y_max'], lim['z_max'])
    return {
        'x_max': lim['x_max'],
        'y_max': lim['y_max'],
        'z_max': lim['z_max'],
        'unit': unit,
        'margin': margin,
    }


# ============================================================
# 绘图函数 (适配 tuple 格式框架)
# ============================================================
class FramePlots:
    """基于 tuple 格式框架的绘图函数集合"""

    @staticmethod
    def _iter_columns(frame):
        """生成柱线段 [(x,y,z_bottom,z_top)]"""
        for col in frame.get('columns', []):
            if len(col) == 5:
                x, y, z_bottom, z_top, section = col
            else:  # dict 兼容
                x, y, z_bottom, z_top, section = (col['x'], col['y'],
                                                  col['z_bottom'], col['z_top'],
                                                  col['section'])
            yield (x, y, z_bottom, z_top)

    @staticmethod
    def _iter_beams(frame):
        """生成梁线段 dict: x1,y1,x2,y2,z,direction"""
        for beam in frame.get('beams', []):
            if len(beam) == 8:
                x1, x2, y1, y2, z, w, h, direction = beam
            else:  # dict 兼容
                x1, x2, y1, y2, z, w, h, direction = (
                    beam['x1'], beam['x2'], beam['y1'], beam['y2'],
                    beam['z'], beam['width'], beam['height'], beam['direction'])
            yield {
                'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2,
                'z': z, 'direction': direction,
            }

    @staticmethod
    def _iter_slabs(frame):
        """生成板四角坐标"""
        for slab in frame.get('slabs', []):
            if len(slab) == 6:
                x_start, x_end, y_start, y_end, z, th = slab
            else:  # dict 兼容
                x_start, x_end, y_start, y_end, z, th = (
                    slab['x_start'], slab['x_end'], slab['y_start'],
                    slab['y_end'], slab['z'], slab['thickness'])
            yield (x_start, x_end, y_start, y_end, z)

    @classmethod
    def plot_axonometric(cls, ax, frame, title=None, show_slabs=True, scale=None):
        """轴测图 (3D)

        scale: 全局统一缩放 dict (含 x_max/y_max/z_max/unit)；None 时用样本自身范围
        所有轴均从 0 开始，X/Y/Z 用同一 unit 比例 (等长单位)。
        """
        # 柱
        for (x, y, z_bottom, z_top) in cls._iter_columns(frame):
            ax.plot([x, x], [y, y], [z_bottom, z_top],
                    color=COLOR_COLUMN, linewidth=3, alpha=0.9)
        # 梁
        for beam in cls._iter_beams(frame):
            if beam['direction'] == 'x':
                ax.plot([beam['x1'], beam['x2']], [beam['y1'], beam['y1']],
                        [beam['z'], beam['z']], color=COLOR_BEAM_X, linewidth=2, alpha=0.8)
            else:
                ax.plot([beam['x1'], beam['x1']], [beam['y1'], beam['y2']],
                        [beam['z'], beam['z']], color=COLOR_BEAM_Y, linewidth=2, alpha=0.8)
        # 板 (半透明)
        if show_slabs:
            polys = []
            for (xs, xe, ys, ye, z) in cls._iter_slabs(frame):
                polys.append([[xs, ys, z], [xe, ys, z], [xe, ye, z], [xs, ye, z]])
            if polys:
                ax.add_collection3d(Poly3DCollection(polys, alpha=0.08,
                                                     facecolors=COLOR_SLAB,
                                                     edgecolors='none'))

        if scale is None:
            scale = get_single_scale(frame)
        m = scale['margin']
        # 所有轴从 0 开始，长度取统一 unit (XYZ 同比例)
        ax.set_xlim([0, scale['unit'] + m])
        ax.set_ylim([0, scale['unit'] + m])
        ax.set_zlim([0, scale['unit'] + m])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        # XYZ 等比例 (三轴同单位长度)
        try:
            ax.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass
        ax.view_init(elev=25, azim=-45)
        if title:
            ax.set_title(title, fontsize=12)

    @classmethod
    def plot_plan(cls, ax, frame, title=None, scale=None):
        """平面图 (俯视图 X-Y)

        scale: 全局统一缩放；None 时用样本自身范围。
        X/Y 均从 0 开始，长度取统一 unit (XY 同比例)。
        """
        # 柱平面位置
        for (x, y, z_bottom, z_top) in cls._iter_columns(frame):
            if z_bottom < 0.01:  # 只画底层柱位置
                ax.plot(x, y, 'o', color=COLOR_COLUMN, markersize=6)
        # 梁投影
        for beam in cls._iter_beams(frame):
            if beam['direction'] == 'x':
                ax.plot([beam['x1'], beam['x2']], [beam['y1'], beam['y1']],
                        color=COLOR_BEAM_X, linewidth=1.5, alpha=0.8)
            else:
                ax.plot([beam['x1'], beam['x1']], [beam['y1'], beam['y2']],
                        color=COLOR_BEAM_Y, linewidth=1.5, alpha=0.8)
        if scale is None:
            scale = get_single_scale(frame)
        m = scale['margin']
        ax.set_xlim([0, scale['unit'] + m])
        ax.set_ylim([0, scale['unit'] + m])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if title:
            ax.set_title(title, fontsize=12)

    @classmethod
    def plot_elevation(cls, ax, frame, view='x', title=None, scale=None):
        """
        立面图
        view='x': X-Z 立面 (看向 Y 方向)
        view='y': Y-Z 立面 (看向 X 方向)

        scale: 全局统一缩放；None 时用样本自身范围。
        水平/垂直轴均从 0 开始，长度取统一 unit (等比例)。
        """
        # 柱
        for (x, y, z_bottom, z_top) in cls._iter_columns(frame):
            h = x if view == 'x' else y
            ax.plot([h, h], [z_bottom, z_top], color=COLOR_COLUMN, linewidth=3, alpha=0.9)
        # 梁
        for beam in cls._iter_beams(frame):
            if view == 'x':
                if beam['direction'] == 'x':
                    ax.plot([beam['x1'], beam['x2']], [beam['z'], beam['z']],
                            color=COLOR_BEAM_X, linewidth=2, alpha=0.8)
                else:
                    ax.plot([beam['y1'], beam['y1']], [beam['z'], beam['z']],
                            color=COLOR_BEAM_Y, linewidth=2, alpha=0.8)
            else:  # view='y'
                if beam['direction'] == 'y':
                    ax.plot([beam['y1'], beam['y2']], [beam['z'], beam['z']],
                            color=COLOR_BEAM_Y, linewidth=2, alpha=0.8)
                else:
                    ax.plot([beam['x1'], beam['x1']], [beam['z'], beam['z']],
                            color=COLOR_BEAM_X, linewidth=2, alpha=0.8)
        if scale is None:
            scale = get_single_scale(frame)
        m = scale['margin']
        ax.set_xlim([0, scale['unit'] + m])
        ax.set_ylim([0, scale['unit'] + m])
        ax.set_xlabel('X (m)' if view == 'x' else 'Y (m)')
        ax.set_ylabel('Z (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if title:
            ax.set_title(title, fontsize=12)

    @classmethod
    def plot_displacement_time(cls, ax, displacement, dt=0.02, title=None):
        """Roof / top displacement time history (mm)"""
        T = len(displacement)
        t = np.arange(T) * dt
        ax.plot(t, displacement, 'b-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Top displacement (mm)')
        ax.grid(True, alpha=0.3)
        max_idx = int(np.argmax(np.abs(displacement)))
        ax.scatter(t[max_idx], displacement[max_idx], color='red', s=30, zorder=5)
        ax.annotate(f'Max={abs(displacement[max_idx]):.2f} mm',
                    xy=(t[max_idx], displacement[max_idx]),
                    xytext=(8, 8), textcoords='offset points', fontsize=9, color='red')
        max_disp = np.max(np.abs(displacement))
        rms = np.sqrt(np.mean(displacement ** 2))
        ax.text(0.02, 0.95, f'Peak: {max_disp:.2f} mm\nRMS: {rms:.2f} mm',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        if title:
            ax.set_title(title, fontsize=12)

    @classmethod
    def plot_motion_time(cls, ax, motion, dt=0.02, title=None):
        """Ground acceleration time history (input earthquake, unit g)"""
        T = len(motion)
        t = np.arange(T) * dt
        ax.plot(t, motion, 'r-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Ground acceleration (g)')
        ax.grid(True, alpha=0.3)
        peak_idx = int(np.argmax(np.abs(motion)))
        ax.scatter(t[peak_idx], motion[peak_idx], color='blue', s=30, zorder=5)
        ax.annotate(f'PGA={abs(motion[peak_idx]):.4f} g',
                    xy=(t[peak_idx], motion[peak_idx]),
                    xytext=(8, 8), textcoords='offset points', fontsize=9, color='blue')
        ax.text(0.02, 0.95, f'PGA: {np.max(np.abs(motion)):.4f} g',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        if title:
            ax.set_title(title, fontsize=12)


# ============================================================
# 综合可视化器 (单个样本)
# ============================================================
class ComprehensiveVisualizer:
    """综合可视化器: 读取样本集，输出各样本结构图与时程曲线"""

    def __init__(self, config=None, output_dir='./plots/comprehensive'):
        self.config = config or Config()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.motion_pool = self._load_motion_pool()

    def _load_motion_pool(self):
        motion_file = os.path.join(self.config.CACHE_DIR, 'motion_pool.pkl')
        if os.path.exists(motion_file):
            try:
                with open(motion_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None

    # ------------------------------------------------------------
    # 单个样本综合图 (一张大图: 轴测/平面/立面/位移时程/加速度时程)
    # ------------------------------------------------------------
    def visualize_sample(self, params, displacement, motion_index=0, sample_id=0,
                         save_format='png', dpi=110, scale=None,
                         motion=None, meta=None, struct=None,
                         formats=None):
        """
        绘制单个样本的综合图 (SCI 配图: 矢量 PDF + 高分辨率 PNG)

        Args:
            params: [8] 结构参数 或 struct dict (DB 模式)
            displacement: [T] 顶点位移时程 (mm)
            motion_index: 地震动池索引 (pkl 模式回退用)
            sample_id: 样本编号
            dpi: 位图分辨率 (默认用 SCI_DPI=300)
            scale: 全局统一缩放 dict；None 时用样本自身范围
            motion: 实际输入地震动加速度时程 (g) (DB 模式直接提供)
            meta: 样本附加信息 dict (PGA/截面等, DB 模式提供)
            struct: 数据库结构 dict (DB 模式提供, 优先用真实截面重建框架)
            formats: 保存格式列表 (默认 ['pdf','png'])
        """
        # 优先用 struct (DB) 重建框架, 保证与仿真完全一致
        if struct is not None:
            frame = rebuild_frame_from_struct(struct)
            num_stories = int(struct['num_stories'])
        else:
            frame = rebuild_frame(params)
            num_stories = int(params[0])
        dt = self.config.TARGET_DT
        dpi = dpi or SCI_DPI
        formats = formats or SCI_FORMATS

        # 还原输入加速度 (g): 优先用实际地震波, 否则用位移二阶差分近似
        accel = None
        if motion is not None and len(motion) > 0:
            accel = np.asarray(motion, dtype=np.float64)
            if len(accel) > len(displacement):
                accel = accel[:len(displacement)]
            elif len(accel) < len(displacement):
                accel = np.pad(accel, (0, len(displacement) - len(accel)))
        if accel is None:
            try:
                accel = np.gradient(np.gradient(displacement, dt), dt) / 9.81
            except Exception:
                accel = np.zeros_like(displacement)

        # 若 DB 未提供且地震动池可用，用实际地震波 (pkl 回退)
        if motion is None and self.motion_pool is not None and motion_index < len(self.motion_pool):
            m = np.asarray(self.motion_pool[motion_index], dtype=np.float64)
            if len(m) < len(displacement):
                m = np.pad(m, (0, len(displacement) - len(m)))
            elif len(m) > len(displacement):
                m = m[:len(displacement)]
            accel = m

        sample_dir = os.path.join(self.output_dir, f'sample_{sample_id:04d}')
        os.makedirs(sample_dir, exist_ok=True)

        # ---- SCI 大图布局 ----
        fig = plt.figure(figsize=(10.5, 7.0))
        gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        FramePlots.plot_axonometric(ax1, frame, title='(a)', scale=scale)
        _apply_sci_axes(ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        FramePlots.plot_plan(ax2, frame, title='(b)', scale=scale)
        _apply_sci_axes(ax2)

        ax3 = fig.add_subplot(gs[0, 2])
        FramePlots.plot_elevation(ax3, frame, view='x', title='(c)', scale=scale)
        _apply_sci_axes(ax3)

        ax4 = fig.add_subplot(gs[1, 0])
        FramePlots.plot_elevation(ax4, frame, view='y', title='(d)', scale=scale)
        _apply_sci_axes(ax4)

        ax5 = fig.add_subplot(gs[1, 1])
        FramePlots.plot_displacement_time(ax5, displacement, dt=dt, title='(e)')
        _apply_sci_axes(ax5)

        ax6 = fig.add_subplot(gs[1, 2])
        FramePlots.plot_motion_time(ax6, accel, dt=dt, title='(f)')
        _apply_sci_axes(ax6)

        # 标题信息 (SCI: 简洁, 参数放图内/下方)
        total_height = frame.get('total_height', 0.0)
        peak = np.max(np.abs(displacement))
        pga = float(meta.get('pga', 0.0)) if meta else 0.0
        info = (f"Sample {sample_id}: {num_stories}-storey, "
                f"{frame.get('num_spans_x', 0)}\u00d7{frame.get('num_spans_y', 0)} bays, "
                f"H={total_height:.1f} m")
        fig.suptitle(info, fontsize=12, y=0.99)
        # 图内标注 (避免标题过挤, 符合期刊去噪原则)
        ax1.text2D(0.03, 0.95,
                   f"PGA={pga:.3f} g\nPeak disp.={peak:.2f} mm",
                   transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))

        save_path = os.path.join(sample_dir, f'comprehensive_report.{save_format}')
        # SCI 保存: 矢量 PDF + 高分辨率 PNG
        base = os.path.join(sample_dir, 'comprehensive_report')
        saved_paths = save_figure(fig, base, dpi=dpi, formats=formats)
        plt.close(fig)
        if saved_paths:
            save_path = saved_paths[0]

        # ---- 单图保存 (SCI 矢量 + 高分辨率) ----
        self._save_individual_plots(sample_dir, frame, displacement, accel,
                                    motion_index, params, dt, scale=scale,
                                    formats=formats, dpi=dpi)
        return save_path

    def _save_individual_plots(self, sample_dir, frame, displacement, accel,
                               motion_index, params, dt, dpi=None, scale=None,
                               formats=None):
        """保存单个样本的独立 SCI 配图 (矢量 PDF + 高分辨率 PNG)"""
        dpi = dpi or SCI_DPI
        formats = formats or SCI_FORMATS

        # 轴测图
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        FramePlots.plot_axonometric(ax, frame, scale=scale)
        _apply_sci_axes(ax)
        save_figure(fig, os.path.join(sample_dir, 'axonometric'), dpi=dpi,
                    formats=formats)
        plt.close(fig)

        # 平面图
        fig, ax = plt.subplots(figsize=(6, 6))
        FramePlots.plot_plan(ax, frame, scale=scale)
        _apply_sci_axes(ax)
        save_figure(fig, os.path.join(sample_dir, 'plan'), dpi=dpi, formats=formats)
        plt.close(fig)

        # 两个立面图
        for view, name in [('x', 'elevation_x'), ('y', 'elevation_y')]:
            fig, ax = plt.subplots(figsize=(5, 6))
            FramePlots.plot_elevation(ax, frame, view=view, scale=scale)
            _apply_sci_axes(ax)
            save_figure(fig, os.path.join(sample_dir, name), dpi=dpi,
                        formats=formats)
            plt.close(fig)

        # 顶点位移时程
        fig, ax = plt.subplots(figsize=(7, 3.5))
        FramePlots.plot_displacement_time(ax, displacement, dt=dt)
        _apply_sci_axes(ax)
        save_figure(fig, os.path.join(sample_dir, 'displacement_time'), dpi=dpi,
                    formats=formats)
        plt.close(fig)

        # 加速度时程
        fig, ax = plt.subplots(figsize=(7, 3.5))
        FramePlots.plot_motion_time(ax, accel, dt=dt)
        _apply_sci_axes(ax)
        save_figure(fig, os.path.join(sample_dir, 'motion_time'), dpi=dpi,
                    formats=formats)
        plt.close(fig)


# ============================================================
# 批量可视化: 读取样本集
# ============================================================
def load_samples_from_db(num_samples=None, limit=None, pga_range=None,
                         n_stories=None, db=None):
    """从 PostgreSQL 数据库加载样本数据 (用于可视化)

    Returns:
        dict: {params_list, displacement_list, motion_list, struct_list,
               meta_list, sample_ids, gm_ids}
    """
    from db_manager import SLFDatabase
    from dataset_db import SLFDbDataset
    db = db or SLFDatabase()
    rows = db.query_samples(pga_range=pga_range, n_stories=n_stories, limit=limit)
    n_rows = len(rows)
    n_use = n_rows if num_samples is None else min(num_samples, n_rows)

    # 多维分层均匀抽样 (与训练一致: 楼层+响应量级+结构形态), 保证可视化样本
    # 反映训练样本集分布 (最新原则)
    if n_use < n_rows:
        from dataset import _stratified_sample_uniform
        rng = np.random.default_rng(42)
        cfg = Config()
        n_peak = int(getattr(cfg, 'DB_STRATIFY_RESPONSE_BINS', 3) or 0)
        use_struct = bool(getattr(cfg, 'DB_STRATIFY_STRUCT', True))
        peak_bins = None if n_peak < 2 else []
        struct_bins = {} if use_struct else None
        rows = _stratified_sample_uniform(rows, n_use, rng,
                                          peak_bins=peak_bins,
                                          struct_bins=struct_bins)
        print(f"  多维分层均匀抽样 {len(rows)} 个样本 (楼层+响应量级+结构形态)")

    params_list, disp_list, motion_list = [], [], []
    struct_list, meta_list = [], []
    sample_ids, gm_ids = [], []
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

        p = SLFDbDataset._params_from_struct(struct)
        params_list.append(p)
        disp_list.append(np.asarray(resp['roof_disp'], dtype=np.float64))
        motion_list.append(np.asarray(gm['motion'], dtype=np.float64))
        struct_list.append(struct)
        sample_ids.append(sid)
        gm_ids.append(resp['gm_id'])

        loads = struct.get('floor_loads') or []
        masses = struct.get('floor_masses') or []
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
            'sample_id': sid,
            'gm_id': resp['gm_id'],
        })
    return {
        'params_list': params_list,
        'displacement_list': disp_list,
        'motion_list': motion_list,
        'struct_list': struct_list,
        'meta_list': meta_list,
        'sample_ids': sample_ids,
        'gm_ids': gm_ids,
    }


def visualize_samples_from_db(num_samples=None, output_dir='./plots/sci',
                              pga_range=None, n_stories=None,
                              formats=None, dpi=None):
    """从 PostgreSQL 数据库读取样本并批量可视化 (SCI 配图)

    Args:
        num_samples: 可视化样本数 (None=全部)
        output_dir: 输出目录
        pga_range: 可选过滤 (g)
        n_stories: 可选过滤层数
        formats: 输出格式 (默认 ['pdf','png'])
        dpi: 位图分辨率 (默认 300)
    """
    print("=" * 60)
    print("综合可视化 (SCI 配图) - 从 PostgreSQL 读取样本集")
    print("=" * 60)

    cfg = Config()
    visualizer = ComprehensiveVisualizer(cfg, output_dir)

    data = load_samples_from_db(num_samples=num_samples, pga_range=pga_range,
                                n_stories=n_stories)
    n_use = len(data['params_list'])
    if n_use == 0:
        print("  [X] 数据库中没有可用样本")
        return None

    print(f"  [*] 数据库样本: {n_use} 个")
    print(f"  [*] 输出格式: {formats or SCI_FORMATS}, DPI: {dpi or SCI_DPI}")

    # 全局统一缩放 (SCI 要求多图对比时比例一致)
    frames_for_scale = [rebuild_frame_from_struct(s) for s in data['struct_list']]
    global_scale = compute_unified_scale(frames_for_scale, margin=0.5)

    saved = 0
    try:
        from tqdm import tqdm
        iterator = tqdm(range(n_use), desc="  可视化进度", unit="样本")
    except ImportError:
        iterator = range(n_use)

    for i in iterator:
        try:
            visualizer.visualize_sample(
                params=data['params_list'][i],
                displacement=data['displacement_list'][i],
                motion=data['motion_list'][i],
                motion_index=0,
                sample_id=i,
                meta=data['meta_list'][i],
                struct=data['struct_list'][i],
                scale=global_scale,
                formats=formats, dpi=dpi,
            )
            saved += 1
        except Exception as e:
            import traceback
            print(f"  [W] 样本 {i} 可视化失败: {e}")
            traceback.print_exc()
        if not hasattr(iterator, 'set_description'):
            if (i + 1) % 5 == 0 or (i + 1) == n_use:
                print(f"  [*] 进度: {i+1}/{n_use}")

    print(f"\n[OK] SCI 配图完成! {saved} 个样本已保存: {output_dir}")
    return output_dir


def visualize_samples_from_cache(num_samples=None, output_dir='./plots/comprehensive'):
    """从仿真缓存读取样本并批量可视化 (pkl 回退, 兼容旧数据)

    Args:
        num_samples: 可视化样本数 (None=全部)
        output_dir: 输出目录
    """
    print("=" * 60)
    print("综合可视化 - 从缓存读取样本集")
    print("=" * 60)

    cfg = Config()
    visualizer = ComprehensiveVisualizer(cfg, output_dir)

    sim_cache = SimulationCache(cfg)
    if not sim_cache.load():
        print("  [X] 仿真缓存不存在，请先运行仿真生成器")
        return None

    sim_data = sim_cache.get_all()
    n_total = len(sim_data['params'])
    n_vis = n_total if num_samples is None else min(num_samples, n_total)

    print(f"  [*] 仿真缓存: {n_total} 个样本, 可视化 {n_vis} 个")
    if visualizer.motion_pool is not None:
        print(f"  [*] 地震动池: {len(visualizer.motion_pool)} 条")

    motion_indices = sim_data.get('motion_indices', np.zeros(n_total, dtype=np.int32))

    # 计算所有待可视化样本的全局统一缩放 (所有样本共用同一套比例)
    print("  [*] 计算全局统一缩放尺度...")
    frames_for_scale = [rebuild_frame(sim_data['params'][i]) for i in range(n_vis)]
    global_scale = compute_unified_scale(frames_for_scale, margin=0.5)
    print(f"      全局 Xmax={global_scale['x_max']:.1f}, Ymax={global_scale['y_max']:.1f}, "
          f"Zmax={global_scale['z_max']:.1f}, 统一unit={global_scale['unit']:.1f}")

    saved = 0
    # 用 tqdm 显示实时进度 (避免看起来卡死)
    try:
        from tqdm import tqdm
        iterator = tqdm(range(n_vis), desc="  可视化进度", unit="样本")
    except ImportError:
        iterator = range(n_vis)

    for i in iterator:
        params = sim_data['params'][i]
        displacement = sim_data['displacements'][i]
        motion_idx = int(motion_indices[i])
        try:
            visualizer.visualize_sample(
                params=params,
                displacement=displacement,
                motion_index=motion_idx,
                sample_id=i,
                scale=global_scale,
            )
            saved += 1
        except Exception as e:
            print(f"  [W] 样本 {i} 可视化失败: {e}")
        # 非 tqdm 模式时打印每 5 个样本进度
        if not hasattr(iterator, 'set_description'):
            if (i + 1) % 5 == 0 or (i + 1) == n_vis:
                print(f"  [*] 进度: {i+1}/{n_vis} 个样本已完成")

    print(f"\n[OK] 可视化完成! {saved} 个样本已保存: {output_dir}")
    return output_dir


# ============================================================
# 50 个样本轴测图拼接 (每 10 个拼一张)
# ============================================================
def visualize_tiled_axonometric(num_samples=50, per_tile=10,
                                output_dir='./plots/comprehensive_tiled'):
    """
    选 num_samples 个样本，每 per_tile 个样本的轴测图拼成一张大图

    默认: 50 个样本，每 10 个一张 → 5 张拼接图
    """
    print("=" * 60)
    print(f"轴测图拼接 - {num_samples} 个样本，每 {per_tile} 个拼一张")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    cfg = Config()

    sim_cache = SimulationCache(cfg)
    if not sim_cache.load():
        print("  [X] 仿真缓存不存在")
        return None

    sim_data = sim_cache.get_all()
    n_total = len(sim_data['params'])
    n_sel = min(num_samples, n_total)

    # 均匀采样 n_sel 个样本 (保持多样性)
    indices = np.linspace(0, n_total - 1, n_sel).astype(int)
    indices = np.unique(indices)
    if len(indices) > n_sel:
        indices = indices[:n_sel]

    n_tiles = int(np.ceil(len(indices) / per_tile))
    print(f"  [*] 选择 {len(indices)} 个样本, 生成 {n_tiles} 张拼接图")

    # 重建框架 (串行即可, 框架重建很快; 避免 Windows 多进程 spawn 造成的卡顿)
    print("  [*] 重建框架...")
    frames = [rebuild_frame(sim_data['params'][i]) for i in indices]

    # 所有样本共用同一套缩放比例 (XYZ 取同一 unit)
    global_scale = compute_unified_scale(frames, margin=0.5)
    print(f"      全局 Xmax={global_scale['x_max']:.1f}, Ymax={global_scale['y_max']:.1f}, "
          f"Zmax={global_scale['z_max']:.1f}, 统一unit={global_scale['unit']:.1f}")

    for t in range(n_tiles):
        start = t * per_tile
        end = min(start + per_tile, len(indices))
        chunk = indices[start:end]
        n_in_tile = end - start

        # 3x4 网格 (最多12个, 兼容10个)
        ncols = 4 if n_in_tile > 4 else n_in_tile
        nrows = int(np.ceil(n_in_tile / ncols))
        fig = plt.figure(figsize=(ncols * 4.5, nrows * 4.2))
        gs = GridSpec(nrows, ncols, figure=fig, hspace=0.2, wspace=0.1)

        for j, idx in enumerate(chunk):
            ax = fig.add_subplot(gs[j // ncols, j % ncols], projection='3d')
            frame = frames[start + j]
            num_stories = int(sim_data['params'][idx][0])
            FramePlots.plot_axonometric(ax, frame, show_slabs=False, scale=global_scale)
            ax.set_title(f"样本 {idx}: {num_stories}层", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

        # 隐藏空子图
        for j in range(n_in_tile, nrows * ncols):
            ax = fig.add_subplot(gs[j // ncols, j % ncols], projection='3d')
            ax.set_axis_off()

        fig.suptitle(f"样本轴测图拼接 第 {t+1}/{n_tiles} 组 "
                     f"(样本 {chunk[0]}~{chunk[-1]})",
                     fontsize=14, fontweight='bold')
        save_path = os.path.join(output_dir, f'tile_{t+1:02d}_samples_{chunk[0]}_{chunk[-1]}.png')
        plt.savefig(save_path, dpi=130, facecolor='white')
        plt.close(fig)
        print(f"  [*] 已保存 ({t+1}/{n_tiles}): {save_path}")

    print(f"\n[OK] 拼接完成! 共 {n_tiles} 张图: {output_dir}")
    return output_dir


# ============================================================
# 命令行入口
# ============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='综合可视化（默认从 PostgreSQL 读取，SCI 配图：矢量 PDF+300dpi PNG）')
    parser.add_argument('--num', type=int, default=10,
                        help='可视化的样本数量 (默认10)')
    parser.add_argument('--output', type=str, default='./plots/sci',
                        help='单个样本输出目录 (默认 ./plots/sci)')
    parser.add_argument('--source', type=str, default='db',
                        choices=['db', 'pkl'],
                        help='数据源: db=PostgreSQL(默认), pkl=simulation_cache.pkl')
    parser.add_argument('--pga_min', type=float, default=None,
                        help='过滤: 目标PGA下限 (g)')
    parser.add_argument('--pga_max', type=float, default=None,
                        help='过滤: 目标PGA上限 (g)')
    parser.add_argument('--n_stories', type=int, default=None,
                        help='过滤: 层数')
    parser.add_argument('--dpi', type=int, default=SCI_DPI,
                        help='位图分辨率 (默认300)')
    parser.add_argument('--formats', type=str, default='pdf,png',
                        help='输出格式, 逗号分隔 (默认 pdf,png)')
    parser.add_argument('--tile', action='store_true',
                        help='执行50样本轴测图拼接 (仅 pkl 源)')
    parser.add_argument('--tile_num', type=int, default=50,
                        help='拼接样本数 (默认50)')
    parser.add_argument('--per_tile', type=int, default=10,
                        help='每张拼接图样本数 (默认10)')
    parser.add_argument('--tile_output', type=str, default='./plots/comprehensive_tiled',
                        help='拼接图输出目录')
    args = parser.parse_args()

    formats = [f.strip() for f in args.formats.split(',') if f.strip()]

    if args.source == 'db':
        pga_range = None
        if args.pga_min is not None or args.pga_max is not None:
            pga_range = (args.pga_min or 0.0, args.pga_max or 1.0)
        visualize_samples_from_db(
            num_samples=args.num,
            output_dir=args.output,
            pga_range=pga_range,
            n_stories=args.n_stories,
            formats=formats,
            dpi=args.dpi,
        )
    else:
        visualize_samples_from_cache(num_samples=args.num, output_dir=args.output)
        if args.tile:
            visualize_tiled_axonometric(num_samples=args.tile_num,
                                        per_tile=args.per_tile,
                                        output_dir=args.tile_output)
