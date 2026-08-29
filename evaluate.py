# evaluate.py
"""
模型评估模块 - 使用训练时保存的数据集划分
功能：
1. 加载训练时保存的数据集划分索引，确保评估一致性
2. 在验证集上评估模型性能
3. 生成论文风格的可视化图表
4. 计算多种评估指标
python evaluate.py --model models_voxel_token/model/best_model.pth --use_db --max_samples 30000 --out plots/eval_token
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
import argparse
from config import Config
from dataset import OctreeDataset
from transformer_model import SLFormer
from train import weighted_mm_metric
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SCI 论文配图样式 (英文标签, 避免中文字体方框)
# ============================================================
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['legend.frameon'] = False
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['mathtext.fontset'] = 'stix'

SCI_DPI = 300
SCI_FORMATS = ['pdf', 'png']


def save_sci_figure(fig, path, dpi=None, formats=None):
    """按 SCI 要求保存图形 (矢量 PDF + 高分辨率 PNG)"""
    dpi = dpi or SCI_DPI
    formats = formats or SCI_FORMATS
    saved = []
    for fmt in formats:
        p = f"{path}.{fmt}"
        fig.savefig(p, dpi=dpi, format=fmt, bbox_inches='tight', facecolor='white')
        saved.append(p)
    return saved


def _apply_sci_axes(ax, fs_tick=None, fs_label=None):
    """统一 SCI 坐标轴样式: 刻度朝内、统一字号"""
    fs_tick = fs_tick or 9
    fs_label = fs_label or 11
    ax.tick_params(axis='both', which='major', labelsize=fs_tick, direction='in')
    ax.xaxis.label.set_fontsize(fs_label)
    ax.yaxis.label.set_fontsize(fs_label)
    if hasattr(ax, 'zaxis'):
        try:
            ax.zaxis.label.set_fontsize(fs_label)
        except Exception:
            pass


class Evaluator:
    """模型评估与可视化（论文风格）"""
    
    def __init__(self, model, config, device='cuda', plot_dir='./plots'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model.eval()
        
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # 存储样本数据用于可视化
        self.sample_data = []
    
    def evaluate(self, val_loader, num_samples=20):
        """在验证集上评估模型"""
        print("\n" + "="*60)
        print("模型评估 (验证集)")
        print("="*60)
        
        all_preds = []
        all_targets = []
        all_errors = []
        all_heights = []
        all_E_avg = []
        all_accelerations = []
        # 加权 mm 逐 batch 累加 (与 train 验证循环完全一致, 按 batch 平均)
        w_peak = float(getattr(self.config, 'LOSS_PEAK_W', 1.0))
        w_high = float(getattr(self.config, 'LOSS_HIGH_W', 3.0))
        loss_norm = str(getattr(self.config, 'LOSS_NORM', 'absolute')).lower()
        if loss_norm == 'relative':
            # 相对模式: 大位移阈值 = 峰值比例
            thresh = float(getattr(self.config, 'LOSS_HIGH_THRESH_RATIO', 0.2))
        else:
            thresh = float(getattr(self.config, 'LOSS_HIGH_THRESH_MM', 8.0))
        wmm_score_sum = 0.0
        wmm_mae_sum = 0.0
        wmm_peak_sum = 0.0
        wmm_high_sum = 0.0
        wmm_nbatch = 0
        
        with torch.no_grad():
            for batch in val_loader:
                octree_features = batch['octree_features'].to(self.device)
                target_disp = batch['disp'].to(self.device)
                height = batch['height'].to(self.device)
                E_avg = batch['E_avg'].to(self.device)
                # 真实地震动输入 + 结构参数 + 杆系特征 (与 train 完全一致)
                motion = batch['motion'].to(self.device)
                params = batch['params'].to(self.device)
                ffeat = (batch['frame_features'].to(self.device)
                         if batch.get('frame_features') is not None else None)
                batch_size, T = target_disp.shape

                # 预测 (与 train 相同的调用签名)
                pred_disp, _ = self.model(octree_features, motion.unsqueeze(-1),
                                          cond_params=params, frame_features=ffeat)

                # 加权 mm (与 train 验证完全一致: 每 batch 算, 最后平均; no_peak)
                if torch.isfinite(pred_disp).all():
                    v_score_b, v_mae_b, v_peak_b, v_high_b = weighted_mm_metric(
                        pred_disp, target_disp, w_peak=w_peak, w_high=w_high,
                        thresh_mm=thresh, loss_norm=loss_norm,
                        use_peak=False, use_shape=True, use_high=True)
                    wmm_score_sum += v_score_b.item()
                    wmm_mae_sum += v_mae_b.item()
                    wmm_peak_sum += v_peak_b.item()
                    wmm_high_sum += v_high_b.item()
                    wmm_nbatch += 1

                all_preds.append(pred_disp.cpu().numpy())
                all_targets.append(target_disp.cpu().numpy())
                all_errors.append((pred_disp - target_disp).cpu().numpy())
                all_heights.append(height.cpu().numpy())
                all_E_avg.append(E_avg.cpu().numpy())
                all_accelerations.append(motion.cpu().numpy())

                # 存储部分样本用于详细可视化
                if len(self.sample_data) < num_samples:
                    for i in range(min(batch_size, num_samples - len(self.sample_data))):
                        self.sample_data.append({
                            'octree_features': octree_features[i].cpu().numpy(),
                            'target_disp': target_disp[i].cpu().numpy(),
                            'pred_disp': pred_disp[i].cpu().numpy(),
                            'acceleration': motion[i].cpu().numpy().flatten(),
                            'height': height[i].cpu().numpy().item(),
                            'E_avg': E_avg[i].cpu().numpy().item()
                        })
        
        # 合并数据
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_errors = np.concatenate(all_errors, axis=0)
        all_heights = np.concatenate(all_heights, axis=0).flatten()
        all_E_avg = np.concatenate(all_E_avg, axis=0).flatten()
        
        # 加权 mm (逐 batch 平均, 与 train 验证完全一致)
        weighted_mm = wmm_score_sum / max(wmm_nbatch, 1)
        weighted_mm_mae = wmm_mae_sum / max(wmm_nbatch, 1)
        weighted_mm_peak = wmm_peak_sum / max(wmm_nbatch, 1)
        weighted_mm_high = wmm_high_sum / max(wmm_nbatch, 1)
        
        # 计算指标 (绝对误差, mm)
        flat_preds = all_preds.flatten()
        flat_targets = all_targets.flatten()
        
        r2 = r2_score(flat_targets, flat_preds)
        rmse = np.sqrt(mean_squared_error(flat_targets, flat_preds))
        mae = mean_absolute_error(flat_targets, flat_preds)
        corr = np.corrcoef(flat_preds, flat_targets)[0, 1] if len(flat_preds) > 1 else 0
        
        # 峰值指标 (绝对误差, mm)
        peak_preds = np.max(np.abs(all_preds), axis=1)
        peak_targets = np.max(np.abs(all_targets), axis=1)
        peak_errors = peak_preds - peak_targets
        peak_mae = np.mean(np.abs(peak_errors))
        peak_rmse = np.sqrt(np.mean(peak_errors**2))
        peak_r2 = r2_score(peak_targets, peak_preds)

        # ============================================================
        # 相对误差指标 (百分比 %)
        # ============================================================
        eps = 1e-3
        flat_errors = all_errors.flatten()
        
        # 1) 整体 MAPE: 平均绝对百分比误差, 分母加 eps 避免除零
        #    位移时程接近 0 的点百分比会偏大, 因此同时给出"按阈值过滤"版本
        mape_all = np.mean(np.abs(flat_errors) / (np.abs(flat_targets) + eps)) * 100.0
        
        # 按峰值阈值过滤: 仅统计 |target| > 10% 峰值的点 (避开近零点)
        peak_abs = np.abs(all_targets).max(axis=1, keepdims=True)
        mask = np.abs(all_targets) > 0.1 * peak_abs
        if mask.any():
            mape_peakzone = (np.abs(all_errors[mask]) /
                             (np.abs(all_targets[mask]) + eps)).mean() * 100.0
        else:
            mape_peakzone = 0.0
        
        # 2) 峰值相对误差: |peak_pred - peak_target| / |peak_target| * 100
        peak_rel = np.abs(peak_errors) / (np.abs(peak_targets) + eps) * 100.0
        peak_rel_mae = np.mean(peak_rel)          # 平均峰值相对误差 (%)
        peak_rel_median = np.median(peak_rel)     # 中位峰值相对误差 (%)
        peak_rel_std = np.std(peak_rel)           # 峰值相对误差标准差 (%)
        peak_rel_p90 = np.percentile(peak_rel, 90)  # 90% 分位 (%)
        
        # 3) 样本级 RMSE 相对误差: sqrt(mean(e^2)) / (peak_target) * 100
        sample_rmse = np.sqrt(np.mean(all_errors**2, axis=1))
        sample_rel_rmse = sample_rmse / (np.abs(peak_targets) + eps) * 100.0
        rel_rmse_mean = np.mean(sample_rel_rmse)      # 平均样本相对RMSE (%)
        rel_rmse_median = np.median(sample_rel_rmse)

        # ============================================================
        # 相对偏差 R² (每样本按峰值归一化, 大位移样本不主导)
        # ============================================================
        flat_norm_p = (all_preds / (np.abs(peak_targets)[:, None] + eps)).flatten()
        flat_norm_t = (all_targets / (np.abs(peak_targets)[:, None] + eps)).flatten()
        rel_r2 = r2_score(flat_norm_t, flat_norm_p)
        # 按位移幅值分组: 小位移 (<中位峰值) vs 大位移 (>=中位峰值)
        med_peak = float(np.median(peak_targets))
        small_idx = peak_targets < med_peak
        large_idx = peak_targets >= med_peak
        r2_small_all = r2_score(all_targets[small_idx].flatten(),
                                all_preds[small_idx].flatten()) if small_idx.sum() > 1 else 0.0
        r2_large_all = r2_score(all_targets[large_idx].flatten(),
                                all_preds[large_idx].flatten()) if large_idx.sum() > 1 else 0.0
        r2_small_peak = r2_score(peak_targets[small_idx], peak_preds[small_idx]) if small_idx.sum() > 1 else 0.0
        r2_large_peak = r2_score(peak_targets[large_idx], peak_preds[large_idx]) if large_idx.sum() > 1 else 0.0

        height_groups = self._group_by_height(all_preds, all_targets, all_heights)
        
        results = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'corr': corr,
            'peak_mae': peak_mae,
            'peak_rmse': peak_rmse,
            'peak_r2': peak_r2,
            # 加权 mm (与 train 训练/验证一致)
            'weighted_mm': weighted_mm,
            'weighted_mm_mae': weighted_mm_mae,
            'weighted_mm_peak': weighted_mm_peak,
            'weighted_mm_high': weighted_mm_high,
            'loss_norm': loss_norm,   # 标记加权指标量纲: relative=无量纲比值 / absolute=mm
            # 相对误差 (%)
            'mape_all': mape_all,
            'mape_peakzone': mape_peakzone,
            'peak_rel_mae': peak_rel_mae,
            'peak_rel_median': peak_rel_median,
            'peak_rel_std': peak_rel_std,
            'peak_rel_p90': peak_rel_p90,
            'rel_rmse_mean': rel_rmse_mean,
            'rel_rmse_median': rel_rmse_median,
            'peak_rel_all': peak_rel,      # 各样本峰值相对误差数组 (%)
            'sample_rel_rmse': sample_rel_rmse,  # 各样本相对RMSE (%)
            # 相对偏差 R² + 分组
            'rel_r2': float(rel_r2),
            'median_peak_mm': med_peak,
            'n_small': int(small_idx.sum()), 'n_large': int(large_idx.sum()),
            'r2_small_all': float(r2_small_all), 'r2_large_all': float(r2_large_all),
            'r2_small_peak': float(r2_small_peak), 'r2_large_peak': float(r2_large_peak),
            'height_groups': height_groups,
            'predictions': all_preds,
            'targets': all_targets,
            'errors': all_errors,
            'heights': all_heights,
            'E_avg': all_E_avg,
            'num_samples': len(all_preds)
        }
        
        self._print_results(results)
        
        # 生成图表
        print("\n生成可视化图表...")
        self.plot_results(results)
        self.plot_error_analysis(results)
        self.plot_relative_error(results)
        self.plot_peak_scatter(results)
        self.plot_height_analysis(results)
        self.plot_sample_comparison(results, num_samples=min(num_samples, 10))
        self.plot_single_sample_detail(0)
        
        print(f"\n✅ 所有图表已保存至: {self.plot_dir}")
        
        return results
    
    def _group_by_height(self, preds, targets, heights):
        """按高度分组统计"""
        height_bins = [0, 10, 20, 30, 40, 50, 100]
        groups = {}
        
        for i, h in enumerate(heights):
            for j in range(len(height_bins) - 1):
                if height_bins[j] <= h < height_bins[j+1]:
                    key = f"{height_bins[j]}-{height_bins[j+1]}m"
                    if key not in groups:
                        groups[key] = {'preds': [], 'targets': []}
                    groups[key]['preds'].extend(preds[i].flatten())
                    groups[key]['targets'].extend(targets[i].flatten())
                    break
        
        for key in groups:
            p = np.array(groups[key]['preds'])
            t = np.array(groups[key]['targets'])
            groups[key]['r2'] = r2_score(t, p) if len(t) > 1 else 0
            groups[key]['rmse'] = np.sqrt(mean_squared_error(t, p))
            groups[key]['mae'] = mean_absolute_error(t, p)
            groups[key]['n'] = len(t)
        
        return groups
    
    def _print_results(self, results):
        """打印评估结果"""
        print("\n" + "-"*60)
        print("评估结果")
        print("-"*60)
        print(f"  验证样本数:      {results['num_samples']}")
        print(f"  R² (整体):       {results['r2']:.4f}")
        print(f"  RMSE (整体):     {results['rmse']:.4f} mm")
        print(f"  MAE (整体):      {results['mae']:.4f} mm")
        print(f"  相关系数:        {results['corr']:.4f}")
        print(f"  峰值 MAE:        {results['peak_mae']:.4f} mm")
        print(f"  峰值 RMSE:       {results['peak_rmse']:.4f} mm")
        print(f"  峰值 R²:         {results['peak_r2']:.4f}")
        # 加权指标 (relative=无量纲比值显示%, absolute=绝对mm)
        if 'weighted_mm' in results:
            if results.get('loss_norm') == 'relative':
                wu = '%'
                scale = 100.0
            else:
                wu = 'mm'
                scale = 1.0
            print(f"  加权指标:        {results['weighted_mm']*scale:.4f}{wu} "
                  f"[{results.get('loss_norm','absolute')}]")
            print(f"     其中: 平均位移={results['weighted_mm_mae']*scale:.4f}{wu}, "
                  f"峰值={results['weighted_mm_peak']*scale:.4f}{wu}, "
                  f"大位移惩罚={results['weighted_mm_high']*scale:.4f}{wu}")
        # 相对偏差 R² + 分组
        if 'rel_r2' in results:
            print(f"  相对偏差 R²:     {results['rel_r2']:.4f}  (按样本峰值归一化)")
            print(f"  按峰值分组 (中位={results['median_peak_mm']:.2f} mm):")
            print(f"    小位移组 (n={results['n_small']}): 整体R²={results['r2_small_all']:.4f}, "
                  f"峰值R²={results['r2_small_peak']:.4f}")
            print(f"    大位移组 (n={results['n_large']}): 整体R²={results['r2_large_all']:.4f}, "
                  f"峰值R²={results['r2_large_peak']:.4f}")
        print("-"*60)
        print("相对误差 (百分比):")
        print(f"  整体 MAPE:       {results['mape_all']:.2f} %")
        print(f"  峰值区间 MAPE:   {results['mape_peakzone']:.2f} %  (|target|>10%峰值)")
        print(f"  峰值相对误差 MAE:{results['peak_rel_mae']:.2f} %")
        print(f"  峰值相对误差中位:{results['peak_rel_median']:.2f} %")
        print(f"  峰值相对误差 P90: {results['peak_rel_p90']:.2f} %")
        print(f"  峰值相对误差标准差: {results['peak_rel_std']:.2f} %")
        print(f"  样本相对 RMSE 均值:{results['rel_rmse_mean']:.2f} %")
        print(f"  样本相对 RMSE 中位:{results['rel_rmse_median']:.2f} %")
        print("-"*60)
        print("\n按高度分组:")
        for key, group in results['height_groups'].items():
            print(f"  {key}: R²={group['r2']:.4f}, RMSE={group['rmse']:.4f}mm, n={group['n']}")
    
    def plot_results(self, results):
        """绘制预测 vs 真实对比图"""
        preds = results['predictions']
        targets = results['targets']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        flat_preds = preds.flatten()
        flat_targets = targets.flatten()
        
        # 采样加速
        if len(flat_preds) > 10000:
            idx = np.random.choice(len(flat_preds), 10000, replace=False)
            flat_preds_s = flat_preds[idx]
            flat_targets_s = flat_targets[idx]
        else:
            flat_preds_s = flat_preds
            flat_targets_s = flat_targets
        
        # 预测 vs 真实
        axes[0, 0].scatter(flat_targets_s, flat_preds_s, alpha=0.3, s=2, c='#4A7DB4')
        lims = [min(flat_targets.min(), flat_preds.min()), 
                max(flat_targets.max(), flat_preds.max())]
        axes[0, 0].plot(lims, lims, 'k--', linewidth=1.5, label='y = x')
        axes[0, 0].set_xlabel('Target Displacement (mm)')
        axes[0, 0].set_ylabel('Predicted Displacement (mm)')
        axes[0, 0].set_title(f'Prediction vs Target (R²={results["r2"]:.4f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 误差分布
        flat_errors = results['errors'].flatten()
        axes[0, 1].hist(flat_errors, bins=80, edgecolor='black', alpha=0.7, 
                       color='#6B8E6B', density=True)
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        axes[0, 1].axvline(x=np.mean(flat_errors), color='blue', linestyle='--', 
                          linewidth=1.5, label=f'Mean: {np.mean(flat_errors):.3f}')
        axes[0, 1].set_xlabel('Prediction Error (mm)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title(f'Error Distribution (RMSE={results["rmse"]:.4f}mm)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 峰值误差
        peak_errors = np.max(np.abs(results['predictions']), axis=1) - \
                      np.max(np.abs(results['targets']), axis=1)
        axes[1, 0].hist(peak_errors, bins=40, edgecolor='black', alpha=0.7, 
                       color='#B85C4A', density=True)
        axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        axes[1, 0].set_xlabel('Peak Error (mm)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title(f'Peak Error Distribution (MAE={results["peak_mae"]:.4f}mm)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 峰值预测
        peak_preds = np.max(np.abs(results['predictions']), axis=1)
        peak_targets = np.max(np.abs(results['targets']), axis=1)
        lims_p = [min(peak_targets.min(), peak_preds.min()), 
                  max(peak_targets.max(), peak_preds.max())]
        xp = np.linspace(lims_p[0], lims_p[1], 100)
        axes[1, 1].fill_between(xp, xp * 1.10, xp * 0.90, color='red', alpha=0.06)
        axes[1, 1].fill_between(xp, xp * 1.05, xp * 0.95, color='green', alpha=0.08)
        axes[1, 1].scatter(peak_targets, peak_preds, alpha=0.5, s=15, c='#4A7DB4')
        axes[1, 1].plot(lims_p, lims_p, 'k--', linewidth=1.5)
        axes[1, 1].set_xlabel('Target Peak Displacement (mm)')
        axes[1, 1].set_ylabel('Predicted Peak Displacement (mm)')
        axes[1, 1].set_title(f'Peak Prediction (R²={results["peak_r2"]:.4f})')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, 'evaluation_overview')
        save_sci_figure(fig, save_path)
        print(f"✓ 评估总览图: {save_path}.pdf/.png")
        plt.close()
    
    def plot_error_analysis(self, results):
        """绘制误差分析图"""
        errors = results['errors']
        targets = results['targets']
        preds = results['predictions']
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 13))
        
        flat_targets = targets.flatten()
        flat_preds = preds.flatten()
        flat_errors = errors.flatten()
        
        # 误差 vs 真实值
        axes[0, 0].scatter(flat_targets[::10], flat_errors[::10], alpha=0.1, s=1, c='#4A7DB4')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
        axes[0, 0].set_xlabel('Target Displacement (mm)')
        axes[0, 0].set_ylabel('Error (mm)')
        axes[0, 0].set_title('Error vs Target')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 误差 vs 预测值
        axes[0, 1].scatter(flat_preds[::10], flat_errors[::10], alpha=0.1, s=1, c='#6B8E6B')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
        axes[0, 1].set_xlabel('Predicted Displacement (mm)')
        axes[0, 1].set_ylabel('Error (mm)')
        axes[0, 1].set_title('Error vs Prediction')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q图
        from scipy import stats
        stats.probplot(flat_errors[::10], dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('Q-Q Plot')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 样本RMSE分布
        sample_rmse = np.sqrt(np.mean(errors**2, axis=1))
        axes[1, 0].hist(sample_rmse, bins=40, edgecolor='black', alpha=0.7, 
                       color='#B85C4A')
        axes[1, 0].axvline(x=np.mean(sample_rmse), color='red', linestyle='--', 
                          linewidth=1.5, label=f'Mean: {np.mean(sample_rmse):.4f}')
        axes[1, 0].set_xlabel('Sample RMSE (mm)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('RMSE Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 累积误差
        sorted_errors = np.sort(np.abs(flat_errors))
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1, 1].plot(sorted_errors, cumulative, linewidth=2, c='#4A7DB4')
        axes[1, 1].axvline(x=results['mae'], color='red', linestyle='--', 
                          linewidth=1.5, label=f'MAE: {results["mae"]:.4f}')
        axes[1, 1].axvline(x=results['rmse'], color='blue', linestyle='--', 
                          linewidth=1.5, label=f'RMSE: {results["rmse"]:.4f}')
        axes[1, 1].set_xlabel('Absolute Error (mm)')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Error Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 误差 vs 高度
        heights = results['heights']
        axes[1, 2].scatter(heights, sample_rmse, alpha=0.5, s=20, c='#4A7DB4')
        z = np.polyfit(heights, sample_rmse, 1)
        p = np.poly1d(z)
        h_sorted = np.sort(heights)
        axes[1, 2].plot(h_sorted, p(h_sorted), 'r--', linewidth=1.5, 
                       label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
        axes[1, 2].set_xlabel('Building Height (m)')
        axes[1, 2].set_ylabel('Sample RMSE (mm)')
        axes[1, 2].set_title('RMSE vs Building Height')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        # ============================================================
        # 第三行: 相对误差 (百分比) vs Target + 95% 保证率曲线
        # ============================================================
        eps = 1e-3
        # 百分比相对误差 (避免过零点爆炸, 用 |tgt|+eps 分母; 同时过滤极小位移)
        rel_err_pct = np.abs(flat_errors) / (np.abs(flat_targets) + eps) * 100.0
        # 过滤: 目标位移太小 (<0.1mm) 时相对误差无意义, 避免散点爆炸
        mask_valid = np.abs(flat_targets) > 0.1
        # x 轴用 |target| (绝对位移), 更符合"位移越大相对误差越小"的可读性
        tx = np.abs(flat_targets[mask_valid])
        rel = rel_err_pct[mask_valid]

        # (2,0) 相对误差散点 vs target
        if len(tx) > 2000:
            idx = np.random.choice(len(tx), 2000, replace=False)
            axes[2, 0].scatter(tx[idx], rel[idx], alpha=0.2, s=2, c='#B85C4A')
        else:
            axes[2, 0].scatter(tx, rel, alpha=0.2, s=2, c='#B85C4A')
        axes[2, 0].set_xlabel('Target Displacement (mm)')
        axes[2, 0].set_ylabel('Relative Error (%)')
        axes[2, 0].set_title('Relative Error vs Target')
        axes[2, 0].set_yscale('log')
        axes[2, 0].grid(True, alpha=0.3)

        # (2,1) 95% 保证率包络: 按 target 分箱, 每箱画 P95(及 P50) 相对误差阶梯线
        # 用分位数回归式分箱: 按 target 分位数分 20 箱保证每箱样本数接近
        n_bins = 20
        bin_edges = np.quantile(tx, np.linspace(0, 1, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        bin_centers = []
        p95_vals = []
        p50_vals = []
        n_per_bin = []
        for i in range(len(bin_edges) - 1):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            m = (tx >= lo) & (tx <= hi) if i == len(bin_edges) - 2 else (tx >= lo) & (tx < hi)
            if m.sum() == 0:
                continue
            bin_centers.append((lo + hi) / 2)
            p95_vals.append(np.percentile(rel[m], 95))
            p50_vals.append(np.percentile(rel[m], 50))
            n_per_bin.append(int(m.sum()))
        bin_centers = np.array(bin_centers)
        p95_vals = np.array(p95_vals)
        p50_vals = np.array(p50_vals)
        axes[2, 1].plot(bin_centers, p95_vals, 'r-o', linewidth=2, markersize=4,
                        label='P95 (95% guarantee)')
        axes[2, 1].plot(bin_centers, p50_vals, 'b--s', linewidth=1.5, markersize=3,
                        label='P50 (median)')
        axes[2, 1].fill_between(bin_centers, p50_vals, p95_vals, alpha=0.15, color='red')
        axes[2, 1].set_xlabel('Target Displacement (mm)')
        axes[2, 1].set_ylabel('Relative Error (%)')
        axes[2, 1].set_title('Relative Error Quantiles vs Target (95% guarantee)')
        axes[2, 1].set_yscale('log')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

        # (2,2) 95% 保证率数值表 (文本): 各 target 分箱的 P95 相对误差
        axes[2, 2].axis('off')
        lines = ['95% 保证率 (相对误差):', '']
        for c, p95, p50, n in zip(bin_centers, p95_vals, p50_vals, n_per_bin):
            lines.append(f'  |t|~{c:5.1f}mm: P95={p95:6.1f}%  P50={p50:6.1f}%  (n={n})')
        lines.append('')
        lines.append('含义: 该位移区间内, 95% 的预测')
        lines.append('相对误差不超过 P95 值')
        axes[2, 2].text(0.02, 0.98, '\n'.join(lines), transform=axes[2, 2].transAxes,
                        fontsize=8, va='top', family='monospace')

        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, 'error_analysis')
        save_sci_figure(fig, save_path)
        print(f"✓ 误差分析图: {save_path}.pdf/.png")
        plt.close()
    
    def plot_relative_error(self, results):
        """绘制相对误差百分比分布图"""
        peak_rel = results.get('peak_rel_all', None)
        sample_rel_rmse = results.get('sample_rel_rmse', None)
        if peak_rel is None or sample_rel_rmse is None or len(peak_rel) == 0:
            print("  ⚠️ 无相对误差数据，跳过相对误差图")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 峰值相对误差分布
        axes[0, 0].hist(peak_rel, bins=40, edgecolor='black', alpha=0.7, 
                       color='#B85C4A')
        axes[0, 0].axvline(x=results['peak_rel_mae'], color='red', linestyle='--', 
                          linewidth=1.5, label=f'MAE: {results["peak_rel_mae"]:.2f}%')
        axes[0, 0].axvline(x=results['peak_rel_median'], color='blue', linestyle='--', 
                          linewidth=1.5, label=f'Median: {results["peak_rel_median"]:.2f}%')
        axes[0, 0].set_xlabel('Peak Relative Error (%)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title(f'Peak Relative Error Distribution\n'
                             f'(MAE={results["peak_rel_mae"]:.2f}%, '
                             f'P90={results["peak_rel_p90"]:.2f}%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 样本相对 RMSE 分布
        axes[0, 1].hist(sample_rel_rmse, bins=40, edgecolor='black', alpha=0.7, 
                       color='#6B8E6B')
        axes[0, 1].axvline(x=results['rel_rmse_mean'], color='red', linestyle='--', 
                          linewidth=1.5, label=f'Mean: {results["rel_rmse_mean"]:.2f}%')
        axes[0, 1].axvline(x=results['rel_rmse_median'], color='blue', linestyle='--', 
                          linewidth=1.5, label=f'Median: {results["rel_rmse_median"]:.2f}%')
        axes[0, 1].set_xlabel('Sample Relative RMSE (%)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Sample Relative RMSE Distribution\n'
                             f'(Mean={results["rel_rmse_mean"]:.2f}%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 峰值相对误差 vs 峰值目标位移
        peak_targets = np.max(np.abs(results['targets']), axis=1)
        axes[1, 0].scatter(peak_targets, peak_rel, alpha=0.5, s=15, c='#4A7DB4')
        axes[1, 0].set_xlabel('Peak Target Displacement (mm)')
        axes[1, 0].set_ylabel('Peak Relative Error (%)')
        axes[1, 0].set_title('Peak Relative Error vs Peak Target')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 累积分布 (CDF)
        sorted_rel = np.sort(peak_rel)
        cumulative = np.arange(1, len(sorted_rel) + 1) / len(sorted_rel)
        axes[1, 1].plot(sorted_rel, cumulative, linewidth=2, c='#4A7DB4')
        axes[1, 1].axhline(y=0.9, color='gray', linestyle='--', linewidth=1, 
                          label='P90')
        axes[1, 1].axvline(x=results['peak_rel_p90'], color='red', linestyle='--', 
                          linewidth=1.5, label=f'P90: {results["peak_rel_p90"]:.2f}%')
        axes[1, 1].set_xlabel('Peak Relative Error (%)')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Relative Error Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, 'relative_error')
        save_sci_figure(fig, save_path)
        print(f"✓ 相对误差分布图: {save_path}.pdf/.png")
        plt.close()
    
    def _add_peak_bias_bands(self, ax, peak_targets, peak_preds):
        """在 peak_scatter 图上绘制 y=x 两侧 ±5% / ±10% 偏差区间填充."""
        lo = min(peak_targets.min(), peak_preds.min())
        hi = max(peak_targets.max(), peak_preds.max())
        x = np.linspace(lo, hi, 100)
        # 填充 ±10% 区间 (最外层, 浅红)
        ax.fill_between(x, x * 1.10, x * 0.90, color='red', alpha=0.06,
                        label='±10% band', zorder=1)
        # 填充 ±5% 区间 (内层, 浅绿)
        ax.fill_between(x, x * 1.05, x * 0.95, color='green', alpha=0.08,
                        label='±5% band', zorder=1)

    def plot_peak_scatter(self, results):
        """绘制峰值散点图 (含 ±5% / ±10% 偏差区间填充 + 按高度分组多图)."""
        heights = results['heights']
        peak_preds = np.max(np.abs(results['predictions']), axis=1)
        peak_targets = np.max(np.abs(results['targets']), axis=1)

        # ---------- 主图: 全部样本, 带偏差区间 ----------
        fig, ax = plt.subplots(figsize=(8, 7))
        self._add_peak_bias_bands(ax, peak_targets, peak_preds)
        scatter = ax.scatter(peak_targets, peak_preds, c=heights, cmap='viridis',
                             alpha=0.7, s=25, edgecolors='none', zorder=3)
        lims = [min(peak_targets.min(), peak_preds.min()),
                max(peak_targets.max(), peak_preds.max())]
        ax.plot(lims, lims, 'k--', linewidth=1.5, label='y = x', zorder=4)
        ax.set_xlabel('Target Peak Displacement (mm)')
        ax.set_ylabel('Predicted Peak Displacement (mm)')
        ax.set_title(f'Peak Displacement Prediction (R²={results["peak_r2"]:.4f})')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Building Height (m)')
        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, 'peak_scatter')
        save_sci_figure(fig, save_path)
        print(f"✓ 峰值散点图: {save_path}.pdf/.png")
        plt.close()

        # ---------- 按楼层高度分组的多个 peak_scatter ----------
        # 高度分箱 (用实际高度范围自适应): 每组样本数尽量均衡
        self._plot_peak_scatter_by_height(results)

    def _plot_peak_scatter_by_height(self, results, n_groups=6):
        """按楼层高度分组, 每组绘制一个 peak_scatter (含偏差区间)."""
        heights = np.asarray(results['heights']).flatten()
        peak_preds = np.max(np.abs(results['predictions']), axis=1)
        peak_targets = np.max(np.abs(results['targets']), axis=1)

        if len(heights) < 10:
            print("  ⚠️ 样本太少, 跳过按高度分组峰值散点图")
            return

        # 按高度分位数分箱 (每箱样本数接近, 标签用实际高度范围)
        quantiles = np.percentile(heights, np.linspace(0, 100, n_groups + 1))
        quantiles = np.unique(quantiles)
        labels = []
        group_idx = np.zeros(len(heights), dtype=int)
        for gi in range(len(quantiles) - 1):
            lo, hi = quantiles[gi], quantiles[gi + 1]
            if gi == len(quantiles) - 2:
                m = (heights >= lo) & (heights <= hi)
            else:
                m = (heights >= lo) & (heights < hi)
            group_idx[m] = gi
            labels.append(f'{lo:.0f}-{hi:.0f}m')
        n_groups_actual = len(labels)

        # 每样本 R² 标注 (该组内)
        ncols = 3
        nrows = int(np.ceil(n_groups_actual / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                                 squeeze=False)
        for gi in range(n_groups_actual):
            ax = axes[gi // ncols, gi % ncols]
            m = group_idx == gi
            if m.sum() < 2:
                ax.set_title(f'{labels[gi]} (n={m.sum()})', fontsize=10)
                ax.axis('off')
                continue
            pt, pp = peak_targets[m], peak_preds[m]
            # 组内 R²
            ss_res = np.sum((pt - pp) ** 2)
            ss_tot = np.sum((pt - pt.mean()) ** 2)
            r2_g = float(1.0 - ss_res / (ss_tot + 1e-12))
            self._add_peak_bias_bands(ax, pt, pp)
            ax.scatter(pt, pp, alpha=0.7, s=20, c='#4A7DB4',
                       edgecolors='none', zorder=3)
            lo = min(pt.min(), pp.min()); hi = max(pt.max(), pp.max())
            ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.2, zorder=4)
            ax.set_title(f'H {labels[gi]}  (n={m.sum()}, R²={r2_g:.3f})',
                         fontsize=10)
            ax.set_xlabel('Target Peak (mm)', fontsize=9)
            ax.set_ylabel('Predicted Peak (mm)', fontsize=9)
            ax.grid(True, alpha=0.3)
        # 隐藏多余子图
        for gi in range(n_groups_actual, nrows * ncols):
            axes[gi // ncols, gi % ncols].axis('off')
        fig.suptitle('Peak Scatter by Building Height Group',
                     fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(self.plot_dir, 'peak_scatter_by_height')
        save_sci_figure(fig, save_path)
        print(f"✓ 按高度分组峰值散点图: {save_path}.pdf/.png")
        plt.close()
    
    def plot_height_analysis(self, results):
        """绘制按高度分组的性能分析"""
        groups = results['height_groups']
        
        if not groups:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        
        names = list(groups.keys())
        r2s = [groups[n]['r2'] for n in names]
        rmses = [groups[n]['rmse'] for n in names]
        ns = [groups[n]['n'] for n in names]
        
        axes[0].bar(names, r2s, color='#4A7DB4', edgecolor='black', alpha=0.8)
        axes[0].axhline(y=0.8, color='red', linestyle='--', linewidth=1.5, label='R²=0.8')
        axes[0].set_xlabel('Height Range')
        axes[0].set_ylabel('R²')
        axes[0].set_title('R² by Height Group')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        axes[1].bar(names, rmses, color='#6B8E6B', edgecolor='black', alpha=0.8)
        axes[1].set_xlabel('Height Range')
        axes[1].set_ylabel('RMSE (mm)')
        axes[1].set_title('RMSE by Height Group')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        axes[2].bar(names, ns, color='#B85C4A', edgecolor='black', alpha=0.8)
        axes[2].set_xlabel('Height Range')
        axes[2].set_ylabel('Sample Count')
        axes[2].set_title('Sample Count by Height Group')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, 'height_analysis')
        save_sci_figure(fig, save_path)
        print(f"✓ 高度分析图: {save_path}.pdf/.png")
        plt.close()
    
    def plot_sample_comparison(self, results, num_samples=6):
        """绘制多个样本的时程对比"""
        if len(self.sample_data) == 0:
            print("  无样本数据，跳过时程对比图")
            return
        
        n_samples = min(num_samples, len(self.sample_data))
        indices = np.random.choice(len(self.sample_data), n_samples, replace=False)
        
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3.5 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        dt = self.config.TARGET_DT
        T = self.sample_data[0]['target_disp'].shape[0]
        t = np.arange(T) * dt
        
        for i, idx in enumerate(indices):
            data = self.sample_data[idx]
            
            # 加速度 (g → m/s², ×9.81)
            acc = data['acceleration'] * 9.81
            if len(acc) > len(t):
                acc = acc[:len(t)]
            elif len(acc) < len(t):
                acc = np.pad(acc, (0, len(t) - len(acc)), 'constant')
            
            axes[i, 0].plot(t, acc, 'b-', linewidth=1.5)
            axes[i, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 0].set_ylabel('Acceleration (m/s²)')
            axes[i, 0].set_title(f'Sample {i+1}: Input Acceleration (H={data["height"]:.1f}m)')
            axes[i, 0].grid(True, alpha=0.3)
            
            peak_idx = np.argmax(np.abs(acc))
            axes[i, 0].scatter(t[peak_idx], acc[peak_idx], color='red', s=30, zorder=5)
            
            # 位移对比
            target = data['target_disp']
            pred = data['pred_disp']
            if len(target) > len(t):
                target = target[:len(t)]
                pred = pred[:len(t)]
            elif len(target) < len(t):
                target = np.pad(target, (0, len(t) - len(target)), 'constant')
                pred = np.pad(pred, (0, len(t) - len(pred)), 'constant')
            
            axes[i, 1].plot(t, target, 'b-', linewidth=1.8, label='Target')
            axes[i, 1].plot(t, pred, 'r--', linewidth=1.8, label='Predicted')
            axes[i, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
            axes[i, 1].set_xlabel('Time (s)')
            axes[i, 1].set_ylabel('Displacement (mm)')
            axes[i, 1].set_title(f'Sample {i+1}: Top Displacement')
            axes[i, 1].legend(loc='upper right')
            axes[i, 1].grid(True, alpha=0.3)
            
            # 误差
            error = pred - target
            if len(error) > len(t):
                error = error[:len(t)]
            axes[i, 2].plot(t, error, 'g-', linewidth=1.5)
            axes[i, 2].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
            axes[i, 2].fill_between(t, 0, error, alpha=0.2, color='green')
            axes[i, 2].set_xlabel('Time (s)')
            axes[i, 2].set_ylabel('Error (mm)')
            rmse_val = np.sqrt(np.mean(error**2))
            axes[i, 2].set_title(f'Sample {i+1}: Error (RMSE={rmse_val:.3f}mm)')
            axes[i, 2].grid(True, alpha=0.3)
            
            mae_val = np.mean(np.abs(error))
            axes[i, 2].text(0.02, 0.95, f'RMSE={rmse_val:.3f}mm, MAE={mae_val:.3f}mm',
                           transform=axes[i, 2].transAxes, fontsize=9,
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, 'sample_comparison')
        save_sci_figure(fig, save_path)
        print(f"✓ 样本时程对比图: {save_path}.pdf/.png")
        plt.close()

    def plot_single_sample_detail(self, sample_idx=0):
        """绘制单个样本的详细图表"""
        if len(self.sample_data) == 0:
            print("无样本数据")
            return
        
        idx = min(sample_idx, len(self.sample_data) - 1)
        data = self.sample_data[idx]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        dt = self.config.TARGET_DT
        T = data['target_disp'].shape[0]
        t = np.arange(T) * dt
        
        # 加速度 (g → m/s², ×9.81)
        acc = data['acceleration'] * 9.81
        if len(acc) > len(t):
            acc = acc[:len(t)]
        axes[0].plot(t, acc, 'b-', linewidth=1.5)
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Acceleration (m/s²)')
        axes[0].set_title('Input Acceleration')
        axes[0].grid(True, alpha=0.3)
        
        # 位移对比
        target = data['target_disp']
        pred = data['pred_disp']
        if len(target) > len(t):
            target = target[:len(t)]
            pred = pred[:len(t)]
        axes[1].plot(t, target, 'b-', linewidth=2, label='Target')
        axes[1].plot(t, pred, 'r--', linewidth=2, label='Predicted')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Displacement (mm)')
        axes[1].set_title(f'Top Displacement (H={data["height"]:.1f}m)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 误差
        error = pred - target
        if len(error) > len(t):
            error = error[:len(t)]
        axes[2].plot(t, error, 'g-', linewidth=1.5)
        axes[2].axhline(y=0, color='black', linestyle='--')
        axes[2].fill_between(t, 0, error, alpha=0.2, color='green')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Error (mm)')
        rmse_val = np.sqrt(np.mean(error**2))
        axes[2].set_title(f'Prediction Error (RMSE={rmse_val:.3f}mm)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, f'sample_detail_{idx}')
        save_sci_figure(fig, save_path)
        print(f"✓ 样本细节图: {save_path}.pdf/.png")
        plt.close()


# ============================================================
# 主测试函数 - 使用训练时保存的数据集划分
# ============================================================

def evaluate_model(model_path=None, use_db=False, output_dir=None, num_eval=None,
                   max_samples=None, enc_mode=None):
    """运行完整评估 - 使用训练时保存的验证集 (与 train 一致)
python evaluate.py --model models_voxel_token/model/best_model.pth --use_db --max_samples 30000 --out plots/eval_token
    Args:
        model_path: 模型权重路径 (默认 ./models/model/best_model.pth)
        use_db: 从 PostgreSQL 读取数据 (与 train --use_db 一致)
        output_dir: 图表输出目录 (默认 ./plots/eval_sci)
        num_eval: 评估样本数上限 (None=全部验证集)
        max_samples: 数据库模式样本上限 (须与训练时一致, 保证同一批验证集)
        enc_mode: 切杆系编码模式 'token'/'direct'/'cont'
                  (None=自动从 checkpoint config 的 USE_VOXEL_TOKEN 推断)
    """
    print("=" * 60)
    print("模型评估 (与 train 一致的输入) - SCI 配图")
    print("=" * 60)

    cfg = Config()
    if max_samples:
        cfg.DB_MAX_SAMPLES = max_samples
    cfg.print_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # ============================================================
    # 加载数据集 (与 train 使用完全相同的 build_loaders, 支持 DB)
    # ============================================================
    print("\n加载数据集...")
    from train import build_loaders
    # 若未显式指定 enc_mode, 从 checkpoint 的 config 自动推断
    use_voxel_feature = False
    voxel_enc_mode = None
    if enc_mode is not None:
        voxel_enc_mode = enc_mode
        use_voxel_feature = True
    elif model_path and os.path.exists(model_path):
        try:
            ck = torch.load(model_path, map_location='cpu')
            scfg = ck.get('config') or {}
            # checkpoint 的 vars(cfg) 只含实例属性 (类属性如 USE_VOXEL_TOKEN 不在其中),
            # 但 VOXEL_VOCAB_SIZE / FRAME_FEATURE_DIM 会被显式 setattr 保存。
            # 判定规则:
            #   - 有 VOXEL_VOCAB_SIZE 且 > 1  -> token 模式 (train_voxel 设 cfg.VOXEL_VOCAB_SIZE=n_tok)
            #   - FRAME_FEATURE_DIM == VOXEL_GRID³×6 -> cont 模式
            #   - FRAME_FEATURE_DIM == VOXEL_GRID³ 但无 vocab -> direct 模式
            _vg = int(getattr(cfg, 'VOXEL_GRID', 64))
            if scfg.get('VOXEL_VOCAB_SIZE'):
                voxel_enc_mode = 'token'
                use_voxel_feature = True
            else:
                fd = scfg.get('FRAME_FEATURE_DIM')
                if fd == _vg ** 3 * 6:
                    voxel_enc_mode = 'cont'
                    use_voxel_feature = True
                elif fd == _vg ** 3:
                    voxel_enc_mode = 'direct'
                    use_voxel_feature = True
        except Exception:
            pass
    if use_voxel_feature:
        print(f"  🧱 使用切杆系编码特征模式: enc_mode={voxel_enc_mode}")
    tr_loader, va_loader, dataset = build_loaders(
        cfg, use_db=use_db, use_voxel_feature=use_voxel_feature,
        voxel_enc_mode=voxel_enc_mode)
    if va_loader is None:
        print("  ❌ 无法加载数据")
        return None

    val_dataset = va_loader.dataset
    n_val = len(val_dataset)
    print(f"  ✓ 验证集: {n_val} 样本")

    # 限制评估样本数 (加速 SCI 出图); 不限制时直接用 va_loader
    if num_eval and num_eval < n_val:
        from torch.utils.data import Subset
        indices = np.linspace(0, n_val - 1, num_eval).astype(int)
        indices = np.unique(indices)
        val_dataset = Subset(val_dataset, indices)
        print(f"  限制评估样本: {len(val_dataset)}")
        # 重建 loader
        batch = min(16, len(val_dataset))
        val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False,
                                num_workers=0, pin_memory=True)
    else:
        val_loader = va_loader

    # ============================================================
    # 加载模型 (与 train 相同的架构/路径)
    # ============================================================
    use_v2 = bool(getattr(cfg, 'USE_V2', False))

    # checkpoint 保存的 model_kwargs (含 use_sa), 用于重建相同架构
    saved_mk = None

    def _build_model(mcfg, v2, bypass, mk_extra=None):
        mk = dict(use_v2=v2, use_bypass=bypass, drop_path=0.0,  # 推理关闭随机深度
                  film=getattr(mcfg, 'V2_FILM', True))
        # 默认关闭自注意力 (no_sa); checkpoint 保存的 model_kwargs 优先
        mk['use_sa'] = False
        if mk_extra and 'use_sa' in mk_extra:
            mk['use_sa'] = bool(mk_extra['use_sa'])
        # 用已推断的 voxel_enc_mode 判定 (优先), 回退到 config 类默认
        is_token = (voxel_enc_mode == 'token' or
                    (voxel_enc_mode is None and
                     getattr(mcfg, 'VOXEL_VOCAB_SIZE', 0) and
                     getattr(mcfg, 'FRAME_FEATURE_DIM', 0) ==
                     int(getattr(cfg, 'VOXEL_GRID', 64)) ** 3))
        if is_token:
            mk['use_voxel_token'] = True
            mk['vocab_size'] = int(getattr(mcfg, 'VOXEL_VOCAB_SIZE', 300))
        return SLFormer(mcfg, **mk)

    if model_path is None:
        # 默认优先新目录 models, 回退旧目录 models_causal (兼容旧模型)
        default_path = os.path.join('models', 'model', 'best_model.pth')
        if not os.path.exists(default_path):
            default_path = os.path.join('models_causal', 'model', 'best_model.pth')
        model_path = default_path

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        saved_mk = checkpoint.get('model_kwargs')
        if 'model_state_dict' in checkpoint:
            sd = checkpoint['model_state_dict']
        else:
            sd = checkpoint

        # 优先用 checkpoint 保存的 config 重建模型 (确保 N_HEAD/D_FF/drop_path 与训练一致)
        saved_cfg = checkpoint.get('config')
        model_cfg = cfg
        if saved_cfg is not None and isinstance(saved_cfg, dict):
            model_cfg = Config()
            for k, v in saved_cfg.items():
                if isinstance(v, (int, float, str, bool, list, tuple, type(None))):
                    setattr(model_cfg, k, v)
            use_v2 = bool(getattr(model_cfg, 'USE_V2', True))
            print(f"  🔧 使用 checkpoint 保存的 config 重建模型 "
                  f"(N_HEAD={getattr(model_cfg, 'N_HEAD', '?')}, "
                  f"D_FF={getattr(model_cfg, 'D_FF', '?')}, "
                  f"drop_path={getattr(model_cfg, 'V2_DROP_PATH', 0.0):.4f})")
            if voxel_enc_mode == 'token' or \
               (voxel_enc_mode is None and
                getattr(model_cfg, 'VOXEL_VOCAB_SIZE', 0) and
                getattr(model_cfg, 'FRAME_FEATURE_DIM', 0) ==
                int(getattr(cfg, 'VOXEL_GRID', 64)) ** 3):
                print(f"  🧱 微元 Token 模型 (vocab_size={getattr(model_cfg, 'VOXEL_VOCAB_SIZE', '?')})")

        # 尝试用模型 config 加载 (v1/v2 自动回退)
        loaded = False
        for try_v2 in [use_v2, not use_v2]:
            try:
                model = _build_model(model_cfg, try_v2, try_v2, mk_extra=saved_mk).to(device)
                model.load_state_dict(sd)
                if try_v2 != use_v2:
                    print(f"  ⚠️ 权重架构不匹配, 已切换 use_v2={try_v2} 加载")
                loaded = True
                break
            except Exception:
                continue
        if not loaded:
            model = _build_model(model_cfg, use_v2, use_v2, mk_extra=saved_mk).to(device)
            print("  ⚠️ 权重无法匹配, 使用随机初始化模型")
        print(f"✓ 加载模型: {model_path}")
        if 'epoch' in checkpoint:
            print(f"  训练轮次: {checkpoint['epoch']+1}")
        if 'best_val_loss' in checkpoint:
            print(f"  最佳验证损失: {checkpoint['best_val_loss']:.6f}")
    else:
        print(f"⚠️ 模型文件不存在: {model_path}")
        print("  将使用随机初始化的模型进行评估（仅供参考）")
        model = _build_model(cfg, use_v2, use_v2).to(device)

    # ============================================================
    # 评估 (SCI 图表)
    # ============================================================
    plot_dir = output_dir or os.path.join('plots', 'eval_sci')
    evaluator = Evaluator(model, cfg, device, plot_dir=plot_dir)
    results = evaluator.evaluate(val_loader, num_samples=min(20, len(val_dataset)))

    # 保存评估结果
    if results:
        results_path = os.path.join(plot_dir, 'evaluation_results.pkl')
        with open(results_path, 'wb') as f:
            results_serializable = {
                'r2': results['r2'],
                'rmse': results['rmse'],
                'mae': results['mae'],
                'corr': results['corr'],
                'peak_mae': results['peak_mae'],
                'peak_rmse': results['peak_rmse'],
                'peak_r2': results['peak_r2'],
                'height_groups': results['height_groups'],
                'num_samples': results['num_samples']
            }
            pickle.dump(results_serializable, f)
        print(f"✓ 评估结果已保存: {results_path}")

    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型评估 (与 train 一致, SCI 配图)')
    parser.add_argument('--model', type=str, default=None,
                        help='模型权重路径 (默认 ./models/model/best_model.pth)')
    parser.add_argument('--use_db', action='store_true',
                        help='从 PostgreSQL 读取数据 (与 train --use_db 一致)')
    parser.add_argument('--out', type=str, default=None,
                        help='图表输出目录 (默认 ./plots/eval_sci)')
    parser.add_argument('--num', type=int, default=None,
                        help='评估样本数上限')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='数据库模式: 最多取多少样本 (须与训练时一致, 保证同一验证集)')
    parser.add_argument('--enc', type=str, default=None, choices=['token', 'direct', 'cont'],
                        help='切杆系编码模式 (默认自动从 checkpoint config 推断)')
    args = parser.parse_args()
    evaluate_model(model_path=args.model, use_db=args.use_db,
                   output_dir=args.out, num_eval=args.num,
                   max_samples=args.max_samples, enc_mode=args.enc)