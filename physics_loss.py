# physics_loss.py
import torch
import torch.nn.functional as F

def physical_regularization_loss(pred_disp, height, E_avg, config, lambda_reg=None, delta=None):
    """
    基于经验周期的频谱正则化
    
    参数:
        pred_disp: [batch, T] 预测位移时程
        height: [batch, 1] 建筑总高 (米)
        E_avg: [batch, 1] 平均弹性模量 (GPa)
        config: Config对象
        lambda_reg: 正则化权重 (默认使用config中的值)
        delta: 周期容限 (秒)
    """
    if lambda_reg is None:
        lambda_reg = config.PHYSICS_LAMBDA
    if delta is None:
        delta = config.PHYSICS_DELTA
    
    batch_size, T = pred_disp.shape
    device = pred_disp.device
    
    # 1. 估算基本周期 T_est = alpha * H^0.75 / sqrt(E_avg)
    # 规范经验系数 (框架结构)
    alpha = 0.085
    gamma = 0.75
    
    # 防止除零
    E_avg_clamped = torch.clamp(E_avg, min=1.0)  # 至少1 GPa
    T_est = alpha * (height ** gamma) / torch.sqrt(E_avg_clamped)  # [batch, 1]
    T_est = torch.clamp(T_est, min=0.1, max=5.0)  # 物理合理范围
    
    # 2. 对预测位移做FFT，找峰值频率
    # 对每个样本独立处理
    T_pred_list = []
    for i in range(batch_size):
        # 去除直流分量
        signal = pred_disp[i] - pred_disp[i].mean()
        # 加窗减少频谱泄漏
        window = torch.hann_window(T, device=device)
        signal_windowed = signal * window
        
        # 实数FFT (只取正频率)
        fft_vals = torch.fft.rfft(signal_windowed)
        magnitudes = torch.abs(fft_vals)
        
        # 忽略直流分量 (索引0)
        magnitudes[0] = 0
        
        # 找到峰值频率索引
        peak_idx = torch.argmax(magnitudes)
        
        # 转换为频率 (Hz)
        # 假设采样频率 fs = T / 总时长，总时长设为10秒
        total_duration = 10.0  # 秒
        fs = T / total_duration
        f_peak = peak_idx.float() * (fs / T)
        
        # 转换为周期
        T_pred_i = 1.0 / (f_peak + 1e-6)
        T_pred_list.append(T_pred_i)
    
    T_pred = torch.stack(T_pred_list).unsqueeze(1)  # [batch, 1]
    
    # 3. 计算惩罚项 (仅当偏差超过delta才惩罚)
    deviation = torch.abs(T_pred - T_est) - delta
    penalty = F.relu(deviation)  # max(0, deviation)
    loss_reg = lambda_reg * penalty.mean()
    
    return loss_reg, T_est, T_pred