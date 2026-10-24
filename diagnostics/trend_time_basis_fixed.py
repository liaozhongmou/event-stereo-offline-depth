#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
纯时间维度自适应视差平滑

为每个像素位置独立维护时间序列趋势拟合器
仅在时间维度上做趋势预测和平滑更新，不做任何空间平滑
"""

import argparse
import pathlib
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class PixelTrendTracker:
    """为单个像素跟踪视差的时间趋势"""
    
    def __init__(self, window_size=20, deviation_threshold=2.0, 
                 outlier_tolerance=3, trend_degree=1, min_samples=5):
        self.window_size = window_size
        self.deviation_threshold = deviation_threshold
        self.outlier_tolerance = outlier_tolerance
        self.trend_degree = trend_degree
        self.min_samples = min_samples
        
        # 时间序列数据
        self.timestamps = []
        self.values = []
        
        # 拟合状态
        self.coefficients = None
        self.fit_time_origin = None
        self.fit_time_scale = None
        self.mean_deviation = None
        self.std_deviation = 1.0
        
        # 异常计数
        self.consecutive_outliers = 0
        
        # 统计
        self.total_points = 0
        self.refit_count = 0
        self.correction_count = 0
        
    def add_value(self, timestamp, value):
        """
        添加新观测值，返回拟合后的值和状态信息
        
        返回: (fitted_value, status_dict)
        """
        self.total_points += 1
        
        # 添加到窗口
        self.timestamps.append(timestamp)
        self.values.append(value)
        
        # 保持窗口大小
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
            self.values.pop(0)
        
        # 样本不足，直接返回原值
        if len(self.values) < self.min_samples:
            return value, {
                'status': 'insufficient_data',
                'fitted': value,
                'predicted': None,
                'deviation': 0,
                'is_outlier': False
            }
        
        # 初次拟合
        if self.coefficients is None:
            self._fit_trend()
            return value, {
                'status': 'initial_fit',
                'fitted': value,
                'predicted': None,
                'deviation': 0,
                'is_outlier': False
            }
        
        # 预测当前值
        predicted_value = self._predict(timestamp)
        deviation = abs(value - predicted_value)
        
        # 判断是否为异常点
        threshold = self.deviation_threshold * self.std_deviation
        is_outlier = deviation > threshold
        
        if is_outlier:
            self.consecutive_outliers += 1
            
            # 连续异常点过多，认为趋势已变化，重新拟合
            if self.consecutive_outliers >= self.outlier_tolerance:
                self._fit_trend()
                self.consecutive_outliers = 0
                self.refit_count += 1
                fitted_value = self._predict(timestamp)
                status = 'refitted'
            else:
                # 使用预测值替代异常值
                fitted_value = predicted_value
                status = 'outlier_corrected'
                self.correction_count += 1
        else:
            # 正常点，重置异常计数
            self.consecutive_outliers = 0
            # 使用实际值，并更新拟合
            fitted_value = value
            self._update_fit()
            status = 'normal'
        
        return fitted_value, {
            'status': status,
            'fitted': fitted_value,
            'predicted': predicted_value,
            'deviation': deviation,
            'threshold': threshold,
            'is_outlier': is_outlier
        }
    
    def _fit_trend(self):
        """拟合趋势线"""
        if len(self.values) < self.min_samples:
            return
        
        t = np.array(self.timestamps)
        y = np.array(self.values)
        
        # 归一化时间戳（提高数值稳定性）
        t_normalized = (t - t[0]) / (t[-1] - t[0] + 1e-10)
        
        try:
            # 多项式拟合
            degree = min(self.trend_degree, len(t) - 1)
            self.coefficients = np.polyfit(t_normalized, y, degree)
            self.fit_time_origin = t[0]
            self.fit_time_scale = t[-1] - t[0] + 1e-10
            
            # 计算拟合误差统计
            predictions = np.polyval(self.coefficients, t_normalized)
            residuals = y - predictions
            
            self.mean_deviation = np.mean(np.abs(residuals))
            self.std_deviation = np.std(residuals) + 0.01  # 避免除零
        except:
            # 拟合失败，保持原状态
            pass
    
    def _update_fit(self):
        """增量更新拟合（使用所有窗口内数据重新拟合）"""
        self._fit_trend()
    
    def _predict(self, timestamp):
        """预测指定时间戳的值"""
        if self.coefficients is None or len(self.timestamps) == 0:
            return self.values[-1] if self.values else 0.0
        
        # 使用相同的归一化方式
        t0 = self.fit_time_origin
        t_range = self.fit_time_scale
        t_normalized = (timestamp - t0) / t_range
        
        return float(np.polyval(self.coefficients, t_normalized))
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'total_points': self.total_points,
            'refit_count': self.refit_count,
            'correction_count': self.correction_count,
            'window_size': len(self.values),
            'mean_deviation': self.mean_deviation,
            'std_deviation': self.std_deviation,
            'consecutive_outliers': self.consecutive_outliers
        }


def process_temporal_adaptive(x, y, t, disp, args):
    """
    纯时间维度自适应处理
    
    流程：
    1. 按时间排序所有事件
    2. 为每个像素位置 (x,y) 创建独立的趋势跟踪器
    3. 顺序处理每个事件，用跟踪器获取平滑后的视差值
    """
    
    # 按时间排序
    print("[info] 按时间排序事件...", flush=True)
    order = np.argsort(t)
    x_sorted = x[order]
    y_sorted = y[order]
    t_sorted = t[order]
    disp_sorted = disp[order]
    
    # 为每个像素位置创建跟踪器
    print("[info] 初始化像素跟踪器...", flush=True)
    trackers = defaultdict(lambda: PixelTrendTracker(
        window_size=args.window_size,
        deviation_threshold=args.deviation_threshold,
        outlier_tolerance=args.outlier_tolerance,
        trend_degree=args.trend_degree,
        min_samples=args.min_samples
    ))
    
    disp_out = np.zeros_like(disp_sorted)
    
    # 统计信息
    stats = {
        'insufficient_data': 0,
        'initial_fit': 0,
        'normal': 0,
        'outlier_corrected': 0,
        'refitted': 0
    }
    
    print(f"[info] 开始处理 {len(t_sorted)} 个事件...", flush=True)
    print(f"       时间范围: [{t_sorted[0]:.6f}, {t_sorted[-1]:.6f}]", flush=True)
    
    # 逐事件处理
    report_interval = max(1, len(t_sorted) // 20)  # 报告20次进度
    
    for i in range(len(t_sorted)):
        xi = x_sorted[i]
        yi = y_sorted[i]
        ti = t_sorted[i]
        di = disp_sorted[i]
        
        # 只处理有效视差
        if di <= 0:
            disp_out[i] = di
            continue
        
        # 使用像素的趋势跟踪器
        pixel_key = (int(xi), int(yi))
        fitted_val, info = trackers[pixel_key].add_value(ti, di)
        
        disp_out[i] = fitted_val
        stats[info['status']] += 1
        
        # 进度报告
        if (i + 1) % report_interval == 0:
            progress = (i + 1) / len(t_sorted) * 100
            print(f"  进度: {progress:.1f}% ({i+1}/{len(t_sorted)}) | "
                  f"活跃跟踪器: {len(trackers)}", flush=True)
    
    # 恢复原始顺序
    disp_final = np.zeros_like(disp)
    disp_final[order] = disp_out
    
    # 打印统计信息
    print("\n" + "="*60)
    print("处理完成统计:")
    print("="*60)
    print(f"总事件数:         {len(t_sorted)}")
    print(f"活跃像素数:       {len(trackers)}")
    print(f"\n状态分布:")
    print(f"  样本不足:       {stats['insufficient_data']:8d} ({stats['insufficient_data']/len(t_sorted)*100:5.2f}%)")
    print(f"  初次拟合:       {stats['initial_fit']:8d} ({stats['initial_fit']/len(t_sorted)*100:5.2f}%)")
    print(f"  正常更新:       {stats['normal']:8d} ({stats['normal']/len(t_sorted)*100:5.2f}%)")
    print(f"  异常校正:       {stats['outlier_corrected']:8d} ({stats['outlier_corrected']/len(t_sorted)*100:5.2f}%)")
    print(f"  趋势重拟:       {stats['refitted']:8d} ({stats['refitted']/len(t_sorted)*100:5.2f}%)")
    
    # 计算平均修正量
    corrections = np.abs(disp_final - disp)
    print(f"\n修正统计:")
    print(f"  平均修正量:     {np.mean(corrections):.4f}")
    print(f"  最大修正量:     {np.max(corrections):.4f}")
    print(f"  修正中位数:     {np.median(corrections):.4f}")
    
    # 采样显示部分像素的跟踪器统计
    print(f"\n像素跟踪器采样（前5个活跃像素）:")
    for i, (pixel_key, tracker) in enumerate(list(trackers.items())[:5]):
        tracker_stats = tracker.get_stats()
        print(f"  像素 {pixel_key}: 观测={tracker_stats['total_points']}, "
              f"重拟={tracker_stats['refit_count']}, "
              f"校正={tracker_stats['correction_count']}, "
              f"std={tracker_stats['std_deviation']:.4f}")
    
    print("="*60)
    
    return disp_final, trackers


def main():
    pa = argparse.ArgumentParser(description="纯时间维度自适应视差平滑")
    pa.add_argument("--in_npz", type=pathlib.Path, default="matches10.npz",
                    help="输入npz文件")
    pa.add_argument("--out_npz", type=pathlib.Path, default="matches10_1.npz",
                    help="输出npz文件")
    
    # 自适应趋势跟踪参数
    pa.add_argument("--window_size", type=int, default=20,
                    help="每个像素的时间窗口大小（保留多少历史观测）")
    pa.add_argument("--deviation_threshold", type=float, default=0.1,
                    help="异常检测阈值（标准差的倍数）")
    pa.add_argument("--outlier_tolerance", type=int, default=3,
                    help="容忍的连续异常点数量，超过则重新拟合趋势")
    pa.add_argument("--trend_degree", type=int, default=1,
                    help="趋势拟合的多项式阶数（1=线性，2=二次）")
    pa.add_argument("--min_samples", type=int, default=4,
                    help="最小样本数量才能进行拟合")
    
    args = pa.parse_args()
    
    print("\n" + "="*60)
    print("纯时间维度自适应视差平滑")
    print("="*60)
    print(f"输入文件: {args.in_npz}")
    print(f"输出文件: {args.out_npz}")
    print(f"\n参数配置:")
    print(f"  时间窗口大小:   {args.window_size}")
    print(f"  异常检测阈值:   {args.deviation_threshold} × std")
    print(f"  异常容忍度:     {args.outlier_tolerance}")
    print(f"  趋势拟合阶数:   {args.trend_degree}")
    print(f"  最小样本数:     {args.min_samples}")
    print("="*60 + "\n")
    
    # 加载数据
    print("[1/4] 加载数据...", flush=True)
    data = np.load(args.in_npz)
    
    need = {"x_L", "y_L", "t", "disp"}
    if not need.issubset(data.files):
        raise ValueError(f"缺少必要字段: {need}")
    
    x = data["x_L"].astype(np.int32)
    y = data["y_L"].astype(np.int32)
    t = data["t"].astype(np.float64)
    disp = data["disp"].astype(np.float32)
    
    if x.size == 0:
        print("[warn] 空文件，直接保存", flush=True)
        out_path = pathlib.Path(args.out_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out_path), x_L=x, y_L=y, t=t, disp=disp)
        return
    
    W = int(x.max()) + 1
    H = int(y.max()) + 1
    
    print(f"     图像尺寸: {W} × {H}")
    print(f"     事件总数: {x.size}")
    print(f"     唯一像素: {len(set(zip(x, y)))}")
    
    # 处理
    print(f"\n[2/4] 时间维度自适应处理...", flush=True)
    disp_out, trackers = process_temporal_adaptive(x, y, t, disp, args)
    
    # 保存
    print(f"\n[3/4] 保存结果...", flush=True)
    out_path = pathlib.Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), x_L=x, y_L=y, t=t, disp=disp_out)
    
    print(f"[4/4] 完成！")
    print(f"\n输出文件: {out_path.resolve()}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

#python Adapative_Trend_Fitter.py 