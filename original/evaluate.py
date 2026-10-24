#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVSEC disparity-domain evaluator (Enhanced with comprehensive metrics)
============================================================
* extract_gt  – 读取 indoor_flying*_gt.hdf5 深度 Z(m) → 视差(px) 并保存 gt_disp.npz。
* evaluate    – 将 matches.npz 的 (x_L, y_L, t, disp) 与 GT 视差按时间对齐后，计算
                多种评估指标，并输出逐帧 MAE 曲线（可选仅绘制时间窗口）。

【新增评估指标】
1. MAE (Mean Absolute Error) - 平均绝对误差
2. RMSE (Root Mean Square Error) - 均方根误差
3. Bad-X% - 多个阈值下的错误率 (1.0, 2.0, 3.0 pixels)
4. Median Error - 中位数误差
5. Mean Relative Error (MRE) - 平均相对误差
6. Mean Absolute Relative Error (MARE) - 平均绝对相对误差
7. A1, A2, A3 - 阈值准确度 (δ < 1.25, 1.25², 1.25³)
8. Percentile errors (50th, 75th, 90th, 95th, 99th)
9. Inlier ratio - 不同阈值下的内点率
10. Standard deviation - 标准差

【用法示例】
1) 生成 GT 视差：
   python evaluate_enhanced.py extract_gt \
       --hdf5 indoor_flying1_gt.hdf5 \
       --fx 226.38018519795807 --baseline 0.09988137641750752

2) 评估（指定窗口中心与时长）：
   python evaluate_enhanced.py evaluate --match matches.npz --gt gt_disp.npz \
       --t_center 42.5 --dt_window 6 --t_units s

3) 评估（自定义视差范围和MAE阈值）：
   python evaluate_enhanced.py evaluate --match matches.npz --gt gt_disp.npz \
       --t0 40.0 --until 46.0 --t_units s --disp_min 5.0 --disp_max 30.0 --mae_thresh 10.0
"""
from __future__ import annotations
import argparse, pathlib
import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import Dict


# ---------------------- 提取 GT（深度→视差） ---------------------- #
def extract_gt(hdf5_path: pathlib.Path, out_path: pathlib.Path,
               *, fx: float, baseline: float):
    """读取 GT 深度，转为视差并保存（disp=fx*baseline/depth；无效深度置0）。"""
    with h5py.File(hdf5_path, 'r') as f:
        depth = f['davis/left/depth_image_rect'][:].astype(np.float32)
        ts    = f['davis/left/depth_image_rect_ts'][:].astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        disp = np.where(depth > 0, fx * baseline / depth, 0.0).astype(np.float32)
    np.savez_compressed(out_path, disp=disp, ts=ts)
    print(f"✓ disparity GT saved → {out_path}")


# -------------------------- 工具函数 -------------------------- #
def _resolve_window(ts_gt: np.ndarray,
                    t0: float | None,
                    until: float | None,
                    t_center: float | None,
                    dt_window: float | None) -> tuple[float, float]:
    """解析评估窗口"""
    gt_lo, gt_hi = float(ts_gt[0]), float(ts_gt[-1])

    if (t_center is not None) and (dt_window is not None):
        lo = t_center - 0.5 * dt_window
        hi = t_center + 0.5 * dt_window
    else:
        lo = t0 if t0 is not None else gt_lo
        hi = until if until is not None else gt_hi

    lo = max(lo, gt_lo)
    hi = min(hi, gt_hi)
    if hi <= lo:
        lo, hi = gt_lo, min(gt_hi, gt_lo + 1e-6)
    return lo, hi


def _median_dt(arr: np.ndarray) -> float:
    """返回已排序数组相邻差分的中位数"""
    if arr.size < 2:
        return 0.0
    diffs = np.diff(np.sort(arr))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def _align_only(t_sec: np.ndarray, ts_gt: np.ndarray) -> tuple[np.ndarray, float]:
    """仅估计全局平移，使事件时间对齐到最近的GT帧"""
    t0, ts0 = float(t_sec[0]), float(ts_gt[0])
    t_rel  = t_sec - t0
    ts_rel = ts_gt - ts0

    idx_prev = np.searchsorted(ts_rel, t_rel, side='right') - 1
    idx_prev = np.clip(idx_prev, 0, ts_rel.size - 1)
    idx_next = np.clip(idx_prev + 1, 0, ts_rel.size - 1)

    dt_prev  = np.abs(t_rel - ts_rel[idx_prev])
    dt_next  = np.abs(ts_rel[idx_next] - t_rel)
    nearest  = np.where(dt_next < dt_prev, ts_rel[idx_next], ts_rel[idx_prev])

    delta = float(np.median(nearest - t_rel))
    t_aligned = t_rel + delta + ts0
    return t_aligned, delta


def _robust_time_scale_and_align(t: np.ndarray, ts_gt: np.ndarray) -> tuple[np.ndarray, float, float]:
    """自动识别时间单位并对齐（兜底用）"""
    t0, ts0 = float(t[0]), float(ts_gt[0])
    t_rel  = t - t0
    ts_rel = ts_gt - ts0

    # 判定粒度
    diffs_e = np.diff(np.sort(t_rel))
    diffs_e = diffs_e[diffs_e > 0]
    dt_e = float(np.median(diffs_e)) if diffs_e.size else 0.0
    dt_g = _median_dt(ts_rel)

    cands = np.array([1.0, 1e-3, 1e-6, 1e3, 1e6], dtype=np.float64)
    if dt_e <= 0 or dt_g <= 0:
        span_e = max(1e-12, float((t_rel.max() if t_rel.size else 0.0) - (t_rel.min() if t_rel.size else 0.0)))
        span_g = max(1e-12, float(ts_rel[-1] - ts_rel[0]))
        errs = np.abs(cands * span_e - span_g)
    else:
        errs = np.abs(cands * dt_e - dt_g)
    scale = float(cands[int(np.argmin(errs))])

    t_rel_scaled = t_rel * scale

    # 对齐平移
    idx_prev = np.searchsorted(ts_rel, t_rel_scaled, side='right') - 1
    idx_prev = np.clip(idx_prev, 0, ts_rel.size - 1)
    idx_next = np.clip(idx_prev + 1, 0, ts_rel.size - 1)

    dt_prev  = np.abs(t_rel_scaled - ts_rel[idx_prev])
    dt_next  = np.abs(ts_rel[idx_next] - t_rel_scaled)
    nearest  = np.where(dt_next < dt_prev, ts_rel[idx_next], ts_rel[idx_prev])

    delta = float(np.median(nearest - t_rel_scaled))
    t_aligned = t_rel_scaled + delta + ts0
    return t_aligned, scale, delta


# -------------------------- 评估指标计算 -------------------------- #
def compute_comprehensive_metrics(disp_gt: np.ndarray, disp_pred: np.ndarray,
                                  bad_thresholds: list[float] = [1.0, 2.0, 3.0]) -> Dict[str, float]:
    """
    计算全面的立体匹配评估指标
    
    Args:
        disp_gt: Ground truth disparity
        disp_pred: Predicted disparity
        bad_thresholds: Thresholds for bad pixel percentage
    
    Returns:
        Dictionary containing all metrics
    """
    errors = np.abs(disp_pred - disp_gt)
    squared_errors = (disp_pred - disp_gt) ** 2
    
    # 避免除零
    valid_gt = disp_gt > 1e-6
    relative_errors = np.where(valid_gt, errors / disp_gt, 0.0)
    relative_errors_with_sign = np.where(valid_gt, (disp_pred - disp_gt) / disp_gt, 0.0)
    
    metrics = {}
    
    # 1. 基础误差指标
    metrics['MAE'] = float(np.mean(errors))
    metrics['RMSE'] = float(np.sqrt(np.mean(squared_errors)))
    metrics['Median_Error'] = float(np.median(errors))
    metrics['Std_Dev'] = float(np.std(errors))
    
    # 2. Bad pixel percentages (不同阈值)
    for thresh in bad_thresholds:
        bad_ratio = np.mean(errors > thresh) * 100.0
        metrics[f'Bad_{thresh:.1f}px'] = float(bad_ratio)
    
    # 3. 相对误差指标
    if np.any(valid_gt):
        metrics['MRE'] = float(np.mean(relative_errors_with_sign[valid_gt]))  # Mean Relative Error (can be negative)
        metrics['MARE'] = float(np.mean(relative_errors[valid_gt]))  # Mean Absolute Relative Error
    else:
        metrics['MRE'] = float('nan')
        metrics['MARE'] = float('nan')
    
    # 4. 阈值准确度 (A1, A2, A3) - 常用于深度估计
    # δ < 1.25, 1.25², 1.25³
    if np.any(valid_gt):
        ratio = np.maximum(disp_pred[valid_gt] / disp_gt[valid_gt], 
                          disp_gt[valid_gt] / disp_pred[valid_gt])
        metrics['A1'] = float(np.mean(ratio < 1.25) * 100.0)  # δ < 1.25
        metrics['A2'] = float(np.mean(ratio < 1.25**2) * 100.0)  # δ < 1.5625
        metrics['A3'] = float(np.mean(ratio < 1.25**3) * 100.0)  # δ < 1.953125
    else:
        metrics['A1'] = metrics['A2'] = metrics['A3'] = float('nan')
    
    # 5. 百分位数误差
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        metrics[f'P{p}_Error'] = float(np.percentile(errors, p))
    
    # 6. Inlier ratios (不同阈值下的内点率)
    inlier_thresholds = [0.5, 1.0, 2.0, 3.0]
    for thresh in inlier_thresholds:
        inlier_ratio = np.mean(errors <= thresh) * 100.0
        metrics[f'Inlier_{thresh:.1f}px'] = float(inlier_ratio)
    
    # 7. 相对误差百分位数
    if np.any(valid_gt):
        for p in [50, 95, 99]:
            metrics[f'P{p}_RelError'] = float(np.percentile(relative_errors[valid_gt], p) * 100.0)
    else:
        for p in [50, 95, 99]:
            metrics[f'P{p}_RelError'] = float('nan')
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """打印格式化的评估指标"""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    
    # 基础误差指标
    print(f"\n{'Basic Error Metrics':-^70}")
    print(f"  MAE (Mean Absolute Error)        : {metrics['MAE']:8.3f} px")
    print(f"  RMSE (Root Mean Square Error)    : {metrics['RMSE']:8.3f} px")
    print(f"  Median Error                     : {metrics['Median_Error']:8.3f} px")
    print(f"  Standard Deviation               : {metrics['Std_Dev']:8.3f} px")
    
    # Bad pixel percentages
    print(f"\n{'Bad Pixel Percentages':-^70}")
    for key in sorted([k for k in metrics.keys() if k.startswith('Bad_')]):
        print(f"  {key:35s} : {metrics[key]:7.2f} %")
    
    # Inlier ratios
    print(f"\n{'Inlier Ratios':-^70}")
    for key in sorted([k for k in metrics.keys() if k.startswith('Inlier_')]):
        print(f"  {key:35s} : {metrics[key]:7.2f} %")
    
    # 相对误差
    print(f"\n{'Relative Error Metrics':-^70}")
    if not np.isnan(metrics['MRE']):
        print(f"  MRE (Mean Relative Error)        : {metrics['MRE']:8.3f} ({metrics['MRE']*100:6.2f} %)")
        print(f"  MARE (Mean Abs. Relative Error)  : {metrics['MARE']:8.3f} ({metrics['MARE']*100:6.2f} %)")
    else:
        print(f"  MRE / MARE                       : N/A (no valid GT)")
    
    # 阈值准确度
    print(f"\n{'Threshold Accuracy (δ < threshold)':-^70}")
    if not np.isnan(metrics['A1']):
        print(f"  A1 (δ < 1.25)                    : {metrics['A1']:7.2f} %")
        print(f"  A2 (δ < 1.25²)                   : {metrics['A2']:7.2f} %")
        print(f"  A3 (δ < 1.25³)                   : {metrics['A3']:7.2f} %")
    else:
        print(f"  A1 / A2 / A3                     : N/A (no valid GT)")
    
    # 百分位数误差
    print(f"\n{'Percentile Errors (Absolute)':-^70}")
    print(f"  50th percentile (Median)         : {metrics['P50_Error']:8.3f} px")
    print(f"  75th percentile                  : {metrics['P75_Error']:8.3f} px")
    print(f"  90th percentile                  : {metrics['P90_Error']:8.3f} px")
    print(f"  95th percentile                  : {metrics['P95_Error']:8.3f} px")
    print(f"  99th percentile                  : {metrics['P99_Error']:8.3f} px")
    
    # 相对误差百分位数
    if not np.isnan(metrics['P50_RelError']):
        print(f"\n{'Percentile Errors (Relative)':-^70}")
        print(f"  50th percentile                  : {metrics['P50_RelError']:7.2f} %")
        print(f"  95th percentile                  : {metrics['P95_RelError']:7.2f} %")
        print(f"  99th percentile                  : {metrics['P99_RelError']:7.2f} %")
    
    print(f"{'='*70}\n")


# -------------------------- 评估主流程 -------------------------- #
def evaluate(match_path: pathlib.Path, gt_npz: pathlib.Path,
             *, bad_px: float = 2.0, subsample: int = 10_000_000,
             t0: float | None = None, until: float | None = None,
             t_center: float | None = None, dt_window: float | None = None,
             plot_full: bool = False, dt_tol_ms: float | None = None,
             t_units: str = 'auto', disp_min: float = 4.0, disp_max: float = 32.0,
             mae_thresh: float = 8.0):
    """
    评估匹配文件中的视差与 GT 视差之间的误差，并输出全面的评估指标
    """
    # ---------- 读取 GT ----------
    gt = np.load(gt_npz)
    disp_gt: np.ndarray = gt['disp']
    ts_gt:   np.ndarray = gt['ts'].astype(np.float64)
    F, H, W = disp_gt.shape
    if F < 2:
        print("✗ GT 帧数不足"); return

    print(f"[Disparity filter] 仅评估真值视差范围 [{disp_min:.1f}, {disp_max:.1f}] 的点")

    # ---------- 读取预测匹配 ----------
    data = np.load(match_path)
    x_L = data['x_L'].astype(np.float32)
    y_L = data['y_L'].astype(np.float32)
    t   = data['t'].astype(np.float64)
    disp_pred = data['disp'].astype(np.float32)

    if x_L.size == 0:
        print("✗ 没有匹配点"); return

    # ---------- 时间对齐 ----------
    if t_units == 's':
        t_aligned, delta = _align_only(t, ts_gt)
        print(f"[Time] 用户指定单位=秒 → 仅做平移对齐，delta={delta:.6f}s")
    elif t_units == 'ms':
        t_sec = t * 1e-3
        t_aligned, delta = _align_only(t_sec, ts_gt)
        print(f"[Time] 用户指定单位=毫秒 → 转秒后对齐，delta={delta:.6f}s")
    elif t_units == 'us':
        t_sec = t * 1e-6
        t_aligned, delta = _align_only(t_sec, ts_gt)
        print(f"[Time] 用户指定单位=微秒 → 转秒后对齐，delta={delta:.6f}s")
    else:
        t_aligned, scale, delta = _robust_time_scale_and_align(t, ts_gt)
        print(f"[Time] auto 识别 scale={scale:.2e}, delta={delta:.6f}s")

    # ---------- 窗口与离帧门控 ----------
    win_lo, win_hi = _resolve_window(ts_gt, t0, until, t_center, dt_window)
    if dt_tol_ms is None:
        dt_tol = max(5e-3, 0.5 * _median_dt(ts_gt))
    else:
        dt_tol = dt_tol_ms * 1e-3
    print(f"[Window] [{win_lo:.3f}, {win_hi:.3f}]s | dt_tol={dt_tol*1e3:.1f}ms")

    in_win = (t_aligned >= win_lo) & (t_aligned <= win_hi)
    if not np.any(in_win):
        print("✗ 窗口内无匹配点"); return

    x_L = x_L[in_win]
    y_L = y_L[in_win]
    t_aligned = t_aligned[in_win]
    disp_pred = disp_pred[in_win]

    # ---------- 线性插值 GT 视差 ----------
    prev_idx = np.searchsorted(ts_gt, t_aligned, side='right') - 1
    prev_idx = np.clip(prev_idx, 0, F-2)
    next_idx = prev_idx + 1

    x_L_int = np.round(x_L).astype(int)
    y_L_int = np.round(y_L).astype(int)
    valid_xy = (x_L_int >= 0) & (x_L_int < W) & (y_L_int >= 0) & (y_L_int < H)
    if not np.any(valid_xy):
        print("✗ 无有效坐标"); return

    x_L_int = x_L_int[valid_xy]
    y_L_int = y_L_int[valid_xy]
    t_aligned = t_aligned[valid_xy]
    disp_pred = disp_pred[valid_xy]
    prev_idx = prev_idx[valid_xy]
    next_idx = next_idx[valid_xy]

    d_prev = disp_gt[prev_idx, y_L_int, x_L_int]
    d_next = disp_gt[next_idx, y_L_int, x_L_int]
    t_prev = ts_gt[prev_idx]
    t_next = ts_gt[next_idx]

    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.where(t_next > t_prev, (t_aligned - t_prev) / (t_next - t_prev), 0.5)
    w = np.clip(w, 0.0, 1.0)
    disp_gt_interp = d_prev + w * (d_next - d_prev)

    # ---------- 离帧门控 ----------
    dt_prev = np.abs(t_aligned - t_prev)
    dt_next = np.abs(t_next - t_aligned)
    dt_nearest = np.minimum(dt_prev, dt_next)
    within_tol = (dt_nearest <= dt_tol)

    if not np.any(within_tol):
        print("✗ 离帧门控后无样本"); return

    disp_gt_interp = disp_gt_interp[within_tol]
    disp_pred = disp_pred[within_tol]
    t_aligned = t_aligned[within_tol]
    prev_idx = prev_idx[within_tol]
    next_idx = next_idx[within_tol]

    # ---------- 下采样（如需要） ----------
    if disp_pred.size > subsample:
        sel = np.random.choice(disp_pred.size, subsample, replace=False)
        disp_gt_interp = disp_gt_interp[sel]
        disp_pred = disp_pred[sel]
        t_aligned = t_aligned[sel]
        prev_idx = np.searchsorted(ts_gt, t_aligned, side='right') - 1
        prev_idx = np.clip(prev_idx, 0, F-1)
        next_idx = np.clip(prev_idx + 1, 0, F-1)
        use_next = (np.abs(ts_gt[next_idx] - t_aligned) < np.abs(t_aligned - ts_gt[prev_idx]))
        fidx_ok = np.where(use_next, next_idx, prev_idx)
    else:
        use_next = (np.abs(ts_gt[next_idx] - t_aligned) < np.abs(t_aligned - ts_gt[prev_idx]))
        fidx_ok = np.where(use_next, next_idx, prev_idx)

    # ---------- 异常值过滤 + 视差范围过滤 ----------
    finite_gt_all = disp_gt[(disp_gt > 0) & np.isfinite(disp_gt)]
    disp_hi = float(np.percentile(finite_gt_all, 99.9)) if finite_gt_all.size else 512.0
    
    ok = (
        (disp_gt_interp >= disp_min) & (disp_gt_interp <= disp_max) &
        (disp_gt_interp > 0) & np.isfinite(disp_gt_interp) &
        np.isfinite(disp_pred) & (disp_pred >= 0) & (disp_pred < 4.0 * disp_hi)
    )
    
    if not np.any(ok):
        print(f"✗ 没有在视差范围 [{disp_min:.1f}, {disp_max:.1f}] 内的有效样本")
        return

    disp_gt_s   = disp_gt_interp[ok].astype(np.float32)
    disp_pred_s = disp_pred[ok].astype(np.float32)
    fidx_ok     = fidx_ok[ok]

    total_valid_before = np.sum(
        (disp_gt_interp > 0) & np.isfinite(disp_gt_interp) &
        np.isfinite(disp_pred) & (disp_pred >= 0) & (disp_pred < 4.0 * disp_hi)
    )
    print(f"[Disparity filter] {ok.sum():,}/{total_valid_before:,} "
          f"({ok.sum()/max(1, total_valid_before)*100:.1f}%) 在范围 [{disp_min:.1f}, {disp_max:.1f}]")

    # ---------- 逐帧 MAE 计算（用于帧过滤） ----------
    err_px = np.abs(disp_pred_s - disp_gt_s)
    sum_e = np.bincount(fidx_ok, weights=err_px, minlength=F).astype(np.float32)
    cnt   = np.bincount(fidx_ok, minlength=F).astype(np.int32)
    mae_frame = np.full(F, np.nan, np.float32)
    nz = cnt > 0
    mae_frame[nz] = sum_e[nz] / cnt[nz]

    # ---------- MAE帧过滤 ----------
    good_frames = np.where(np.isfinite(mae_frame) & (mae_frame <= mae_thresh))[0]
    
    if good_frames.size == 0:
        print(f"✗ 所有帧的MAE都超过阈值 {mae_thresh:.1f}"); return
    
    frames_with_data = np.sum(nz)
    frames_filtered = frames_with_data - good_frames.size
    print(f"[MAE frame filter] 阈值: {mae_thresh:.1f} px")
    print(f"[MAE frame filter] 剔除帧数: {frames_filtered}/{frames_with_data} "
          f"({frames_filtered/max(1, frames_with_data)*100:.1f}%)")
    
    frame_mask = np.isin(fidx_ok, good_frames)
    samples_before = disp_pred_s.size
    disp_gt_s_filtered = disp_gt_s[frame_mask]
    disp_pred_s_filtered = disp_pred_s[frame_mask]
    fidx_ok_filtered = fidx_ok[frame_mask]
    
    samples_after = disp_pred_s_filtered.size
    samples_removed = samples_before - samples_after
    print(f"[MAE frame filter] 剔除样本数: {samples_removed:,}/{samples_before:,} "
          f"({samples_removed/max(1, samples_before)*100:.1f}%)\n")
    
    if samples_after == 0:
        print("✗ MAE帧过滤后无剩余样本"); return

    # ---------- 计算所有评估指标 ----------
    # 过滤前的指标
    metrics_before = compute_comprehensive_metrics(disp_gt_s, disp_pred_s, bad_thresholds=[1.0, 2.0, 3.0])
    print_metrics(metrics_before, title="Metrics BEFORE MAE Frame Filter")
    
    # 过滤后的指标
    metrics_after = compute_comprehensive_metrics(disp_gt_s_filtered, disp_pred_s_filtered, 
                                                  bad_thresholds=[1.0, 2.0, 3.0])
    print_metrics(metrics_after, title="Metrics AFTER MAE Frame Filter")
    
    # 打印改进情况
    print(f"{'='*70}")
    print(f"{'Improvement Summary':^70}")
    print(f"{'='*70}")
    print(f"  MAE improvement       : {metrics_before['MAE'] - metrics_after['MAE']:+7.3f} px "
          f"({(metrics_before['MAE'] - metrics_after['MAE'])/metrics_before['MAE']*100:+6.1f}%)")
    print(f"  RMSE improvement      : {metrics_before['RMSE'] - metrics_after['RMSE']:+7.3f} px "
          f"({(metrics_before['RMSE'] - metrics_after['RMSE'])/metrics_before['RMSE']*100:+6.1f}%)")
    print(f"  Bad-2.0px improvement : {metrics_before['Bad_2.0px'] - metrics_after['Bad_2.0px']:+7.2f} %")
    print(f"{'='*70}\n")
    
    print(f"[GT range] disp ∈ [{disp_gt_s_filtered.min():.2f}, {disp_gt_s_filtered.max():.2f}] px")
    print(f"[Samples] {samples_after:,} points evaluated")

    # ---------- 重新计算逐帧 MAE（用于绘图） ----------
    sum_e_filtered = np.bincount(fidx_ok_filtered, weights=np.abs(disp_pred_s_filtered - disp_gt_s_filtered), 
                                  minlength=F).astype(np.float32)
    cnt_filtered = np.bincount(fidx_ok_filtered, minlength=F).astype(np.int32)
    mae_frame_filtered = np.full(F, np.nan, np.float32)
    nz_filtered = cnt_filtered > 0
    mae_frame_filtered[nz_filtered] = sum_e_filtered[nz_filtered] / cnt_filtered[nz_filtered]

    # ---------- 绘图 ----------
    if plot_full:
        plot_mask = np.ones(F, dtype=bool)
        x_ts = ts_gt - ts_gt[0]
        title_suffix = f" (full, disp∈[{disp_min:.1f},{disp_max:.1f}], MAE≤{mae_thresh:.1f})"
        out_suffix = f".enhanced_mae{mae_thresh:.1f}_disp{disp_min:.1f}-{disp_max:.1f}.full.png"
    else:
        plot_mask = (ts_gt >= win_lo) & (ts_gt <= win_hi)
        x_ts = ts_gt[plot_mask] - win_lo
        title_suffix = f" [{win_lo:.2f},{win_hi:.2f}]s (disp∈[{disp_min:.1f},{disp_max:.1f}], MAE≤{mae_thresh:.1f})"
        out_suffix = f".enhanced_mae{mae_thresh:.1f}_disp{disp_min:.1f}-{disp_max:.1f}.{win_lo:.2f}-{win_hi:.2f}.png"

    plt.figure(figsize=(10, 4))
    plt.plot(x_ts, mae_frame_filtered[plot_mask], lw=1.5, label='Per-frame MAE (filtered)')
    plt.axhline(y=metrics_after['MAE'], color='r', linestyle='--', alpha=0.7, 
                label=f'Overall MAE: {metrics_after["MAE"]:.3f} px')
    plt.xlabel("Time (s)" + ("" if not plot_full else " (from GT start)"))
    plt.ylabel("MAE (px)")
    plt.title(f"Per-frame MAE (after frame filter){title_suffix}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    finite_mae = mae_frame_filtered[plot_mask][np.isfinite(mae_frame_filtered[plot_mask])]
    if finite_mae.size:
        y95 = np.percentile(finite_mae, 95)
        y_max = max(1e-3, y95 * 1.2)
        plt.ylim(0, y_max)

    plt.tight_layout()
    out_png = match_path.with_suffix(out_suffix)
    plt.savefig(out_png, dpi=150)
    print(f"\n✓ Per-frame MAE curve saved → {out_png}")


# ------------------------------- CLI ------------------------------- #
def main():
    pa = argparse.ArgumentParser('MVSEC evaluator – Enhanced with comprehensive metrics')
    sub = pa.add_subparsers(dest='cmd', required=True)

    ex = sub.add_parser('extract_gt')
    ex.add_argument('--hdf5', type=pathlib.Path, required=True)
    ex.add_argument('--out',  type=pathlib.Path, default='indoor2_gt_disp.npz')
    ex.add_argument('--fx', type=float, required=True)
    ex.add_argument('--baseline', type=float, required=True)

    ev = sub.add_parser('evaluate')
    ev.add_argument('--match', type=pathlib.Path, default='matches11_2.npz')
    ev.add_argument('--gt',    type=pathlib.Path, default='indoor3_gt_disp.npz')
    ev.add_argument('--bad_px', type=float, default=2.0)
    ev.add_argument('--subsample', type=int, default=30_000_000)

    ev.add_argument('--t0', type=float, help='窗口起始时间（秒）')
    ev.add_argument('--until', type=float, help='窗口结束时间（秒）')
    ev.add_argument('--t_center', type=float, help='窗口中心时间（秒）')
    ev.add_argument('--dt_window', type=float, help='窗口总时长（秒）')
    
    ev.add_argument('--plot_full', action='store_true',
                    help='绘图显示全时长（默认仅显示窗口）')
    ev.add_argument('--dt_tol_ms', type=float, default=15.0,
                    help='允许的事件与GT帧时间差上限（毫秒）')
    ev.add_argument('--t_units', choices=['auto', 's', 'ms', 'us'], default='s',
                    help='匹配时间戳单位（建议明确指定 s）')
    ev.add_argument('--disp_min', type=float, default=4.0,
                    help='真值视差最小值（默认4.0）')
    ev.add_argument('--disp_max', type=float, default=32.0,
                    help='真值视差最大值（默认32.0）')
    ev.add_argument('--mae_thresh', type=float, default=8.0,
                    help='逐帧MAE阈值，高于此值的帧会被剔除（默认8.0）')

    args = pa.parse_args()
    if args.cmd == 'extract_gt':
        extract_gt(args.hdf5, args.out, fx=args.fx, baseline=args.baseline)
    else:
        evaluate(args.match, args.gt,
                 bad_px=args.bad_px,
                 subsample=args.subsample,
                 t0=args.t0, until=args.until,
                 t_center=args.t_center, dt_window=args.dt_window,
                 plot_full=args.plot_full,
                 dt_tol_ms=args.dt_tol_ms,
                 t_units=args.t_units,
                 disp_min=args.disp_min, disp_max=args.disp_max,
                 mae_thresh=args.mae_thresh)


if __name__ == '__main__':
    main()


#python evaluate.py evaluate --match matches9_1.npz --gt gt_disp.npz --plot_full --t0 1504645183.425 --until 1504645239.425