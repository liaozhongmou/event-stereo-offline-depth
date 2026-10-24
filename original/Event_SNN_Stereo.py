#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Event-based stereo (row-wise) — V2+ (adaptive, cross-checked, subpixel):

Key ideas added over the user's baseline:
1) Adaptive time window (ERAW): expand/shrink around each left-event to
   reach a target amount of support. This keeps windows tight in textured
   regions and wide in sparse ones → lower outliers.
2) Dual-consistency cross-check: do a cheap right→left IRLS fit around the
   selected right position/time (using the *same row*), and penalize or reject
   if the reprojected x_L drifts from the real x_L.
3) Local evidence field with subpixel refinement: evaluate a small discrete
   kernelized “like−conflict” score at integer x_R bins around the IRLS
   intercept, then refine the peak by a 1D parabolic fit for subpixel x_R.
4) Temporal prior per pixel: keep last (t, d) at each (y, x). Matches that are
   consistent in time (small Δt and |d−d_prev|) get a boost; large jumps are
   down-weighted. This stabilizes tracks and reduces flicker.
5) Polarity-flexible scoring: you can enforce same-polarity or simply weight
   conflicts; the latter helps in low-texture where polarity flips occur.
6) Robustness tricks: Huber IRLS (as before), residual gating, LR symmetry
   check, and a light row-wise bilateral median post-filter.

Designed to remain fast with Numba and pure NumPy prepacking (CSR by rows).

CLI quickstart (MVSEC indoor_flying1 suggested):
python event_stereo_v2_plus.py \
  --left left.npz --right right.npz --out matches_v2.npz \
  --W 346 --H 260 \
  --disp_min 4 --disp_max 28 \
  --dt_win_min 0.0015 --dt_win_max 0.008 --target_support 12 \
  --tau_w 0.0015 --sigma_x 2.25 --alpha_like 1.6 --beta_conf 2.2 \
  --gamma_res 1.2 --cross_eps 1.2 --bins_radius 6 --expand 1.6 \
  --temporal_prior --tau_disp 0.25 --sigma_disp 3.0

Notes:
- Time units follow your input; if MVSEC timestamps are in µs, set --time_scale 1e-6.
- Post-filter is intentionally mild; crank --row_bimed_k up if you want more smoothing.
"""

from __future__ import annotations
import numpy as np
import argparse, pathlib, json
from typing import Tuple
import time

# ---- Numba import (with graceful fallback) ----
try:
    from numba import njit, prange
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco
    def prange(*args, **kwargs):
        return range(*args, **kwargs)

# ---- Performance profiler import ----
try:
    from profiler import PerformanceProfiler
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
    print("[INFO] Profiler module not found. Performance metrics disabled.")

# ------------------------------ Robust loader ------------------------------ #
def _get(data: np.lib.npyio.NpzFile, keys, *, optional=False):
    for k in keys:
        if k in data.files:
            return data[k]
    if optional:
        return None
    raise ValueError(f"缺少字段之一: {keys}，实际字段: {list(data.files)}")


def load_events_npz(path: pathlib.Path, time_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load events (x,y,t,p) from npz; accept aliases like 'polarity', 'ts', 'timestamps'."""
    data = np.load(path)
    x = _get(data, ["x", "xs"]).astype(np.int32)
    y = _get(data, ["y", "ys"]).astype(np.int32)
    t = _get(data, ["t", "ts", "timestamps"]).astype(np.float64)
    p = _get(data, ["p", "polarity", "pol", "pols"]).astype(np.int16)

    # 标准化极性到 ±1
    u = np.unique(p)
    if set(u.tolist()).issubset({0, 1}):
        p = p * 2 - 1
    p = p.astype(np.int8)

    # 时间缩放并排序（保证单调）
    t = t * float(time_scale)
    ord_t = np.argsort(t)
    return x[ord_t], y[ord_t], t[ord_t], p[ord_t]

# ------------------------------ Numba kernels ------------------------------ #
@njit(cache=True, fastmath=True)
def passthrough_numba(x, y, t, p, W, H):
    """只做越界剔除；输入已按时间升序。返回 (x,y,t,p) 子序列。"""
    N = t.shape[0]
    out_x = np.empty(N, np.int32)
    out_y = np.empty(N, np.int32)
    out_t = np.empty(N, np.float64)
    out_p = np.empty(N, np.int8)
    w = 0
    for i in range(N):
        xi = x[i]; yi = y[i]
        if 0 <= xi < W and 0 <= yi < H:
            out_x[w] = xi
            out_y[w] = yi
            out_t[w] = t[i]
            out_p[w] = p[i]
            w += 1
    return out_x[:w], out_y[:w], out_t[:w], out_p[:w]

@njit(cache=True)
def lower_bound(a, left, right, x):
    lo = left
    hi = right
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo

@njit(cache=True)
def upper_bound(a, left, right, x):
    lo = left
    hi = right
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo

# ------------------------------ CSR builders ------------------------------ #

def build_rows_csr(X, Y, T, P, H):
    """
    Pack events by (row, time) into CSR-like buffers:
      - flat arrays: X_flat, T_flat, P_flat
      - row_ptr: len H+1, points into the flat arrays
    Pure NumPy (Numba lacks lexsort in nopython).
    """
    mask = (Y >= 0) & (Y < H)
    X = X[mask]; Y = Y[mask]; T = T[mask]; P = P[mask]
    idx = np.lexsort((T, Y))  # primary: Y, secondary: T
    X = X[idx]; Y = Y[idx]; T = T[idx]; P = P[idx]
    counts = np.bincount(Y, minlength=H)
    row_ptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return X, T, P, row_ptr

# ---------------------- Huber weight & helpers ---------------------------- #
@njit(cache=True, fastmath=True)
def huber_weight(r, delta):
    ar = r if r >= 0.0 else -r
    if ar <= delta:
        return 1.0
    return delta / (ar + 1e-12)

@njit(cache=True, fastmath=True)
def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

# ------------------- Core: adaptive, cross-checked matching --------------- #
@njit(cache=True, fastmath=True)
def match_v2_adaptive(
    Lx, Ly, Lt, Lp,
    Rx_flat, Rt_flat, Rp_flat, R_row_ptr,
    Lx_flat, Lt_flat, Lp_flat, L_row_ptr,
    W, H,
    disp_min, disp_max,
    dt_win_min, dt_win_max, expand_factor, target_support,
    tau_w,
    same_polarity_flag,
    # IRLS / evidence params
    huber_delta, max_irls_iters,
    bins_radius, sigma_x,
    alpha_like, beta_conflict,
    gamma_residual,
    # temporal prior
    use_temporal_prior, prev_t, prev_d, tau_disp, sigma_disp, temp_boost,
    # cross-check
    cross_eps,
    # filters
    min_support, max_residual_norm, min_score,
    # mild row-wise post filter control
    row_bimed_k
):
    M = Lt.shape[0]
    out_xL = np.empty(M, np.int32)
    out_yL = np.empty(M, np.int32)
    out_t  = np.empty(M, np.float64)
    out_d  = np.empty(M, np.float32)
    out_score = np.empty(M, np.float32)
    out_resid = np.empty(M, np.float32)
    out_supp  = np.empty(M, np.int32)
    m = 0

    eps = 1e-12
    K = bins_radius * 2 + 1

    # row-wise scratch for light bilateral median (post) — store last k disps
    row_win_disp = np.empty((H, row_bimed_k), np.float32)
    row_win_cnt  = np.zeros(H, np.int32)

    for i in range(M):
        xL = Lx[i]; yL = Ly[i]; tL = Lt[i]; pL = Lp[i]
        if yL < 0 or yL >= H:
            continue
        rstart = R_row_ptr[yL]
        rend   = R_row_ptr[yL + 1]
        if rend <= rstart:
            continue

        # --- Adaptive time window to reach target_support ---
        dt = dt_win_min
        lo = lower_bound(Rt_flat, rstart, rend, tL - dt)
        hi = upper_bound(Rt_flat, rstart, rend, tL + dt)
        # count candidates under disparity & polarity constraints
        def count_cands(lo_, hi_):
            cnt = 0
            xr_min = xL - disp_max
            if xr_min < 0: xr_min = 0
            xr_max = xL - disp_min
            if xr_max > W - 1: xr_max = W - 1
            if xr_min > xr_max:
                return 0
            for j in range(lo_, hi_):
                xr = Rx_flat[j]
                if xr < xr_min or xr > xr_max:
                    continue
                if same_polarity_flag == 1 and Rp_flat[j] != pL:
                    continue
                cnt += 1
            return cnt

        cand = count_cands(lo, hi)
        while cand < target_support and dt < dt_win_max:
            dt *= expand_factor
            if dt > dt_win_max: dt = dt_win_max
            lo = lower_bound(Rt_flat, rstart, rend, tL - dt)
            hi = upper_bound(Rt_flat, rstart, rend, tL + dt)
            cand = count_cands(lo, hi)
        if cand < min_support:
            continue

        # --- Initial weighted LS (like baseline) to get (a, b) ---
        S_w = 0.0; S_t = 0.0; S_tt = 0.0; S_x = 0.0; S_tx = 0.0
        xr_min = xL - disp_max
        if xr_min < 0: xr_min = 0
        xr_max = xL - disp_min
        if xr_max > W - 1: xr_max = W - 1
        for j in range(lo, hi):
            xr = Rx_flat[j]
            if xr < xr_min or xr > xr_max:
                continue
            if same_polarity_flag == 1 and Rp_flat[j] != pL:
                continue
            dtj = Rt_flat[j] - tL
            adt = dtj if dtj >= 0.0 else -dtj
            wt  = np.exp(-adt / max(1e-12, tau_w))
            S_w  += wt
            S_t  += wt * dtj
            S_tt += wt * dtj * dtj
            S_x  += wt * xr
            S_tx += wt * dtj * xr
        if S_w <= eps:
            continue
        Delta = S_w * S_tt - S_t * S_t
        if Delta <= eps:
            a = 0.0; b = S_x / S_w
        else:
            a = (S_w * S_tx - S_t * S_x) / Delta
            b = (S_tt * S_x - S_t * S_tx) / Delta

        # --- IRLS refinement ---
        for _ in range(max_irls_iters):
            Sw = 0.0; St = 0.0; Stt = 0.0; Sx = 0.0; Stx = 0.0
            for j in range(lo, hi):
                xr = Rx_flat[j]
                if xr < xr_min or xr > xr_max:
                    continue
                if same_polarity_flag == 1 and Rp_flat[j] != pL:
                    continue
                dtj = Rt_flat[j] - tL
                adt = dtj if dtj >= 0.0 else -dtj
                wt  = np.exp(-adt / max(1e-12, tau_w))
                r   = xr - (a * dtj + b)
                wr  = huber_weight(r, huber_delta)
                w   = wt * wr
                Sw  += w
                St  += w * dtj
                Stt += w * dtj * dtj
                Sx  += w * xr
                Stx += w * dtj * xr
            if Sw <= eps:
                break
            D = Sw * Stt - St * St
            if D <= eps:
                a = 0.0; b = Sx / Sw
            else:
                a = (Sw * Stx - St * Sx) / D
                b = (Stt * Sx - St * Stx) / D
        xR_hat = b

        # --- Evidence field over integer bins around xR_hat ---
        center_i = int(np.floor(xR_hat + 0.5))
        E = np.zeros(K, np.float64)
        # also gather weighted right-time anchor near the (as yet unknown) best bin
        # we will compute after we know the argmax; for now accumulate per-bin scores
        for k in range(-bins_radius, bins_radius + 1):
            xbin = center_i + k
            if xbin < 0 or xbin > W - 1:
                E[k + bins_radius] = -1e9
                continue
            acc = 0.0
            for j in range(lo, hi):
                xr = Rx_flat[j]
                if xr < xr_min or xr > xr_max:
                    continue
                dtj = Rt_flat[j] - tL
                adt = dtj if dtj >= 0.0 else -dtj
                wt  = np.exp(-adt / max(1e-12, tau_w))
                dx  = xr - xbin
                adx = dx if dx >= 0.0 else -dx
                wx  = np.exp(- adx / max(1e-12, sigma_x))
                sp  = 1 if Rp_flat[j] == pL else -1
                acc += (alpha_like * (sp == 1) - beta_conflict * (sp == -1)) * wt * wx
            E[k + bins_radius] = acc
        # choose best bin and (optional) parabolic subpixel refine
        best_k = 0
        best_v = E[0]
        for k in range(1, K):
            v = E[k]
            if v > best_v:
                best_v = v; best_k = k
        # margin for confidence
        second_v = -1e18
        for k in range(K):
            if k == best_k: continue
            v = E[k]
            if v > second_v: second_v = v
        # parabola through (k-1,k,k+1)
        k0 = best_k
        xR_peak = center_i + (k0 - bins_radius)
        if 1 <= k0 <= K - 2:
            y1 = E[k0 - 1]; y2 = E[k0]; y3 = E[k0 + 1]
            denom = (y1 - 2*y2 + y3)
            if denom != 0.0:
                offset = 0.5 * (y1 - y3) / denom  # in [-0.5,0.5] typically
                if offset < -1.0: offset = -1.0
                if offset >  1.0: offset =  1.0
                xR_peak = (center_i + (k0 - bins_radius)) + offset
        # base score combines peak value and margin
        score_hist = best_v - 0.3 * second_v

        # --- Compute residual around (a,b) for gating ---
        sum_w = 0.0; sse = 0.0; support = 0
        for j in range(lo, hi):
            xr = Rx_flat[j]
            if xr < xr_min or xr > xr_max:
                continue
            if same_polarity_flag == 1 and Rp_flat[j] != pL:
                continue
            dtj = Rt_flat[j] - tL
            adt = dtj if dtj >= 0.0 else -dtj
            wt  = np.exp(-adt / max(1e-12, tau_w))
            r   = xr - (a * dtj + b)
            wr  = huber_weight(r, huber_delta)
            w   = wt * wr
            sum_w += w
            sse   += w * r * r
            support += 1
        if support < min_support or sum_w <= eps:
            continue
        resid_norm = sse / (sum_w + eps)

        # --- Cross-check (right→left) near the chosen right peak time ---
        # Compute a right-time anchor as the weighted mean time of right events
        # close to xR_peak.
        wsum = 0.0; t_anchor = tL
        for j in range(lo, hi):
            xr = Rx_flat[j]
            if xr < xr_min or xr > xr_max:
                continue
            dtj = Rt_flat[j] - tL
            adt = dtj if dtj >= 0.0 else -dtj
            wt  = np.exp(-adt / max(1e-12, tau_w))
            dx  = xr - xR_peak
            adx = dx if dx >= 0.0 else -dx
            wx  = np.exp(- adx / max(1e-12, sigma_x))
            w   = wt * wx
            wsum += w
            t_anchor += (Rt_flat[j] - t_anchor) * (w / (wsum + eps))
        # Left row segment
        lstart = L_row_ptr[yL]
        lend   = L_row_ptr[yL + 1]
        if lend > lstart:
            lo2 = lower_bound(Lt_flat, lstart, lend, t_anchor - dt)
            hi2 = upper_bound(Lt_flat, lstart, lend, t_anchor + dt)
            # quick IRLS fit xL ≈ a2*(t-t_anchor) + b2 with candidates near disparity band
            S_w2 = 0.0; S_t2 = 0.0; S_tt2 = 0.0; S_x2 = 0.0; S_tx2 = 0.0
            xl_min = xR_peak + disp_min
            if xl_min < 0: xl_min = 0
            xl_max = xR_peak + disp_max
            if xl_max > W - 1: xl_max = W - 1
            supp2 = 0
            for j in range(lo2, hi2):
                xl = Lx_flat[j]
                if xl < xl_min or xl > xl_max:
                    continue
                dtj = Lt_flat[j] - t_anchor
                adt = dtj if dtj >= 0.0 else -dtj
                wt  = np.exp(-adt / max(1e-12, tau_w))
                S_w2  += wt
                S_t2  += wt * dtj
                S_tt2 += wt * dtj * dtj
                S_x2  += wt * xl
                S_tx2 += wt * dtj * xl
                supp2 += 1
            if S_w2 > eps and supp2 >= min_support:
                D2 = S_w2 * S_tt2 - S_t2 * S_t2
                if D2 <= eps:
                    a2 = 0.0; b2 = S_x2 / S_w2
                else:
                    a2 = (S_w2 * S_tx2 - S_t2 * S_x2) / D2
                    b2 = (S_tt2 * S_x2 - S_t2 * S_tx2) / D2
                xL_hat_cc = b2  # at t_anchor
                # penalty if left reprojection disagrees with true xL
                cc_pen = (xL_hat_cc - xL)
                if cc_pen < 0.0: cc_pen = -cc_pen
            else:
                cc_pen = cross_eps * 2.0  # force penalty if no support
        else:
            cc_pen = cross_eps * 2.0

        # --- Temporal prior
        temp_bonus = 0.0
        if use_temporal_prior == 1:
            d_temp = float(xL) - float(xR_peak)
            pd = prev_d[yL, xL]
            pt = prev_t[yL, xL]
            if pt > 0.0:  # seen before
                dtp = tL - pt
                if dtp < 0.0: dtp = -dtp
                # exp time and disparity proximity
                td = np.exp(- dtp / max(1e-12, tau_disp))
                dd = d_temp - pd
                if dd < 0.0: dd = -dd
                sd = np.exp(- dd / max(1e-12, sigma_disp))
                temp_bonus = temp_boost * td * sd

        # --- Final score and gating ---
        score = score_hist - gamma_residual * resid_norm - 0.9 * cc_pen + temp_bonus

        if (resid_norm <= max_residual_norm) and (score >= min_score) and (cc_pen <= cross_eps):
            d = float(xL) - float(xR_peak)

            # light row-wise bilateral median (keep last K disps in the row)
            if row_bimed_k > 0:
                idxr = row_win_cnt[yL] % max(1, row_bimed_k)
                row_win_disp[yL, idxr] = d
                row_win_cnt[yL] += 1
                # use the median of the last up-to-k disps as a gentle smoother
                nwin = row_bimed_k if row_win_cnt[yL] >= row_bimed_k else row_win_cnt[yL]
                # simple insertion-sort median for small k
                med = 0.0
                if nwin > 0:
                    # copy small window into local buffer for sorting
                    tmp = np.empty(nwin, np.float32)
                    for q in range(nwin): tmp[q] = row_win_disp[yL, q]
                    # insertion sort
                    for aidx in range(1, nwin):
                        key = tmp[aidx]
                        bidx = aidx - 1
                        while bidx >= 0 and tmp[bidx] > key:
                            tmp[bidx + 1] = tmp[bidx]
                            bidx -= 1
                        tmp[bidx + 1] = key
                    if (nwin & 1) == 1:
                        med = tmp[nwin // 2]
                    else:
                        med = 0.5 * (tmp[nwin // 2 - 1] + tmp[nwin // 2])
                d_out = 0.7 * d + 0.3 * med
            else:
                d_out = d

            out_xL[m] = xL
            out_yL[m] = yL
            out_t[m]  = tL
            out_d[m]  = d_out
            out_score[m] = score
            out_resid[m] = resid_norm
            out_supp[m]  = support

            # update temporal memory
            if use_temporal_prior == 1:
                prev_t[yL, xL] = tL
                prev_d[yL, xL] = d_out

            m += 1

    return out_xL[:m], out_yL[:m], out_t[:m], out_d[:m], out_score[:m], out_resid[:m], out_supp[:m]

# ----------------------------------- CLI ----------------------------------- #
def main():
    pa = argparse.ArgumentParser("Event stereo V2+: adaptive window + subpixel evidence + cross-check + temporal prior")
    pa.add_argument("--left",  type=pathlib.Path, default=r"indoor2_data\indoor2_left2.npz")
    pa.add_argument("--right", type=pathlib.Path, default=r"indoor2_data\indoor2_right2.npz")
    pa.add_argument("--out",   type=pathlib.Path, default="matches10.npz")

    # Geometry
    pa.add_argument("--W", type=int, default=346)
    pa.add_argument("--H", type=int, default=260)

    # Time scaling (e.g., microseconds -> seconds use 1e-6)
    pa.add_argument("--time_scale", type=float, default=1.0, help="scale factor applied to t in input files")

    # Matching params
    pa.add_argument("--disp_min", type=int, default=4)
    pa.add_argument("--disp_max", type=int, default=32)

    # Adaptive window
    pa.add_argument("--dt_win_min", type=float, default=0.001, help="min time window (s)")
    pa.add_argument("--dt_win_max", type=float, default=0.3, help="max time window (s)")
    pa.add_argument("--expand",     type=float, default=1.5,    help="window expansion factor")
    pa.add_argument("--target_support", type=int, default=5,   help="target #cands in window")

    pa.add_argument("--tau_w",      type=float, default=0.05, help="temporal decay (s)")
    pa.add_argument("--same_polarity", action="store_true", help="enforce polarity consistency (instead of soft penalty)")

    # IRLS / evidence params
    pa.add_argument("--huber_delta", type=float, default=2.8, help="Huber delta in pixels")
    pa.add_argument("--irls_iters",  type=int,   default=3,   help="IRLS iterations (2~3 is enough)")
    pa.add_argument("--bins_radius", type=int,   default=4,   help="#integer bins on each side for evidence field")
    pa.add_argument("--sigma_x",     type=float, default=6.0,help="spatial kernel (pixels) for evidence field")
    pa.add_argument("--alpha_like",  type=float, default=8.5, help="weight for like counts")
    pa.add_argument("--beta_conf",   type=float, default=8.2, help="weight for conflict counts")
    pa.add_argument("--gamma_res",   type=float, default=0.08, help="weight for residual penalty")

    # temporal prior per (y,x)
    pa.add_argument("--temporal_prior", action="store_true", help="enable temporal prior")
    pa.add_argument("--tau_disp",   type=float, default=0.55, help="time decay (s) for prior")
    pa.add_argument("--sigma_disp", type=float, default=5.0,  help="disp diff scale (px) for prior")
    pa.add_argument("--temp_boost", type=float, default=0.6,  help="bonus for temporal consistency")

    # cross-check
    pa.add_argument("--cross_eps",  type=float, default=3.0,  help="max |xL_hat_cc - xL| to accept")

    # Filters
    pa.add_argument("--min_support", type=int,   default=2,    help="min candidates within time window")
    pa.add_argument("--max_resid",   type=float, default=20.0,  help="max normalized residual (px^2)")
    pa.add_argument("--min_score",   type=float, default=0.05,  help="min total score to accept")

    # light post-filter
    pa.add_argument("--row_bimed_k", type=int,   default=10,    help="row-wise bilateral median window (0=off)")
    
    # Performance profiling
    pa.add_argument("--profile", action="store_true", help="enable performance profiling")
    pa.add_argument("--profile_output", type=pathlib.Path, default="performance_report.json",
                    help="profiler output JSON file")

    args = pa.parse_args()
    
    # Initialize profiler if enabled
    profiler = None
    if args.profile and PROFILER_AVAILABLE:
        profiler = PerformanceProfiler(sample_interval=0.1)
        profiler.start()
        print("[PROFILE] Performance monitoring enabled")
    elif args.profile:
        print("[WARNING] --profile requested but profiler module not available")

    # Load events
    t_load_start = time.perf_counter()
    Lx_raw, Ly_raw, Lt_raw, Lp_raw = load_events_npz(args.left,  time_scale=args.time_scale)
    Rx_raw, Ry_raw, Rt_raw, Rp_raw = load_events_npz(args.right, time_scale=args.time_scale)

    # Filter by image bounds
    Lx, Ly, Lt, Lp = passthrough_numba(Lx_raw, Ly_raw, Lt_raw, Lp_raw, args.W, args.H)
    Rx, Ry, Rt, Rp = passthrough_numba(Rx_raw, Ry_raw, Rt_raw, Rp_raw, args.W, args.H)
    print(f"[PASS] left events: {Lt.size} | right events: {Rt.size}")
    
    t_preprocess = time.perf_counter() - t_load_start
    if profiler:
        profiler.set_stage_time("preprocessing", t_preprocess)
        profiler.sample()

    # CSR by rows (both eyes, for cross-check)
    Rx_flat, Rt_flat, Rp_flat, R_row_ptr = build_rows_csr(Rx, Ry, Rt, Rp, args.H)
    Lx_flat, Lt_flat, Lp_flat, L_row_ptr = build_rows_csr(Lx, Ly, Lt, Lp, args.H)

    # Temporal memories (for prior)
    prev_t = np.zeros((args.H, args.W), dtype=np.float64)
    prev_d = np.zeros((args.H, args.W), dtype=np.float32)

    # Run matching
    t_match_start = time.perf_counter()
    same_pol = 1 if args.same_polarity else 0
    xL, yL, tL, disp, score, resid, supp = match_v2_adaptive(
        Lx, Ly, Lt, Lp,
        Rx_flat, Rt_flat, Rp_flat, R_row_ptr,
        Lx_flat, Lt_flat, Lp_flat, L_row_ptr,
        args.W, args.H,
        args.disp_min, args.disp_max,
        args.dt_win_min, args.dt_win_max, args.expand, args.target_support,
        args.tau_w,
        same_pol,
        args.huber_delta, args.irls_iters,
        args.bins_radius, args.sigma_x,
        args.alpha_like, args.beta_conf,
        args.gamma_res,
        1 if args.temporal_prior else 0, prev_t, prev_d, args.tau_disp, args.sigma_disp, args.temp_boost,
        args.cross_eps,
        args.min_support, args.max_resid, args.min_score,
        args.row_bimed_k
    )
    t_match = time.perf_counter() - t_match_start
    if profiler:
        profiler.set_stage_time("matching", t_match)
        profiler.sample()
    
    print(f"[MATCH] matches: {xL.size} | time: {t_match:.3f}s")

    # Save
    t_save_start = time.perf_counter()
    # Save
    t_save_start = time.perf_counter()
    meta_dict = {
        "dt_win_min": args.dt_win_min,
        "dt_win_max": args.dt_win_max,
        "expand": args.expand,
        "target_support": args.target_support,
        "tau_w": args.tau_w,
        "huber_delta": args.huber_delta,
        "irls_iters": args.irls_iters,
        "bins_radius": args.bins_radius,
        "sigma_x": args.sigma_x,
        "alpha_like": args.alpha_like,
        "beta_conf": args.beta_conf,
        "gamma_res": args.gamma_res,
        "min_support": args.min_support,
        "max_resid": args.max_resid,
        "min_score": args.min_score,
        "same_polarity": int(args.same_polarity),
        "temporal_prior": int(args.temporal_prior),
        "tau_disp": args.tau_disp,
        "sigma_disp": args.sigma_disp,
        "temp_boost": args.temp_boost,
        "cross_eps": args.cross_eps,
        "row_bimed_k": args.row_bimed_k,
    }
    meta_json = json.dumps(meta_dict)

    np.savez_compressed(
        args.out,
        x_L=xL, y_L=yL, t=tL, disp=disp,
        score=score, resid=resid, support=supp,
        meta_json=np.array(meta_json)
    )
    t_save = time.perf_counter() - t_save_start
    if profiler:
        profiler.set_stage_time("postprocessing", t_save)
    
    print(f"[OK] saved -> {args.out.resolve()}")
    
    # Finalize profiling
    if profiler:
        profiler.stop(left_events=int(Lt.size), right_events=int(Rt.size), matches=int(xL.size))
        profiler.print_summary()
        profiler.save_report(args.profile_output)


if __name__ == "__main__":
    main()


#python new_methods\Event_SNN_Stereo.py --temporal_prior --same_polarity --profile