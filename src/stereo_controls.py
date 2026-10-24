"""Isolated matcher controls; original code preserved in original/."""
import numpy as np
from numba import njit

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
    row_bimed_k, cross_enabled=1, same_time=0
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
            if xbin < xr_min or xbin > xr_max:
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
            xbin = center_i + k - bins_radius
            if xbin < xr_min or xbin > xr_max:
                continue
            v = E[k]
            if v > second_v: second_v = v
        if second_v == -1e18:
            second_v = 0.0  # A singleton has peak evidence but no runner-up margin.
        # parabola through (k-1,k,k+1)
        k0 = best_k
        xR_peak = center_i + (k0 - bins_radius)
        if 1 <= k0 <= K - 2 and xr_min < xR_peak < xr_max:
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
                xL_hat_cc = b2 + (a2 * (tL - t_anchor) if same_time == 1 else 0.0)
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
        score = score_hist - gamma_residual * resid_norm - 0.9 * cc_pen * cross_enabled + temp_bonus

        if (resid_norm <= max_residual_norm) and (score >= min_score) and (cross_enabled == 0 or cc_pen <= cross_eps):
            d = float(xL) - float(xR_peak)
            if d < disp_min or d > disp_max:
                continue

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
