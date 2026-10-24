#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIF (no-reset) + log-threshold adaptation for event streams.

- Input:  left1.npz  with keys: (x,y,t,p) OR (x,y,t,polarity) etc.
- Output: left2.npz  with keys: (x,y,t,p)  (p in {-1,+1})

每个像素(346x260)一个 LIF 神经元：
  V(t) 事件到来时按 ±v_step 积分，且按 tau_mem 指数衰减；
  触发规则：p=+1 时 V>θ 输出 +1；p=-1 时 V<−θ 输出 −1；
  触发后不复位 V，后续继续指数衰减；
  事件频率 R 做指数滑动累积（tau_rate），阈值 θ = clip(θ0 + k_log*log1p(R*s_rate))，
  高频→θ 上升（对数），低频→R 衰减→θ 回落到 θ0。
"""

import numpy as np
import argparse, pathlib, math
from typing import Tuple

# ---- Numba import (with graceful fallback) ----
try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco

# ------------------------------ Robust loader ------------------------------ #
def _get(data: np.lib.npyio.NpzFile, keys, *, optional=False):
    for k in keys:
        if k in data.files:
            return data[k]
    if optional:
        return None
    raise ValueError(f"缺少字段之一: {keys}，实际字段: {list(data.files)}")

def load_events_npz(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load events (x,y,t,p) from npz; accept aliases like 'polarity', 'ts', 'timestamps'."""
    data = np.load(path)
    x = _get(data, ["x","xs"]).astype(np.int32)
    y = _get(data, ["y","ys"]).astype(np.int32)
    t = _get(data, ["t","ts","timestamps"]).astype(np.float64)

    # polarity/p：支持 {-1,+1} 或 {0,1}
    p = _get(data, ["p","polarity","pol","pols"]).astype(np.int16)
    u = np.unique(p)
    if set(u.tolist()).issubset({0,1}):
        p = (p * 2 - 1)  # 0/1 -> -1/+1
    p = p.astype(np.int8)

    # 时间排序
    order = np.argsort(t)
    return x[order], y[order], t[order], p[order]

# ---------------------------- LIF kernel (numba) --------------------------- #
@njit(cache=True, fastmath=False)
def _lif_kernel(x, y, t, p, W, H,
                tau_mem, tau_rate, v_step,
                theta0, theta_min, theta_max, k_log, s_rate):
    """事件驱动的逐像素 LIF 更新；与原先的 Python 循环逐比特等价。

    状态保持原样：阈值先写入 float32 的 T 再读回参与比较（参考实现即如此，
    该次舍入会影响触发判定），因此这里不做代数化简，以保证输出可复现。
    """
    V      = np.zeros((H, W), dtype=np.float32)          # 膜电位
    R      = np.zeros((H, W), dtype=np.float32)          # 事件频率指数滑动“漏桶”
    T      = np.zeros((H, W), dtype=np.float32)          # 动态阈值
    t_last = np.zeros((H, W), dtype=np.float64)          # 该像素上次更新时间
    for i in range(H):
        for j in range(W):
            T[i, j] = theta0

    N = x.shape[0]
    out_x = np.empty(N, dtype=np.int32)
    out_y = np.empty(N, dtype=np.int32)
    out_t = np.empty(N, dtype=np.float64)
    out_p = np.empty(N, dtype=np.int8)
    m = 0

    for i in range(N):
        xi = x[i]; yi = y[i]; ti = t[i]; pi = p[i]
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            continue

        # 指数衰减至当前时刻
        dt = ti - t_last[yi, xi]
        if dt < 0:
            # 若时间戳异常乱序，安全跳过
            continue
        if dt > 0:
            V[yi, xi] *= math.exp(-dt / max(1e-9, tau_mem))
            R[yi, xi] *= math.exp(-dt / max(1e-9, tau_rate))
            t_last[yi, xi] = ti

        # 积分：按极性 ±v_step
        V[yi, xi] += v_step * (1.0 if pi > 0 else -1.0)
        # 频率计数 +1（与极性无关）
        R[yi, xi] += 1.0

        # 阈值：对数上升，低频时因 R 衰减而自然回落
        thr = theta0 + k_log * math.log1p(R[yi, xi] * s_rate)
        if thr < theta_min:
            thr = theta_min
        elif thr > theta_max:
            thr = theta_max
        T[yi, xi] = thr

        # 触发（极性一致的对称阈值）；触发后不复位膜电位，继续指数衰减
        if (V[yi, xi] > T[yi, xi] and pi > 0) or (V[yi, xi] < -T[yi, xi] and pi < 0):
            out_x[m] = xi; out_y[m] = yi; out_t[m] = ti
            out_p[m] = 1 if pi > 0 else -1
            m += 1

    return out_x[:m], out_y[:m], out_t[:m], out_p[:m]

# ---------------------------- LIF Main Function ---------------------------- #
def lif_filter_events(
    in_path: pathlib.Path = pathlib.Path("left1.npz"),
    out_path: pathlib.Path = pathlib.Path("left2.npz"),
    W: int = 346, H: int = 260,
    tau_mem: float = 0.05,      # 膜电位时间常数 (s)
    tau_rate: float = 0.20,     # 事件频率时间常数 (s)
    v_step: float = 1.0,        # 每事件电位步长
    theta0: float = 3.0,        # 阈值基线
    theta_min: float = 1.0,     # 阈值下限
    theta_max: float = 12.0,    # 阈值上限
    k_log: float = 2.0,         # 阈值随频率上升的对数系数
    s_rate: float = 1.0         # 频率缩放（调节 log1p 的速度）
):
    x, y, t, p = load_events_npz(in_path)

    out_x, out_y, out_t, out_p = _lif_kernel(
        x, y, t, p, W, H,
        tau_mem, tau_rate, v_step,
        theta0, theta_min, theta_max, k_log, s_rate
    )

    # 保存输出
    np.savez_compressed(
        out_path,
        x=out_x,
        y=out_y,
        t=out_t,
        p=out_p
    )
    # 使用 ASCII 字符避免 Windows GBK 控制台编码问题
    print(f"[OK] 输入事件: {len(x)} | 输出事件: {len(out_x)} -> {out_path}")

# --------------------------------- CLI ------------------------------------ #
def main():
    pa = argparse.ArgumentParser("LIF（不复位）+ log阈值自适应 事件滤波")
    pa.add_argument("--in",  dest="in_path",  type=pathlib.Path, default=r"indoor1_data\indoor1_left1.npz")
    pa.add_argument("--out", dest="out_path", type=pathlib.Path, default=r"indoor1_data\indoor1_left2.npz")
    pa.add_argument("--W", type=int, default=346)
    pa.add_argument("--H", type=int, default=260)
    pa.add_argument("--tau_mem",  type=float, default=0.70)
    pa.add_argument("--tau_rate", type=float, default=0.01)
    pa.add_argument("--v_step",   type=float, default=1.0)
    pa.add_argument("--theta0",   type=float, default=1.05)
    pa.add_argument("--theta_min",type=float, default=1.05)
    pa.add_argument("--theta_max",type=float, default=1000.0)
    pa.add_argument("--k_log",    type=float, default=2.4)
    pa.add_argument("--s_rate",   type=float, default=2.1)
    args = pa.parse_args()

    lif_filter_events(
        in_path=args.in_path, out_path=args.out_path,
        W=args.W, H=args.H,
        tau_mem=args.tau_mem, tau_rate=args.tau_rate,
        v_step=args.v_step,
        theta0=args.theta0, theta_min=args.theta_min, theta_max=args.theta_max,
        k_log=args.k_log, s_rate=args.s_rate
    )

if __name__ == "__main__":
    main()


