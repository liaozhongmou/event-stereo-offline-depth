#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按时间窗逐帧补洞（保留原事件 + 逐帧追加插值像素）——安全读写版

修复点：
- 读取 .npz 时默认 `allow_pickle=False`，并**避免**访问 object 数组（典型是 `meta`）。
- 写出时**不复制**任何 object 数组；若发现 legacy `meta`，会尝试转成 `meta_json`（字符串）。

输入：matches.npz（至少含 x、y、t、disp；键名支持常见别名）
处理：按 dt 切帧，每帧用固定大小卷积核做“占比判据”，对空洞做邻域均值插值
输出：--out 指定的新 npz（原事件 + 每帧新增像素），且不会包含 object 数组

依赖：仅 numpy
"""

import numpy as np
import argparse, pathlib, json
from typing import Dict, Tuple

# ---------- 自动识别键名 ----------

def detect_keys(files) -> Dict[str, str]:
    cand = {
        "x": ["x_L", "x", "xs"],
        "y": ["y_L", "y", "ys"],
        "t": ["ts", "t", "time", "timestamp", "timestamps"],
        "d": ["disp", "disparity", "d"],
    }
    out = {}
    for k, opts in cand.items():
        for o in opts:
            if o in files:
                out[k] = o
                break
        if k not in out:
            raise KeyError(f"缺少 {k} 键（可接受: {opts}），实际: {list(files)}")
    return out

# ---------- 稀疏 -> 网格（同像素多命中做均值/中位数） ----------

def points_to_grid(x, y, val, H, W, agg="mean"):
    dtype = val.dtype
    flat = (y.astype(np.int64) * W + x.astype(np.int64))
    uniq, inv = np.unique(flat, return_inverse=True)
    if agg == "mean":
        sums = np.bincount(inv, weights=val.astype(np.float64))
        cnts = np.bincount(inv)
        vals = (sums / np.maximum(cnts, 1)).astype(dtype)
    else:  # median
        order = np.argsort(inv)
        invs, vs = inv[order], val[order]
        ends = np.flatnonzero(np.r_[True, invs[1:] != invs[:-1], True])
        vals = np.empty(len(ends) - 1, dtype)
        for i in range(len(vals)):
            seg = vs[ends[i] : ends[i + 1]]
            vals[i] = np.median(seg).astype(dtype, copy=False)
    grid = np.full(H * W, np.nan, dtype)
    grid[uniq] = vals
    return grid.reshape(H, W)

# ---------- 积分图盒滤（固定半径窗口和；输出尺寸=输入） ----------

def box_sum(arr: np.ndarray, r: int) -> np.ndarray:
    """O(1) 取 (2r+1)x(2r+1) 窗口和；边界 0 填充；返回与 arr 同尺寸。"""
    if r <= 0:
        return arr.astype(np.float64, copy=True)
    A = np.pad(arr, ((r, r), (r, r)), mode="constant", constant_values=0)
    S = A.cumsum(axis=0).cumsum(axis=1)
    return S[2 * r :, 2 * r :] - S[:-2 * r, 2 * r :] - S[2 * r :, :-2 * r] + S[:-2 * r, :-2 * r]

# ---------- 有效像素判定（统一口径） ----------

def make_valid_mask(disp: np.ndarray, rule: str) -> np.ndarray:
    valid = np.isfinite(disp)
    if rule == ">0":
        valid &= disp > 0
    elif rule == "!=0":
        valid &= disp != 0
    elif rule == "finite":
        pass
    else:
        raise ValueError("valid_rule 仅支持 '>0'、'!=0'、'finite'")
    return valid

# ---------- 单帧：占比判据 + 邻域均值插值 ----------

def densify_one_frame(
    xi,
    yi,
    ti,
    di,
    H,
    W,
    r: int,
    thr: float,
    avg_radius: int,
    iters: int,
    agg: str,
    area_map: np.ndarray,
    valid_rule: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回：新增像素的 (x_new, y_new, t_new, d_new)"""
    # 网格化
    disp = points_to_grid(xi, yi, di, H, W, agg=agg).astype(np.float32)
    time = points_to_grid(xi, yi, ti, H, W, agg=agg).astype(np.float64)

    # 统一有效判定
    valid = make_valid_mask(disp, valid_rule)
    valid0 = valid.copy()

    # 迭代填充
    for _ in range(iters):
        # 占比（边界用有效窗口面积归一化）
        cnt_occ = box_sum(valid.astype(np.float32), r)
        occ_ratio = cnt_occ / np.maximum(area_map, 1.0)

        # 邻域均值（用 avg_radius），仅由“有效”邻域贡献
        sum_d = box_sum(np.where(valid, disp, 0.0), avg_radius)
        sum_t = box_sum(np.where(valid, time, 0.0), avg_radius)
        cnt_avg = box_sum(valid.astype(np.float32), avg_radius)
        mean_d = np.where(cnt_avg > 0, sum_d / np.maximum(cnt_avg, 1e-6), np.nan)
        mean_t = np.where(cnt_avg > 0, sum_t / np.maximum(cnt_avg, 1e-6), np.nan)

        # 空洞且占比达标且邻域有样本
        hole = ~valid
        fillable = hole & (occ_ratio >= thr) & (cnt_avg > 0)
        if not np.any(fillable):
            break

        disp[fillable] = mean_d[fillable].astype(np.float32)
        time[fillable] = mean_t[fillable].astype(np.float64)
        valid[fillable] = True

    # 新增像素（仅本帧）
    new_mask = valid & (~valid0)
    yy_new, xx_new = np.where(new_mask)
    dd_new = disp[yy_new, xx_new].astype(np.float32)
    tt_new = time[yy_new, xx_new].astype(np.float64)
    return xx_new.astype(np.int32), yy_new.astype(np.int32), tt_new, dd_new

# ---------- 安全复制：保留非 object 字段，并尝试把 legacy meta → meta_json ----------

def copy_safe_fields(in_path: pathlib.Path, data: np.lib.npyio.NpzFile, skip_keys) -> Dict[str, np.ndarray]:
    out = {}
    for k in data.files:
        if k in skip_keys:
            continue
        try:
            out[k] = data[k]
        except ValueError as e:
            # 典型：Object arrays cannot be loaded when allow_pickle=False
            if "Object arrays cannot be loaded" in str(e):
                # 尝试单独用 allow_pickle 读取，并序列化为 meta_json
                try:
                    with np.load(in_path, allow_pickle=True) as data2:
                        obj = data2[k]
                        # 尝试将 numpy 对象转成 python 基本类型后再 JSON
                        try:
                            payload = obj.tolist()
                        except Exception:
                            payload = str(obj)
                        try:
                            out["meta_json"] = np.array(json.dumps(payload, ensure_ascii=False))
                        except Exception:
                            out["meta_str"] = np.array(str(payload))
                except Exception:
                    # 读不出来就静默跳过该字段
                    pass
            else:
                # 其他异常继续抛出，避免吞错
                raise
    return out

# ---------- 主流程：按时间窗切帧逐帧追加 ----------

def densify_by_time(
    in_path: pathlib.Path,
    out_path: pathlib.Path,
    H: int = None,
    W: int = None,
    dt: float = 0.1,  # 默认 100ms
    stride: float = None,  # 默认与 dt 相同
    ksize: int = 11,
    thr: float = 0.7,
    avg_radius: int = None,
    iters: int = 1,
    agg: str = "mean",
    sort_by_time: bool = True,
    valid_rule: str = ">0",  # 统一的有效像素口径
    print_frame_stats: bool = True,
):
    # 安全方式加载：默认禁止 pickle
    data = np.load(in_path, allow_pickle=False)
    keys = detect_keys(data.files)
    x = data[keys["x"]].astype(np.int32)
    y = data[keys["y"]].astype(np.int32)
    t = data[keys["t"]].astype(np.float64)
    d = data[keys["d"]].astype(np.float32)

    # 尺寸：优先读文件 H/W，否则由坐标推断
    if H is None:
        try:
            H = int(data["H"]) if "H" in data.files else int(y.max()) + 1
        except Exception:
            H = int(y.max()) + 1
    if W is None:
        try:
            W = int(data["W"]) if "W" in data.files else int(x.max()) + 1
        except Exception:
            W = int(x.max()) + 1

    r = ksize // 2
    if avg_radius is None:
        avg_radius = r
    if stride is None:
        stride = dt

    # 预计算：有效窗口面积图（边界不降权）
    area_map = box_sum(np.ones((H, W), np.float32), r)

    # 时间范围
    t0, t1 = float(np.min(t)), float(np.max(t))

    # 汇总新增
    new_x_list, new_y_list, new_t_list, new_d_list = [], [], [], []

    k = 0
    cur = t0
    while cur < t1 + 1e-12:
        t_start = cur
        t_end = min(cur + dt, t1 + 1e-9)
        idx = (t >= t_start) & (t < t_end)
        n = int(idx.sum())
        if n > 0:
            xi, yi, ti, di = x[idx], y[idx], t[idx], d[idx]
            xx_new, yy_new, tt_new, dd_new = densify_one_frame(
                xi, yi, ti, di, H, W, r, thr, avg_radius, iters, agg, area_map, valid_rule
            )

            if print_frame_stats:
                if valid_rule == ">0":
                    new_ok = int(np.sum(dd_new > 0))
                elif valid_rule == "!=0":
                    new_ok = int(np.sum(dd_new != 0))
                else:  # finite
                    new_ok = int(dd_new.size)
                print(
                    f"[Frame {k:05d}] t=[{t_start:.6f},{t_end:.6f})  pts={n}  new={len(xx_new)}  new(valid)={new_ok}"
                )

            if len(xx_new):
                new_x_list.append(xx_new)
                new_y_list.append(yy_new)
                new_t_list.append(tt_new)
                new_d_list.append(dd_new)
            k += 1
        cur += stride

    if new_x_list:
        new_x = np.concatenate(new_x_list)
        new_y = np.concatenate(new_y_list)
        new_t = np.concatenate(new_t_list)
        new_d = np.concatenate(new_d_list)
    else:
        new_x = np.empty(0, np.int32)
        new_y = np.empty(0, np.int32)
        new_t = np.empty(0, np.float64)
        new_d = np.empty(0, np.float32)

    # 追加到原事件（原事件不丢）
    x_out = np.concatenate([x, new_x])
    y_out = np.concatenate([y, new_y])
    t_out = np.concatenate([t, new_t])
    d_out = np.concatenate([d, new_d])

    # （可选）按时间排序
    if sort_by_time and len(t_out) > 0:
        order = np.argsort(t_out, kind="mergesort")  # 稳定排序
        x_out, y_out, t_out, d_out = x_out[order], y_out[order], t_out[order], d_out[order]

    # 写出：复制安全字段 + 新数组；完全避免 object 数组
    skip = {keys["x"], keys["y"], keys["t"], keys["d"]}
    extras = copy_safe_fields(in_path, data, skip_keys=skip)

    # 回写覆盖四大键
    extras[keys["x"]] = x_out
    extras[keys["y"]] = y_out
    extras[keys["t"]] = t_out
    extras[keys["d"]] = d_out

    # 保证 H/W 存在且为标量数值（非 object）
    extras["H"] = np.int32(H)
    extras["W"] = np.int32(W)

    np.savez_compressed(out_path, **extras)

    # 调试：打印文件点数，确认新增插值的像素点
    print(f"[OK] 原事件: {len(x)}  逐帧新增: {len(new_x)}  总计: {len(x_out)}")
    print(
        f"[参数] dt={dt}s stride={stride}s ksize={ksize} thr={thr} avg_radius={avg_radius} iters={iters} valid={valid_rule}"
    )
    print(f"[写出] {out_path}")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        "按时间窗逐帧：滑窗占比 + 邻域均值插值（保留原事件并追加新点，安全读写版）"
    )
    ap.add_argument("--in", dest="in_path", type=pathlib.Path, default="matches11_1.npz")
    ap.add_argument("--out", dest="out_path", type=pathlib.Path, default="matches11_2.npz")
    ap.add_argument("--H", type=int, default=None, help="图像高（可省略：优先读文件H或由坐标推断）")
    ap.add_argument("--W", type=int, default=None, help="图像宽（可省略：优先读文件W或由坐标推断）")

    ap.add_argument("--dt", type=float, default=0.1, help="时间窗（秒）")
    ap.add_argument("--stride", type=float, default=0.01, help="步长（秒），默认=dt，可设 <dt 做滑窗）")

    ap.add_argument("--ksize", type=int, default=17, help="滑窗大小(奇数)，如 7/9/11/13")
    ap.add_argument("--thr", type=float, default=0.2, help="窗口内有效像素占比阈值 0~1")
    ap.add_argument("--avg_radius", type=int, default=9, help="邻域均值半径；默认等于 ksize//2")
    ap.add_argument("--iters", type=int, default=3, help="插值迭代次数（>1 可层层外扩）")
    ap.add_argument("--agg", choices=["mean", "median"], default="median", help="同像素多命中的聚合")
    ap.add_argument(
        "--valid",
        choices=[">0", "!=0", "finite"],
        default=">0",
        help="有效像素判定：'>0'（默认）/ '!=0' / 'finite'",
    )
    ap.add_argument("--no_sort", action="store_true", help="不要按时间排序输出")
    ap.add_argument("--no_frame_stats", action="store_true", help="不打印逐帧统计")
    args = ap.parse_args()

    if args.ksize % 2 == 0 or args.ksize < 1:
        raise ValueError("ksize 必须为正奇数，例如 7/9/11/13")
    if not (0.0 <= args.thr <= 1.0):
        raise ValueError("thr 必须在 0~1 之间")

    densify_by_time(
        in_path=args.in_path,
        out_path=args.out_path,
        H=args.H,
        W=args.W,
        dt=args.dt,
        stride=args.stride,
        ksize=args.ksize,
        thr=args.thr,
        avg_radius=args.avg_radius,
        iters=args.iters,
        agg=args.agg,
        sort_by_time=(not args.no_sort),
        valid_rule=args.valid,
        print_frame_stats=(not args.no_frame_stats),
    )


if __name__ == "__main__":
    main()





