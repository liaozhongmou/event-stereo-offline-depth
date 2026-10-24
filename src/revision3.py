"""Small, explicit kernels for the unified JEI revision experiments."""
import numpy as np
from numba import njit


def box_sum(image, radius):
    padded = np.pad(image.astype(np.float64), radius)
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    width = 2 * radius + 1
    return (integral[width:, width:] - integral[:-width, width:]
            - integral[width:, :-width] + integral[:-width, :-width])


def complete_grid(grid, radius=8, threshold=.2, iterations=3, average_radius=9):
    result = grid.copy()
    area = box_sum(np.ones(grid.shape), radius)
    for _ in range(iterations):
        valid = np.isfinite(result) & (result > 0)
        count = box_sum(valid, average_radius)
        fill = (~valid & (box_sum(valid, radius) >= threshold * area) & (count > 0))
        if not fill.any():
            break
        result[fill] = box_sum(np.where(valid, result, 0), average_radius)[fill] / count[fill]
    return result


@njit(cache=True)
def temporal_median(x, y, t, disparity, window=5, gap=.3):
    width = int(x.max()) + 1
    pixels = (int(y.max()) + 1) * width
    history = np.empty((pixels, window), np.float32)
    count = np.zeros(pixels, np.int64)
    last = np.full(pixels, -np.inf)
    output = np.empty_like(disparity)
    for i in range(len(t)):
        p = int(y[i]) * width + int(x[i])
        if t[i] - last[p] > gap:
            count[p] = 0
        history[p, count[p] % window] = disparity[i]
        count[p] += 1
        last[p] = t[i]
        output[i] = np.median(history[p, :min(count[p], window)])
    return output


def rank_subset(scores, count):
    return np.sort(np.lexsort((np.arange(len(scores)), -scores))[:count])


@njit(cache=True)
def median_grid(x, y, values, height, width):
    flat = y.astype(np.int64) * width + x
    order = np.argsort(flat)
    grid = np.full((height, width), np.nan)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and flat[order[end]] == flat[order[start]]:
            end += 1
        pixel = flat[order[start]]
        grid[pixel // width, pixel % width] = np.median(values[order[start:end]])
        start = end
    return grid


def complete_events(prediction, shape, interval, dt=.1, stride=.01, radius=8,
                    threshold=.2, iterations=3, average_radius=9):
    x, y, t, d = (prediction[k] for k in ('x_L','y_L','t','disp'))
    height, width = shape
    origin = interval[0]
    additions = []
    area = box_sum(np.ones(shape), radius)
    for start in np.arange(interval[0], interval[1] - dt + 1e-8, stride):
        lo, hi = np.searchsorted(t, [start, start + dt])
        if hi == lo:
            continue
        grid = median_grid(x[lo:hi], y[lo:hi], d[lo:hi], height, width)
        time_grid = median_grid(x[lo:hi], y[lo:hi], t[lo:hi]-origin, height, width)
        initial = np.isfinite(grid) & (grid > 0)
        valid = initial.copy()
        for _ in range(iterations):
            count = box_sum(valid, average_radius)
            fill = (~valid & (box_sum(valid, radius) >= threshold * area) & (count > 0))
            if not fill.any():
                break
            grid[fill] = box_sum(np.where(valid,grid,0), average_radius)[fill]/count[fill]
            time_grid[fill] = box_sum(np.where(valid,time_grid,0), average_radius)[fill]/count[fill]
            valid[fill] = True
        yy, xx = np.where(valid & ~initial)
        additions.append((xx, yy, time_grid[yy,xx]+origin, grid[yy,xx]))
    output = {}
    for col, key in enumerate(('x_L','y_L','t','disp')):
        extra = np.concatenate([a[col] for a in additions]) if additions else np.array([])
        output[key] = np.concatenate([prediction[key],extra]).astype(prediction[key].dtype)
    output['kind'] = np.r_[np.zeros(len(t),np.uint8), np.ones(len(output['t'])-len(t),np.uint8)]
    order = np.argsort(output['t'], kind='stable')
    return {k:v[order] for k,v in output.items()}


@njit(cache=True)
def nearest_time_match(left, right, width, height, dt=.05, dmin=4, dmax=28):
    """Same-polarity nearest-time candidate in the same rectified row."""
    lx,ly,lt,lp = left
    rx,ry,rt,rp = right
    latest = np.full((height,width,2), -np.inf)
    # Offline bilateral support: latest past and earliest future candidates.
    best = np.full(len(lt), np.inf)
    disp = np.zeros(len(lt), np.float32)
    for direction in (1,-1):
        latest[:,:,:] = -np.inf if direction == 1 else np.inf
        j = 0 if direction == 1 else len(rt)-1
        for ii in range(len(lt)):
            i = ii if direction == 1 else len(lt)-1-ii
            while 0 <= j < len(rt) and (rt[j] <= lt[i] if direction == 1 else rt[j] >= lt[i]):
                latest[ry[j],rx[j],int(rp[j]>0)] = rt[j]
                j += direction
            for d in range(dmin,dmax+1):
                xr = lx[i]-d
                if xr < 0:
                    continue
                difference = abs(lt[i]-latest[ly[i],xr,int(lp[i]>0)])
                if difference < best[i] and difference <= dt:
                    best[i],disp[i] = difference,d
    keep = np.isfinite(best)
    return lx[keep],ly[keep],lt[keep],disp[keep],(dt-best[keep]).astype(np.float32)
