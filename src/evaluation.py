import hashlib

import numpy as np


def associate(prediction, gt, ts, interval, gt_range=(4, 28)):
    """Outcome-independent association on unique GT frame-pixel units."""
    x, y, t, d = [np.asarray(prediction[k]) for k in ('x_L', 'y_L', 't', 'disp')]
    ts = np.asarray(ts)
    if gt.ndim != 3 or ts.ndim != 1 or len(ts) != gt.shape[0] or not len(ts):
        raise ValueError('GT must be [frame,y,x] with one timestamp per frame')
    if not np.isfinite(ts).all() or not (np.diff(ts) > 0).all():
        raise ValueError('GT timestamps must be finite and strictly increasing')
    if any(a.ndim != 1 or len(a) != len(t) for a in (x, y, t, d)):
        raise ValueError('prediction fields must be equal-length vectors')
    h, w = gt.shape[1:]
    if (not np.isfinite(t).all() or not np.isfinite(x).all() or not np.isfinite(y).all()
            or (x != np.floor(x)).any() or (y != np.floor(y)).any()
            or (x < 0).any() or (x >= w).any() or (y < 0).any() or (y >= h).any()):
        raise ValueError('finite seconds and integer in-bounds coordinates required')
    lo, hi = interval
    gt_min, gt_max = gt_range
    if not np.isfinite([lo, hi]).all() or hi < lo:
        raise ValueError('explicit finite closed interval required')
    frames = np.flatnonzero((ts >= lo) & (ts <= hi))
    # Count frame by frame to avoid materializing a second dense GT volume.
    valid_counts = np.array([np.count_nonzero(np.isfinite(gt[f]) & (gt[f] >= gt_min)
                                             & (gt[f] <= gt_max)) for f in frames])
    source = np.flatnonzero(np.isfinite(d) & (t >= lo) & (t <= hi))
    pt = t[source]
    nxt = np.clip(np.searchsorted(ts, pt), 0, len(ts) - 1)
    prev = np.maximum(nxt - 1, 0)
    frame = np.where(np.abs(ts[nxt] - pt) < np.abs(ts[prev] - pt), nxt, prev)
    delta = np.abs(ts[frame] - pt)
    value = gt[frame, y[source].astype(np.int64), x[source].astype(np.int64)]
    keep = ((delta <= .015) & (ts[frame] >= lo) & (ts[frame] <= hi)
            & np.isfinite(value) & (value >= gt_min) & (value <= gt_max))
    source, frame, delta, value = (a[keep] for a in (source, frame, delta, value))
    unit = frame * h * w + y[source].astype(np.int64) * w + x[source].astype(np.int64)
    order = np.lexsort((source, delta, unit))
    unique = np.r_[True, np.diff(unit[order]) != 0] if len(order) else np.array([], bool)
    chosen = order[unique]
    fingerprint = hashlib.sha256()
    for a in (ts, gt):
        fingerprint.update(str((a.shape, a.dtype)).encode('ascii'))
        fingerprint.update(np.ascontiguousarray(a).view(np.uint8))
    return dict(unit=unit[chosen], frame=frame[chosen], source=source[chosen],
                prediction=d[source[chosen]], gt=value[chosen], delta=delta[chosen],
                frames=frames, frame_times=ts[frames], image_shape=(h, w), interval=(lo, hi),
                gt_fingerprint=fingerprint.hexdigest(), gt_range=tuple(gt_range),
                valid_counts=valid_counts, denominator=int(valid_counts.sum()))


def select(association, index):
    result = association.copy()
    for key in ('unit', 'frame', 'source', 'prediction', 'gt', 'delta'):
        result[key] = association[key][index]
    return result


def common_support(associations):
    if not associations:
        raise ValueError('at least one association required')
    first = associations[0]
    if any(a['gt_fingerprint'] != first['gt_fingerprint'] for a in associations):
        raise ValueError('common support requires identical complete ground truth and time axes')
    if any(not np.array_equal(a['frames'], first['frames'])
           or not np.array_equal(a['gt_range'], first['gt_range'])
           or not np.array_equal(a['frame_times'], first['frame_times'])
           or not np.array_equal(a['image_shape'], first['image_shape'])
           or not np.array_equal(a['interval'], first['interval'])
           or not np.array_equal(a['valid_counts'], first['valid_counts']) for a in associations):
        raise ValueError('common support requires the same GT frame population')
    common = first['unit']
    for a in associations[1:]:
        common = np.intersect1d(common, a['unit'], assume_unique=True)
    selected = [select(a, np.searchsorted(a['unit'], common)) for a in associations]
    if any(not np.array_equal(a['gt'], selected[0]['gt']) for a in selected):
        raise ValueError('common units must have the same ground truth')
    return selected


def metrics(association):
    error = np.abs(association['prediction'].astype(float) - association['gt'].astype(float))
    n, denominator = len(error), association['denominator']
    result = dict(units=n, valid_gt_units=denominator,
                  coverage=n / denominator if denominator else None)
    for name in ('mae', 'rmse', 'median', 'p90', 'bad1', 'bad2', 'bad3'):
        result[name] = None
    if n:
        result.update(mae=float(error.mean()), rmse=float(np.sqrt(np.mean(error ** 2))),
                      median=float(np.median(error)), p90=float(np.quantile(error, .9)))
    for threshold in (1, 2, 3):
        result[f'bad{threshold}'] = float(np.mean(error > threshold)) if n else None
        result[f'correct{threshold}_all_gt'] = (int(np.count_nonzero(error <= threshold))
                                               / denominator if denominator else None)
    return result


def frame_metrics(association):
    frames = association['frames']
    index = np.searchsorted(frames, association['frame'])
    error = np.abs(association['prediction'].astype(float) - association['gt'].astype(float))
    counts = np.bincount(index, minlength=len(frames))
    sums = np.bincount(index, weights=error, minlength=len(frames))
    means = np.full(len(frames), np.nan)
    np.divide(sums, counts, out=means, where=counts > 0)
    return dict(frame=frames, units=counts, mae=means, valid_gt_units=association['valid_counts'])


def frame_bootstrap(association, repeats=2000, seed=20260905):
    means = frame_metrics(association)['mae']
    if not np.isfinite(means).any():
        return dict(mean=None, ci95=None, nonempty_frames=0, valid_replicates=0)
    block = min(6, len(means))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repeats):
        starts = rng.integers(0, len(means) - block + 1, size=(len(means) + block - 1) // block)
        indices = (starts[:, None] + np.arange(block)).ravel()[:len(means)]
        selected = means[indices]
        finite = selected[np.isfinite(selected)]
        if len(finite):
            samples.append(float(finite.mean()))
    return dict(mean=float(np.nanmean(means)), ci95=np.quantile(samples, [.025, .975]).tolist(),
                nonempty_frames=int(np.isfinite(means).sum()), valid_replicates=len(samples))
