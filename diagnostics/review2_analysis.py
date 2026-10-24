"""Paired archive diagnostics; separate from unreproduced historical benchmarks."""
from pathlib import Path
import argparse
from contextlib import redirect_stdout
import importlib.util
import json
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from src.evaluation import metrics, select
from src.run_historical import load_prediction, save_output_hashes, sha256

ROOT = Path(__file__).resolve().parents[1]
COEFFICIENT = 19.941771812941038


def new_support_mask(reference_units, complete_units):
    """Partition locations; this does not assign provenance to saved rows."""
    return ~np.isin(complete_units, reference_units)


def depth_summary(prediction, gt, coefficient):
    positive = prediction > 0
    error = np.abs(coefficient / prediction[positive].astype(float)
                   - coefficient / gt[positive].astype(float))
    return dict(units=len(prediction), positive_prediction_units=int(positive.sum()),
                nonpositive_prediction_units=int((~positive).sum()),
                mae_m=float(error.mean()) if len(error) else None,
                rmse_m=float(np.sqrt(np.mean(error ** 2))) if len(error) else None,
                p90_m=float(np.quantile(error, .9)) if len(error) else None)


def boundary_mask(disparity, jump=1.0, radius=2):
    edge = np.zeros(disparity.shape, bool)
    horizontal = np.abs(np.diff(disparity, axis=1)) > jump
    vertical = np.abs(np.diff(disparity, axis=0)) > jump
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    edge[:-1] |= vertical
    edge[1:] |= vertical
    for axis in (0, 1):
        padded = np.pad(edge, ((radius, radius), (radius, radius)))
        edge = np.logical_or.reduce([padded[radius + dy:radius + dy + edge.shape[0],
                                             radius + dx:radius + dx + edge.shape[1]]
                                     for dy, dx in ([(i, 0) for i in range(-radius, radius + 1)]
                                                    if axis == 0 else [(0, i) for i in range(-radius, radius + 1)])])
    return edge


def describe(association):
    result = metrics(association)
    result['metric_depth'] = depth_summary(association['prediction'], association['gt'], COEFFICIENT)
    return result


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draw_panels(gt, association, newly_reached, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    frames = association['complete']['frames']
    selected = frames[np.rint(np.array([.1, .5, .9]) * (len(frames) - 1)).astype(int)]
    height, width = gt.shape[1:]
    figures = []
    for names in [('GT', 'Temporal', 'Completed', 'Absolute error'), ('Matcher', 'New support')]:
        fig, axes = plt.subplots(3, len(names), figsize=(7.1, 4.35 if len(names) == 4 else 6.8),
                                 constrained_layout=True, squeeze=False)
        for row, frame in enumerate(selected):
            for col, name in enumerate(names):
                ax = axes[row, col]
                image = np.full((height, width), np.nan)
                if name == 'GT':
                    valid = np.isfinite(gt[frame]) & (gt[frame] >= 4) & (gt[frame] <= 28)
                    image[valid] = gt[frame][valid]
                else:
                    stage = {'Temporal': 'smooth', 'Matcher': 'raw'}.get(name, 'complete')
                    a = association[stage]
                    keep = a['frame'] == frame
                    y, x = np.unravel_index(a['unit'][keep] % (height * width), (height, width))
                    image[y, x] = (np.abs(a['prediction'][keep] - a['gt'][keep]) if name == 'Absolute error'
                                    else newly_reached[keep].astype(float) if name == 'New support'
                                    else a['prediction'][keep])
                cmap = (ListedColormap(['#0072B2', '#D55E00']) if name == 'New support'
                        else plt.get_cmap('magma' if name == 'Absolute error' else 'viridis').copy())
                cmap.set_bad('white')
                limits = (0, 1) if name == 'New support' else (0, 8) if name == 'Absolute error' else (4, 28)
                im = ax.imshow(image, cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation='none')
                if row == 0:
                    ax.set_title(name, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if col == 0:
                    ax.set_ylabel(f'Frame {frame}', fontsize=9)
                if row == 2 and (col == len(names) - 1 or name == 'Completed'):
                    cb = fig.colorbar(im, ax=axes[:, col], location='bottom', fraction=.025, pad=.02)
                    if name == 'New support':
                        cb.set_ticks([.25, .75], labels=['Shared', 'New'])
                    else:
                        cb.set_label('Absolute error (px)' if name == 'Absolute error' else 'Disparity (px)')
        filename = 'stage_maps' if len(names) == 4 else 'source_masks'
        fig.savefig(out / (filename + '.pdf'))
        fig.savefig(out / (filename + '.png'), dpi=160)
        plt.close(fig)
        figures.append(filename)
    return selected.tolist(), figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out
    out.mkdir(exist_ok=False)
    contract = dict(status='B_DIAGNOSTIC_NOT_HISTORICAL_REPLACEMENT',
                    source_association='A_historical_v2_20261020',
                    newly_reached='S_complete minus S_temporal; location partition, not per-row provenance',
                    boundary=dict(disparity_jump_px=1., chebyshev_radius_px=2, valid_gt_range_px=[4, 28]),
                    depth_bins_m=[1., 2.], frame_fractions=[.1, .5, .9],
                    temporal_parameters=dict(window_size=20, deviation_threshold=2., outlier_tolerance=3,
                                             trend_degree=1, min_samples=4))
    (out / 'contract.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    baseline = ROOT / 'runs/A_historical_v2_20261020'
    historical = json.loads((baseline / 'metrics.json').read_text())
    cache_hashes = json.loads((baseline / 'output_hashes.json').read_text())
    association = {}
    for stage in ['raw', 'smooth', 'complete']:
        path = baseline / f'{stage}_associated.npz'
        assert sha256(path) == cache_hashes[path.name]['sha256']
        with np.load(path, allow_pickle=False) as z:
            association[stage] = {k: z[k] for k in z.files}
        association[stage]['denominator'] = int(association[stage]['denominator'])
    inputs = {}
    for old_path, record in historical['manifest']['inputs'].items():
        path = Path(old_path)
        assert sha256(path) == record['sha256'], str(path)
        inputs[path.name] = record
    a = association['complete']
    new = new_support_mask(association['smooth']['unit'], a['unit'])
    assert len(a['unit']) - int(new.sum()) == len(association['smooth']['unit'])
    results = dict(contract=contract, interval=a['interval'].tolist(), valid_gt_units=int(a['denominator']),
                   partition={name: describe(value) for name, value in [
                       ('matcher', association['raw']), ('temporal', association['smooth']),
                       ('completed_shared_locations', select(a, ~new)),
                       ('newly_reached', select(a, new)), ('all_completed', a)]})
    assert np.isclose(results['partition']['all_completed']['mae'] * len(new),
                      sum(results['partition'][key]['mae'] * results['partition'][key]['units']
                          for key in ['completed_shared_locations', 'newly_reached']))
    with np.load(ROOT.parent / 'indoor1_data/indoor1_gt_disp.npz') as z:
        gt = z['disp'] * np.float32(.8819476)
    height, width = gt.shape[1:]
    edge = np.zeros(len(a['unit']), bool)
    for frame in np.unique(a['frame']):
        valid = np.isfinite(gt[frame]) & (gt[frame] >= 4) & (gt[frame] <= 28)
        mask = boundary_mask(np.where(valid, gt[frame], np.nan))
        chosen = a['frame'] == frame
        edge[chosen] = mask.ravel()[a['unit'][chosen] % (height * width)]
    results['new_support_boundary'] = {name: describe(select(a, new & index))
                                       for name, index in [('within_2px', edge), ('away_from_boundary', ~edge)]}
    depth = COEFFICIENT / a['gt'].astype(float)
    results['depth_strata'] = {name: describe(select(a, index)) for name, index in [
        ('below_1m', depth < 1), ('1_to_2m', (depth >= 1) & (depth < 2)), ('2m_and_above', depth >= 2)]}
    frame_metrics = {}
    for name, stage in association.items():
        counts = np.bincount(np.searchsorted(stage['frames'], stage['frame']), minlength=len(stage['frames']))
        denominator = stage['valid_counts']
        cov = counts[denominator > 0] / denominator[denominator > 0]
        frame_metrics[name] = dict(valid_gt_frames=int((denominator > 0).sum()),
                                   coverage_quantiles=np.quantile(cov, [0, .1, .5, .9, 1]).tolist())
    results['frame_coverage'] = frame_metrics
    results['selected_frame_indices'], results['figures'] = draw_panels(gt, association, new, out)
    print('Completed archive partition, boundary and depth diagnostics.', flush=True)
    raw = load_prediction(ROOT.parent / 'new_methods/new_methods_results_indoor1/matches9.npz')
    replay = {}
    for label, path in [('legacy', ROOT / 'original/Adapative_Trend_Fitter.py'),
                        ('consistent_basis', ROOT / 'diagnostics/trend_time_basis_fixed.py')]:
        module = load_module(path, 'review2_' + label)
        started = time.perf_counter()
        with (out / f'{label}_replay.txt').open('w', encoding='utf-8') as log, redirect_stdout(log):
            value, _ = module.process_temporal_adaptive(raw['x_L'], raw['y_L'], raw['t'], raw['disp'],
                                                       SimpleNamespace(**contract['temporal_parameters']))
        assert len(value) == len(raw['disp']) and np.isfinite(value).all()
        replay[label] = value
        scored = association['raw'].copy()
        scored['prediction'] = value[scored['source']]
        results.setdefault('paired_temporal_replay', {})[label] = dict(
            input_rows=len(value), changed_rows_from_matcher=int(np.count_nonzero(value != raw['disp'])),
            native=describe(scored), source_code_sha256=sha256(path),
            invocation_seconds=time.perf_counter() - started,
            timing_scope='single diagnostic invocation; not an end-to-end benchmark')
        print('Replayed', label, 'on', len(value), 'input rows.', flush=True)
    np.savez_compressed(out / 'paired_temporal_disparities.npz', **replay)
    results['manifest'] = dict(inputs=inputs, python=sys.version, numpy=np.__version__, platform=platform.platform(),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT / 'src/evaluation.py']})
    (out / 'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
    save_output_hashes(out)
    print(json.dumps({'partitions': results['partition'], 'paired_replay': results['paired_temporal_replay']}, indent=2))


if __name__ == '__main__':
    main()
