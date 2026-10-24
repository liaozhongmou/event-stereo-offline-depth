"""Inspect archived GT scales and stage pairing without changing any input."""
from pathlib import Path
import ast
import argparse
import hashlib
import json
import re
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True)
OUT = parser.parse_args().out
OUT.mkdir(exist_ok=False)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stored_rows(path):
    with zipfile.ZipFile(path) as archive, archive.open('t.npy') as stream:
        version = np.lib.format.read_magic(stream)
        assert version == (1, 0), version
        shape, _, _ = np.lib.format.read_array_header_1_0(stream)
        return shape[0]


calib_path = WORKSPACE / 'camchain-imucam-indoor_flying.yaml'
text = calib_path.read_text()
left = text.split('cam0:', 1)[1].split('cam1:', 1)[0]
right = text.split('cam1:', 1)[1]
raw_fx = ast.literal_eval(re.search(r'intrinsics:\s*(\[[^\]]+\])', left)[1])[0]
rect_fx = ast.literal_eval(re.search(r'projection_matrix:\s*- (\[[^\]]+\])', left)[1])[0]
rect_fb = -ast.literal_eval(re.search(r'projection_matrix:\s*- (\[[^\]]+\])', right)[1])[3]
result = {'calibration': {'path': str(calib_path), 'sha256': digest(calib_path),
                         'raw_fx_px': raw_fx, 'rect_fx_px': rect_fx,
                         'rect_fb_px_m': rect_fb, 'focal_ratio': rect_fx / raw_fx,
                         'archived_disparity_multiplier': .8819476}, 'sequences': []}

for sequence in [1, 2, 3]:
    item = {'sequence': sequence, 'inputs': {}, 'GT_frame_index_samples': []}
    saved = {}
    for kind in ['depth', 'disp']:
        path = WORKSPACE / f'indoor{sequence}_data/indoor{sequence}_gt_{kind}.npz'
        with np.load(path, allow_pickle=False) as archive:
            times = archive['ts']
            array = archive[kind]
            indices = np.linspace(0, len(array) - 1, 5, dtype=int)
            saved[kind] = array[indices].copy()
            item['inputs'][kind] = {'path': str(path), 'sha256': digest(path),
                                   'shape': list(array.shape), 'sample_frame_indices': indices.tolist(),
                                   'sample_timestamps': times[indices].tolist()}
            del array
    depth, disp = saved['depth'], saved['disp']
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(disp) & (disp > 0)
    coefficient = depth[valid].astype(float) * disp[valid].astype(float)
    item['rowwise_depth_times_archived_disparity_quantiles'] = np.quantile(coefficient, [0, .05, .5, .95, 1]).tolist()
    item['rowwise_pairing_status'] = 'Same array indices only; physical frame correspondence is not established by these files.'
    for k, frame in enumerate(indices):
        locations = np.flatnonzero(valid[k])
        for flat in locations[np.linspace(0, len(locations) - 1, 3, dtype=int)]:
            y, x = np.unravel_index(flat, depth[k].shape)
            z, d = float(depth[k, y, x]), float(disp[k, y, x])
            item['GT_frame_index_samples'].append({'frame_index': int(frame), 'x': int(x), 'y': int(y),
                'stored_depth_value': z, 'stored_disparity_px': d, 'scaled_disparity_px': d * .8819476,
                'rect_fb_over_stored_depth_px': rect_fb / z})
    number = sequence + 8
    stage_dir = WORKSPACE / f'new_methods/new_methods_results_indoor{sequence}'
    item['stored_stage_rows'] = {stage: stored_rows(stage_dir / f'matches{number}{suffix}.npz')
                               for stage, suffix in [('raw', ''), ('smooth', '_1'), ('complete', '_2')]}
    with (np.load(WORKSPACE / f'new_methods/new_methods_results_indoor{sequence}/matches{number}.npz') as raw,
          np.load(WORKSPACE / f'new_methods/new_methods_results_indoor{sequence}/matches{number}_1.npz') as smooth):
        raw_t, smooth_t = raw['t'], smooth['t']
        item['stage_pairing'] = {'raw_rows': len(raw_t), 'smooth_rows': len(smooth_t),
            'raw_time_range': [float(raw_t.min()), float(raw_t.max())],
            'smooth_time_range': [float(smooth_t.min()), float(smooth_t.max())],
            'same_coordinates_and_time': {key: bool(np.array_equal(raw[key], smooth[key])) for key in ['x_L', 'y_L', 't']}}
    result['sequences'].append(item)
    print('audited sequence', sequence, item['rowwise_depth_times_archived_disparity_quantiles'], item['stage_pairing'], flush=True)
archived_coefficient = float(np.median([s['rowwise_depth_times_archived_disparity_quantiles'][2] for s in result['sequences']]))
result['calibration'].update({
    'effective_archived_coefficient_px_m': archived_coefficient,
    'rect_over_archived_coefficient': rect_fb / archived_coefficient,
    'max_sample_disparity_difference_px': max(abs(p['scaled_disparity_px'] - p['rect_fb_over_stored_depth_px'])
        for s in result['sequences'] for p in s['GT_frame_index_samples'])})
(OUT / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
