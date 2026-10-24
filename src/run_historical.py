"""Reconstruct the Indoor 1 historical table; never fit time offsets."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from src.evaluation import associate, common_support, frame_bootstrap, frame_metrics, metrics


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_prediction(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in ('x_L', 'y_L', 't', 'disp')}


def save_association(path, association):
    np.savez_compressed(path, **association)


def save_output_hashes(directory):
    records = {p.name: dict(bytes=p.stat().st_size, sha256=sha256(p))
               for p in sorted(directory.iterdir()) if p.is_file() and p.name != 'output_hashes.json'}
    (directory / 'output_hashes.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    workspace = Path(__file__).resolve().parents[2]
    gt_path = workspace / 'indoor1_data/indoor1_gt_disp.npz'
    sources = {name: workspace / f'new_methods/new_methods_results_indoor1/{file}'
               for name, file in [('raw', 'matches9.npz'), ('smooth', 'matches9_1.npz'),
                                  ('complete', 'matches9_2.npz')]}
    start = time.perf_counter()
    with np.load(gt_path, allow_pickle=False) as z:
        gt, ts = z['disp'] * np.float32(.8819476), z['ts']
    predictions = {name: load_prediction(path) for name, path in sources.items()}
    # Fix one shared interval before any association or metric calculation.
    interval = (max(float(ts[0]), *(float(p['t'].min()) for p in predictions.values())),
                min(float(ts[-1]), *(float(p['t'].max()) for p in predictions.values())))
    associations, results = {}, {}
    for name, path in sources.items():
        prediction = predictions[name]
        a = associate(prediction, gt, ts, interval)
        associations[name] = a
        results[name] = dict(interval=list(interval), native=metrics(a), frame=frame_bootstrap(a))
        save_association(args.out / f'{name}_associated.npz', a)
        np.savez_compressed(args.out / f'{name}_frames.npz', **frame_metrics(a))
        print(name, json.dumps(results[name]), flush=True)
    common = common_support(list(associations.values()))
    results['three_stage_common'] = {name: metrics(a) for name, a in zip(associations, common)}
    # Compare to the complete sequence-level row, at the printed precision.
    expected = dict(units=1058170, coverage=.01163, mae=3.322, rmse=4.141, bad3=.4838)
    observed = results['complete']['native']
    checks = dict(units=observed['units'] == expected['units'],
                  coverage=round(observed['coverage'] * 100, 3) == 1.163,
                  mae=round(observed['mae'], 3) == 3.322,
                  rmse=round(observed['rmse'], 3) == 4.141,
                  bad3=round(observed['bad3'] * 100, 2) == 48.38,
                  frame_mean=round(results['complete']['frame']['mean'], 3) == 4.561,
                  frame_ci=[round(v, 3) for v in results['complete']['frame']['ci95']]
                  == [4.345, 4.770])
    results['historical_check'] = dict(expected=expected, checks=checks, passed=all(checks.values()),
                                       evaluator='reconstructed-v1, not the original frozen evaluator')
    results['manifest'] = dict(command=sys.argv, python=sys.version, platform=platform.platform(),
                               numpy=np.__version__, elapsed_seconds=time.perf_counter() - start,
                               git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=Path(__file__).resolve().parents[1], text=True).strip(),
                               inputs={str(p): dict(bytes=p.stat().st_size, sha256=sha256(p))
                                       for p in [gt_path, *sources.values()]},
                               code={str(p): sha256(p) for p in Path(__file__).parent.glob('*.py')})
    (args.out / 'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
    save_output_hashes(args.out)
    print('HISTORICAL_CHECK', json.dumps(results['historical_check']), flush=True)


if __name__ == '__main__':
    main()
