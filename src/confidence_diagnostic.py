"""Fixed-unit confidence diagnostic; not a deployment rejection evaluation."""
import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation import associate, metrics, select
from src.run_historical import load_prediction, save_output_hashes, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    paths = [root / f'new_methods/new_methods_results_indoor{i}/matches{number}.npz'
             for i, number in [(1, 9), (2, 10), (3, 11)]]
    ratios = [1., .5, .25, .1, .05]
    with np.load(paths[0], allow_pickle=False) as z:
        confidence = {field: z[field] for field in ('score', 'resid')}
    if not all(np.isfinite(a).all() for a in confidence.values()):
        raise ValueError('confidence fields must be finite')
    thresholds = {field: [None if ratio == 1 else float(np.quantile(a, 1-ratio if field == 'score' else ratio))
                          for ratio in ratios] for field, a in confidence.items()}
    frozen = dict(thresholds=thresholds, ratios=ratios, fit_prediction_sha256=sha256(paths[0]),
                  seed=20261020, random_repeats=100,
                  full_population_control='one deterministic replicate; no subsampling at ratio=1',
                  population='fixed unique units after time-only association',
                  stage='development diagnostic on saved matcher output')
    # This file is written before loading any GT in this process.
    (args.out / 'thresholds_frozen.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    rng = np.random.default_rng(frozen['seed'])
    records, repeats, manifests = [], [], []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    started = time.perf_counter()
    for sequence, path in enumerate(paths, 1):
        prediction = load_prediction(path)
        with np.load(path, allow_pickle=False) as z:
            confidence = {field: z[field] for field in ('score', 'resid')}
        if not all(np.isfinite(a).all() for a in confidence.values()):
            raise ValueError('confidence fields must be finite')
        gt_path = root / f'indoor{sequence}_data/indoor{sequence}_gt_disp.npz'
        with np.load(gt_path, allow_pickle=False) as z:
            gt, ts = z['disp'] * np.float32(.8819476), z['ts']
        interval = (max(float(ts[0]), float(prediction['t'].min())),
                    min(float(ts[-1]), float(prediction['t'].max())))
        a = associate(prediction, gt, ts, interval)
        del gt
        manifests.append(dict(sequence=sequence, interval=list(interval),
                              prediction_sha256=sha256(path), gt_sha256=sha256(gt_path)))
        for field in ('score', 'resid'):
            curve = []
            values = confidence[field][a['source']]
            for ratio, threshold in zip(ratios, thresholds[field]):
                keep = np.ones(len(values), bool) if threshold is None else (
                    values >= threshold if field == 'score' else values <= threshold)
                subset = select(a, keep)
                row = dict(sequence=sequence, field=field, training_fraction=ratio,
                           threshold=threshold, **metrics(subset))
                random_results = []
                for rep in range(1 if ratio == 1 else frozen['random_repeats']):
                    chosen = rng.choice(len(values), size=int(keep.sum()), replace=False)
                    random_results.append(metrics(select(a, chosen)))
                row['random_control'] = {key: np.quantile([m[key] for m in random_results], [.025, .5, .975]).tolist()
                                         for key in ('mae', 'bad1', 'bad2', 'bad3', 'p90', 'correct3_all_gt')
                                         if all(m[key] is not None for m in random_results)}
                row['random_control_replicates'] = len(random_results)
                repeats.append(dict(sequence=sequence, field=field, training_fraction=ratio,
                                    metrics=random_results))
                records.append(row)
                curve.append(row)
            color = 'C0' if field == 'score' else 'C1'
            axes[sequence-1].plot([100*r['coverage'] for r in curve], [r['mae'] for r in curve],
                                  'o-', color=color, label=field)
            axes[sequence-1].plot([100*r['coverage'] for r in curve],
                                  [r['random_control']['mae'][1] for r in curve], '--', color=color,
                                  alpha=.55, label=f'{field}: equal-count random')
            axes[sequence-1].fill_between([100*r['coverage'] for r in curve],
                    [r['random_control']['mae'][0] for r in curve],
                    [r['random_control']['mae'][2] for r in curve], color=color, alpha=.12)
        axes[sequence-1].set(title=f'Indoor {sequence} (development)', xlabel='All valid GT covered (%)',
                             ylabel='Native MAE (px)')
        axes[sequence-1].grid(alpha=.2)
        print('confidence sequence', sequence, 'units', len(a['unit']), flush=True)
    axes[0].legend(fontsize=7)
    fig.savefig(args.out / 'error_coverage.png', dpi=180)
    fig.savefig(args.out / 'error_coverage.pdf')
    plt.close(fig)
    result = dict(protocol=frozen, results=records, inputs=manifests,
                  command=sys.argv, elapsed_seconds=time.perf_counter()-started,
                  python=sys.version, numpy=np.__version__, matplotlib=matplotlib.__version__,
                  platform=platform.platform(),
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                           cwd=Path(__file__).resolve().parents[1], text=True).strip(),
                  code={str(p):sha256(p) for p in Path(__file__).parent.glob('*.py')})
    (args.out / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    (args.out / 'random_replicates.json').write_text(json.dumps(repeats, allow_nan=False), encoding='utf-8')
    columns = ['sequence', 'field', 'training_fraction', 'threshold', 'units', 'valid_gt_units',
               'coverage', 'mae', 'rmse', 'median', 'p90', 'bad1', 'bad2', 'bad3',
               'correct1_all_gt', 'correct2_all_gt', 'correct3_all_gt']
    with (args.out / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    save_output_hashes(args.out)


if __name__ == '__main__':
    main()
