"""Check the stored depth/disparity value conversion over every development frame."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

from src.run_historical import sha256, save_output_hashes

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out
    out.mkdir(exist_ok=False)
    coefficient, multiplier = 19.941771812941038, .8819476
    previous = json.loads((ROOT / 'runs/R3_review_input_audit_20261021/audit.json').read_text())
    result = dict(scope='All same-index stored arrays; physical time alignment is not inferred',
                  coefficient_px_m=coefficient, disparity_multiplier=multiplier,
                  python=sys.version, numpy=np.__version__, script_sha256=sha256(Path(__file__)), sequences=[])
    for sequence in (1, 2, 3):
        paths = {kind: ROOT.parent / f'indoor{sequence}_data/indoor{sequence}_gt_{kind}.npz'
                 for kind in ('depth', 'disp')}
        source = previous['sequences'][sequence - 1]['inputs']
        for kind, path in paths.items():
            assert sha256(path) == source[kind]['sha256']
        with np.load(paths['depth']) as z:
            depth, depth_times = z['depth'], z['ts']
        with np.load(paths['disp']) as z:
            disparity, disparity_times = z['disp'], z['ts']
        assert depth.shape == disparity.shape
        count, eval_count, max_error, max_eval_error = 0, 0, 0., 0.
        product_min, product_max = float('inf'), -float('inf')
        checked_frames = 0
        for start in range(0, len(depth), 16):
            z, d = depth[start:start + 16], disparity[start:start + 16]
            valid = np.isfinite(z) & (z > 0) & np.isfinite(d) & (d > 0)
            checked_frames += int(np.any(valid, axis=(1, 2)).sum())
            zz, dd = z[valid].astype(float), d[valid].astype(float)
            product, converted = zz * dd, dd * multiplier
            error = np.abs(converted - coefficient / zz)
            evaluated = (converted >= 4) & (converted <= 28)
            count += len(zz)
            eval_count += int(evaluated.sum())
            if len(error):
                max_error = max(max_error, float(error.max()))
                product_min, product_max = min(product_min, float(product.min())), max(product_max, float(product.max()))
            if evaluated.any():
                max_eval_error = max(max_eval_error, float(error[evaluated].max()))
        item = dict(sequence=sequence, shape=list(depth.shape), array_frames=len(depth),
                    frames_with_positive_pairs=checked_frames, positive_value_pairs=count,
                    evaluated_range_pairs=eval_count, product_range_px_m=[product_min, product_max],
                    max_conversion_difference_px=max_error, max_evaluated_range_difference_px=max_eval_error,
                    depth_time_range=[float(depth_times[0]), float(depth_times[-1])],
                    disparity_time_range=[float(disparity_times[0]), float(disparity_times[-1])],
                    input_sha256={kind: source[kind]['sha256'] for kind in source})
        result['sequences'].append(item)
        del depth, disparity
        print(json.dumps(item), flush=True)
    (out / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    save_output_hashes(out)


if __name__ == '__main__':
    main()
