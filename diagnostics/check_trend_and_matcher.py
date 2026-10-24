"""Synthetic software diagnostics; no real-event or accuracy evaluation."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
os.environ["NUMBA_DISABLE_JIT"] = "1"
ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Original = load("original_trend", ROOT / "original/Adapative_Trend_Fitter.py").PixelTrendTracker
CASES = [
    ("rising", np.arange(12.0), 1.0, 10.0, 20),
    ("falling_window", np.arange(12.0), -0.5, 20.0, 5),
    ("offset_window", 100.0 + np.arange(12.0) * 0.1, 2.0, 10.0, 5),
    ("irregular_window", np.array([0, .1, .4, .9, 1.5, 2.8, 3.0, 4.6]), 3.0, 5.0, 5),
]


def linear_case(tracker_class, timestamps, slope, intercept, window_size=20):
    tracker = tracker_class(window_size=window_size)
    expected = slope * (timestamps - timestamps[0]) + intercept
    rows = []
    for t, value in zip(timestamps, expected):
        fitted, info = tracker.add_value(float(t), float(value))
        rows.append({"t": float(t), "expected": float(value), "output": float(fitted),
                     "predicted": info["predicted"], "status": info["status"]})
    return rows, tracker


def regression(tracker_class):
    for _, t, slope, intercept, window in CASES:
        rows, tracker = linear_case(tracker_class, t, slope, intercept, window)
        np.testing.assert_allclose([r["output"] for r in rows], [r["expected"] for r in rows], atol=1e-8)
        assert all(r["status"] == "normal" for r in rows[5:])
        assert tracker.correction_count == tracker.refit_count == 0


def matcher_case(stereo, Lx, Lt, Rx, Rt, row_k, cross_eps):
    Lx = np.array(Lx, dtype=np.int32)
    Lt = np.array(Lt, dtype=np.float64)
    Rx = np.array(Rx, dtype=np.int32)
    Rt = np.array(Rt, dtype=np.float64)
    Ly = np.zeros(Lx.size, dtype=np.int32)
    Lp = np.ones(Lx.size, dtype=np.int8)
    Rp = np.ones(Rx.size, dtype=np.int8)
    Rptr = np.array([0, Rx.size], dtype=np.int64)
    Lptr = np.array([0, Lx.size], dtype=np.int64)
    records = []
    source = (ROOT / "original/Event_SNN_Stereo.py").read_text(encoding="utf-8")
    gate_line = next(i for i, line in enumerate(source.splitlines(), 1) if "if (resid_norm <=" in line)

    def trace(frame, event, arg):
        if frame.f_code.co_name == "match_v2_adaptive" and event == "line" and frame.f_lineno == gate_line:
            fields = ("tL", "xL", "a", "b", "xR_peak", "t_anchor", "a2", "b2", "cc_pen", "score", "resid_norm")
            records.append({key: float(frame.f_locals[key]) for key in fields})
        return trace

    sys.settrace(trace)
    result = stereo.match_v2_adaptive(
        Lx, Ly, Lt, Lp, Rx, Rt, Rp, Rptr, Lx, Lt, Lp, Lptr,
        100, 1, 1.0, 30.0, .3, .3, 1.6, 2, 1.0, 1,
        3.0, 3, 2, 1.0, 1.0, 2.0, 1.0,
        0, np.zeros((1, 100)), np.zeros((1, 100)), .5, 3.0, 1.0,
        cross_eps, 1, 100.0, -100.0, row_k)
    sys.settrace(None)
    return {"disp": result[3].tolist(), "score": result[4].tolist(), "gates": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regression", choices=("original", "fixed"))
    args = parser.parse_args()
    if args.regression:
        cls = Original if args.regression == "original" else load("fixed_trend", ROOT / "diagnostics/trend_time_basis_fixed.py").PixelTrendTracker
        regression(cls)
        print("PASS: four exact linear trajectories, including sliding window and irregular times")
        return

    rows, tracker = linear_case(Original, np.arange(12.0), 1.0, 10.0)
    np.testing.assert_allclose([r["output"] for r in rows], [10, 11, 12, 13, 14, 14, 14, 17, 17, 17, 20, 20], atol=1e-8)
    assert rows[5]["status"] == "outlier_corrected"
    assert tracker.correction_count == 5 and tracker.refit_count == 2
    constant, _ = linear_case(Original, np.arange(12.0), 0.0, 10.0)
    np.testing.assert_allclose([r["output"] for r in constant], 10.0, atol=1e-8)
    print(json.dumps({"linear_original": rows, "original_stats": tracker.get_stats()}, indent=2))
    fixed = load("fixed_trend", ROOT / "diagnostics/trend_time_basis_fixed.py").PixelTrendTracker
    metrics = []
    for name, t, slope, intercept, window in CASES:
        for implementation, cls in [("original", Original), ("fixed", fixed)]:
            sequence, state = linear_case(cls, t, slope, intercept, window)
            error = np.abs([r["output"] - r["expected"] for r in sequence])
            metrics.append({"case": name, "implementation": implementation,
                            "max_abs_error": float(error.max()), "MAE": float(error.mean()),
                            "corrections": state.correction_count, "refits": state.refit_count})
    fixed_constant, _ = linear_case(fixed, np.arange(12.0), 0.0, 10.0)
    assert [r["status"] for r in constant] == [r["status"] for r in fixed_constant]
    np.testing.assert_allclose([r["output"] for r in constant], [r["output"] for r in fixed_constant])
    print(json.dumps({"linear_metrics": metrics}, indent=2))
    stereo = load("original_stereo", ROOT / "original/Event_SNN_Stereo.py")
    raw = matcher_case(stereo, [30, 50], [1, 2], [20, 20, 30, 30], [.999, 1.001, 1.999, 2.001], 0, 5)
    median = matcher_case(stereo, [30, 50], [1, 2], [20, 20, 30, 30], [.999, 1.001, 1.999, 2.001], 10, 5)
    np.testing.assert_allclose(raw["disp"], [10, 20])
    np.testing.assert_allclose(median["disp"], [10, 18.5])
    np.testing.assert_array_equal(raw["score"], median["score"])
    anchor = matcher_case(stereo, [40, 41, 42], [1, 1.1, 1.2], [31, 32], [1.1, 1.2], 0, 5)
    first = anchor["gates"][0]
    np.testing.assert_allclose(first["b"], 30, atol=1e-8)
    np.testing.assert_allclose(first["a2"], 10, atol=1e-8)
    np.testing.assert_allclose(first["cc_pen"], 10 * (first["t_anchor"] - 1), atol=1e-8)
    assert first["cc_pen"] > 1
    narrow = matcher_case(stereo, [40, 41, 42], [1, 1.1, 1.2], [31, 32], [1.1, 1.2], 0, 1)
    assert len(anchor["disp"]) == 3 and len(narrow["disp"]) == 2
    print(json.dumps({"before_median": raw, "with_median": median, "anchor": anchor,
                      "anchor_cross_eps_1": narrow}, indent=2))
    print("PASS: original behavior reproduced; scores precede median; asynchronous cross-check penalty reproduced")


if __name__ == "__main__":
    main()
