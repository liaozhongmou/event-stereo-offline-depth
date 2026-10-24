import numpy as np
import pytest

from src.evaluation import associate, common_support, frame_bootstrap, frame_metrics, metrics, select


def pred(x, t, d):
    return dict(x_L=np.array(x), y_L=np.zeros(len(x), dtype=int),
                t=np.array(t, dtype=float), disp=np.array(d, dtype=float))


def test_nearest_duplicate_is_selected_without_reading_error():
    gt = np.full((1, 1, 2), 10.)
    a = associate(pred([0, 0, 1], [.001, .002, .001], [30, 10, 10]),
                  gt, np.array([0.]), (0., .01))
    assert a['source'].tolist() == [0, 2]
    assert metrics(a)['coverage'] == 1
    assert metrics(a)['mae'] == 10


def test_equal_distance_uses_first_source_and_earlier_frame():
    a = associate(pred([0, 0], [.01, .01], [30, 10]),
                  np.full((2, 1, 1), 10.), np.array([0., .02]), (0., .02))
    assert a['frame'].tolist() == [0]
    assert a['source'].tolist() == [0]


def test_tolerance_boundary_invalid_gt_and_outside_predictions():
    gt = np.array([[[4., 28., 0., np.nan]]])
    a = associate(pred([0, 1, 2, 3], [.015, .0150001, 0., 0.], [-10, 50, 10, 10]),
                  gt, np.array([0.]), (0., .02))
    assert a['prediction'].tolist() == [-10]
    assert a['denominator'] == 2
    assert metrics(a)['mae'] == 14


def test_common_is_exact_frame_pixel_intersection_and_fixed_denominator():
    gt = np.full((1, 1, 3), 10.)
    a = associate(pred([0, 1], [0, 0], [10, 20]), gt, np.array([0.]), (0, 0))
    b = associate(pred([1, 2], [0, 0], [15, 10]), gt, np.array([0.]), (0, 0))
    ca, cb = common_support([a, b])
    assert ca['unit'].tolist() == cb['unit'].tolist() == [1]
    assert metrics(ca)['mae'] == 10
    assert metrics(cb)['coverage'] == 1 / 3


def test_unordered_times_are_associated_without_mutating_source_order():
    a = associate(pred([0, 0], [1, 0], [20, 10]), np.full((2, 1, 1), 10.),
                  np.array([0., 1.]), (0, 1))
    assert a['source'].tolist() == [1, 0]


@pytest.mark.parametrize('bad', ['nan_time', 'fractional_x', 'outside_x', 'length'])
def test_invalid_input_fails_explicitly(bad):
    p = pred([0], [0], [10])
    if bad == 'nan_time':
        p['t'][0] = np.nan
    elif bad == 'fractional_x':
        p['x_L'] = np.array([.5])
    elif bad == 'outside_x':
        p['x_L'][0] = 1
    else:
        p['disp'] = np.array([])
    with pytest.raises(ValueError):
        associate(p, np.full((1, 1, 1), 10.), np.array([0.]), (0, 0))


def test_nan_disparity_abstains_and_empty_frames_stay_in_denominator():
    a = associate(pred([0, 0], [0, 1], [np.nan, 10]), np.full((3, 1, 1), 10.),
                  np.array([0., 1., 2.]), (0, 2))
    assert metrics(a)['coverage'] == 1 / 3
    assert a['frames'].tolist() == [0, 1, 2]
    empty = associate(pred([], [], []), np.full((1, 1, 1), 10.), np.array([0.]), (0, 0))
    assert metrics(empty)['mae'] is None


def test_bad_thresholds_are_strict_and_gt_time_order_is_validated():
    gt = np.full((1, 1, 3), 10.)
    a = associate(pred([0, 1, 2], [0, 0, 0], [11, 12, 13]), gt, np.array([0.]), (0, 0))
    m = metrics(a)
    assert m['bad1'] == 2 / 3
    assert m['bad2'] == 1 / 3
    assert m['bad3'] == 0
    with pytest.raises(ValueError):
        associate(pred([], [], []), np.full((2, 1, 1), 10.), np.array([1., 0.]), (0, 1))


def test_common_rejects_different_gt_with_equal_valid_counts():
    p = pred([0], [0], [10])
    a = associate(p, np.full((1, 1, 1), 10.), np.array([0.]), (0, 0))
    b = associate(p, np.full((1, 1, 1), 11.), np.array([0.]), (0, 0))
    with pytest.raises(ValueError, match='ground truth'):
        common_support([a, b])


def test_common_rejects_different_gt_population_outside_common_units():
    p = pred([0], [0], [10])
    a = associate(p, np.array([[[10., 10., np.nan]]]), np.array([0.]), (0, 0))
    b = associate(p, np.array([[[10., np.nan, 10.]]]), np.array([0.]), (0, 0))
    with pytest.raises(ValueError, match='ground truth'):
        common_support([a, b])


def test_rejection_preserves_denominator_and_empty_frame_positions():
    a = associate(pred([0, 0], [0, 2], [10, 12]), np.full((3, 1, 1), 10.),
                  np.array([0., 1., 2.]), (0, 2))
    reduced = select(a, np.array([True, False]))
    assert reduced['denominator'] == a['denominator'] == 3
    f = frame_metrics(a)
    assert f['units'].tolist() == [1, 0, 1]
    assert np.isnan(f['mae'][1])
    bootstrap = frame_bootstrap(a, repeats=20)
    assert bootstrap['mean'] == 1
    assert bootstrap['ci95'] == [1, 1]


def test_nearest_frame_outside_interval_is_not_reassigned_to_inside_frame():
    a = associate(pred([0], [.009], [10]), np.full((2, 1, 1), 10.),
                  np.array([0., .02]), (.005, .025))
    assert len(a['unit']) == 0
    assert a['denominator'] == 1


def test_epoch_tolerance_uses_stored_float_distance_without_added_epsilon():
    epoch = 1504645177.
    rounded_up = epoch + .015
    rounded_down = np.nextafter(rounded_up, -np.inf)
    assert rounded_up - epoch > .015
    assert rounded_down - epoch < .015
    a = associate(pred([0, 1], [rounded_up, rounded_down], [10, 10]),
                  np.full((1, 1, 2), 10.), np.array([epoch]), (epoch, epoch + .02))
    assert a['source'].tolist() == [1]
