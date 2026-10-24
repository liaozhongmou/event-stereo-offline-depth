import numpy as np
import pytest

from diagnostics.review2_analysis import boundary_mask, depth_summary, new_support_mask


def test_new_support_is_a_location_partition_even_when_shared_values_change():
    # A completed estimate at unit 7 can differ from the temporal estimate;
    # its location still belongs to shared support, not newly reached support.
    mask = new_support_mask(np.array([7, 12]), np.array([3, 7, 12, 20]))
    np.testing.assert_array_equal(mask, [True, False, False, True])


def test_metric_depth_reports_nonpositive_predictions_without_silent_clipping():
    result = depth_summary(np.array([5., 0., -1., 10.]), np.full(4, 10.), 20.)
    assert result['units'] == 4 and result['positive_prediction_units'] == 2
    assert result['nonpositive_prediction_units'] == 2
    assert result['mae_m'] == pytest.approx(1.)
    assert result['rmse_m'] == pytest.approx(np.sqrt(2.))


def test_boundary_definition_uses_finite_neighbors_and_marks_both_sides():
    gt = np.array([[10., np.nan, 12.], [10., 10., 12.]])
    np.testing.assert_array_equal(boundary_mask(gt, radius=0),
                                  [[False, False, False], [False, True, True]])
    assert boundary_mask(gt, radius=1).all()
