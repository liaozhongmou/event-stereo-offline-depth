import numpy as np
import pytest

from src.revision3 import (box_sum, complete_grid, temporal_median, rank_subset,
                           median_grid, complete_events, nearest_time_match)


@pytest.mark.parametrize('radius', [0, 1, 2])
def test_box_sum_is_centered_including_edges(radius):
    image = np.arange(20).reshape(4, 5)
    expected = np.array([[image[max(0,y-radius):y+radius+1,
                               max(0,x-radius):x+radius+1].sum()
                          for x in range(5)] for y in range(4)])
    np.testing.assert_array_equal(box_sum(image, radius), expected)


def test_completion_preserves_sources_and_only_fills_supported_neighborhood():
    grid = np.full((7, 7), np.nan)
    grid[3, 3] = 12
    out = complete_grid(grid, radius=1, threshold=.1, iterations=1)
    np.testing.assert_array_equal(out[2:5, 2:5], np.full((3, 3), 12))
    assert np.isfinite(out).sum() == 9
    assert out[3, 3] == grid[3, 3]


def test_temporal_median_resets_after_gap_and_keeps_pixels_separate():
    x = np.array([0, 1, 0, 0, 0])
    y = np.zeros(5, int)
    t = np.array([0., .01, .02, .03, 2.])
    d = np.array([10., 20., 12., 14., 8.])
    np.testing.assert_allclose(temporal_median(x,y,t,d,window=3,gap=.3),
                               [10,20,11,12,8])


def test_equal_count_selection_uses_score_and_source_order_without_gt():
    np.testing.assert_array_equal(rank_subset(np.array([2.,5.,5.,1.]), 2), [1,2])


def test_pixel_median_merge_retains_empty_pixels():
    grid=median_grid(np.array([1,1,2]),np.array([0,0,1]),np.array([8.,12.,20.]),2,3)
    assert grid[0,1]==10
    assert grid[1,2]==20
    assert np.isfinite(grid).sum()==2


def test_nearest_time_obeys_row_polarity_and_positive_disparity():
    left=(np.array([20],np.int32),np.array([0],np.int32),np.array([1.]),np.array([1],np.int8))
    right=(np.array([10,9,8],np.int32),np.array([0,0,1],np.int32),
           np.array([.999,1.,1.]),np.array([1,-1,1],np.int8))
    result=nearest_time_match(left,right,30,2,.01)
    np.testing.assert_allclose(result[3],[10])


def test_completion_rows_keep_originals_and_record_interpolation_kind():
    prediction=dict(x_L=np.array([3],np.int32),y_L=np.array([3],np.int32),
                    t=np.array([.05]),disp=np.array([12.],np.float32))
    out=complete_events(prediction,(7,7),[0,.1],radius=1,threshold=.1,iterations=1,average_radius=1)
    assert (out['kind']==0).sum()==1
    assert (out['kind']==1).sum()==8
    np.testing.assert_allclose(out['disp'],12.)
    np.testing.assert_allclose(out['t'],.05)


def test_bounded_matcher_never_emits_out_of_search_range():
    from src.run_revision3 import matching, PARAMS
    from src.stereo_controls import match_v2_adaptive
    rng=np.random.default_rng(42)
    left=(rng.integers(20,50,500,dtype=np.int32),np.zeros(500,np.int32),
          np.linspace(0,1,500),rng.choice(np.array([-1,1],np.int8),500))
    right=(rng.integers(1,45,800,dtype=np.int32),np.zeros(800,np.int32),
           np.linspace(0,1,800),rng.choice(np.array([-1,1],np.int8),800))
    pred=matching(left,right,(1,60),PARAMS|dict(cross_eps=100.),kernel=match_v2_adaptive)
    assert len(pred['disp'])>0
    assert ((pred['disp']>=4)&(pred['disp']<=28)).all()


def test_external_all_positive_gt_retains_out_of_search_gt_errors():
    from src.evaluation import associate,metrics
    gt=np.array([[[2.,10.,40.]]]);ts=np.array([0.])
    pred=dict(x_L=np.array([0,1,2]),y_L=np.zeros(3,int),t=np.zeros(3),disp=np.array([4.,10.,28.]))
    a=associate(pred,gt,ts,[0,0],gt_range=(np.finfo(np.float32).tiny,np.inf))
    assert metrics(a)['valid_gt_units']==3
    assert metrics(a)['units']==3
    assert metrics(a)['mae']==pytest.approx(14/3)


def test_single_valid_evidence_bin_has_finite_ordinary_score():
    from src.run_revision3 import matching,PARAMS
    from src.stereo_controls import match_v2_adaptive
    left=(np.array([20,22,22],np.int32),np.zeros(3,np.int32),
          np.array([1.,1.001,1.00125]),np.ones(3,np.int8))
    right=(np.array([16,15],np.int32),np.zeros(2,np.int32),
           np.array([1.001,1.00125]),np.ones(2,np.int8))
    pred=matching(left,right,(1,40),PARAMS|dict(row_bimed_k=0),kernel=match_v2_adaptive)
    assert len(pred['score'])>0
    assert pred['score'].max()<100
