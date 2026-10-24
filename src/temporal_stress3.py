"""Controlled temporal-state audit; reported separately from real-data accuracy."""
from pathlib import Path
import json
import numpy as np
from original.Adapative_Trend_Fitter import PixelTrendTracker as Legacy
from diagnostics.trend_time_basis_fixed import PixelTrendTracker as Consistent
from src.revision3 import temporal_median
from src.run_revision3 import dump

OUT=Path(__file__).resolve().parents[1]/'runs/C3_temporal_stress_20261021'
OUT.mkdir(exist_ok=False)
rng=np.random.default_rng(20261021)
t=np.arange(40,dtype=float)*.05
truths=dict(constant=np.full(40,12.),linear=8+4*t,accelerated=8+2*t+t*t,
            step=np.where(t<1,10.,20.),noise=np.full(40,12.))
results={}
for name,truth in truths.items():
    observation=truth+(rng.normal(0,.5,40) if name=='noise' else 0)
    outputs=dict(bypass=observation.copy(),median5=temporal_median(np.zeros(40,np.int32),
        np.zeros(40,np.int32),t,observation,window=5,gap=.3))
    for key,klass in [('legacy',Legacy),('consistent_basis',Consistent)]:
        tracker=klass(window_size=20,deviation_threshold=2,outlier_tolerance=3,trend_degree=1,min_samples=4)
        outputs[key]=np.array([tracker.add_value(float(ti),float(di))[0] for ti,di in zip(t,observation)])
    results[name]={key:dict(mae=float(np.abs(value-truth).mean()),max_error=float(np.abs(value-truth).max()),
        nonpositive=int(np.count_nonzero(value<=0)),values=value.tolist()) for key,value in outputs.items()}
gap_t=np.r_[np.arange(5)*.01,5.,5.01]
gap_d=np.r_[np.full(5,10.),20.,20.]
gap=temporal_median(np.zeros(7,np.int32),np.zeros(7,np.int32),gap_t,gap_d,window=5,gap=.3)
assert np.array_equal(gap,gap_d)
results['gap']=dict(timestamps=gap_t.tolist(),observed=gap_d.tolist(),median=gap.tolist(),
                    rule='No output is emitted during an interval without observations; reset after 0.3 s')
dump(OUT/'metrics.json',results)
