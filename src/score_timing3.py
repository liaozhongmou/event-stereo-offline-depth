"""Score the measured path outputs, checking repeated invocations agree."""
import json
import numpy as np
from src.run_revision3 import ROOT,dump
from src.evaluation import associate,metrics

directory=ROOT/'runs/C3_joint_timing_20261021'
with np.load(ROOT.parent/'indoor1_data/indoor1_gt_disp.npz') as z:
    gt,ts=z['disp']*np.float32(.8819476),z['ts']
result={}
for mode in ['matcher','median','completed']:
    rows=[]
    for repeat in range(1,6):
        path=directory/f'{repeat:02d}_{mode}'
        with np.load(path/'prediction.npz') as z:
            pred={k:z[k] for k in ['x_L','y_L','t','disp']}
        a=associate(pred,gt,ts,[float(ts[0]),float(ts[-1])])
        rows.append(metrics(a))
    assert all(v==rows[0] for v in rows[1:]),mode
    result[mode]=dict(metrics=rows[0],five_repeats_identical_metrics=True)
dump(directory/'quality.json',result)
print(json.dumps(result,indent=2))
