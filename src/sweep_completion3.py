"""Prespecified completion cost-quality controls on a fixed 12 s development clip."""
from pathlib import Path
import argparse
import time
import numpy as np
from src.run_revision3 import ROOT,dump
from src.revision3 import complete_events,median_grid
from src.evaluation import associate,metrics,select
from src.run_historical import sha256,save_output_hashes


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=False)
    path=ROOT/'runs/C3_controls_v2_indoor1_20261021/same_time.npz'
    with np.load(path) as z:
        pred={k:z[k] for k in z.files}
    interval=[float(pred['t'][0])+5,float(pred['t'][0])+17]
    keep=(pred['t']>=interval[0])&(pred['t']<=interval[1])
    pred={k:v[keep] for k,v in pred.items()}
    changes=dict(default={},radius4=dict(radius=4,average_radius=5),radius12=dict(radius=12,average_radius=13),
        threshold10=dict(threshold=.1),threshold30=dict(threshold=.3),one_pass=dict(iterations=1),
        window50=dict(dt=.05),stride50=dict(stride=.05),local_mean=dict(threshold=0.,iterations=1))
    dump(out/'contract.json',dict(interval=interval,changes=changes,prediction_sha256=sha256(path),
        comparison='one completion parameter at a time; fixed timeline clip chosen without errors'))
    times={}
    median_grid(pred['x_L'][:10],pred['y_L'][:10],pred['disp'][:10],260,346)
    median_grid(pred['x_L'][:10],pred['y_L'][:10],pred['t'][:10],260,346)
    # Generate all predictions before loading any GT.
    for name,params in changes.items():
        start=time.perf_counter();result=complete_events(pred,(260,346),interval,**params)
        times[name]=time.perf_counter()-start
        np.savez_compressed(out/f'{name}.npz',**result)
        print(name,times[name],len(result['t']),flush=True)
    with np.load(ROOT.parent/'indoor1_data/indoor1_gt_disp.npz') as z:
        gt,ts=z['disp']*np.float32(.8819476),z['ts']
    base=associate(pred,gt,ts,interval);results={}
    for name in changes:
        with np.load(out/f'{name}.npz') as z:
            complete={k:z[k] for k in z.files}
        ass=associate(complete,gt,ts,interval)
        new=~np.isin(ass['unit'],base['unit'])
        results[name]=dict(seconds=times[name],native=metrics(ass),new=metrics(select(ass,new)),
            original=metrics(base),seconds_per_added_correct3=None)
        correct=int(np.count_nonzero(np.abs(ass['prediction'][new]-ass['gt'][new])<=3))
        results[name]['new_correct3_units']=correct
        results[name]['seconds_per_added_correct3']=times[name]/correct if correct else None
    dump(out/'metrics.json',results);save_output_hashes(out)
