"""Prespecified LIF-rate and temporal-window sweeps on a fixed development clip."""
from pathlib import Path
from functools import partial
import argparse
import numpy as np
from src.run_revision3 import ROOT,PARAMS,events,matching,dump,_lif_kernel
from src.stereo_controls import match_v2_adaptive
from src.evaluation import associate,metrics
from src.run_historical import sha256,save_output_hashes

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=False)
    raw=[events(ROOT.parent/f'indoor1_data/indoor1_{side}1.npz') for side in ['left','right']]
    start=max(side[2][0] for side in raw)+5;interval=[start,start+12]
    raw=[tuple(v[side[2]<=interval[1]+.3] for v in side) for side in raw]
    cases={f'lif_{k:g}':dict(k_log=k) for k in [0.,1.2,2.4,3.6]}
    cases.update({f'window_{dt:g}':dict(k_log=2.4,dt=dt) for dt in [.005,.02,.05,.1]})
    dump(out/'contract.json',dict(interval=interval,cases=cases,
        context='LIF starts at recording start; +/-0.3 s matcher context; score central fixed12 s',
        source={str(path.relative_to(ROOT)):sha256(path) for path in [Path(__file__),ROOT/'src/stereo_controls.py']}))
    for name,case in cases.items():
        pair=[]
        for side in raw:
            filtered=_lif_kernel(*side,346,260,.7,.01,1.,1.05,1.05,1000.,case['k_log'],2.1)
            mask=(filtered[2]>=interval[0]-.3)&(filtered[2]<=interval[1]+.3)
            pair.append(tuple(v[mask] for v in filtered))
        params=PARAMS.copy()
        if 'dt' in case:params.update(dt_win_min=case['dt'],dt_win_max=case['dt'])
        pred=matching(*pair,(260,346),params,kernel=partial(match_v2_adaptive,same_time=1))
        np.savez_compressed(out/f'{name}.npz',**pred)
        print(name,len(pred['t']),flush=True)
    with np.load(ROOT.parent/'indoor1_data/indoor1_gt_disp.npz') as z:
        gt,ts=z['disp']*np.float32(.8819476),z['ts']
    results={}
    for name in cases:
        with np.load(out/f'{name}.npz') as z:pred={k:z[k] for k in z.files}
        results[name]=metrics(associate(pred,gt,ts,interval))
    dump(out/'metrics.json',results);save_output_hashes(out)
