"""Paired physical-range and reverse-consistency controls on frozen inputs."""
from functools import partial
from pathlib import Path
import argparse
import json
import time
import numpy as np

from src.run_revision3 import ROOT,PARAMS,events,filtered,matching,dump
from src.stereo_controls import match_v2_adaptive
from src.run_historical import sha256,save_output_hashes
from src.revision3 import complete_events


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--sequence',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=False)
    controls=dict(bounded=dict(cross_enabled=1,same_time=0),
                  same_time=dict(cross_enabled=1,same_time=1),
                  no_lr_true=dict(cross_enabled=0,same_time=0))
    folder=ROOT.parent/f'indoor{args.sequence}_data'
    inputs=[folder/f'indoor{args.sequence}_{side}1.npz' for side in ('left','right')]
    dump(out/'contract.json',dict(params=PARAMS,controls=controls,sequence=args.sequence,
        input_hashes={str(p):sha256(p) for p in inputs},
        source={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'src/stereo_controls.py',
            ROOT/'src/run_controls3.py',ROOT/'src/run_revision3.py',ROOT/'src/revision3.py',
            ROOT/'original/neuron_filter_vis.py',ROOT/'original/Event_SNN_Stereo.py']},
        selection='same_time is the selected consistent candidate before external GT; bounded and no_lr_true explanatory',
        geometry='evidence bins and emitted disparity within configured range'))
    left,right=[events(folder/f'indoor{args.sequence}_{side}1.npz') for side in ('left','right')]
    pair=[filtered(v,(260,346)) for v in (left,right)]
    times={}
    for name,kwargs in controls.items():
        start=time.perf_counter()
        pred=matching(*pair,(260,346),PARAMS,kernel=partial(match_v2_adaptive,**kwargs))
        assert ((pred['disp']>=4)&(pred['disp']<=28)).all()
        np.savez_compressed(out/f'{name}.npz',**pred)
        times[name]=dict(seconds=time.perf_counter()-start,rows=len(pred['t']))
        print(args.sequence,name,times[name],flush=True)
    interval=[max(left[2][0],right[2][0]),min(left[2][-1],right[2][-1])]
    with np.load(out/'same_time.npz') as z:
        main={k:z[k] for k in z.files}
    begin=time.perf_counter()
    completed=complete_events(main,(260,346),interval)
    np.savez_compressed(out/'completed_consistent.npz',**completed)
    times['completed_consistent']=dict(seconds=time.perf_counter()-begin,rows=len(completed['t']))
    dump(out/'inference.json',dict(interval=interval,shape=[260,346],variants=times))
    save_output_hashes(out)
