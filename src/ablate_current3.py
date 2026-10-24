"""One-factor ablations of the corrected, bounded, same-time matcher."""
from functools import partial
from pathlib import Path
import argparse
import time
import numpy as np
from src.run_revision3 import ROOT,PARAMS,events,filtered,matching,dump
from src.stereo_controls import match_v2_adaptive
from src.run_historical import sha256,save_output_hashes

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=int,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    variants=dict(consistent_fixed_lif={},consistent_fixed50=dict(dt_win_min=.05,dt_win_max=.05),
                  consistent_ols=dict(irls_iters=0),consistent_no_row=dict(row_bimed_k=0))
    files=[ROOT.parent/f'indoor{args.sequence}_data/indoor{args.sequence}_{side}1.npz' for side in ['left','right']]
    dump(out/'contract.json',dict(params=PARAMS,sequence=args.sequence,variants=variants,same_time=1,
        input_hashes={str(f):sha256(f) for f in files},
        source={str(f.relative_to(ROOT)):sha256(f) for f in [ROOT/'src/stereo_controls.py',ROOT/'src/ablate_current3.py',
            ROOT/'src/run_revision3.py',ROOT/'original/neuron_filter_vis.py']}))
    raw=[events(f) for f in files];pair=[filtered(v,(260,346)) for v in raw];results={}
    for name,changes in variants.items():
        begin=time.perf_counter()
        inputs=[filtered(v,(260,346),True) for v in raw] if name=='consistent_fixed_lif' else pair
        pred=matching(*inputs,(260,346),PARAMS|changes,kernel=partial(match_v2_adaptive,same_time=1))
        assert ((pred['disp']>=4)&(pred['disp']<=28)).all()
        np.savez_compressed(out/f'{name}.npz',**pred)
        results[name]=dict(rows=len(pred['t']),seconds=time.perf_counter()-begin)
        print(args.sequence,name,results[name],flush=True)
    dump(out/'inference.json',results);save_output_hashes(out)
