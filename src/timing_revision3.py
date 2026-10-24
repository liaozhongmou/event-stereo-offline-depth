"""Five separately timed complete invocations per output mode; sampled peak RSS."""
from pathlib import Path
from functools import partial
import argparse
import json
import os
import subprocess
import sys
import multiprocessing as mp
import time
import numpy as np
import psutil
from src.run_revision3 import ROOT,PARAMS,events,filtered,matching,dump
from src.stereo_controls import match_v2_adaptive
from src.revision3 import complete_events,temporal_median,median_grid
from src.run_historical import sha256


def sample_process(pid,stop,ready,path):
    process=psutil.Process(pid)
    resident=[process.memory_info().rss]
    ready.set()
    while not stop.wait(.02):
        resident.append(process.memory_info().rss)
    dump(path,dict(peak_rss_mib=max(resident)/2**20,samples=len(resident)))


def invocation(mode,out):
    out.mkdir(parents=True,exist_ok=False)
    kernel=partial(match_v2_adaptive,same_time=1)
    # Identical dtypes and geometry to the measured workload; warm all used kernels.
    tiny=(np.array([20,21,22],np.int32),np.zeros(3,np.int32),
          np.array([1.,1.01,1.02]),np.ones(3,np.int8))
    filtered(tiny,(260,346));matching(tiny,tiny,(260,346),PARAMS,kernel=kernel)
    median_grid(tiny[0],tiny[1],tiny[2],260,346)
    median_grid(tiny[0],tiny[1],tiny[2].astype(np.float32),260,346)
    temporal_median(tiny[0],tiny[1],tiny[2],np.array([10.,10.,10.],np.float32))
    context=mp.get_context('spawn');stop=context.Event();ready=context.Event()
    worker=context.Process(target=sample_process,args=(os.getpid(),stop,ready,out/'rss.json'))
    worker.start();ready.wait()
    begin=time.perf_counter()
    pair=[events(ROOT.parent/f'indoor1_data/indoor1_{side}1.npz') for side in ['left','right']]
    selected=[filtered(v,(260,346)) for v in pair]
    pred=matching(*selected,(260,346),PARAMS,kernel=kernel)
    if mode=='median':
        pred['disp']=temporal_median(pred['x_L'],pred['y_L'],pred['t'],pred['disp'])
    if mode=='completed':
        interval=[max(pair[0][2][0],pair[1][2][0]),min(pair[0][2][-1],pair[1][2][-1])]
        pred=complete_events(pred,(260,346),interval)
    np.savez_compressed(out/'prediction.npz',**pred)
    seconds=time.perf_counter()-begin
    stop.set();worker.join();assert worker.exitcode==0
    rss=json.loads((out/'rss.json').read_text())
    dump(out/'timing.json',dict(mode=mode,seconds=seconds,peak_rss_mib=rss['peak_rss_mib'],rss_samples=rss['samples'],
        scope='load NPZ + LIF + CSR + matching + selected output mode + compressed write',
        excluded='imports, JIT warm-up, SHA-256 and evaluation',rows=len(pred['t']),
        thread_settings={k:os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS','NUMBA_NUM_THREADS']},python=sys.version,
        output_sha256=sha256(out/'prediction.npz')))
    print(mode,seconds,rss['peak_rss_mib'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','matcher','median','completed'],required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.mode!='all':
        invocation(a.mode,a.out)
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        records=[]
        for repeat in range(5):
            for mode in ['matcher','median','completed']:
                path=a.out/f'{repeat+1:02d}_{mode}'
                subprocess.run([sys.executable,'-m','src.timing_revision3','--mode',mode,'--out',str(path)],
                               cwd=ROOT,check=True)
                records.append(json.loads((path/'timing.json').read_text()))
                dump(a.out/'records.json',records)
        dump(a.out/'summary.json',{mode:dict(
            seconds_quantiles=np.quantile([r['seconds'] for r in records if r['mode']==mode],[.25,.5,.75]).tolist(),
            peak_rss_mib_quantiles=np.quantile([r['peak_rss_mib'] for r in records if r['mode']==mode],[.25,.5,.75]).tolist())
            for mode in ['matcher','median','completed']})
