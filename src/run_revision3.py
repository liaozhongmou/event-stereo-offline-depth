"""Frozen full-recording inference and outcome-independent scoring."""
from pathlib import Path
import argparse
import json
import os
import platform
import subprocess
import sys
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
import numba
import numpy as np
import psutil

from original import Event_SNN_Stereo as stereo
from original.neuron_filter_vis import _lif_kernel
from src.evaluation import associate, common_support, frame_metrics, metrics, select
from src.revision3 import complete_events, temporal_median, rank_subset, nearest_time_match
from src.run_historical import sha256, save_output_hashes

ROOT = Path(__file__).resolve().parents[1]
PARAMS = dict(dt_win_min=.001,dt_win_max=.3,expand=1.5,target_support=5,tau_w=.05,
              huber_delta=2.8,irls_iters=3,bins_radius=4,sigma_x=6.,alpha_like=8.5,
              beta_conf=8.2,gamma_res=.08,cross_eps=3.,min_support=2,max_resid=20.,
              min_score=.05,row_bimed_k=10,disp_min=4,disp_max=28)
VARIANTS = {'full':{},'fixed_lif':{},'fixed50':dict(dt_win_min=.05,dt_win_max=.05),
            'ols':dict(irls_iters=0),'no_row':dict(row_bimed_k=0),
            'no_lr':dict(cross_eps=1e9)}


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def events(path):
    with np.load(path,allow_pickle=False) as z:
        p = z['polarity'].copy()
        if np.all(p >= 0):
            p = (2*p-1).astype(np.int8)
        arrays = (z['x'].astype(np.int32),z['y'].astype(np.int32),
                  z['t'].astype(np.float64),p.astype(np.int8))
    assert (np.diff(arrays[2])>=0).all()
    return arrays


def filtered(arrays, shape, fixed=False):
    h,w = shape
    return _lif_kernel(*arrays,w,h,.70,.01,1.,1.05,1.05,1000.,0. if fixed else 2.4,2.1)


def matching(left,right,shape,params,kernel=stereo.match_v2_adaptive):
    h,w = shape
    lx,ly,lt,lp = left
    rx,ry,rt,rp = right
    right_csr = stereo.build_rows_csr(rx,ry,rt,rp,h)
    left_csr = stereo.build_rows_csr(lx,ly,lt,lp,h)
    p = params
    result = kernel(*left,*right_csr,*left_csr,w,h,
        p['disp_min'],p['disp_max'],p['dt_win_min'],p['dt_win_max'],p['expand'],p['target_support'],
        p['tau_w'],0,p['huber_delta'],p['irls_iters'],p['bins_radius'],p['sigma_x'],
        p['alpha_like'],p['beta_conf'],p['gamma_res'],0,np.zeros(shape),np.zeros(shape,np.float32),
        .55,5.,.6,p['cross_eps'],p['min_support'],p['max_resid'],p['min_score'],p['row_bimed_k'])
    return dict(zip(('x_L','y_L','t','disp','score','resid','support'),result))


def infer(sequence,out):
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'contract.json',dict(sequence=sequence,params=PARAMS,variants=VARIANTS,
        status='NEW_UNIFIED_RUN_NOT_HISTORICAL_REPRODUCTION',temporal_default='bypass',
        score_scan=[.05,2.,5.,10.,20.],matched_count_fractions=[.1,.25,.5,.75,1.],
        interval='full intersection of raw left/right events; 0.3 s boundary margin',
        association_ms=15,gt_range_px=[4,28],completion='centered sum, median pixel merge',
        fixed_parameter_external='DSEC thun_00_a; no retuning',
        git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in [
            ROOT/'src/revision3.py',ROOT/'src/run_revision3.py',ROOT/'original/Event_SNN_Stereo.py',
            ROOT/'original/neuron_filter_vis.py',ROOT/'src/evaluation.py']}))
    start = time.perf_counter()
    folder = ROOT.parent/f'indoor{sequence}_data'
    paths = [folder/f'indoor{sequence}_{side}1.npz' for side in ('left','right')]
    left,right = [events(p) for p in paths]
    interval = [max(left[2][0],right[2][0])+.3,min(left[2][-1],right[2][-1])-.3]
    load_seconds = time.perf_counter()-start
    process = psutil.Process()
    run = dict(interval=interval,shape=[260,346],input_hashes={str(p):sha256(p) for p in paths},
               python=sys.version,numpy=np.__version__,numba=numba.__version__,
               platform=platform.platform(),load_seconds=load_seconds,variants={})
    # Trigger compilation on a tiny actual-event subset; measured separately.
    warm = time.perf_counter()
    small = [tuple(v[:min(100,len(v))] for v in side) for side in (left,right)]
    tiny = [filtered(v,(260,346)) for v in small]
    matching(*tiny,(260,346),PARAMS)
    run['jit_warmup_seconds'] = time.perf_counter()-warm
    begin = time.perf_counter()
    fl,fr = [filtered(v,(260,346)) for v in (left,right)]
    run['filter_seconds'] = time.perf_counter()-begin
    run['event_counts'] = dict(raw=[len(left[2]),len(right[2])],filtered=[len(fl[2]),len(fr[2])])
    for name,change in VARIANTS.items():
        begin = time.perf_counter()
        pair = [filtered(v,(260,346),True) for v in (left,right)] if name=='fixed_lif' else [fl,fr]
        pred = matching(*pair,(260,346),PARAMS|change)
        np.savez_compressed(out/f'{name}.npz',**pred)
        run['variants'][name] = dict(seconds=time.perf_counter()-begin,rows=len(pred['t']),
                                     process_rss_mib=process.memory_info().rss/2**20)
        print(sequence,name,run['variants'][name],flush=True)
        if name=='full':
            default = pred
    for name,dt in [('nearest50',.05),('nearest5',.005)]:
        begin=time.perf_counter()
        result=nearest_time_match(fl,fr,346,260,dt)
        pred=dict(zip(('x_L','y_L','t','disp','score'),result))
        np.savez_compressed(out/f'{name}.npz',**pred)
        run['variants'][name]=dict(seconds=time.perf_counter()-begin,rows=len(pred['t']))
        print(sequence,name,run['variants'][name],flush=True)
    begin=time.perf_counter()
    med = {k:default[k] for k in ('x_L','y_L','t','disp')}
    med['disp']=temporal_median(med['x_L'],med['y_L'],med['t'],med['disp'])
    np.savez_compressed(out/'median.npz',**med)
    run['variants']['median']=dict(seconds=time.perf_counter()-begin,rows=len(med['t']))
    begin=time.perf_counter()
    complete=complete_events(default,(260,346),interval)
    np.savez_compressed(out/'completed.npz',**complete)
    run['variants']['completed']=dict(seconds=time.perf_counter()-begin,rows=len(complete['t']))
    print(sequence,'completed',run['variants']['completed'],flush=True)
    dump(out/'inference.json',run)
    save_output_hashes(out)


def score(sequence,out):
    inference=json.loads((out/'inference.json').read_text())
    path=ROOT.parent/f'indoor{sequence}_data/indoor{sequence}_gt_disp.npz'
    with np.load(path) as z:
        gt,ts=z['disp']*np.float32(.8819476),z['ts']
    associations={}
    predictions={}
    results=dict(gt_file_sha256=sha256(path),interval=inference['interval'],native={},
                 curves={},equal_count={})
    for name in inference['variants']:
        with np.load(out/f'{name}.npz') as z:
            pred={k:z[k] for k in z.files}
        a=associate(pred,gt,ts,inference['interval'])
        associations[name]=a
        predictions[name]=pred
        result=metrics(a)
        fm=frame_metrics(a)
        valid=fm['valid_gt_units']>0
        coverage=fm['units'][valid]/fm['valid_gt_units'][valid]
        result['frame_coverage_p10_median_p90']=np.quantile(coverage,[.1,.5,.9]).tolist()
        result['zero_prediction_frame_fraction']=float(np.mean(fm['units'][valid]==0))
        result['nonpositive']=int(np.count_nonzero(a['prediction']<=0))
        results['native'][name]=result
        np.savez_compressed(out/f'{name}_associated.npz',**a)
        if 'score' in pred:
            results['curves'][name]=[]
            for threshold in [.05,2.,5.,10.,20.]:
                if name.startswith('nearest'):
                    continue
                # Gate rows before association; denoms never depend on prediction support.
                keep=pred['score']>=threshold
                selected=associate({k:pred[k][keep] for k in ('x_L','y_L','t','disp')},
                                   gt,ts,inference['interval'])
                results['curves'][name].append(dict(threshold=threshold,**metrics(selected)))
    names=['full','fixed_lif','fixed50','ols','no_row','no_lr']
    common=common_support([associations[n] for n in names])
    results['common']={n:metrics(a) for n,a in zip(names,common)}
    minimum=min(len(associations[n]['unit']) for n in names)
    for fraction in [.1,.25,.5,.75,1.]:
        count=int(minimum*fraction)
        results['equal_count'][str(count)]={}
        for n in names:
            a=associations[n]
            scores=predictions[n]['score'][a['source']]
            results['equal_count'][str(count)][n]=metrics(select(a,rank_subset(scores,count)))
    complete=associations['completed']
    new=~np.isin(complete['unit'],associations['full']['unit'])
    results['new_support']=metrics(select(complete,new))
    results['temporal_common']={n:metrics(a) for n,a in zip(
        ['full','median'],common_support([associations['full'],associations['median']]))}
    dump(out/'metrics.json',results)
    save_output_hashes(out)
    print(json.dumps(results['native'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--sequence',type=int,required=True,choices=[1,2,3])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=['infer','score'],required=True)
    args=parser.parse_args()
    (infer if args.mode=='infer' else score)(args.sequence,args.out)
