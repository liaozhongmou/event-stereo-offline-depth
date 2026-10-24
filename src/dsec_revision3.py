"""Official rectification and frozen-parameter DSEC thun_00_a test."""
from pathlib import Path
from functools import partial
import argparse
import json
import time
import hdf5plugin
import h5py
import numpy as np
from PIL import Image
import yaml

from src.run_revision3 import ROOT,PARAMS,filtered,matching,dump
from src.stereo_controls import match_v2_adaptive
from src.revision3 import complete_events,nearest_time_match
from src.evaluation import associate,metrics,common_support,frame_metrics
from src.run_historical import sha256,save_output_hashes

DATA=ROOT/'datasets/dsec/thun_00_a'


def load_events(side):
    folder=DATA/f'events_{side}'
    with h5py.File(folder/'rectify_map.h5','r') as f:
        mapping=f['rectify_map'][:]
    with h5py.File(folder/'events.h5','r') as f:
        x,y=f['events/x'][:],f['events/y'][:]
        t=(f['events/t'][:].astype(np.float64)+float(f['t_offset'][()]))*1e-6
        p=(f['events/p'][:].astype(np.int8)*2-1)
    xy=np.rint(mapping[y,x]).astype(np.int32)
    keep=(xy[:,0]>=0)&(xy[:,0]<640)&(xy[:,1]>=0)&(xy[:,1]<480)
    assert (np.diff(t)>=0).all()
    return xy[keep,0],xy[keep,1],t[keep],p[keep]


def infer(out):
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'contract.json',dict(params=PARAMS,geometry=[480,640],
        default='bounded matching with same_time=1; temporal bypass; centered completion',
        timestamp='(events/t + t_offset) * 1e-6; official clock',
        rectification='official rectify_map[y,x], nearest integer, discard outside field',
        gt='not loaded in inference; external test uses parameters frozen on MVSEC',
        source={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'src/stereo_controls.py',
            ROOT/'src/run_revision3.py',ROOT/'src/revision3.py',ROOT/'src/dsec_revision3.py']}))
    selected=[];counts=[];ranges=[]
    for side in ['left','right']:
        raw=load_events(side)
        counts.append(len(raw[2]));ranges.append([float(raw[2][0]),float(raw[2][-1])])
        selected.append(filtered(raw,(480,640)))
        del raw
        print(side,'rectified',counts[-1],'filtered',len(selected[-1][2]),flush=True)
    interval=[max(ranges[0][0],ranges[1][0]),min(ranges[0][1],ranges[1][1])]
    times={}
    for name,kernel in [('consistent',partial(match_v2_adaptive,same_time=1)),
                         ('bounded_legacy_time',partial(match_v2_adaptive,same_time=0))]:
        begin=time.perf_counter()
        pred=matching(*selected,(480,640),PARAMS,kernel=kernel)
        np.savez_compressed(out/f'{name}.npz',**pred)
        times[name]=dict(rows=len(pred['t']),seconds=time.perf_counter()-begin)
        print(name,times[name],flush=True)
        if name=='consistent':
            default=pred
    for name,dt in [('nearest50',.05),('nearest5',.005)]:
        begin=time.perf_counter()
        pred=dict(zip(('x_L','y_L','t','disp','score'),nearest_time_match(*selected,640,480,dt)))
        np.savez_compressed(out/f'{name}.npz',**pred)
        times[name]=dict(rows=len(pred['t']),seconds=time.perf_counter()-begin)
    completed=complete_events(default,(480,640),interval)
    np.savez_compressed(out/'completed.npz',**completed)
    times['completed']=dict(rows=len(completed['t']))
    dump(out/'inference.json',dict(interval=interval,variants=times,
        events=dict(raw=counts,filtered=[len(v[2]) for v in selected]),
        download_manifest_sha256=sha256(DATA/'download_manifest.json')))
    save_output_hashes(out)


def score(out):
    timestamps=np.loadtxt(DATA/'disparity_timestamps.txt')*1e-6
    frames=sorted(DATA.glob('[0-9]*.png'))
    gt=np.stack([np.asarray(Image.open(p)).astype(np.float32)/256. for p in frames])
    assert len(gt)==len(timestamps)
    interval=[float(timestamps[0]),float(timestamps[-1])]
    associations={};result=dict(frames=len(frames),interval=interval,native={},all_positive_gt={},shared={})
    denominator_positive=int(np.count_nonzero(gt>0))
    denominator_range=int(np.count_nonzero((gt>=4)&(gt<=28)))
    result['in_search_range_gt_fraction']=denominator_range/denominator_positive
    for path in sorted(out.glob('*.npz')):
        if path.stem.endswith('_associated'):
            continue
        with np.load(path) as z:
            pred={k:z[k] for k in z.files}
        a=associate(pred,gt,timestamps,interval)
        associations[path.stem]=a
        result['native'][path.stem]=metrics(a)
        result['all_positive_gt'][path.stem]=metrics(associate(pred,gt,timestamps,interval,
            gt_range=(np.finfo(np.float32).tiny,np.inf)))
        np.savez_compressed(out/f'{path.stem}_associated.npz',**a)
        print(path.stem,result['native'][path.stem],flush=True)
    names=['consistent','nearest50','nearest5']
    result['shared']={n:metrics(a) for n,a in zip(names,common_support([associations[n] for n in names]))}
    dump(out/'metrics.json',result);save_output_hashes(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['infer','score'],required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    (infer if a.mode=='infer' else score)(a.out)
