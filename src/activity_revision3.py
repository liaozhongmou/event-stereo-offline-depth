"""Descriptive frame strata by raw left-event count, without error-based bins."""
import hdf5plugin
import h5py
import numpy as np
from src.run_revision3 import ROOT,dump
from src.evaluation import select,metrics
from src.run_historical import sha256

out=ROOT/'runs/C3_activity_20261021'
out.mkdir(exist_ok=False)
result={}
for sequence in [1,2,3,'dsec']:
    if sequence=='dsec':
        data=ROOT/'datasets/dsec/thun_00_a'
        with h5py.File(data/'events_left/events.h5') as f:
            times=(f['events/t'][:].astype(np.float64)+float(f['t_offset'][()]))*1e-6
        timestamps=np.loadtxt(data/'disparity_timestamps.txt')*1e-6
        directory=ROOT/'runs/D3_dsec_thun00a_v2_20261021'
        names=['consistent','completed','nearest5']
    else:
        data=ROOT.parent/f'indoor{sequence}_data'
        with np.load(data/f'indoor{sequence}_left1.npz') as z:times=z['t']
        with np.load(data/f'indoor{sequence}_gt_disp.npz') as z:timestamps=z['ts']
        directory=ROOT/f'runs/C3_full_gt_v2_indoor{sequence}_20261021'
        names=['same_time','completed_consistent','nearest5']
    counts=np.searchsorted(times,timestamps+.015,side='right')-np.searchsorted(times,timestamps-.015)
    del times
    # All timeline frames, including inactive frames; ties resolve by frame index.
    groups=np.empty(len(counts),np.int8)
    for i,indices in enumerate(np.array_split(np.argsort(counts,kind='stable'),3)):groups[indices]=i
    entry=dict(raw_left_events_in_30ms=counts.tolist(),frame_groups=groups.tolist(),methods={})
    for name in names:
        path=directory/f'{name}_associated.npz'
        with np.load(path) as z:a={key:z[key] for key in z.files}
        strata=[]
        for group in range(3):
            subset=select(a,groups[a['frame']]==group)
            frames=np.flatnonzero(groups==group)
            subset['denominator']=int(a['valid_counts'][frames].sum())
            strata.append(dict(frames=len(frames),event_count_min=int(counts[frames].min()),
                event_count_max=int(counts[frames].max()),**metrics(subset)))
        entry['methods'][name]=dict(source_sha256=sha256(path),strata=strata)
    result[str(sequence)]=entry
dump(out/'metrics.json',result)
print('Saved activity strata for all four recordings.')
