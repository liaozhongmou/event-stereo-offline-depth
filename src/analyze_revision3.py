"""Score complete GT timelines and paired controls under one explicit protocol."""
from pathlib import Path
import argparse
import json
import numpy as np
from src.run_revision3 import ROOT,dump
from src.evaluation import associate,common_support,metrics,select,frame_metrics
from src.revision3 import rank_subset
from src.run_historical import sha256,save_output_hashes
from diagnostics.review2_analysis import boundary_mask


def run(sequence,out):
    out.mkdir(parents=True,exist_ok=False)
    source=ROOT/f'runs/C3_unified_indoor{sequence}_20261021'
    controls=ROOT/f'runs/C3_controls_v2_indoor{sequence}_20261021'
    path=ROOT.parent/f'indoor{sequence}_data/indoor{sequence}_gt_disp.npz'
    with np.load(path) as z:
        gt,ts=z['disp']*np.float32(.8819476),z['ts']
    interval=[float(ts[0]),float(ts[-1])]
    result=dict(interval=interval,frames=len(ts),gt_sha256=sha256(path),
                scorer_sha256=sha256(Path(__file__)),evaluator_sha256=sha256(ROOT/'src/evaluation.py'),
                source_contracts={str(p):sha256(p) for p in [source/'contract.json',controls/'contract.json']},
                gt_range_px=[4,28],association_ms=15,native={},equal_count={},curves={},
                source_hashes={},frame_distributions={})
    associations={};scores={}
    paths={n:source/f'{n}.npz' for n in ['full','fixed_lif','fixed50','ols','no_row',
                                      'nearest50','nearest5','median','completed']}
    paths.update({n:controls/f'{n}.npz' for n in ['bounded','same_time','no_lr_true','completed_consistent']})
    ablations=ROOT/f'runs/C3_current_ablation_indoor{sequence}_20261021'
    paths.update({n:ablations/f'{n}.npz' for n in ['consistent_fixed_lif','consistent_fixed50','consistent_ols','consistent_no_row']})
    for name,path in paths.items():
        with np.load(path) as z:
            pred={k:z[k] for k in z.files}
        a=associate(pred,gt,ts,interval)
        associations[name]=a
        result['source_hashes'][name]=dict(path=str(path),sha256=sha256(path))
        m=metrics(a)
        fm=frame_metrics(a);valid=fm['valid_gt_units']>0
        cov=fm['units'][valid]/fm['valid_gt_units'][valid]
        m.update(frame_coverage_p10_median_p90=np.quantile(cov,[.1,.5,.9]).tolist(),
                 zero_prediction_frame_fraction=float(np.mean(fm['units'][valid]==0)),
                 nonpositive=int(np.count_nonzero(a['prediction']<=0)),
                 output_rows=len(pred['t']),
                 output_outside_range=int(np.count_nonzero((pred['disp']<4)|(pred['disp']>28))))
        result['native'][name]=m
        np.savez_compressed(out/f'{name}_associated.npz',**a)
        result['frame_distributions'][name]={k:np.asarray(v).tolist() for k,v in fm.items() if k!='mae'}
        if 'score' in pred:
            scores[name]=pred['score'][a['source']]
            result['curves'][name]=[]
            thresholds=[0,.0025,.01,.025] if name.startswith('nearest') else [.05,2.,5.,10.,20.]
            for threshold in thresholds:
                keep=pred['score']>=threshold
                filtered={k:pred[k][keep] for k in ('x_L','y_L','t','disp')}
                result['curves'][name].append(dict(threshold=threshold,
                    **metrics(associate(filtered,gt,ts,interval))))
        if name=='completed_consistent':
            result['selected_interpolant_fraction']=float(np.mean(pred['kind'][a['source']]==1))
        print(sequence,name,m['units'],m['mae'],flush=True)
    comparisons=[['full','fixed_lif','fixed50','ols','no_row'],
                 ['same_time','consistent_fixed_lif','consistent_fixed50','consistent_ols','consistent_no_row','no_lr_true'],
                 ['full','bounded','same_time','no_lr_true'],['same_time','nearest50','nearest5'],
                 ['full','median'],['same_time','completed_consistent']]
    result['shared']={}
    for names in comparisons:
        result['shared']['+'.join(names)]={n:metrics(a) for n,a in zip(
            names,common_support([associations[n] for n in names]))}
    names=['full','fixed_lif','fixed50','ols','no_row','same_time','no_lr_true','nearest50','nearest5',
           'consistent_fixed_lif','consistent_fixed50','consistent_ols','consistent_no_row']
    minimum=min(len(associations[n]['unit']) for n in names)
    for frac in [.1,.25,.5,.75,1.]:
        count=int(frac*minimum)
        result['equal_count'][str(count)]={n:metrics(select(associations[n],rank_subset(scores[n],count)))
                                          for n in names}
    a=associations['completed_consistent']
    new=~np.isin(a['unit'],associations['same_time']['unit'])
    result['newly_reached']=metrics(select(a,new))
    depth=19.941771812941038/a['gt'].astype(float)
    result['depth_strata']={}
    for name,keep in [('below1',depth<1),('1to2',(depth>=1)&(depth<2)),('above2',depth>=2)]:
        subset=select(a,keep)
        errors=np.abs(19.941771812941038/subset['prediction']-19.941771812941038/subset['gt'])
        result['depth_strata'][name]=dict(**metrics(subset),depth_mae_m=float(errors.mean()) if len(errors) else None,
             depth_rmse_m=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
             depth_p90_m=float(np.quantile(errors,.9)) if len(errors) else None,
             relative_depth_error=float(np.mean(np.abs(subset['gt']/subset['prediction']-1))) if len(errors) else None)
    # Event-active pixels are defined from the common raw left stream, independent of method.
    raw_path=ROOT.parent/f'indoor{sequence}_data/indoor{sequence}_left1.npz'
    with np.load(raw_path) as z:
        ex,ey,et=z['x'],z['y'],z['t']
    active_counts={n:0 for n in associations};active_denominator=0
    edge=np.zeros(len(a['unit']),bool);h,w=gt.shape[1:]
    for f,timestamp in enumerate(ts):
        lo,hi=np.searchsorted(et,[timestamp-.015,timestamp+.015])
        active=np.zeros((h,w),bool);active[ey[lo:hi],ex[lo:hi]]=True
        valid=np.isfinite(gt[f])&(gt[f]>=4)&(gt[f]<=28)
        active_denominator+=int(np.count_nonzero(active&valid))
        for n,ass in associations.items():
            start,end=np.searchsorted(ass['frame'],[f,f+1])
            active_counts[n]+=int(np.count_nonzero(active.ravel()[ass['unit'][start:end]%(h*w)]))
        start,end=np.searchsorted(a['frame'],[f,f+1])
        edge[start:end]=boundary_mask(np.where(valid,gt[f],np.nan)).ravel()[a['unit'][start:end]%(h*w)]
    result['event_active_coverage']=dict(definition='raw left events within +/-15ms of GT time; valid GT only',
        denominator=active_denominator,methods={n:dict(units=count,coverage=count/active_denominator)
                                                 for n,count in active_counts.items()})
    result['new_support_boundary']={n:metrics(select(a,new&mask)) for n,mask in [('within2px',edge),('away',~edge)]}
    dump(out/'metrics.json',result)
    save_output_hashes(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=int,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();run(args.sequence,args.out)
