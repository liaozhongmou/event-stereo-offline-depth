"""Generate tables and current-result plots directly from frozen score files."""
from pathlib import Path
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
MAIN=ROOT/'manuscript/main'
SUPP=ROOT/'manuscript/supplementary'
FIG=MAIN/'figures/revision3'
RESULTS=[json.loads((ROOT/f'runs/C3_full_gt_v2_indoor{s}_20261021/metrics.json').read_text()) for s in [1,2,3]]
EXTERNAL=json.loads((ROOT/'runs/D3_dsec_thun00a_v2_20261021/metrics.json').read_text())
TIMING=json.loads((ROOT/'runs/C3_joint_timing_20261021/summary.json').read_text())
TIMING_QUALITY=json.loads((ROOT/'runs/C3_joint_timing_20261021/quality.json').read_text())
SWEEP=json.loads((ROOT/'runs/C3_completion_sweep_v2_20261021/metrics.json').read_text())
SWEEP_LABELS=dict(default='Default',radius4='Radii 4/5 px',radius12='Radii 12/13 px',
    threshold10='Occupancy 0.10',threshold30='Occupancy 0.30',one_pass='One pass',
    window50='50 ms window',stride50='50 ms stride',local_mean='Local mean')
SENSITIVITY=json.loads((ROOT/'runs/C3_sensitivity_20261021/metrics.json').read_text())
ACTIVITY=json.loads((ROOT/'runs/C3_activity_20261021/metrics.json').read_text())


def table(caption,label,columns,head,rows):
    return '\n'.join([r'\begin{table}[tbp]',r'\centering\footnotesize\setlength{\tabcolsep}{3pt}',
        r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+columns+'}',r'\toprule',
        head+r' \\',r'\midrule',*[row+r' \\' for row in rows],r'\bottomrule',r'\end{tabular}',r'\end{table}'])


def f(value,digits=3):
    return '--' if value is None else f'{value:.{digits}f}'


rows=[]
for s,result in enumerate(RESULTS,1):
    for name,label in [('same_time','Matcher'),('completed_consistent','Completed')]:
        v=result['native'][name]
        rows.append(f"Indoor {s} & {label} & {v['units']:,} & {100*v['coverage']:.3f} & {v['mae']:.3f} & {v['rmse']:.3f} & {100*v['bad3']:.2f}")
(MAIN/'revision3_main_table.tex').write_text(table(
    'Unified current outputs over complete MVSEC GT timelines. Each sequence has one fixed valid-GT denominator. Matcher outputs use bounded evidence search, same-time reverse checking and row-history blending; completed outputs additionally contain tagged interpolants.',
    'tab:unified','llrrrrr',r'Sequence & Output & Units & Coverage (\%) & MAE (px) & RMSE (px) & Bad-3 (\%)',rows),encoding='utf-8')

rows=[]
for name,label in [('consistent','Matcher'),('completed','Completed'),('nearest5','Nearest 5 ms'),('nearest50','Nearest 50 ms')]:
    v=EXTERNAL['native'][name];a=EXTERNAL['all_positive_gt'][name]
    rows.append(f"{label} & {100*v['coverage']:.3f} & {f(v['mae'])} & {100*v['bad3']:.2f} & {100*a['coverage']:.3f} & {f(a['mae'])}")
(MAIN/'revision3_external_table.tex').write_text(table(
    'Fixed-parameter DSEC thun\\_00\\_a test on all 120 GT frames. In-range columns score GT disparity in [4,28] px; all-positive columns include GT beyond the search range. No threshold was selected from this sequence.',
    'tab:dsec','lrrrrr',r'Output & In-range cov. (\%) & MAE (px) & Bad-3 (\%) & All-GT cov. (\%) & MAE (px)',rows),encoding='utf-8')

rows=[]
for mode,label in [('matcher','Matcher'),('median','Matcher + median'),('completed','Matcher + completion')]:
    v=TIMING[mode];q=v['seconds_quantiles'];rss=v['peak_rss_mib_quantiles']
    quality=TIMING_QUALITY[mode]['metrics']
    rows.append(f"{label} & {q[1]:.2f} & [{q[0]:.2f}, {q[2]:.2f}] & {rss[1]:.0f} & {quality['mae']:.3f}")
(MAIN/'revision3_timing_table.tex').write_text(table(
    'Five independent complete invocations per output mode on Indoor Flying 1. Wall-clock includes NPZ loading, both-camera LIF, CSR packing, matching, the selected output stage and compressed output writing. Imports, JIT warm-up, hashing and evaluation are excluded. An independent process samples pipeline RSS every 20 ms.',
    'tab:joint_timing','lrrrr',r'Output mode & Median (s) & [Q1,Q3] (s) & Median peak (MiB) & MAE (px)',rows),encoding='utf-8')

supp=[]
names=['same_time','consistent_fixed_lif','consistent_fixed50','consistent_ols','consistent_no_row','no_lr_true']
labels=['Current matcher','Fixed LIF','Fixed 50 ms','OLS','No row history','No LR check']
for s,result in enumerate(RESULTS,1):
    common=result['shared']['+'.join(names)]
    count=next(iter(common.values()))['units']
    rows=[]
    for name,label in zip(names,labels):
        v=result['native'][name]
        rows.append(f"{label} & {v['units']:,} & {100*v['coverage']:.3f} & {v['mae']:.3f} & {common[name]['mae']:.3f}")
    supp.append(table(f'Current-version one-factor controls on Indoor Flying {s}; shared errors use exactly {count:,} common units. All configurations use the same bounded evidence and time-consistent reverse implementation. The no-LR control removes its score penalty and gate.',
        f'tab:current_ablation_{s}','lrrrr',r'Configuration & Units & Coverage (\%) & Native MAE & Shared MAE',rows))

rows=[]
for s,result in enumerate(RESULTS,1):
    v=result['native']['completed_consistent'];q=v['frame_coverage_p10_median_p90'];active=result['event_active_coverage']
    rows.append(f"Indoor {s} & {100*q[0]:.3f} & {100*q[1]:.3f} & {100*q[2]:.3f} & {100*v['zero_prediction_frame_fraction']:.2f} & {100*active['methods']['completed_consistent']['coverage']:.3f}")
supp.append(table('Frame-coverage distribution of current completed outputs. P10/median/P90 include zero-output frames. Active coverage uses valid GT pixels with a raw left event within 15 ms of the GT time; this is an activity mask, not an independently annotated object-edge mask.',
    'tab:current_frames','lrrrrr',r'Sequence & P10 (\%) & Median (\%) & P90 (\%) & Zero frames (\%) & Active cov. (\%)',rows))
rows=[]
for s,result in enumerate(RESULTS,1):
    for name,label in [('below1','Below 1 m'),('1to2','1--2 m'),('above2','At least 2 m')]:
        v=result['depth_strata'][name]
        rows.append(f"Indoor {s} & {label} & {v['units']:,} & {f(v['mae'])} & {f(v['depth_mae_m'])} & {f(v['depth_rmse_m'])} & {f(v['relative_depth_error'])}")
supp.append(table('Current completed-output depth strata defined by GT depth. Relative error is the mean absolute metric-depth error divided by GT depth, reported as a fraction.',
    'tab:current_depth','llrrrrr',r'Sequence & GT depth & Units & Disp. MAE & Depth MAE (m) & RMSE (m) & Relative error',rows))
rows=[]
for s,result in enumerate(RESULTS,1):
    v=result['newly_reached'];b=result['new_support_boundary']
    rows.append(f"Indoor {s} & {v['units']:,} & {v['mae']:.3f} & {100*v['bad3']:.2f} & {b['within2px']['units']:,} & {b['within2px']['mae']:.3f} & {b['away']['mae']:.3f}")
supp.append(table('Newly reached completed support, excluding every matcher unit. A boundary is within two pixels (Chebyshev distance) of a valid adjacent GT disparity jump greater than 1 px. This GT-derived diagnostic does not control inference.',
    'tab:current_new','lrrrrrr',r'Sequence & New units & MAE & Bad-3 (\%) & Boundary units & Boundary MAE & Away MAE',rows))
stress=json.loads((ROOT/'runs/C3_temporal_stress_20261021/metrics.json').read_text())
rows=[]
for name in ['constant','linear','accelerated','step','noise']:
    rows.append(name.capitalize()+' & '+' & '.join(f"{stress[name][key]['mae']:.3f} / {stress[name][key]['max_error']:.3f}" for key in ['bypass','median5','legacy','consistent_basis']))
supp.append(table('Controlled temporal trajectories: MAE / maximum absolute error (px) against the specified truth. Each trajectory has 40 observations at 50 ms spacing; the noisy constant case adds Gaussian noise with standard deviation 0.5 px, seed 20261021. Both polynomial versions use the archived settings. The gap test has observations at 0--0.04 and 5--5.01 s; median reset emits the observed 20 px immediately after the gap and no values within the gap.',
    'tab:temporal_stress','lrrrr',r'Trajectory & Bypass & Median 5 & Legacy & Consistent basis',rows))
rows=[]
for sequence,entry in ACTIVITY.items():
    matcher='consistent' if sequence=='dsec' else 'same_time'
    completed='completed' if sequence=='dsec' else 'completed_consistent'
    for group,label in enumerate(['Low','Middle','High']):
        m=entry['methods'][matcher]['strata'][group];c=entry['methods'][completed]['strata'][group]
        rows.append(f"{'DSEC' if sequence=='dsec' else 'Indoor '+sequence} & {label} & {c['event_count_min']:,}--{c['event_count_max']:,} & {f(m['mae'])} & {100*c['coverage']:.3f} & {f(c['mae'])}")
supp.append(table('Descriptive raw-left-event activity strata. Frames are sorted by the number of raw left events in a 30 ms interval centered on GT time, then divided into three nearly equal groups before inspecting error. Ties use timeline order. Coverage divides by all valid GT units in the group. DSEC uses the unchanged in-range GT definition. These within-recording strata do not isolate illumination or speed.',
    'tab:activity','lllrrr',r'Sequence & Activity & Raw events / 30 ms & Matcher MAE & Completed cov. (\%) & MAE',rows))
rows=[]
for name,v in SWEEP.items():
    rows.append(f"{SWEEP_LABELS[name]} & {v['seconds']:.2f} & {100*v['native']['coverage']:.3f} & {v['native']['mae']:.3f} & {f(v['new']['mae'])} & {v['new_correct3_units']:,}")
supp.append(table('Centered-completion sweep on a fixed 12 s development clip, selected by time before error computation. The neighborhood-scale factor couples occupancy/mean radii at 4/5, 8/9 (default) or 12/13 px; other controls change one setting. The local-mean reference removes occupancy rejection and uses one pass. Timings cover completion only after dtype-specific warm-up. Added good units have absolute disparity error at most 3 px.',
    'tab:completion_sweep','lrrrrr',r'Setting & Time (s) & Coverage (\%) & Native MAE & New MAE & Added good units',rows))
(SUPP/'revision3_tables.tex').write_text('\n\n'.join(supp),encoding='utf-8')
rows=[f"{name.replace('_',' ')} & {100*v['coverage']:.3f} & {v['mae']:.3f} & {100*v['bad3']:.2f}"
      for name,v in SENSITIVITY.items()]
with (SUPP/'revision3_tables.tex').open('a',encoding='utf-8') as stream:
    stream.write('\n\n'+table('LIF rate-term and fixed-window sensitivity on the same time-selected 12 s development clip. The LIF state is primed from the start of the recording; matching has 0.3 s temporal context around the scoring interval. All other parameters are fixed.',
        'tab:lif_window_sensitivity','lrrr',r'Setting & Coverage (\%) & MAE (px) & Bad-3 (\%)',rows))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})
plot_names=['same_time','consistent_fixed_lif','consistent_fixed50','consistent_ols','consistent_no_row','nearest5']
plot_labels=['Current matcher','Fixed LIF','Fixed 50 ms','OLS','No row history','Nearest 5 ms']
colors=['#202020','#0072B2','#D55E00','#009E73','#CC79A7','#8B6F00']
markers=['o','s','^','D','v','P']
fig,axes=plt.subplots(1,3,figsize=(7.1,2.8))
for ax,result,s in zip(axes,RESULTS,[1,2,3]):
    for name,label,color,marker in zip(plot_names,plot_labels,colors,markers):
        points=[row[name] for row in result['equal_count'].values()]
        ax.plot([100*v['coverage'] for v in points],[v['mae'] for v in points],
                label=label,color=color,marker=marker,ms=3,lw=.85)
    ax.set_title(f'Indoor Flying {s}',fontsize=9);ax.set_xlabel('Valid-GT coverage (%)')
    ax.grid(color='.9',lw=.5)
axes[0].set_ylabel('MAE at equal count (px)')
handles,labs=axes[0].get_legend_handles_labels()
fig.legend(handles,labs,loc='lower center',ncol=3,frameon=False,fontsize=8)
fig.subplots_adjust(left=.08,right=.99,bottom=.36,top=.88,wspace=.36)
for extension in ['svg','pdf','png']:fig.savefig(FIG/f'Fig4_EqualCoverage.{extension}',dpi=200)
plt.close(fig)

# Pre-association score sweeps complement the post-association equal-count view.
fig,axes=plt.subplots(2,3,figsize=(7.1,4.8))
for col,(result,s) in enumerate(zip(RESULTS,[1,2,3])):
    for name,label,color,marker in zip(plot_names,plot_labels,colors,markers):
        points=[v for v in result['curves'][name] if v['mae'] is not None]
        axes[0,col].plot([100*v['coverage'] for v in points],[v['mae'] for v in points],label=label,color=color,marker=marker,ms=3,lw=.85)
        axes[1,col].plot([100*v['coverage'] for v in points],[100*v['bad3'] for v in points],color=color,marker=marker,ms=3,lw=.85)
    axes[0,col].set_title(f'Indoor Flying {s}',fontsize=9)
    axes[1,col].set_xlabel('Valid-GT coverage (%)')
    for ax in axes[:,col]:ax.grid(color='.9',lw=.5)
axes[0,0].set_ylabel('MAE (px)');axes[1,0].set_ylabel('Bad-3 (%)')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=8)
fig.subplots_adjust(left=.08,right=.99,bottom=.22,top=.92,hspace=.30,wspace=.36)
for extension in ['svg','pdf','png']:fig.savefig(FIG/f'Fig6_ScoreSweep.{extension}',dpi=200)
plt.close(fig)

# Fixed middle timeline indices; every displayed scientific pixel is scored data.
fig,axes=plt.subplots(4,4,figsize=(7.1,7.1))
frame_sources=[]
for row in range(4):
    if row<3:
        sequence=row+1
        with np.load(ROOT.parent/f'indoor{sequence}_data/indoor{sequence}_gt_disp.npz') as z:
            index=len(z['ts'])//2;gt=z['disp'][index]*np.float32(.8819476)
        directory=ROOT/f'runs/C3_full_gt_v2_indoor{sequence}_20261021'
        names=['same_time_associated','completed_consistent_associated'];label=f'Indoor {sequence}'
    else:
        frames=sorted((ROOT/'datasets/dsec/thun_00_a').glob('[0-9]*.png'))
        index=len(frames)//2;gt=np.asarray(Image.open(frames[index])).astype(np.float32)/256
        directory=ROOT/'runs/D3_dsec_thun00a_v2_20261021'
        names=['consistent_associated','completed_associated'];label='DSEC'
    valid=np.isfinite(gt)&(gt>=4)&(gt<=28);gt=np.where(valid,gt,np.nan)
    shape=gt.shape;maps=[]
    for name in names:
        grid=np.full(shape,np.nan)
        with np.load(directory/f'{name}.npz') as z:
            keep=z['frame']==index;pixels=z['unit'][keep]%(shape[0]*shape[1])
            grid.ravel()[pixels]=z['prediction'][keep]
        maps.append(grid)
    error=np.abs(maps[1]-gt)
    frame_sources.append(dict(sequence=label,frame_index=index,shape=shape,association_folder=str(directory.relative_to(ROOT)),finite=[int(np.isfinite(v).sum()) for v in [gt,*maps,error]]))
    for col,grid in enumerate([gt,*maps,error]):
        ax=axes[row,col]
        plotted=ax.imshow(grid,cmap='magma' if col==3 else 'viridis',vmin=0 if col==3 else 4,vmax=8 if col==3 else 28,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
        if row==0:ax.set_title(['GT','Matcher','Completed','Absolute error'][col],fontsize=9)
        if col==0:ax.set_ylabel(f'{label}\nframe {index}',fontsize=8)
        if col==3:error_image=plotted
        else:disparity_image=plotted
fig.subplots_adjust(left=.075,right=.97,top=.95,bottom=.12,wspace=.06,hspace=.11)
cax=fig.add_axes([.13,.055,.40,.012]);fig.colorbar(disparity_image,cax=cax,orientation='horizontal',label='Disparity (px)',ticks=[4,12,20,28])
cax=fig.add_axes([.67,.055,.25,.012]);fig.colorbar(error_image,cax=cax,orientation='horizontal',label='Absolute error (px; saturated at 8)',ticks=[0,4,8])
for extension in ['svg','pdf','png']:fig.savefig(FIG/f'Fig5_CurrentMaps.{extension}',dpi=220)
plt.close(fig)
(FIG/'Fig5_source.json').write_text(json.dumps(frame_sources,indent=2),encoding='utf-8')

fig,axes=plt.subplots(1,2,figsize=(7.1,3.3))
sweep_colors=['#202020','#0072B2','#009E73','#D55E00','#CC79A7','#8B6F00','#56B4E9','#555555','#882255']
sweep_markers=['o','s','^','D','v','P','X','>','*']
for (name,v),color,marker in zip(SWEEP.items(),sweep_colors,sweep_markers):
    axes[0].scatter(v['seconds'],v['new_correct3_units']/1000,color=color,marker=marker,s=28,label=SWEEP_LABELS[name])
    axes[1].scatter(100*v['native']['coverage'],v['new']['mae'],color=color,marker=marker,s=28)
axes[0].set(xlabel='Completion time (s)',ylabel='Added good units (thousands)\n(error ≤3 px)')
axes[1].set(xlabel='Valid-GT coverage (%)',ylabel='Newly reached MAE (px)')
for ax in axes:ax.grid(color='.9',lw=.5)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=8)
fig.subplots_adjust(left=.14,right=.985,top=.94,bottom=.36,wspace=.45)
for extension in ['svg','pdf','png']:fig.savefig(FIG/f'Fig7_CompletionCost.{extension}',dpi=200)
plt.close(fig)

evidence=ROOT/'manuscript/review_evidence/revision3'
evidence.mkdir(exist_ok=True)
for s in [1,2,3]:
    shutil.copy2(ROOT/f'runs/C3_full_gt_v2_indoor{s}_20261021/metrics.json',evidence/f'indoor{s}_metrics.json')
shutil.copy2(ROOT/'runs/D3_dsec_thun00a_v2_20261021/metrics.json',evidence/'dsec_metrics.json')
shutil.copy2(ROOT/'runs/C3_joint_timing_20261021/summary.json',evidence/'timing_summary.json')
shutil.copy2(ROOT/'runs/C3_joint_timing_20261021/quality.json',evidence/'timing_quality.json')
shutil.copy2(ROOT/'runs/C3_activity_20261021/metrics.json',evidence/'activity_strata.json')
shutil.copy2(ROOT/'runs/C3_temporal_stress_20261021/metrics.json',evidence/'temporal_stress.json')
shutil.copy2(ROOT/'runs/C3_sensitivity_20261021/metrics.json',evidence/'lif_window_sensitivity.json')
shutil.copy2(ROOT/'runs/C3_completion_sweep_v2_20261021/metrics.json',evidence/'completion_sweep.json')
print('Generated current tables and equal-coverage figure directly from scored runs.')
