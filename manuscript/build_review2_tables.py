"""Generate revision-stage tables directly from diagnostic JSON records."""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).parent
RUN = ROOT.parent / 'runs/B2_review2_stage_audit_20261021'
data = json.loads((RUN / 'metrics.json').read_text())
evidence = ROOT / 'review_evidence'
for source, name in [(RUN / 'metrics.json', 'review2_stage_metrics.json'),
                     (RUN / 'contract.json', 'review2_contract.json'),
                     (RUN / 'archive_replay_equality.json', 'review2_replay_equality.json'),
                     (RUN / 'evaluator_equality.json', 'review2_evaluator_equality.json'),
                     (RUN / 'output_hashes.json', 'review2_output_hashes.json'),
                     (ROOT.parent / 'runs/A2_review2_all_gt_20261021/audit.json', 'review2_all_gt_audit.json')]:
    record = json.loads(source.read_text())
    if 'manifest' in record:
        record['manifest']['platform'] = 'Windows; see full project run manifest'
    (evidence / name).write_text(json.dumps(record, indent=2), encoding='utf-8')
for name in ['stage_maps', 'source_masks']:
    shutil.copy2(RUN / (name + '.pdf'), ROOT / 'supplementary/figures' / ('review2_' + name + '.pdf'))

rows = []
for label, result in [
    ('Matcher', data['partition']['matcher']),
    ('Temporal: archive / legacy replay', data['partition']['temporal']),
    ('Temporal: consistent-basis replay', data['paired_temporal_replay']['consistent_basis']['native']),
    ('Completed: shared locations', data['partition']['completed_shared_locations']),
    ('Completed: newly reached locations', data['partition']['newly_reached']),
    ('All completed', data['partition']['all_completed'])]:
    rows.append(f"{label} & {result['units']:,} & {100*result['coverage']:.3f} & {result['mae']:.3f} & {result['rmse']:.3f} & {100*result['bad3']:.2f} " + r'\\')
table = r'''\begin{table}[tbp]
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Reproducible Indoor Flying~1 stage audit under the reconstructed association contract. The first three rows use the same 201,345 units. Completed shared and newly reached rows partition all completed support. Their common denominator is 91,962,488 valid GT units over 1399 frames. The legacy replay reproduces the stored temporal disparity array exactly.}
\label{tab:stage_audit}
\begin{tabular}{lrrrrr}
\toprule
Output / version & Units & Coverage (\%) & MAE (px) & RMSE (px) & Bad-3 (\%) \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{table}
'''
(ROOT / 'main/review2_stage_table.tex').write_text(table, encoding='utf-8')

rows = []
for label, result in [('Within 2 px of boundary', data['new_support_boundary']['within_2px']),
                       ('Away from boundary', data['new_support_boundary']['away_from_boundary'])]:
    rows.append(f"{label} & {result['units']:,} & {result['mae']:.3f} & {result['rmse']:.3f} & {100*result['bad3']:.2f} " + r'\\')
boundary = r'''\begin{table}[htbp]
\centering\footnotesize
\caption{Newly reached support split by the fixed GT-boundary definition: adjacent valid disparities differ by more than 1 px, expanded by a two-pixel Chebyshev radius.}
\label{tab:new_support_boundary}
\begin{tabular}{lrrrr}
\toprule
Region & Units & MAE (px) & RMSE (px) & Bad-3 (\%) \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{table}
'''
rows = []
for label, result in [('Below 1 m', data['depth_strata']['below_1m']),
                       ('1 to below 2 m', data['depth_strata']['1_to_2m']),
                       ('2 m and above', data['depth_strata']['2m_and_above']),
                       ('All completed', data['partition']['all_completed'])]:
    depth = result['metric_depth']
    rows.append(f"{label} & {result['units']:,} & {result['mae']:.3f} & {depth['mae_m']:.3f} & {depth['rmse_m']:.3f} & {depth['p90_m']:.3f} " + r'\\')
depth_table = r'''\begin{table}[htbp]
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Depth-stratified audit of the stored completed output. Strata use GT depth $c_{rect}/d_{GT}$. All completed predictions are positive, so every listed unit contributes to the metric-depth errors.}
\label{tab:depth_strata}
\begin{tabular}{lrrrrr}
\toprule
GT depth & Units & MAE (px) & MAE (m) & RMSE (m) & P90 (m) \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{table}
'''
rows = []
for label, key in [('Matcher / temporal', 'raw'), ('Completed', 'complete')]:
    values = data['frame_coverage'][key]['coverage_quantiles']
    rows.append(label + ' & ' + ' & '.join(f'{100*x:.3f}' for x in values) + r' \\')
coverage = r'''\begin{table}[htbp]
\centering\footnotesize
\caption{Frame-level valid-GT coverage in percent over all 1399 frames, including zero-output frames. P10 and P90 are empirical frame quantiles.}
\label{tab:frame_coverage}
\begin{tabular}{lrrrrr}
\toprule
Stage & Minimum & P10 & Median & P90 & Maximum \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{table}
'''
(ROOT / 'supplementary/review2_tables.tex').write_text(boundary+'\n'+depth_table+'\n'+coverage, encoding='utf-8')
print('Generated one main and three supplementary tables from saved diagnostic JSON.')
