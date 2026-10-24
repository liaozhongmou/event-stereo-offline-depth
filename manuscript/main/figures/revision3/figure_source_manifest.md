# Scientific figure source manifest — revision 3

All mechanism graphics are drawn with Matplotlib primitives. No generative image
model, stock photograph, supposed physical robot deployment or invented scientific
image is used. Editable SVG text is retained as text; PDF fonts are embedded.

## Content audit and scientific choices

| Figure / panel | Scientific content | Source / status |
|---|---|---|
| Fig.1a | Left/right optical centers, baseline, shared world point, separate rectified horizontal rows; positive d and Z=fb/d | Conceptual vector geometry, not measured hardware |
| Fig.1b | Signed events, exponential membrane/rate decay, logarithmic threshold, accepted/suppressed pulses without reset | Method equations and original/neuron_filter_vis.py; explicitly synthetic pulse input |
| Fig.1c | Time versus right horizontal location, IRLS intercept at zero relative time, subsequent peak refinement | Method equations and src/stereo_controls.py; schematic local image trajectory, not a 3D trajectory |
| Fig.1d | Sparse M estimates, separate legacy T audit, current M→C route and tagged interpolants | Current unified path and source labels in src/revision3.py; schematic grids |
| Fig.1 evaluation | Recorded GT enters only outcome-independent association and metrics | src/evaluation.py, 15 ms, one selected unit |
| Fig.2 | GT / archived pipeline / EMatch and identical enlarged crop | Retained Figure6_holdout_midframe.pdf image objects13/14/15 plus alpha masks24/27/30; original scientific pixels preserved |
| Fig.3 | Six historical component operating points | Preserved historical Table7; exact values below; 89,532 common units |
| Fig.4 | Five score-ranked equal-count selections per current configuration on each complete sequence | C3_full_gt_v2_indoor{1,2,3}_20261021/metrics.json, equal_count |
| Fig.5 | Fixed middle GT indices and scored current prediction/error maps | Associated arrays from those runs and D3_dsec_thun00a_v2_20261021; Fig5_source.json gives indices and counts |
| Fig.6 | Pre-association score threshold MAE/Bad3–coverage sweeps | Same metrics.json files, curves; empty selections retain null in source and have no plotted error |
| Fig.7 | Completion-only time, added good units and newly reached MAE | C3_completion_sweep_v2_20261021/metrics.json; fixed time-selected development clip |

The main manuscript adopts layout A, the horizontal editorial arrangement.
Layout B is an alternate two-row technical arrangement. Both use conceptual
hardware rather than unauthorised or misidentified photographs. Retaining T as an
audit branch follows M5; C in the current path preserves M, not a polynomial output.
The fitted intercept initializes the evidence search; the refined peak, rather
than the intercept alone, determines emitted disparity.

## Fig.1 中文布局说明（作者预览）

左侧以概念双目几何交代左右传感器、基线、矫正行和正视差。中部依次展示有符号事件、无重置膜电位与动态阈值，以及相对时间坐标中的稳健拟合和峰值搜索。右侧区分匹配、历史时间审计与局部插值；当前路线直接从匹配进入补全。底部真值仅流入关联与评价，不参与推理。

## Fig.2 provenance and limits

The source image panels are 211×159 pixels. The figure retains their finite-value
alpha masks, white unsupported regions and colours. Enlargements use an identical
75×45 crop beginning at image coordinate(65,78), with nearest-neighbour display.
The image-plane visibility annotations47.1%,8.9%,43.2% reproduce the confirmed
single-frame source, not valid-GT coverage. The common colourbar is4–28px.
Raw numerical arrays for this historical view were not recovered: extracted images
and masks are retained as raster objects, while frames, labels and colourbar are
vector. The SVG is explicitly mixed raster/vector, not a raster disguised as an
all-vector scientific plot. No values are inferred by inverting colour pixels.

## Fig.3 numeric source

| Configuration | Coverage (%) | Native MAE(px) | Common MAE(px) |
|---|---:|---:|---:|
| Full |1.151|3.322|4.094|
| Fixed LIF |8.477|3.889|3.807|
| Fixed50ms |0.687|3.260|3.934|
| OLS |0.891|3.264|4.051|
| No trend |1.151|3.356|4.193|
| No completion |0.219|4.431|4.094|

These points use native and common sets as labelled. Their coverage axis is
explicitly logarithmic; shared MAE has a separately labelled linear axis.
Fig3_source.json is the machine-readable source. It is not a new experiment.

## Reuse and editing

Fig.1 and Fig.3–4/6–7 contain vector axes, marks and editable text. Fig.2 and Fig.5
embed genuine scientific raster images with vector annotation. Semantic group IDs
in Fig.1 identify geometry, event selection, correspondence, stored outputs and
GT-only evaluation. Labels use DejaVu Sans with shapes as well as colour coding.
Manuscript width is about175–180mm; labels are designed for about7–9pt after
scaling. Editing individual SVG labels in a vector editor requires the font or an
explicit font replacement.

Dataset-derived panels retain the attribution to Zhu et al.(MVSEC) and Gehrig et
al.(DSEC), with CC BY-SA4.0 dataset terms stated by their official sites. See
REVIEW3_REPRODUCE_20261021_ZH.md for links and exact data references. Other retained
paper graphics are author-provided. The source ZIP supplies rendering scripts;
numeric JSON and associated-array hashes link current plots to the experiments.
