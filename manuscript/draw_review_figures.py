"""Render recorded ablation operating points, without interpolating threshold curves."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
labels = ['Full', 'Fixed LIF', 'Fixed window', 'OLS', 'No trend', 'No completion']
coverage = np.array([1.151, 8.477, .687, .891, 1.151, .219])
native = np.array([3.322, 3.889, 3.260, 3.264, 3.356, 4.431])
common = np.array([4.094, 3.807, 3.934, 4.051, 4.193, 4.094])
colors = ['#202020', '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#7A6610']
markers = ['o', 's', '^', 'D', 'v', 'P']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), gridspec_kw={'width_ratios': [1.15, 1]})
offsets = [(6, -17), (-38, 10), (-20, -20), (8, -3), (7, 10), (6, -13)]
for i, label in enumerate(labels):
    axes[0].scatter(coverage[i], native[i], c=colors[i], marker=markers[i], s=38, zorder=3)
    axes[0].annotate(label, (coverage[i], native[i]), xytext=offsets[i],
                     textcoords='offset points', fontsize=8)
    axes[1].scatter(common[i], i, c=colors[i], marker=markers[i], s=38, zorder=3)
    axes[1].annotate(f'{common[i]:.3f}', (common[i], i), xytext=(6, -3),
                     textcoords='offset points', fontsize=8)
axes[0].set(xscale='log', xlim=(.17, 12), ylim=(3.05, 4.6),
            xlabel='Valid-GT coverage (%)', ylabel='Native MAE (px)')
axes[0].set_xticks([.2, .5, 1, 2, 5, 10], labels=['0.2', '0.5', '1', '2', '5', '10'])
axes[0].minorticks_off()
axes[1].set(yticks=np.arange(6), yticklabels=labels, xlim=(3.75, 4.30),
            xlabel='Common-support MAE (px)')
axes[1].invert_yaxis()
axes[1].axvline(common[0], color='0.6', lw=.8, ls='--', zorder=1)
for ax, title in zip(axes, ['(a) Native operating points', '(b) Same 89,532 units']):
    ax.set_title(title, fontsize=10, loc='left', pad=10)
    ax.grid(axis='x', color='0.9', lw=.6)
fig.subplots_adjust(left=.09, right=.98, bottom=.17, top=.85, wspace=.62)
fig.savefig(ROOT / 'main/figures/ablation_operating_points.pdf')
fig.savefig(ROOT / 'main/figures/ablation_operating_points.png', dpi=180)
print('Rendered six recorded ablation configurations.')
