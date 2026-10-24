"""Assemble the unchanged historical panels into one PDF per figure."""
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
original = (ROOT / 'archive_before_JEI_20261021/supplementary/supplementary.tex').read_text(encoding='utf-8')
panels = re.findall(r'\\includegraphics\[width=\\linewidth\]\{([^}]+)\}\s*\\caption\{([^}]+)\}', original)
assert len(panels) == 27
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'pdf.fonttype': 42})
for sequence in range(3):
    fig, axes = plt.subplots(3, 3, figsize=(6.5, 4.5))
    for index, (ax, (filename, interval)) in enumerate(zip(axes.flat, panels[sequence * 9:(sequence + 1) * 9])):
        ax.imshow(plt.imread(ROOT / 'supplementary/figures' / filename), interpolation='none')
        ax.axis('off')
        text = interval.replace('$', '').replace('--', '\N{EN DASH}').replace(r'\,s', ' s')
        ax.text(0.5, -0.04, f'({chr(97 + index)}) {text}', transform=ax.transAxes, ha='center', va='top', fontsize=9)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0.05, wspace=0.02, hspace=0.26)
    fig.savefig(ROOT / f'supplementary/figures/development_indoor{sequence + 1}.pdf')
    plt.close(fig)
