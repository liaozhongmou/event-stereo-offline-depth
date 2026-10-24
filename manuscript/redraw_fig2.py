"""Recompose the three retained PDF image objects, including their exact masks."""
from pathlib import Path
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'main/figures/revision3'
doc=fitz.open(ROOT/'main/figures/Figure6_holdout_midframe.pdf')
images=[]
for i,record in enumerate(doc[0].get_images()[:3]):
    pix=fitz.Pixmap(doc,record[0])
    pix=fitz.Pixmap(pix,fitz.Pixmap(doc,record[1]))
    path=OUT/f'Fig2_source_{i}.png'
    pix.save(path)
    images.append(plt.imread(path))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
fig,axes=plt.subplots(2,3,figsize=(7.1,3.5),gridspec_kw={'height_ratios':[1,.42]})
for i,(image,title,percent) in enumerate(zip(images,['Ground truth','Archived pipeline','EMatch'],[47.1,8.9,43.2])):
    ax=axes[0,i];ax.imshow(image,interpolation='none')
    ax.add_patch(Rectangle((65,78),75,45,fill=False,edgecolor='#202020',lw=.6,ls='--'))
    ax.set_title(f'({chr(97+i)}) {title}\nImage-plane visibility {percent}%',fontsize=9)
    axes[1,i].imshow(image[78:123,65:140],interpolation='none')
    axes[1,i].set_xlabel('Same retained image crop',fontsize=8)
    for a in axes[:,i]:
        a.set_xticks([]);a.set_yticks([])
        for spine in a.spines.values():spine.set_visible(False)
fig.subplots_adjust(left=.015,right=.87,top=.84,bottom=.10,wspace=.08,hspace=.07)
cax=fig.add_axes([.90,.15,.015,.64])
bar=plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(4,28),cmap='viridis'),cax=cax)
bar.set_label('Disparity (px)');bar.set_ticks([4,10,16,22,28])
for extension in ['svg','pdf','png']:
    fig.savefig(OUT/f'Fig2_Qualitative.{extension}',dpi=200,facecolor='white')
plt.close(fig)
