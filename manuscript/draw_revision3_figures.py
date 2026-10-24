"""Editable scientific figures: conceptual mechanism and recorded numeric data."""
from pathlib import Path
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'main/figures/revision3'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none',
                     'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
BLUE='#0072B2'; TEAL='#009E73'; ORANGE='#D55E00'; GRAY='#737373'


def arrow(ax,a,b,color='#333333',style='-'):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=.85,color=color,linestyle=style))


def save(fig,name):
    for extension in ('svg','pdf','png'):
        fig.savefig(OUT/f'{name}.{extension}',dpi=200,facecolor='white')
    if name=='Fig1_Method_NatureTRO':
        shutil.copy2(OUT/f'{name}.png',OUT/'Fig1_Concept_Layout.png')
    plt.close(fig)


def method(direction='A'):
    fig=plt.figure(figsize=(7.1,4.8) if direction=='A' else (7.1,6.1))
    cols=[(.025,.18),(.235,.18),(.455,.235),(.725,.25)]
    panels=[fig.add_axes([x,.36,w,.50]) for x,w in cols]
    for ax,name in zip(panels,['acquisition_geometry','signed_lif_selection','spatiotemporal_correspondence','stored_outputs']):
        ax.set_gid(name)
    if direction=='B':
        for ax,position in zip(panels,[(.025,.62,.23,.23),(.37,.62,.23,.23),
                                     (.69,.62,.285,.23),(.28,.26,.44,.20)]):
            ax.set_position(position)
    for ax,title in zip(panels,['(a) Rectified stereo','(b) Event selection',
                               '(c) Correspondence','(d) Stored outputs']):
        ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
        ax.text(0,1.12,title,fontsize=9,fontweight='bold',transform=ax.transAxes)
    ax=panels[0]
    ax.plot([.22,.78],[.31,.31],color=GRAY,lw=1)
    for x,label,color in [(.22,'Left',BLUE),(.78,'Right',TEAL)]:
        ax.add_patch(Rectangle((x-.12,.20),.24,.16,facecolor='white',edgecolor=color,lw=1.1))
        ax.add_patch(Circle((x,.36),.055,facecolor='white',edgecolor=color,lw=1.2))
        ax.plot([x,.5],[.36,.92],color=color,lw=.9)
        ax.plot([x,x],[.36,.98],color=GRAY,lw=.6,ls=':')
        ax.text(x,.14,label,ha='center',color=color)
    ax.plot(.5,.92,'o',ms=3,color='black'); ax.text(.55,.92,'P',va='center')
    for x,point,label,color in [(.00,.28,'xL',BLUE),(.57,.72,'xR',TEAL)]:
        ax.add_patch(Rectangle((x,.51),.43,.12,facecolor='white',edgecolor=color,lw=.7,zorder=4))
        ax.plot([x+.02,x+.41],[.57,.57],color=GRAY,lw=.6,ls='--',zorder=5)
        ax.plot(point,.57,'o',ms=2.5,color=color,zorder=6)
        ax.text(point,.65,label,ha='center',fontsize=7,color=color,zorder=6)
    ax.annotate('',xy=(.78,.07),xytext=(.22,.07),arrowprops=dict(arrowstyle='<->',lw=.7))
    ax.text(.5,.015,'baseline b',ha='center')
    ax.text(.5,.66,'Z',ha='center',bbox=dict(facecolor='white',edgecolor='none',pad=.5))
    ax.text(.5,-.12,'Conceptual geometry',ha='center',fontsize=8)
    ax.text(.5,-.24,'same row; d = xL − xR > 0\nZ = f b / d',ha='center',fontsize=8)
    ax=panels[1]
    pulses=np.array([.04,.12,.125,.13,.135,.14,.145,.15,.155,.16,.165,.35,.52,.66,.78,.94])
    polarity=np.array([1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,1])
    for t,p in zip(pulses,polarity):
        ax.plot([t,t],[.87,.87+.06*p],color=BLUE if p>0 else ORANGE,lw=1)
    ax.text(.5,.98,'(x, y, t, p)',ha='center')
    ax.text(0,.77,'signed pulses',fontsize=7)
    voltage=[];threshold=[];v=0.;rate=0.;last=0.
    for t,p in zip(pulses,polarity):
        v=v*np.exp(-(t-last)/.7)+p
        rate=rate*np.exp(-(t-last)/.01)+1
        voltage.append(v);threshold.append(1.05+2.4*np.log1p(rate*2.1));last=t
    scale=30;voltage=np.array(voltage);threshold=np.array(threshold)
    dense_t=[];dense_v=[];dense_threshold=[];v=0.;rate=0.;last=0.
    for t,p in zip(pulses,polarity):
        segment=np.linspace(last,t,20);decay=segment-last
        dense_t.extend(segment);dense_v.extend(v*np.exp(-decay/.7))
        dense_threshold.extend(1.05+2.4*np.log1p(rate*np.exp(-decay/.01)*2.1))
        v=v*np.exp(-(t-last)/.7)+p;rate=rate*np.exp(-(t-last)/.01)+1;last=t
        dense_t.append(t);dense_v.append(v);dense_threshold.append(1.05+2.4*np.log1p(rate*2.1))
    ax.plot(dense_t,.32+np.asarray(dense_v)/scale,color=BLUE,lw=1,label='V(t)')
    ax.plot(dense_t,.32+np.asarray(dense_threshold)/scale,color='black',lw=.8,ls='--',label='+θ(t)')
    ax.plot(dense_t,.32-np.asarray(dense_threshold)/scale,color='black',lw=.8,ls=':')
    ax.plot([0,1],[.32,.32],color='0.8',lw=.6)
    keep=((voltage>threshold)&(polarity>0))|((voltage<-threshold)&(polarity<0))
    ax.scatter(pulses[keep],np.full(keep.sum(),.025),s=10,color=BLUE,marker='o')
    ax.scatter(pulses[~keep],np.full((~keep).sum(),.025),s=9,color=GRAY,marker='x')
    ax.text(.02,.42,'V(t)',color=BLUE);ax.text(.92,.49,'+θ',ha='right')
    ax.text(.92,.13,'−θ',ha='right')
    ax.text(.5,-.12,'Signed charge; no reset',ha='center',fontsize=8)
    ax.text(.5,-.24,'● kept; × suppressed\nSynthetic pulse input',ha='center',fontsize=8)
    ax=panels[2]
    # Coordinate axes explicitly show time and horizontal image location.
    ax.plot([.10,.94],[.18,.18],color='black',lw=.7)
    ax.plot([.12,.12],[.16,.93],color='black',lw=.7)
    ax.plot([.52,.52],[.18,.9],color=GRAY,lw=.6,ls=':')
    time=np.array([-.85,-.68,-.55,-.35,-.12,.10,.26,.48,.70,.82])
    yy=.53+.20*time+np.array([.02,-.01,.015,.13,-.02,.0,-.17,.01,-.015,.02])
    ax.scatter(.52+.40*time,yy,s=12,c=TEAL,marker='o')
    ax.plot([.18,.86],[.36,.70],color=BLUE,lw=1)
    ax.scatter([.52],[.53],s=20,facecolors='white',edgecolors=BLUE,zorder=5)
    ax.text(.57,.49,'bfit',color=BLUE,fontsize=8)
    ax.text(.02,.97,'xR',ha='left');ax.text(.52,.02 if direction=='B' else .075,'tR − tL',ha='center')
    ax.text(.18,.14,'−Δt',ha='center',va='top');ax.text(.86,.14,'+Δt',ha='center',va='top')
    ax.text(.52,.14,'0',ha='center',va='top')
    ax.text(.5,1.01,'xR = a(tR − tL) + bfit',ha='center',fontsize=8)
    ax.text(.5,-.12,'Peak vote → xR*; d = xL − xR*',ha='center',fontsize=8)
    ax.text(.5,-.24,'Adaptive window · Huber IRLS\nSame-time reverse check',ha='center',fontsize=8)
    ax=panels[3]
    for row,label,color in [(.83,'M  Matcher estimates',BLUE),
                             (.49,'T  Temporal audit only',GRAY),(.15,'C  Local completion',ORANGE)]:
        ax.text(0,row+.14,label,fontsize=8,color=color)
        for x in range(12):
            for y in range(3):
                ax.add_patch(Rectangle((.015+x*.075,row-.045+y*.040),.075,.040,
                                      facecolor='white',edgecolor='0.8',lw=.25))
        direct=[(1,1),(3,0),(5,1),(7,2),(10,1)]
        fill=direct+([(2,1),(4,0),(6,1),(8,2)] if row==.15 else [])
        for x,y in fill:
            color_point=BLUE if (x,y) in direct else ORANGE
            ax.add_patch(Rectangle((.015+x*.075,row-.045+y*.040),.075,.040,
                                      facecolor=color_point,edgecolor='white',lw=.25,
                                      hatch='///' if (x,y) not in direct else None))
    arrow(ax,(.98,.87),(.98,.20),ORANGE)
    arrow(ax,(.46,.76),(.46,.66),GRAY,'--')
    ax.text(.5,-.12,'T: audited legacy version',ha='center',fontsize=8)
    ax.text(.5,-.24,'Current route: M → C\nSeparate source tags',ha='center',fontsize=8)
    if direction=='B':
        for panel in panels:
            captions=[item for item in panel.texts if item.get_position()[1]<0]
            caption='\n'.join(item.get_text() for item in captions)
            for item in captions:item.set_visible(False)
            panel.text(.5,-.12,caption,ha='center',va='top',fontsize=8,linespacing=1.25)
    # GT is confined to the evaluation branch.
    lower=fig.add_axes([.025,.055,.95,.12]);lower.set_gid('ground_truth_evaluation_only');lower.set(xlim=(0,1),ylim=(0,1));lower.axis('off')
    lower.text(.0,.7,'Recorded GT + timestamps',fontsize=8)
    arrow(lower,(.29,.76),(.35,.76),GRAY,'--')
    lower.text(.38,.69,'≤15 ms association → one prediction / valid frame–pixel',fontsize=8)
    lower.text(.38,.23,'Native coverage · shared-support MAE · Bad-3',fontsize=8)
    overlay=fig.add_axes([0,0,1,1],frameon=False)
    overlay.set(xlim=(0,1),ylim=(0,1));overlay.axis('off')
    if direction=='A':
        overlay.plot([.696,.71,.71],[.61,.61,.78],color='#333333',lw=.85)
        for a,b in [((.209,.61),(.231,.61)),((.421,.61),(.451,.61)),((.71,.78),(.724,.78)),
                    ((.975,.39),(.975,.15)),((.975,.15),(.91,.15))]:
            arrow(overlay,a,b)
    else:
        for a,b in [((.255,.735),(.36,.735)),((.60,.735),(.68,.735)),
                    ((.976,.735),(.99,.735)),((.99,.43),(.72,.43)),((.72,.29),(.975,.29)),
                    ((.975,.29),(.975,.14)),((.975,.14),(.91,.14))]:
            arrow(overlay,a,b)
        overlay.plot([.99,.99],[.735,.43],color='#333333',lw=.85)
    fig.text(.025,.95,'Offline non-causal support; conceptual mechanism, not measured scene data',fontsize=8,color=GRAY)
    if direction=='B':
        fig.text(.025,.915,'Robot perception view: stereo geometry → events → correspondence → inspection',fontsize=8)
    save(fig,'Fig1_Method_NatureTRO' if direction=='A' else 'Fig1_Layout_B')


def historical_ablation():
    labels=['Full','Fixed LIF','Fixed 50 ms','OLS','No trend','No completion']
    coverage=[1.151,8.477,.687,.891,1.151,.219]
    native=[3.322,3.889,3.260,3.264,3.356,4.431]
    common=[4.094,3.807,3.934,4.051,4.193,4.094]
    colors=['#333333',BLUE,ORANGE,TEAL,'#CC79A7','#8B6F00'];markers=['o','s','^','D','v','P']
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.05),gridspec_kw={'width_ratios':[1.15,1]})
    for i in range(6):
        axes[0].scatter(coverage[i],native[i],color=colors[i],marker=markers[i],s=32)
        axes[1].scatter(common[i],i,color=colors[i],marker=markers[i],s=32)
        axes[1].text(common[i]+.012,i,f'{common[i]:.3f}',va='center',fontsize=8)
    positions=[(12,-16),(-55,9),(-12,-20),(-36,4),(7,14),(8,-10)]
    for x,y,label,offset in zip(coverage,native,labels,positions):
        axes[0].annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=8)
    axes[0].set(xscale='log',xlim=(.17,12),ylim=(3.0,4.65),
                xlabel='Valid-GT coverage (%) · log scale',ylabel='Native MAE (px)')
    axes[0].set_xticks([.2,.5,1,2,5,10],labels=['0.2','0.5','1','2','5','10'])
    axes[0].minorticks_off()
    axes[1].set(yticks=range(6),yticklabels=labels,xlim=(3.74,4.34),xlabel='Shared-support MAE (px)')
    axes[1].invert_yaxis();axes[1].axvline(4.094,color=GRAY,lw=.6,ls='--')
    for ax,label in zip(axes,['(a) Native support','(b) Same 89,532 units']):
        ax.set_title(label,loc='left',fontsize=9);ax.grid(axis='x',color='.9',lw=.5)
    fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.86,wspace=.66)
    save(fig,'Fig3_Ablation')
    (OUT/'Fig3_source.json').write_text(json.dumps(dict(source='Historical manuscript Table 7',
        labels=labels,coverage_percent=coverage,native_mae_px=native,common_mae_px=common,
        common_units=89532),indent=2),encoding='utf-8')
    (OUT/'Fig3_source.csv').write_text('configuration,coverage_percent,native_mae_px,common_mae_px,common_units\n'+
        '\n'.join(f'{label},{cov:.3f},{nat:.3f},{com:.3f},89532' for label,cov,nat,com in zip(labels,coverage,native,common))+'\n',encoding='utf-8')


if __name__=='__main__':
    method('A');method('B');historical_ablation()
