"""Replot the audit-rounded baseline estimates. Optional: requires matplotlib."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'docs/data/baseline.json').read_text())
OUT = ROOT / 'docs/assets'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'svg.fonttype':'none',
                     'axes.labelcolor':'#647086', 'text.color':'#172038',
                     'xtick.color':'#647086', 'ytick.color':'#172038'})

def draw(rows, name, xmin, xmax, height, left):
    fig, ax = plt.subplots(figsize=(8.4,height))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=left, right=.74, top=.82, bottom=.23)
    colors = ['#2455e6', '#087461', '#7d4db0']
    for i,row in enumerate(rows):
        y=len(rows)-1-i
        v,lo,hi=(row[k] for k in ('estimate_pp','ci_low_pp','ci_high_pp'))
        ax.plot([lo,hi],[y,y],color=colors[i],linewidth=2.1,solid_capstyle='round',zorder=3)
        ax.plot([lo,lo],[y-.055,y+.055],color=colors[i],linewidth=1.5)
        ax.plot([hi,hi],[y-.055,y+.055],color=colors[i],linewidth=1.5)
        ax.scatter([v],[y],s=65,color=colors[i],zorder=4,edgecolors='white',linewidths=1.3)
        ax.text(1.065,y,f'{v:+.2f} pp',transform=ax.get_yaxis_transform(),va='center',ha='left',fontsize=12,fontweight='medium')
        ax.text(1.065,y-.20,f'[{lo:+.2f}, {hi:+.2f}]',transform=ax.get_yaxis_transform(),va='center',ha='left',fontsize=9.5,color='#647086')
    ax.set_yticks(range(len(rows)),[r['label'] for r in reversed(rows)])
    ax.tick_params(axis='y',length=0,pad=14,labelsize=11)
    ax.tick_params(axis='x',length=0,pad=9,labelsize=10)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(-.55,len(rows)-.35)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis='x',color='#e3e8f1',linewidth=.8)
    ax.axvline(0,color='#a5b2ca',linewidth=1,linestyle=(0,(3,3)))
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xlabel('Change in approval probability (percentage points)',fontsize=10,labelpad=14)
    for ext in ('svg','png'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=200,facecolor='white',metadata={'Creator':'Peer Approval Security: reconstructed from the dated run audit'} if ext=='svg' else None)
    plt.close(fig)

if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    draw(DATA['cooperative_effects']['peer_sensitivity'],'peer-sensitivity',0,35,3.5,.24)
    draw(DATA['cooperative_effects']['history_dependence'],'history-dependence',0,35,3.5,.24)
    draw(DATA['intervention_effects'],'interventions',-27,3,4.2,.31)
    print('Rendered three audit-based figures as SVG and PNG.')
