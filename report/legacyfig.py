from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
rows=json.loads((ROOT/'results/summary.json').read_text())['exp8_annotation_audit']
configs=[('audit_pipeline',.05,'Robust pipeline\n979–1007 reviews','#206493'),('random_audit_fixed_cp',.1,'Random clean audit\n1000 reviews','#b75e19')]
fig,axes=plt.subplots(1,2,figsize=(8.4,2.7),layout='constrained')
for i,(arm,alpha,label,color) in enumerate(configs):
 data=sorted([r for r in rows if r['audit']==arm and r['alpha']==alpha],key=lambda r:r['seed'])
 assert len(data)==3
 for ax,key in zip(axes,['sharp_suspect_coverage','sharp_suspect_size']):
  v=np.array([r[key] for r in data]);ax.scatter(i+np.array([-.055,0,.055]),v,s=27,color=color,zorder=4)
  ax.errorbar(i,v.mean(),yerr=v.std(ddof=1),fmt='_',markersize=24,capsize=5,color=color,lw=1.5,zorder=3)
for ax in axes:
 ax.set_xticks([0,1],[c[2] for c in configs],fontsize=10);ax.set_xlim(-.5,1.5);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=10)
axes[0].axhline(.9,color='#666666',ls='--',lw=1);axes[0].text(.03,.903,'90% marginal target',fontsize=9,color='#555555');axes[0].set_ylim(.86,.99);axes[0].set_ylabel('Measured clean coverage');axes[0].set_title('(a) Coverage at a common target',fontsize=11)
axes[1].set_ylabel('Mean prediction-set size');axes[1].set_ylim(1.1,2.2);axes[1].set_title('(b) Set efficiency',fontsize=11)
fig.savefig(ROOT/'figures/fig_primary_audit.pdf');fig.savefig(ROOT/'figures/fig_primary_audit.png',dpi=180);plt.close(fig)
