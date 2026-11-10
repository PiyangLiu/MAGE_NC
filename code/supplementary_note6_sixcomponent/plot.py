"""Publication-style two-panel plot from *recorded* audit data."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
with np.load(HERE/'data'/'confirmatory.npz') as d:
    a={k:d[k] for k in d.files}
report=json.loads((HERE/'results'/'verified_benchmark.json').read_text())
time=report['five_repeats_us_per_query']
labels=['MAGE, direct','PR, analytic Jacobian','PR, finite-difference Jacobian']
keys=['MAGE_direct','PR_analytic_cold','PR_fd_cold']
colours=['#177e89','#dd6b20','#69768a']

fig,(left,right)=plt.subplots(1,2,figsize=(10.3,3.65),gridspec_kw={'width_ratios':[1.2,1]})
for phase,col,name in [(0,'#177e89','Liquid stable'),(1,'#dd6b20','Vapour stable')]:
    mask=a['phase']==phase
    left.scatter(a['anchor_pressure'][mask]/1e6,a['perturbation'][mask],
                 s=7,c=col,alpha=.58,rasterized=True,label=f'{name} ({mask.sum()})')
left.set_xlabel('Reference coexistence anchor pressure (MPa)')
left.set_ylabel(r'Perturbation in $\eta_{\mathrm{nC_6}}$')
left.set_xlim(1.91,2.79)
left.legend(loc='center',bbox_to_anchor=(.5,.53),frameon=False,fontsize=8)
left.set_title('All 1,000 independent phase labels matched',fontsize=10)
left.grid(alpha=.15)

for j,(label,key,col) in enumerate(zip(labels,keys,colours)):
    values=np.asarray(time[key]);med=np.median(values)
    right.barh(j,med,height=.53,color=col,alpha=.84)
    right.scatter(values,np.full(len(values),j),s=21,color='#222222',alpha=.75,zorder=3)
    right.text(max(values)+.65,j,f'{med:.2f} (median)',va='center',fontsize=8.5)
right.invert_yaxis();right.set_yticks(np.arange(len(keys)),labels)
right.set_xlabel(r'Online time per query ($\mu$s)')
right.set_xlim(0,max(max(time[k]) for k in keys)+10.)
right.set_title('Five alternating repeats; lower is faster',fontsize=10)
right.grid(axis='x',alpha=.15)
fig.tight_layout(pad=1.5,w_pad=2.8)
for ext in ('png','pdf'):
    fig.savefig(HERE/'results'/f'Figure_noncritical_grand_PR_MAGE.{ext}',dpi=240)
print('Wrote scientific plot in results/Figure_noncritical_grand_PR_MAGE.{png,pdf}')
