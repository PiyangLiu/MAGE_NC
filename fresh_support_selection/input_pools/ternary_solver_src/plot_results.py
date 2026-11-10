#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'; F=R/'figures'; F.mkdir(parents=True,exist_ok=True)
D=pd.read_csv(R/'combined_global_states.csv')
ME=pd.read_csv(R/'mage_phase_events.csv'); PE=pd.read_csv(R/'pr_phase_events.csv')
S=json.load(open(R/'summary.json'))
CFG=json.load(open(ROOT/'benchmark_config.json'))

# Journal-style constants
BLACK='#222222'; GRAY='#777777'; LIGHT='#D5D5D5'; BLUE='#4477AA'; ORANGE='#CC8963'; GREEN='#228833'; MAGENTA='#AA3377'
PHASE_COLORS=['#A9C2D8','#B7B7B7','#D99A73'] # liquid, two, vapor
mpl.rcParams.update({
    'font.family':'DejaVu Sans','font.size':7.0,'axes.labelsize':7.2,'axes.titlesize':7.2,
    'xtick.labelsize':6.2,'ytick.labelsize':6.2,'legend.fontsize':6.1,
    'axes.linewidth':0.65,'xtick.major.width':0.6,'ytick.major.width':0.6,
    'xtick.major.size':3,'ytick.major.size':3,'lines.linewidth':1.0,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.transparent':False,
})

def clean(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')

def panel(ax,label,x=-0.12,y=1.06):
    ax.text(x,y,label,transform=ax.transAxes,ha='left',va='bottom',fontsize=8.2,fontweight='bold',color=BLACK)

def save(fig,name):
    fig.savefig(F/f'{name}.svg',bbox_inches='tight')
    fig.savefig(F/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(F/f'{name}_600dpi.png',dpi=600,bbox_inches='tight')
    plt.close(fig)

def stage_spans(ax):
    bounds=[0,12000,30000,60000]
    shades=['#F5F5F5','#ECECEC','#F5F5F5']
    for a,b,c in zip(bounds[:-1],bounds[1:],shades): ax.axvspan(a/3600,b/3600,color=c,zorder=-10)
    for x in bounds[1:-1]: ax.axvline(x/3600,color='#AAAAAA',lw=.55,ls='--',zorder=-5)

# Common matrices
phase_map={'liquid':0,'two':1,'vapor':2}
steps=sorted(D.accepted_step.unique()); times=np.array([D.loc[D.accepted_step==k,'time_s_mage'].iloc[0] for k in steps])
Mmap=np.full((6,len(steps)),np.nan); Pmap=np.full_like(Mmap,np.nan)
for i,k in enumerate(steps):
    g=D[D.accepted_step==k].sort_values('cell')
    Mmap[:,i]=[phase_map[x] for x in g.phase_mage]
    Pmap[:,i]=[phase_map[x] for x in g.phase_pr]

# -------- Figure 5 candidate overview --------
fig=plt.figure(figsize=(7.20,5.45))
gs=fig.add_gridspec(3,12,height_ratios=[1.05,1.3,1.05],hspace=.68,wspace=.9)

# (a) setup / schedule
ax=fig.add_subplot(gs[0,:4]); ax.set_axis_off(); panel(ax,'(a)',x=-.05,y=1.03)
# 1D cells
x0=.08; y0=.47; width=.78; h=.20
for j in range(6):
    rect=plt.Rectangle((x0+j*width/6,y0),width/6,h,fc='white',ec=BLACK,lw=.7,transform=ax.transAxes)
    ax.add_patch(rect); ax.text(x0+(j+.5)*width/6,y0+h/2,str(j+1),transform=ax.transAxes,ha='center',va='center',fontsize=6.5)
ax.annotate('',xy=(x0,y0+h/2),xytext=(.00,y0+h/2),xycoords=ax.transAxes,arrowprops=dict(arrowstyle='-|>',lw=.85,color=BLUE))
ax.annotate('',xy=(.96,y0+h/2),xytext=(x0+width,y0+h/2),xycoords=ax.transAxes,arrowprops=dict(arrowstyle='-|>',lw=.85,color=BLACK))
ax.text(.02,.76,r'$P_{in}=3.2$ MPa',transform=ax.transAxes,ha='left')
ax.text(.98,.76,r'$P_{out}=2.0$ MPa',transform=ax.transAxes,ha='right')
ax.text(.50,.93,r'$T=320$ K   $L=0.6$ m',transform=ax.transAxes,ha='center')
ax.text(.03,.28,r'$\mathbf{z}_0=(0.04,0.08,0.88)$',transform=ax.transAxes,ha='left')
# compact three-stage injection rail
ax.text(.03,.13,'Injection',transform=ax.transAxes,ha='left',va='center',color=GRAY)
segments=[(.25,.49,BLUE,'A  two phase'),(.49,.71,BLACK,'B  liquid'),(.71,.98,ORANGE,'C  vapour-rich')]
for xa,xb,c,lab in segments:
    ax.plot([xa,xb],[.07,.07],transform=ax.transAxes,color=c,lw=3.2,solid_capstyle='butt')
    ax.text((xa+xb)/2,.14,lab,transform=ax.transAxes,ha='center',va='center',fontsize=5.3,color=c)

# (b) probe saturation cycle
ax=fig.add_subplot(gs[0,4:]); panel(ax,'(b)',x=-.08,y=1.03); clean(ax); stage_spans(ax)
g=D[D.cell==1].sort_values('accepted_step')
t=g.time_s_mage.to_numpy()/3600
ax.plot(t,g.S_pr,color=BLACK,ls='--',lw=1.0,label='fully implicit PR')
ax.plot(t,g.S_mage,color=BLUE,lw=1.0,marker='o',ms=2.1,mew=0,label='MAGE')
for x in [12000,30000]: ax.axvline(x/3600,color='#999999',ls='--',lw=.55)
# event markers MAGE/PR identical; use MAGE events cell 1
for _,e in ME[ME.cell==1].iterrows():
    ax.plot(e.time_s/3600,e.S_mage,marker='o',ms=4.0,mfc='white',mec=MAGENTA,mew=.9,zorder=6)
ax.set_xlim(0,60000/3600); ax.set_ylim(-.035,1.035); ax.set_xlabel('Time (h)'); ax.set_ylabel(r'Gas saturation, $S_g$')
ax.legend(frameon=False,loc='center',bbox_to_anchor=(.50,.16),ncol=2,handlelength=2.4)
ax.text(1.7,.92,'two-phase injection',ha='center',va='center',fontsize=5.8,color=GRAY)
ax.text(5.8,.92,'liquid wash',ha='center',va='center',fontsize=5.8,color=GRAY)
ax.text(12.5,.92,'vapour-rich injection',ha='center',va='center',fontsize=5.8,color=GRAY)

# (c) phase maps - two stacked strips in one panel
outer=fig.add_subplot(gs[1,:]); outer.set_axis_off(); panel(outer,'(c)',x=-.02,y=1.01)
sg=gs[1,:].subgridspec(2,1,hspace=.18)
cmap=ListedColormap(PHASE_COLORS); norm=BoundaryNorm([-.5,.5,1.5,2.5],cmap.N)
for ii,(mat,label) in enumerate([(Pmap,'Fully implicit PR'),(Mmap,'MAGE')]):
    ax=fig.add_subplot(sg[ii,0]); clean(ax)
    extent=[times.min()/3600,times.max()/3600,.5,6.5]
    ax.imshow(mat,origin='lower',aspect='auto',interpolation='nearest',cmap=cmap,norm=norm,extent=extent)
    for x in [12000,30000]: ax.axvline(x/3600,color='white',lw=.8,ls='--')
    ax.set_yticks(range(1,7)); ax.set_ylabel('Cell')
    ax.text(.01,.86,label,transform=ax.transAxes,ha='left',va='center',fontsize=6.4,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.78,pad=1.0))
    if ii==0: ax.set_xticklabels([]); ax.set_xlabel('')
    else: ax.set_xlabel('Time (h)')
    ax.set_xlim(0,60000/3600)
legend=[Line2D([0],[0],marker='s',linestyle='none',markerfacecolor=c,markeredgecolor='none',markersize=6,label=l)
        for c,l in zip(PHASE_COLORS,['liquid','two phase','vapour'])]
outer.legend(handles=legend,loc='upper right',bbox_to_anchor=(.995,.995),frameon=False,ncol=3,handletextpad=.3,columnspacing=.8)

# (d) error distributions
ax=fig.add_subplot(gs[2,:6]); panel(ax,'(d)',x=-.08,y=1.04); clean(ax)
vals=[D.P_relative_error.to_numpy(),
      D[['z_CH4_abs_error','z_CO2_abs_error','z_nC4_abs_error']].to_numpy().ravel(),
      D.S_abs_error.to_numpy()]
labels=['Pressure rel. error','Composition abs. error',r'$S_g$ abs. error']
# floor only for log rendering, preserve zeros conceptually
vv=[np.maximum(v,1e-9) for v in vals]
bp=ax.boxplot(vv,vert=False,positions=[3,2,1],widths=.42,showfliers=False,patch_artist=True,
              medianprops=dict(color=BLACK,lw=.9),whiskerprops=dict(color=GRAY,lw=.65),capprops=dict(color=GRAY,lw=.65))
for b in bp['boxes']: b.set(facecolor='#E5E5E5',edgecolor=GRAY,lw=.65)
for pos,v,c in zip([3,2,1],vv,[BLUE,GREEN,ORANGE]):
    q95=np.quantile(v,.95); mean=np.mean(v)
    ax.plot(mean,pos,marker='D',ms=3.4,mfc=c,mec='white',mew=.4,zorder=5)
    ax.plot(q95,pos,marker='|',ms=7,mew=1.1,color=c,zorder=5)
ax.set_xscale('log'); ax.set_xlim(1e-8,2e-1); ax.set_yticks([1,2,3],labels[::-1]); ax.set_xlabel('Error (-)')
ax.text(.02,.98,'diamond: mean    vertical tick: 95th percentile',transform=ax.transAxes,ha='left',va='top',fontsize=5.6,color=GRAY)

# (e) workload
outer2=fig.add_subplot(gs[2,6:]); outer2.set_axis_off(); panel(outer2,'(e)',x=-.02,y=1.04)
se=gs[2,6:].subgridspec(1,2,wspace=.6)
ax1=fig.add_subplot(se[0,0]); clean(ax1)
# local flash requests; scientific lollipop
names=['MAGE','PR']; xvals=[0,S['PR_solver']['online_local_flash_requests_in_global_residual']]
for y,xv,c in [(1,xvals[0],BLUE),(0,xvals[1],BLACK)]:
    ax1.hlines(y,0,xv,color=c,lw=1.1); ax1.plot(xv,y,'o',color=c,ms=4)
    ax1.text(xv + max(xvals)*.035,y,f'{xv:,}',va='center',ha='left',fontsize=6.0,color=c)
ax1.set_yticks([0,1],['PR','MAGE']); ax1.set_xlabel('Local flash requests'); ax1.set_xlim(-1500,max(xvals)*1.24); ax1.set_ylim(-.6,1.6)
ax1.text(.02,.94,'nested equilibrium solves',transform=ax1.transAxes,ha='left',va='top',fontsize=5.8,color=GRAY)

ax2=fig.add_subplot(se[0,1]); clean(ax2)
vals=[S['MAGE_solver']['online_MAGE_branch_state_evaluations'],S['PR_solver']['online_PR_EOS_evaluations']]
for y,xv,c in [(1,vals[0],BLUE),(0,vals[1],BLACK)]:
    ax2.hlines(y,1e4,xv,color=c,lw=1.1); ax2.plot(xv,y,'o',color=c,ms=4)
    ax2.text(xv*1.12,y,f'{xv:,}',va='center',ha='left',fontsize=6.0,color=c)
ax2.set_xscale('log'); ax2.set_xlim(1e4,8e5); ax2.set_yticks([0,1],['PR EOS','MAGE branches']); ax2.set_xlabel('Thermodynamic evaluations')
ax2.text(.02,.94,'online evaluations',transform=ax2.transAxes,ha='left',va='top',fontsize=5.8,color=GRAY)

fig.subplots_adjust(left=.07,right=.985,top=.975,bottom=.08)
save(fig,'benchmark_overview')

# -------- Phase-front comparison only --------
fig,axs=plt.subplots(2,1,figsize=(7.2,2.85),sharex=True,constrained_layout=True)
for ax,mat,label in zip(axs,[Pmap,Mmap],['Fully implicit PR','MAGE']):
    ax.imshow(mat,origin='lower',aspect='auto',interpolation='nearest',cmap=cmap,norm=norm,
              extent=[times.min()/3600,times.max()/3600,.5,6.5])
    for x in [12000,30000]: ax.axvline(x/3600,color='white',ls='--',lw=.85)
    clean(ax); ax.set_yticks(range(1,7)); ax.set_ylabel('Cell')
    ax.text(.01,.84,label,transform=ax.transAxes,bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1),fontsize=7)
axs[-1].set_xlabel('Time (h)')
fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.55,1.02),frameon=False,ncol=3)
save(fig,'phase_front_comparison')

# -------- Probe cycle --------
fig,ax=plt.subplots(figsize=(7.2,2.75),constrained_layout=True); clean(ax); stage_spans(ax)
ax.plot(t,g.S_pr,color=BLACK,ls='--',label='fully implicit PR')
ax.plot(t,g.S_mage,color=BLUE,marker='o',ms=2.4,label='MAGE')
for _,e in ME[ME.cell==1].iterrows(): ax.plot(e.time_s/3600,e.S_mage,'o',ms=4.5,mfc='white',mec=MAGENTA,mew=1)
ax.set(xlabel='Time (h)',ylabel=r'Gas saturation, $S_g$',xlim=(0,60000/3600),ylim=(-.035,1.035))
ax.legend(frameon=False,loc='upper left',ncol=2)
for x,txt in [(2,'two-phase injection'),(5.8,'liquid wash'),(12.5,'vapour-rich injection')]: ax.text(x,.93,txt,ha='center',fontsize=6,color=GRAY)
save(fig,'probe_phase_cycle_comparison')

# -------- Selected pressure profiles --------
sel=[1,10,21,47,75,89,92]
fig,ax=plt.subplots(figsize=(5.2,3.2),constrained_layout=True); clean(ax)
cm=plt.get_cmap('cividis'); cols=cm(np.linspace(.08,.92,len(sel)))
for c,k in zip(cols,sel):
    gg=D[D.accepted_step==k].sort_values('cell')
    ax.plot(gg.cell,gg.P_MPa_pr,ls='--',color=c,lw=.8)
    ax.plot(gg.cell,gg.P_MPa_mage,marker='o',ms=2.4,color=c,lw=.9,label=f'{gg.time_s_mage.iloc[0]/3600:.1f} h')
ax.set(xlabel='Cell',ylabel='Pressure (MPa)',xticks=range(1,7)); ax.legend(frameon=False,ncol=2,title='Time')
save(fig,'pressure_profiles_comparison')

# -------- Workload standalone --------
fig,axs=plt.subplots(1,2,figsize=(7.2,2.5),constrained_layout=True)
ax=axs[0]; clean(ax)
for y,xv,c,n in [(1,0,BLUE,'MAGE'),(0,S['PR_solver']['online_local_flash_requests_in_global_residual'],BLACK,'PR')]:
    ax.hlines(y,0,xv,color=c); ax.plot(xv,y,'o',color=c); ax.text(xv+1400,y,f'{xv:,}',va='center',fontsize=7)
ax.set_yticks([0,1],['PR','MAGE']); ax.set_xlabel('Online local flash requests'); ax.set_xlim(-1500,49000); ax.set_ylim(-.5,1.5)
ax=axs[1]; clean(ax)
for y,xv,c,label in [(1,S['MAGE_solver']['online_MAGE_branch_state_evaluations'],BLUE,'MAGE branches'),(0,S['PR_solver']['online_PR_EOS_evaluations'],BLACK,'PR EOS')]:
    ax.hlines(y,1e4,xv,color=c); ax.plot(xv,y,'o',color=c); ax.text(xv*1.12,y,f'{xv:,}',va='center',fontsize=7)
ax.set_xscale('log'); ax.set_xlim(1e4,8e5); ax.set_yticks([0,1],['PR EOS','MAGE branches']); ax.set_xlabel('Online thermodynamic evaluations')
save(fig,'workload_comparison')

# -------- summary tables --------
summary_rows=[
    ['space-time states',S['comparison']['space_time_states'],'-'],
    ['phase-state match',S['comparison']['phase_state_match_fraction'],'fraction'],
    ['phase-state mismatches',S['comparison']['phase_state_mismatches'],'count'],
    ['phase events (MAGE)',S['comparison']['phase_events_MAGE'],'count'],
    ['phase events (PR)',S['comparison']['phase_events_PR'],'count'],
    ['mean pressure relative error',S['comparison']['pressure_relative_error_mean'],'-'],
    ['max pressure relative error',S['comparison']['pressure_relative_error_max'],'-'],
    ['mean composition absolute error',S['comparison']['composition_absolute_error_mean'],'-'],
    ['max composition absolute error',S['comparison']['composition_absolute_error_max'],'-'],
    ['mean gas-saturation absolute error',S['comparison']['gas_saturation_absolute_error_mean'],'-'],
    ['max gas-saturation absolute error',S['comparison']['gas_saturation_absolute_error_max'],'-'],
    ['MAGE online local flash calls',S['MAGE_solver']['online_local_flash_calls'],'count'],
    ['PR local flash requests',S['PR_solver']['online_local_flash_requests_in_global_residual'],'count'],
    ['PR executed local flashes (cached)',S['PR_solver']['online_executed_local_flash_calls_with_exact_cache'],'count'],
    ['MAGE branch evaluations',S['MAGE_solver']['online_MAGE_branch_state_evaluations'],'count'],
    ['PR EOS evaluations',S['PR_solver']['online_PR_EOS_evaluations'],'count'],
]
pd.DataFrame(summary_rows,columns=['metric','value','unit']).to_csv(R/'summary_table.csv',index=False)
# event comparison in a convenient compact form
ec=ME[['accepted_step','time_s','stage','cell','from_phase','to_phase','S_mage']].merge(
   PE[['accepted_step','time_s','cell','from_phase','to_phase','S_pr']],
   on=['accepted_step','time_s','cell','from_phase','to_phase'],how='outer',indicator=True)
ec.to_csv(R/'phase_event_comparison.csv',index=False)
print(f'Figures written to {F}')
