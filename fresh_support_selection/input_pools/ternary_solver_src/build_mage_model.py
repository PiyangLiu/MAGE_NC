#!/usr/bin/env python3
"""Rebuild the 1280-support MAGE model from the frozen teacher candidate pool.
No flow-trajectory states are used in support selection.
"""
from pathlib import Path
import json, numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1]
POOL=np.load(ROOT/'teacher/ternary_branch_pools.npz')
BASE=json.load(open(ROOT/'teacher/original_ternary_summary.json'))
M=1280; TAU=0.5
orders={}
for ph in ['liquid','vapor']:
    eta=POOL[f'{ph}_eta']; phi=POOL[f'{ph}_phi']; q=POOL[f'{ph}_q']
    order=list(map(int,BASE[f'support_order_{ph}']))
    chosen=np.zeros(len(phi),bool); chosen[np.asarray(order)]=True
    maxL=np.full(len(phi),-np.inf)
    for j in order: maxL=np.maximum(maxL,phi[j]+(eta-eta[j])@q[j])
    while len(order)<M:
        gap=phi-maxL; gap[chosen]=-np.inf; j=int(np.argmax(gap)); order.append(j); chosen[j]=True
        maxL=np.maximum(maxL,phi[j]+(eta-eta[j])@q[j])
    orders[ph]=np.asarray(order[:M],int)
# one shared additive gauge calibrated over both candidate pools
all_diff=[]
for ph in ['liquid','vapor']:
    eta=POOL[f'{ph}_eta']; phi=POOL[f'{ph}_phi']; q=POOL[f'{ph}_q']; idx=orders[ph]
    se=eta[idx]; sf=phi[idx]; sq=q[idx]; pred=[]
    for st in range(0,len(phi),300):
        E=eta[st:st+300]; L=sf[None,:]+np.einsum('nmd,md->nm',E[:,None,:]-se[None,:,:],sq)
        pred.append(TAU*(logsumexp(L/TAU,axis=1)-np.log(M)))
    all_diff.append(phi-np.concatenate(pred))
offset=float(np.mean(np.concatenate(all_diff)))
np.savez_compressed(ROOT/'model/mage_ternary_1280.npz',
    liquid_eta=POOL['liquid_eta'][orders['liquid']],liquid_phi=POOL['liquid_phi'][orders['liquid']],liquid_q=POOL['liquid_q'][orders['liquid']],
    vapor_eta=POOL['vapor_eta'][orders['vapor']],vapor_phi=POOL['vapor_phi'][orders['vapor']],vapor_q=POOL['vapor_q'][orders['vapor']])
# Reconstruction diagnostics on the declared candidate pool (not an independent test set)
metrics={}
for ph in ['liquid','vapor']:
    eta=POOL[f'{ph}_eta']; phi=POOL[f'{ph}_phi']; q=POOL[f'{ph}_q']; idx=orders[ph]
    se=eta[idx]; sf=phi[idx]; sq=q[idx]
    pp=[]; qq=[]
    for st in range(0,len(phi),300):
        E=eta[st:st+300]
        L=sf[None,:]+np.einsum('nmd,md->nm',E[:,None,:]-se[None,:,:],sq)
        A=L/TAU; lz=logsumexp(A,axis=1); w=np.exp(A-lz[:,None])
        pp.append(TAU*(lz-np.log(M))+offset)
        qq.append(w@sq)
    pp=np.concatenate(pp); qq=np.vstack(qq)
    pre=np.abs(pp-phi)/np.maximum(np.abs(phi),1e-30)
    gre=np.linalg.norm(qq-q,axis=1)/np.maximum(np.linalg.norm(q,axis=1),1e-30)
    cre=np.linalg.norm(qq[:,1:]-q[:,1:],axis=1)/np.maximum(np.linalg.norm(q[:,1:],axis=1),1e-30)
    metrics[ph]={
        'phi_relative_mean':float(pre.mean()),'phi_relative_max':float(pre.max()),
        'gradient_relative_mean':float(gre.mean()),'gradient_relative_max':float(gre.max()),
        'concentration_relative_mean':float(cre.mean()),'concentration_relative_max':float(cre.max())}
summary={'supports_per_phase':M,'tau':TAU,'shared_phi_offset':offset,
         'support_selection':'first 512 supports from the frozen teacher model, extended to 1280 by greedy maximum convex support-plane gap over the declared candidate pool',
         'flow_trajectory_used_for_support_selection':False,
         'candidate_pool_counts':{'liquid':int(len(POOL['liquid_phi'])),'vapor':int(len(POOL['vapor_phi']))},
         'candidate_pool_reconstruction':metrics}
json.dump(summary,open(ROOT/'model/mage_model_summary.json','w'),indent=2)
print(json.dumps(summary,indent=2))
