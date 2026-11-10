"""Validation-greedy MAGE supports, directly from the same six-component PR.

Equal physical-reference gauge and same M for both phases: no phase-offset
fitting, no test-set use, no phase labels used by the selector.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from pr_six import R,T,N

HERE=Path(__file__).resolve().parent
DATA=HERE/'data';MODEL=HERE/'model'
PHI_SCALE=1000.0
TAU=.00004

def predict(x,X,F,G):
    planes=x@G.T + (F-np.einsum('ij,ij->i',G,X))[None,:]
    z=planes/TAU
    lz=logsumexp(z,axis=1)
    w=np.exp(z-lz[:,None])
    return TAU*(lz-np.log(len(X))),w@G

def subset(a,phase,center,scale):
    return ((a['eta']-center)/scale,
            a[f'phi_{phase}']/PHI_SCALE,
            a[f'q_{phase}']*scale[None,:]/PHI_SCALE)

def select(cand,val,phase,M):
    center=cand['eta'].mean(axis=0)
    scale=cand['eta'].std(axis=0)
    xc,fc,gc=subset(cand,phase,center,scale)
    xv,fv,gv=subset(val,phase,center,scale)
    fsize=max(np.ptp(fv),1e-12)
    gsize=max(np.linalg.norm(np.ptp(gv,axis=0)),1e-12)
    first=int(np.argmin(np.sum((xc-xc.mean(axis=0))**2,axis=1)))
    used=np.zeros(len(xc),bool);used[first]=True
    ids=[first];trace=[]
    for k in range(M):
        IX=np.array(ids,int)
        pf,pg=predict(xv,xc[IX],fc[IX],gc[IX])
        ep=np.abs(pf-fv)/fsize
        eg=np.linalg.norm(pg-gv,axis=1)/np.maximum(np.linalg.norm(gv,axis=1),.1*gsize)
        score=np.maximum(ep,eg)
        trace.append({'M':len(ids),'mean_validation_score':float(score.mean()),
                      'max_validation_score':float(score.max()),
                      'max_pressure_abs_MPa':float(np.max(np.abs(pf-fv))*PHI_SCALE*R*T/1e6)})
        if k==M-1:break
        worst=xv[int(np.argmax(score))];dist=np.sum((xc-worst)**2,axis=1)
        dist[used]=np.inf
        j=int(np.argmin(dist));ids.append(j);used[j]=True
    ix=np.array(ids)
    return {'center':center,'scale':scale,'X':xc[ix],'F':fc[ix],'G':gc[ix]},trace,ix

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--supports',type=int,default=128)
    args=ap.parse_args();M=args.supports
    MODEL.mkdir(exist_ok=True)
    with np.load(DATA/'candidate.npz') as d:cand={k:d[k] for k in d.files}
    with np.load(DATA/'validation.npz') as d:val={k:d[k] for k in d.files}
    t0=time.perf_counter();archive={};reports={}
    for phase in ('liquid','vapor'):
        model,trace,ix=select(cand,val,phase,M)
        for k,v in model.items():archive[f'{phase}_{k}']=v
        reports[phase]={'indices':ix.tolist(),'trace':trace}
        print(phase,'support',M,'max pressure deviation MPa',trace[-1]['max_pressure_abs_MPa'],flush=True)
    np.savez_compressed(MODEL/'mage_six.npz',**archive)
    meta={'M_each_phase':M,'n_components':N,
          'tau_nondim':TAU,'phi_scale_mol_per_m3':PHI_SCALE,
          'offline_selection_seconds':time.perf_counter()-t0,
          'validation_errors':{p:v['trace'][-1] for p,v in reports.items()},
          'selection_uses_heldout':False,
          'support_indices':{p:v['indices'] for p,v in reports.items()}}
    (MODEL/'model_summary.json').write_text(json.dumps(meta,indent=2))
    print('Offline support selection seconds',meta['offline_selection_seconds'])

if __name__=='__main__':main()
