"""Offline high-accuracy PR reference for 6-component isofugacity queries.

Candidate, validation and test are independent. The query distribution has
two intrinsic parameters (coexistence-anchor pressure, heavy-species fugacity
perturbation) in an ambient 6-component chemical-potential space.
"""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from pr_six import R,T,N,PREF,state,ptz_flash,NAMES

HERE=Path(__file__).resolve().parent
DATA=HERE/'data'
FEED=np.array([.46,.12,.08,.12,.11,.11])


def anchors():
    p=np.linspace(1.86e6,2.84e6,101)
    x=[];y=[];residual=[]
    for P in p:
        beta,liq,vap,err,nit=ptz_flash(P,FEED.copy())
        if err>5e-10 or not (.2<beta<.8):
            raise RuntimeError('PR coexistence anchor failed')
        x.append(liq);y.append(vap);residual.append(err)
    return p,np.array(x),np.array(y)


def inv(eta,p0,x0,phase):
    def decode(v):
        vv=np.r_[v[1:],0.];w=np.exp(vv-vv.max());return 1e6*np.exp(v[0]),w/w.sum()
    start=np.r_[np.log(p0/1e6),np.log(x0[:-1]/x0[-1])]
    fun=lambda v:state(*decode(v),phase)[0]-eta
    sol=least_squares(fun,start,xtol=3e-12,ftol=3e-12,
                      gtol=3e-12,max_nfev=80)
    if np.max(np.abs(sol.fun))>1e-8:
        raise RuntimeError(f'Six-PR inverse failed, {phase}: residual {sol.fun}')
    pp,x=decode(sol.x);_,phi,q,Z=state(pp,x,phase)
    return pp,phi,q,Z,x,sol.nfev


def split(n,seed,table):
    rng=np.random.default_rng(seed)
    pbase=rng.uniform(1.95e6,2.75e6,n)
    d=rng.choice([-1.,1.],n)*rng.uniform(.012,.030,n)
    ptab,xtab,ytab=table
    out={'eta':np.zeros((n,N)), 'anchor_pressure':pbase,'perturbation':d,
         'phase':np.zeros(n,dtype=np.int32)}
    for ph in ('liquid','vapor'):
        out[f'P_{ph}']=np.zeros(n);out[f'phi_{ph}']=np.zeros(n)
        out[f'q_{ph}']=np.zeros((n,N));out[f'Z_{ph}']=np.zeros(n)
        out[f'x_{ph}']=np.zeros((n,N));out[f'nfev_{ph}']=np.zeros(n,dtype=np.int32)
    for i,(pressure,delta) in enumerate(zip(pbase,d)):
        xl=np.array([np.interp(pressure,ptab,xtab[:,k]) for k in range(N)])
        xv=np.array([np.interp(pressure,ptab,ytab[:,k]) for k in range(N)])
        eta=state(pressure,xl,'liquid')[0];eta[-1]+=delta
        out['eta'][i]=eta
        for phase,init in [('liquid',xl),('vapor',xv)]:
            pp,phi,q,Z,x,nfev=inv(eta,pressure,init,phase)
            out[f'P_{phase}'][i]=pp;out[f'phi_{phase}'][i]=phi
            out[f'q_{phase}'][i]=q;out[f'Z_{phase}'][i]=Z
            out[f'x_{phase}'][i]=x;out[f'nfev_{phase}'][i]=nfev
    out['phase']=(out['P_vapor']>out['P_liquid']).astype(np.int32)
    return out


def main():
    DATA.mkdir(exist_ok=True)
    start=time.perf_counter();p,x,y=anchors()
    np.savez_compressed(DATA/'anchors.npz',P=p,x=x,y=y)
    summary={'name':'six-component noncritical PR fugacity-controlled equilibrium',
             'components':list(NAMES), 'temperature_K':T,'feed_composition_anchor':FEED.tolist(),
             'kij':'zero classical PR mixing',
             'distribution':'P_anchor in 1.95-2.75 MPa; eta_nC6 offset +/- 0.012-0.030',
             'independent_splits':{},'support_selection_uses_test':False}
    # The 'heldout' stream was used for an early implementation audit; the
    # separate 1,000-state confirmatory stream is reserved for final reporting.
    for name,count,seed in [('candidate',1200,880101),('validation',360,880102),
                            ('heldout',800,880103),('confirmatory',1000,880104)]:
        split_start=time.perf_counter()
        data=split(count,seed,(p,x,y))
        np.savez_compressed(DATA/f'{name}.npz',**data)
        summary['independent_splits'][name]={
            'count':count,'seed':seed,
            'generation_wall_s':time.perf_counter()-split_start,
            'liquid_stable':int((data['phase']==0).sum()),
            'vapor_stable':int((data['phase']==1).sum()),
            'pressure_margin_min_MPa':float(np.min(np.abs(data['P_vapor']-data['P_liquid']))/1e6),
            'Z_liquid_range':[float(data['Z_liquid'].min()),float(data['Z_liquid'].max())],
            'Z_vapor_range':[float(data['Z_vapor'].min()),float(data['Z_vapor'].max())]}
        print(name,summary['independent_splits'][name],flush=True)
    summary['reference_generation_wall_s']=time.perf_counter()-start
    summary['model_reference_generation_wall_s']=sum(
        summary['independent_splits'][name]['generation_wall_s']
        for name in ('candidate','validation'))
    (DATA/'reference_summary.json').write_text(json.dumps(summary,indent=2))
    print('Reference generation wall seconds',summary['reference_generation_wall_s'])

if __name__=='__main__':main()
