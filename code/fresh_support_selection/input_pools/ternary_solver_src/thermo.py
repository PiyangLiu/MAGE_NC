from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

R = 8.314462618
T = 320.0
PREF = 1.0e5

class MAGEModel:
    def __init__(self, model_dir: Path):
        s = json.load(open(model_dir/'mage_model_summary.json'))
        p = np.load(model_dir/'mage_ternary_1280.npz')
        self.M = int(s['supports_per_phase'])
        self.tau = float(s['tau'])
        self.common_offset = float(s['shared_phi_offset'])
        self.branch_state_evals = 0
        self.branches = {}
        for ph in ['liquid','vapor']:
            self.branches[ph] = {
                'eta': p[f'{ph}_eta'],
                'phi': p[f'{ph}_phi'],
                'q': p[f'{ph}_q'],
            }
        self.M_by_branch = {ph: len(self.branches[ph]['eta'])
                            for ph in ('liquid', 'vapor')}

    def reset_counter(self):
        self.branch_state_evals = 0

    def branch(self, nu, phase):
        nu = np.atleast_2d(nu)
        b = self.branches[phase]
        eta = np.c_[np.full(len(nu), -1.0/(R*T)), nu]
        L = b['phi'][None,:] + np.einsum(
            'nmd,md->nm', eta[:,None,:]-b['eta'][None,:,:], b['q'])
        A = L/self.tau
        lz = logsumexp(A, axis=1)
        w = np.exp(A-lz[:,None])
        phi = self.tau*(lz-np.log(self.M_by_branch[phase])) + self.common_offset
        q = w@b['q']
        c = q[:,1:]
        cs = b['q'][:,1:]
        Ecc = np.einsum('nm,mi,mj->nij', w, cs, cs)
        H = (Ecc - np.einsum('ni,nj->nij', c, c))/self.tau
        self.branch_state_evals += len(nu)
        return R*T*phi, c, H

    def props(self, Y):
        nu = Y[:,:3]
        S = Y[:,3]
        pL,cL,HL = self.branch(nu,'liquid')
        pV,cV,HV = self.branch(nu,'vapor')
        pm = (1-S)*pL + S*pV
        C = (1-S)[:,None]*cL + S[:,None]*cV
        z = C/C.sum(axis=1)[:,None]
        F = (pV-pL)/1.0e6
        Hmix = (1-S)[:,None,None]*HL + S[:,None,None]*HV
        return dict(pL=pL,pV=pV,cL=cL,cV=cV,HL=HL,HV=HV,
                    pm=pm,C=C,z=z,F=F,Hmix=Hmix)


def _cubic_real_roots(a,b,c):
    p=b-a*a/3.0
    q=2*a*a*a/27.0-a*b/3.0+c
    disc=(q/2.0)**2+(p/3.0)**3
    if disc>1e-14:
        sd=math.sqrt(disc)
        return np.array([np.cbrt(-q/2+sd)+np.cbrt(-q/2-sd)-a/3.0])
    if abs(p)<1e-16:
        return np.array([-a/3.0])
    rr=2*math.sqrt(max(0.0,-p/3.0))
    den=math.sqrt(max(1e-300,-(p/3.0)**3))
    arg=max(-1.0,min(1.0,(-q/2.0)/den))
    th=math.acos(arg)
    out=np.array([rr*math.cos((th+2*k*math.pi)/3.0)-a/3.0 for k in range(3)])
    out.sort()
    return out

class PengRobinsonFlash:
    """PR EOS + local isothermal flash used inside the conventional global solve.

    The implementation uses analytic cubic roots, Wilson/previous-K initialization,
    Rachford--Rice and successive fugacity substitution. A homogeneous Gibbs check
    selects the stable single-phase root. This implementation was cross-checked
    against the slower nonlinear least-squares PR flash on an independent random set.
    """
    def __init__(self):
        self.Tc=np.array([190.564,304.1282,425.125])
        self.Pc=np.array([4.5992e6,7.3773e6,3.796e6])
        self.omega=np.array([0.01142,0.22394,0.2002])
        self.kappa=0.37464+1.54226*self.omega-0.26992*self.omega**2
        self.ai0=0.45724*R**2*self.Tc**2/self.Pc
        self.bi=0.07780*R*self.Tc/self.Pc
        self.ai=self.ai0*(1+self.kappa*(1-np.sqrt(T/self.Tc)))**2
        self.aij=np.sqrt(self.ai[:,None]*self.ai[None,:])
        self.sq2=np.sqrt(2.0)
        self.calls=0; self.eos_evals=0; self.iterations=0

    def roots_lnphi(self,P,x):
        self.eos_evals += 1
        x=np.asarray(x,float); x=x/x.sum()
        am=float(x@self.aij@x); bm=float(x@self.bi)
        A=am*P/(R**2*T**2); B=bm*P/(R*T)
        rr=_cubic_real_roots(B-1.0,A-3*B*B-2*B,-(A*B-B*B-B**3))
        rr=rr[rr>B+1e-10]
        if len(rr)==0: raise RuntimeError('No valid PR root')
        sa=self.aij@x; out=[]
        for Z in rr:
            term=math.log((Z+(1+self.sq2)*B)/(Z+(1-self.sq2)*B))
            lp=self.bi/bm*(Z-1)-math.log(Z-B)-A/(2*self.sq2*B)*(2*sa/am-self.bi/bm)*term
            out.append((float(Z),lp))
        return out

    def lnphi(self,P,x,phase):
        rr=self.roots_lnphi(P,x)
        return rr[-1] if phase=='vapor' else rr[0]

    @staticmethod
    def rr_beta(z,K):
        d=K-1.0
        f0=float(z@d); f1=float(z@(d/K))
        if f0<=0: return 0.0
        if f1>=0: return 1.0
        lo=0.0; hi=1.0
        for _ in range(30):
            m=0.5*(lo+hi)
            f=float(z@(d/(1+m*d)))
            if f>0: lo=m
            else: hi=m
        return 0.5*(lo+hi)

    def _ssi(self,P,z,lnK):
        lnK=np.asarray(lnK,float).copy()
        for it in range(35):
            K=np.exp(np.clip(lnK,-20,20))
            beta=self.rr_beta(z,K)
            den=1+beta*(K-1)
            x=z/den; y=K*x; x/=x.sum(); y/=y.sum()
            ZL,lpl=self.lnphi(P,x,'liquid')
            ZV,lpv=self.lnphi(P,y,'vapor')
            target=lpl-lpv
            err=float(np.max(np.abs(target-lnK)))
            if err<3e-9:
                lnK=target; break
            damp=0.6 if err>0.15 else 1.0
            lnK=(1-damp)*lnK+damp*target
        self.iterations += it+1
        K=np.exp(np.clip(lnK,-20,20))
        beta=self.rr_beta(z,K)
        den=1+beta*(K-1)
        x=z/den; y=K*x; x/=x.sum(); y/=y.sum()
        ZL,lpl=self.lnphi(P,x,'liquid')
        ZV,lpv=self.lnphi(P,y,'vapor')
        return beta,x,y,ZL,ZV,lpl,lpv,lnK

    def flash(self,P,z,lnK0=None):
        self.calls += 1
        z=np.asarray(z,float); z=np.clip(z,1e-14,None); z/=z.sum()
        wil=np.log((self.Pc/P)*np.exp(5.373*(1+self.omega)*(1-self.Tc/T)))
        beta,x,y,ZL,ZV,lpl,lpv,lnK=self._ssi(P,z,wil if lnK0 is None else lnK0)
        roots=self.roots_lnphi(P,z)
        singles=[(float(np.sum(z*(np.log(z)+lp))),Z) for Z,lp in roots]
        gs,Zs=min(singles,key=lambda v:v[0])
        if 1e-7<beta<1-1e-7:
            g2=(1-beta)*np.sum(x*(np.log(np.clip(x,1e-300,1))+lpl))+beta*np.sum(y*(np.log(np.clip(y,1e-300,1))+lpv))
            if g2<gs-1e-8:
                rhoL=P/(ZL*R*T); rhoV=P/(ZV*R*T)
                S=(beta/rhoV)/(beta/rhoV+(1-beta)/rhoL)
                cL=rhoL*x; cV=rhoV*y
                return dict(phase='two',S=float(S),beta=float(beta),x=x,y=y,cL=cL,cV=cV,
                            C=(1-S)*cL+S*cV,lnK=lnK)
        phase='vapor' if Zs>0.5 else 'liquid'
        S=1.0 if phase=='vapor' else 0.0
        Z,lp=self.lnphi(P,z,phase); rho=P/(Z*R*T)
        cL=rho*z if phase=='liquid' else np.zeros(3)
        cV=rho*z if phase=='vapor' else np.zeros(3)
        return dict(phase=phase,S=S,beta=S,x=z if phase=='liquid' else np.full(3,np.nan),
                    y=z if phase=='vapor' else np.full(3,np.nan),cL=cL,cV=cV,
                    C=cL if phase=='liquid' else cV,lnK=lnK)

    def chemical_potential_coordinate(self,P,z,flash_result=None):
        f=self.flash(P,z) if flash_result is None else flash_result
        if f['phase']=='two':
            x=f['x']; Z,lp=self.lnphi(P,x,'liquid')
            return np.log(x*P*np.exp(lp)/PREF)
        z=np.asarray(z,float); Z,lp=self.lnphi(P,z,f['phase'])
        return np.log(z*P*np.exp(lp)/PREF)
