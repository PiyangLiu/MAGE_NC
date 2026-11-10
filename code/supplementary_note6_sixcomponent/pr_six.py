"""Six-hydrocarbon Peng--Robinson EOS, classical zero-kij mixture.

CH4 / C2H6 / C3H8 / nC4 / nC5 / nC6, at T=300 K.
Species Tc/Pc/omega are explicit; no external EOS package is required.
"""
from __future__ import annotations
import numpy as np

R=8.31446261815324
T=300.0
PREF=1e5
N=6
NAMES=('CH4','C2H6','C3H8','nC4','nC5','nC6')
TC=np.array([190.564,305.322,369.83,425.125,469.7,507.6])
PC=np.array([4.5992e6,4.872e6,4.248e6,3.796e6,3.370e6,3.025e6])
OMEGA=np.array([.01142,.0995,.1523,.2002,.2515,.3013])
KAPPA=.37464+1.54226*OMEGA-.26992*OMEGA**2
AI=.45724*(R*TC)**2/PC*(1+KAPPA*(1-np.sqrt(T/TC)))**2
BI=.07780*R*TC/PC
AIJ=np.sqrt(np.outer(AI,AI))
SQ2=np.sqrt(2.)


def state(P,x,phase):
    x=np.asarray(x,float);x=x/x.sum()
    am=float(x@AIJ@x);bm=float(x@BI)
    A=am*P/(R*T)**2;B=bm*P/(R*T)
    roots=np.roots([1.,B-1.,A-3*B**2-2*B,-(A*B-B**2-B**3)])
    good=sorted(float(z.real) for z in roots if abs(z.imag)<1e-8 and z.real>B+1e-10)
    if not good:raise ValueError('No admissible PR root')
    Z=good[0] if phase=='liquid' else good[-1]
    log_term=np.log((Z+(1+SQ2)*B)/(Z+(1-SQ2)*B))
    lnphi=BI/bm*(Z-1)-np.log(Z-B)-A/(2*SQ2*B)*(2*(AIJ@x)/am-BI/bm)*log_term
    eta=np.log(x*P/PREF)+lnphi
    return eta,P/(R*T),x*P/(Z*R*T),Z


def rr_beta(z,K):
    d=K-1.;f0=float(z@d);f1=float(z@(d/K))
    if f0<=0:return 0.
    if f1>=0:return 1.
    lo,hi=0.,1.
    for _ in range(55):
        mid=(lo+hi)/2
        if z@(d/(1+mid*d))>0:lo=mid
        else:hi=mid
    return (lo+hi)/2


def ptz_flash(P,z):
    z=np.asarray(z,float);z/=z.sum()
    lnK=np.log(PC/P)+5.373*(1+OMEGA)*(1-TC/T)
    for k in range(150):
        K=np.exp(np.clip(lnK,-25,25));beta=rr_beta(z,K)
        x=z/(1+beta*(K-1));x/=x.sum()
        y=K*x;y/=y.sum()
        eta_l,*_=state(P,x,'liquid');eta_v,*_=state(P,y,'vapor')
        # ln K = lnphi_L-lnphi_V; eta_l-eta_v = ln(x/y)+lnphiL-lnphiV
        lnK_target=lnK+(eta_l-eta_v)
        err=np.max(np.abs(lnK_target-lnK))
        if err<2e-11:break
        lnK=.6*lnK+.4*lnK_target
    return beta,x,y,err,k+1

