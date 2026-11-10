"""Check exact C++ MAGE tangent implementation and analytic PR Jacobians."""
from __future__ import annotations
import ctypes,json
from pathlib import Path
import numpy as np
from benchmark import CompiledMAGE,DP,dptr
from fit import PHI_SCALE,predict
from pr_six import R,T,N

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results';RESULTS.mkdir(exist_ok=True)

def main():
    with np.load(HERE/'data'/'confirmatory.npz') as d:t={k:d[k] for k in d.files}
    with np.load(HERE/'model'/'mage_six.npz') as d:m={k:d[k] for k in d.files}
    eta=np.ascontiguousarray(t['eta']);actual=CompiledMAGE()(eta)
    per_phase={}
    for phase,idx in [('liquid',0),('vapor',1)]:
        center=m[f'{phase}_center'];scale=m[f'{phase}_scale']
        X=m[f'{phase}_X'];F=m[f'{phase}_F'];G=m[f'{phase}_G']
        pred,grad=predict((eta-center)/scale,X,F,G)
        pressure=pred*PHI_SCALE*R*T
        density=grad*PHI_SCALE/scale
        pe=np.abs(pressure-actual[:,idx])/pressure
        qe=(np.abs(density-actual[:,2+idx*N:2+(idx+1)*N]) /
            np.maximum(np.abs(density),1.))
        per_phase[phase]={'max_relative_pressure_discrepancy':float(pe.max()),
                          'max_relative_component_density_discrepancy':float(qe.max())}
        if pe.max()>1e-11 or qe.max()>1e-10:
            raise AssertionError(f'C++ and Python MAGE differ on {phase}')
    if np.count_nonzero(actual[:,16]!=(actual[:,1]>actual[:,0])):
        raise AssertionError('C++ MAGE phase label inconsistent with branch pressures')
    lib=ctypes.CDLL(str((HERE/'libpr_six_analytic.so').resolve()))
    f=lib.pr_six_audit_jac
    f.argtypes=[DP,ctypes.c_int,DP,DP];f.restype=ctypes.c_int
    largest=0.;largest_relative=0.;tested=0
    for ind in np.linspace(0,len(eta)-1,24,dtype=int):
        for phase,p in [('liquid',0),('vapor',1)]:
            x=t[f'x_{phase}'][ind]
            v=np.ascontiguousarray(np.r_[np.log(t[f'P_{phase}'][ind]/1e6),
                                          np.log(x[:-1]/x[-1])])
            analytic=np.empty((N,N));finite=np.empty((N,N))
            if f(dptr(v),p,dptr(analytic),dptr(finite)):
                raise RuntimeError('PR analytic derivative audit call failed')
            err=np.abs(analytic-finite);tested+=1
            largest=max(largest,float(err.max()))
            largest_relative=max(largest_relative,float(np.linalg.norm(err)/np.linalg.norm(finite)))
    if largest_relative>1e-6:raise AssertionError('Analytic PR Jacobian failed finite differences')
    report={'MAGE_CPP_vs_Python':per_phase,
            'PR_analytic_Jacobian_test_states':tested,
            'PR_Jacobian_largest_element_absolute_error':largest,
            'PR_Jacobian_max_relative_Frobenius_discrepancy':largest_relative}
    (RESULTS/'compiled_implementation_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
