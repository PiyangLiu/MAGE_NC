"""Independent noncritical six-component GRAND-canonical PR--MAGE benchmark.

Both compiled C++ methods receive exactly the same six chemical potentials and
return both PR phase pressures, twelve component densities, two Z values and
the stable-phase label. No TPz-vs-dual-specification timing mismatch exists.
"""
from __future__ import annotations
import ctypes,json,statistics,time
from pathlib import Path
import numpy as np
import pandas as pd
from pr_six import N,NAMES,R,T

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results';RESULTS.mkdir(exist_ok=True)
DP=ctypes.POINTER(ctypes.c_double);IP=ctypes.POINTER(ctypes.c_int)
def dptr(a):return a.ctypes.data_as(DP)
def iptr(a):return a.ctypes.data_as(IP)

class CompiledPR:
    def __init__(self,analytic=False):
        filename='libpr_six_analytic.so' if analytic else 'libpr_six.so'
        f=ctypes.CDLL(str((HERE/filename).resolve())).pr_six_batch
        f.argtypes=[ctypes.c_int,DP,DP,IP,IP,ctypes.c_int];f.restype=ctypes.c_int
        self.f=f

    def __call__(self,eta,warm):
        n=len(eta);out=np.empty((n,18));iters=np.empty((n,2),dtype=np.int32)
        eos=np.empty((n,2),dtype=np.int32)
        failure=self.f(n,dptr(eta),dptr(out),iptr(iters),iptr(eos),int(warm))
        if failure:raise RuntimeError(f'Compiled PR failed at zero-based query {failure-1}')
        return out,iters,eos

class CompiledMAGE:
    def __init__(self):
        with np.load(HERE/'model'/'mage_six.npz') as d:
            def stack(prop):return np.ascontiguousarray(np.stack(
                [d[f'{phase}_{prop}'] for phase in ('liquid','vapor')]))
            self.center=stack('center');self.scale=stack('scale')
            self.X=stack('X');self.F=stack('F');self.G=stack('G')
        self.intercepts=np.ascontiguousarray(self.F-np.einsum('pmi,pmi->pm',self.G,self.X))
        cfg=json.loads((HERE/'model'/'model_summary.json').read_text())
        self.tau=cfg['tau_nondim'];self.pscale=cfg['phi_scale_mol_per_m3']
        self.M=self.F.shape[1]
        f=ctypes.CDLL(str((HERE/'libmage_six.so').resolve())).mage_six_batch
        f.argtypes=[ctypes.c_int,ctypes.c_int,DP,DP,DP,DP,DP,DP,
                    ctypes.c_double,ctypes.c_double,ctypes.c_double,DP]
        f.restype=ctypes.c_int;self.f=f

    def __call__(self,eta):
        n=len(eta);out=np.empty((n,18))
        rc=self.f(n,self.M,dptr(eta),dptr(self.center),dptr(self.scale),
                  dptr(self.X),dptr(self.intercepts),dptr(self.G),
                  self.tau,self.pscale,R*T,dptr(out))
        if rc:raise RuntimeError('Compiled MAGE failure')
        return out


def error(actual,ref):
    pe=np.abs(actual[:,:2]-ref[:,:2])/ref[:,:2]
    qe=np.abs(actual[:,2:14]-ref[:,2:14])/np.maximum(np.abs(ref[:,2:14]),1e-9)
    ze=np.abs(actual[:,14:16]-ref[:,14:16])
    return {'pressure_relative_mean':float(pe.mean()),'pressure_relative_max':float(pe.max()),
            'component_density_relative_mean':float(qe.mean()),
            'component_density_relative_max':float(qe.max()),
            'compressibility_Z_absolute_max':float(ze.max())}


def main():
    # This confirmatory stream is independent of candidate and validation data.
    with np.load(HERE/'data'/'confirmatory.npz') as d:
        test={k:d[k] for k in d.files}
    eta=np.ascontiguousarray(test['eta'],dtype=np.float64);n=len(eta)
    ref=np.zeros((n,18),float)
    for ph,p in [('liquid',0),('vapor',1)]:
        ref[:,p]=test[f'P_{ph}']
        ref[:,2+p*N:2+(p+1)*N]=test[f'q_{ph}']
        ref[:,14+p]=test[f'Z_{ph}']
    ref[:,16]=test['phase']
    pr=CompiledPR();pr_ad=CompiledPR(analytic=True);mage=CompiledMAGE()
    pw,iters,eos=pr(eta,True);pc,ic,ec=pr(eta,False)
    pa,ai,ae=pr_ad(eta,True);pac,aic,aec=pr_ad(eta,False)
    mm=mage(eta)
    count={'PR_warm':int(np.count_nonzero(pw[:,16]==ref[:,16])),
           'PR_cold':int(np.count_nonzero(pc[:,16]==ref[:,16])),
           'PR_analytic_warm':int(np.count_nonzero(pa[:,16]==ref[:,16])),
           'PR_analytic_cold':int(np.count_nonzero(pac[:,16]==ref[:,16])),
           'MAGE':int(np.count_nonzero(mm[:,16]==ref[:,16]))}
    if any(v!=n for v in count.values()):
        raise AssertionError(f'100% phase classification requirement failed: {count}')
    perr=error(pw,ref);cerr=error(pc,ref)
    aerr=error(pa,ref);acerr=error(pac,ref)
    merr=error(mm,ref)
    if max(perr['pressure_relative_max'],cerr['pressure_relative_max'],
           aerr['pressure_relative_max'],acerr['pressure_relative_max'])>1e-8:
        raise AssertionError('Compiled PR failed independent reference EOS audit')
    if max(perr['component_density_relative_max'],aerr['component_density_relative_max'])>1e-8:
        raise AssertionError('Compiled PR composition/density parity failed')
    if merr['pressure_relative_max']>0.002 or merr['component_density_relative_max']>0.06:
        raise AssertionError(f'MAGE finite-support accuracy not adequate: {merr}')
    for _ in range(7):pr(eta,True);pr(eta,False);pr_ad(eta,True);pr_ad(eta,False);mage(eta)
    methods={'PR_analytic_warm':lambda:pr_ad(eta,True),
             'PR_analytic_cold':lambda:pr_ad(eta,False),
             'PR_fd_warm':lambda:pr(eta,True),
             'PR_fd_cold':lambda:pr(eta,False),
             'MAGE_direct':lambda:mage(eta)}
    times={k:[] for k in methods}
    batches=30
    for rep in range(5):
        names=list(methods) if rep%2==0 else list(methods)[::-1]
        for name in names:
            start=time.perf_counter()
            for _ in range(batches):methods[name]()
            times[name].append((time.perf_counter()-start)/(batches*n)*1e6)
        print(f'run {rep+1}: analytic PR {times["PR_analytic_warm"][-1]:.2f} us, '
              f'FD PR {times["PR_fd_warm"][-1]:.2f} us, '
              f'MAGE {times["MAGE_direct"][-1]:.2f} us',flush=True)
    median={k:float(statistics.median(v)) for k,v in times.items()}
    fastest=min(median[k] for k in median if k.startswith('PR_'))
    g=json.loads((HERE/'data'/'reference_summary.json').read_text())
    f=json.loads((HERE/'model'/'model_summary.json').read_text())
    training_generation=(g['reference_generation_wall_s']
        -g['independent_splits']['heldout']['generation_wall_s']
        -g['independent_splits']['confirmatory']['generation_wall_s'])
    online_saving=(fastest-median['MAGE_direct'])*1e-6
    report={'benchmark':'six-component 300 K noncritical grand-canonical PR EOS inversion',
        'components':list(NAMES),'N':N,'n_heldout':n,'phase_agreement_counts':count,
        'reference_stable_liquid_count':int((ref[:,16]==0).sum()),
        'reference_stable_vapor_count':int((ref[:,16]==1).sum()),
        'min_absolute_phase_pressure_margin_MPa':float(np.min(np.abs(ref[:,1]-ref[:,0]))/1e6),
        'liquid_Z_range':[float(ref[:,14].min()),float(ref[:,14].max())],
        'vapor_Z_range':[float(ref[:,15].min()),float(ref[:,15].max())],
        'PR_warm_vs_independent_Python_reference':perr,
        'PR_cold_vs_independent_Python_reference':cerr,
        'PR_analytic_warm_vs_Python_reference':aerr,
        'PR_analytic_cold_vs_Python_reference':acerr,
        'MAGE_vs_independent_Python_PR_reference':merr,
        'PR_max_chemical_potential_residual':float(np.max(pa[:,17])),
        'PR_analytic_total_Newton_steps_per_query':{'median':float(np.median(ai.sum(axis=1))),
            'min':int(ai.sum(axis=1).min()),'max':int(ai.sum(axis=1).max())},
        'PR_analytic_total_EOS_evaluations_per_query':{'median':float(np.median(ae.sum(axis=1))),
            'min':int(ae.sum(axis=1).min()),'max':int(ae.sum(axis=1).max())},
        'PR_fd_total_EOS_evaluations_per_query':{'median':float(np.median(eos.sum(axis=1))),
            'min':int(eos.sum(axis=1).min()),'max':int(eos.sum(axis=1).max())},
        'PR_analytic_cold_EOS_evaluations_median':float(np.median(aec.sum(axis=1))),
        'MAGE_supports_per_phase':mage.M,
        'five_repeats_us_per_query':times,'median_us_per_query':median,
        'speedup_MAGE_vs_fastest_PR':fastest/median['MAGE_direct'],
        'offline_PR_reference_generation_s':g['reference_generation_wall_s'],
        'offline_training_reference_generation_s':training_generation,
        'offline_support_selection_s':f['offline_selection_seconds'],
        'idealized_offline_amortization_query_count':(
            training_generation+f['offline_selection_seconds'])/online_saving,
        'idealized_full_experiment_amortization_query_count':(
            g['reference_generation_wall_s']+f['offline_selection_seconds'])/online_saving,
        'limitations':['not ordinary PTz flash; difficult fugacity-specified two-branch inverse',
            'PR used analytic cubic roots and damped Newton, including analytical PR fugacity Jacobians and continuation',
            'MAGE trained on a restricted two-parameter manifold in six-dimensional chemical space',
            'headline labels evaluated on 1,000 confirmatory states; a separate 800-state development-held-out stream is retained; no near-critical state selected',
            'hardware/runtime and offline database generation affect amortization']}
    (RESULTS/'verified_benchmark.json').write_text(json.dumps(report,indent=2))
    cols={'eta_'+name:eta[:,i] for i,name in enumerate(NAMES)}
    cols.update({'reference_phase':ref[:,16].astype(int),'PR_phase':pa[:,16].astype(int),
                 'MAGE_phase':mm[:,16].astype(int),
                 'PR_newton_evals':ai.sum(axis=1),'PR_EOS_evals':ae.sum(axis=1)})
    for j,ph in enumerate(('liquid','vapor')):
        cols[f'{ph}_P_PR_MPa']=pa[:,j]/1e6
        cols[f'{ph}_P_MAGE_MPa']=mm[:,j]/1e6
        cols[f'{ph}_P_reference_MPa']=ref[:,j]/1e6
        cols[f'{ph}_Z_reference']=ref[:,14+j]
    pd.DataFrame(cols).to_csv(RESULTS/'heldout_all_states.csv',index=False)
    print(json.dumps({k:report[k] for k in ('phase_agreement_counts',
        'MAGE_vs_independent_Python_PR_reference','PR_analytic_total_EOS_evaluations_per_query',
        'median_us_per_query','speedup_MAGE_vs_fastest_PR',
        'idealized_offline_amortization_query_count')},indent=2))

if __name__=='__main__':main()
