#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from solvers import MAGEGlobalSolver,PRGlobalSolver

def compare(mage,pr,me,pe):
    d=mage.merge(pr,on=['accepted_step','cell'],suffixes=('_mage','_pr'))
    d['phase_match']=d.phase_mage==d.phase_pr
    d['P_relative_error']=np.abs(d.P_MPa_mage-d.P_MPa_pr)/np.maximum(np.abs(d.P_MPa_pr),1e-30)
    for c in ['CH4','CO2','nC4']:
        d[f'z_{c}_abs_error']=np.abs(d[f'z_{c}_mage']-d[f'z_{c}_pr'])
    d['S_abs_error']=np.abs(d.S_mage-d.S_pr)
    evcols=['accepted_step','cell','from_phase','to_phase']
    em=me[evcols].sort_values(evcols).reset_index(drop=True)
    ep=pe[evcols].sort_values(evcols).reset_index(drop=True)
    events_identical=em.equals(ep)
    zerr=d[[f'z_{c}_abs_error' for c in ['CH4','CO2','nC4']]].to_numpy()
    summary={
      'space_time_states':int(len(d)),
      'phase_state_match_fraction':float(d.phase_match.mean()),
      'phase_state_mismatches':int((~d.phase_match).sum()),
      'phase_events_MAGE':int(len(me)),'phase_events_PR':int(len(pe)),
      'phase_event_step_cell_transition_identity':bool(events_identical),
      'pressure_relative_error_mean':float(d.P_relative_error.mean()),
      'pressure_relative_error_max':float(d.P_relative_error.max()),
      'composition_absolute_error_mean':float(zerr.mean()),
      'composition_absolute_error_max':float(zerr.max()),
      'gas_saturation_absolute_error_mean':float(d.S_abs_error.mean()),
      'gas_saturation_absolute_error_max':float(d.S_abs_error.max()),
    }
    return d,summary

def main():
    results=ROOT/'results';results.mkdir(exist_ok=True)
    grid=pd.read_csv(ROOT/'time_grid.csv')
    mage_solver=MAGEGlobalSolver(ROOT/'benchmark_config.json',ROOT/'model')
    pr_solver=PRGlobalSolver(ROOT/'benchmark_config.json')
    print('Running MAGE global solver...',flush=True)
    M,ME,MM,MS=mage_solver.run(grid,results)
    print('Running fully implicit PR global solver...',flush=True)
    P,PE,PM,PS=pr_solver.run(grid,results)
    C,CS=compare(M,P,ME,PE)
    C.to_csv(results/'combined_global_states.csv',index=False)
    # combined event comparison
    key=['accepted_step','cell','from_phase','to_phase']
    ev=ME.merge(PE,on=key,how='outer',suffixes=('_mage','_pr'),indicator=True)
    ev.to_csv(results/'combined_phase_events.csv',index=False)
    summary={'benchmark':json.load(open(ROOT/'benchmark_config.json')),'MAGE_solver':MS,'PR_solver':PS,'comparison':CS}
    json.dump(summary,open(results/'summary.json','w'),indent=2)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
