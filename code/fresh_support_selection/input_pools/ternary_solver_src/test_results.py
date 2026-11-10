#!/usr/bin/env python3
from pathlib import Path
import json, pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results'
s=json.load(open(R/'summary.json')); d=pd.read_csv(R/'combined_global_states.csv'); me=pd.read_csv(R/'mage_phase_events.csv'); pe=pd.read_csv(R/'pr_phase_events.csv'); mm=pd.read_csv(R/'mage_mass_balance.csv'); pm=pd.read_csv(R/'pr_mass_balance.csv')
checks={}
checks['both_solvers_complete_92_steps']=s['MAGE_solver']['accepted_steps']==92 and s['PR_solver']['accepted_steps']==92
checks['MAGE_has_zero_online_local_flash']=s['MAGE_solver']['online_local_flash_calls']==0
checks['MAGE_has_zero_online_TPD']=s['MAGE_solver']['online_TPD_calls']==0
checks['MAGE_has_zero_explicit_phase_switches']=s['MAGE_solver']['online_explicit_phase_switches']==0
checks['PR_executes_nested_local_flash']=s['PR_solver']['online_executed_local_flash_calls_with_exact_cache']>0
checks['phase_state_match_is_100_percent']=bool((d.phase_mage==d.phase_pr).all())
cols=['accepted_step','cell','from_phase','to_phase']; checks['all_24_phase_events_identical']=bool(len(me)==24 and len(pe)==24 and me[cols].sort_values(cols).reset_index(drop=True).equals(pe[cols].sort_values(cols).reset_index(drop=True)))
probe=me[me.cell==1].sort_values('accepted_step'); cycle=[probe.iloc[0].from_phase]+probe.to_phase.tolist(); checks['probe_complete_phase_cycle']=cycle==['liquid','two','liquid','two','vapor']
checks['each_cell_has_four_phase_events']=bool(all((me.cell==j).sum()==4 for j in range(1,7)))
checks['MAGE_residual_within_tolerance']=s['MAGE_solver']['max_scaled_residual']<2e-6
checks['PR_residual_within_tolerance']=s['PR_solver']['max_scaled_residual']<2e-5
checks['mass_balance_within_tolerance']=bool(mm.relative_balance_error.max()<1e-9 and pm.relative_balance_error.max()<1e-6)
checks['pressure_monotone_both_solvers']=bool(all(np.all(np.diff(g.P_MPa_mage.values)<0) and np.all(np.diff(g.P_MPa_pr.values)<0) for _,g in d.sort_values(['accepted_step','cell']).groupby('accepted_step'))) 
checks['mean_solution_errors_within_declared_limits']=s['comparison']['pressure_relative_error_mean']<1e-3 and s['comparison']['composition_absolute_error_mean']<2e-3 and s['comparison']['gas_saturation_absolute_error_mean']<3e-3
checks={k:bool(v) for k,v in checks.items()}
ok=all(checks.values())
with open(R/'TEST_REPORT.txt','w') as f:
    for k,v in checks.items(): f.write(f"{'PASS' if v else 'FAIL'}  {k}\n")
    f.write(f"\n{sum(checks.values())}/{len(checks)} checks passed\n")
print(json.dumps(checks,indent=2)); print(f"{sum(checks.values())}/{len(checks)} checks passed")
if not ok: raise SystemExit(1)
