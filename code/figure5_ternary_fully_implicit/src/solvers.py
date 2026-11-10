from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from thermo import R,T,PREF,MAGEModel,PengRobinsonFlash


def phase_label(S, tol=1e-4):
    if S<tol: return 'liquid'
    if S>1-tol: return 'vapor'
    return 'two'

def proj01(y): return np.clip(y,0.0,1.0)
def dproj01(y):
    y=np.asarray(y); d=np.zeros_like(y,float); d[(y>0)&(y<1)]=1.0
    d[np.isclose(y,0,atol=1e-12)|np.isclose(y,1,atol=1e-12)]=0.5
    return d

class PhysicalCase:
    def __init__(self, config_path: Path):
        c=json.load(open(config_path)); self.config=c
        self.N=int(c['cells']); self.length=float(c['length_m']); self.dx=self.length/self.N
        self.por=float(c['porosity']); self.k=float(c['permeability_m2'])
        self.muL=float(c['liquid_viscosity_Pa_s']); self.muV=float(c['vapor_viscosity_Pa_s'])
        self.Pin=float(c['inlet_pressure_MPa'])*1e6; self.Pout=float(c['outlet_pressure_MPa'])*1e6
        self.initial_z=np.array(c['initial_composition'],float)
        self.stages={x['name']:np.array(x['composition'],float) for x in c['injection_stages']}
        self.Cscale=1500.0; self.Pclose=1e5; self.Pscale=1e6

class MAGEGlobalSolver(PhysicalCase):
    def __init__(self, config_path:Path, model_dir:Path):
        super().__init__(config_path); self.mage=MAGEModel(model_dir); self.pr_map=PengRobinsonFlash()
        self.global_residual_evals=0

    def map_state(self,P,z):
        ref=self.pr_map.flash(P,z)
        nu0=self.pr_map.chemical_potential_coordinate(P,z,ref)
        x0=np.r_[nu0,ref['S']]
        lo=np.array([-.45,-.2,-.45,0.0]); hi=np.array([3.35,3.0,1.55,1.0])
        def fun(v):
            Y=np.r_[v,P][None,:]; a=self.mage.props(Y); y=v[3]+a['F'][0]
            return np.r_[(a['z'][0,:2]-z[:2])/.01,(a['pm'][0]-P)/1e5,v[3]-proj01(y)]
        sol=least_squares(fun,x0,bounds=(lo,hi),x_scale=[1,1,1,.2],max_nfev=300,
                          xtol=1e-11,ftol=1e-11,gtol=1e-11)
        return np.r_[sol.x,P],float(np.max(np.abs(sol.fun))),ref

    def _face_flux(self,a,j,Pdn,dist):
        S=a['Y'][j,3]; dP=a['Y'][j,4]-Pdn
        uL=self.k*(1-S)**2/self.muL*dP/dist; uV=self.k*S**2/self.muV*dP/dist
        F=a['cL'][j]*uL+a['cV'][j]*uV
        dF=np.zeros((3,5)); dF[:,:3]=a['HL'][j]*uL+a['HV'][j]*uV
        dF[:,3]=a['cL'][j]*(self.k*(-2*(1-S))/self.muL*dP/dist)+a['cV'][j]*(self.k*(2*S)/self.muV*dP/dist)
        dF[:,4]=a['cL'][j]*(self.k*(1-S)**2/self.muL/dist)+a['cV'][j]*(self.k*S**2/self.muV/dist)
        return F,dF,-dF[:,4]

    def _residual_jac(self,x,Cold,bstate,bprops,dt,need_jac=True):
        self.global_residual_evals += 1
        Y=x.reshape(self.N,5); a=self.mage.props(Y); a['Y']=Y
        flux=np.zeros((self.N+1,3)); df_up=[None]*(self.N+1); df_dn=[None]*(self.N+1)
        Sb=bstate[3]; dP=self.Pin-Y[0,4]; dist=self.dx/2
        uL=self.k*(1-Sb)**2/self.muL*dP/dist; uV=self.k*Sb**2/self.muV*dP/dist
        flux[0]=bprops['cL'][0]*uL+bprops['cV'][0]*uV
        df_dn[0]=-(bprops['cL'][0]*self.k*(1-Sb)**2/self.muL/dist+bprops['cV'][0]*self.k*Sb**2/self.muV/dist)
        for f in range(1,self.N): flux[f],df_up[f],df_dn[f]=self._face_flux(a,f-1,Y[f,4],self.dx)
        flux[self.N],df_up[self.N],_=self._face_flux(a,self.N-1,self.Pout,self.dx/2)
        rr=np.zeros((self.N,5)); J=np.zeros((5*self.N,5*self.N)) if need_jac else None
        for j in range(self.N):
            rr[j,:3]=(self.por*a['C'][j]-Cold[j]+dt/self.dx*(flux[j+1]-flux[j]))/self.Cscale
            rr[j,3]=(Y[j,4]-a['pm'][j])/self.Pclose
            y=Y[j,3]+a['F'][j]; rr[j,4]=Y[j,3]-proj01(y); dpj=float(dproj01(y))
            if not need_jac: continue
            row=5*j; col=5*j
            J[row:row+3,col:col+3]+=self.por*a['Hmix'][j]/self.Cscale
            J[row:row+3,col+3]+=self.por*(a['cV'][j]-a['cL'][j])/self.Cscale
            J[row:row+3,col:col+5]+=dt/self.dx*df_up[j+1]/self.Cscale
            if j<self.N-1: J[row:row+3,5*(j+1)+4]+=dt/self.dx*df_dn[j+1]/self.Cscale
            if j==0: J[row:row+3,col+4]+=-dt/self.dx*df_dn[0]/self.Cscale
            else:
                J[row:row+3,5*(j-1):5*(j-1)+5]+=-dt/self.dx*df_up[j]/self.Cscale
                J[row:row+3,col+4]+=-dt/self.dx*df_dn[j]/self.Cscale
            J[row+3,col:col+3]=-R*T*a['C'][j]/self.Pclose
            J[row+3,col+3]=-(a['pV'][j]-a['pL'][j])/self.Pclose; J[row+3,col+4]=1/self.Pclose
            dFnu=R*T*(a['cV'][j]-a['cL'][j])/self.Pscale
            J[row+4,col:col+3]=-dpj*dFnu; J[row+4,col+3]=1-dpj
        return rr.ravel(),J,flux,a

    def run(self,time_grid:pd.DataFrame,outdir:Path):
        outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
        bmaps={}; mapping=[]
        for name,z in self.stages.items():
            state,err,ref=self.map_state(self.Pin,z); bmaps[name]=state
            mapping.append(dict(kind='boundary',label=name,P_MPa=self.Pin/1e6,z_CH4=z[0],z_CO2=z[1],z_nC4=z[2],map_residual=err,PR_phase=ref['phase'],PR_S=ref['S']))
        xc=(np.arange(self.N)+.5)*self.dx; P0=self.Pin+(self.Pout-self.Pin)*xc/self.length
        Y=[]
        for j,P in enumerate(P0):
            state,err,ref=self.map_state(P,self.initial_z); Y.append(state)
            mapping.append(dict(kind='initial',label=f'cell {j+1}',P_MPa=P/1e6,z_CH4=self.initial_z[0],z_CO2=self.initial_z[1],z_nC4=self.initial_z[2],map_residual=err,PR_phase=ref['phase'],PR_S=ref['S']))
        Y=np.asarray(Y)
        offline_pr_calls=self.pr_map.calls; offline_pr_eos=self.pr_map.eos_evals
        self.mage.reset_counter(); self.global_residual_evals=0
        rows=[];events=[];mass=[];prev=[phase_label(x) for x in Y[:,3]]
        a0=self.mage.props(Y); Mprev=self.por*self.dx*a0['C'].sum(axis=0)
        t0=time.time()
        for k,s in enumerate(time_grid.itertuples(),1):
            bstate=bmaps[s.stage]; bprops=self.mage.props(bstate[None,:]); old=self.mage.props(Y); Cold=self.por*old['C']; cache={}
            def evaluate(x):
                key=x.tobytes()
                if cache.get('key')==key: return cache['r'],cache['J'],cache['flux'],cache['a']
                r,J,f,a=self._residual_jac(x,Cold,bstate,bprops,s.dt_s,True); cache.update(key=key,r=r,J=J,flux=f,a=a); return r,J,f,a
            lo=np.tile(np.array([-.45,-.2,-.45,0.0,1.25e6]),self.N); hi=np.tile(np.array([3.35,3.0,1.55,1.0,3.25e6]),self.N); xs=np.tile(np.array([1,1,1,.2,1e6]),self.N)
            sol=least_squares(lambda x:evaluate(x)[0],Y.ravel(),jac=lambda x:evaluate(x)[1],bounds=(lo,hi),x_scale=xs,max_nfev=180,xtol=1e-10,ftol=1e-10,gtol=1e-10)
            r,J,flux,a=evaluate(sol.x); rmax=float(np.max(np.abs(r)))
            if (not sol.success) or rmax>2e-6: raise RuntimeError(f'MAGE global failure at step {k}: residual={rmax}')
            Y=sol.x.reshape(self.N,5); a=self.mage.props(Y); _,_,ff,_=self._residual_jac(Y.ravel(),Cold,bstate,bprops,s.dt_s,False)
            Mnow=self.por*self.dx*a['C'].sum(axis=0); bal=Mnow-Mprev-s.dt_s*(ff[0]-ff[-1]); sc=np.maximum(abs(Mnow)+abs(Mprev)+abs(s.dt_s*(ff[0]-ff[-1])),1e-30); Mprev=Mnow
            for i,n in enumerate(['CH4','CO2','nC4']): mass.append(dict(accepted_step=k,time_s=s.time_s,component=n,balance_error=bal[i],relative_balance_error=abs(bal[i])/sc[i]))
            labs=[]
            for j in range(self.N):
                lab=phase_label(Y[j,3]); labs.append(lab)
                if lab!=prev[j]: events.append(dict(accepted_step=k,time_s=s.time_s,stage=s.stage,cell=j+1,from_phase=prev[j],to_phase=lab,S_mage=Y[j,3],P_MPa=Y[j,4]/1e6,z_CH4=a['z'][j,0],z_CO2=a['z'][j,1],z_nC4=a['z'][j,2]))
                rows.append(dict(accepted_step=k,time_s=s.time_s,stage=s.stage,dt_s=s.dt_s,cell=j+1,P_MPa=Y[j,4]/1e6,z_CH4=a['z'][j,0],z_CO2=a['z'][j,1],z_nC4=a['z'][j,2],S_mage=Y[j,3],phase_mage=lab,F_scaled=a['F'][j],residual_max=rmax,nfev=sol.nfev))
            prev=labs
        D=pd.DataFrame(rows);E=pd.DataFrame(events);MB=pd.DataFrame(mass)
        D.to_csv(outdir/'mage_global_states.csv',index=False);E.to_csv(outdir/'mage_phase_events.csv',index=False);MB.to_csv(outdir/'mage_mass_balance.csv',index=False);pd.DataFrame(mapping).to_csv(outdir/'mage_offline_mapping.csv',index=False)
        summary=dict(runtime_s=time.time()-t0,accepted_steps=len(time_grid),online_local_flash_calls=0,online_TPD_calls=0,online_explicit_phase_switches=0,online_MAGE_branch_state_evaluations=int(self.mage.branch_state_evals),global_residual_evaluations=int(self.global_residual_evals),offline_PR_flash_calls=int(offline_pr_calls),offline_PR_EOS_evaluations=int(offline_pr_eos),max_scaled_residual=float(D.residual_max.max()),max_component_relative_mass_balance_error=float(MB.relative_balance_error.max()),phase_events=len(E))
        json.dump(summary,open(outdir/'mage_solver_summary.json','w'),indent=2)
        return D,E,MB,summary

class PRGlobalSolver(PhysicalCase):
    def __init__(self,config_path:Path):
        super().__init__(config_path); self.pr=PengRobinsonFlash()
        S=lil_matrix((3*self.N,3*self.N),dtype=int)
        for j in range(self.N):
            for r in range(3*j,3*j+3):
                for c in range(3*j,3*j+3): S[r,c]=1
                if j>0:
                    for c in range(3*(j-1),3*(j-1)+3): S[r,c]=1
                if j<self.N-1: S[r,3*(j+1)+2]=1
        # Some SciPy/NumPy combinations pass ``copy=None`` to NumPy while
        # converting sparse matrices.  Keep the sparse acceleration when it
        # is available and fall back to a dense finite-difference Jacobian for
        # the small six-cell system when it is not.
        try:
            self.sparsity=S.tocsr(copy=False)
        except (TypeError, ValueError):
            self.sparsity=None

    def decode(self,X):
        V=np.asarray(X).reshape(self.N,3); L=np.c_[V[:,:2],np.zeros(self.N)]; L-=L.max(axis=1,keepdims=True); e=np.exp(L); z=e/e.sum(axis=1,keepdims=True); P=V[:,2]*1e6; return z,P
    def encode(self,z,P): return np.c_[np.log(z[:,:2]/z[:,2,None]),P/1e6].ravel()
    def flux(self,q,P,b):
        F=np.zeros((self.N+1,3)); S=b['S']; dP=self.Pin-P[0]; dist=self.dx/2
        uL=self.k*(1-S)**2/self.muL*dP/dist; uV=self.k*S**2/self.muV*dP/dist; F[0]=b['cL']*uL+b['cV']*uV
        for f in range(1,self.N):
            up=q[f-1]; dP=P[f-1]-P[f]; uL=self.k*(1-up['S'])**2/self.muL*dP/self.dx; uV=self.k*up['S']**2/self.muV*dP/self.dx; F[f]=up['cL']*uL+up['cV']*uV
        up=q[-1]; dP=P[-1]-self.Pout; dist=self.dx/2; uL=self.k*(1-up['S'])**2/self.muL*dP/dist; uV=self.k*up['S']**2/self.muV*dP/dist; F[-1]=up['cL']*uL+up['cV']*uV
        return F

    def run(self,time_grid:pd.DataFrame,outdir:Path):
        outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
        xc=(np.arange(self.N)+.5)*self.dx; P=self.Pin+(self.Pout-self.Pin)*xc/self.length; z=np.tile(self.initial_z,(self.N,1))
        q=[self.pr.flash(P[j],z[j]) for j in range(self.N)]; guess=[v['lnK'] for v in q]; X=self.encode(z,P)
        boundary={name:self.pr.flash(self.Pin,zz) for name,zz in self.stages.items()}
        preprocessing_calls=self.pr.calls; preprocessing_eos=self.pr.eos_evals
        rows=[];events=[];mass=[];prev=[v['phase'] for v in q]; Mprev=self.por*self.dx*np.sum([v['C'] for v in q],axis=0)
        residual_calls=0; flash_requests=0; cache_hits=0; t0=time.time()
        for k,s in enumerate(time_grid.itertuples(),1):
            Cold=np.array([self.por*v['C'] for v in q]); b=boundary[s.stage]; baseguess=[g.copy() for g in guess]; flash_cache={}
            def residual(x):
                nonlocal residual_calls,flash_requests,cache_hits
                residual_calls+=1; zz,PP=self.decode(x); qq=[]
                for j in range(self.N):
                    flash_requests+=1; key=(float(PP[j]),float(zz[j,0]),float(zz[j,1]))
                    if key in flash_cache: v=flash_cache[key]; cache_hits+=1
                    else: v=self.pr.flash(PP[j],zz[j],baseguess[j]); flash_cache[key]=v
                    qq.append(v)
                FF=self.flux(qq,PP,b)
                return np.concatenate([(self.por*qq[j]['C']-Cold[j]+s.dt_s/self.dx*(FF[j+1]-FF[j]))/self.Cscale for j in range(self.N)])
            solve_options=dict(bounds=(np.tile([-7,-7,1.3],self.N),np.tile([7,7,3.3],self.N)),jac='2-point',x_scale=np.tile([1,1,1],self.N),max_nfev=80,xtol=1e-9,ftol=1e-9,gtol=1e-9,diff_step=2e-5)
            if self.sparsity is not None:
                solve_options.update(jac_sparsity=self.sparsity, tr_solver='lsmr')
            sol=least_squares(residual,X,**solve_options)
            rmax=float(np.max(np.abs(sol.fun))); z,P=self.decode(sol.x); q=[self.pr.flash(P[j],z[j],baseguess[j]) for j in range(self.N)]; guess=[v['lnK'] for v in q]; X=sol.x; FF=self.flux(q,P,b)
            if (not sol.success) or rmax>2e-5: raise RuntimeError(f'PR global failure at step {k}: residual={rmax}')
            Mnow=self.por*self.dx*np.sum([v['C'] for v in q],axis=0); bal=Mnow-Mprev-s.dt_s*(FF[0]-FF[-1]); sc=np.maximum(abs(Mnow)+abs(Mprev)+abs(s.dt_s*(FF[0]-FF[-1])),1e-30); Mprev=Mnow
            for i,n in enumerate(['CH4','CO2','nC4']): mass.append(dict(accepted_step=k,time_s=s.time_s,component=n,balance_error=bal[i],relative_balance_error=abs(bal[i])/sc[i]))
            for j,v in enumerate(q):
                if v['phase']!=prev[j]: events.append(dict(accepted_step=k,time_s=s.time_s,stage=s.stage,cell=j+1,from_phase=prev[j],to_phase=v['phase'],S_pr=v['S'],P_MPa=P[j]/1e6,z_CH4=z[j,0],z_CO2=z[j,1],z_nC4=z[j,2]))
                rows.append(dict(accepted_step=k,time_s=s.time_s,stage=s.stage,dt_s=s.dt_s,cell=j+1,P_MPa=P[j]/1e6,z_CH4=z[j,0],z_CO2=z[j,1],z_nC4=z[j,2],S_pr=v['S'],phase_pr=v['phase'],beta_pr=v['beta'],residual_max=rmax,nfev=sol.nfev))
            prev=[v['phase'] for v in q]
        D=pd.DataFrame(rows);E=pd.DataFrame(events);MB=pd.DataFrame(mass)
        D.to_csv(outdir/'pr_global_states.csv',index=False);E.to_csv(outdir/'pr_phase_events.csv',index=False);MB.to_csv(outdir/'pr_mass_balance.csv',index=False)
        summary=dict(runtime_s=time.time()-t0,accepted_steps=len(time_grid),preprocessing_flash_calls=int(preprocessing_calls),preprocessing_PR_EOS_evaluations=int(preprocessing_eos),online_local_flash_requests_in_global_residual=int(flash_requests),online_executed_local_flash_calls_with_exact_cache=int(self.pr.calls-preprocessing_calls),online_PR_EOS_evaluations=int(self.pr.eos_evals-preprocessing_eos),global_residual_evaluations=int(residual_calls),exact_flash_cache_hits=int(cache_hits),mean_flash_iterations=float(self.pr.iterations/max(self.pr.calls,1)),max_scaled_residual=float(D.residual_max.max()),max_component_relative_mass_balance_error=float(MB.relative_balance_error.max()),phase_events=len(E))
        json.dump(summary,open(outdir/'pr_solver_summary.json','w'),indent=2)
        return D,E,MB,summary
