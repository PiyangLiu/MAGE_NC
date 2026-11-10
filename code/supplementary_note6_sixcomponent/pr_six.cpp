// Fast, deterministic C++17 -O3 six-component PR inverse for FIXED (T, mu_i).
// Thermodynamic task is exactly the same as mage_six.cpp, including both
// phase-root pressures, all twelve phase component densities and stable phase.
// Newton has damped backtracking, analytical cubic roots, and previous-state
// continuation; composition/pressure Jacobians are computed locally by FD.
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
constexpr int N=6,D=6;
constexpr double PI=3.14159265358979323846,SQ2=1.4142135623730951;
constexpr double R=8.31446261815324,T=300.,RT=R*T,PREF=1e5;
constexpr double TC[N]={190.564,305.322,369.83,425.125,469.7,507.6};
constexpr double PC[N]={4.5992e6,4.872e6,4.248e6,3.796e6,3.370e6,3.025e6};
constexpr double W[N]={.01142,.0995,.1523,.2002,.2515,.3013};
double AI[N],BI[N],AIJ[N][N];

void coefficients(){
  for(int i=0;i<N;i++){
    double k=.37464+1.54226*W[i]-.26992*W[i]*W[i];
    AI[i]=.45724*R*R*TC[i]*TC[i]/PC[i]*std::pow(1+k*(1-std::sqrt(T/TC[i])),2);
    BI[i]=.07780*R*TC[i]/PC[i];
  }
  for(int i=0;i<N;i++)for(int j=0;j<N;j++)AIJ[i][j]=std::sqrt(AI[i]*AI[j]);
}

int cubic(double a,double b,double c,double roots[3]){
  double p=b-a*a/3,q=2*a*a*a/27-a*b/3+c;
  double discr=q*q/4+p*p*p/27;
  if(discr>1e-14){
    double s=std::sqrt(discr);
    roots[0]=std::cbrt(-q/2+s)+std::cbrt(-q/2-s)-a/3;return 1;
  }
  if(std::abs(p)<1e-16){roots[0]=-a/3;return 1;}
  double d=std::sqrt(std::max(1e-300,-p*p*p/27));
  double arg=std::clamp((-q/2)/d,-1.,1.);
  double mag=2*std::sqrt(std::max(0.,-p/3)),angle=std::acos(arg);
  for(int k=0;k<3;k++)roots[k]=mag*std::cos((angle+2*k*PI)/3)-a/3;
  std::sort(roots,roots+3);return 3;
}

struct State{double eta[N],q[N],p,Z;};

bool calc(const double vars[D],int phase,State& result,int& eos){
  double P=1e6*std::exp(vars[0]);
  if(!(P>2e5 && P<7e6))return false;
  double logits[N];double largest=0;
  for(int j=0;j<N-1;j++){logits[j]=vars[j+1];largest=std::max(largest,logits[j]);}
  logits[N-1]=0;
  double composition[N],total=0;
  for(int j=0;j<N;j++){composition[j]=std::exp(logits[j]-largest);total+=composition[j];}
  for(int j=0;j<N;j++)composition[j]/=total;
  ++eos;
  double bm=0,am=0,sa[N]={};
  for(int i=0;i<N;i++){
    bm+=composition[i]*BI[i];
    for(int j=0;j<N;j++){
      double term=AIJ[i][j]*composition[j];
      sa[i]+=term;am+=composition[i]*term;
    }
  }
  double A=am*P/(RT*RT),B=bm*P/RT;
  double roots[3];int nr=cubic(B-1,A-3*B*B-2*B,-(A*B-B*B-B*B*B),roots);
  double good[3];int cnt=0;
  for(int j=0;j<nr;j++)if(roots[j]>B+1e-10)good[cnt++]=roots[j];
  if(cnt==0)return false;
  double Z=phase==0?good[0]:good[cnt-1];
  double logratio=std::log((Z+(1+SQ2)*B)/(Z+(1-SQ2)*B));
  result.p=P;result.Z=Z;
  for(int i=0;i<N;i++){
    double lp=BI[i]/bm*(Z-1)-std::log(Z-B)
     -A/(2*SQ2*B)*(2*sa[i]/am-BI[i]/bm)*logratio;
    result.eta[i]=std::log(composition[i]*P/PREF)+lp;
    result.q[i]=composition[i]*P/(Z*RT);
  }
  return std::isfinite(result.eta[0]);
}

// Exact first derivatives of each PR fugacity coordinate with respect to
// (log P, five composition logits). The compressibility-root derivative is
// obtained by implicit differentiation of the PR cubic; this is nonsingular
// precisely because the chosen states are well away from criticality.
bool calc_analytic(const double vars[D],int phase,State& result,
                   double jac[D][D],int& eos){
  double P=1e6*std::exp(vars[0]);
  if(!(P>2e5&&P<7e6))return false;
  double logits[N],largest=0;
  for(int i=0;i<N-1;i++){logits[i]=vars[i+1];largest=std::max(largest,logits[i]);}
  logits[N-1]=0;
  double x[N],total=0;
  for(int i=0;i<N;i++){x[i]=std::exp(logits[i]-largest);total+=x[i];}
  for(int i=0;i<N;i++)x[i]/=total;
  ++eos;
  double bm=0,am=0,s[N]={};
  for(int i=0;i<N;i++){
    bm+=x[i]*BI[i];
    for(int j=0;j<N;j++){
      double term=AIJ[i][j]*x[j];s[i]+=term;am+=x[i]*term;
    }
  }
  double A=am*P/(RT*RT),B=bm*P/RT;
  double roots[3];int nr=cubic(B-1,A-3*B*B-2*B,-(A*B-B*B-B*B*B),roots);
  double good[3];int cnt=0;
  for(int j=0;j<nr;j++)if(roots[j]>B+1e-10)good[cnt++]=roots[j];
  if(cnt==0)return false;
  double Z=phase==0?good[0]:good[cnt-1];
  double cplus=1+SQ2,cminus=1-SQ2;
  double lg=std::log((Z+cplus*B)/(Z+cminus*B));
  double coef=A/(2*SQ2*B);
  double t[N],h[N];
  result.p=P;result.Z=Z;
  for(int i=0;i<N;i++){
    t[i]=BI[i]/bm;h[i]=2*s[i]/am-t[i];
    result.eta[i]=std::log(x[i]*P/PREF)+t[i]*(Z-1)-std::log(Z-B)-coef*h[i]*lg;
    result.q[i]=x[i]*P/(Z*RT);
  }
  double FZ=3*Z*Z+2*(B-1)*Z+(A-3*B*B-2*B);
  double FA=Z-B;
  double FB=Z*Z+(-6*B-2)*Z-A+2*B+3*B*B;
  if(std::abs(FZ)<1e-10)return false;
  for(int j=0;j<D;j++){
    double dx[N]={};
    if(j>0){
      int k=j-1;
      for(int i=0;i<N;i++)dx[i]=x[i]*((i==k?1.:0.)-x[k]);
    }
    double dbm=0,dam=0,ds[N]={};
    for(int i=0;i<N;i++){
      dbm+=dx[i]*BI[i];dam+=2*dx[i]*s[i];
      for(int k=0;k<N;k++)ds[i]+=AIJ[i][k]*dx[k];
    }
    double dA=(P/(RT*RT))*dam+(j==0?A:0.);
    double dB=(P/RT)*dbm+(j==0?B:0.);
    double dZ=-(FA*dA+FB*dB)/FZ;
    double dcoef=coef*(dA/A-dB/B);
    double dlg=(dZ+cplus*dB)/(Z+cplus*B)-(dZ+cminus*dB)/(Z+cminus*B);
    for(int i=0;i<N;i++){
      double dt=-t[i]*dbm/bm;
      double dh=2*(ds[i]/am-s[i]*dam/(am*am))-dt;
      double dlnphi=dt*(Z-1)+t[i]*dZ-(dZ-dB)/(Z-B)
        -dcoef*h[i]*lg-coef*dh*lg-coef*h[i]*dlg;
      double dlogx=j>0?((i==j-1?1.:0.)-x[j-1]):0.;
      jac[i][j]=dlogx+(j==0?1.:0.)+dlnphi;
    }
  }
  return std::isfinite(result.eta[0]);
}

bool linear(double A[D][D],double b[D],double x[D]){
  double aug[D][D+1];
  for(int i=0;i<D;i++){
    for(int j=0;j<D;j++)aug[i][j]=A[i][j];
    aug[i][D]=b[i];
  }
  for(int k=0;k<D;k++){
    int pivot=k;
    for(int i=k+1;i<D;i++)if(std::abs(aug[i][k])>std::abs(aug[pivot][k]))pivot=i;
    if(std::abs(aug[pivot][k])<1e-11)return false;
    for(int j=k;j<=D;j++)std::swap(aug[k][j],aug[pivot][j]);
    double scale=aug[k][k];for(int j=k;j<=D;j++)aug[k][j]/=scale;
    for(int i=0;i<D;i++)if(i!=k){
      double factor=aug[i][k];
      for(int j=k;j<=D;j++)aug[i][j]-=factor*aug[k][j];
    }
  }
  for(int i=0;i<D;i++)x[i]=aug[i][D];
  return true;
}

bool invert(const double target[N],int phase,double v[D],State& out,
            int& steps,int& evaluations,double& residual){
  steps=0;evaluations=0;
  for(int it=0;it<30;it++){
    State s;
    double J[D][D];
#ifdef ANALYTIC_JAC
    if(!calc_analytic(v,phase,s,J,evaluations))return false;
#else
    if(!calc(v,phase,s,evaluations))return false;
#endif
    double r[D],norm=0;
    for(int i=0;i<D;i++){r[i]=s.eta[i]-target[i];norm=std::max(norm,std::abs(r[i]));}
    if(norm<5e-11){out=s;steps=it;residual=norm;return true;}
#ifndef ANALYTIC_JAC
    constexpr double h=2e-5;
    for(int j=0;j<D;j++){
      double w[D];for(int k=0;k<D;k++)w[k]=v[k];w[j]+=h;
      State moved;if(!calc(w,phase,moved,evaluations))return false;
      for(int i=0;i<D;i++)J[i][j]=(moved.eta[i]-s.eta[i])/h;
    }
#endif
    double rhs[D],dx[D];for(int i=0;i<D;i++)rhs[i]=-r[i];
    if(!linear(J,rhs,dx))return false;
    double scale=1.;
    for(int j=0;j<D;j++){
      double bound=j==0?.6:.9;
      if(std::abs(dx[j])>bound)scale=std::min(scale,bound/std::abs(dx[j]));
    }
    bool accepted=false;
    for(int ls=0;ls<18;ls++){
      double w[D];for(int k=0;k<D;k++)w[k]=v[k]+scale*dx[k];
      State next;
      if(calc(w,phase,next,evaluations)){
        double nextnorm=0;
        for(int i=0;i<D;i++)nextnorm=std::max(nextnorm,std::abs(next.eta[i]-target[i]));
        if(nextnorm<norm*(1-1e-5*scale)){
          for(int k=0;k<D;k++)v[k]=w[k];accepted=true;break;
        }
      }
      scale*=.5;
    }
    if(!accepted)return false;
  }
  return false;
}
}

// 18 columns: [P_L,P_V, qL_0..5, qV_0..5, Z_L,Z_V,label,max_eta_residual]
// Return 0 on success, otherwise 1-based index of failed query.
extern "C" int pr_six_batch(int n,const double *eta,double *out,
                            int *iterations,int *eos_counts,int warm){
  coefficients();
  double cached[2][D]={{std::log(2.35),-.86386514,-.98036718,-.79103361,-.07980421,-.04494129},
                       {std::log(2.35),5.00548495,3.35687506,2.42413345,2.02238111,.99146031}};
  for(int j=0;j<n;j++){
    State res[2];double maxerr=0;
    for(int ph=0;ph<2;ph++){
      double v[D];for(int k=0;k<D;k++)v[k]=cached[ph][k];
      if(!warm){
        v[0]=std::log(2.35);
        const double q[D-1]={-.86386514,-.98036718,-.79103361,-.07980421,-.04494129};
        const double s[D-1]={5.00548495,3.35687506,2.42413345,2.02238111,.99146031};
        for(int k=1;k<D;k++)v[k]=ph==0?q[k-1]:s[k-1];
      }
      int steps=0,eos=0;double err=0;
      if(!invert(eta+N*j,ph,v,res[ph],steps,eos,err))return j+1;
      if(warm)for(int k=0;k<D;k++)cached[ph][k]=v[k];
      iterations[2*j+ph]=steps;eos_counts[2*j+ph]=eos;
      maxerr=std::max(maxerr,err);
    }
    double* o=out+18*j;o[0]=res[0].p;o[1]=res[1].p;
    for(int ph=0;ph<2;ph++){
      for(int c=0;c<N;c++)o[2+ph*N+c]=res[ph].q[c];
      o[14+ph]=res[ph].Z;
    }
    o[16]=res[1].p>res[0].p?1.:0.;o[17]=maxerr;
  }
  return 0;
}

// Independent derivative audit: analytic 6x6 PR chemical-potential Jacobian
// against central finite differences at a provided physical phase state.
extern "C" int pr_six_audit_jac(const double *variables,int phase,
                               double *analytic,double *finite){
  coefficients();State s;double J[D][D];int calls=0;
  if(!calc_analytic(variables,phase,s,J,calls))return 1;
  constexpr double h=2e-6;
  for(int j=0;j<D;j++){
    double up[D],down[D];for(int k=0;k<D;k++){up[k]=variables[k];down[k]=variables[k];}
    up[j]+=h;down[j]-=h;
    State sp,sm;
    if(!calc(up,phase,sp,calls)||!calc(down,phase,sm,calls))return 2;
    for(int i=0;i<D;i++){
      analytic[i*D+j]=J[i][j];
      finite[i*D+j]=(sp.eta[i]-sm.eta[i])/(2*h);
    }
  }
  return 0;
}
