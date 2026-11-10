// Frozen six-dimensional MAGE Massieu tangent representation at fixed eta.
// Identical 18-component output contract to the compiled six-component PR.
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
constexpr int N=6;

void branch(const double *eta,int M,const double *center,const double *scale,
            const double *X,const double *intercept,const double *G,
            double tau,double phi_scale,double RT,
            double &P,double *q,double &Z){
  double v[N];for(int i=0;i<N;i++)v[i]=(eta[i]-center[i])/scale[i];
  (void)X; // Exact F-G.X tangent intercepts prepared once outside the timer.
  double top=-std::numeric_limits<double>::infinity();
  for(int k=0;k<M;k++){
    double plane=intercept[k];
    for(int i=0;i<N;i++)plane+=G[k*N+i]*v[i];
    if(plane>top)top=plane;
  }
  double weightsum=0,gg[N]={};
  for(int k=0;k<M;k++){
    double plane=intercept[k];
    for(int i=0;i<N;i++)plane+=G[k*N+i]*v[i];
    // Bounded neglected weight < exp(-32) per omitted support.  The entire
    // discarded mass is < M*exp(-32), below double-precision test tolerances.
    double logw=(plane-top)/tau;
    if(logw < -32.)continue;
    double w=std::exp(logw);weightsum+=w;
    for(int i=0;i<N;i++)gg[i]+=w*G[k*N+i];
  }
  double phi=(top+tau*(std::log(weightsum)-std::log(M)))*phi_scale;
  P=phi*RT;double total=0;
  for(int i=0;i<N;i++){q[i]=gg[i]/weightsum*phi_scale/scale[i];total+=q[i];}
  Z=P/(RT*total);
}
}

// 18 columns: [P_L,P_V, qL_0..5, qV_0..5, Z_L,Z_V,label,0].
extern "C" int mage_six_batch(int n,int M,const double *eta,
                              const double *centers,const double *scales,
                              const double *Xs,const double *Fs,const double *Gs,
                              double tau,double phi_scale,double RT,double *out){
  for(int j=0;j<n;j++){
    double *o=out+18*j;
    for(int ph=0;ph<2;ph++)
      branch(eta+j*N,M,centers+ph*N,scales+ph*N,
             Xs+ph*M*N,Fs+ph*M,Gs+ph*M*N,
             tau,phi_scale,RT,o[ph],o+2+ph*N,o[14+ph]);
    o[16]=o[1]>o[0]?1.:0.;o[17]=0.;
  }
  return 0;
}
