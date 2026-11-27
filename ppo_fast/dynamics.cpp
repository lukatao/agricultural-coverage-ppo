// Scalar translation of the frozen V1 constrained RK4 loop. Compiled without contraction.
#include <algorithm>
#include <cmath>
using std::min;
using std::max;
static double clip(double x, double a, double b) { return min(max(x,a),b); }
struct Robot {
    double kmax, vmax, omega, lateral, accel, decel, rate, alpha, axle, steer;
    double auxiliary, distance, cubic, turning;
    double speed(double k) const {
        double a = std::abs(k);
        return a <= 1e-12 ? vmax : min({vmax, omega/a, std::sqrt(lateral/a)});
    }
    double curvature(double v) const {
        return v <= 1e-9 ? kmax : min({kmax, omega/v, lateral/(v*v)});
    }
};
enum Flag {speed_sat, turn_limit, braking, curvature_sat, rate_sat, accel_sat, alpha_sat, stationary};
static void controls(const Robot& r, bool target, double v, double k, double c0, double c1,
                     double h, unsigned char* flags, double& a, double& sigma) {
    double limit = r.speed(k);
    if (target) {
        if (c0<0 || c0>r.vmax+1e-12) flags[speed_sat]=1;
        if (std::abs(c1)>r.kmax+1e-12) flags[curvature_sat]=1;
        double requested=clip(c0,0,r.vmax), turn=r.speed(clip(c1,-r.kmax,r.kmax));
        if (requested>turn+1e-9) {
            flags[turn_limit]=1;
            if(v>turn+1e-6) flags[braking]=1;
        }
        double desired=(min({requested,limit,turn})-v)/h;
        a=clip(desired,-r.decel,r.accel);
        if(std::abs(a-desired)>1e-9) flags[accel_sat]=1;
    } else {
        a=clip(c0,-r.decel,r.accel);
        if(std::abs(a-c0)>1e-9) flags[accel_sat]=1;
    }
    if(v+a*h<0) {a=-v/h;flags[speed_sat]=1;}
    else if(v+a*h>limit) {
        double cap=max(-r.decel,(limit-v)/h);
        if(a>cap+1e-12) {a=cap;flags[limit<r.vmax-1e-9 ? turn_limit : speed_sat]=1;}
    }
    double next=max(0.,v+a*h), mid=0.5*(v+next), ds=mid*h;
    if(ds<=1e-12) {
        if(std::abs(target ? c1-k : c1)>1e-6) flags[stationary]=1;
        sigma=0;return;
    }
    double temporal=mid*r.rate;
    if(r.axle>0) {
        double lk=r.axle*k;
        temporal=min(temporal,((1.+lk*lk)/r.axle)*r.steer);
    }
    double bound=min(r.rate,temporal/max(mid,1e-12));
    double desired=target ? (clip(c1,-r.kmax,r.kmax)-k)/ds : c1;
    sigma=clip(desired,-bound,bound);
    if(std::abs(sigma-desired)>1e-9) flags[rate_sat]=1;
    double env=min({r.kmax,r.curvature(next),r.curvature(mid)}), prop=k+sigma*ds;
    if(std::abs(prop)>env+1e-12 && std::abs(prop)>std::abs(k)) {
        sigma=clip((clip(prop,-env,env)-k)/ds,-bound,bound);
        flags[env<r.kmax-1e-9 ? turn_limit : curvature_sat]=1;
    }
    double vmax=max(v,next), vv=vmax*vmax;
    if(vmax>1e-9 && std::abs(a*k+vv*sigma)>r.alpha+1e-12) {
        double restricted=clip(sigma,max(-bound,(-r.alpha-a*k)/vv),min(bound,(r.alpha-a*k)/vv));
        if(std::abs(restricted-sigma)>1e-9) {sigma=restricted;flags[alpha_sat]=1;}
    }
    double final=k+sigma*ds;
    if(std::abs(final)>r.kmax) {sigma=(clip(final,-r.kmax,r.kmax)-k)/ds;flags[curvature_sat]=1;}
}
extern "C" void integrate(const double* spec,int target,double c0,double c1,double h,int count,
                          double* states,double* d,unsigned char* flags) {
    Robot r{spec[0],spec[1],spec[2],spec[3],spec[4],spec[5],spec[6],spec[7],spec[8],spec[9],
            spec[10],spec[11],spec[12],spec[13]};
    double x=states[0], y=states[1], th=states[2], v=states[3], k=states[4];
    d[0]=d[1]=v;d[2]=std::abs(k);d[3]=std::abs(v*k);d[7]=max(0.,v-r.speed(k));
    for(int i=1;i<=count;i++) {
        double a,sig;controls(r,target!=0,v,k,c0,c1,h,flags,a,sig);
        double half=0.5*h, v0=max(0.,v), v1=max(0.,v+half*a), v3=max(0.,v+h*a);
        double dk0=v0*sig, dk1=v1*sig, dk3=v3*sig;
        double w0=v0*k, th1=th+half*w0, w1=v1*(k+half*dk0);
        double th2=th+half*w1, w2=v1*(k+half*dk1), th3=th+h*w2, w3=v3*(k+h*dk1);
        double c=h/6.0;
        x+=c*(v0*std::cos(th)+2.*(v1*std::cos(th1))+2.*(v1*std::cos(th2))+v3*std::cos(th3));
        y+=c*(v0*std::sin(th)+2.*(v1*std::sin(th1))+2.*(v1*std::sin(th2))+v3*std::sin(th3));
        th=std::fmod((th+c*(w0+2.*w1+2.*w2+w3))+M_PI,2*M_PI);
        if(th<0) th+=2*M_PI;
        th-=M_PI;
        double nk=k+c*(dk0+2.*dk1+2.*dk1+dk3), mid=0.5*(v+v3);
        d[8]+=mid*h;
        d[9]+=(r.auxiliary+r.distance*mid+r.cubic*std::pow(mid,3.)+r.turning*std::abs(mid*(0.5*(k+nk))))*h;
        d[10]+=a*h;d[11]+=sig*h;d[12]+=(mid*sig)*h;
        d[6]=max(d[6],std::abs(a*k+max(v,v3)*max(v,v3)*sig));
        v=v3;k=nk;
        d[0]=max(d[0],v);d[1]=min(d[1],v);d[2]=max(d[2],std::abs(k));d[3]=max(d[3],std::abs(v*k));
        d[4]=max(d[4],std::abs(sig));d[5]=max(d[5],std::abs(a));d[7]=max(d[7],v-r.speed(k));
        double* dst=states+5*i;dst[0]=x;dst[1]=y;dst[2]=th;dst[3]=v;dst[4]=k;
    }
}
