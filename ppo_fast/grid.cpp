#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <cstdint>
using std::min;
using std::max;
struct Point {double x,y;};
struct Grid {
    double x,y,inv;
    int nx,ny;
    int col(double px) const {return int(std::floor((px-x)*inv));}
    int row(double py) const {return int(std::floor((py-y)*inv));}
    int index(double px,double py) const {
        int i=col(px),j=row(py);
        return i>=0 && j>=0 && i<nx && j<ny ? j*nx+i : -1;
    }
};
static double cross(Point a, Point b, Point p) {return (b.x-a.x)*(p.y-a.y)-(b.y-a.y)*(p.x-a.x);}
static bool triangle(Point a,Point b,Point c,Point p) {
    double ab=cross(a,b,p),bc=cross(b,c,p),ca=cross(c,a,p);
    return (ab>=-1e-9 && bc>=-1e-9 && ca>=-1e-9) || (ab<=1e-9 && bc<=1e-9 && ca<=1e-9);
}
extern "C" void treatment(const double* geom,int nx,int ny,const double* xs,const double* ys,
                          const unsigned char* field,const unsigned char* obstacle,unsigned char* covered,
                          uint16_t* count,const double* path,int n,int stride,const double* lengths,
                          const double* offsets,double width,double* result) {
    Grid grid{geom[0],geom[1],geom[2],nx,ny};
    double xmin=path[0],xmax=xmin,ymin=path[1],ymax=ymin;
    for(int i=1;i<n;i++){
        xmin=min(xmin,path[i*stride]);xmax=max(xmax,path[i*stride]);
        ymin=min(ymin,path[i*stride+1]);ymax=max(ymax,path[i*stride+1]);
    }
    double half=0.5*width,pad=half+geom[3];
    int x0=max(0,grid.col(xmin-pad)),x1=min(nx,1+int(std::ceil((xmax+pad-grid.x)*grid.inv)));
    int y0=max(0,grid.row(ymin-pad)),y1=min(ny,1+int(std::ceil((ymax+pad-grid.y)*grid.inv)));
    if(x0>=x1 || y0>=y1)return;
    std::vector<unsigned char> visited(size_t(x1-x0)*(y1-y0));
    double distance=0;
    int first=0;
    for(int i=1;i<n;i++) {
        distance+=lengths[i-1];
        if(distance<0.25 && i!=n-1)continue;
        const double *p=path+first*stride,*q=path+i*stride;
        double length=distance;distance=0;first=i;
        if(length<=1e-12)continue;
        double sx=-std::sin(p[2]),sy=std::cos(p[2]),tx=-std::sin(q[2]),ty=std::cos(q[2]);
        Point a{p[0]-half*sx,p[1]-half*sy},b{p[0]+half*sx,p[1]+half*sy};
        Point c{q[0]+half*tx,q[1]+half*ty},d{q[0]-half*tx,q[1]-half*ty};
        int lo=max(x0,grid.col(min({a.x,b.x,c.x,d.x}))),bottom=max(y0,grid.row(min({a.y,b.y,c.y,d.y})));
        int hi=min(x1,1+int(std::ceil((max({a.x,b.x,c.x,d.x})-grid.x)*grid.inv)));
        int top=min(y1,1+int(std::ceil((max({a.y,b.y,c.y,d.y})-grid.y)*grid.inv)));
        for(int j=bottom;j<top;j++)for(int k=lo;k<hi;k++) {
            int v=(j-y0)*(x1-x0)+k-x0;
            Point pixel{xs[k],ys[j]};
            if(!visited[v] && (triangle(a,b,c,pixel)||triangle(a,c,d,pixel))) {
                visited[v]=1;
                int idx=j*nx+k;
                if(!covered[idx])result[field[idx]?0:(obstacle[idx]?1:2)]++;
                covered[idx]=1;count[idx]++;
            }
        }
        double angle=std::fmod(q[2]-p[2]+M_PI,2*M_PI);
        if(angle<0)angle+=2*M_PI;
        double mid=p[2]+0.5*(angle-M_PI),mx=0.5*(p[0]+q[0]),my=0.5*(p[1]+q[1]);
        double ux=-std::sin(mid),uy=std::cos(mid);
        int inside=0;
        for(int j=0;j<11;j++){
            int idx=grid.index(mx+offsets[j]*ux,my+offsets[j]*uy);
            if(idx>=0)inside+=field[idx]!=0;
        }
        result[3]+=(width*length)*(inside/11.);
    }
}
// Fused V1 nearest-raster sampling and 1x1 or 2x2 pooling. Gaussian markers remain NumPy float32.
extern "C" void sample(const double* geom,int nx,int ny,const unsigned char* field,
                       const unsigned char* obstacle,const unsigned char* covered,const unsigned char* nav,
                       const double* axis,int side,int sub,double cx,double cy,double ct,double st,int local,float* out) {
    Grid grid{geom[0],geom[1],geom[2],nx,ny};
    int plane=side*side;
    for(int j=0;j<side;j++)for(int i=0;i<side;i++) {
        int f=0,o=0,c=0,m=0;
        for(int a=0;a<sub;a++)for(int b=0;b<sub;b++) {
            double u=axis[i*sub+b],v=axis[j*sub+a];
            int id=grid.index(cx+ct*u-st*v,cy+st*u+ct*v);
            if(id<0)continue;
            f+=field[id];o|=obstacle[id];c+=(field[id]&&covered[id]);m|=nav[id];
        }
        int idx=j*side+i;
        float factor=1.0f/(sub*sub);
        out[idx]=f*factor;out[plane+idx]=o;out[2*plane+idx]=c*factor;
        out[3*plane+idx]=(f-c)*factor;
        if(local)out[4*plane+idx]=m;
    }
}
extern "C" void rays(double x,double y,const double* dx,const double* dy,int n,
                     const double* edges,int m,double range,float* out) {
    for(int i=0;i<n;i++){
        double best=range;
        for(int j=0;j<m;j++){
            const double* e=edges+4*j;
            double sx=e[2]-e[0],sy=e[3]-e[1],px=e[0]-x,py=e[1]-y;
            double den=dx[i]*sy-dy[i]*sx;
            if(std::abs(den)<=1e-12)continue;
            double t=(px*sy-py*sx)/den,u=(px*dy[i]-py*dx[i])/den;
            if(t>=0 && t<=range && u>=-1e-9 && u<=1.+1e-9)best=min(best,t);
        }
        out[i]=float(best);
    }
}
static double point_dist(Point p,Point a,Point b,bool relative=false) {
    double x=b.x-a.x,y=b.y-a.y,px=p.x-a.x,py=p.y-a.y;
    double s=std::clamp((px*x+py*y)/max(x*x+y*y,1e-18),0.,1.);
    double dx=relative ? px-s*x : p.x-(a.x+s*x),dy=relative ? py-s*y : p.y-(a.y+s*y);
    return dx*dx+dy*dy;
}
extern "C" double clearance(const double* path,int n,const double* edges,int m) {
    double best=std::numeric_limits<double>::infinity();
    for(int i=0;i<n;i++){
        Point p{path[2*i],path[2*i+1]};
        if(n>1 && i==n-1)break;
        for(int j=0;j<m;j++){
            const double* e=edges+4*j;
            Point a{e[0],e[1]},b{e[2],e[3]};
            best=min(best,point_dist(p,a,b,n==1));
            if(n==1)continue;
            Point q{path[2*i+2],path[2*i+3]};
            if(cross(a,b,p)*cross(a,b,q)<0 && cross(p,q,a)*cross(p,q,b)<0)return 0;
            best=min(best,point_dist(q,a,b));
            double dx=q.x-p.x,dy=q.y-p.y;
            if(dx*dx+dy*dy>1e-18)best=min({best,point_dist(a,p,q),point_dist(b,p,q)});
        }
        if(n>1 && best<=1e-18)return 0;
    }
    return std::sqrt(best);
}
