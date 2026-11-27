import math
from functools import lru_cache
import numpy as np
from ppo_coverage_planning_v1.coverage_map import IncrementalCoverageMap, CoverageStepDelta
from .native import lib,ptr

@lru_cache(maxsize=64)
def global_axes(size,span,res):
    cell=span/size
    centers=-0.5*span+(np.arange(size,dtype=np.float64)+0.5)*cell
    axis=centers
    if cell>3.5*res:
        axis=np.empty(2*size)
        axis[0::2]=centers-0.25*cell
        axis[1::2]=centers+0.25*cell
    markers=-0.5*span+(np.arange(size,dtype=np.float32)+0.5)*np.float32(cell)
    return axis,markers,np.float32(1./(2.*(0.9*cell)**2))

@lru_cache(maxsize=64)
def local_axis(size,extent):
    return np.linspace(-0.5*extent,0.5*extent,size)

@lru_cache(maxsize=32)
def bar_offsets(width):
    return np.linspace(-0.5*width,0.5*width,11)

class FastMap(IncrementalCoverageMap):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.grid=np.array([self.x0,self.y0,1./self.resolution_m,self.resolution_m])
        self._nav_key=object()

    def set_navigable_polygon(self,navigable_polygon):
        key=navigable_polygon.wkb if navigable_polygon is not None else None
        if key!=self._nav_key:
            super().set_navigable_polygon(navigable_polygon)
            self._nav_key=key

    def extract_local_map(self,x,y,theta,*,patch_size=24,patch_extent_m=36.):
        # Match the reference's four-decimal per-instance lookup key.
        key=(patch_size,round(patch_extent_m,4))
        if key not in self._local_grid_cache:
            self._local_grid_cache[key]=local_axis(patch_size,patch_extent_m)
        axis=self._local_grid_cache[key]
        out=np.empty((5,patch_size,patch_size),np.float32)
        lib.sample(ptr(self.grid),self.nx,self.ny,ptr(self.in_field),ptr(self.in_obstacle),ptr(self.covered),
                   ptr(self.in_navigable),ptr(axis),patch_size,1,x,y,math.cos(theta),math.sin(theta),1,ptr(out))
        return out

    def extract_global_map(self,x,y,theta,*,global_size=32,frame="ego_aligned"):
        ego=frame=="ego_aligned"
        span=max(24.,4.+(self.field_diag_m if ego else max(self.field_width_m,self.field_height_m)))
        axis,markers,inv=global_axes(global_size,span,self.resolution_m)
        cell=span/global_size
        ct,st=math.cos(theta),math.sin(theta)
        dx,dy=x-self.field_cx,y-self.field_cy
        if ego:
            rx,ry=ct*dx+st*dy,-st*dx+ct*dy
            fx,fy=rx+1.5*cell,ry
        else:
            rx,ry=dx,dy
            fx,fy=rx+1.5*cell*ct,ry+1.5*cell*st
        out=np.empty((6,global_size,global_size),np.float32)
        lib.sample(ptr(self.grid),self.nx,self.ny,ptr(self.in_field),ptr(self.in_obstacle_dilated),
                   ptr(self.covered),ptr(self.in_navigable),ptr(axis),global_size,len(axis)//global_size,
                   self.field_cx,self.field_cy,ct if ego else 1.,st if ego else 0.,0,ptr(out))
        for channel,px,py in [(4,rx,ry),(5,fx,fy)]:
            out[channel]=np.exp(-((markers[None,:]-np.float32(px))**2+(markers[:,None]-np.float32(py))**2)*inv)
        return out

    def update_from_substeps(self,substeps,working_width_m,*,tool_active=True):
        if not tool_active or len(substeps)<2:
            return super().update_from_substeps(substeps,working_width_m,tool_active=tool_active)
        arr=np.ascontiguousarray(substeps,dtype=np.float64)
        if arr.ndim!=2 or arr.shape[1]<3:
            raise ValueError("Swept poses need x, y and heading")
        delta=np.diff(arr[:,:2],axis=0)
        lengths=np.hypot(delta[:,0],delta[:,1])
        totals=np.zeros(4)
        if lengths.sum()>1e-12:
            offsets=bar_offsets(working_width_m)
            lib.treatment(ptr(self.grid),self.nx,self.ny,ptr(self.xs),ptr(self.ys),ptr(self.in_field),
                          ptr(self.in_obstacle),ptr(self.covered),ptr(self.treated_count),ptr(arr),len(arr),
                          arr.shape[1],ptr(lengths),ptr(offsets),working_width_m,ptr(totals))
        fields,holes,outside=map(int,totals[:3])
        gross=float(totals[3])
        self._covered_field_cells+=fields
        self._covered_obs_cells+=holes
        self._covered_out_cells+=outside
        self._gross_in_field_m2+=gross
        pct=100.*(self._covered_field_cells/max(1,self.field_cell_count))
        treated=self.source_area_m2*(pct/100.)
        repeated=max(0.,self._gross_in_field_m2-treated)
        change=repeated-self._cum_repeated_m2
        self._cum_repeated_m2=repeated
        return CoverageStepDelta(fields*self.cell_area_m2*self._area_scale,change,gross,outside*self.cell_area_m2,
                                 holes*self.cell_area_m2,pct,max(0.,self.source_area_m2-treated))
