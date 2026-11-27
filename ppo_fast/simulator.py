import math
import numpy as np
from ppo_coverage_planning_v1.dynamics import VehicleSimulator, StepDiagnostics, ControlInterface
from .native import lib,ptr

class FastSimulator(VehicleSimulator):
    def __init__(self,spec,**kwargs):
        super().__init__(spec,**kwargs)
        e=spec.energy
        power=[e.auxiliary_power_w+(e.tool_power_w if spec.tool_always_active else 0),
               e.distance_energy_j_per_m,e.speed_cubic_coeff_w_s3_m3,e.turning_energy_j_per_rad] if e.enabled else [0.]*4
        self.parameters=np.array([spec.max_curvature_inv_m,spec.max_straight_speed_m_s,spec.max_angular_velocity_rad_s,
            spec.implied_max_lateral_accel_m_s2,spec.max_linear_accel_m_s2,spec.max_linear_decel_m_s2,
            spec.max_curvature_rate_inv_m2,spec.max_angular_accel_rad_s2,
            (spec.wheelbase_m or 0.) if spec.max_steering_rate_rad_s is not None else 0.,
            spec.max_steering_rate_rad_s or 0.,*power],dtype=np.float64)

    def step(self,cmd,dt,*,normalized_action=False,record_substeps=True):
        if not (math.isfinite(dt) and dt>0):
            raise ValueError("dt must be positive and finite")
        if normalized_action:
            c0,c1=self.map_rl_action(cmd)
        elif self.interface==ControlInterface.CURVATURE_ONLY:
            c1=float(np.asarray(cmd).ravel()[0])
            c0=self.spec.allowed_speed(c1)
        else:
            c0,c1=float(cmd[0]),float(cmd[1])
        if not (math.isfinite(c0) and math.isfinite(c1)):
            raise ValueError("Non-finite command")
        n=max(1,math.ceil(dt/self.dt_phys-1e-12))
        h=dt/n
        st=self.state
        trace=np.empty((n+1,5),dtype=np.float64)
        trace[0]=(st.x,st.y,st.theta,st.v,st.kappa)
        d=np.zeros(13)
        flags=np.zeros(8,dtype=np.uint8)
        lib.integrate(ptr(self.parameters),self.interface!=ControlInterface.DIRECT_ACCEL_RATE,c0,c1,h,n,ptr(trace),ptr(d),ptr(flags))
        st.x,st.y,st.theta,st.v,st.kappa=trace[-1].tolist()
        st.t+=dt
        st.s+=float(d[8])
        st.energy_j+=float(d[9])
        e=self.spec.energy
        if e.enabled and e.battery_capacity_wh and e.battery_capacity_wh>0:
            st.soc=max(0.,st.soc-float(d[9])/(e.battery_capacity_wh*3600.))
        diag=StepDiagnostics(self.interface.value,dt,h,n,(c0,c1),float(d[10])/dt,float(d[11])/dt,float(d[12])/dt,
                             st.v,st.kappa,*map(bool,flags),*map(float,d[:10]),
                             substep_states=list(map(tuple,trace.tolist())) if record_substeps else [])
        return st.copy(),diag
