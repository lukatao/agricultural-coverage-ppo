"""Opt-in accelerators for the sealed V1.0 environment. Call in each worker."""
import hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
from ppo_coverage_planning_v1 import env,field_world

_REFERENCE=(env.VehicleSimulator,env.IncrementalCoverageMap,field_world._min_dist_segments_to_edges,field_world._raycast_against_edges)
@lru_cache(maxsize=1)
def check_reference():
    pins={
        "env":"977f7a5c213127442d7581501726230eb7b8a5eb1b6367ce38f0b4ac2ee90525",
        "dynamics":"141ed8e5bbe4f7bb424155e3e9ec8e3f6446da3e8b6f616f97cceb05ccde72ad",
        "coverage_map":"617c9e861d179c22ca8a15f39ef70f5eb454503812b89f9c1a3129e4ba157129",
        "field_world":"d3f7454c589deeb2889ab55e430ab0356b02a12bec060ec00e20fed9755510b2",
        "robot_specs":"0df55cd5cd04aaa5296724ca7542a4d8343f5c8b815cca3f090694a141c091b2",
    }
    for name,digest in pins.items():
        path=Path(env.__file__).with_name(name+".py")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError("Unsupported reference module: "+name)

def clearance(path,edges):
    from .native import lib,ptr,matrix
    if not len(path) or not len(edges):return float("inf")
    path,edges=matrix(path,2),matrix(edges,4)
    return lib.clearance(ptr(path),len(path),ptr(edges),len(edges))

def raycast(x,y,rdx,rdy,edges,max_range_m):
    from .native import lib,ptr,matrix
    if not len(edges):return np.full(len(rdx),max_range_m,np.float32)
    dx,dy=np.ascontiguousarray(rdx,dtype=np.float64),np.ascontiguousarray(rdy,dtype=np.float64)
    if dx.ndim!=1 or dx.shape!=dy.shape:raise ValueError("Expected matching ray directions")
    edges=matrix(edges,4)
    out=np.empty(len(dx),np.float32)
    lib.rays(x,y,ptr(dx),ptr(dy),len(dx),ptr(edges),len(edges),max_range_m,ptr(out))
    return out

def enable(backend="fast"):
    if backend not in ("reference","fast","dynamics","maps","geometry"):
        raise ValueError("Unknown backend: "+backend)
    if backend!="reference":
        check_reference()
        from .simulator import FastSimulator
        from .maps import FastMap
    env.VehicleSimulator=FastSimulator if backend in ("fast","dynamics") else _REFERENCE[0]
    env.IncrementalCoverageMap=FastMap if backend in ("fast","maps") else _REFERENCE[1]
    field_world._min_dist_segments_to_edges=clearance if backend in ("fast","geometry") else _REFERENCE[2]
    field_world._raycast_against_edges=raycast if backend in ("fast","geometry") else _REFERENCE[3]
