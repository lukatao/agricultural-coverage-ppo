"""Build the host-native C ABI library once, including under forkserver workers."""
import ctypes as C
import fcntl
import hashlib
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np

def load():
    here = Path(__file__).parent
    sources = [here/"dynamics.cpp", here/"grid.cpp"]
    compiler = os.environ.get("CXX", "g++")
    flags = ["-std=c++17", "-O3", "-fPIC", "-shared", "-fno-fast-math", "-ffp-contract=off"]
    identity = [subprocess.check_output([compiler, "--version"]), subprocess.check_output([compiler, "-dumpmachine"]),
                repr(flags).encode(), platform.machine().encode(), *(f.read_bytes() for f in sources)]
    key = hashlib.sha256(b"\0".join(identity)).hexdigest()
    directory = Path(os.environ.get("PPO_FAST_CACHE", Path.home()/".cache"/"thesis-ppo-fast"))
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    so = directory/(key+".so")
    with (directory/(key+".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not so.exists():
            with tempfile.TemporaryDirectory(dir=directory) as stage:
                output = Path(stage)/"kernel.so"
                subprocess.run([compiler,*flags,*sources,"-o",output],check=True)
                os.replace(output,so)
    lib = C.CDLL(str(so))
    ptr,integer,real = C.c_void_p,C.c_int,C.c_double
    def bind(name,args,result=None):
        f=getattr(lib,name)
        f.argtypes=args
        f.restype=result
    bind("integrate",[ptr,integer,real,real,real,integer,ptr,ptr,ptr])
    bind("sample",[ptr,integer,integer,ptr,ptr,ptr,ptr,ptr,integer,integer,real,real,real,real,integer,ptr])
    bind("treatment",[ptr,integer,integer,ptr,ptr,ptr,ptr,ptr,ptr,ptr,integer,integer,ptr,ptr,real,ptr])
    bind("rays",[real,real,ptr,ptr,integer,ptr,integer,real,ptr])
    bind("clearance",[ptr,integer,ptr,integer],real)
    return lib

lib=load()

def ptr(array):
    return array.ctypes.data

def matrix(x,width):
    a=np.ascontiguousarray(x,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=width:
        raise ValueError(f"Expected N by {width} geometry, received {a.shape}")
    return a
