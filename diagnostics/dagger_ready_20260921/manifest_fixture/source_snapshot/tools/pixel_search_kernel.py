"""Compiled equivalent of one search physics step; optional local dependency."""
from pathlib import Path
import importlib
import sys
import numpy as np

try:
    from numba import njit
except ImportError:
    vendor=Path(__file__).resolve().parents[1]/'.runtime/pixel_search'
    if not vendor.is_dir():
        raise RuntimeError('Install requirements-pixel-search.txt in the barrage environment')
    sys.path.insert(0,str(vendor))
    # A failed import or an IDE namespace package can survive in sys.modules.
    # Retry against the complete local distribution rather than that cache.
    for name in tuple(sys.modules):
        if name == 'numba' or name.startswith('numba.'):
            del sys.modules[name]
    importlib.invalidate_caches()
    from numba import njit


@njit(cache=False)
def rounded(x):
    return int(np.floor(np.float32(x+np.float32(.5))))


@njit(cache=False)
def search_step(positions, movement, half_size, bullets, error, table, integral, cull_far=False):
    n=len(positions); bcount=len(bullets)
    result=np.empty_like(positions)
    hits=np.zeros(n,np.bool_)
    mass=np.zeros((n,bcount),np.float64)
    for i in range(n):
        for axis in range(2):
            result[i,axis]=min(max(np.float32(positions[i,axis]+movement[i,axis]),
                                  half_size[axis]),np.float32(820-half_size[axis]))
        p=result[i]
        for j in range(bcount):
            b=bullets[j]
            # Outside the complete lookup support, both overlap terms are zero.
            # Include rounding-cell headroom; keep the legacy path for parity.
            if cull_far and (abs(b[0]-p[0])>34+error[j] or abs(b[1]-p[1])>34+error[j]):
                continue
            cx=rounded(b[0])-rounded(p[0]); cy=rounded(b[1])-rounded(p[1])
            if -32<=cx<=32 and -32<=cy<=32 and table[cy+32,cx+32]>0:
                hits[i]=True
            lo0=rounded(np.float32(b[0]-error[j]))-rounded(np.float32(np.float32(p[0]+np.float32(.5))-np.float32(1e-5)))
            lo1=rounded(np.float32(b[1]-error[j]))-rounded(np.float32(np.float32(p[1]+np.float32(.5))-np.float32(1e-5)))
            hi0=rounded(np.float32(np.float32(b[0]+error[j])-np.float32(1e-5)))-rounded(np.float32(p[0]-np.float32(.5)))
            hi1=rounded(np.float32(np.float32(b[1]+error[j])-np.float32(1e-5)))-rounded(np.float32(p[1]-np.float32(.5)))
            lx=min(max(lo0+32,0),65); ly=min(max(lo1+32,0),65)
            hx=min(max(hi0+33,0),65); hy=min(max(hi1+33,0),65)
            count=integral[hy,hx]-integral[ly,hx]-integral[hy,lx]+integral[ly,lx]
            mass[i,j]=count/((hi0-lo0+1)*(hi1-lo1+1))
    return result,hits,mass


def warm_search(guard):
    search_step(np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),
        guard.half_size,np.zeros((1,2),np.float32),np.ones(1,np.float32),
        guard.table,guard.integral)
