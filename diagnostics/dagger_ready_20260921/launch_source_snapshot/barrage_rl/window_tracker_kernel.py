"""Exact float32 association scan for the window tracker; no fast math."""
import numpy as np
from tools.pixel_search_kernel import njit

@njit(cache=False,fastmath=False)
def association_scan(predictions,detections,gates_squared,occlusion_squared):
    n,m=len(predictions),len(detections)
    distance=np.empty((n,m),np.float32)
    valid=np.empty((n,m),np.bool_)
    nearest_detection=np.zeros(n,np.int64)
    nearest_track=np.zeros(m,np.int64)
    groups=np.zeros(m,np.int64)
    column_min=np.empty(m,np.float32)
    for i in range(n):
        best=np.float32(np.inf)
        for j in range(m):
            dx=np.float32(predictions[i,0]-detections[j,0])
            dy=np.float32(predictions[i,1]-detections[j,1])
            d=np.float32(np.float32(dx*dx)+np.float32(dy*dy))
            distance[i,j]=d
            valid[i,j]=d<=gates_squared[i]
            if j==0 or d<best:
                best=d;nearest_detection[i]=j
            if i==0 or d<column_min[j]:
                column_min[j]=d;nearest_track[j]=i
            if d<=occlusion_squared:groups[j]+=1
    return distance,valid,nearest_detection,nearest_track,groups

_FLOAT32_DTYPE=np.dtype(np.float32)

def stack_vectors(values,dtype=None):
    """Join existing two-float buffers; preserve NumPy fallback for other inputs."""
    try:
        if (dtype is None or np.dtype(dtype)==_FLOAT32_DTYPE) and values and all(v.dtype==_FLOAT32_DTYPE and v.shape==(2,) for v in values):
            return np.frombuffer(b''.join(values),dtype=np.float32).reshape(len(values),2)
    except (AttributeError,TypeError,BufferError,ValueError):
        pass
    return np.asarray(values,dtype=dtype)
