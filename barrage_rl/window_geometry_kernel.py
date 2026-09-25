"""Optional fused float32 geometry for CPU window inference, without fast math."""
import numpy as np

try:
    from tools.pixel_search_kernel import njit
except (ImportError,RuntimeError):
    clearance_kernel=None
else:
    @njit(cache=False,fastmath=False)
    def clearance_kernel(relative,velocity,masks,actions,horizons,speed,radius):
        result=np.empty((len(relative),len(actions),relative.shape[1],len(horizons)),np.float32)
        epsilon=np.float32(1e-6)
        for b in range(len(relative)):
            for a in range(len(actions)):
                ax=np.float32(actions[a,0]*speed)
                ay=np.float32(actions[a,1]*speed)
                for n in range(relative.shape[1]):
                    if not masks[b,n]:
                        for h in range(len(horizons)):result[b,a,n,h]=np.inf
                        continue
                    x,y=relative[b,n,0],relative[b,n,1]
                    vx=np.float32(velocity[b,n,0]-ax)
                    vy=np.float32(velocity[b,n,1]-ay)
                    speed_squared=np.float32(np.float32(vx*vx)+np.float32(vy*vy))
                    projection=np.float32(np.float32(x*vx)+np.float32(y*vy))
                    closest_time=np.float32(-projection/np.maximum(speed_squared,epsilon))
                    closest_time=np.maximum(closest_time,np.float32(0.))
                    for h in range(len(horizons)):
                        t=np.minimum(closest_time,horizons[h])
                        cx=np.float32(x+np.float32(vx*t))
                        cy=np.float32(y+np.float32(vy*t))
                        norm=np.sqrt(np.float32(np.float32(cx*cx)+np.float32(cy*cy)))
                        result[b,a,n,h]=np.float32(norm-radius)
        return result


def warmup():
    if clearance_kernel is not None:
        clearance_kernel(np.zeros((1,1,2),np.float32),np.zeros((1,1,2),np.float32),
                         np.ones((1,1),np.bool_),np.zeros((9,2),np.float32),
                         np.array([.1,.3,.6,1.2],np.float32),np.float32(240),np.float32(11.5))
