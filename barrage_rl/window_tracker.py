"""Window-only exact tracker implementations; shared reference remains available."""
from operator import itemgetter
import numpy as np
from .image_oracle import PersistentImageTracker
from .window_tracker_kernel import association_scan,stack_vectors

_timestamp=itemgetter(0)

class WindowImageTracker(PersistentImageTracker):
    _track_vectors=staticmethod(stack_vectors)

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        association_scan(np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),
                         np.ones(1,np.float32),self.occlusion_gate*self.occlusion_gate)

    def _history_groups(self,tracks):
        groups={}
        for track in tracks:
            if track.velocity_known and not self.refit_known_velocity:
                continue
            history=track.history if len(track.history)<=8 else track.history[-8:]
            if len(history)>=4:
                timestamps=tuple(map(_timestamp,history))
                group=groups.get(timestamps)
                if group is None:groups[timestamps]=[(track,history)]
                else:group.append((track,history))
        return groups

    def _association_arrays(self,predictions,detections):
        if (self.occlusion_gate!=7.5 or predictions.dtype!=np.float32 or detections.dtype!=np.float32
                or not np.isfinite(predictions).all() or not np.isfinite(detections).all()):
            return super()._association_arrays(predictions,detections)
        gates=np.asarray([15. if t.velocity_known else 24. for t in self.tracks],np.float32)
        distance,valid,nearest_detection,nearest_track,groups=association_scan(
            predictions,detections,np.square(gates),self.occlusion_gate*self.occlusion_gate)
        return distance,valid,nearest_detection,nearest_track,groups.tolist()

    @staticmethod
    def _assignment_pairs(mutual_pairs,pair_order,valid,detection_count):
        # pair_order comes exclusively from flatnonzero(valid), in stable distance order.
        return list(mutual_pairs)+[divmod(int(i),detection_count) for i in pair_order]
