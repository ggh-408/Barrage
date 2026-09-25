"""Standalone candidate for one-observation reuse of prior-track predictions."""
import numpy as np
import inspect
import textwrap


def source_predictions(tracker, decision_steps=1):
    tracks = tracker.tracks
    if not tracks:
        return np.empty((0, 2), np.float32)
    elapsed = tracker.decision_dt * max(1, int(decision_steps))
    positions = np.asarray([t.position for t in tracks])
    velocities = np.asarray([t.velocity for t in tracks])
    if all(t.position.dtype == np.float32 and t.velocity.dtype == np.float32 for t in tracks):
        known = np.asarray([t.velocity_known for t in tracks])
        displacement = np.zeros_like(positions)
        displacement[known] = velocities[known] * elapsed
        return positions + displacement
    return np.stack([t.position + (t.velocity * elapsed if t.velocity_known else 0.) for t in tracks])


def shared_hints(tracker, rgb_shape, decision_steps=1):
    """Return detector hints plus source coordinates for the immediate tracker update.

    The caller must pass source coordinates through the same observation, before
    changing or reordering tracks. No persistent cache or normalized round trip.
    """
    tracks = tracker.tracks
    positions = source_predictions(tracker, decision_steps)
    if not tracks:
        return (np.empty((0, 2), np.float32), np.empty(0, np.float32)), positions
    uncertainty = np.asarray([t.position_uncertainty for t in tracks], dtype=np.float32)
    known = np.asarray([t.velocity_known for t in tracks], dtype=np.bool_)
    steps = max(1, int(decision_steps))
    known_radius = np.clip(uncertainty + 4.0 + 2.0 * (steps - 1), 6.0, 24.0)
    unknown_radius = np.clip(uncertainty + 8.0 * steps, 24.0, 64.0)
    source_radii = np.where(known, known_radius, unknown_radius)
    height, width = rgb_shape[:2]
    pixel_scale = max(width, height) / max(tracker.source_size, 1.0)
    hints = positions.astype(np.float32) / tracker.source_size, source_radii * pixel_scale
    return hints, positions


def build_shared_update():
    """Compile a source-identical update with explicit per-observation predictions.

    This experimental adapter leaves the live tracker class and its state intact.
    """
    from barrage_rl.image_oracle import PersistentImageTracker
    original = PersistentImageTracker._update_measurements
    source = textwrap.dedent(inspect.getsource(original))
    marker = ('    cached = self.__dict__.pop' if '    cached = self.__dict__.pop' in source
              else '    positions = np.asarray([track.position for track in self.tracks])')
    start = source.index(marker)
    end = source.index('    occlusion_distance_squared =', start)
    source = source[:start] + '    predictions = source_predictions\n' + source[end:]
    source = source.replace('    decision_steps: int = 1,',
                            '    decision_steps: int = 1,\n    source_predictions: np.ndarray,', 1)
    namespace = dict(original.__globals__)
    exec(compile(source, '<shared_prediction_measurements>', 'exec'), namespace)
    measurement = namespace['_update_measurements']
    original = PersistentImageTracker._update_detections
    source = textwrap.dedent(inspect.getsource(original))
    source = source.replace('    decision_steps: int = 1,',
                            '    decision_steps: int = 1,\n    source_predictions: np.ndarray,', 1)
    source = source.replace('self._update_measurements(', '_shared_measurements(self,')
    source = source.replace('detections, current_plane, decision_steps=decision_steps',
        'detections, current_plane, decision_steps=decision_steps, source_predictions=source_predictions')
    namespace = dict(original.__globals__, _shared_measurements=measurement)
    exec(compile(source, '<shared_prediction_detections>', 'exec'), namespace)
    update = namespace['_update_detections']
    def step(extractor, bullets, plane, source_positions, *, decision_steps=1):
        if not extractor.initialized:
            return extractor.reset_detections(bullets, plane)
        update(extractor.tracker, bullets, plane, normalized=True,
               decision_steps=decision_steps, source_predictions=source_positions)
        return extractor._features()
    return step
