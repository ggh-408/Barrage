"""Process-local instrumentation for the standalone visible-window diagnostic."""
import time
from array import array
import numpy as np


class WindowProfiler:
    def __init__(self, deep_model=False, initial_capacity=4096):
        self.deep_model = deep_model
        self.model_structure = []
        self.active = False
        self._capacity = max(1, int(initial_capacity))
        self._width = 128
        self._values = array('d', [0.0]) * (self._capacity * self._width * 2)
        self._seen = bytearray(self._capacity * self._width)
        self._totals = array('d', [0.0]) * self._capacity
        self._labels = []
        self._label_indices = {}
        self._count = 0
        self._depth = 0
        self._starts = [0.0] * 128
        self._children = [0.0] * 128
        self.restores = []

    def _begin(self):
        if self._count == self._capacity:
            self._values.extend(array('d', [0.0]) * (self._capacity * self._width * 2))
            self._seen.extend(bytearray(self._capacity * self._width))
            self._totals.extend(array('d', [0.0]) * self._capacity)
            self._capacity *= 2
        self.active = True

    def _finish(self, total):
        self.active = False
        self._totals[self._count] = total
        self._count += 1

    @property
    def rows(self):
        # Materialize containers only after measurement, preserving the JSON schema.
        if self.active:
            raise RuntimeError('Export timing rows after measurement')
        return [dict(total_ms=self._totals[i], stages={
            label: list(self._values[(i*self._width+j)*2:(i*self._width+j)*2+2])
            for j,label in enumerate(self._labels) if self._seen[i*self._width+j]
        }) for i in range(self._count)]

    def wrap(self, obj, name, label):
        original = getattr(obj, name)
        if label not in self._label_indices:
            if len(self._labels) == self._width:
                raise RuntimeError('Profiler stage capacity exceeded')
            self._label_indices[label] = len(self._labels)
            self._labels.append(label)
        column = self._label_indices[label]
        def measured(*args, **kwargs):
            if not self.active:
                return original(*args, **kwargs)
            depth = self._depth
            self._starts[depth] = time.perf_counter()
            self._children[depth] = 0.0
            self._depth = depth + 1
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - self._starts[depth]
                self._depth = depth
                if depth:
                    self._children[depth-1] += elapsed
                slot = self._count * self._width + column
                offset = slot * 2
                self._seen[slot] = 1
                self._values[offset] += elapsed * 1000
                self._values[offset+1] += (elapsed - self._children[depth]) * 1000
        setattr(obj, name, measured)
        self.restores.append((obj, name, original))

    def install(self):
        import barrage_rl.live_screen as live
        from barrage_rl.tracked_policy import TrackedFeatureExtractor
        self.wrap(live, 'snapshot_surface_rgb', 'RGB capture')
        self.wrap(TrackedFeatureExtractor, 'step_detections', 'Tracking and features')
        original = live.LiveVisualController.observe_due_surface
        def observe(controller, surface):
            if not getattr(controller, '_diagnostic_profile_ready', False):
                self.wrap(controller.semanticizer, 'detect', 'RGB detection')
                self.wrap(controller, '_prediction_hints', 'Detection prediction hints')
                self.wrap(controller.agent, 'act_features', 'Agent orchestration')
                self.wrap(controller.agent.model, 'forward_with_geometry', 'Model forward')
                if hasattr(controller.agent.model, 'forward_policy'):
                    self.wrap(controller.agent.model, 'forward_policy', 'Model forward')
                self.wrap(controller.agent._action_selector, 'select', 'Action selection')
                for name in ('prepare_action_geometry', 'action_geometry_features_from_shared'):
                    if hasattr(controller.agent.model, name):
                        self.wrap(controller.agent.model, name, 'Model geometry: ' + name)
                import torch
                model = controller.agent.model
                from dataclasses import asdict
                self.model_config = dict(width=model.width, attention_layers=model.attention_layers,
                                         attention_heads=model.attention_heads, action_count=model.action_count,
                                         spec=asdict(model.spec), safety_horizons=model.safety_horizons,
                                         geometry_statistics=model.geometry_statistics,
                                         continuation_weight=model.continuation_weight)
                modules = model.named_modules() if self.deep_model else model.named_children()
                for name, module in modules:
                    if not name:
                        continue
                    self.model_structure.append(dict(name=name, type=type(module).__name__,
                                                     configuration=module.extra_repr(),
                                                     direct_parameters=sum(p.numel() for p in module.parameters(recurse=False))))
                    if not isinstance(module, (torch.nn.ModuleList, torch.nn.ModuleDict)):
                        self.wrap(module, 'forward', 'Neural module: ' + name)
                if self.deep_model:
                    from barrage_rl import tracked_policy
                    for name in ('build_image_geometry_belief', 'constant_action_clearance_by_object'):
                        self.wrap(tracked_policy, name, 'Shared geometry: ' + name)
                    self.wrap(model._window_action_query_cache, 'get', 'Action query cache')
                guard = controller.agent._receding_pixel_guard
                for name in ('apply', '_geometry', '_assess', 'search'):
                    if hasattr(guard, name):
                        self.wrap(guard, name, 'Pixel planner: ' + name)
                controller._diagnostic_profile_ready = True
            self._begin()
            started = time.perf_counter()
            try:
                return original(controller, surface)
            finally:
                total = (time.perf_counter() - started) * 1000
                self._finish(total)
        live.LiveVisualController.observe_due_surface = observe
        self.restores.append((live.LiveVisualController, 'observe_due_surface', original))

    def close(self):
        for obj, name, original in reversed(self.restores):
            setattr(obj, name, original)

    def report(self):
        rows = self.rows
        names = sorted({name for row in rows for name in row['stages']})
        total = sum(row['total_ms'] for row in rows)
        ranking = []
        for name in names:
            values = np.array([row['stages'].get(name, [0., 0.]) for row in rows])
            exclusive = values[:, 1]
            ranking.append(dict(stage=name, exclusive_total_ms=float(exclusive.sum()),
                                exclusive_mean_per_decision_ms=float(exclusive.mean()),
                                exclusive_p95_ms=float(np.percentile(exclusive, 95)),
                                inclusive_mean_per_decision_ms=float(values[:, 0].mean()),
                                percent_of_decision_time=float(exclusive.sum() / total * 100) if total else 0.))
        ranking.sort(key=lambda row: row['exclusive_total_ms'], reverse=True)
        return dict(decisions=len(rows), total_decision_ms=total,
                    model_config=getattr(self,'model_config',{}), model_structure=self.model_structure,
                    uninstrumented_ms=total-sum(row['exclusive_total_ms'] for row in ranking),
                    ranking=ranking,
                    note='Sorted by exclusive wall time; child time removed from parents. Includes lazy compilation and instrumentation overhead. Means use all measured decisions.')
