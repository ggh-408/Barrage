"""Isolated removal of repeated retained-route scoring within one decision."""
import inspect
import textwrap
import types


def install(guard):
    original = guard.apply
    source = textwrap.dedent(inspect.getsource(original))
    old = '        metrics=self._assess(plane,self.half_size,bullets,velocity,error,paths,lengths)'
    assert source.count(old) == 1, 'Installed planner source differs from reviewed code'
    new = '''        if retained_metrics is None:
            metrics=self._assess(plane,self.half_size,bullets,velocity,error,paths,lengths)
        else:
            # Preserve every candidate row and its index, including duplicates.
            fresh_metrics=self._assess(plane,self.half_size,bullets,velocity,error,
                paths[:prior_index],lengths[:prior_index])
            metrics=np.concatenate((fresh_metrics,retained_metrics),axis=0)'''
    namespace = dict(original.__func__.__globals__)
    namespace['_original_apply'] = original
    source = source.replace('return super().apply(selection,objects,masks,globals_)',
                            'return _original_apply(selection,objects,masks,globals_)')
    exec(compile(source.replace(old, new), '<reuse_retained_assessment>', 'exec'), namespace)
    guard.apply = types.MethodType(namespace['apply'], guard)
    return original
