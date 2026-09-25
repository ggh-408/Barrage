"""Run focused dependency smoke checks with local runtime access disabled."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')


def block_local_runtime(event, args):
    if event in ('open', 'os.listdir', 'os.scandir', 'ctypes.dlopen') and args:
        if isinstance(args[0], (str, bytes)):
            path = os.fsdecode(args[0]).replace('\\', '/').lower()
            if '/.runtime/' in path or path.endswith('/.runtime') or path.startswith('.runtime'):
                raise RuntimeError('Local runtime access blocked: ' + path)


sys.addaudithook(block_local_runtime)


def main():
    import pytest
    names = (
        'pixel_guard_candidate', 'pixel_guard_refined', 'pixel_guard_receding',
        'pixel_guard_continuation', 'parallel_rgb', 'rgb_sprite_kernel',
        'window_inference', 'window_shared_entry', 'window_geometry',
        'window_tracker_algorithms', 'tracked_collection', 'commit_safe_ranking',
    )
    result = pytest.main(['-q', '-p', 'no:cacheprovider'] + [
        str(ROOT / 'tests' / ('test_' + name + '.py')) for name in names
    ])
    import numba
    import llvmlite
    from llvmlite import binding
    print('Numba:', numba.__version__, numba.__file__)
    print('llvmlite:', llvmlite.__version__, binding.ffi.lib._name)
    for module in list(sys.modules.values()):
        assert '.runtime' not in str(getattr(module, '__file__', ''))
    print('PASS: no loaded module uses .runtime')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
