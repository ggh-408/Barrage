"""Parallelize independent route assessments while retaining evaluated sources."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
kernel=(HERE/'readiness_kernel.py').read_text()
for name in ('assess_paths','assess_path_intervals','assess_path_exposure'):
    old='@njit(cache=False)\ndef '+name+'('
    assert old in kernel
    kernel=kernel.replace(old,'@njit(cache=False, parallel=True)\ndef '+name+'(')
kernel=kernel.replace('for i in range(len(paths)):', 'for i in prange(len(paths)):')
with (HERE/'readiness_kernel_parallel.py').open('x') as f:f.write(kernel)
source=(HERE/'candidate_controller.py').read_text(encoding='utf-8')
source=source.replace('from readiness_kernel import','from readiness_kernel_parallel import')
with (HERE/'candidate_parallel.py').open('x',encoding='utf-8') as f:f.write(source)
print('Independent assessment loops parallelized; evaluated serial files retained.')
