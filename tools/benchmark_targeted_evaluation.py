"""Paired timing of the existing targeted evaluator; no policy changes."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import argparse
import csv
import ctypes
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import numba
import llvmlite
from barrage_rl import evaluate_tracked_policy as evaluation
from barrage_rl.artifacts import atomic_write_json, prepare_new_output
from tools.evaluate_targeted_dagger import install_evaluation_runtime


def profile_worker(*args):
    import cProfile
    from barrage_rl import parallel_evaluation
    if os.environ.get('BARRAGE_BENCH_TRACKER') == 'window':
        from functools import partial
        from barrage_rl.window_tracker import WindowImageTracker
        parallel_evaluation.TrackedFeatureExtractor = partial(parallel_evaluation.TrackedFeatureExtractor,
                                                              tracker_class=WindowImageTracker)
    profiler = cProfile.Profile() if os.environ.get('BARRAGE_BENCH_PROFILE_DIR') else None
    if profiler is not None:
        profiler.enable()
    try:
        return parallel_evaluation._rollout_worker(*args)
    finally:
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(str(Path(os.environ['BARRAGE_BENCH_PROFILE_DIR']) / f'worker_{os.getpid()}.pstats'))


class Resources:
    def __init__(self):
        self.stop = threading.Event()
        self.rows = []
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        previous = None
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
        while not self.stop.is_set():
            idle, kernel, user = (ctypes.c_ulonglong() for _ in range(3))
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
            total = kernel.value + user.value
            row = {'time': time.time()}
            if previous and total > previous[1]:
                row['system_cpu_percent'] = 100 * (1 - (idle.value - previous[0]) / (total - previous[1]))
            previous = (idle.value, total)
            memory = Memory(); memory.length = ctypes.sizeof(memory)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
            row.update(memory_total_bytes=memory.total, memory_available_bytes=memory.available)
            try:
                value = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                    text=True, creationflags=subprocess.CREATE_NO_WINDOW, timeout=5).strip().split(',')
                row.update(gpu_percent=float(value[0]), gpu_memory_mib=float(value[1]))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.rows.append(row)
            self.stop.wait(3)

    def __enter__(self):
        self.thread.start(); return self

    def __exit__(self, *args):
        self.stop.set(); self.thread.join(timeout=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--configs', default='9:20:9,9:20:1,9:20:2,9:20:4,6:20:1')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--limit', type=float, default=10)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--planner', choices=('reference', 'window'), default='reference')
    parser.add_argument('--tracker', choices=('reference', 'window'), default='reference')
    args = parser.parse_args()
    prepare_new_output(args.output)
    seed_source = ROOT / 'diagnostics/targeted_dagger_fresh_20260924_101738_193311/evaluation/evaluation_episodes.csv'
    with seed_source.open(newline='') as stream:
        seeds = [int(row['seed']) for row in csv.DictReader(stream)][:args.episodes]
    assert len(seeds) == args.episodes
    checkpoint_before = (ROOT / 'best.pt').read_bytes()
    source_paths = [*sorted((ROOT/'barrage_rl').glob('*.py')), *sorted((ROOT/'tools').glob('*.py')),
                    *sorted((ROOT/'diagnostics/planner_readiness_20260921').glob('*.py'))]
    sources_before = {p: p.read_bytes() for p in source_paths}
    torch.set_num_threads(1)
    install_evaluation_runtime(15)
    if args.planner == 'window':
        from barrage_rl import deployment
        from barrage_rl.window_runtime import configure_window_controller
        def configure(agent, kind='receding', *, search_workers=9):
            if kind != 'receding' or agent.model.continuation_head is not None:
                raise ValueError('Window planner benchmark requires the current policy without a continuation head')
            return {**configure_window_controller(agent, search_workers=search_workers),
                    'evaluation_planner_implementation': 'window', 'planning_horizon_seconds': .5}
        deployment.configure_image_controller = configure
    original_load = evaluation.load_tracked_agent
    original_rollout = evaluation.run_parallel_rollout
    records = []
    report = dict(python=platform.python_version(), platform=platform.platform(), logical_processors=os.cpu_count(),
        torch=torch.__version__, numba=numba.__version__, llvmlite=llvmlite.__version__,
        gpu=torch.cuda.get_device_name(), checkpoint='best.pt', episode_seeds=seeds,
        environment={key: os.environ.get(key) for key in ('OMP_WAIT_POLICY', 'KMP_BLOCKTIME', 'NUMBA_THREADING_LAYER')},
        planner_implementation=args.planner,
        tracker_implementation=args.tracker,
        seed_source=str(seed_source.relative_to(ROOT)), evaluation_episode_limit_seconds=args.limit,
        scope='Paired throughput benchmark using the existing evaluator. Short episodes do not establish long-episode reliability.',
        records=records)
    reference_actions = np.load(args.reference/'actions.npy') if args.reference else None
    reference_csv = (args.reference/'evaluation/evaluation_episodes.csv').read_text() if args.reference else None
    configs = [tuple(map(int, value.split(':'))) for value in args.configs.split(',')]
    for repeat in range(args.repeat):
        for workers, batch, search in configs:
            name = f'r{repeat}_w{workers}_b{batch}_s{search}'
            output = args.output/name
            trace = np.full((args.episodes, int(round(args.limit*30))), 255, dtype=np.uint8)
            timings = dict(agent_seconds=0., guard_seconds=0., environment_wait_seconds=0., startup_seconds=0., rollout_seconds=0., calls=0)
            state = {'offset': 0, 'first': None, 'last': None, 'started': None}

            def load(*pos, **kw):
                agent, spec, checkpoint = original_load(*pos, **kw)
                original_act = agent.act_features
                def act(*values, **options):
                    tick = time.perf_counter()
                    if state['first'] is None:
                        state['first'] = tick
                        timings['startup_seconds'] += tick-state['started']
                    elif state['last'] is not None:
                        timings['environment_wait_seconds'] += tick-state['last']
                    result = original_act(*values, **options)
                    end = time.perf_counter()
                    timings['agent_seconds'] += end-tick
                    timings['calls'] += 1
                    state['last'] = end
                    indices = np.asarray(options['episode_indices']) + state['offset']
                    decisions = np.asarray(options['decision_indices'])
                    trace[indices, decisions] = result
                    return result
                agent.act_features = act
                return agent, spec, checkpoint

            def rollout(**kw):
                guard = kw['agent']._receding_pixel_guard
                original_apply = guard.apply
                def apply(*pos, **options):
                    tick = time.perf_counter()
                    try:
                        return original_apply(*pos, **options)
                    finally:
                        timings['guard_seconds'] += time.perf_counter()-tick
                guard.apply = apply
                state.update(first=None, last=None, started=time.perf_counter())
                profiler = None
                if args.profile or args.tracker == 'window':
                    from barrage_rl import parallel_evaluation
                    original_worker = parallel_evaluation._rollout_worker
                    parallel_evaluation._rollout_worker = profile_worker
                    os.environ['BARRAGE_BENCH_TRACKER'] = args.tracker
                if args.profile:
                    import cProfile
                    os.environ['BARRAGE_BENCH_PROFILE_DIR'] = str(output.resolve())
                    profiler = cProfile.Profile()
                    profiler.enable()
                try:
                    return original_rollout(**kw)
                finally:
                    if profiler is not None:
                        profiler.disable()
                        profiler.dump_stats(str(output / f'profile_{state["offset"]}.pstats'))
                    if args.profile or args.tracker == 'window':
                        parallel_evaluation._rollout_worker = original_worker
                    timings['rollout_seconds'] += time.perf_counter()-state['started']
                    state['offset'] += kw['episodes']
                    guard.apply = original_apply

            evaluation.load_tracked_agent = load
            evaluation.run_parallel_rollout = rollout
            print(f'START {name} episodes={args.episodes} limit={args.limit}', flush=True)
            started = time.perf_counter()
            with Resources() as resources:
                result = evaluation.evaluate_tracked_checkpoint(str(ROOT/'best.pt'), episodes=args.episodes,
                    episode_seeds=seeds, seed=20260924, workers=workers, evaluation_batch_size=batch,
                    output_dir=str(output/'evaluation'), device_name='cuda', episode_limit_seconds=args.limit,
                    bullet_count=300, targeted_bullet_probability=.10, rendered_rgb=True,
                    causal_action_delay_steps=0, analytic_shield=False, pixel_guard='receding',
                    search_workers=search, smoke_test=args.limit!=120, supplemental_test=args.limit==120)
            elapsed = time.perf_counter()-started
            np.save(output/'actions.npy', trace, allow_pickle=False)
            episode_csv = (output/'evaluation/evaluation_episodes.csv').read_text()
            if reference_actions is None:
                reference_actions, reference_csv = trace.copy(), episode_csv
            mismatches = int(np.count_nonzero(trace != reference_actions))
            decisions = int(np.count_nonzero(trace != 255))
            row = dict(name=name, workers=workers, batch_size=batch, search_workers=search, elapsed_seconds=elapsed,
                decisions=decisions, decisions_per_second=decisions/elapsed,
                rollout_decisions_per_second=decisions/timings['rollout_seconds'],
                steady_decisions_per_second=decisions/max(.001,timings['agent_seconds']+timings['environment_wait_seconds']),
                action_mismatches=mismatches, episode_csv_exact=episode_csv==reference_csv,
                success_at_limit=result['success_at_limit'], timings=timings, resource_samples=resources.rows)
            records.append(row)
            report['checkpoint_contents_unchanged'] = (ROOT/'best.pt').read_bytes() == checkpoint_before
            report['source_contents_unchanged'] = all(p.read_bytes()==v for p,v in sources_before.items())
            atomic_write_json(args.output/'report.json', report)
            print(json.dumps({k:v for k,v in row.items() if k!='resource_samples'}), flush=True)
            assert report['checkpoint_contents_unchanged'] and report['source_contents_unchanged']
    evaluation.load_tracked_agent = original_load
    evaluation.run_parallel_rollout = original_rollout


if __name__ == '__main__':
    main()
