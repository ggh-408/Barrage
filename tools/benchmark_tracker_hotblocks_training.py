"""Spawn-safe before/after collection throughput with direct trajectory checks."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.validate_tracker_hotblocks import OUT, variants
from tools.benchmark_tracker_exact import switch

# Windows spawn re-executes this module; each worker installs its own baseline.
if os.environ.get('BARRAGE_TRACKER_BASELINE') == '1':
    switch(variants(), True)


class Memory(ctypes.Structure):
    _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD),
               ('total', ctypes.c_ulonglong), ('available', ctypes.c_ulonglong),
               ('total_page', ctypes.c_ulonglong), ('available_page', ctypes.c_ulonglong),
               ('total_virtual', ctypes.c_ulonglong), ('available_virtual', ctypes.c_ulonglong),
               ('extended', ctypes.c_ulonglong)]


class Resources:
    def __init__(self):
        self.rows = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise ctypes.WinError()
        idle, kernel, user = (ctypes.c_ulonglong() for _ in range(3))
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            raise ctypes.WinError()
        self.rows.append((idle.value, kernel.value+user.value, memory.available, memory.total))

    def run(self):
        while not self.stop.wait(1):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()

    def report(self):
        idle = self.rows[-1][0]-self.rows[0][0]
        total = self.rows[-1][1]-self.rows[0][1]
        return dict(system_cpu_percent=100*(1-idle/max(total,1)),
                    minimum_available_memory_mib=min(r[2] for r in self.rows)/2**20,
                    total_memory_bytes=self.rows[0][3], samples=len(self.rows))


def main():
    from tools import benchmark_tracked_collection as collection
    benchmark = collection.benchmark
    worker_observation = {}
    def cpu_seconds(pid):
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(ctypes.c_ulonglong)]*4
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            creation, exit_time, kernel_time, user_time = (ctypes.c_ulonglong() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time)):
                raise ctypes.WinError(ctypes.get_last_error())
            return (kernel_time.value+user_time.value)/1e7
        finally:
            kernel.CloseHandle(handle)

    class ObservedPipeline(collection.ParallelTrackedDaggerEnv):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._cpu_before = [cpu_seconds(p.pid) for p in self._processes]
            self._observation_start = time.perf_counter()

        def close(self):
            if not self._closed and hasattr(self, '_cpu_before'):
                worker_observation.update(
                    pids=[p.pid for p in self._processes],
                    all_workers_alive=all(p.is_alive() for p in self._processes),
                    cpu_seconds=[cpu_seconds(p.pid)-start for p, start in zip(self._processes, self._cpu_before)],
                    wall_seconds=time.perf_counter()-self._observation_start)
            super().close()

    collection.ParallelTrackedDaggerEnv = ObservedPipeline
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, nargs='+', default=[4, 6, 9])
    parser.add_argument('--decisions', type=int, default=64)
    parser.add_argument('--optimizer', action='store_true')
    args = parser.parse_args()
    items = variants()
    rows = []
    reference = []
    destination = OUT/f'collection_{args.decisions}.json'
    if destination.exists():
        raise FileExistsError(destination)
    try:
        for workers in args.workers:
            for before in (True, False):
                os.environ['BARRAGE_TRACKER_BASELINE'] = '1' if before else '0'
                switch(items, before)
                with Resources() as resources:
                    row = benchmark(workers, 36, args.decisions, 'recovery', 192, True, (),
                                    300, 0, reference_trajectory=reference,
                                    teacher_reaction_seconds=0.10)
                row.update(variant='before' if before else 'after', resources=resources.report(),
                           worker_execution=dict(worker_observation))
                rows.append(row)
                destination.write_text(json.dumps(rows, indent=2))
                print(json.dumps(row), flush=True)
        if args.optimizer:
            import torch
            from tools.benchmark_tracked_policy_runtime import benchmark as optimizer
            optimizer_rows = []
            for before in (True, False):
                switch(items, before)
                torch.manual_seed(925)
                with Resources() as resources:
                    row = optimizer(512, 30, 'full', 300)
                row.update(variant='before' if before else 'after', resources=resources.report())
                optimizer_rows.append(row)
                print(json.dumps(row), flush=True)
            (OUT/'optimizer.json').write_text(json.dumps(optimizer_rows, indent=2))
    finally:
        switch(items, False)
        os.environ.pop('BARRAGE_TRACKER_BASELINE', None)


if __name__ == '__main__':
    main()
