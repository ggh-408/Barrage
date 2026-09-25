"""Temporarily suspend this evaluation tree while measuring the current window."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def processes():
    command = '[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress'
    return json.loads(subprocess.check_output(['powershell', '-NoProfile', '-Command', command], encoding='utf-8-sig'))


def main():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    nt = ctypes.WinDLL('ntdll')
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    nt.NtSuspendProcess.argtypes = nt.NtResumeProcess.argtypes = [wintypes.HANDLE]
    output = ROOT / 'diagnostics' / ('current_window_' + time.strftime('%Y%m%d_%H%M%S'))
    handles = []
    suspended = []

    def pause(pid):
        handle = kernel.OpenProcess(0x0800, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if nt.NtSuspendProcess(handle) != 0:
            kernel.CloseHandle(handle)
            raise RuntimeError(f'Cannot suspend process {pid}')
        handles.append(handle)
        suspended.append(pid)

    try:
        parent = next((p for p in processes() if p['ProcessId'] == 12068), None)
        if parent:
            assert 'evaluate_combined_latency.py' in (parent['CommandLine'] or '')
            pause(12068)
            rows = processes()
            pending = [12068]
            while pending:
                parent_id = pending.pop()
                for row in rows:
                    if row['ParentProcessId'] == parent_id:
                        pause(row['ProcessId'])
                        pending.append(row['ProcessId'])
        print('Suspended evaluation processes: ' + str(suspended), flush=True)
        print('Window output: ' + str(output), flush=True)
        subprocess.run([sys.executable, str(ROOT / 'tools/diagnose_window_pacing.py'),
                        '--seconds', '120', '--record-window-state', '--output-dir', str(output)],
                       cwd=ROOT, check=True)
    finally:
        failures = []
        for pid, handle in reversed(list(zip(suspended, handles))):
            if nt.NtResumeProcess(handle) != 0:
                failures.append(pid)
            kernel.CloseHandle(handle)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'isolation.json').write_text(json.dumps(dict(suspended_pids=suspended,
            resume_failures=failures, candidate_installed=False), indent=2), encoding='utf-8')
        print('Evaluation resumed; failures: ' + str(failures), flush=True)
        if failures:
            raise RuntimeError('Some evaluation processes need resume recovery')


if __name__ == '__main__':
    main()
