"""Resource sampling and exclusive atomic report publication."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any


def _resources() -> dict[str, int | float | str]:
    result: dict[str, int | float | str] = {
        "process_cpu_seconds": time.process_time(),
    }
    try:
        import psutil

        memory = psutil.Process().memory_info()
        system = psutil.virtual_memory()
        result.update(
            source="psutil", process_rss_bytes=int(memory.rss),
            system_total_bytes=int(system.total),
            system_available_bytes=int(system.available),
        )
    except ImportError:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", wintypes.DWORD), ("load", wintypes.DWORD),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong),
                    ("available_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended", ctypes.c_ulonglong),
                ]

            class ProcessMemory(ctypes.Structure):
                _fields_ = [
                    ("size", wintypes.DWORD), ("page_faults", wintypes.DWORD),
                    ("peak_working_set", ctypes.c_size_t),
                    ("working_set", ctypes.c_size_t),
                    ("peak_paged_pool", ctypes.c_size_t),
                    ("paged_pool", ctypes.c_size_t),
                    ("peak_nonpaged_pool", ctypes.c_size_t),
                    ("nonpaged_pool", ctypes.c_size_t),
                    ("pagefile", ctypes.c_size_t),
                    ("peak_pagefile", ctypes.c_size_t),
                    ("private_usage", ctypes.c_size_t),
                ]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD
            ]
            system = MemoryStatus()
            system.length = ctypes.sizeof(system)
            memory = ProcessMemory()
            memory.size = ctypes.sizeof(memory)
            if not kernel.GlobalMemoryStatusEx(ctypes.byref(system)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not psapi.GetProcessMemoryInfo(
                kernel.GetCurrentProcess(), ctypes.byref(memory), memory.size
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            result.update(
                source="windows_ctypes", process_rss_bytes=int(memory.working_set),
                process_peak_rss_bytes=int(memory.peak_working_set),
                process_private_bytes=int(memory.private_usage),
                system_total_bytes=int(system.total_physical),
                system_available_bytes=int(system.available_physical),
            )
        else:
            result["source"] = "stdlib_cpu_only"
    return result


def _atomic_publish(temporary: Path, destination: Path) -> None:
    """An exclusive hard link publishes complete bytes without replacing a file."""
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(destination: Path, result: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as temporary:
        path = Path(temporary.name)
        try:
            json.dump(result, temporary, indent=2, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        except BaseException:
            temporary.close()
            path.unlink(missing_ok=True)
            raise
    _atomic_publish(path, destination)
