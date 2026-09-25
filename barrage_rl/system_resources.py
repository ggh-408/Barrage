"""Read Windows process CPU time and working set without extra dependencies."""
import ctypes
from ctypes import wintypes


class _MemoryStatus(ctypes.Structure):
    _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
        (name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total',
            'page_available', 'virtual_total', 'virtual_available', 'extended')]


class _Counters(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in ('peak_rss', 'rss', 'paged_peak',
            'paged', 'nonpaged_peak', 'nonpaged', 'pagefile', 'pagefile_peak')]


def memory_status():
    data = _MemoryStatus(); data.length = ctypes.sizeof(data)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(data)):
        raise ctypes.WinError()
    return dict(total_bytes=data.total, available_bytes=data.available)


def process_resources(pid):
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),)*4
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD)
    handle = kernel.OpenProcess(0x1400, False, int(pid))
    if not handle: raise ctypes.WinError()
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(x) for x in times)):
            raise ctypes.WinError()
        data = _Counters(); data.size = ctypes.sizeof(data)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(data), data.size):
            raise ctypes.WinError()
        cpu = sum((x.dwHighDateTime << 32) + x.dwLowDateTime for x in times[2:]) / 1e7
        return dict(cpu_seconds=cpu, rss_bytes=data.rss, peak_rss_bytes=data.peak_rss)
    finally:
        kernel.CloseHandle(handle)
