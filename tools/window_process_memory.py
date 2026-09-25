"""Read Windows process memory without additional runtime dependencies."""
def process_memory():
    import ctypes
    from ctypes import wintypes
    class Memory(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('page_faults',wintypes.DWORD),
            ('peak_rss',ctypes.c_size_t),('rss',ctypes.c_size_t),
            ('peak_paged',ctypes.c_size_t),('paged',ctypes.c_size_t),
            ('peak_nonpaged',ctypes.c_size_t),('nonpaged',ctypes.c_size_t),
            ('pagefile',ctypes.c_size_t),('peak_pagefile',ctypes.c_size_t),('private',ctypes.c_size_t)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD]
    info=Memory();info.cb=ctypes.sizeof(info)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(info),info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(rss_bytes=info.rss,peak_rss_bytes=info.peak_rss,private_bytes=info.private)
