from __future__ import annotations

import ctypes
import gc
import os
import platform
import sys
import time


def _sync_filesystem() -> bool:
    sync = getattr(os, "sync", None)
    if sync is None:
        return False
    sync()
    return True


def _trim_linux_heap() -> bool:
    if platform.system() != "Linux":
        return False

    try:
        libc = ctypes.CDLL("libc.so.6")
        return libc.malloc_trim(0) == 1
    except Exception:
        return False


def _trim_windows_working_set() -> bool:
    if platform.system() != "Windows":
        return False

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        process = kernel32.GetCurrentProcess()
        return bool(kernel32.SetProcessWorkingSetSize(process, -1, -1))
    except Exception:
        return False


def main() -> int:
    _ = gc.collect()
    _ = _sync_filesystem()
    _ = _trim_linux_heap()
    _ = _trim_windows_working_set()

    # Give the OS scheduler a tiny chance to reclaim pages before the next run.
    time.sleep(0.2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
