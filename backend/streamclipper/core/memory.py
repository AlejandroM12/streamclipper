"""Medición aproximada de RAM (Windows): proceso Python + procesos hijos (FFmpeg) + RAM libre del sistema.

El pico de Python se lee del contador del SO (PeakWorkingSetSize, exacto desde el inicio del proceso).
Para el total (Python + FFmpeg a la vez) se muestrea cada 0.5 s, así que es una cota aproximada.
"""
import ctypes
import os
import threading
from ctypes import wintypes

from streamclipper.core.sysinfo import available_ram_gb

_MB = 1024 * 1024


class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_IS_WIN = os.name == "nt"
if _IS_WIN:
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _k32.GetCurrentProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]

_QUERY = 0x0400 | 0x0010  # PROCESS_QUERY_INFORMATION | PROCESS_VM_READ

_children: set[int] = set()
_children_lock = threading.Lock()


def register_child(pid: int) -> None:
    with _children_lock:
        _children.add(pid)


def unregister_child(pid: int) -> None:
    with _children_lock:
        _children.discard(pid)


def _counters(handle) -> _PMC | None:
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    return pmc if _psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb) else None


def own_memory_mb() -> tuple[float, float] | None:
    """(working set actual, pico histórico) del proceso actual en MB."""
    if not _IS_WIN:
        return None
    pmc = _counters(_k32.GetCurrentProcess())
    return (pmc.WorkingSetSize / _MB, pmc.PeakWorkingSetSize / _MB) if pmc else None


def _pid_working_set_mb(pid: int) -> float | None:
    h = _k32.OpenProcess(_QUERY, False, pid)
    if not h:
        return None
    try:
        pmc = _counters(h)
        return pmc.WorkingSetSize / _MB if pmc else None
    finally:
        _k32.CloseHandle(h)


class ResourceMonitor:
    """Muestrea memoria en un hilo mientras dura un trabajo."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.python_peak_mb = 0.0
        self.child_peak_mb = 0.0
        self.total_peak_mb = 0.0
        self.min_available_gb: float | None = None
        self.samples = 0

    def _sample(self) -> None:
        own = own_memory_mb()
        py = own[0] if own else 0.0
        with _children_lock:
            pids = list(_children)
        kids = [w for w in (_pid_working_set_mb(p) for p in pids) if w]
        child = sum(kids)
        self.python_peak_mb = max(self.python_peak_mb, py)
        self.child_peak_mb = max(self.child_peak_mb, child)
        self.total_peak_mb = max(self.total_peak_mb, py + child)
        avail = available_ram_gb()
        if avail is not None:
            self.min_available_gb = avail if self.min_available_gb is None else min(self.min_available_gb, avail)
        self.samples += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self) -> "ResourceMonitor":
        if _IS_WIN:
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if _IS_WIN:
            self._stop.set()
            self._thread.join(timeout=2)
            self._sample()

    def summary(self) -> dict:
        own = own_memory_mb()
        return {
            "python_peak_mb": round(self.python_peak_mb),
            "python_peak_os_counter_mb": round(own[1]) if own else None,
            "ffmpeg_peak_mb": round(self.child_peak_mb),
            "total_peak_mb_approx": round(self.total_peak_mb),
            "system_min_available_gb": round(self.min_available_gb, 2) if self.min_available_gb is not None else None,
            "samples": self.samples,
            "note": "Muestreo cada 0.5 s: valores aproximados. 'python_peak_os_counter_mb' es el pico exacto "
                    "del proceso desde que arrancó.",
        }
