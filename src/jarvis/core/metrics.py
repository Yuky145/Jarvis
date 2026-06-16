"""System resource metrics.

Provides a background sampler that records peak RAM usage (system-wide and for
the Ollama process tree) while a block of code runs. CPU-only inference loads
weights into the Ollama server process, so we track the whole machine plus the
``ollama`` process family to attribute memory correctly.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import psutil


@dataclass
class MemorySample:
    """Result of a memory-sampling window."""

    peak_system_used_mb: float = 0.0
    peak_ollama_rss_mb: float = 0.0
    baseline_system_used_mb: float = 0.0
    samples: int = 0
    duration_s: float = 0.0
    delta_system_mb: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.delta_system_mb = max(
            0.0, self.peak_system_used_mb - self.baseline_system_used_mb
        )


def _ollama_rss_mb() -> float:
    """Total resident memory (MB) of all processes whose name contains 'ollama'."""
    total = 0
    for proc in psutil.process_iter(["name"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if "ollama" in name:
                total += proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / (1024 * 1024)


class MemoryMonitor:
    """Context manager that samples memory on a background thread.

    Example
    -------
    >>> with MemoryMonitor(interval=0.25) as mon:
    ...     do_work()
    >>> print(mon.result.peak_ollama_rss_mb)
    """

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_system = 0.0
        self._peak_ollama = 0.0
        self._baseline = 0.0
        self._count = 0
        self._start_t = 0.0
        self.result: MemorySample | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            sys_used = psutil.virtual_memory().used / (1024 * 1024)
            oll = _ollama_rss_mb()
            self._peak_system = max(self._peak_system, sys_used)
            self._peak_ollama = max(self._peak_ollama, oll)
            self._count += 1
            time.sleep(self.interval)

    def __enter__(self) -> "MemoryMonitor":
        self._baseline = psutil.virtual_memory().used / (1024 * 1024)
        self._peak_system = self._baseline
        self._start_t = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.result = MemorySample(
            peak_system_used_mb=round(self._peak_system, 1),
            peak_ollama_rss_mb=round(self._peak_ollama, 1),
            baseline_system_used_mb=round(self._baseline, 1),
            samples=self._count,
            duration_s=round(time.perf_counter() - self._start_t, 3),
        )


def total_ram_gb() -> float:
    """Total physical RAM in GB."""
    return round(psutil.virtual_memory().total / (1024 ** 3), 1)
