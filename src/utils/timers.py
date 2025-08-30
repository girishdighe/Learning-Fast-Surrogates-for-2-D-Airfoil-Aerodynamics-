# src/utils/timers.py
"""
Simple timing/profiling helpers.
"""

from __future__ import annotations
import time
from contextlib import contextmanager
from dataclasses import dataclass

@dataclass
class TickTock:
    start: float | None = None
    elapsed: float = 0.0

    def tic(self) -> None:
        self.start = time.perf_counter()

    def toc(self) -> float:
        if self.start is None:
            return 0.0
        self.elapsed = time.perf_counter() - self.start
        self.start = None
        return self.elapsed

@contextmanager
def timer(name: str = "block"):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[TIMER] {name}: {dt:.3f}s")
