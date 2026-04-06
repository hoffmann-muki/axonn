"""
CUDA event-based timers for collective and pruning operations.

Environment variable (read once at import time, zero overhead when disabled):
    AXONN_TIME_OPS=1   -- enable timing (default: 0)

Design
------
Each timer is either a ``_CudaOpTimer`` instance or ``None``. Call sites
receive the timer as an optional argument and guard all accesses with
``if timer is not None`` so disabled timing has negligible overhead.
"""

import os
from typing import Dict, Optional

import torch

_ENABLED: bool = os.environ.get("AXONN_TIME_OPS", "0") == "1"
TIMERS: Dict[str, "_CudaOpTimer"] = {}


class _CudaOpTimer:
    """Accumulates CUDA event-pairs for later elapsed-time reads."""

    __slots__ = ("_pending", "_current", "_cur_start")

    def __init__(self):
        self._pending = []
        self._current = []
        self._cur_start = None

    def start(self, stream=None):
        e = torch.cuda.Event(enable_timing=True)
        e.record(stream)
        self._cur_start = e

    def stop(self, stream=None):
        e = torch.cuda.Event(enable_timing=True)
        e.record(stream)
        self._current.append((self._cur_start, e))
        self._cur_start = None

    def flush_and_get_ms(self) -> float:
        total = 0.0
        for s, e in self._pending + self._current:
            try:
                total += s.elapsed_time(e)
            except RuntimeError:
                pass
        self._pending = []
        self._current = []
        return total

    def reset(self):
        self._pending = []
        self._current = []
        self._cur_start = None


def _make_timer(name: str) -> Optional[_CudaOpTimer]:
    if not _ENABLED:
        return None
    t = _CudaOpTimer()
    TIMERS[name] = t
    return t


def flush_all_and_get_ms() -> Dict[str, float]:
    return {name: t.flush_and_get_ms() for name, t in TIMERS.items()}


# Named timers
allreduce_timer: Optional[_CudaOpTimer] = _make_timer("allreduce_timer")
dp_prune_timer: Optional[_CudaOpTimer] = _make_timer("dp_prune_timer")
