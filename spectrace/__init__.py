"""spectrace — bake-off harness for serving tricks on multi-step agent traces."""

from __future__ import annotations

__version__ = "0.1.0"

from spectrace.metrics import BakeoffSummary, RunMetrics
from spectrace.replay import run_bakeoff, run_trace
from spectrace.traces import AgentTrace, load_trace, load_traces

__all__ = [
    "AgentTrace",
    "BakeoffSummary",
    "RunMetrics",
    "load_trace",
    "load_traces",
    "run_bakeoff",
    "run_trace",
    "__version__",
]
