"""Observability (perf log).

Records the duration of MCP tool calls and startup preload to
data_dir/perf/perf_log.jsonl, one JSON object per line. Always-on
instrumentation (toggled via settings.perf_log) for diagnosing "something
feels slow" with data instead of gut feeling.

Log format (fixed contract - do not change):
    {"ts": epoch float, "kind": "tool" | "preload", "name": str, "ms": float, "ok": bool}

Rotation and write-failure behavior are kept consistent with surface.py's
_append_log (the spontaneous-recall log).
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import Settings

_LOG_ROTATE_BYTES = 5 * 1024 * 1024


def append_perf(settings: Settings, entry: dict) -> None:
    """Append one line to the perf log (creates the directory and handles rotation).

    Log write failures (OSError) are swallowed so they never affect the availability of the memory store.
    """
    d = settings.data_dir / "perf"
    try:
        d.mkdir(parents=True, exist_ok=True)
        log = d / "perf_log.jsonl"
        if log.is_file() and log.stat().st_size > _LOG_ROTATE_BYTES:
            log.replace(log.with_suffix(".jsonl.old"))
        # never let stray malformed surrogates etc. in the input break the log write
        with log.open("a", encoding="utf-8", errors="replace") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


@contextmanager
def timed(settings: Settings, kind: str, name: str) -> Iterator[None]:
    """Context manager that times a block and records the duration to the perf log.

    When settings.perf_log is False, the timer isn't even started - it's a
    no-op (the only overhead is a single bool check). If an exception occurs
    inside the block, it's recorded with ok=False and then re-raised
    (behavior at the call site is unchanged).
    """
    if not settings.perf_log:
        yield
        return

    start = time.perf_counter()
    ok = True
    try:
        yield
    except BaseException:
        ok = False
        raise
    finally:
        ms = (time.perf_counter() - start) * 1000.0
        append_perf(
            settings,
            {"ts": time.time(), "kind": kind, "name": name, "ms": ms, "ok": ok},
        )
