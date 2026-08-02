"""Regression tests for MCP server startup.

These guard against the "startup timeout" class of incident (which actually
occurred on 2026-07-02 through 03):
- the initialize handshake gets dragged into a heavy import and blocks for tens of seconds
- heavyweight modules such as torch get loaded merely by importing the server module

Both can be verified without a real model (by design, with ENGRAM_PRELOAD=off the
handshake returns without building the engine).
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")

# The handshake normally takes 2-3 seconds. Even allowing for a slow CI runner,
# 15 seconds is plenty; exceeding it means a regression where a heavy import blocks the handshake.
HANDSHAKE_DEADLINE_SECONDS = 15


def _spawn_server(tmp_path: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC_DIR
    env["ENGRAM_PRELOAD"] = "off"
    env["ENGRAM_HOME"] = str(tmp_path)
    return subprocess.Popen(
        [sys.executable, "-c", "from engram.server import main; main()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def test_initialize_handshake_is_fast(tmp_path):
    """initialize should respond within a few seconds (not blocked by a heavy import)."""
    proc = _spawn_server(tmp_path)
    try:
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "startup-test", "version": "0"},
            },
        }
        proc.stdin.write((json.dumps(request) + "\n").encode())
        proc.stdin.flush()

        # Push the read into a thread and wait on a deadline, so that readline hanging
        # on a stuck server doesn't freeze the whole test
        lines: queue.Queue = queue.Queue()

        def _reader():
            for raw in proc.stdout:
                lines.put(raw)

        threading.Thread(target=_reader, daemon=True).start()

        response = None
        import time
        deadline = time.monotonic() + HANDSHAKE_DEADLINE_SECONDS
        while time.monotonic() < deadline:
            try:
                raw = lines.get(timeout=0.5)
            except queue.Empty:
                assert proc.poll() is None, "サーバーが起動中に死にました"
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == 1:
                response = msg
                break

        assert response is not None, (
            f"initialize が {HANDSHAKE_DEADLINE_SECONDS} 秒以内に応答しません"
            "(重い import がハンドシェイクを塞ぐ回帰の疑い)"
        )
        assert "result" in response
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_server_import_stays_light():
    """Importing engram.server alone must not pull in heavyweight modules.

    Not just torch / sentence_transformers -- onnxruntime too must stay lazily
    loaded (on the first embed call). If this regresses, every client's startup slows down.
    """
    code = (
        "import sys\n"
        "import engram.server\n"
        "heavy = [m for m in ('torch', 'sentence_transformers', 'onnxruntime')"
        " if m in sys.modules]\n"
        "print('HEAVY:' + ','.join(heavy))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC_DIR
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    marker = [ln for ln in out.stdout.splitlines() if ln.startswith("HEAVY:")]
    assert marker, out.stdout
    heavy = marker[0].removeprefix("HEAVY:")
    assert heavy == "", f"import engram.server が重量級を読み込みました: {heavy}"
