"""Tests for ENGRAM_PRELOAD's auto resolution and the ONNX readiness helper (v0.10.0).

As of v0.12.0, tests for idle unload (ENGRAM_IDLE_UNLOAD_SEC /
_maybe_idle_unload) live here too (both verify the resident process's
model-loading strategy).
"""

from __future__ import annotations

import time

import engram.server as server
from engram.config import onnx_model_ready
from engram.embedder import OnnxRuriEmbedder
from engram.server import (
    _maybe_idle_unload,
    _resolve_idle_unload_sec,
    _resolve_preload_mode,
)


# --- _resolve_preload_mode (pure function) ---

def test_auto_resolves_by_onnx_availability():
    """auto (the default) resolves to background if ONNX has been generated, blocking otherwise."""
    assert _resolve_preload_mode("auto", onnx_ready=True) == "background"
    assert _resolve_preload_mode("auto", onnx_ready=False) == "blocking"
    # Unset (None) resolves the same as auto
    assert _resolve_preload_mode(None, onnx_ready=True) == "background"
    assert _resolve_preload_mode(None, onnx_ready=False) == "blocking"


def test_explicit_modes_are_respected():
    """An explicit value passes through unchanged, regardless of ONNX availability."""
    for ready in (True, False):
        assert _resolve_preload_mode("blocking", onnx_ready=ready) == "blocking"
        assert _resolve_preload_mode("background", onnx_ready=ready) == "background"
        assert _resolve_preload_mode("off", onnx_ready=ready) == "off"


def test_unknown_values_fall_back_to_auto():
    """Unknown values (e.g. typos) resolve the same as auto (changed from the old
    implementation's "anything but off means background")."""
    assert _resolve_preload_mode("bckground", onnx_ready=False) == "blocking"
    assert _resolve_preload_mode("bckground", onnx_ready=True) == "background"
    assert _resolve_preload_mode("  Blocking  ", onnx_ready=True) == "blocking"  # whitespace and case are normalized


# --- onnx_model_ready (the lightweight check on the config side) ---

def _make_model_dir(tmp_path, *files):
    d = tmp_path / "onnx" / "model"
    d.mkdir(parents=True)
    for name in files:
        (d / name).write_text("dummy", encoding="utf-8")
    return d


def test_onnx_model_ready_requires_all_three_files(tmp_path):
    complete = _make_model_dir(tmp_path, "model.onnx", "tokenizer.json", "meta.json")
    assert onnx_model_ready(complete) is True

    missing = _make_model_dir(tmp_path / "x", "model.onnx", "tokenizer.json")
    assert onnx_model_ready(missing) is False

    assert onnx_model_ready(tmp_path / "nonexistent") is False


def test_embedder_is_available_agrees_with_config_helper(tmp_path):
    """embedder.is_available delegates to config.onnx_model_ready (prevents the two from drifting apart)."""
    complete = _make_model_dir(tmp_path, "model.onnx", "tokenizer.json", "meta.json")
    partial = _make_model_dir(tmp_path / "y", "meta.json")
    for d in (complete, partial, tmp_path / "nope"):
        assert OnnxRuriEmbedder.is_available(d) == onnx_model_ready(d)


# --- _resolve_idle_unload_sec (pure function) ---

def test_idle_unload_default_is_600():
    assert _resolve_idle_unload_sec(None) == 600.0
    assert _resolve_idle_unload_sec("") == 600.0
    assert _resolve_idle_unload_sec("   ") == 600.0


def test_idle_unload_explicit_values():
    assert _resolve_idle_unload_sec("120") == 120.0
    assert _resolve_idle_unload_sec("120.5") == 120.5
    assert _resolve_idle_unload_sec(" 60 ") == 60.0


def test_idle_unload_zero_or_negative_disables():
    assert _resolve_idle_unload_sec("0") == 0.0
    assert _resolve_idle_unload_sec("-5") == 0.0  # negatives clamp to 0 (disabled)


def test_idle_unload_garbage_falls_back_to_default():
    assert _resolve_idle_unload_sec("abc") == 600.0
    assert _resolve_idle_unload_sec("10min") == 600.0


# --- _maybe_idle_unload (one round of the idle check) ---

class _FakeUnloadableEmbedder:
    """Reproduces just the behavior of an embedder with unload() (= ONNX path)."""

    def __init__(self, loaded: bool = True) -> None:
        self.loaded = loaded
        self.unload_calls = 0

    def unload(self) -> bool:
        self.unload_calls += 1
        was = self.loaded
        self.loaded = False
        return was


class _FakeSettings:
    perf_log = False


class _FakeEngine:
    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.settings = _FakeSettings()


def _set_idle_state(monkeypatch, engine, idle_elapsed: float) -> None:
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(
        server, "_last_activity", time.monotonic() - idle_elapsed
    )


def test_no_engine_no_unload(monkeypatch):
    """No engine built -> do nothing (never trigger construction just to unload)."""
    _set_idle_state(monkeypatch, None, idle_elapsed=9999)
    assert _maybe_idle_unload(600) is False


def test_unloads_after_idle_threshold(monkeypatch):
    emb = _FakeUnloadableEmbedder(loaded=True)
    _set_idle_state(monkeypatch, _FakeEngine(emb), idle_elapsed=601)
    assert _maybe_idle_unload(600) is True
    assert emb.unload_calls == 1
    assert not emb.loaded


def test_does_not_unload_before_threshold(monkeypatch):
    emb = _FakeUnloadableEmbedder(loaded=True)
    _set_idle_state(monkeypatch, _FakeEngine(emb), idle_elapsed=10)
    assert _maybe_idle_unload(600) is False
    assert emb.unload_calls == 0
    assert emb.loaded


def test_not_loaded_embedder_is_skipped(monkeypatch):
    emb = _FakeUnloadableEmbedder(loaded=False)
    _set_idle_state(monkeypatch, _FakeEngine(emb), idle_elapsed=9999)
    assert _maybe_idle_unload(600) is False
    assert emb.unload_calls == 0


def test_embedder_without_unload_is_skipped(monkeypatch):
    """The torch path (no unload) is excluded — never expose it to the reload pathology."""

    class _TorchLikeEmbedder:
        loaded = True  # even when loaded, no unload method means hands off

    _set_idle_state(monkeypatch, _FakeEngine(_TorchLikeEmbedder()), 9999)
    assert _maybe_idle_unload(600) is False


def test_touch_resets_idle_timer(monkeypatch):
    """A tool call (_touch) resets the idle measurement."""
    emb = _FakeUnloadableEmbedder(loaded=True)
    _set_idle_state(monkeypatch, _FakeEngine(emb), idle_elapsed=9999)
    server._touch()  # a tool was just called
    assert _maybe_idle_unload(600) is False
    assert emb.loaded
