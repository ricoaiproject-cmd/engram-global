"""Tests for ENGRAM_PRELOAD's auto resolution and the ONNX readiness helper (v0.10.0)."""

from __future__ import annotations

from engram.config import onnx_model_ready
from engram.embedder import OnnxRuriEmbedder
from engram.server import _resolve_preload_mode


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
