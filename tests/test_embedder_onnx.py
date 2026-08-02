"""Tests for the ONNX embedding runtime (no real model used).

- mean_pool_normalize: verifies the properties of the pure-function pooling computation,
  identical to the torch path
- make_embedder: verifies the embed_backend selection logic with a tmp_path Settings
  (RuriEmbedder is only checked via isinstance; embed is never called = no model download)
- Verifies that the ENGRAM_EMBED_BACKEND environment variable reaches get_settings
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from engram import config as cfg
from engram.config import Settings
from engram.embedder import (
    OnnxRuriEmbedder,
    RuriEmbedder,
    make_embedder,
    mean_pool_normalize,
)


# ---------------------------------------------------------------------------
# mean_pool_normalize (pure function)
# ---------------------------------------------------------------------------

class TestMeanPoolNormalize:
    def test_masked_mean_hand_computed(self):
        """Only tokens with mask=1 should be averaged (hand-computed example).

        hidden: (1, 3, 2), mask=[1,1,0] -> the 3rd token is ignored,
        so the average is ([1,2] + [3,4]) / 2 = [2,3]. L2-normalized this is [2,3]/sqrt(13).
        """
        hidden = np.array([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
        mask = np.array([[1, 1, 0]])
        out = mean_pool_normalize(hidden, mask)
        expected = np.array([2.0, 3.0]) / np.sqrt(13.0)
        np.testing.assert_allclose(out[0], expected, rtol=1e-6)

    def test_output_is_l2_normalized(self):
        rng = np.random.default_rng(42)
        hidden = rng.normal(size=(4, 7, 16)).astype(np.float32)
        mask = np.ones((4, 7), dtype=np.int64)
        mask[2, 4:] = 0  # some padding present
        out = mean_pool_normalize(hidden, mask)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_all_zero_mask_does_not_crash_or_nan(self):
        """An all-zero attention mask must not produce NaN or a division by zero (defensive check)."""
        hidden = np.ones((2, 3, 4), dtype=np.float32)
        mask = np.zeros((2, 3), dtype=np.int64)
        out = mean_pool_normalize(hidden, mask)
        assert out.shape == (2, 4)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_batch_shape(self):
        """Shape transformation (B, S, D) -> (B, D)."""
        B, S, D = 5, 11, 8
        hidden = np.zeros((B, S, D), dtype=np.float32)
        hidden[:, :, 0] = 1.0
        mask = np.ones((B, S), dtype=np.int64)
        out = mean_pool_normalize(hidden, mask)
        assert out.shape == (B, D)


# ---------------------------------------------------------------------------
# make_embedder selection logic
# ---------------------------------------------------------------------------

def _make_settings(tmp_path, backend: str) -> Settings:
    return Settings(
        data_dir=tmp_path,
        memories_dir=tmp_path / "memories",
        embed_backend=backend,
    )


def _fabricate_onnx_dir(settings: Settings) -> None:
    """Fabricate a directory that looks like it has been export-onnx'd (contents can be
    empty, since make_embedder only checks file existence and meta.json)."""
    d = settings.onnx_model_dir
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"")
    (d / "tokenizer.json").write_bytes(b"")
    (d / "meta.json").write_text(
        json.dumps({
            "dim": 512,
            "max_seq_length": 8192,
            "pad_token_id": 3,
            "pad_token": "<pad>",
        }),
        encoding="utf-8",
    )


class TestMakeEmbedder:
    def test_backend_torch_returns_ruri(self, tmp_path):
        s = _make_settings(tmp_path, "torch")
        emb = make_embedder(s)
        assert isinstance(emb, RuriEmbedder)  # embed is never called (avoids model download)

    def test_auto_without_onnx_falls_back_to_torch(self, tmp_path):
        s = _make_settings(tmp_path, "auto")
        assert not s.onnx_model_dir.is_dir()
        emb = make_embedder(s)
        assert isinstance(emb, RuriEmbedder)

    def test_auto_with_onnx_dir_returns_onnx(self, tmp_path):
        s = _make_settings(tmp_path, "auto")
        _fabricate_onnx_dir(s)
        emb = make_embedder(s)
        assert isinstance(emb, OnnxRuriEmbedder)

    def test_onnx_dim_read_without_loading_session(self, tmp_path):
        """dim is answered immediately from meta.json, and the ONNX session is not loaded
        (model.onnx is an empty file, so a load attempt would always fail)."""
        s = _make_settings(tmp_path, "auto")
        _fabricate_onnx_dir(s)
        emb = make_embedder(s)
        assert emb.dim == 512
        assert emb._session is None  # session remains unloaded

    def test_backend_onnx_forced_missing_dir_raises(self, tmp_path):
        s = _make_settings(tmp_path, "onnx")
        with pytest.raises(FileNotFoundError) as exc_info:
            make_embedder(s)
        assert "export-onnx" in str(exc_info.value)

    def test_backend_onnx_forced_with_dir_returns_onnx(self, tmp_path):
        s = _make_settings(tmp_path, "onnx")
        _fabricate_onnx_dir(s)
        emb = make_embedder(s)
        assert isinstance(emb, OnnxRuriEmbedder)

    def test_invalid_backend_raises_value_error(self, tmp_path):
        s = _make_settings(tmp_path, "tensorflow")
        with pytest.raises(ValueError):
            make_embedder(s)

    def test_partial_onnx_dir_is_not_available(self, tmp_path):
        """An incomplete directory missing meta.json is not treated as ONNX."""
        s = _make_settings(tmp_path, "auto")
        d = s.onnx_model_dir
        d.mkdir(parents=True)
        (d / "model.onnx").write_bytes(b"")
        (d / "tokenizer.json").write_bytes(b"")
        # no meta.json
        emb = make_embedder(s)
        assert isinstance(emb, RuriEmbedder)


# ---------------------------------------------------------------------------
# ENGRAM_EMBED_BACKEND environment variable -> Settings
# ---------------------------------------------------------------------------

class TestEmbedBackendEnv:
    def test_env_override_reaches_settings(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
        monkeypatch.setenv("ENGRAM_EMBED_BACKEND", "torch")
        s = cfg.get_settings()
        assert s.embed_backend == "torch"

    def test_default_is_auto(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
        monkeypatch.delenv("ENGRAM_EMBED_BACKEND", raising=False)
        s = cfg.get_settings()
        assert s.embed_backend == "auto"

    def test_onnx_model_dir_derived_from_model_name(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
        monkeypatch.delenv("ENGRAM_EMBED_BACKEND", raising=False)
        s = cfg.get_settings()
        assert s.onnx_model_dir == (
            tmp_path / "onnx" / s.embed_model.replace("/", "--")
        )
