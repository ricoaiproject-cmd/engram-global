"""Embedding layer. The default model is intfloat/multilingual-e5-small
(100+ languages); other models such as Ruri-v3 (Japanese-specialized) remain
available via config. Tests use FakeEmbedder.

Two runtime backends:
- OnnxRuriEmbedder — ONNX Runtime. Import+load is light (1-2s); this is the
  default. Reads the model directory (with meta.json) produced by
  `engram export-onnx`.
- RuriEmbedder — sentence-transformers (torch). Import alone takes 12-24s warm /
  50s+ cold (see the ENGRAM_PRELOAD comment in server.py). Used as a fallback
  when no ONNX model has been exported yet, and as the reference for
  export-onnx's parity check.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed_query(self, text: str) -> np.ndarray: ...
    def embed_docs(self, texts: list[str]) -> np.ndarray: ...


def mean_pool_normalize(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean-pool over the attention mask, weighted, then L2-normalize.

    Matches sentence-transformers' Pooling(pooling_mode_mean_tokens=True) +
    encode(normalize_embeddings=True). The parity test guarantees the ONNX path
    produces the same vector distribution as the torch path (the db's dim is
    fixed and existing vectors depend on it, so do not change this function
    lightly).

    hidden: (batch, seq, dim) final hidden state
    mask:   (batch, seq) attention mask (0/1)
    """
    m = mask.astype(np.float32)[:, :, np.newaxis]
    summed = (hidden.astype(np.float32) * m).sum(axis=1)
    counts = np.clip(m.sum(axis=1), 1e-9, None)
    pooled = summed / counts
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.clip(norms, 1e-12, None)


class RuriEmbedder:
    """Wraps a sentence-transformers model (e.g. the default
    multilingual-e5-small, or cl-nagoya/ruri-v3). Query/doc prefixes are
    configurable — the E5 family uses "query: "/"passage: " (the defaults
    here) and the Ruri family uses its own Japanese prefixes, while models
    such as all-MiniLM use empty prefixes.

    sentence-transformers is heavy, so loading is lazy. The stdio MCP server
    is a long-lived process, so the model is only loaded once.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        query_prefix: str = "query: ",
        doc_prefix: str = "passage: ",
    ) -> None:
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._model = None
        # FastMCP runs synchronous tools on worker threads. This lock prevents
        # double-loading the model if background preloading races with the
        # first tool call.
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                import contextlib
                import os
                import sys

                # Checking connectivity to the HF Hub can hang indefinitely on
                # a flaky network, which would freeze recall (and the whole
                # session) on a stdio MCP server. Load offline immediately if
                # the model is already cached, and only reach the network
                # when it isn't.
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
                from sentence_transformers import SentenceTransformer

                # On a stdio MCP server, stdout is reserved for JSON-RPC.
                # Stray output from the library would corrupt the protocol,
                # so redirect it to stderr while loading.
                with contextlib.redirect_stdout(sys.stderr):
                    try:
                        self._model = SentenceTransformer(
                            self._model_name, local_files_only=True
                        )
                    except Exception:
                        # Download online only on the very first run, before
                        # the cache is populated
                        self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        model = self._load()
        # Compatible with both old and new sentence-transformers versions
        getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension"
        )
        return int(getter())

    def embed_query(self, text: str) -> np.ndarray:
        vec = self._load().encode(
            [self._query_prefix + text], normalize_embeddings=True
        )[0]
        return np.asarray(vec, dtype=np.float32)

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        vecs = self._load().encode(
            [self._doc_prefix + t for t in texts], normalize_embeddings=True
        )
        return np.asarray(vecs, dtype=np.float32)


class OnnxRuriEmbedder:
    """Embeds using the ONNX model produced by `engram export-onnx` (the
    default runtime backend).

    Never imports torch, so it starts up in seconds even on a cold start. The
    model directory must contain model.onnx / tokenizer.json / meta.json;
    dim is read from meta.json, so the DB can be opened without loading the
    model.
    """

    def __init__(
        self,
        model_dir: Path,
        query_prefix: str = "query: ",
        doc_prefix: str = "passage: ",
    ) -> None:
        self._dir = Path(model_dir)
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._session = None
        self._tokenizer = None
        self._input_names: list[str] = []
        self._meta: dict | None = None
        self._lock = threading.Lock()

    @classmethod
    def is_available(cls, model_dir: Path) -> bool:
        # Delegates to config.onnx_model_ready (kept there so it stays a
        # lightweight import)
        from .config import onnx_model_ready

        return onnx_model_ready(model_dir)

    def _load_meta(self) -> dict:
        if self._meta is None:
            with (self._dir / "meta.json").open(encoding="utf-8") as f:
                self._meta = json.load(f)
        return self._meta

    @property
    def dim(self) -> int:
        return int(self._load_meta()["dim"])

    def _load(self):
        if self._session is not None:
            return self._session
        with self._lock:
            if self._session is None:
                import onnxruntime as ort
                from tokenizers import Tokenizer

                meta = self._load_meta()
                tok = Tokenizer.from_file(str(self._dir / "tokenizer.json"))
                tok.enable_truncation(max_length=int(meta["max_seq_length"]))
                tok.enable_padding(
                    pad_id=int(meta["pad_token_id"]),
                    pad_token=str(meta["pad_token"]),
                )

                opts = ort.SessionOptions()
                # stdout is reserved for JSON-RPC on a stdio MCP server, so
                # keep ORT's warnings out of it too (3 = ERROR and above only)
                opts.log_severity_level = 3
                self._session = ort.InferenceSession(
                    str(self._dir / "model.onnx"),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._input_names = [
                    i.name for i in self._session.get_inputs()
                ]
                self._tokenizer = tok
        return self._session

    def _encode(self, texts: list[str]) -> np.ndarray:
        session = self._load()
        encodings = self._tokenizer.encode_batch(texts)
        ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)

        feeds: dict[str, np.ndarray] = {}
        for name in self._input_names:
            if name == "input_ids":
                feeds[name] = ids
            elif name == "attention_mask":
                feeds[name] = mask
            elif name == "token_type_ids":
                feeds[name] = np.zeros_like(ids)
        hidden = session.run(None, feeds)[0]  # last_hidden_state
        vecs = mean_pool_normalize(hidden, mask)
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self._query_prefix + text])[0]

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return self._encode([self._doc_prefix + t for t in texts])


def make_embedder(settings) -> Embedder:
    """Selects the runtime backend according to settings.embed_backend (used
    by build_engine).

    auto  — use the ONNX model if it has already been exported, otherwise
            fall back to torch
    onnx  — force ONNX; error if no model has been exported (avoids an
            implicit 12-24s torch startup)
    torch — force sentence-transformers (also the parity reference for
            export-onnx)
    """
    backend = getattr(settings, "embed_backend", "auto")
    onnx_dir = settings.onnx_model_dir
    if backend not in ("auto", "onnx", "torch"):
        raise ValueError(
            f"Invalid embed_backend: {backend!r} (auto | onnx | torch)"
        )
    if backend in ("auto", "onnx") and OnnxRuriEmbedder.is_available(onnx_dir):
        return OnnxRuriEmbedder(
            onnx_dir,
            query_prefix=settings.query_prefix,
            doc_prefix=settings.doc_prefix,
        )
    if backend == "onnx":
        raise FileNotFoundError(
            f"No ONNX model found: {onnx_dir}\n"
            "Generate one with `engram export-onnx` (no extra dependencies required)"
        )
    return RuriEmbedder(
        model_name=settings.embed_model,
        query_prefix=settings.query_prefix,
        doc_prefix=settings.doc_prefix,
    )


class FakeEmbedder:
    """Deterministic embedding for tests. Feature hashing over character
    n-grams (1..3).

    Texts that share substrings end up with nearby vectors, which lets
    similarity-dependent tests (duplicate detection, nearest-neighbor
    search) run without a real model.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for n in (1, 2, 3):
            for i in range(len(text) - n + 1):
                gram = text[i : i + n]
                h = int.from_bytes(
                    hashlib.md5(gram.encode("utf-8")).digest()[:4], "little"
                )
                vec[h % self.dim] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed(t) for t in texts])
