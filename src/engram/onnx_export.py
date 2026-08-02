"""ONNX export (`engram export-onnx`).

Converts an embedding model running on sentence-transformers (torch) to ONNX
once, so subsequent runtime uses only onnxruntime + tokenizers. The goal is
to remove the torch import (12-24s warm / 50s+ cold) from the startup path.

Conversion uses torch.onnx.export directly. Not using optimum is deliberate:
optimum pins transformers to a narrow version range, and pulling it in can
downgrade the transformers already installed in the environment, breaking
sentence-transformers itself (this actually happened on 2026-07-03, when
transformers was downgraded from 5.12 to 4.57). torch is already a required
dependency via sentence-transformers, so this conversion adds zero new
dependencies.

Safety net: after conversion, the same set of texts is embedded through both
the torch path (RuriEmbedder) and the ONNX path (OnnxRuriEmbedder). If the
minimum cosine similarity falls below PARITY_MIN, the conversion is treated
as a failure and the model directory is discarded. Because the DB pins its
dimensionality and vector distribution to the model (see the dim-mismatch
exception in db.py), silently adopting a skewed ONNX model would silently
break search over every existing memory.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings

# Conversion error between torch fp32 and ONNX fp32 is normally around 1e-6.
# Dropping below 0.999 means the conversion diverged in some way (pooling,
# prefix, or tokenizer mismatch).
PARITY_MIN = 0.999

# Parity-check sample texts. A mix of Japanese sentences with varied length
# and content, chosen to resemble real memories. Lengths are deliberately
# spread out so that any length-dependent branch baked into the traced graph
# is reliably exposed by the check.
PARITY_TEXTS = [
    "Windows では MCP サーバーのスレッドで torch を import すると劣化する",
    "ユーザーの好み: 業務文書は和暦・右詰めヘッダーで作成すること",
    "pip の再インストール失敗で site-packages に残骸が残り import 不能になった",
    "令和8年度 研究データ基盤開発委員会 今後の計画書",
    "engram recall のクエリ",
    "短い文",
    "SQLite の仮想テーブル vec0 は埋め込み次元を作成時に固定する。"
    "次元が変わると既存 DB は開けず、reindex 以前に DB の作り直しが必要になる。"
    "運用上は export-onnx のパリティ検証がこの事故を防ぐ最後の砦になる。",
    "会議は毎週火曜 10 時から。議事録は MeetingRecords フォルダに保存する。",
    # Above ~128 tokens, ModernBERT-based models (e.g. Ruri-v3) switch to
    # sliding-window attention. Whether that code path traces correctly can
    # only be verified with a long input, so one text intentionally crosses
    # that boundary by a wide margin (~400 tokens). The default MiniLM model
    # is a plain BERT and doesn't have this switch, but the long-text check
    # is kept so the parity harness still covers ModernBERT-based configs.
    "長期記憶の設計では、意味(埋め込み)と思い出しやすさ(活性度)を分離する。"
    "埋め込みベクトルは固定し、検索順位は使用履歴に基づく活性度で変調する。"
    "使うほど活性化し、放置するとべき乗則で減衰するが、完全には消えない。"
    "印象的な文脈で符号化された記憶は減衰が遅く、訂正された誤りは間違えた経験ごと"
    "高い活性で刻み直される。これらの力学は ACT-R の宣言的記憶モジュールに由来し、"
    "実装では各記憶のアクセスイベント列から活性度を都度計算する。検索時は関連度・"
    "活性度・重要度の加重和で最終スコアを決め、重複検知にはコサイン類似の閾値を使う。"
    "この閾値は同一話題の短い日本語文で誤併合が起きた実例に基づいて調整された。"
    "多マシン運用では記憶の正本を Markdown として共有し、インデックスはマシンごとに"
    "ローカルへ置く。起動時に件数の乖離を検知して自動的に再インデックスする。",
]


def _resolve_pad_token(model_name: str, target_dir: Path) -> tuple[str, int]:
    """Place tokenizer.json in target_dir and return (pad_token, pad_token_id).

    Fetched directly from the HF Hub cache. AutoTokenizer is deliberately not
    used, since for some models it falls back to the slow sentencepiece path
    and fails (the runtime only needs the tokenizers library + tokenizer.json
    anyway). The pad token is read from special_tokens_map.json.
    """
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok_path = Path(hf_hub_download(model_name, "tokenizer.json"))
    shutil.copy(tok_path, target_dir / "tokenizer.json")

    pad_token = None
    try:
        stm_path = Path(hf_hub_download(model_name, "special_tokens_map.json"))
        with stm_path.open(encoding="utf-8") as f:
            entry = json.load(f).get("pad_token")
        if isinstance(entry, dict):
            pad_token = entry.get("content")
        elif isinstance(entry, str):
            pad_token = entry
    except Exception:
        pass
    if not pad_token:
        raise RuntimeError("Cannot determine the pad token for this model; unsupported")

    pad_id = Tokenizer.from_file(str(target_dir / "tokenizer.json")).token_to_id(
        pad_token
    )
    if pad_id is None:
        raise RuntimeError(f"Pad token {pad_token!r} is not present in the vocabulary")
    return pad_token, int(pad_id)


def _export_transformer(st_model, out_path: Path) -> None:
    """Write the transformer backbone held by the SentenceTransformer out to
    an ONNX file.

    Input is (input_ids, attention_mask); output is last_hidden_state only.
    Pooling and normalization are not included in the ONNX graph (handled by
    embedder.mean_pool_normalize). Both the batch and seq axes are made
    dynamic.
    """
    import torch

    auto_model = st_model[0].auto_model
    auto_model.eval()

    class _LastHidden(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

    wrapper = _LastHidden(auto_model)
    # Sample input that includes padding (so the mask's zero path is traced too)
    ids = torch.ones((2, 16), dtype=torch.int64)
    mask = torch.ones((2, 16), dtype=torch.int64)
    mask[1, 8:] = 0

    try:
        # New exporter (dynamo). More reliable for modern models (e.g. ModernBERT)
        batch = torch.export.Dim("batch")
        seq = torch.export.Dim("seq")
        program = torch.onnx.export(
            wrapper,
            (ids, mask),
            dynamo=True,
            dynamic_shapes={
                "input_ids": {0: batch, 1: seq},
                "attention_mask": {0: batch, 1: seq},
            },
        )
        program.save(str(out_path))
    except Exception as e:
        print(
            f"[export-onnx] dynamo exporter failed, retrying with the legacy exporter: {e}",
            file=sys.stderr,
        )
        torch.onnx.export(
            wrapper,
            (ids, mask),
            str(out_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "last_hidden_state": {0: "batch", 1: "seq"},
            },
            dynamo=False,
        )


def export_onnx(settings: Settings, *, force: bool = False) -> dict:
    """Convert settings.embed_model to ONNX and place it at
    settings.onnx_model_dir.

    Returns a report dict (dim, parity stats, output path, etc). Raises on
    failure.
    """
    target = settings.onnx_model_dir
    if target.is_dir() and not force:
        raise FileExistsError(
            f"Already exists: {target}\nPass --force to overwrite"
        )

    tmp = target.with_name(target.name + ".tmp")
    if tmp.is_dir():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    try:
        from .embedder import OnnxRuriEmbedder, RuriEmbedder

        print(
            f"[1/4] Loading {settings.embed_model} via torch (reference path)...",
            file=sys.stderr,
        )
        ref = RuriEmbedder(
            model_name=settings.embed_model,
            query_prefix=settings.query_prefix,
            doc_prefix=settings.doc_prefix,
        )
        st_model = ref._load()
        ref_docs = ref.embed_docs(PARITY_TEXTS)
        ref_query = ref.embed_query(PARITY_TEXTS[0])
        dim = ref.dim
        max_seq = int(getattr(st_model, "max_seq_length", 8192))

        print("[2/4] Converting to ONNX...", file=sys.stderr)
        _export_transformer(st_model, tmp / "model.onnx")
        pad_token, pad_token_id = _resolve_pad_token(settings.embed_model, tmp)

        meta = {
            "source_model": settings.embed_model,
            "dim": dim,
            "max_seq_length": max_seq,
            "pad_token_id": pad_token_id,
            "pad_token": pad_token,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        with (tmp / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print("[3/4] Verifying parity via the ONNX path...", file=sys.stderr)
        onnx = OnnxRuriEmbedder(
            tmp,
            query_prefix=settings.query_prefix,
            doc_prefix=settings.doc_prefix,
        )
        if onnx.dim != dim:
            raise RuntimeError(f"Dimension mismatch: torch={dim}, onnx={onnx.dim}")
        onnx_docs = onnx.embed_docs(PARITY_TEXTS)
        onnx_query = onnx.embed_query(PARITY_TEXTS[0])

        # Both paths are L2-normalized, so dot product = cosine similarity
        doc_cos = (ref_docs * onnx_docs).sum(axis=1)
        query_cos = float((ref_query * onnx_query).sum())
        min_cos = float(min(doc_cos.min(), query_cos))
        if min_cos < PARITY_MIN:
            raise RuntimeError(
                f"Parity check failed: min cosine = {min_cos:.6f} < {PARITY_MIN}\n"
                "The ONNX path's distribution diverges from the torch path. Aborted "
                "because adopting this model would break search over the existing "
                "index.db."
            )

        meta["parity"] = {
            "n_texts": len(PARITY_TEXTS),
            "min_cosine": min_cos,
            "threshold": PARITY_MIN,
        }
        with (tmp / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print("[4/4] Deploying...", file=sys.stderr)
        if target.is_dir():
            shutil.rmtree(target)
        tmp.replace(target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    return {
        "model": settings.embed_model,
        "dim": dim,
        "max_seq_length": max_seq,
        "min_cosine": min_cos,
        "target": str(target),
        "onnx_size_mb": round(
            (target / "model.onnx").stat().st_size / 1024 / 1024, 1
        ),
    }
