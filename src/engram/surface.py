"""Fast-path search for proactive recall (2. proactive memory).

Surfaces memories relevant to what the user just said and injects them into
context, without the agent having to ask. Called on every prompt from the
UserPromptSubmit hook, so it does not load the embedding model (torch), numpy,
or sqlite-vec:

- Relevance: IDF-weighted character-bigram containment (how much of the
  utterance's vocabulary is contained in the memory). Bigrams work reasonably
  well for Japanese even without morphological analysis.
- Activation: dynamics.activation_norm (ACT-R, math only)
- Final score: the same weighted sum used by recall (dynamics.final_score)

Read-only against the DB (does not even record recall_hit events). Deciding
whether a surfaced memory was actually useful, and calling reinforce, is the
agent's responsibility.

Modes (surface_mode in config.toml):
- off:    do nothing
- shadow: don't inject anything, just log "this is what would have been
          injected" (for tuning)
- active: actually inject memories that clear the threshold into context

Both shadow and active log every candidate (surface/surface_log.jsonl), so the
threshold's validity can be reviewed later.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

from . import dynamics
from .config import Settings

_SESSION_ID_SAFE = re.compile(r"[^A-Za-z0-9_-]")
_STATE_MAX_AGE_SECONDS = 7 * 86400
_LOG_ROTATE_BYTES = 5 * 1024 * 1024


# ---------------------------------------------------------------------------
# Lexical relevance (IDF-weighted bigram containment)
# ---------------------------------------------------------------------------

def bigrams(text: str) -> set[str]:
    """Return the set of character bigrams after NFKC normalization and
    casefolding.

    Built per whitespace-separated segment (bigrams never cross a word
    boundary). A single-character segment is treated as a 1-gram as-is.
    """
    norm = unicodedata.normalize("NFKC", text).casefold()
    grams: set[str] = set()
    for seg in norm.split():
        if len(seg) == 1:
            grams.add(seg)
            continue
        for i in range(len(seg) - 1):
            grams.add(seg[i : i + 2])
    return grams


def lexical_scores(prompt: str, docs: list[str]) -> list[float]:
    """For each document, return the IDF-weighted bigram containment (0..1)
    against the prompt's bigrams.

    score = sum_{g in P∩D} idf(g) / sum_{g in P} idf(g)
    Common bigrams (function-word fragments etc.) appear in many documents and
    so get a low idf, letting content-word matches dominate.
    """
    p_grams = bigrams(prompt)
    doc_grams = [bigrams(d) for d in docs]
    n = len(docs)
    if not p_grams or n == 0:
        return [0.0] * n

    df: dict[str, int] = {}
    for grams in doc_grams:
        for g in grams & p_grams:  # only grams that appear in the prompt need counting
            df[g] = df.get(g, 0) + 1

    idf = {g: math.log(1.0 + n / (1.0 + df.get(g, 0))) for g in p_grams}
    denom = sum(idf.values())
    if denom <= 0:
        return [0.0] * n

    scores: list[float] = []
    for grams in doc_grams:
        num = sum(idf[g] for g in p_grams & grams)
        scores.append(num / denom)
    return scores


# ---------------------------------------------------------------------------
# DB reads (lightweight path: stdlib sqlite3 only)
# ---------------------------------------------------------------------------

def _fetch_candidates(db_path: Path, rooms: list[str] | None) -> list[dict]:
    """Return every tier=hot memory (with body text). Empty if the DB doesn't
    exist.

    Never touches vec_memories (sqlite-vec), so no extension loading is
    needed. FTS5 is part of stock SQLite, so reading fts_memories is fine.
    """
    if not Path(db_path).is_file():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=2000")
        # Exclude episodes. Proactive recall is meant to be a lightweight
        # equivalent of fast, and:
        # (1) episodes are raw experience logs and tend to be noisy; following
        #     them associatively is what deep recall is for (recall's fast
        #     mode also excludes episodes)
        # (2) since auto-encoding (1. automatic encoding) saves user
        #     utterances as summarized episodes, not excluding them creates a
        #     feedback loop where a similar utterance surfaces an episode
        #     containing the user's own words as an exact match — i.e. an
        #     echo effect (confirmed in real logs)
        # Only surface memories that have been distilled into knowledge,
        # preferences, or project memories
        sql = (
            "SELECT m.id, m.type, m.importance, m.room, m.created_at, "
            "       f.content AS content "
            "FROM memories m JOIN fts_memories f ON f.memory_id = m.id "
            "WHERE m.tier = 'hot' AND m.type != 'episode'"
        )
        params: list = []
        if rooms:
            ph = ",".join("?" * len(rooms))
            sql += f" AND m.room IN ({ph})"
            params.extend(rooms)
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        if not rows:
            return []
        ids = [r["id"] for r in rows]
        ph = ",".join("?" * len(ids))
        events: dict[str, list[tuple[float, float]]] = {i: [] for i in ids}
        for row in conn.execute(
            f"SELECT memory_id, ts, weight FROM access_events "
            f"WHERE memory_id IN ({ph}) ORDER BY ts",
            ids,
        ):
            events[row["memory_id"]].append((row["ts"], row["weight"]))
        for r in rows:
            r["events"] = events[r["id"]]
        return rows
    except sqlite3.Error:
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Session state and logging
# ---------------------------------------------------------------------------

def _surface_dir(settings: Settings) -> Path:
    return settings.data_dir / "surface"

def _state_path(settings: Settings, session_id: str) -> Path:
    safe = _SESSION_ID_SAFE.sub("_", session_id) or "unknown"
    return _surface_dir(settings) / f"session-{safe}.json"


def _load_surfaced_ids(settings: Settings, session_id: str) -> set[str]:
    try:
        data = json.loads(
            _state_path(settings, session_id).read_text(encoding="utf-8")
        )
        return set(data.get("surfaced", []))
    except Exception:
        return set()


def _save_surfaced_ids(settings: Settings, session_id: str,
                       ids: set[str], now: float) -> None:
    d = _surface_dir(settings)
    d.mkdir(parents=True, exist_ok=True)
    _state_path(settings, session_id).write_text(
        json.dumps({"surfaced": sorted(ids)}, ensure_ascii=False),
        encoding="utf-8",
    )
    # While we're at it, clean up old session state (cheap even every time
    # since the count is small)
    try:
        for f in d.glob("session-*.json"):
            if now - f.stat().st_mtime > _STATE_MAX_AGE_SECONDS:
                f.unlink(missing_ok=True)
    except OSError:
        pass


def _append_log(settings: Settings, entry: dict) -> None:
    d = _surface_dir(settings)
    d.mkdir(parents=True, exist_ok=True)
    log = d / "surface_log.jsonl"
    try:
        if log.is_file() and log.stat().st_size > _LOG_ROTATE_BYTES:
            log.replace(log.with_suffix(".jsonl.old"))
    except OSError:
        pass
    # Don't let stray invalid surrogates etc. in the input crash the log write
    with log.open("a", encoding="utf-8", errors="replace") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_surface(
    prompt: str,
    *,
    settings: Settings,
    room: str = "common",
    session_id: str = "unknown",
    mode: str | None = None,
    now: float | None = None,
    dry_run: bool = False,
) -> dict:
    """Find memories relevant to the prompt, and update the log/state
    according to the mode.

    dry_run=True is for manual inspection (the `engram surface` command) — it
    writes neither log nor state, and just returns the candidates.
    Returns: {"mode", "room", "candidates": [...], "surfaced": [...]}
    candidates is the top 5 (sorted by descending score); surfaced is the list
    of ids that cleared the threshold and duplicate-suppression and are
    actually injected (or would be, under shadow mode).
    """
    ts = now if now is not None else time.time()
    mode = mode or settings.surface_mode
    rooms = sorted({room, "common"})

    result: dict = {"mode": mode, "room": room, "candidates": [],
                    "surfaced": []}
    if mode == "off":
        return result

    rows = _fetch_candidates(settings.db_path, rooms)
    if not rows:
        return result

    lex = lexical_scores(prompt, [r["content"] for r in rows])

    scored: list[tuple[float, float, float, dict]] = []
    for r, rel in zip(rows, lex):
        d = dynamics.decay_rate(r["importance"])
        act = dynamics.activation_norm(
            r["events"], ts, d, min_elapsed=settings.min_elapsed_seconds
        )
        score = dynamics.final_score(
            rel, act, r["importance"],
            w_relevance=settings.w_relevance,
            w_activation=settings.w_activation,
            w_importance=settings.w_importance,
        )
        scored.append((score, rel, act, r))
    scored.sort(key=lambda t: t[0], reverse=True)

    top = scored[:5]
    result["candidates"] = [
        {
            "id": r["id"],
            "score": round(score, 4),
            "relevance": round(rel, 4),
            "activation": round(act, 4),
            "importance": r["importance"],
            "type": r["type"],
            "room": r["room"],
            "content": r["content"],
        }
        for score, rel, act, r in top
    ]

    # Threshold + relevance gate + suppress re-surfacing within the same session
    already = set() if dry_run else _load_surfaced_ids(settings, session_id)
    surfaced: list[dict] = []
    for cand in result["candidates"]:
        if len(surfaced) >= settings.surface_max_items:
            break
        if cand["score"] < settings.surface_threshold:
            continue
        if cand["relevance"] < settings.surface_min_relevance:
            continue
        if cand["id"] in already:
            continue
        surfaced.append(cand)
    result["surfaced"] = [c["id"] for c in surfaced]
    result["surfaced_items"] = surfaced

    if dry_run:
        return result

    # Update state the same way in shadow mode as in active mode (so shadow
    # mode is a faithful simulation of production). The log is raw data for
    # tuning and auditing.
    if surfaced:
        _save_surfaced_ids(
            settings, session_id, already | set(result["surfaced"]), ts
        )
    _append_log(settings, {
        "ts": ts,
        "session_id": session_id,
        "mode": mode,
        "room": room,
        "prompt": " ".join(prompt.split())[:120],
        "candidates": [
            {k: (v[:80] if k == "content" else v) for k, v in c.items()}
            for c in result["candidates"]
        ],
        "surfaced": result["surfaced"],
    })
    return result


def format_context(surfaced_items: list[dict]) -> str:
    """Build the text injected into context in active mode."""
    lines = [
        "(engram proactive recall) The following memories were automatically "
        "surfaced from your memory store as possibly relevant to what you "
        "just said.",
        "",
    ]
    for c in surfaced_items:
        content = " ".join(c["content"].split())
        if len(content) > 300:
            content = content[:299] + "…"
        lines.append(f"- [{c['id']}] ({c['type']}/{c['room']}) {content}")
    lines += [
        "",
        "Only if this actually turns out to be useful, pass its id to "
        "engram's reinforce to reinforce it. If the content is wrong, use "
        "correct; if it's irrelevant, ignore it.",
    ]
    return "\n".join(lines)
