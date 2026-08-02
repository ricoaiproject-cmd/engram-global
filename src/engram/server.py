"""MCP server (owner: Agent B).

Exposes engine operations as tools via the official mcp SDK's FastMCP (stdio).

    from mcp.server.fastmcp import FastMCP

Implementation requirements:
- The engine is lazily constructed on the first tool call (build_engine), to keep
  startup fast.
- Each tool delegates to the engine method of the same name and returns the dict
  as-is.
- The docstring becomes the tool's description verbatim, so word it so an agent
  can judge "when should I call this."
- Tools: remember / recall / reinforce / correct / link / forget /
         consolidation_candidates / mark_consolidated / skill_candidates /
         reindex / stats
- Arguments follow the engine's method signatures (`now` is not exposed).
- main() runs over stdio: mcp.run().
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import perf
from .config import resolve_room
from .engine import MemoryEngine, build_engine

mcp = FastMCP("engram")

# Module-level lazy singleton (constructed on the first tool call)
_engine: MemoryEngine | None = None
_engine_lock = threading.Lock()

# The server's default room. Determined at process startup from the working
# directory (= the project the agent was launched from) via config's room_paths
_room: str | None = None


def _get_engine() -> MemoryEngine:
    """Lazily construct and return the engine (thread-safe)."""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            _engine = build_engine()
            _run_startup_index_check(_engine)
    return _engine


def _timed(name: str):
    """Thin delegate to perf.timed (settings is fetched via _get_engine and cached).

    FastMCP inspects a tool function's signature to build its schema, so rather
    than wrapping the tool function itself in a decorator, use
    `with _timed(...):` inside each tool's body (this never touches the
    function's signature or docstring).
    """
    return perf.timed(_get_engine().settings, "tool", name)


def _run_startup_index_check(engine: MemoryEngine) -> None:
    """Detect drift between the Markdown files and the index at startup;
    reindex if mode is auto, warn if mode is warn.

    Memory Markdown files can be shared (e.g. via Google Drive) while index.db
    is local to each machine, which creates a blind spot: memories written by
    another machine aren't picked up into the index and don't show up in
    recall. This resolves that at startup. Failure here never blocks serving
    the engine (availability of the memory substrate is the top priority).
    """
    import sys

    try:
        mode = getattr(engine.settings, "startup_index_check", "auto")
        res = engine.check_index_freshness(mode=mode)
        action = res.get("action")
        if action == "reindexed":
            print(
                f"engram: index out of sync (markdown={res.get('markdown')} "
                f"index={res.get('index')}) -> reindexed {res.get('reindex')}",
                file=sys.stderr,
            )
        elif action == "warn":
            print(
                f"engram: WARNING memory index out of sync "
                f"(markdown={res.get('markdown')} vs index={res.get('index')}). "
                f"Run 'engram reindex' to sync this machine.",
                file=sys.stderr,
            )
    except Exception as e:
        print(f"engram: startup index check skipped: {e}", file=sys.stderr)


def _default_room() -> str:
    """Resolve the memory room from the server's startup directory (computed
    once, on first use)."""
    global _room
    if _room is None:
        try:
            settings = _get_engine().settings
            _room = resolve_room(Path.cwd(), settings.room_paths)
        except Exception:
            _room = "common"
    return _room


@mcp.tool()
def remember(
    content: str,
    type: str,
    importance: int,
    tags: list[str] | None = None,
    source: str = "unknown",
    related_ids: list[str] | None = None,
    room: str | None = None,
) -> dict:
    """Save a new memory.

    Call this whenever you discover something important during a task (a fact,
    a preference, project status, an event). Self-score `importance` from 1-10
    based on how significant it is in context.
    If a sufficiently similar existing memory is found (cosine similarity >=
    dup_threshold=0.95), it is reinforced as a duplicate and returned instead.
    You normally don't need to pass `room` (it's inferred automatically from
    the working directory). Only pass room="common" explicitly for universal
    memories that apply across every context.
    """
    engine = _get_engine()
    with _timed("remember"):
        return engine.remember(
            content=content,
            type=type,
            importance=importance,
            tags=tags,
            source=source,
            related_ids=related_ids,
            room=room if room is not None else _default_room(),
        )


@mcp.tool()
def recall(
    query: str,
    mode: str = "fast",
    limit: int = 5,
    type: str | None = None,
    record_hits: bool = True,
    room: str | None = None,
) -> dict:
    """Search for and return memories.

    Always call this at the start of a task. It surfaces relevant past
    knowledge, preferences, and project context.
    mode="fast" searches only tier=hot, quickly. If the score is low it
    automatically falls back to deep.
    mode="deep" searches more broadly, including cold/superseded/episode
    memories, and follows associative links.
    mode="exhaustive" ignores activation and ranks purely by relevance across
    every memory. Use this as a last resort to dig up a "sunk" memory you're
    sure you recorded but that isn't surfacing under fast/deep.
    You normally don't need to pass `room` (it searches only the current room
    plus common). room="*" searches across all rooms — use it only when
    necessary, since it breaks the separation between work and personal
    contexts.
    """
    engine = _get_engine()
    with _timed("recall"):
        return engine.recall(
            query=query,
            mode=mode,
            limit=limit,
            type=type,
            record_hits=record_hits,
            room=room if room is not None else _default_room(),
        )


@mcp.tool()
def reinforce(
    ids: list[str],
    strength: float = 1.0,
) -> dict:
    """Report which memories actually turned out to be useful.

    When a task finishes, report the ids of memories that actually helped.
    Reinforced memories are more likely to surface near the top on the next
    recall.
    Passing multiple ids at once links those memories together via a
    co-occurrence link (Hebbian learning).
    strength ranges 0.1-3.0 and controls how strong the reinforcement is.
    """
    engine = _get_engine()
    with _timed("reinforce"):
        return engine.reinforce(ids=ids, strength=strength)


@mcp.tool()
def correct(
    id: str,
    corrected_content: str,
    reason: str,
    source: str = "unknown",
) -> dict:
    """Use this instead of forget when a memory turns out to be wrong.

    Demotes the old memory to superseded (corrected) and creates a new memory
    that records the reason for the correction. Explicitly recording the
    mistake prevents the same error from being repeated (a hypercorrection
    effect).
    """
    engine = _get_engine()
    with _timed("correct"):
        return engine.correct(
            id=id,
            corrected_content=corrected_content,
            reason=reason,
            source=source,
        )


@mcp.tool()
def link(src: str, dst: str) -> dict:
    """Create an explicit link between two memories.

    Call this when you want to manually connect related memories.
    Deep recall can then follow this link to surface memories associatively.
    """
    engine = _get_engine()
    with _timed("link"):
        return engine.link(src=src, dst=dst)


@mcp.tool()
def forget(id: str) -> dict:
    """Soft-delete a memory (move it to trash).

    Call this when a memory is no longer needed and you want it excluded from
    search.
    This moves the memory to trash rather than physically deleting it, so it
    can be restored if deleted by mistake.
    If you want to correct an error rather than remove a memory, use correct
    instead of forget.
    """
    engine = _get_engine()
    with _timed("forget"):
        return engine.forget(id=id)


@mcp.tool()
def consolidation_candidates() -> dict:
    """Return clusters of episode memories that are candidates for
    consolidation.

    Call this before ending a session (at session end), to surface clusters of
    similar older episodes that are candidates for compressing into knowledge
    or project memories.
    The LLM generates the summary, then calls mark_consolidated to complete the
    consolidation.
    """
    engine = _get_engine()
    with _timed("consolidation_candidates"):
        return engine.consolidation_candidates()


@mcp.tool()
def mark_consolidated(episode_ids: list[str], new_memory_id: str) -> dict:
    """Record that a consolidation has been completed.

    Call this after the LLM has summarized a cluster surfaced by
    consolidation_candidates and created the new memory via remember.
    The original episodes are demoted to cold (long-term storage) and linked
    to the new memory via a derived_from link.
    Also use this tool to demote the target episodes to cold after acting on a
    skill_candidates cluster.
    """
    engine = _get_engine()
    with _timed("mark_consolidated"):
        result = engine.mark_consolidated(
            episode_ids=episode_ids,
            new_memory_id=new_memory_id,
        )
    # Consolidation changes the cluster count, so refresh the nudge state
    # immediately. Skipping this would keep nudging with the stale cluster
    # count until the next session-end (updates both the consolidation and
    # skill-candidate nudge state).
    try:
        import time

        from .hooks import _write_consolidation_state

        n = len(engine.consolidation_candidates().get("clusters", []))
        n_skill = len(engine.skill_candidates().get("clusters", []))
        _write_consolidation_state(
            engine.settings,
            {
                "clusters": n,
                "skill_clusters": n_skill,
                "checked_at": time.time(),
            },
        )
    except Exception:
        pass
    return result


@mcp.tool()
def skill_candidates() -> dict:
    """Return clusters of episode memories that are candidates for
    extraction into a reusable skill.

    When 3 or more (default; the "three-times rule") episodes recording the
    same shape of work (procedure) form a similar cluster, use this as input
    for judging whether that procedure is worth extracting into a reusable
    skill (a how-to document — a SKILL.md for Claude Code, etc.). Unlike
    consolidation_candidates, there is no age filter here (recently repeated
    work is exactly the target).
    Even when a cluster is found, always propose turning it into a skill to
    the user and get their approval first. Never create or deploy a skill on
    your own.
    Once the decision (adopt or pass) is made, record the reasoning via
    remember(type=knowledge), then clean up the original episodes with
    mark_consolidated(episode_ids, new_memory_id).
    """
    engine = _get_engine()
    with _timed("skill_candidates"):
        return engine.skill_candidates()


@mcp.tool()
def reindex() -> dict:
    """Rebuild the DB index from the Markdown files.

    Call this after manually editing files, or when you suspect the DB is
    corrupted.
    Only memories that differ are re-embedded, so this is faster than a full
    rebuild.
    """
    engine = _get_engine()
    with _timed("reindex"):
        return engine.reindex()


@mcp.tool()
def stats() -> dict:
    """Return memory statistics.

    Shows memory counts (by type and tier), the number of access events, the
    number of links, and so on.
    """
    engine = _get_engine()
    with _timed("stats"):
        return engine.stats()


def _preload() -> None:
    """Preload the embedding model (startup continues even on failure). Times
    the whole preload duration."""
    import sys
    import time

    # settings can be obtained from config alone, even before the engine is
    # built, so fetch it here first so timing works even if engine
    # construction itself fails
    from .config import get_settings

    settings = get_settings()
    start = time.perf_counter()
    ok = True
    try:
        engine = _get_engine()
        engine.embedder.embed_query("warm-up")
        print("engram: engine preloaded", file=sys.stderr)
    except Exception as e:  # keep starting up even on failure; retry lazily on the first tool call
        ok = False
        print(f"engram: preload failed, will retry lazily: {e}", file=sys.stderr)
    finally:
        if settings.perf_log:
            ms = (time.perf_counter() - start) * 1000.0
            perf.append_perf(
                settings,
                {"ts": time.time(), "kind": "preload", "name": "preload", "ms": ms, "ok": ok},
            )


def _resolve_preload_mode(raw: str | None, onnx_ready: bool) -> str:
    """Resolve the ENGRAM_PRELOAD value into the actual preload strategy
    (pure function, covered by tests).

    Explicit values of blocking / background / off pass through unchanged.
    auto (the default) and unknown values resolve to background if the ONNX
    model has already been generated, or blocking in a torch-fallback
    environment. See the comment at the top of main() for the rationale.
    """
    mode = (raw or "auto").strip().lower()
    if mode in ("blocking", "background", "off"):
        return mode
    return "background" if onnx_ready else "blocking"


def main() -> None:
    """Start the stdio MCP server."""
    # [Since v0.6.0] The default runtime is ONNX (embed_backend=auto +
    # export-onnx already run). The record of the torch pathology below is
    # kept as documentation of what happens if you fall back to an
    # environment where the ONNX model hasn't been generated yet.
    #
    # Importing torch / sentence_transformers is extremely heavy (measured:
    # import alone takes 50+ seconds cold). Running this on the main thread
    # before mcp.run() blocks the initialize handshake for that whole time,
    # and MCP clients with a short startup timeout give up and disconnect
    # (observed cases: Antigravity IDE's "context canceled"; Claude Code's
    # default 30-second startup timeout causing intermittent connection
    # failures on cold starts).
    #
    # On the other hand, doing this import "on a separate thread" while
    # mcp.run()'s event loop is running is drastically slower on Windows
    # (measured: main thread 12-24s -> daemon/worker thread ~184s;
    # reproduced across two Claude Code MCP log sessions on 2026-07-02).
    # Background preloading does return the handshake quickly (1.5s), but the
    # first recall then has to wait for this slow load, turning into a
    # 180-second-class stall that surfaces as a client tool timeout. Since a
    # thread import while the event loop is stopped only takes 6 seconds (not
    # reproducible in isolation), this looks like GIL/DLL-loader contention
    # with the event loop. Lesson: the only reliably fast path for a heavy
    # import is to do it on the main thread before the event loop starts.
    #
    # [Since v0.10.0] We've confirmed by measurement that the above pathology
    # does not occur on the ONNX path (2026-07-12: loading the embedder on a
    # daemon thread while the event loop is running took 4.9s — almost the
    # same as the main thread's 4.7s. The 184-second-class degradation is
    # specific to torch's DLL loader). So the default is now auto: background
    # (handshake responds immediately, the first tool call waits at most a
    # few seconds) if the ONNX model has already been generated, otherwise
    # the traditional blocking behavior in a torch-fallback environment.
    # Background: some real MCP clients don't respect a longer startup
    # timeout setting (observed case: Codex Desktop 26.707 recognizes
    # startup_timeout_sec=120 but doesn't wait out blocking's dozen-plus
    # seconds, and gets stuck without completing initialization).
    #
    # ENGRAM_PRELOAD lets you force a specific preload strategy:
    #   auto (default)  — background if ONNX has been generated, otherwise
    #                     blocking.
    #   blocking        — preload on the main thread at startup. The
    #                     handshake waits for the import to finish (on the
    #                     torch path this is warm 12-24s / cold 50+s, so set
    #                     the client's MCP startup timeout to 120+ seconds —
    #                     for Claude Code, set MCP_TIMEOUT=120000 in the env
    #                     section of settings.json). Every recall after
    #                     connecting responds immediately.
    #   background      — run preload on a daemon thread and start mcp.run()
    #                     immediately. Safe on the ONNX path (the first tool
    #                     call just waits a few seconds for preload to
    #                     finish). On the torch path, an explicit choice of
    #                     this is discouraged, since the pathology above can
    #                     make the first recall on Windows take 3-minute-class
    #                     time.
    #   off             — don't preload. Loads lazily on the first tool call
    #                     (hits the same pathology as background on the torch
    #                     path).
    # Because FastMCP runs synchronous tools on a worker thread, both
    # _get_engine and RuriEmbedder._load use a lock to prevent duplicate
    # loading.
    from .config import get_settings, onnx_model_ready

    mode = _resolve_preload_mode(
        os.environ.get("ENGRAM_PRELOAD"),
        onnx_model_ready(get_settings().onnx_model_dir),
    )
    if mode == "blocking":
        _preload()
    elif mode == "background":
        threading.Thread(
            target=_preload, name="engram-preload", daemon=True
        ).start()

    mcp.run()
