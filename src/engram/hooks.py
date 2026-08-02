"""Entry points invoked by agent hooks (1. automatic encoding, 2. spontaneous recall).

Registered in Claude Code's hooks (~/.claude/settings.json); receives JSON on
stdin and acts on it:

- SessionEnd        -> run_session_end : auto-saves the session as an episode
- UserPromptSubmit  -> run_user_prompt : spontaneous surfacing of related memories

Design principle: hooks must never block the agent. Every failure is swallowed
and the hook exits 0; the details are left in ~/.engram/hooks.log.
"""

from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path

from .config import Settings, get_settings, resolve_room

_ENCODED_SESSIONS_KEEP = 500

# Prefixes of text that is injected by the agent/harness rather than typed by
# a human. If spontaneous recall reacted to these, unrelated memories would
# surface on every system notification and become noise (confirmed against
# real logs). Slash commands are excluded too.
_NON_HUMAN_PREFIXES = (
    "/",                       # Slash commands
    "<",                       # <task-notification> / <command-name> / <!-- ... etc.
    "Caveat:",                 # Notice shown when running a local command
    "[Request interrupted",    # Interruption notice
)


def _is_non_human(text: str) -> bool:
    """Determine whether an utterance did not originate from a human (system/harness text)."""
    return any(text.startswith(p) for p in _NON_HUMAN_PREFIXES)


def _log(settings: Settings, message: str) -> None:
    try:
        path = settings.data_dir / "hooks.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{stamp} {message}\n")
    except OSError:
        pass


def _read_stdin_json(stdin_text: str | None) -> dict:
    """Read JSON from stdin.

    The hook caller (Claude Code = Node) writes UTF-8, but on Windows, Python
    interprets redirected stdin as cp932, which corrupts non-ASCII prompts
    and paths (observed in the wild). Bypass the text layer entirely and
    decode the raw bytes as UTF-8 directly.
    """
    if stdin_text is None:
        try:
            stdin_text = sys.stdin.buffer.read().decode("utf-8",
                                                        errors="replace")
        except Exception:
            try:
                stdin_text = sys.stdin.read()
            except Exception:
                return {}
    try:
        data = json.loads(stdin_text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 1. Automatic encoding (SessionEnd)
# ---------------------------------------------------------------------------

def _already_encoded(settings: Settings, session_id: str) -> bool:
    path = settings.data_dir / "encoded_sessions.txt"
    if not path.is_file():
        return False
    try:
        lines = path.read_text(encoding="utf-8").split()
        return session_id in lines
    except OSError:
        return False


def _mark_encoded(settings: Settings, session_id: str) -> None:
    path = settings.data_dir / "encoded_sessions.txt"
    try:
        lines: list[str] = []
        if path.is_file():
            lines = path.read_text(encoding="utf-8").split()
        lines.append(session_id)
        path.write_text(
            "\n".join(lines[-_ENCODED_SESSIONS_KEEP:]) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Automatic consolidation nudge
#
# Only an LLM (the agent) can produce a summary, so engram itself cannot
# perform consolidation. Instead, SessionEnd tallies the candidate cluster
# count into a state file, and the next session's UserPromptSubmit
# (lightweight path) prompts the agent to do the consolidation work once the
# threshold is exceeded. This moves the human notion of "memory
# consolidation during sleep" to whenever the agent next has a free moment.
# ---------------------------------------------------------------------------

def _consolidation_state_path(settings: Settings) -> Path:
    return settings.data_dir / "consolidation_state.json"


def _read_consolidation_state(settings: Settings) -> dict:
    try:
        return json.loads(
            _consolidation_state_path(settings).read_text(encoding="utf-8")
        )
    except Exception:
        return {}


def _write_consolidation_state(settings: Settings, updates: dict) -> None:
    state = _read_consolidation_state(settings)
    state.update(updates)
    path = _consolidation_state_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def _consolidation_nudge(settings: Settings, now: float | None = None) -> str | None:
    """Return a nudge message if consolidation candidates have piled up (lightweight check, no engine).

    Nudging too often is just noise, so this is gated both by a cluster-count
    threshold and by a minimum interval between nudges. If writing to the
    state file fails, don't nudge (to avoid nudging on every single prompt).
    """
    if not settings.consolidate_nudge:
        return None
    state = _read_consolidation_state(settings)
    clusters = int(state.get("clusters", 0) or 0)
    if clusters < settings.consolidate_nudge_min_clusters:
        return None
    ts = now if now is not None else time.time()
    last = float(state.get("last_nudged_at", 0.0) or 0.0)
    if ts - last < settings.consolidate_nudge_interval_days * 86400.0:
        return None
    try:
        _write_consolidation_state(settings, {"last_nudged_at": ts})
    except OSError:
        return None
    return (
        f"(engram memory consolidation) {clusters} clusters of consolidation-candidate "
        "episodes have piled up. At a good stopping point, call "
        "consolidation_candidates, summarize each cluster's content and save it with "
        "remember (type=knowledge, preserving the source cluster's context), then "
        "demote the original episodes with mark_consolidated(episode_ids, "
        "new_memory_id). (This is maintenance work equivalent to memory "
        "consolidation during human sleep.)"
    )


# ---------------------------------------------------------------------------
# Automatic skill-candidate nudge
#
# Structurally identical to the consolidation nudge. When episodes recording
# the same shape of work form a cluster of at least a certain size (the
# "three strikes" rule), it's judged worth extracting as a procedure (i.e.
# turning it into a skill), and the agent is prompted to propose this to the
# user. The state file is shared with consolidation (consolidation_state.json),
# just under different keys.
# ---------------------------------------------------------------------------

def _skill_nudge(settings: Settings, now: float | None = None) -> str | None:
    """Return a nudge message if skill candidates have piled up (lightweight check, no engine).

    Same gate structure as consolidate_nudge: returns None if the setting is
    off, the cluster count is below threshold, the minimum interval hasn't
    elapsed, or writing to the state file fails.
    """
    if not settings.skill_nudge:
        return None
    state = _read_consolidation_state(settings)
    clusters = int(state.get("skill_clusters", 0) or 0)
    if clusters < settings.skill_nudge_min_clusters:
        return None
    ts = now if now is not None else time.time()
    last = float(state.get("last_skill_nudged_at", 0.0) or 0.0)
    if ts - last < settings.skill_nudge_interval_days * 86400.0:
        return None
    try:
        _write_consolidation_state(settings, {"last_skill_nudged_at": ts})
    except OSError:
        return None
    return (
        f"(engram skill candidate) Episodes recording the same shape of work have "
        f"formed {clusters} clusters (each with at least {settings.skill_min_count} "
        "items - the three-strikes rule). At a good stopping point, call "
        "skill_candidates, and if the procedure looks reusable, propose to the user "
        "turning it into a skill (extracting it as a written procedure). Never create "
        "one without asking first - always get approval. Once adopted or declined, "
        "record the outcome with remember (type=knowledge), then clean up the "
        "original episodes with mark_consolidated(episode_ids, new_memory_id)."
    )


def run_session_end(stdin_text: str | None = None) -> int:
    """SessionEnd hook body. Always returns 0 (never blocks session termination)."""
    try:
        settings = get_settings()
    except Exception:
        return 0
    try:
        if not settings.auto_encode:
            return 0

        data = _read_stdin_json(stdin_text)
        session_id = str(data.get("session_id", "")) or "unknown"
        transcript_path = data.get("transcript_path", "")
        cwd = data.get("cwd", "")

        if not transcript_path or not Path(transcript_path).is_file():
            _log(settings, f"session-end {session_id}: no transcript, skipping")
            return 0
        if session_id != "unknown" and _already_encoded(settings, session_id):
            _log(settings, f"session-end {session_id}: already encoded, skipping")
            return 0

        from .transcript import build_episode, extract_messages

        messages = extract_messages(transcript_path)
        project = Path(cwd or messages.get("cwd") or "").name
        episode = build_episode(
            messages,
            date_str=datetime.date.today().isoformat(),
            project=project,
            min_chars=settings.auto_encode_min_chars,
        )
        if episode is None:
            _log(settings, f"session-end {session_id}: content too thin, not recording")
            return 0

        room = resolve_room(cwd or messages.get("cwd"), settings.room_paths)

        # Only load the heavy dependency (the embedding model) here. It's
        # after session end, so this doesn't block the user's interaction.
        from .engine import build_engine

        engine = build_engine(settings)
        result = engine.remember(
            content=episode,
            type="episode",
            importance=settings.auto_episode_importance,
            tags=["auto", "session"],
            source="auto-encode",
            room=room,
        )
        # Tally consolidation candidates (the engine is already built, so
        # this adds negligible cost). Write the count to the state file; the
        # next session's user-prompt hook will nudge if it's over threshold.
        if settings.consolidate_nudge:
            try:
                n_clusters = len(
                    engine.consolidation_candidates().get("clusters", [])
                )
                _write_consolidation_state(
                    settings,
                    {"clusters": n_clusters, "checked_at": time.time()},
                )
            except Exception as e:
                _log(settings, f"session-end consolidation candidate check failed: {e!r}")

        # Tally skill candidates (computed here for the same reason as
        # consolidation). Write the count to the state file; the next
        # session's user-prompt hook will nudge if it's over threshold.
        if settings.skill_nudge:
            try:
                n_skill_clusters = len(
                    engine.skill_candidates().get("clusters", [])
                )
                _write_consolidation_state(
                    settings,
                    {"skill_clusters": n_skill_clusters, "checked_at": time.time()},
                )
            except Exception as e:
                _log(settings, f"session-end skill candidate check failed: {e!r}")

        engine.db.close()
        _mark_encoded(settings, session_id)
        _log(
            settings,
            f"session-end {session_id}: {result.get('status')} "
            f"id={result.get('id')} room={room}",
        )
    except Exception as e:
        _log(settings, f"session-end error: {e!r}")
    return 0


# ---------------------------------------------------------------------------
# 2. Spontaneous recall (UserPromptSubmit)
# ---------------------------------------------------------------------------

def run_user_prompt(stdin_text: str | None = None) -> int:
    """UserPromptSubmit hook body. Always returns 0 (never blocks the prompt).

    In active mode, surfaced memories are emitted as
    hookSpecificOutput.additionalContext and injected into the agent's
    context. In shadow mode, only logging occurs.
    """
    try:
        settings = get_settings()
    except Exception:
        return 0
    try:
        data = _read_stdin_json(stdin_text)
        prompt = str(data.get("prompt", ""))
        session_id = str(data.get("session_id", "")) or "unknown"
        cwd = data.get("cwd", "")

        stripped = prompt.strip()
        if len(stripped) < settings.surface_min_prompt_chars:
            return 0
        if _is_non_human(stripped):  # Skip system/harness text
            return 0

        context_parts: list[str] = []

        if settings.surface_mode != "off":
            room = resolve_room(cwd, settings.room_paths)

            from .surface import format_context, run_surface

            result = run_surface(
                prompt,
                settings=settings,
                room=room,
                session_id=session_id,
            )

            if settings.surface_mode == "active" and result.get("surfaced_items"):
                context_parts.append(format_context(result["surfaced_items"]))

        # Consolidation nudge (independent of surface; gated by consolidate_nudge setting)
        nudge = _consolidation_nudge(settings)
        if nudge:
            context_parts.append(nudge)

        # Skill-candidate nudge (independent of consolidation; gated by skill_nudge setting)
        skill_nudge = _skill_nudge(settings)
        if skill_nudge:
            context_parts.append(skill_nudge)

        if context_parts:
            payload = {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": "\n\n".join(context_parts),
                }
            }
            print(json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        _log(settings, f"user-prompt error: {e!r}")
    return 0
