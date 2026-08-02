"""Build a session summary from a Claude Code transcript (JSONL) (1. automatic
encoding).

A deterministic, LLM-free summary: since a session's backbone is what the user
said, mechanically assemble "the first request + the main messages along the
way + the gist of the final response." It's a coarse record, but its value is
that "a session automatically becomes an episode memory once it ends."
Precise insights are remembered by the agent on the spot instead (division of
labor).
"""

from __future__ import annotations

import json
from pathlib import Path

# Prefixes of non-utterance text that Claude Code mixes into the transcript
# under the user role
_NOISE_PREFIXES = (
    "<",            # <command-name> / <system-reminder> / <local-command-stdout> etc.
    "Caveat:",      # notice shown when a local command is run
    "[Request interrupted",
)


def _texts_from_content(content) -> list[str]:
    """Extract text from message.content (a str, or a list of blocks)."""
    if isinstance(content, str):
        return [content]
    texts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    texts.append(t)
    return texts


def _is_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return any(stripped.startswith(p) for p in _NOISE_PREFIXES)


def extract_messages(transcript_path: str | Path) -> dict:
    """Scan the transcript JSONL and return the raw material for the summary.

    Returns: {
        "user_texts": [the user's actual messages, in chronological order],
        "last_assistant": the last assistant message (empty string if none),
        "summary": the session title Claude Code assigned (empty string if none),
        "cwd": the working directory recorded in the transcript (empty string if none),
    }
    Malformed lines and unknown formats are silently skipped (never crash the hook).
    """
    user_texts: list[str] = []
    last_assistant = ""
    summary = ""
    cwd = ""

    path = Path(transcript_path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            if not cwd and isinstance(obj.get("cwd"), str):
                cwd = obj["cwd"]

            kind = obj.get("type")
            if kind == "summary":
                s = obj.get("summary")
                if isinstance(s, str) and not summary:
                    summary = s.strip()
                continue
            if obj.get("isMeta"):
                continue

            message = obj.get("message")
            if not isinstance(message, dict):
                continue

            if kind == "user" and message.get("role") == "user":
                for t in _texts_from_content(message.get("content")):
                    if not _is_noise(t):
                        user_texts.append(t.strip())
            elif kind == "assistant":
                for t in _texts_from_content(message.get("content")):
                    if t.strip():
                        last_assistant = t.strip()

    return {
        "user_texts": user_texts,
        "last_assistant": last_assistant,
        "summary": summary,
        "cwd": cwd,
    }


def _clip(text: str, limit: int) -> str:
    """Flatten to a single line and truncate to `limit` characters."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def build_episode(
    messages: dict,
    *,
    date_str: str,
    project: str = "",
    min_chars: int = 24,
    max_user_items: int = 6,
) -> str | None:
    """Assemble the episode body from the result of extract_messages.

    Returns None for sessions not worth recording (no actual user messages,
    or extremely short).
    """
    user_texts: list[str] = messages.get("user_texts", [])
    if not user_texts:
        return None
    if sum(len(t) for t in user_texts) < min_chars:
        return None

    title = messages.get("summary") or _clip(user_texts[0], 60)
    where = f", {project}" if project else ""

    lines: list[str] = [f"Session ({date_str}{where}): {_clip(title, 80)}", ""]
    lines.append("User's requests / messages:")
    lines.append(f"1. {_clip(user_texts[0], 200)}")

    # From the 2nd message on: if there are many, pull from the start and the
    # end (more informative than the back-and-forth in the middle)
    rest = user_texts[1:]
    if len(rest) > max_user_items - 1:
        head_n = (max_user_items - 1) // 2
        tail_n = max_user_items - 1 - head_n
        picked = rest[:head_n] + rest[-tail_n:]
        omitted = len(rest) - len(picked)
    else:
        picked = rest
        omitted = 0
    for i, t in enumerate(picked, start=2):
        lines.append(f"{i}. {_clip(t, 110)}")
    if omitted > 0:
        lines.append(f"(and {omitted} more messages)")

    last_assistant = messages.get("last_assistant", "")
    if last_assistant:
        lines.append("")
        lines.append(f"Outcome (summary of the last response): {_clip(last_assistant, 280)}")

    return "\n".join(lines)
