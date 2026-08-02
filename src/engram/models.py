"""Shared data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

MEMORY_TYPES = ("knowledge", "preference", "project", "episode")
TIERS = ("hot", "cold", "superseded", "trash")
EVENT_KINDS = ("create", "recall_hit", "reinforce", "correction")
LINK_KINDS = ("explicit", "co_recall", "derived_from", "superseded_by")


@dataclass
class MemoryRecord:
    """A single memory. The Markdown file is the source of truth; the DB is just an index."""

    id: str                      # ULID
    type: str                    # one of MEMORY_TYPES
    created: str                 # ISO 8601 (with local TZ)
    importance: int              # 1-10, scored by the calling agent from context
    tags: list[str] = field(default_factory=list)
    source: str = "unknown"      # claude-code / codex / antigravity / etc.
    tier: str = "hot"
    links: list[str] = field(default_factory=list)  # list of linked memory ids
    content: str = ""            # body text (frontmatter excluded)
    path: Path | None = None     # absolute path to the Markdown file
    content_hash: str = ""       # sha256 of the body (for detecting manual edits)
    room: str = "common"         # memory room (work/personal context separation; defaults to common)


@dataclass
class RecallHit:
    """A single recall result. Returned with a score breakdown so the agent can judge it."""

    id: str
    content: str
    type: str
    tags: list[str]
    tier: str
    score: float                 # final score
    relevance: float             # semantic relevance to the query (0-1)
    activation: float            # activation (normalized 0-1)
    importance: float            # importance/10
    via: str = "direct"          # "direct" | "associative" (reached via an associative link)
    note: str = ""               # e.g. "-> corrected by [id]"
    room: str = "common"         # memory room
