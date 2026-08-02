"""Markdown source-of-truth store (owner: Agent A).

1 memory = 1 file (atomic, Zettelkasten-style). Viewable and editable directly in
Obsidian.

Layout:
    memories/knowledge/   memories/preferences/   memories/projects/
    memories/episodes/YYYY/MM/    memories/_trash/
type → subdirectory mapping: knowledge→knowledge, preference→preferences,
project→projects, episode→episodes/YYYY/MM (derived from created).

File format (read/written via python-frontmatter):
    ---
    id: 01JXXXX...            # ULID
    type: knowledge
    created: 2026-06-11T09:00:00+09:00
    tags: [sqlite, windows]
    importance: 7
    source: claude-code
    tier: hot
    links: ["[[01JYYY...]]"]   # stored in Obsidian wiki-link format
    ---
    body (plain Markdown)

Implementation requirements:
- Filename: YYYYMMDD-{slug}-{last 6 chars of id}.md
  slug = generated from the first line of the body (max 40 chars, with Windows
  forbidden chars <>:"/\\|?* and newlines stripped, spaces replaced with '-'.
  Japanese text is left as-is).
- links are stored in frontmatter as "[[id]]" strings, but as a plain list of
  ids on MemoryRecord.links. Converted both ways on read/write.
- content_hash = sha256 hex of the body (frontmatter excluded, after strip).
- Tier changes and body updates rewrite the file in place. Only episodes
  involve a directory move (tier lives purely in frontmatter). forget moves
  the file to _trash/.
- scan_all() yields every .md file except those under _trash as a
  MemoryRecord (used for reindex).
- Files with broken frontmatter are collected as warnings and skipped
  (never raises).
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import warnings
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import frontmatter

from .models import MemoryRecord

logger = logging.getLogger(__name__)

# Windows forbidden filename characters + control chars
_WIN_FORBIDDEN = re.compile(r'[<>:"/\\|?*\r\n\x00-\x1f]')

# Mapping type → subdirectory name (non-episode)
_TYPE_TO_DIR: dict[str, str] = {
    "knowledge": "knowledge",
    "preference": "preferences",
    "project": "projects",
}


def content_hash(content: str) -> str:
    """Normalized hash of the body (sha256 hex after strip)."""
    normalized = content.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _make_slug(content: str) -> str:
    """Generate a slug from the first line of the body."""
    first_line = content.strip().splitlines()[0] if content.strip() else ""
    # Remove Windows forbidden chars and control chars
    slug = _WIN_FORBIDDEN.sub("", first_line)
    # Replace spaces with hyphens
    slug = slug.replace(" ", "-")
    # Truncate to 40 chars
    slug = slug[:40]
    # If empty, use a fallback
    if not slug:
        slug = "memory"
    return slug


def _parse_created_date(created: str) -> datetime:
    """Return a datetime parsed from an ISO 8601 string."""
    # Handle timezone offset like +09:00
    try:
        # Python 3.7+ fromisoformat handles most cases but not all timezone formats
        return datetime.fromisoformat(created)
    except (ValueError, TypeError):
        return datetime.now(tz=timezone.utc)


def _links_to_frontmatter(links: list[str]) -> list[str]:
    """Convert a plain list of ids → a list of "[[id]]" strings."""
    return [f"[[{lid}]]" for lid in links]


def _links_from_frontmatter(raw: list | None) -> list[str]:
    """Convert frontmatter links → a plain list of ids (strips "[[id]]")."""
    if not raw:
        return []
    result = []
    for item in raw:
        s = str(item).strip()
        if s.startswith("[[") and s.endswith("]]"):
            result.append(s[2:-2])
        else:
            result.append(s)
    return result


def _subdir_for_type(root: Path, type: str, created: str) -> Path:
    """Determine the destination directory from type and created."""
    if type == "episode":
        dt = _parse_created_date(created)
        return root / "episodes" / dt.strftime("%Y") / dt.strftime("%m")
    return root / _TYPE_TO_DIR.get(type, type)


def _filename_from_record(id: str, type: str, created: str, content: str) -> str:
    """YYYYMMDD-{slug}-{last 6 chars of id}.md"""
    dt = _parse_created_date(created)
    date_str = dt.strftime("%Y%m%d")
    slug = _make_slug(content)
    short_id = id[-6:]
    return f"{date_str}-{slug}-{short_id}.md"


class MarkdownStore:
    def __init__(self, root: Path) -> None:
        """root = the memories directory. Created along with its subdirectories if missing."""
        self._root = Path(root)
        # Create all standard subdirectories
        for subdir in ("knowledge", "preferences", "projects", "_trash"):
            (self._root / subdir).mkdir(parents=True, exist_ok=True)
        # episodes dir
        (self._root / "episodes").mkdir(parents=True, exist_ok=True)

    def _write_record(self, record: MemoryRecord) -> None:
        """Write a MemoryRecord out to a file."""
        post = frontmatter.Post(
            content=record.content,
            id=record.id,
            type=record.type,
            created=record.created,
            tags=record.tags,
            importance=record.importance,
            source=record.source,
            tier=record.tier,
            room=record.room,
            links=_links_to_frontmatter(record.links),
        )
        text = frontmatter.dumps(post)
        assert record.path is not None
        record.path.parent.mkdir(parents=True, exist_ok=True)
        record.path.write_text(text, encoding="utf-8")

    def create(
        self,
        *,
        content: str,
        type: str,
        importance: int,
        tags: list[str] | None = None,
        source: str = "unknown",
        links: list[str] | None = None,
        id: str | None = None,
        created: str | None = None,
        room: str = "common",
    ) -> MemoryRecord:
        """Write a new memory file and return the MemoryRecord."""
        from ulid import ULID

        if id is None:
            id = str(ULID())
        if created is None:
            created = datetime.now().astimezone().isoformat()

        subdir = _subdir_for_type(self._root, type, created)
        subdir.mkdir(parents=True, exist_ok=True)
        filename = _filename_from_record(id, type, created, content)
        path = subdir / filename

        ch = content_hash(content)
        record = MemoryRecord(
            id=id,
            type=type,
            created=created,
            importance=importance,
            tags=tags or [],
            source=source,
            tier="hot",
            links=links or [],
            content=content,
            path=path,
            content_hash=ch,
            room=room,
        )
        self._write_record(record)
        return record

    def read(self, path: Path) -> MemoryRecord:
        path = Path(path)
        post = frontmatter.load(str(path))
        raw_links = post.get("links", [])
        return MemoryRecord(
            id=str(post["id"]),
            type=str(post["type"]),
            created=str(post["created"]),
            importance=int(post["importance"]),
            tags=list(post.get("tags") or []),
            source=str(post.get("source", "unknown")),
            tier=str(post.get("tier", "hot")),
            links=_links_from_frontmatter(raw_links),
            content=post.content,
            path=path,
            content_hash=content_hash(post.content),
            room=str(post.get("room", "common")),
        )

    def find_by_id(self, id: str) -> MemoryRecord | None:
        """The proper approach is to use the DB's path rather than a full scan; a full scan is fine when using the store alone."""
        for record in self.scan_all():
            if record.id == id:
                return record
        return None

    def update(self, record: MemoryRecord) -> MemoryRecord:
        """Rewrite the file at record.path with record's content (recomputes content_hash)."""
        updated = MemoryRecord(
            id=record.id,
            type=record.type,
            created=record.created,
            importance=record.importance,
            tags=record.tags,
            source=record.source,
            tier=record.tier,
            links=record.links,
            content=record.content,
            path=record.path,
            content_hash=content_hash(record.content),
            room=record.room,
        )
        self._write_record(updated)
        return updated

    def add_link(self, record: MemoryRecord, target_id: str) -> MemoryRecord:
        """Add target_id to links (ignoring duplicates) and update the file."""
        if target_id in record.links:
            return record
        updated = MemoryRecord(
            id=record.id,
            type=record.type,
            created=record.created,
            importance=record.importance,
            tags=record.tags,
            source=record.source,
            tier=record.tier,
            links=record.links + [target_id],
            content=record.content,
            path=record.path,
            content_hash=record.content_hash,
            room=record.room,
        )
        self._write_record(updated)
        return updated

    def set_tier(self, record: MemoryRecord, tier: str) -> MemoryRecord:
        updated = MemoryRecord(
            id=record.id,
            type=record.type,
            created=record.created,
            importance=record.importance,
            tags=record.tags,
            source=record.source,
            tier=tier,
            links=record.links,
            content=record.content,
            path=record.path,
            content_hash=record.content_hash,
            room=record.room,
        )
        self._write_record(updated)
        return updated

    def move_to_trash(self, record: MemoryRecord) -> MemoryRecord:
        """Move to _trash/ and set tier=trash. Never physically deletes the file."""
        trash_dir = self._root / "_trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        assert record.path is not None
        new_path = trash_dir / record.path.name
        # Handle name collision in trash
        if new_path.exists() and new_path != record.path:
            stem = record.path.stem
            suffix = record.path.suffix
            new_path = trash_dir / f"{stem}-{record.id[-4:]}{suffix}"
        shutil.move(str(record.path), str(new_path))
        updated = MemoryRecord(
            id=record.id,
            type=record.type,
            created=record.created,
            importance=record.importance,
            tags=record.tags,
            source=record.source,
            tier="trash",
            links=record.links,
            content=record.content,
            path=new_path,
            content_hash=record.content_hash,
            room=record.room,
        )
        self._write_record(updated)
        return updated

    def scan_all(self) -> Iterator[MemoryRecord]:
        """Yield every .md file except those under _trash as a MemoryRecord (used for reindex).
        Files with broken frontmatter are skipped with a warning.
        """
        trash_dir = self._root / "_trash"
        for md_file in self._root.rglob("*.md"):
            # Skip files inside _trash
            try:
                md_file.relative_to(trash_dir)
                continue  # it's inside _trash
            except ValueError:
                pass  # not in _trash, proceed

            try:
                record = self.read(md_file)
                yield record
            except Exception as exc:
                warnings.warn(
                    f"Skipping broken frontmatter in {md_file}: {exc}",
                    stacklevel=2,
                )
                continue

    def count_memory_files(self) -> int:
        """Lightweight count of .md files excluding _trash (does not parse frontmatter).

        Used for the startup index-sync check. Excludes _trash the same way as scan_all.
        """
        trash_dir = self._root / "_trash"
        count = 0
        for md_file in self._root.rglob("*.md"):
            try:
                md_file.relative_to(trash_dir)
                continue  # inside _trash
            except ValueError:
                count += 1
        return count
