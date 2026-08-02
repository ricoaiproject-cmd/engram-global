"""Tests for memory rooms: resolution logic, store round-trip, DB migration, and recall filtering."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from engram.config import Settings, resolve_room
from engram.db import IndexDB
from engram.embedder import FakeEmbedder
from engram.engine import MemoryEngine
from engram.store import MarkdownStore


# ---------------------------------------------------------------------------
# resolve_room
# ---------------------------------------------------------------------------

def test_resolve_room_no_config():
    assert resolve_room("C:/anywhere", {}) == "common"
    assert resolve_room(None, {"C:/w": "work"}) == "common"


def test_resolve_room_prefix_match():
    rooms = {"C:/Users/me/work": "work", "C:/Users/me/personal": "personal"}
    assert resolve_room("C:/Users/me/work/proj1", rooms) == "work"
    assert resolve_room("C:/Users/me/personal", rooms) == "personal"
    assert resolve_room("C:/Users/me/other", rooms) == "common"


def test_resolve_room_longest_prefix_wins():
    rooms = {"C:/p": "broad", "C:/p/sub": "narrow"}
    assert resolve_room("C:/p/sub/x", rooms) == "narrow"
    assert resolve_room("C:/p/other", rooms) == "broad"


def test_resolve_room_separator_and_case_insensitive():
    rooms = {r"C:\Users\Me\Work": "work"}
    assert resolve_room("c:/users/me/work/proj", rooms) == "work"


def test_resolve_room_no_partial_component_match():
    # "work" and "workspace" are different (prefix match is on path-component boundaries)
    rooms = {"C:/work": "work"}
    assert resolve_room("C:/workspace", rooms) == "common"


# ---------------------------------------------------------------------------
# store: frontmatter round-trip
# ---------------------------------------------------------------------------

def test_store_room_roundtrip(tmp_path):
    store = MarkdownStore(tmp_path / "memories")
    rec = store.create(content="部屋テスト", type="knowledge", importance=5,
                       room="work")
    loaded = store.read(rec.path)
    assert loaded.room == "work"


def test_store_room_default_for_legacy_files(tmp_path):
    # Existing files without a room (pre-v0.2) are read as "common"
    store = MarkdownStore(tmp_path / "memories")
    rec = store.create(content="旧形式", type="knowledge", importance=5)
    text = rec.path.read_text(encoding="utf-8")
    text = "\n".join(
        ln for ln in text.splitlines() if not ln.startswith("room:")
    )
    rec.path.write_text(text, encoding="utf-8")
    loaded = store.read(rec.path)
    assert loaded.room == "common"


# ---------------------------------------------------------------------------
# db: migration and filtering
# ---------------------------------------------------------------------------

def test_db_migration_adds_room_column(tmp_path):
    db_path = tmp_path / "index.db"
    # Manually build a v0.2-equivalent legacy schema
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO meta VALUES ('dim', '8')")
    conn.execute(
        """CREATE TABLE memories (
            id TEXT PRIMARY KEY, path TEXT, type TEXT, content_hash TEXT,
            created_at REAL, importance INTEGER, tier TEXT)"""
    )
    conn.execute(
        "INSERT INTO memories VALUES ('m1', 'p', 'knowledge', 'h', 0, 5, 'hot')"
    )
    conn.commit()
    conn.close()

    db = IndexDB(db_path, 8)
    mem = db.get_memory("m1")
    assert mem["room"] == "common"
    db.close()


def _make_engine(tmp_path) -> MemoryEngine:
    settings = Settings(
        memories_dir=tmp_path / "memories",
        data_dir=tmp_path / "data",
    )
    embedder = FakeEmbedder(dim=64)
    store = MarkdownStore(settings.memories_dir)
    db = IndexDB(settings.db_path, embedder.dim)
    return MemoryEngine(settings=settings, store=store, db=db,
                        embedder=embedder)


def test_db_room_filters(tmp_path):
    engine = _make_engine(tmp_path)
    engine.remember("仕事の記憶アルファ", "knowledge", 5, room="work")
    engine.remember("個人の記憶ベータ", "knowledge", 5, room="personal")
    engine.remember("共通の記憶ガンマ", "knowledge", 5)  # room omitted = common

    rows = engine.db.all_memories(rooms=["work"])
    assert {r["room"] for r in rows} == {"work"}
    rows = engine.db.all_memories(rooms=["work", "common"])
    assert {r["room"] for r in rows} == {"work", "common"}


def test_recall_room_isolation(tmp_path):
    engine = _make_engine(tmp_path)
    engine.remember("予算要求の書式は財務課の様式7を使う", "knowledge", 5,
                    room="work")
    engine.remember("動画のサムネイルはピンク基調にする", "knowledge", 5,
                    room="personal")
    engine.remember("文章は敬体で書く", "preference", 5)  # common

    # From the "work" room, "personal" is not visible (but "common" is)
    result = engine.recall("予算要求の書式", room="work", record_hits=False)
    rooms = {h["room"] for h in result["hits"]}
    assert "personal" not in rooms

    result = engine.recall("サムネイルの色", room="work", record_hits=False)
    ids_rooms = {h["room"] for h in result["hits"]}
    assert "personal" not in ids_rooms

    # No room specified (None) still sees everything, as before
    result = engine.recall("サムネイルの色", record_hits=False)
    assert any(h["room"] == "personal" for h in result["hits"])

    # room="*" also means every room
    result = engine.recall("サムネイルの色", room="*", record_hits=False)
    assert any(h["room"] == "personal" for h in result["hits"])


def test_remember_dup_detection_is_room_scoped(tmp_path):
    engine = _make_engine(tmp_path)
    r1 = engine.remember("全く同じ内容の記憶", "knowledge", 5, room="work")
    assert r1["status"] == "created"
    # Same room -> duplicate reinforcement
    r2 = engine.remember("全く同じ内容の記憶", "knowledge", 5, room="work")
    assert r2["status"] == "duplicate_reinforced"
    # Different room -> not merged (context isolation)
    r3 = engine.remember("全く同じ内容の記憶", "knowledge", 5, room="personal")
    assert r3["status"] == "created"


def test_correct_inherits_room(tmp_path):
    engine = _make_engine(tmp_path)
    r1 = engine.remember("締切は金曜日", "knowledge", 5, room="work")
    res = engine.correct(r1["id"], "締切は木曜日", "カレンダー確認で判明",
                         source="test")
    new_mem = engine.db.get_memory(res["new_id"])
    assert new_mem["room"] == "work"


def test_reindex_preserves_room(tmp_path):
    engine = _make_engine(tmp_path)
    r1 = engine.remember("部屋付きの記憶", "knowledge", 5, room="work")
    engine.db.delete_memory(r1["id"])  # remove from DB so it gets rebuilt
    engine.reindex()
    mem = engine.db.get_memory(r1["id"])
    assert mem is not None
    assert mem["room"] == "work"
