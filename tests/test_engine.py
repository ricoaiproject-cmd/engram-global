"""Unit tests for engine.py.

Uses FakeEmbedder + mocked db/store + injected now to control time.
db / store are NotImplementedError stubs, so they are replaced with unittest.mock.
Once db/store are complete in the integration phase, the mocks can be removed
to promote these into end-to-end tests.
"""

from __future__ import annotations

import time
import math
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from engram.config import Settings
from engram.embedder import FakeEmbedder
from engram.engine import MemoryEngine, _hit_to_dict, _hybrid_relevances, build_engine
from engram.models import MemoryRecord, RecallHit

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DAY = 86400.0
NOW = 1_750_000_000.0  # fixed "current time"


def _settings(tmp_path: Path) -> Settings:
    """Test Settings (uses a temp directory)."""
    return Settings(
        memories_dir=tmp_path / "memories",
        data_dir=tmp_path / "data",
        dup_threshold=0.92,
        deep_score_threshold=0.35,
        candidate_k=20,
        correction_min_importance=7,
        consolidate_min_age_days=14,
        consolidate_cluster_sim=0.75,
        skill_min_count=3,
        skill_cluster_sim=0.80,
        colink_increment=0.1,
        colink_max=1.0,
        reinforce_weight=1.0,
        reinforce_strength_max=3.0,
        recall_hit_weight=0.3,
        create_alpha=2.0,
        w_relevance=0.6,
        w_activation=0.25,
        w_importance=0.15,
    )


def _fake_record(
    id: str = "01ABC",
    type: str = "knowledge",
    importance: int = 5,
    tier: str = "hot",
    content: str = "テスト記憶",
    tags: list[str] | None = None,
    path: str = "/tmp/test.md",
    content_hash: str | None = None,
) -> MemoryRecord:
    """Dummy MemoryRecord for tests. If content_hash is not given, it is auto-generated from content."""
    import hashlib
    h = content_hash if content_hash is not None else hashlib.sha256(content.strip().encode()).hexdigest()
    return MemoryRecord(
        id=id,
        type=type,
        created="2026-06-11T09:00:00+09:00",
        importance=importance,
        tags=tags or [],
        source="test",
        tier=tier,
        links=[],
        content=content,
        path=Path(path),
        content_hash=h,
    )


def _fake_db_mem(
    id: str = "01ABC",
    type: str = "knowledge",
    importance: int = 5,
    tier: str = "hot",
    path: str = "/tmp/test.md",
    content_hash: str = "",
    created_at: float = NOW - DAY,
) -> dict:
    """The dict shape returned by DB's get_memory / all_memories."""
    return {
        "id": id,
        "type": type,
        "importance": importance,
        "tier": tier,
        "path": path,
        "content_hash": content_hash,
        "created_at": created_at,
    }


def _build_engine(tmp_path: Path, *, embedder=None):
    """Helper that builds an engine with mocked db / store."""
    settings = _settings(tmp_path)
    embedder = embedder or FakeEmbedder(dim=64)
    db = MagicMock()
    store = MagicMock()

    # Default behavior (needed by most tests)
    db.vector_search.return_value = []
    db.keyword_search.return_value = []
    db.get_events.return_value = {}
    db.all_memories.return_value = []
    db.get_links.return_value = []
    db.get_embeddings.return_value = {}
    db.get_memory.return_value = None

    return MemoryEngine(settings=settings, store=store, db=db, embedder=embedder), db, store


# ---------------------------------------------------------------------------
# remember / recall round-trip tests
# ---------------------------------------------------------------------------

class TestRememberRecall:
    def test_remember_then_recall(self, tmp_path):
        """A memory saved via remember should be retrievable via recall."""
        engine, db, store = _build_engine(tmp_path)

        # Mock store.create
        record = _fake_record(id="MEM001", content="Python の非同期処理について")
        store.create.return_value = record
        db.vector_search.return_value = []  # no duplicate

        result = engine.remember(
            "Python の非同期処理について",
            type="knowledge",
            importance=7,
            now=NOW,
        )
        assert result["status"] == "created"
        assert result["id"] == "MEM001"

        # Confirm db.upsert_memory was called
        db.upsert_memory.assert_called_once()

        # Confirm a create event was recorded
        db.add_event.assert_called_once()
        args = db.add_event.call_args
        assert args[0][0] == "MEM001"
        assert args[0][1] == "create"

        # recall setup: return this node
        db.vector_search.return_value = [("MEM001", 0.95)]
        db.keyword_search.return_value = [("MEM001", -1.0)]
        db.all_memories.return_value = [_fake_db_mem(
            id="MEM001", importance=7, created_at=NOW - DAY
        )]
        db.get_events.return_value = {
            "MEM001": [(NOW - 100, 2.4)]  # equivalent to a create event
        }
        store.read.return_value = record

        recall_result = engine.recall("Python 非同期", now=NOW)
        assert recall_result["mode"] == "fast"
        hits = recall_result["hits"]
        assert len(hits) > 0
        assert hits[0]["id"] == "MEM001"

    def test_remember_stores_create_event_weight(self, tmp_path):
        """A create event should be recorded with the initial encoding boost scaled to importance."""
        engine, db, store = _build_engine(tmp_path)

        record = _fake_record(id="MEM002", importance=10)
        store.create.return_value = record
        db.vector_search.return_value = []

        engine.remember("重要な記憶", type="knowledge", importance=10, now=NOW)

        # create_event_weight(10, alpha=2.0) = 1 + 2.0 * (10/10) = 3.0
        event_call = db.add_event.call_args
        assert pytest.approx(event_call[0][2], abs=1e-6) == 3.0


# ---------------------------------------------------------------------------
# Duplicate detection tests
# ---------------------------------------------------------------------------

class TestDuplicate:
    def test_duplicate_reinforced(self, tmp_path):
        """Should return duplicate_reinforced when cos >= dup_threshold."""
        engine, db, store = _build_engine(tmp_path)

        # An existing memory is returned with an almost identical vector
        db.vector_search.return_value = [("EXIST001", 0.95)]  # >= 0.92

        result = engine.remember(
            "Pythonの非同期処理",
            type="knowledge",
            importance=5,
            now=NOW,
        )

        assert result["status"] == "duplicate_reinforced"
        assert result["id"] == "EXIST001"
        # store.create should not be called
        store.create.assert_not_called()
        # A reinforce event is recorded
        db.add_event.assert_called_once()
        assert db.add_event.call_args[0][1] == "reinforce"

    def test_no_duplicate_below_threshold(self, tmp_path):
        """Should create a new memory when cos < dup_threshold."""
        engine, db, store = _build_engine(tmp_path)

        db.vector_search.return_value = [("EXIST001", 0.85)]  # < 0.92
        record = _fake_record(id="NEW001")
        store.create.return_value = record

        result = engine.remember("全然違う内容", type="knowledge", importance=5, now=NOW)
        assert result["status"] == "created"
        store.create.assert_called_once()


# ---------------------------------------------------------------------------
# Tests that rank rises after reinforce
# ---------------------------------------------------------------------------

class TestReinforce:
    def test_reinforce_raises_score(self, tmp_path):
        """The recall score should rise after reinforce."""
        engine, db, store = _build_engine(tmp_path)

        mem = _fake_db_mem(id="MEM_HOT", importance=5, created_at=NOW - 7 * DAY)
        db.all_memories.return_value = [mem]
        db.vector_search.return_value = [("MEM_HOT", 0.8)]
        db.keyword_search.return_value = []
        record = _fake_record(id="MEM_HOT", content="強化される記憶")
        store.read.return_value = record

        # Score before reinforce
        db.get_events.return_value = {"MEM_HOT": [(NOW - 7 * DAY, 1.5)]}
        result_before = engine.recall("強化", now=NOW, record_hits=False)
        score_before = result_before["hits"][0]["score"] if result_before["hits"] else 0.0

        # Add a reinforce event and recompute the score
        db.get_memory.return_value = mem
        engine.reinforce(["MEM_HOT"], strength=2.0, now=NOW)

        # Check the score with the post-reinforce event
        db.get_events.return_value = {
            "MEM_HOT": [
                (NOW - 7 * DAY, 1.5),
                (NOW, 2.0),  # reinforce weight * strength
            ]
        }
        result_after = engine.recall("強化", now=NOW + 1, record_hits=False)
        score_after = result_after["hits"][0]["score"] if result_after["hits"] else 0.0

        assert score_after >= score_before

    def test_reinforce_unknown_ids(self, tmp_path):
        """A nonexistent id should be listed in unknown_ids."""
        engine, db, store = _build_engine(tmp_path)
        db.get_memory.return_value = None

        result = engine.reinforce(["GHOST001", "GHOST002"])
        assert set(result["unknown_ids"]) == {"GHOST001", "GHOST002"}
        assert result["reinforced"] == []

    def test_reinforce_creates_colink(self, tmp_path):
        """Reinforcing multiple ids at once should create co_recall links."""
        engine, db, store = _build_engine(tmp_path)
        mem_a = _fake_db_mem(id="A")
        mem_b = _fake_db_mem(id="B")
        db.get_memory.side_effect = lambda id_: {"A": mem_a, "B": mem_b}.get(id_)

        engine.reinforce(["A", "B"], now=NOW)

        # co_recall links are created in both directions
        link_calls = db.add_link.call_args_list
        kinds = [c[0][2] for c in link_calls]
        assert kinds.count("co_recall") == 2


# ---------------------------------------------------------------------------
# correct flow tests
# ---------------------------------------------------------------------------

class TestCorrect:
    def _setup_correct(self, tmp_path):
        engine, db, store = _build_engine(tmp_path)
        old_record = _fake_record(
            id="OLD001", content="誤った情報", importance=5, tier="hot"
        )
        db.get_memory.return_value = _fake_db_mem(
            id="OLD001", importance=5, tier="hot", path="/tmp/old.md"
        )
        store.read.return_value = old_record

        new_record = _fake_record(id="NEW001", content="正しい情報", importance=7)
        store.create.return_value = new_record
        db.vector_search.return_value = []  # no duplicate

        return engine, db, store, old_record, new_record

    def test_correct_creates_new_and_supersedes_old(self, tmp_path):
        """correct should mark the old memory as superseded and create a new one."""
        engine, db, store, old_record, new_record = self._setup_correct(tmp_path)

        result = engine.correct(
            "OLD001",
            corrected_content="正しい情報",
            reason="検証で誤りが判明",
            now=NOW,
        )

        assert result["status"] == "corrected"
        assert result["old_id"] == "OLD001"
        new_id = result["new_id"]

        # The old memory becomes superseded
        db.set_tier.assert_called_with("OLD001", "superseded")
        store.set_tier.assert_called_with(old_record, "superseded")

        # A superseded_by link is created
        link_calls = db.add_link.call_args_list
        superseded_calls = [c for c in link_calls if c[0][2] == "superseded_by"]
        assert len(superseded_calls) >= 1
        assert superseded_calls[0][0][0] == "OLD001"

    def test_correct_new_memory_has_correction_tag(self, tmp_path):
        """The new memory created by correct should carry the correction tag."""
        engine, db, store, _, _ = self._setup_correct(tmp_path)
        engine.correct("OLD001", "正しい情報", "理由", now=NOW)

        # store.create's arguments should include tags: ["correction", ...]
        create_kwargs = store.create.call_args[1]
        assert "correction" in create_kwargs.get("tags", [])

    def test_correct_new_importance_raised(self, tmp_path):
        """The corrected memory's importance should be at least correction_min_importance."""
        engine, db, store, old_record, _ = self._setup_correct(tmp_path)
        # Correct a low-importance (importance=3) memory
        old_record.importance = 3
        db.get_memory.return_value = _fake_db_mem(id="OLD001", importance=3)

        engine.correct("OLD001", "正しい情報", "理由", now=NOW)

        create_kwargs = store.create.call_args[1]
        assert create_kwargs["importance"] >= 7  # correction_min_importance

    def test_correct_not_found(self, tmp_path):
        """Calling correct on a nonexistent id should return not_found."""
        engine, db, store, _, _ = self._setup_correct(tmp_path)
        db.get_memory.return_value = None

        result = engine.correct("GHOST", "正しい内容", "理由")
        assert result["status"] == "not_found"

    def test_correct_old_not_in_fast_recall(self, tmp_path):
        """After correction, the old (superseded) memory should not appear in fast recall (tier=hot only)."""
        engine, db, store, _, _ = self._setup_correct(tmp_path)
        # fast recall only targets tier=hot
        # a superseded memory is not included in all_memories(tiers=["hot"])
        db.all_memories.return_value = []
        db.vector_search.return_value = []
        db.keyword_search.return_value = []

        result = engine.recall("誤った情報", mode="fast", now=NOW)
        hit_ids = [h["id"] for h in result["hits"]]
        assert "OLD001" not in hit_ids

    def test_correct_deep_note_on_superseded(self, tmp_path):
        """A superseded memory should get a note attached in deep recall."""
        engine, db, store, old_record, _ = self._setup_correct(tmp_path)

        # deep recall setup: return the superseded OLD001
        old_mem = _fake_db_mem(id="OLD001", tier="superseded", importance=5)
        db.all_memories.return_value = [old_mem]
        db.vector_search.return_value = [("OLD001", 0.8)]
        db.keyword_search.return_value = []
        db.get_links.return_value = [("OLD001", "NEW001", "superseded_by", 1.0)]
        db.get_embeddings.return_value = {}
        old_record.tier = "superseded"
        store.read.return_value = old_record

        result = engine.recall("誤った情報", mode="deep", now=NOW)
        old_hits = [h for h in result["hits"] if h["id"] == "OLD001"]
        if old_hits:
            assert "NEW001" in old_hits[0]["note"]


# ---------------------------------------------------------------------------
# co_recall link formation tests
# ---------------------------------------------------------------------------

class TestCoRecallLink:
    def test_co_recall_links_formed_on_reinforce(self, tmp_path):
        """Reinforcing 3 items at once should form co_recall links for 3 pairs."""
        engine, db, store = _build_engine(tmp_path)

        ids = ["A", "B", "C"]
        for id_ in ids:
            db.get_memory.side_effect = lambda i: _fake_db_mem(id=i)

        # Actually use a dict-based side_effect
        mem_map = {i: _fake_db_mem(id=i) for i in ids}
        db.get_memory.side_effect = lambda i: mem_map.get(i)

        engine.reinforce(ids, now=NOW)

        # A-B, A-C, B-C: 3 pairs x 2 directions = 6 co_recall links
        link_calls = db.add_link.call_args_list
        co_recall_calls = [(c[0][0], c[0][1]) for c in link_calls if c[0][2] == "co_recall"]
        assert len(co_recall_calls) == 6


# ---------------------------------------------------------------------------
# forget tests
# ---------------------------------------------------------------------------

class TestForget:
    def test_forget_sets_trash_tier(self, tmp_path):
        """forget should move the file to trash and set DB tier=trash."""
        engine, db, store = _build_engine(tmp_path)

        record = _fake_record(id="DEL001")
        db.get_memory.return_value = _fake_db_mem(id="DEL001")
        store.read.return_value = record

        result = engine.forget("DEL001")

        assert result["status"] == "forgotten"
        store.move_to_trash.assert_called_once_with(record)
        db.set_tier.assert_called_once_with("DEL001", "trash")

    def test_forget_not_found(self, tmp_path):
        """Calling forget on a nonexistent id should return not_found."""
        engine, db, store = _build_engine(tmp_path)
        db.get_memory.return_value = None

        result = engine.forget("GHOST")
        assert result["status"] == "not_found"


# ---------------------------------------------------------------------------
# consolidation_candidates clustering tests
# ---------------------------------------------------------------------------

class TestConsolidationCandidates:
    def test_clusters_similar_episodes(self, tmp_path):
        """Episodes with similar embeddings should be clustered."""
        engine, db, store = _build_engine(tmp_path)
        embedder = engine.embedder

        # Two episodes with identical text (identical embedding -> cos=1.0)
        text = "今日の進捗: PR レビュー"
        ep_ids = ["EP001", "EP002"]
        ep_mems = [
            _fake_db_mem(
                id=eid,
                type="episode",
                tier="hot",
                created_at=NOW - 20 * DAY,  # older than min_age_days=14
            )
            for eid in ep_ids
        ]
        db.all_memories.return_value = ep_mems

        # embeddings: identical vector (cos=1.0 >= 0.75)
        vec = embedder.embed_docs([text])[0]
        db.get_embeddings.return_value = {eid: vec for eid in ep_ids}

        for eid in ep_ids:
            store.read.return_value = _fake_record(id=eid, content=text)

        result = engine.consolidation_candidates(now=NOW)
        clusters = result["clusters"]
        assert len(clusters) >= 1
        assert len(clusters[0]["ids"]) == 2

    def test_no_cluster_for_young_episodes(self, tmp_path):
        """Episodes newer than min_age_days should not appear as cluster candidates."""
        engine, db, store = _build_engine(tmp_path)
        embedder = engine.embedder

        text = "今日の進捗"
        ep_ids = ["EP_NEW1", "EP_NEW2"]
        # Set created_at to NOW - 3 days (< 14 days)
        ep_mems = [
            _fake_db_mem(
                id=eid,
                type="episode",
                tier="hot",
                created_at=NOW - 3 * DAY,
            )
            for eid in ep_ids
        ]
        db.all_memories.return_value = ep_mems

        vec = embedder.embed_docs([text])[0]
        db.get_embeddings.return_value = {eid: vec for eid in ep_ids}

        result = engine.consolidation_candidates(now=NOW)
        assert result["clusters"] == []


# ---------------------------------------------------------------------------
# skill_candidates clustering tests
# ---------------------------------------------------------------------------

class TestSkillCandidates:
    def test_clusters_three_similar_episodes(self, tmp_path):
        """Once 3 similar episodes (skill_min_count) accumulate, they should be returned as a cluster."""
        engine, db, store = _build_engine(tmp_path)
        embedder = engine.embedder

        text = "月次の起案文書チェックを実施した"
        ep_ids = ["EP001", "EP002", "EP003"]
        ep_mems = [
            _fake_db_mem(id=eid, type="episode", tier="hot", created_at=NOW)
            for eid in ep_ids
        ]
        db.all_memories.return_value = ep_mems

        vec = embedder.embed_docs([text])[0]
        db.get_embeddings.return_value = {eid: vec for eid in ep_ids}
        for eid in ep_ids:
            store.read.return_value = _fake_record(id=eid, content=text)

        result = engine.skill_candidates(now=NOW)
        clusters = result["clusters"]
        assert len(clusters) == 1
        assert set(clusters[0]["ids"]) == set(ep_ids)

    def test_no_cluster_below_skill_min_count(self, tmp_path):
        """A cluster with only 2 items is below skill_min_count (default 3), so it should not be returned."""
        engine, db, store = _build_engine(tmp_path)
        embedder = engine.embedder

        text = "月次の起案文書チェックを実施した"
        ep_ids = ["EP001", "EP002"]
        ep_mems = [
            _fake_db_mem(id=eid, type="episode", tier="hot", created_at=NOW)
            for eid in ep_ids
        ]
        db.all_memories.return_value = ep_mems

        vec = embedder.embed_docs([text])[0]
        db.get_embeddings.return_value = {eid: vec for eid in ep_ids}

        result = engine.skill_candidates(now=NOW)
        assert result["clusters"] == []

    def test_no_cluster_below_similarity_threshold(self, tmp_path):
        """Pairs with cosine similarity below skill_cluster_sim should not form a cluster."""
        engine, db, store = _build_engine(tmp_path)

        ep_ids = ["EP001", "EP002", "EP003"]
        ep_mems = [
            _fake_db_mem(id=eid, type="episode", tier="hot", created_at=NOW)
            for eid in ep_ids
        ]
        db.all_memories.return_value = ep_mems

        # Prepare near-orthogonal vectors (cos close to 0, below skill_cluster_sim=0.80)
        dim = engine.embedder.dim
        vecs = {}
        for i, eid in enumerate(ep_ids):
            v = np.zeros(dim, dtype=np.float32)
            v[i % dim] = 1.0
            vecs[eid] = v
        db.get_embeddings.return_value = vecs

        result = engine.skill_candidates(now=NOW)
        assert result["clusters"] == []

    def test_young_episode_is_still_candidate(self, tmp_path):
        """Unlike consolidation, there is no age filter (should be a candidate even right after creation)."""
        engine, db, store = _build_engine(tmp_path)
        embedder = engine.embedder

        text = "決算資料の数値突合を実施した"
        ep_ids = ["EP_NEW1", "EP_NEW2", "EP_NEW3"]
        # Set created_at to NOW (right after creation, well below consolidate_min_age_days=14)
        ep_mems = [
            _fake_db_mem(id=eid, type="episode", tier="hot", created_at=NOW)
            for eid in ep_ids
        ]
        db.all_memories.return_value = ep_mems

        vec = embedder.embed_docs([text])[0]
        db.get_embeddings.return_value = {eid: vec for eid in ep_ids}
        for eid in ep_ids:
            store.read.return_value = _fake_record(id=eid, content=text)

        result = engine.skill_candidates(now=NOW)
        assert len(result["clusters"]) == 1
        assert set(result["clusters"][0]["ids"]) == set(ep_ids)

    def test_cold_tier_episodes_excluded(self, tmp_path):
        """Confirm that tier=cold episodes never even reach us because
        db.all_memories(tiers=["hot"]) already filters them out (only hot is targeted)."""
        engine, db, store = _build_engine(tmp_path)

        # Configure db.all_memories to return empty for a tiers=["hot"] call
        # (mocks the assumption that cold memories are invisible to the engine)
        db.all_memories.return_value = []

        result = engine.skill_candidates(now=NOW)
        assert result["clusters"] == []
        # Confirm the query was restricted to tier=hot
        db.all_memories.assert_called_with(tiers=["hot"], types=["episode"])


# ---------------------------------------------------------------------------
# mark_consolidated tests
# ---------------------------------------------------------------------------

class TestMarkConsolidated:
    def test_mark_consolidated_demotes_to_cold(self, tmp_path):
        """mark_consolidated should demote episodes to cold and create derived_from links."""
        engine, db, store = _build_engine(tmp_path)

        ep_ids = ["EP001", "EP002"]
        new_id = "SUMMARY001"

        ep_records = {eid: _fake_record(id=eid, type="episode") for eid in ep_ids}
        ep_mems = {eid: _fake_db_mem(id=eid, type="episode") for eid in ep_ids}

        db.get_memory.side_effect = lambda i: ep_mems.get(i)
        store.read.side_effect = lambda p: next(
            (r for r in ep_records.values() if str(r.path) == str(p)),
            ep_records[ep_ids[0]],
        )

        result = engine.mark_consolidated(ep_ids, new_id)

        assert result["status"] == "ok"
        assert set(result["consolidated"]) == set(ep_ids)

        # Each episode is demoted to cold
        tier_calls = [c for c in db.set_tier.call_args_list]
        for eid in ep_ids:
            assert any(c[0] == (eid, "cold") for c in tier_calls)

        # A derived_from link is created
        link_calls = db.add_link.call_args_list
        df_calls = [(c[0][0], c[0][1], c[0][2]) for c in link_calls if c[0][2] == "derived_from"]
        for eid in ep_ids:
            assert any(c[0] == eid and c[1] == new_id for c in df_calls)


# ---------------------------------------------------------------------------
# reindex (manual-edit detection) tests
# ---------------------------------------------------------------------------

class TestReindex:
    def test_reindex_detects_new_file(self, tmp_path):
        """Markdown not present in the DB should be counted as added."""
        engine, db, store = _build_engine(tmp_path)

        new_record = _fake_record(id="NEW_FILE", content_hash="abc")
        store.scan_all.return_value = iter([new_record])
        db.get_memory.return_value = None  # not in the DB
        db.all_memories.return_value = []

        result = engine.reindex()
        assert result["added"] == 1
        assert result["updated"] == 0
        assert result["removed"] == 0
        assert result["unchanged"] == 0
        db.upsert_memory.assert_called_once()

    def test_reindex_detects_edit(self, tmp_path):
        """A content_hash mismatch should be counted as updated."""
        engine, db, store = _build_engine(tmp_path)

        record = _fake_record(id="EDIT001", content_hash="new_hash")
        store.scan_all.return_value = iter([record])
        # The DB has the old hash recorded
        db.get_memory.return_value = _fake_db_mem(
            id="EDIT001", content_hash="old_hash"
        )
        db.all_memories.return_value = [_fake_db_mem(id="EDIT001")]

        result = engine.reindex()
        assert result["updated"] == 1
        assert result["added"] == 0
        assert result["unchanged"] == 0

    def test_reindex_removes_orphan(self, tmp_path):
        """A DB entry whose file has disappeared should be counted as removed."""
        engine, db, store = _build_engine(tmp_path)

        # No file in store, but ORPHAN remains in the DB
        store.scan_all.return_value = iter([])
        db.all_memories.return_value = [_fake_db_mem(id="ORPHAN")]

        result = engine.reindex()
        assert result["removed"] == 1
        db.delete_memory.assert_called_once_with("ORPHAN")

    def test_reindex_unchanged(self, tmp_path):
        """A matching hash should be counted as unchanged."""
        engine, db, store = _build_engine(tmp_path)

        record = _fake_record(id="SAME001", content_hash="same_hash")
        store.scan_all.return_value = iter([record])
        db.get_memory.return_value = _fake_db_mem(
            id="SAME001", content_hash="same_hash"
        )
        db.all_memories.return_value = [_fake_db_mem(id="SAME001")]

        result = engine.reindex()
        assert result["unchanged"] == 1
        assert result["added"] == 0
        assert result["updated"] == 0


# ---------------------------------------------------------------------------
# auto_deepened tests
# ---------------------------------------------------------------------------

class TestAutoDeepened:
    def test_auto_deepened_when_low_score(self, tmp_path):
        """auto_deepened should be True when the fast score is below deep_score_threshold."""
        engine, db, store = _build_engine(tmp_path)

        # A situation where the score becomes very low: low similarity
        mem = _fake_db_mem(id="COLD_MEM", importance=1, created_at=NOW - 365 * DAY)
        db.all_memories.return_value = [mem]
        db.vector_search.return_value = [("COLD_MEM", 0.1)]  # low similarity
        db.keyword_search.return_value = []
        db.get_events.return_value = {}  # no events -> activation 0
        db.get_links.return_value = []
        db.get_embeddings.return_value = {}
        record = _fake_record(id="COLD_MEM", importance=1, content="関係なさそうな記憶")
        store.read.return_value = record

        result = engine.recall("全く関係ないクエリ", mode="fast", now=NOW)

        # final_score(0.1, 0, 1) = 0.6*0.1 + 0.25*0 + 0.15*(1/10) = 0.075 < 0.35
        assert result["auto_deepened"] is True
        assert result["mode"] == "deep"

    def test_no_auto_deepened_when_high_score(self, tmp_path):
        """auto_deepened should be False when the fast score is at or above deep_score_threshold."""
        engine, db, store = _build_engine(tmp_path)

        mem = _fake_db_mem(id="HOT_MEM", importance=9, created_at=NOW - DAY)
        db.all_memories.return_value = [mem]
        db.vector_search.return_value = [("HOT_MEM", 0.98)]  # high similarity
        db.keyword_search.return_value = [("HOT_MEM", -1.0)]
        # There is a reinforce event -> high activation
        db.get_events.return_value = {
            "HOT_MEM": [(NOW - DAY, 3.0), (NOW - 2 * DAY, 2.0)]
        }
        record = _fake_record(id="HOT_MEM", importance=9, content="非常に関連性の高い記憶")
        store.read.return_value = record

        result = engine.recall("関連性の高いクエリ", mode="fast", now=NOW)

        assert result["auto_deepened"] is False
        assert result["mode"] == "fast"


# ---------------------------------------------------------------------------
# Tests for link-only nodes in deep recall
# ---------------------------------------------------------------------------

class TestDeepRecall:
    def test_associative_via_link(self, tmp_path):
        """In deep recall, a node reached only via a link should have via=associative."""
        engine, db, store = _build_engine(tmp_path)

        # DIRECT: hit by vector search
        # ASSOC: not hit by vector search, but connected from DIRECT via a co_recall link
        direct_mem = _fake_db_mem(id="DIRECT", importance=5, created_at=NOW - DAY)
        assoc_mem = _fake_db_mem(id="ASSOC", importance=5, created_at=NOW - DAY)

        db.all_memories.return_value = [direct_mem, assoc_mem]
        db.vector_search.return_value = [("DIRECT", 0.85)]
        db.keyword_search.return_value = []
        db.get_events.return_value = {}

        # A co_recall link from DIRECT to ASSOC
        db.get_links.return_value = [("DIRECT", "ASSOC", "co_recall", 0.9)]

        # Return ASSOC's embedding (for relevance calculation)
        embedder = engine.embedder
        assoc_vec = embedder.embed_docs(["連想記憶"])[0]
        db.get_embeddings.return_value = {"ASSOC": assoc_vec}

        direct_record = _fake_record(id="DIRECT", content="直接ヒットした記憶")
        assoc_record = _fake_record(id="ASSOC", content="連想で到達した記憶")
        store.read.side_effect = lambda p: (
            assoc_record if "ASSOC" in str(p) else direct_record
        )

        result = engine.recall("クエリ", mode="deep", now=NOW)

        hit_map = {h["id"]: h for h in result["hits"]}
        if "ASSOC" in hit_map:
            assert hit_map["ASSOC"]["via"] == "associative"
        if "DIRECT" in hit_map:
            assert hit_map["DIRECT"]["via"] == "direct"


# ---------------------------------------------------------------------------
# Tests that the correction tag raises importance
# ---------------------------------------------------------------------------

class TestCorrectionTag:
    def test_correction_tag_raises_importance(self, tmp_path):
        """When tags includes correction, importance should be raised."""
        engine, db, store = _build_engine(tmp_path)

        record = _fake_record(id="CORR001", importance=7)
        store.create.return_value = record
        db.vector_search.return_value = []

        # importance=3 with the correction tag
        engine.remember(
            "訂正情報",
            type="knowledge",
            importance=3,
            tags=["correction"],
            now=NOW,
        )

        # The importance passed to store.create should be at least 7
        create_kwargs = store.create.call_args[1]
        assert create_kwargs["importance"] >= 7


# ---------------------------------------------------------------------------
# exhaustive recall (surfacing sunken memories) tests
# ---------------------------------------------------------------------------


class _FixedQueryEmbedder:
    """A stub whose embed_query always returns a fixed vector (for precise control of relevance)."""

    def __init__(self, qvec):
        self._q = np.asarray(qvec, dtype=np.float32)
        self.dim = len(self._q)

    def embed_query(self, text):
        return self._q

    def embed_docs(self, texts):
        return [self._q for _ in texts]


class TestExhaustiveRecall:
    def test_exhaustive_ranks_by_relevance_ignoring_activation(self, tmp_path):
        """mode=exhaustive ignores activation and ranks purely by relevance:
        a sunken but highly relevant memory should surface, while a memory
        whose relevance is below the floor should be excluded."""
        embedder = _FixedQueryEmbedder([1.0, 0.0, 0.0, 0.0])
        engine, db, store = _build_engine(tmp_path, embedder=embedder)

        # R: sunken (only one old event) but max relevance, G: recent and highly
        # active but mid relevance, L: zero relevance (excluded, below floor)
        r = _fake_db_mem(id="R", type="knowledge", importance=1,
                         created_at=NOW - 1000 * DAY, path="/tmp/R.md")
        g = _fake_db_mem(id="G", type="preference", importance=9,
                         created_at=NOW - DAY, path="/tmp/G.md")
        ll = _fake_db_mem(id="L", type="knowledge", importance=5,
                          created_at=NOW - DAY, path="/tmp/L.md")
        db.all_memories.return_value = [r, g, ll]
        db.get_embeddings.return_value = {
            "R": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),   # cos 1.0
            "G": np.asarray([0.6, 0.8, 0.0, 0.0], dtype=np.float32),   # cos 0.6
            "L": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),   # cos 0.0
        }
        db.get_events.return_value = {
            "R": [(NOW - 1000 * DAY, 1.0)],                 # sunken
            "G": [(NOW - DAY, 3.0), (NOW - 2 * DAY, 2.0)],  # highly active
            "L": [(NOW - DAY, 1.0)],
        }
        store.read.side_effect = lambda p: _fake_record(
            id=Path(p).stem, content=f"{Path(p).stem} の本文"
        )

        result = engine.recall("ターゲット", mode="exhaustive", now=NOW)

        assert result["mode"] == "exhaustive"
        hits = result["hits"]
        ids = [h["id"] for h in hits]
        # R is top (relevance 1.0), followed by G. L is excluded (below the floor).
        assert ids[0] == "R"
        assert "G" in ids
        assert "L" not in ids
        hit_map = {h["id"]: h for h in hits}
        # Sunken R ranks above the more active G (evidence activation is ignored)
        assert hit_map["R"]["activation"] < hit_map["G"]["activation"]
        assert hit_map["R"]["via"] == "exhaustive"
        assert hit_map["R"]["relevance"] == pytest.approx(1.0, abs=1e-5)

    def test_fast_buries_what_exhaustive_surfaces(self, tmp_path):
        """Contrast: the sunken but highly relevant memory R loses the top spot
        to the highly active G under fast only when the relevance gap is
        compressed below the normalization floor (relevance_norm_floor).

        Previously, even a large relevance gap (1.0 vs 0.6) could be reversed
        by the activation boost, but with the introduction of within-candidate
        min-max normalization (2026-07-06), a clear relevance gap now wins.
        fast defers to activation for ranking only when relevance is nearly
        indistinguishable across candidates and the query is not discriminative.
        Surfacing such cases is exhaustive's job."""
        embedder = _FixedQueryEmbedder([1.0, 0.0, 0.0, 0.0])
        engine, db, store = _build_engine(tmp_path, embedder=embedder)

        r = _fake_db_mem(id="R", type="knowledge", importance=1,
                         created_at=NOW - 1000 * DAY, path="/tmp/R.md")
        g = _fake_db_mem(id="G", type="preference", importance=9,
                         created_at=NOW - DAY, path="/tmp/G.md")
        db.all_memories.return_value = [r, g]
        # Compression band: the 0.04 gap is below the floor (0.10) -> relevance is
        # not amplified, so activation decides the ranking
        db.vector_search.return_value = [("R", 0.84), ("G", 0.80)]
        db.keyword_search.return_value = []
        db.get_events.return_value = {
            "R": [(NOW - 1000 * DAY, 1.0)],
            "G": [(NOW - DAY, 3.0), (NOW - 2 * DAY, 2.0)],
        }
        store.read.side_effect = lambda p: _fake_record(
            id=Path(p).stem, content=f"{Path(p).stem} の本文"
        )

        result = engine.recall("ターゲット", mode="fast", now=NOW)

        # In an indistinguishable compression band, highly active G ranks first
        # (fallback to base-level activation)
        assert result["hits"][0]["id"] == "G"

    def test_fast_no_longer_buries_clear_relevance_gap(self, tmp_path):
        """Regression: when the relevance gap is clear (1.0 vs 0.6), the sunken
        R should beat the highly active G even under fast (this used to be
        reversed before normalization was introduced)."""
        embedder = _FixedQueryEmbedder([1.0, 0.0, 0.0, 0.0])
        engine, db, store = _build_engine(tmp_path, embedder=embedder)

        r = _fake_db_mem(id="R", type="knowledge", importance=1,
                         created_at=NOW - 1000 * DAY, path="/tmp/R.md")
        g = _fake_db_mem(id="G", type="preference", importance=9,
                         created_at=NOW - DAY, path="/tmp/G.md")
        db.all_memories.return_value = [r, g]
        db.vector_search.return_value = [("R", 1.0), ("G", 0.6)]
        db.keyword_search.return_value = []
        db.get_events.return_value = {
            "R": [(NOW - 1000 * DAY, 1.0)],
            "G": [(NOW - DAY, 3.0), (NOW - 2 * DAY, 2.0)],
        }
        store.read.side_effect = lambda p: _fake_record(
            id=Path(p).stem, content=f"{Path(p).stem} の本文"
        )

        result = engine.recall("ターゲット", mode="fast", now=NOW)

        assert result["hits"][0]["id"] == "R"

    def test_deep_auto_escalates_to_exhaustive(self, tmp_path):
        """Even under deep, when the top score is weak it should auto-escalate to
        exhaustive and surface a sunken, highly relevant memory."""
        embedder = _FixedQueryEmbedder([1.0, 0.0, 0.0, 0.0])
        engine, db, store = _build_engine(tmp_path, embedder=embedder)

        m = _fake_db_mem(id="M", type="knowledge", importance=5,
                         created_at=NOW - 1000 * DAY, path="/tmp/M.md")
        db.all_memories.return_value = [m]
        db.vector_search.return_value = [("M", 0.2)]   # low relevance under fast/deep
        db.keyword_search.return_value = []
        db.get_events.return_value = {}                # zero activation
        db.get_links.return_value = []
        # The real cosine recomputed under exhaustive is high (it's actually highly relevant)
        db.get_embeddings.return_value = {
            "M": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        }
        store.read.return_value = _fake_record(id="M", content="本来は関連の高い記憶")

        result = engine.recall("ターゲット", mode="fast", now=NOW)

        assert result["auto_deepened"] is True
        assert result["mode"] == "exhaustive"
        assert result["hits"][0]["id"] == "M"
        assert result["hits"][0]["relevance"] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Startup index-freshness check (multi-machine sharing safeguard)
# ---------------------------------------------------------------------------


class TestIndexFreshness:
    @staticmethod
    def _active(n):
        return [_fake_db_mem(id=f"M{i}") for i in range(n)]

    def test_in_sync_no_reindex(self, tmp_path):
        """Raw .md count matches index count -> in_sync, neither scan nor reindex runs."""
        engine, db, store = _build_engine(tmp_path)
        store.count_memory_files.return_value = 5
        db.all_memories.return_value = self._active(5)
        engine.reindex = MagicMock()

        res = engine.check_index_freshness(mode="auto")

        assert res["action"] == "in_sync"
        assert res["markdown"] == 5 and res["index"] == 5
        store.scan_all.assert_not_called()
        engine.reindex.assert_not_called()

    def test_phantom_md_counts_as_in_sync(self, tmp_path):
        """Even if raw .md outnumbers the index, no reindex happens if scan_all's
        valid count matches (an apparent gap caused by empty/broken/non-memory md)."""
        engine, db, store = _build_engine(tmp_path)
        store.count_memory_files.return_value = 6      # includes 1 phantom file
        db.all_memories.return_value = self._active(5)
        store.scan_all.return_value = [object()] * 5    # 5 valid memories
        engine.reindex = MagicMock()

        res = engine.check_index_freshness(mode="auto")

        assert res["action"] == "in_sync"
        assert res["valid"] == 5
        engine.reindex.assert_not_called()

    def test_auto_reindexes_on_real_drift(self, tmp_path):
        """More valid memories than the index (not yet ingested on another machine) -> auto reindexes."""
        engine, db, store = _build_engine(tmp_path)
        store.count_memory_files.return_value = 10
        db.all_memories.return_value = self._active(5)
        store.scan_all.return_value = [object()] * 10
        engine.reindex = MagicMock(return_value={
            "added": 5, "updated": 0, "removed": 0, "unchanged": 5})

        res = engine.check_index_freshness(mode="auto")

        assert res["action"] == "reindexed"
        assert res["valid"] == 10 and res["index"] == 5
        engine.reindex.assert_called_once()

    def test_warn_does_not_reindex(self, tmp_path):
        """warn mode only reports the drift and does not reindex."""
        engine, db, store = _build_engine(tmp_path)
        store.count_memory_files.return_value = 10
        db.all_memories.return_value = self._active(5)
        store.scan_all.return_value = [object()] * 10
        engine.reindex = MagicMock()

        res = engine.check_index_freshness(mode="warn")

        assert res["action"] == "warn"
        assert res["drift"] == 5
        engine.reindex.assert_not_called()

    def test_off_does_nothing(self, tmp_path):
        """off mode does not even count, and does nothing."""
        engine, db, store = _build_engine(tmp_path)
        store.count_memory_files.return_value = 10
        db.all_memories.return_value = self._active(5)
        engine.reindex = MagicMock()

        res = engine.check_index_freshness(mode="off")

        assert res["action"] == "off"
        store.count_memory_files.assert_not_called()
        engine.reindex.assert_not_called()


# ---------------------------------------------------------------------------
# _hybrid_relevances (hybrid-search relevance blending) tests
# ---------------------------------------------------------------------------

class TestHybridRelevances:
    def test_fts_only_rare_token_gives_high_relevance(self):
        """A rare token hit only by FTS (bm25≈-6) should become 1-exp(-6)≈0.9975."""
        rel = _hybrid_relevances([], [("ID1", -6.0)])
        assert rel == {"ID1": pytest.approx(0.9975212478233336, abs=1e-9)}

    def test_common_word_fts_gives_lower_relevance(self):
        """An FTS hit on a common word (bm25≈-0.5) should become 1-exp(-0.5)≈0.3935."""
        rel = _hybrid_relevances([], [("ID1", -0.5)])
        assert rel["ID1"] == pytest.approx(1.0 - math.exp(-0.5), abs=1e-9)

    def test_both_hit_takes_max(self):
        """An id hit by both vector and FTS should take the larger of the two values."""
        # Vector cos=0.5, but FTS's bm25=-6 (lex≈0.9975) is larger -> lex is adopted
        rel = _hybrid_relevances([("ID1", 0.5)], [("ID1", -6.0)])
        assert rel["ID1"] == pytest.approx(1.0 - math.exp(-6.0), abs=1e-9)

        # Conversely, vector cos=0.99 is larger than a weak FTS hit (bm25=-0.1, lex≈0.095)
        rel2 = _hybrid_relevances([("ID2", 0.99)], [("ID2", -0.1)])
        assert rel2["ID2"] == pytest.approx(0.99, abs=1e-9)

    def test_bm25_zero_or_positive_gives_zero(self):
        """When bm25 is 0 or positive, lex should be 0.0 (only bm25<0 is valid)."""
        rel = _hybrid_relevances([], [("ID1", 0.0), ("ID2", 3.5)])
        assert rel["ID1"] == 0.0
        assert rel["ID2"] == 0.0

    def test_empty_inputs_give_empty_dict(self):
        """If both vec_results / kw_results are empty, an empty dict should be returned."""
        assert _hybrid_relevances([], []) == {}

    def test_vector_only_keeps_cosine(self):
        """An id hit only by vector search should keep its cosine similarity as-is."""
        rel = _hybrid_relevances([("ID1", 0.72)], [])
        assert rel == {"ID1": 0.72}


class TestHybridRelevanceIntegration:
    def test_rare_token_ranks_first_via_recall(self, tmp_path):
        """Via recall: a memory containing a rare, exact-match token should rank
        first in search results even when another memory has higher (fake)
        vector similarity.

        FakeEmbedder produces deterministic hash-based pseudo-vectors, so it's
        hard to deliberately manipulate cosine similarity. Instead, this uses a
        real build_engine + real db/store (tmp_path) to verify that FTS's BM25
        actually takes effect.
        """
        settings = Settings(
            memories_dir=tmp_path / "memories",
            data_dir=tmp_path / "data",
            candidate_k=20,
        )
        engine = build_engine(settings, embedder=FakeEmbedder(dim=256))
        try:
            # A memory containing a rare token (an ID-like string)
            rare_token = "ZXQ9981-エラーコード"
            target = engine.remember(
                f"{rare_token} が出た場合はキャッシュを全消去してから再起動する",
                type="knowledge", importance=5, now=NOW,
            )
            # Lexically unrelated, but inserted in bulk to thicken the candidate pool
            topics = ["予算配分", "採用面接", "週次定例", "顧客訪問",
                      "障害対応訓練", "棚卸し作業", "契約更新", "備品発注"]
            for i, topic in enumerate(topics):
                engine.remember(
                    f"メモ{i}: {topic}に関する記録。詳細は別紙参照。",
                    type="knowledge", importance=5, now=NOW,
                )

            result = engine.recall(rare_token, limit=5, now=NOW,
                                   record_hits=False)
            hit_ids = [h["id"] for h in result["hits"]]
            assert hit_ids[0] == target["id"], (
                "希少トークンの完全一致は BM25 由来の高 relevance で首位に"
                "来るべき(ベクトル類似度の順位写像に頼っていた旧実装は"
                "これを満たせなかった)"
            )
        finally:
            engine.db.close()


# ---------------------------------------------------------------------------
# forget: after moving to trash, db.set_path should be called and the actual
# file should move under _trash/
# ---------------------------------------------------------------------------

class TestForgetTrashPath:
    def test_forget_updates_db_path_to_trash(self, tmp_path):
        """After forget, db.get_memory(id)["path"] should point under _trash/,
        and an actual file should exist at that path (integration test using
        real store/db)."""
        settings = Settings(
            memories_dir=tmp_path / "memories",
            data_dir=tmp_path / "data",
            candidate_k=20,
        )
        engine = build_engine(settings, embedder=FakeEmbedder(dim=64))
        try:
            created = engine.remember(
                "忘れられる運命の記憶", type="knowledge", importance=3, now=NOW,
            )
            mem_id = created["id"]

            # Before forget: the path is not under _trash
            before = engine.db.get_memory(mem_id)
            assert "_trash" not in before["path"]

            result = engine.forget(mem_id)
            assert result["status"] == "forgotten"

            after = engine.db.get_memory(mem_id)
            assert after is not None
            after_path = Path(after["path"])
            assert "_trash" in after_path.parts
            assert after_path.is_file(), "db.set_path 後の新パスに実ファイルが存在すること"
        finally:
            engine.db.close()


# ---------------------------------------------------------------------------
# Regression tests for within-candidate relevance normalization
# (countermeasure for cosine compression)
# ---------------------------------------------------------------------------

class TestRelevanceNormalization:
    """Reproduces and confirms the fix for a ranking reversal observed in the
    2026-07-06 baseline measurement.

    In real data, 57.3% of the top-5 results were irrelevant, and 18.7% of
    those were highly active, generic preferences (15 queries x 5 results).
    The cause was that cosine's compressed value range (0.8-0.87) crushed
    relevance's discriminative power, letting the activation + importance
    boost dominate the ranking. Fixed via within-candidate min-max normalization.
    """

    def _two_memories(self, tmp_path):
        """TARGET = highly relevant but sunken specialist knowledge /
        PREF = low relevance but a frequently-recalled general-purpose memory."""
        engine, db, store = _build_engine(tmp_path)

        target = _fake_db_mem(id="TARGET", importance=6, path="/tmp/target.md",
                              created_at=NOW - 90 * DAY)
        pref = _fake_db_mem(id="PREF", importance=9, path="/tmp/pref.md",
                            created_at=NOW - 90 * DAY)
        db.all_memories.return_value = [target, pref]
        db.keyword_search.return_value = []
        db.get_links.return_value = []
        db.get_embeddings.return_value = {}

        records = {
            "/tmp/target.md": _fake_record(id="TARGET", importance=6,
                                           path="/tmp/target.md",
                                           content="ドライブ文字漂流の真因と対処"),
            "/tmp/pref.md": _fake_record(id="PREF", importance=9,
                                         path="/tmp/pref.md",
                                         content="ユーザーの呼び方ルール"),
        }
        store.read.side_effect = lambda p: records[str(p)]
        return engine, db, store

    def test_compressed_relevance_wins_over_activation_gate(self, tmp_path):
        """The compressed-band relevance gap (0.87 vs 0.80) should beat the
        frequent memory's activation + importance boost (regression test for
        the reversal where PREF used to rank first before the fix)."""
        engine, db, store = self._two_memories(tmp_path)

        # Compressed band: the gap is only 0.07
        db.vector_search.return_value = [("TARGET", 0.87), ("PREF", 0.80)]
        # PREF is reinforced every session (highly active); TARGET has been unused for a long time
        db.get_events.return_value = {
            "PREF": [(NOW - i * DAY, 1.0) for i in range(1, 11)],
        }

        result = engine.recall("ドライブ文字変更の影響", mode="fast", now=NOW,
                               record_hits=False)

        ids = [h["id"] for h in result["hits"]]
        assert ids[0] == "TARGET", (
            f"関連度の高い記憶が常連記憶に負けた(修正前の逆転が再発): {ids}")
        # The reported relevance stays as the raw value (the normalized value is not leaked)
        top = result["hits"][0]
        assert abs(top["relevance"] - 0.87) < 1e-6

    def test_tiny_spread_falls_back_to_activation(self, tmp_path):
        """When the relevance gap between candidates is below the floor (i.e.
        the query is not discriminative), ranking should be decided by
        activation/importance (fallback to base-level activation)."""
        engine, db, store = self._two_memories(tmp_path)

        # Nearly indistinguishable: gap of 0.005 (below the 0.10 floor, so not amplified)
        db.vector_search.return_value = [("TARGET", 0.805), ("PREF", 0.800)]
        db.get_events.return_value = {
            "PREF": [(NOW - i * DAY, 1.0) for i in range(1, 11)],
        }

        result = engine.recall("あいまいなクエリ", mode="fast", now=NOW,
                               record_hits=False)

        ids = [h["id"] for h in result["hits"]]
        assert ids[0] == "PREF", f"弁別不能時は高活性側が先頭のはず: {ids}"

    def test_escalation_still_uses_raw_scale(self, tmp_path):
        """Even though normalization changes the top candidate's ranking score,
        the auto-escalation-to-deep decision should still use the raw score
        (it should not trigger when there's a highly relevant hit)."""
        engine, db, store = self._two_memories(tmp_path)

        db.vector_search.return_value = [("TARGET", 0.87), ("PREF", 0.80)]
        db.get_events.return_value = {
            "PREF": [(NOW - i * DAY, 1.0) for i in range(1, 11)],
        }

        result = engine.recall("ドライブ文字変更の影響", mode="fast", now=NOW,
                               record_hits=False)

        # The raw composite score's max (around 0.86 on PREF's side) far exceeds 0.35
        assert result["auto_deepened"] is False
        assert result["mode"] == "fast"
