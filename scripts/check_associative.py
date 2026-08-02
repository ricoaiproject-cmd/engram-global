"""Targeted verification of associative recall.

Confirms that "a memory that is semantically far from the query (won't
surface via vector search) but is link-connected to a related memory"
is found by deep recall as via="associative". A direct test of the core
idea (if it's linked, following the trail will always retrieve it).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from engram.config import Settings
from engram.embedder import FakeEmbedder
from engram.engine import build_engine

DAY = 86400.0
T0 = 1_750_000_000.0

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    # Make candidate_k smaller than the corpus so the vector-search net
    # excludes "semantically far memories" (a microcosm of real usage)
    settings = Settings(memories_dir=tmp / "memories", data_dir=tmp / "data",
                        candidate_k=10)
    # dim=64 causes hash collisions where even unrelated texts get cos≈0.5, too noisy
    engine = build_engine(settings, embedder=FakeEmbedder(dim=256))
    try:
        # Noise: many memories unrelated to the query (vary the wording to avoid dedup detection)
        topics = ["予算配分", "採用面接", "週次定例", "顧客訪問", "障害対応訓練",
                  "棚卸し作業", "契約更新", "備品発注", "勉強会準備", "評価面談"]
        for i in range(60):
            engine.remember(
                f"メモ{i}: {topics[i % 10]}に関する記録その{i}。詳細は別紙参照。",
                type="knowledge", importance=4, now=T0,
            )

        # Hub: a memory semantically close to the query
        hub = engine.remember(
            "SQLiteのWALモードでは書き込みと読み取りが並行できる",
            type="knowledge", importance=6, now=T0,
        )

        # Isolated memory: semantically far from the query (no vocabulary overlap), but linked to the hub
        iso = engine.remember(
            "圧力鍋で豚の角煮を作るときは下茹でを30分する",
            type="knowledge", importance=5, now=T0,
        )
        engine.link(hub["id"], iso["id"])

        # Drop the isolated memory to cold tier (simulates an old, unused memory)
        rec = engine.store.find_by_id(iso["id"])
        engine.store.set_tier(rec, "cold")
        engine.db.set_tier(iso["id"], "cold")

        query = "SQLite WALモード 並行書き込み"
        fast = engine.recall(query, mode="fast", limit=10, now=T0 + 30 * DAY,
                             record_hits=False)
        deep = engine.recall(query, mode="deep", limit=10, now=T0 + 30 * DAY,
                             record_hits=False)

        fast_ids = [h["id"] for h in fast["hits"]]
        deep_hits = {h["id"]: h for h in deep["hits"]}

        print(f"Isolated memory appears in fast (should be False): {iso['id'] in fast_ids}")
        in_deep = iso["id"] in deep_hits
        print(f"Isolated memory appears in deep (should be True): {in_deep}")
        if in_deep:
            h = deep_hits[iso["id"]]
            print(f"  via={h['via']} score={h['score']:.3f} "
                  f"relevance={h['relevance']:.3f} tier={h['tier']}")
        ok = (iso["id"] not in fast_ids) and in_deep \
            and deep_hits[iso["id"]]["via"] == "associative"
        print(f"\nResult: {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    finally:
        engine.db.close()
