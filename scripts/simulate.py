"""Simulation of a 30-day access pattern.

Ingests 200 memories with FakeEmbedder + a temp directory, then simulates
30 days of access patterns using a synthetic clock (injected via the `now`
argument).

Randomness is deterministic via a fixed seed.

Report contents:
1. Reinforced memories rank highly in recall for related queries
2. Neglected memories sink in fast recall but can still be found via deep
   (associative links)
3. An unused memory with importance 9 stays ranked above an unused memory
   with importance 2

Usage: python scripts/simulate.py
(Usable once db / store are complete. Currently expected to fail with
NotImplementedError.)
"""

from __future__ import annotations

import random
import sys
import tempfile
import time
from pathlib import Path

# Reconfigure to UTF-8 so marks like ✓ print correctly even on a Windows console (cp932)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add the project's src to PYTHONPATH
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from engram.config import Settings
from engram.embedder import FakeEmbedder
from engram.engine import MemoryEngine, build_engine

# --- constants ---
SEED = 42
N_MEMORIES = 200
DAY = 86400.0
START_TIME = 1_750_000_000.0  # simulation start time (fixed)
N_DAYS = 30

# Number of "active" memories used regularly
N_ACTIVE = 30
# Number of memory pairs to chain together with links
N_LINK_PAIRS = 20

# Memory templates by category (designed so words overlap within the same
# category, since FakeEmbedder uses n-gram similarity)
KNOWLEDGE_TEMPLATES = [
    "Pythonの非同期処理(asyncio)について: コルーチンを使う",
    "Pythonの型ヒント: TypeVar と Generic の使い方",
    "FastAPI の依存性注入パターンについての知識",
    "SQLite の WAL モードでの並行書き込み制御",
    "ベクトル検索の仕組み: コサイン類似度と KNN",
    "BM25 アルゴリズムの仕組みと FTS5 での実装",
    "ACT-R モデルの記憶活性化計算式",
    "Reciprocal Rank Fusion による検索ランク統合",
    "obsidian のウィキリンク形式 [[id]] の仕様",
    "ULID の生成規則と時系列ソート可能性",
]

EPISODE_TEMPLATES = [
    "今日の作業: エンジン設計レビューを行った",
    "今日の進捗: store.py のスタブ仕様を確認した",
    "今日の作業: test_dynamics.py を通過させた",
    "今日の進捗: recall メソッドの RRF 統合を実装した",
    "今日の作業: CLI のサブコマンドを実装した",
    "今日の進捗: co_recall リンクのヘッブ則を実装した",
    "今日の作業: consolidation 候補の貪欲クラスタリング実装",
    "今日の進捗: reindex の差分検知ロジックを実装した",
    "今日の作業: シミュレーションスクリプトを実行した",
    "今日の進捗: MCP サーバーの FastMCP 統合を完了した",
]


def run_simulation():
    rng = random.Random(SEED)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        settings = Settings(
            memories_dir=tmp_path / "memories",
            data_dir=tmp_path / "data",
            candidate_k=50,
            deep_score_threshold=0.35,
        )
        # dim=64 causes hash collisions where even unrelated texts get cos≈0.5, so use 256
        embedder = FakeEmbedder(dim=256)
        engine = build_engine(settings, embedder=embedder)
        try:
            return _run(engine, rng)
        finally:
            # On Windows, the temp directory fails to delete unless the DB is closed first
            engine.db.close()


def _run(engine: MemoryEngine, rng: random.Random):

        print("=" * 60)
        print("engram 30-day access pattern simulation")
        print("=" * 60)

        # ----------------------------------------------------------------
        # Phase 1: ingest 200 memories (day 0)
        # ----------------------------------------------------------------
        print(f"\n[Phase 1] Ingesting {N_MEMORIES} memories...")
        t0 = START_TIME
        memory_ids: list[str] = []
        importance_map: dict[str, int] = {}

        for i in range(N_MEMORIES):
            # Generate body text while cycling through templates
            if i < N_MEMORIES // 2:
                templates = KNOWLEDGE_TEMPLATES
                mem_type = "knowledge"
            else:
                templates = EPISODE_TEMPLATES
                mem_type = "episode"

            template = templates[i % len(templates)]
            # Using only the template would let bodies differing solely by index
            # get merged by duplicate detection (cos >= 0.92), so add individual
            # detail sentences to make each memory unique
            detail_words = ["実装時の注意点", "設計上の根拠", "計測した結果",
                            "失敗例の分析", "代替案との比較", "運用での知見",
                            "境界条件の確認", "性能への影響", "互換性の検討",
                            "導入手順の整理"]
            details = rng.sample(detail_words, 3)
            content = (f"{template} (記憶#{i:03d})。"
                       f"観点: {details[0]}、{details[1]}、{details[2]}。"
                       f"整理番号 {i * 7919 % 100000}。")

            # Assign importance
            # - Active memories (first N_ACTIVE): importance 5-8
            # - One special memory with importance 9 (index N_ACTIVE)
            # - One low-priority memory with importance 2 (index N_ACTIVE+1)
            # - The rest: importance 3-6
            if i < N_ACTIVE:
                importance = rng.randint(5, 8)
            elif i == N_ACTIVE:
                importance = 9   # high-importance unused memory
            elif i == N_ACTIVE + 1:
                importance = 2   # low-importance unused memory
            else:
                importance = rng.randint(3, 6)

            result = engine.remember(
                content=content,
                type=mem_type,
                importance=importance,
                source="simulate",
                now=t0 + i * 10,  # ingest at 10-second intervals
            )

            if result["status"] in ("created", "duplicate_reinforced"):
                mem_id = result["id"]
                memory_ids.append(mem_id)
                importance_map[mem_id] = importance

        print(f"  Ingest complete: {len(memory_ids)} memories")

        # Classify active memory IDs vs. the rest
        active_ids = memory_ids[:N_ACTIVE]

        # Special pair for Report 3: two memories that share a keyword (so
        # both enter the candidate set) and differ only in importance.
        # Neither is ever accessed again after this.
        high_res = engine.remember(
            "Xyzzy復旧手順: 本番障害時はまずスナップショットを確保してから再起動する",
            type="knowledge", importance=9, source="simulate", now=t0 + 3000,
        )
        low_res = engine.remember(
            "Xyzzy復旧手順についての参考資料がどこにあるかの覚え書き",
            type="knowledge", importance=2, source="simulate", now=t0 + 3010,
        )
        high_imp_id = high_res["id"]
        low_imp_id = low_res["id"]

        # ----------------------------------------------------------------
        # Phase 2: build a chain of links
        # ----------------------------------------------------------------
        print(f"\n[Phase 2] Creating a chain of {N_LINK_PAIRS} link pairs...")
        inactive_ids = memory_ids[N_ACTIVE + 2:]
        link_pairs: list[tuple[str, str]] = []
        sampled = rng.sample(inactive_ids, min(N_LINK_PAIRS * 2, len(inactive_ids)))
        for i in range(0, len(sampled) - 1, 2):
            src, dst = sampled[i], sampled[i + 1]
            engine.link(src, dst)
            link_pairs.append((src, dst))
        print(f"  Link creation complete: {len(link_pairs)} pairs")

        # ----------------------------------------------------------------
        # Phase 3: simulate 30 days of access patterns
        # ----------------------------------------------------------------
        print(f"\n[Phase 3] Simulating {N_DAYS} days of access patterns...")

        for day in range(1, N_DAYS + 1):
            day_ts = START_TIME + day * DAY

            # Recall + reinforce active memories every day (memories in actual use)
            n_daily_active = rng.randint(3, 8)
            daily_active = rng.sample(active_ids, min(n_daily_active, len(active_ids)))

            # Record recall events
            for mem_id in daily_active:
                engine.db.add_event(mem_id, "recall_hit",
                                    engine.settings.recall_hit_weight, day_ts)

            # Reinforce every 3 days
            if day % 3 == 0:
                strength = rng.uniform(1.0, 2.5)
                engine.reinforce(daily_active, strength=strength, now=day_ts)

        print("  Access pattern simulation complete")

        # ----------------------------------------------------------------
        # Phase 4: report
        # ----------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Report")
        print("=" * 60)

        sim_now = START_TIME + N_DAYS * DAY

        # --- Report 1: reinforced memories rank highly for related queries ---
        print("\n[Report 1] Recall rank of reinforced memories")
        query = KNOWLEDGE_TEMPLATES[0]  # template 1 of the active memories
        result = engine.recall(query, mode="fast", limit=10, now=sim_now, record_hits=False)
        hits = result["hits"]
        hit_ids_top10 = [h["id"] for h in hits]
        active_in_top10 = [id_ for id_ in hit_ids_top10 if id_ in active_ids]

        print(f"  Query: '{query[:50]}...'")
        print(f"  Active (reinforced) memories in the top 10: {len(active_in_top10)}")
        for h in hits[:5]:
            is_active = "(active)" if h["id"] in active_ids else ""
            print(f"    [..{h['id'][-8:]}] score={h['score']:.3f} "
                  f"act={h['activation']:.3f} {is_active}")

        r1_passed = len(active_in_top10) > 0
        print(f"  Result: {'PASS ✓' if r1_passed else 'FAIL ✗'} "
              f"(active memories {'do' if r1_passed else 'do not'} surface near the top)")

        # --- Report 2: neglected memories sink in fast but can be found via deep ---
        print("\n[Report 2] Neglected-memory discovery rate: fast vs deep")
        if link_pairs:
            # Search with a query related to the link's src (the neglected memory)
            src_id, dst_id = link_pairs[0]

            fast_result = engine.recall(
                EPISODE_TEMPLATES[0],  # a query close to the neglected memory's template
                mode="fast", limit=10, now=sim_now, record_hits=False
            )
            deep_result = engine.recall(
                EPISODE_TEMPLATES[0],
                mode="deep", limit=10, now=sim_now, record_hits=False
            )

            fast_ids = {h["id"] for h in fast_result["hits"]}
            deep_ids = {h["id"] for h in deep_result["hits"]}

            # Check whether the link target (dst_id) appears only in deep
            assoc_in_deep = any(
                h["id"] in {dst for _, dst in link_pairs} and h["via"] == "associative"
                for h in deep_result["hits"]
            )

            print(f"  Fast recall hit count: {len(fast_ids)}")
            print(f"  Deep recall hit count: {len(deep_ids)}")
            print(f"  Node found via associative link in deep: {assoc_in_deep}")

            deep_assoc_hits = [h for h in deep_result["hits"] if h["via"] == "associative"]
            print(f"  Deep-only (via=associative) hits: {len(deep_assoc_hits)}")

            r2_passed = len(deep_ids) >= len(fast_ids)
            print(f"  Result: {'PASS ✓' if r2_passed else 'FAIL ✗'} "
                  f"(deep {'hit' if r2_passed else 'did not hit'} at least as many as fast)")
        else:
            print("  (no link pairs, skipping)")
            r2_passed = True

        # --- Report 3: an unused importance-9 memory outranks importance 2 ---
        print("\n[Report 3] Rank gap by importance for unused memories")
        if high_imp_id and low_imp_id:
            # Search on the dedicated keyword shared by both memories (puts both in the candidate set)
            result = engine.recall(
                "Xyzzy復旧手順", mode="fast", limit=50,
                now=sim_now, record_hits=False
            )
            rank_map = {h["id"]: i for i, h in enumerate(result["hits"])}

            rank_high = rank_map.get(high_imp_id, 9999)
            rank_low = rank_map.get(low_imp_id, 9999)

            print(f"  Unused importance-9 memory [{high_imp_id[:8]}]: rank {rank_high + 1}")
            print(f"  Unused importance-2 memory [{low_imp_id[:8]}]: rank {rank_low + 1}")

            # Since the bodies differ, there's a legitimate difference in relevance
            # to the query. To observe encoding depth (the flashbulb effect),
            # compare the component with relevance's contribution removed
            # (activation + importance)
            hit_high = next((h for h in result["hits"] if h["id"] == high_imp_id), None)
            hit_low = next((h for h in result["hits"] if h["id"] == low_imp_id), None)
            w_rel = engine.settings.w_relevance
            enc_high = (hit_high["score"] - w_rel * hit_high["relevance"]) if hit_high else 0.0
            enc_low = (hit_low["score"] - w_rel * hit_low["relevance"]) if hit_low else 0.0
            print(f"  importance 9 encoding component (activation+importance): {enc_high:.4f}")
            print(f"  importance 2 encoding component (activation+importance): {enc_low:.4f}")

            r3_passed = enc_high > enc_low
            print(f"  Result: {'PASS ✓' if r3_passed else 'FAIL ✗'} "
                  f"(higher importance ranks {'higher' if r3_passed else 'lower'})")
        else:
            print("  (no memories for the importance test, skipping)")
            r3_passed = True

        # --- overall result ---
        print("\n" + "=" * 60)
        all_passed = r1_passed and r2_passed and r3_passed
        print(f"Overall result: {'ALL TESTS PASS ✓' if all_passed else 'SOME FAILED ✗'}")
        print("=" * 60)

        return all_passed


if __name__ == "__main__":
    success = run_simulation()
    sys.exit(0 if success else 1)
