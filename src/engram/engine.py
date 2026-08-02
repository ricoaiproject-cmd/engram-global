"""Memory engine (owned by: Agent B). Orchestrates db / store / embedder / dynamics.

All methods return JSON-serializable values or models data types. Time defaults
to time.time(), but `now` can be injected as an argument for testability
(now: float | None).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import dynamics
from .config import Settings, get_settings
from .db import IndexDB
from .embedder import Embedder
from .models import MemoryRecord, RecallHit
from .store import MarkdownStore


class MemoryEngine:
    def __init__(self, settings: Settings, store: MarkdownStore, db: IndexDB,
                 embedder: Embedder) -> None:
        self.settings = settings
        self.store = store
        self.db = db
        self.embedder = embedder

    # ------------------------------------------------------------------ remember
    def remember(
        self,
        content: str,
        type: str,
        importance: int,
        *,
        tags: list[str] | None = None,
        source: str = "unknown",
        related_ids: list[str] | None = None,
        room: str | None = None,
        now: float | None = None,
    ) -> dict:
        """Store a memory. Procedure:
        1. Embed content and run vector_search(top1, same type, tier=hot) for
           duplicate detection. If cos >= settings.dup_threshold, skip creating
           a new memory: record a reinforce-equivalent event on the existing
           memory instead and return
           {"id": existing id, "status": "duplicate_reinforced"}.
        2. store.create -> db.upsert_memory.
        3. Record a create event with weight dynamics.create_event_weight(importance)
           (initial encoding boost).
        4. Add explicit links to related_ids (both db and store; the store side
           does not need to be bidirectional -- it is recorded only in the new
           memory's frontmatter).
        5. If tags contains "correction", raise importance to
           max(importance, settings.correction_min_importance).
        Returns: {"id", "status": "created", "path"}.
        """
        ts = now if now is not None else time.time()
        tags = list(tags) if tags else []
        room = room or "common"

        # Raise importance if the correction tag is present
        if "correction" in tags:
            importance = max(importance, self.settings.correction_min_importance)

        # 1. Duplicate detection: top-1 vector search within the same type/room,
        #    tier=hot (merging across rooms would break context isolation, so we
        #    restrict this to the same room)
        vec = self.embedder.embed_docs([content])[0]
        candidates = self.db.vector_search(
            vec, 1, tiers=["hot"], types=[type], rooms=[room]
        )
        if candidates:
            top_id, top_cos = candidates[0]
            if top_cos >= self.settings.dup_threshold:
                # Treat as duplicate: record a reinforce event on the existing memory
                weight = self.settings.reinforce_weight
                self.db.add_event(top_id, "reinforce", weight, ts)
                return {"id": top_id, "status": "duplicate_reinforced"}

        # 2. store.create -> db.upsert_memory
        record = self.store.create(
            content=content,
            type=type,
            importance=importance,
            tags=tags,
            source=source,
            links=related_ids or [],
            room=room,
        )

        self.db.upsert_memory(
            id=record.id,
            path=str(record.path),
            type=record.type,
            content_hash=record.content_hash,
            created_at=ts,
            importance=record.importance,
            tier=record.tier,
            content=content,
            embedding=vec,
            room=record.room,
        )

        # 3. Record the create event with the initial encoding boost
        create_weight = dynamics.create_event_weight(
            importance, alpha=self.settings.create_alpha
        )
        self.db.add_event(record.id, "create", create_weight, ts)

        # 4. Explicit links (db + store)
        if related_ids:
            for rel_id in related_ids:
                self.db.add_link(record.id, rel_id, "explicit", increment=1.0,
                                 max_weight=1.0)
                # store side: new memory's frontmatter only (no bidirectional
                # link needed) -- links were already passed to store.create,
                # so no further update is needed

        return {
            "id": record.id,
            "status": "created",
            "path": str(record.path),
        }

    # -------------------------------------------------------------------- recall
    def recall(
        self,
        query: str,
        *,
        mode: str = "fast",        # "fast" | "deep" | "exhaustive"
        limit: int = 5,
        type: str | None = None,
        room: str | None = None,   # None/"*" = all rooms. If given, restricted to {room, common}
        now: float | None = None,
        record_hits: bool = True,
    ) -> dict:
        """Recall (search). Procedure for fast mode:
        1. Run vector_search top-candidate_k and keyword_search top-candidate_k
           with tier=hot (episodes excluded; if a type is given, it takes
           priority).
        2. Merge with dynamics.rrf_merge -> take the top ~candidate_k as
           candidates.
        3. A candidate's relevance is its vector similarity (FTS-only hits are
           assigned the minimum similarity among the candidates). Ranking uses
           relevance after min-max normalization within the candidate set
           (dynamics.normalize_relevances, to counteract cosine compression);
           activation is computed via db.get_events and candidates are
           re-ranked with dynamics.final_score. Decay uses
           dynamics.decay_rate(importance). RecallHit.relevance holds the raw
           value.
        4. Return the top `limit` hits as RecallHit, and if record_hits is set,
           record a recall_hit event (weight=settings.recall_hit_weight) for
           each.
        5. If the highest composite score computed from raw relevance is below
           settings.deep_score_threshold, auto-trigger deep mode, merge its
           results in, and mark the response "auto_deepened": True (normalized
           scores shift scale per candidate set, so threshold comparisons
           always use raw values).

        Additional steps for deep mode:
        - Redo step 1, also including tier=cold/superseded and type=episode.
        - Seed dynamics.spread with the fast-mode top hits, building an
          adjacency function from db.get_links(kinds=co_recall/explicit/
          derived_from). Memories reached only via links get
          via="associative"; their relevance is computed separately as the
          actual cosine similarity against the query.
        - Superseded memories get a note "→ corrected by [successor id]", and
          the successor (the superseded_by link target) is also included in
          the results.

        Procedure for exhaustive mode (exhaustive recall):
        - Rank every memory across all tiers/types purely by cosine similarity
          to the query (relevance), without narrowing the candidate set.
          Activation is used only as a tiebreaker for equal relevance.
        - Even memories that have gone unused for a long time (low activation)
          will surface as long as they are semantically close.
        - Memories below settings.exhaustive_min_relevance are never returned.
          If deep mode's top score is below settings.exhaustive_score_threshold,
          it auto-escalates to exhaustive (in that case mode returns
          "exhaustive").
        Returns: {"hits": [RecallHit as dicts], "mode", "auto_deepened": bool}.
        """
        ts = now if now is not None else time.time()
        auto_deepened = False

        # Room filter: only look at the specified room plus common
        rooms: list[str] | None = None
        if room is not None and room != "*":
            rooms = sorted({room, "common"})

        if mode == "exhaustive":
            # Explicit exhaustive recall: ignore activation, pull from all
            # memories by relevance only
            hits = self._exhaustive_recall(query, limit=limit, type=type,
                                           rooms=rooms, now=ts)
        else:
            # Search in fast mode
            hits, best_score = self._fast_recall(query, limit=limit, type=type,
                                                 rooms=rooms, now=ts)

            # Auto-trigger deep mode if the top score is below the threshold
            if mode == "fast" and best_score < self.settings.deep_score_threshold:
                mode = "deep"
                auto_deepened = True

            if mode == "deep":
                hits = self._deep_recall(query, fast_hits=hits, limit=limit,
                                         type=type, rooms=rooms, now=ts)
                # Even in deep mode, a weak top score may mean the memory has
                # sunk or lies outside candidate_k's reach. Try relevance-only
                # exhaustive recall, and only adopt it if it yields a
                # higher-relevance result (so a fruitless attempt never
                # degrades the result). hit.score is normalized within the
                # candidate set and its scale shifts, so the threshold
                # comparison uses the raw composite score (same convention as
                # fast mode's best_score)
                deep_best = max(
                    (dynamics.final_score(
                        h.relevance, h.activation, round(h.importance * 10),
                        w_relevance=self.settings.w_relevance,
                        w_activation=self.settings.w_activation,
                        w_importance=self.settings.w_importance,
                    ) for h in hits),
                    default=0.0,
                )
                if deep_best < self.settings.exhaustive_score_threshold:
                    ex_hits = self._exhaustive_recall(
                        query, limit=limit, type=type, rooms=rooms, now=ts)
                    deep_best_rel = max((h.relevance for h in hits), default=0.0)
                    ex_best_rel = max((h.relevance for h in ex_hits), default=0.0)
                    if ex_hits and ex_best_rel > deep_best_rel:
                        mode = "exhaustive"
                        auto_deepened = True
                        hits = ex_hits

        # Record recall_hit events
        if record_hits:
            for hit in hits:
                self.db.add_event(hit.id, "recall_hit",
                                  self.settings.recall_hit_weight, ts)

        return {
            "hits": [_hit_to_dict(h) for h in hits],
            "mode": mode,
            "auto_deepened": auto_deepened,
        }

    def _fast_recall(
        self,
        query: str,
        *,
        limit: int,
        type: str | None,
        rooms: list[str] | None = None,
        now: float,
    ) -> tuple[list[RecallHit], float]:
        """Internal implementation of fast recall. Returns (hits, best_score)."""
        s = self.settings
        k = s.candidate_k

        # Episodes are excluded in fast mode
        if type is not None:
            search_types = [type]
        else:
            search_types = ["knowledge", "preference", "project"]

        search_tiers = ["hot"]

        # Generate the query vector
        qvec = self.embedder.embed_query(query)

        # Vector search + keyword search
        vec_results = self.db.vector_search(
            qvec, k, tiers=search_tiers, types=search_types, rooms=rooms
        )
        kw_results = self.db.keyword_search(
            query, k, tiers=search_tiers, types=search_types, rooms=rooms
        )

        # Vector similarity map
        vec_sim: dict[str, float] = {id_: sim for id_, sim in vec_results}
        # FTS score map (BM25 is lower-is-better, used for the rank list)
        vec_ids = [id_ for id_, _ in vec_results]
        kw_ids = [id_ for id_, _ in kw_results]

        # Merge with RRF
        merged = dynamics.rrf_merge([vec_ids, kw_ids], k=s.rrf_k)

        # Candidate relevance: vector similarity + FTS's BM25-to-rank mapping
        # (_hybrid_relevances)
        relevances = _hybrid_relevances(vec_results, kw_results)
        # Min-max normalize within the candidate set for ranking (to
        # counteract cosine compression). The raw value continues to be used
        # for RecallHit.relevance and escalation decisions.
        rel_norm = dynamics.normalize_relevances(
            relevances, floor=s.relevance_norm_floor)

        # Compute activation and re-rank with the final score
        candidate_ids = list(merged.keys())
        events_map = self.db.get_events(candidate_ids)
        # Fetch importance
        mem_rows = {m["id"]: m for m in self.db.all_memories(
            tiers=search_tiers, types=search_types, rooms=rooms)}

        scored: list[tuple[float, str]] = []
        best_raw_score = 0.0  # Deep-mode auto-trigger decisions use the raw score (scale-invariant)
        for id_ in candidate_ids:
            mem = mem_rows.get(id_)
            if mem is None:
                continue
            imp = mem["importance"]
            d = dynamics.decay_rate(imp)
            act = dynamics.activation_norm(events_map.get(id_, []), now, d,
                                           min_elapsed=s.min_elapsed_seconds)
            score = dynamics.final_score(
                rel_norm[id_], act, imp,
                w_relevance=s.w_relevance,
                w_activation=s.w_activation,
                w_importance=s.w_importance,
            )
            scored.append((score, id_))
            raw_score = dynamics.final_score(
                relevances[id_], act, imp,
                w_relevance=s.w_relevance,
                w_activation=s.w_activation,
                w_importance=s.w_importance,
            )
            if raw_score > best_raw_score:
                best_raw_score = raw_score

        scored.sort(reverse=True)
        top = scored[:limit]

        # Assemble RecallHit (content is read from path)
        hits: list[RecallHit] = []
        for score, id_ in top:
            mem = mem_rows[id_]
            imp = mem["importance"]
            d = dynamics.decay_rate(imp)
            act = dynamics.activation_norm(events_map.get(id_, []), now, d,
                                           min_elapsed=s.min_elapsed_seconds)
            rel = relevances[id_]
            # Fetch content from the store
            try:
                rec = self.store.read(Path(mem["path"]))
                content = rec.content
                tags = rec.tags
                tier = rec.tier
            except Exception:
                content = ""
                tags = []
                tier = mem.get("tier", "hot")
            hit = RecallHit(
                id=id_,
                content=content,
                type=mem["type"],
                tags=tags,
                tier=tier,
                score=score,
                relevance=rel,
                activation=act,
                importance=imp / 10.0,
                via="direct",
                room=mem.get("room", "common"),
            )
            hits.append(hit)

        return hits, best_raw_score

    def _deep_recall(
        self,
        query: str,
        *,
        fast_hits: list[RecallHit],
        limit: int,
        type: str | None,
        rooms: list[str] | None = None,
        now: float,
    ) -> list[RecallHit]:
        """Deep recall: re-search including tier=cold/superseded and episode,
        plus spreading activation."""
        s = self.settings
        k = s.candidate_k

        # Deep mode covers all tiers and types (room filter is still applied)
        search_tiers = ["hot", "cold", "superseded"]
        search_types = [type] if type is not None else None

        qvec = self.embedder.embed_query(query)

        vec_results = self.db.vector_search(
            qvec, k, tiers=search_tiers, types=search_types, rooms=rooms
        )
        kw_results = self.db.keyword_search(
            query, k, tiers=search_tiers, types=search_types, rooms=rooms
        )

        vec_sim: dict[str, float] = {id_: sim for id_, sim in vec_results}
        min_vec_sim = min(vec_sim.values()) if vec_sim else 0.0
        # Relevance of direct candidates (including FTS hits' BM25-to-rank mapping)
        relevances = _hybrid_relevances(vec_results, kw_results)

        vec_ids = [id_ for id_, _ in vec_results]
        kw_ids = [id_ for id_, _ in kw_results]
        merged = dynamics.rrf_merge([vec_ids, kw_ids], k=s.rrf_k)

        # Seeds: the top fast_hits. score is normalized within its candidate
        # set and its scale shifts per set, so the spreading seeds are
        # rebuilt from the raw composite score (to keep propagation amounts
        # stable)
        seed_map: dict[str, float] = {
            h.id: dynamics.final_score(
                h.relevance, h.activation, round(h.importance * 10),
                w_relevance=s.w_relevance,
                w_activation=s.w_activation,
                w_importance=s.w_importance,
            )
            for h in fast_hits
        }

        # Build an adjacency function from links (co_recall/explicit/derived_from)
        all_candidate_ids = list(set(list(merged.keys()) + list(seed_map.keys())))
        link_rows = self.db.get_links(
            all_candidate_ids,
            kinds=["co_recall", "explicit", "derived_from"]
        )
        # Bidirectional adjacency graph
        adjacency: dict[str, list[tuple[str, float]]] = {}
        for src, dst, kind, weight in link_rows:
            adjacency.setdefault(src, []).append((dst, weight))
            adjacency.setdefault(dst, []).append((src, weight))

        def neighbors(id_: str):
            return adjacency.get(id_, [])

        # Spreading activation
        spread_scores = dynamics.spread(
            seed_map, neighbors,
            max_hops=s.max_hops,
            hop_decay=s.hop_decay,
        )

        # All candidates = merged + nodes reached via spreading
        all_ids = set(merged.keys()) | set(spread_scores.keys())
        # Distinguish direct candidates vs. associative ones
        direct_ids = set(merged.keys())

        # Relevance of associative-only nodes: compute the actual cosine similarity
        assoc_ids = all_ids - direct_ids
        if assoc_ids:
            emb_map = self.db.get_embeddings(list(assoc_ids))
            for id_, emb in emb_map.items():
                cos = float(np.dot(qvec, emb))
                vec_sim[id_] = max(0.0, cos)

        # Fetch importance for all candidates. Since the rooms filter is
        # applied here, any node reached into another room via spreading
        # activation gets filtered out here (prevents associative leakage
        # across rooms)
        all_mem_rows = {m["id"]: m for m in self.db.all_memories(
            tiers=search_tiers, types=search_types, rooms=rooms
        )}

        events_map = self.db.get_events(list(all_ids))

        # Successor map for superseded memories
        superseded_links = self.db.get_links(
            list(all_ids), kinds=["superseded_by"]
        )
        successor_map: dict[str, str] = {}
        for src, dst, kind, _ in superseded_links:
            if kind == "superseded_by":
                successor_map[src] = dst

        # Pass 1: settle the raw relevance for every candidate (hybrid
        # relevance for direct candidates, actual cosine for associative
        # ones). Associative nodes are reached via links precisely because
        # their direct similarity to the query is low -- being connected by
        # a strong link is itself evidence of relevance, so we boost their
        # relevance using the spreading activation propagation score.
        # Without this, associative memories would be buried in noise and
        # never surface.)
        rel_map: dict[str, float] = {}
        for id_ in all_ids:
            if all_mem_rows.get(id_) is None:
                continue
            rel = relevances.get(id_, vec_sim.get(id_, min_vec_sim))
            if id_ not in direct_ids:
                rel = max(rel, spread_scores.get(id_, 0.0))
            rel_map[id_] = rel

        # Min-max normalize within the candidate set for ranking (cosine
        # compression counter-measure, same reason as fast mode)
        rel_norm = dynamics.normalize_relevances(
            rel_map, floor=s.relevance_norm_floor)

        scored: list[tuple[float, str, str, float, float]] = []  # (score, id, via, rel, act)
        for id_, rel in rel_map.items():
            mem = all_mem_rows[id_]
            via = "direct" if id_ in direct_ids else "associative"
            imp = mem["importance"]
            d = dynamics.decay_rate(imp)
            act = dynamics.activation_norm(events_map.get(id_, []), now, d,
                                           min_elapsed=s.min_elapsed_seconds)
            score = dynamics.final_score(
                rel_norm[id_], act, imp,
                w_relevance=s.w_relevance,
                w_activation=s.w_activation,
                w_importance=s.w_importance,
            )
            scored.append((score, id_, via, rel, act))

        scored.sort(reverse=True)
        top = scored[:limit]

        hits: list[RecallHit] = []
        for score, id_, via, rel, act in top:
            mem = all_mem_rows[id_]
            imp = mem["importance"]
            try:
                rec = self.store.read(Path(mem["path"]))
                content = rec.content
                tags = rec.tags
                tier = rec.tier
            except Exception:
                content = ""
                tags = []
                tier = mem.get("tier", "hot")

            # Attach a note to superseded memories
            note = ""
            if tier == "superseded" and id_ in successor_map:
                note = f"→ corrected by [{successor_map[id_]}]"

            hit = RecallHit(
                id=id_,
                content=content,
                type=mem["type"],
                tags=tags,
                tier=tier,
                score=score,
                relevance=rel,
                activation=act,
                importance=imp / 10.0,
                via=via,
                note=note,
                room=mem.get("room", "common"),
            )
            hits.append(hit)

        return hits

    def _exhaustive_recall(
        self,
        query: str,
        *,
        limit: int,
        type: str | None,
        rooms: list[str] | None = None,
        now: float,
    ) -> list[RecallHit]:
        """Exhaustive recall: brute-force every tier/type, ignoring activation
        and ranking by relevance alone.

        Because fast/deep let activation affect final_score, memories that
        have gone unused for a long time (low activation) get pushed out
        past `limit` even when highly relevant -- a memory that is never
        forgotten but can never be recalled. Here, without narrowing the
        candidate count, we rank purely by each memory's cosine similarity
        to the query, so a sunk memory will always surface if it is
        semantically close. The room filter still applies, and memories
        below settings.exhaustive_min_relevance are never returned.
        """
        s = self.settings

        # Cover all tiers and types (room filter is still applied)
        search_tiers = ["hot", "cold", "superseded"]
        search_types = [type] if type is not None else None

        mem_rows = self.db.all_memories(
            tiers=search_tiers, types=search_types, rooms=rooms
        )
        if not mem_rows:
            return []

        mem_by_id = {m["id"]: m for m in mem_rows}
        ids = list(mem_by_id.keys())

        qvec = self.embedder.embed_query(query)
        emb_map = self.db.get_embeddings(ids)
        events_map = self.db.get_events(ids)

        # Successor map for superseded memories (for the corrected-by note)
        successor_map: dict[str, str] = {}
        for src, dst, kind, _ in self.db.get_links(ids, kinds=["superseded_by"]):
            if kind == "superseded_by":
                successor_map[src] = dst

        scored: list[tuple[float, float, str]] = []  # (relevance, activation, id)
        for id_ in ids:
            emb = emb_map.get(id_)
            if emb is None:
                continue
            rel = max(0.0, float(np.dot(qvec, emb)))
            if rel < s.exhaustive_min_relevance:
                continue
            imp = mem_by_id[id_]["importance"]
            d = dynamics.decay_rate(imp)
            act = dynamics.activation_norm(events_map.get(id_, []), now, d,
                                           min_elapsed=s.min_elapsed_seconds)
            scored.append((rel, act, id_))

        # Rank primarily by relevance; activation is only a tiebreaker for
        # ties (so sunk memories still get picked up)
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        top = scored[:limit]

        hits: list[RecallHit] = []
        for rel, act, id_ in top:
            mem = mem_by_id[id_]
            imp = mem["importance"]
            try:
                rec = self.store.read(Path(mem["path"]))
                content = rec.content
                tags = rec.tags
                tier = rec.tier
            except Exception:
                content = ""
                tags = []
                tier = mem.get("tier", "hot")

            note = ""
            if tier == "superseded" and id_ in successor_map:
                note = f"→ corrected by [{successor_map[id_]}]"

            hits.append(RecallHit(
                id=id_,
                content=content,
                type=mem["type"],
                tags=tags,
                tier=tier,
                score=rel,            # Exhaustive recall ranks purely by relevance itself
                relevance=rel,
                activation=act,
                importance=imp / 10.0,
                via="exhaustive",
                note=note,
                room=mem.get("room", "common"),
            ))

        return hits

    # ----------------------------------------------------------------- reinforce
    def reinforce(self, ids: list[str], *, strength: float = 1.0,
                  now: float | None = None) -> dict:
        """Report usage. Records a reinforce event for each id
        (weight = settings.reinforce_weight * clamp(strength, 0.1, reinforce_strength_max)).
        For every pair of ids reinforced together, strengthens a co_recall
        link with increment=settings.colink_increment (Hebbian rule).
        Nonexistent ids are skipped and listed in the result under
        "unknown_ids".
        """
        ts = now if now is not None else time.time()
        s = self.settings

        # Clamp strength
        clamped = max(0.1, min(strength, s.reinforce_strength_max))
        weight = s.reinforce_weight * clamped

        reinforced: list[str] = []
        unknown: list[str] = []
        for id_ in ids:
            mem = self.db.get_memory(id_)
            if mem is None:
                unknown.append(id_)
                continue
            self.db.add_event(id_, "reinforce", weight, ts)
            reinforced.append(id_)

        # Hebbian rule: add a co_recall link for pairs reinforced together
        for i in range(len(reinforced)):
            for j in range(i + 1, len(reinforced)):
                self.db.add_link(
                    reinforced[i], reinforced[j], "co_recall",
                    increment=s.colink_increment,
                    max_weight=s.colink_max,
                )
                self.db.add_link(
                    reinforced[j], reinforced[i], "co_recall",
                    increment=s.colink_increment,
                    max_weight=s.colink_max,
                )

        return {
            "reinforced": reinforced,
            "unknown_ids": unknown,
        }

    # ------------------------------------------------------------------- correct
    def correct(self, id: str, corrected_content: str, reason: str, *,
                source: str = "unknown", now: float | None = None) -> dict:
        """Correct an error (the hypercorrection effect).
        1. Fetch the old memory (return {"status": "not_found"} if it does
           not exist).
        2. Assemble the new memory's content:
               {corrected_content}

               > [!note] Correction record
               > Previously misremembered as: "{first 200 chars of the old content}"
               > Reason for correction: {reason}
               > Old memory: [[old id]]
        3. Create it as a new memory, remember-style. Inherits type from the
           old memory; importance = max(old importance,
           settings.correction_min_importance); tags = old tags +
           "correction". Duplicate detection is skipped.
        4. Old memory: demoted to tier=superseded (store+db), with a
           superseded_by link (old -> new).
        Returns: {"new_id", "old_id", "status": "corrected"}.
        """
        ts = now if now is not None else time.time()
        s = self.settings

        # 1. Fetch the old memory
        old_mem = self.db.get_memory(id)
        if old_mem is None:
            return {"status": "not_found"}

        old_path = Path(old_mem["path"])
        try:
            old_record = self.store.read(old_path)
        except Exception:
            return {"status": "not_found"}

        old_content = old_record.content
        old_snippet = old_content[:200]

        # 2. Assemble the new memory's content
        new_content = (
            f"{corrected_content}\n\n"
            f"> [!note] Correction record\n"
            f"> Previously misremembered as: \"{old_snippet}\"\n"
            f"> Reason for correction: {reason}\n"
            f"> Old memory: [[{id}]]"
        )

        # 3. Create the new memory (duplicate detection skipped: uses _remember_direct)
        new_importance = max(old_record.importance, s.correction_min_importance)
        new_tags = list(old_record.tags) + ["correction"]
        if "correction" not in new_tags:
            new_tags.append("correction")

        new_result = self._remember_direct(
            content=new_content,
            type=old_record.type,
            importance=new_importance,
            tags=new_tags,
            source=source,
            related_ids=[id],
            room=old_record.room,
            now=ts,
        )
        new_id = new_result["id"]

        # 4. Demote the old memory to superseded
        self.store.set_tier(old_record, "superseded")
        self.db.set_tier(id, "superseded")

        # superseded_by link (old -> new)
        self.db.add_link(id, new_id, "superseded_by", increment=1.0, max_weight=1.0)

        return {
            "new_id": new_id,
            "old_id": id,
            "status": "corrected",
        }

    def _remember_direct(
        self,
        content: str,
        type: str,
        importance: int,
        *,
        tags: list[str] | None = None,
        source: str = "unknown",
        related_ids: list[str] | None = None,
        room: str = "common",
        now: float,
    ) -> dict:
        """Directly store a memory, bypassing duplicate detection (called from correct)."""
        s = self.settings
        tags = list(tags) if tags else []

        vec = self.embedder.embed_docs([content])[0]

        record = self.store.create(
            content=content,
            type=type,
            importance=importance,
            tags=tags,
            source=source,
            links=related_ids or [],
            room=room,
        )

        self.db.upsert_memory(
            id=record.id,
            path=str(record.path),
            type=record.type,
            content_hash=record.content_hash,
            created_at=now,
            importance=record.importance,
            tier=record.tier,
            content=content,
            embedding=vec,
            room=record.room,
        )

        create_weight = dynamics.create_event_weight(
            importance, alpha=s.create_alpha
        )
        self.db.add_event(record.id, "create", create_weight, now)

        if related_ids:
            for rel_id in related_ids:
                self.db.add_link(record.id, rel_id, "explicit", increment=1.0,
                                 max_weight=1.0)

        return {
            "id": record.id,
            "status": "created",
            "path": str(record.path),
        }

    # --------------------------------------------------------------------- misc
    def link(self, src: str, dst: str) -> dict:
        """Create an explicit link (not bidirectional in db -- a single
        src->dst edge, plus an append to src's frontmatter in the store)."""
        # Add the link in the DB
        self.db.add_link(src, dst, "explicit", increment=1.0, max_weight=1.0)

        # Append to src's frontmatter in the store
        src_mem = self.db.get_memory(src)
        if src_mem is not None:
            try:
                src_record = self.store.read(Path(src_mem["path"]))
                self.store.add_link(src_record, dst)
            except Exception:
                pass

        return {"src": src, "dst": dst, "status": "linked"}

    def forget(self, id: str) -> dict:
        """Soft delete: store.move_to_trash, and tier=trash in the db
        (excluded from recall)."""
        mem = self.db.get_memory(id)
        if mem is None:
            return {"status": "not_found"}

        try:
            record = self.store.read(Path(mem["path"]))
            updated = self.store.move_to_trash(record)
            # Keep the DB path in sync with the new location. If it kept the
            # old path, subsequent store.read calls for this memory would
            # fail and it would stay broken until the next reindex.
            if updated.path is not None:
                self.db.set_path(id, str(updated.path))
        except Exception:
            pass

        self.db.set_tier(id, "trash")
        return {"id": id, "status": "forgotten"}

    def stats(self) -> dict:
        return self.db.stats()

    # ------------------------------------------------------------- consolidation
    @staticmethod
    def _greedy_clusters(
        valid_ids: list[str], emb_map: dict, sim: float, *, min_size: int
    ) -> list[list[str]]:
        """Greedy clustering by embedding cosine similarity >= sim.

        Walking ids in order, each unused id becomes a new cluster's seed,
        and every other unused id with cosine similarity at or above the
        threshold gets merged in (shared logic for consolidation_candidates /
        skill_candidates). Clusters smaller than min_size are discarded.
        """
        used = set()
        clusters: list[list[str]] = []
        for i, id_a in enumerate(valid_ids):
            if id_a in used:
                continue
            cluster = [id_a]
            used.add(id_a)
            vec_a = emb_map[id_a]
            for id_b in valid_ids[i + 1:]:
                if id_b in used:
                    continue
                vec_b = emb_map[id_b]
                cos = float(np.dot(vec_a, vec_b))
                if cos >= sim:
                    cluster.append(id_b)
                    used.add(id_b)
            if len(cluster) >= min_size:
                clusters.append(cluster)
        return clusters

    def _clusters_with_contents(
        self, clusters: list[list[str]], mem_rows: dict
    ) -> list[dict]:
        """Fetch the content for each cluster's ids and format it into the return shape."""
        result_clusters = []
        for cluster_ids in clusters:
            contents = []
            for id_ in cluster_ids:
                mem = mem_rows.get(id_)
                if mem is None:
                    continue
                try:
                    rec = self.store.read(Path(mem["path"]))
                    contents.append(rec.content)
                except Exception:
                    contents.append("")
            result_clusters.append({"ids": cluster_ids, "contents": contents})
        return result_clusters

    def consolidation_candidates(self, *, now: float | None = None) -> dict:
        """Fetch tier=hot episodes older than settings.consolidate_min_age_days
        and greedily cluster them by embedding cosine similarity >=
        settings.consolidate_cluster_sim. Returns only clusters of 2 or more
        as {"clusters": [{"ids": [...], "contents": [...]}]}.
        The summarization itself is done by the calling agent (LLM) -- the
        server has no LLM of its own.
        """
        ts = now if now is not None else time.time()
        s = self.settings
        min_age_seconds = s.consolidate_min_age_days * 86400.0

        # Fetch tier=hot, type=episode memories
        all_mems = self.db.all_memories(tiers=["hot"], types=["episode"])

        # Narrow down to only the old ones
        old_ids = []
        for mem in all_mems:
            age = ts - mem.get("created_at", ts)
            if age >= min_age_seconds:
                old_ids.append(mem["id"])

        if not old_ids:
            return {"clusters": []}

        # Fetch embeddings and cluster
        emb_map = self.db.get_embeddings(old_ids)
        # Only ids present in emb_map are valid
        valid_ids = [id_ for id_ in old_ids if id_ in emb_map]

        if len(valid_ids) < 2:
            return {"clusters": []}

        clusters = self._greedy_clusters(
            valid_ids, emb_map, s.consolidate_cluster_sim, min_size=2
        )

        # Fetch each cluster's contents
        mem_rows = {m["id"]: m for m in all_mems}
        result_clusters = self._clusters_with_contents(clusters, mem_rows)

        return {"clusters": result_clusters}

    def skill_candidates(self, *, now: float | None = None) -> dict:
        """Return "skill candidate" clusters of episodes that recorded work
        of the same shape.

        Differences from consolidation_candidates (consolidation candidates):
        - No age filter: recent, repeated work is exactly what should be
          proposed for skill extraction -- unlike consolidate_min_age_days,
          there is no need to "wait until it's old."
        - The minimum cluster size is settings.skill_min_count (default 3,
          the "rule of three"). Unlike the 2-item clusters used for
          consolidation (compression via summarization), a stricter floor is
          used here so that one-off or coincidental matches don't get
          proposed as skills and turn into noise.
        - The goal is not summarization (memory compression) but surfacing
          material for proposing that the work be extracted into a
          procedure (a skill) to the user. It does not create the skill
          itself.

        Returns the same shape as consolidation_candidates:
        {"clusters": [{"ids": [...], "contents": [...]}]}
        """
        # now is unused since there is no age filter, but the parameter is
        # kept so the call signature matches consolidation_candidates (for
        # now-injection from tests).
        s = self.settings

        # Fetch tier=hot, type=episode memories (no age filter)
        all_mems = self.db.all_memories(tiers=["hot"], types=["episode"])
        all_ids = [mem["id"] for mem in all_mems]

        if not all_ids:
            return {"clusters": []}

        # Fetch embeddings and cluster
        emb_map = self.db.get_embeddings(all_ids)
        valid_ids = [id_ for id_ in all_ids if id_ in emb_map]

        if len(valid_ids) < s.skill_min_count:
            return {"clusters": []}

        clusters = self._greedy_clusters(
            valid_ids, emb_map, s.skill_cluster_sim, min_size=s.skill_min_count
        )

        mem_rows = {m["id"]: m for m in all_mems}
        result_clusters = self._clusters_with_contents(clusters, mem_rows)

        return {"clusters": result_clusters}

    def mark_consolidated(self, episode_ids: list[str], new_memory_id: str) -> dict:
        """Mark consolidation complete: add a derived_from link
        (episode -> new) for each episode and demote it to tier=cold."""
        updated = []
        for ep_id in episode_ids:
            mem = self.db.get_memory(ep_id)
            if mem is None:
                continue
            # derived_from link (episode -> new)
            self.db.add_link(ep_id, new_memory_id, "derived_from",
                             increment=1.0, max_weight=1.0)
            # Demote to tier=cold
            self.db.set_tier(ep_id, "cold")
            try:
                rec = self.store.read(Path(mem["path"]))
                self.store.set_tier(rec, "cold")
            except Exception:
                pass
            updated.append(ep_id)

        return {
            "consolidated": updated,
            "new_memory_id": new_memory_id,
            "status": "ok",
        }

    # ------------------------------------------------------------------- reindex
    def reindex(self) -> dict:
        """Rebuild the DB against the canonical Markdown source:
        - For each record from store.scan_all(), compare content_hash
          against the DB; re-embed and upsert anything that differs (manual
          edits) or is not yet registered.
        - delete_memory for any id that exists in the DB but whose file has
          vanished.
        - Returns counts as {"added", "updated", "removed", "unchanged"}.
        """
        added = 0
        updated = 0
        unchanged = 0

        # Full scan of files from the store
        seen_ids: set[str] = set()
        for record in self.store.scan_all():
            seen_ids.add(record.id)
            db_mem = self.db.get_memory(record.id)
            if db_mem is None:
                # Not yet registered: embed and upsert. created_at is
                # restored from the canonical frontmatter
                vec = self.embedder.embed_docs([record.content])[0]
                created_at = _created_to_epoch(record.created)
                self.db.upsert_memory(
                    id=record.id,
                    path=str(record.path),
                    type=record.type,
                    content_hash=record.content_hash,
                    created_at=created_at,
                    importance=record.importance,
                    tier=record.tier,
                    content=record.content,
                    embedding=vec,
                    room=record.room,
                )
                # If there are no events (i.e. this was rebuilt from
                # scratch), reseed a create event. Without this, activation
                # would be 0 for every memory after the rebuild.
                if not self.db.get_events([record.id]).get(record.id):
                    self.db.add_event(
                        record.id,
                        "create",
                        dynamics.create_event_weight(
                            record.importance, alpha=self.settings.create_alpha
                        ),
                        created_at,
                    )
                added += 1
            elif (db_mem.get("content_hash", "") != record.content_hash
                  or db_mem.get("room", "common") != record.room):
                # Diverges due to a manual edit (content or room): re-embed and upsert
                vec = self.embedder.embed_docs([record.content])[0]
                self.db.upsert_memory(
                    id=record.id,
                    path=str(record.path),
                    type=record.type,
                    content_hash=record.content_hash,
                    created_at=db_mem.get("created_at", 0.0),
                    importance=record.importance,
                    tier=record.tier,
                    content=record.content,
                    embedding=vec,
                    room=record.room,
                )
                updated += 1
            else:
                unchanged += 1

        # Delete ids that exist in the DB but whose file has vanished
        all_db_ids = {m["id"] for m in self.db.all_memories()}
        orphans = all_db_ids - seen_ids
        for orphan_id in orphans:
            self.db.delete_memory(orphan_id)
        removed = len(orphans)

        return {
            "added": added,
            "updated": updated,
            "removed": removed,
            "unchanged": unchanged,
        }

    # ------------------------------------------------ startup index freshness
    def check_index_freshness(self, *, mode: str = "auto") -> dict:
        """Startup index-freshness check (guards against multi-machine
        sharing gaps).

        The memory Markdown may be shared (e.g. via Google Drive) while
        index.db stays local to each machine, creating a blind spot where
        memories written on another machine are missing from the index and
        never surface in recall. Compares the Markdown (non-trash) file
        count against the index's active (hot/cold/superseded) count, and if
        they diverge:
          - mode="auto": reindex to resync
          - mode="warn": return warning info (the caller logs it; nothing is
            written)
          - mode="off" : do nothing
        Returns: {"action", "markdown", "index", ...}. action is one of
        "off"/"in_sync"/"warn"/"reindexed".
        """
        if mode == "off":
            return {"action": "off"}

        md_count = self.store.count_memory_files()
        idx_count = len(self.db.all_memories(
            tiers=["hot", "cold", "superseded"]
        ))

        if md_count == idx_count:
            # Fast path: if the raw .md count matches, treat it as in sync and skip scanning
            return {"action": "in_sync", "markdown": md_count, "index": idx_count}

        # Counts differ. This might just be an apparent mismatch caused by
        # empty files, broken frontmatter, or non-memory .md files without an
        # id (scan_all skips those). To avoid a wasted reindex, recount
        # precisely over the same "valid memory" population the index uses.
        valid_count = sum(1 for _ in self.store.scan_all())
        if valid_count == idx_count:
            return {
                "action": "in_sync",
                "markdown": md_count,
                "index": idx_count,
                "valid": valid_count,
                "note": "The raw .md count difference is only apparent, caused by empty/broken/non-memory md files",
            }

        if mode == "warn":
            return {
                "action": "warn",
                "markdown": md_count,
                "index": idx_count,
                "valid": valid_count,
                "drift": valid_count - idx_count,
            }

        # auto: reindex to resync (pulls in memories from other machines not yet in the index)
        reindex_result = self.reindex()
        return {
            "action": "reindexed",
            "markdown": md_count,
            "index": idx_count,
            "valid": valid_count,
            "reindex": reindex_result,
        }


def _hybrid_relevances(
    vec_results: list[tuple[str, float]],
    kw_results: list[tuple[str, float]],
) -> dict[str, float]:
    """Build the relevance for RRF candidates (the crux of hybrid search).

    For candidates that matched via the vector search, relevance is cosine
    similarity; for candidates that matched via FTS, it's a lexical relevance
    that maps BM25 onto an absolute (0,1) scale. When a candidate matched on
    both, whichever is larger wins.

    An earlier version gave a single min_vec_sim (the lowest similarity among
    the vector candidates) to every FTS-only hit, but that meant exact
    matches on IDs, file paths, or proper nouns -- a case vectors are weak at
    and FTS excels at -- always sank to the bottom of the candidates and
    never surfaced. On top of that, embedding cosine similarity tends to
    compress into a narrow band (e.g. Ruri, the Japanese-oriented embedding
    option, was observed compressing to roughly 0.8-0.87), and mapping BM25
    rank onto that narrow vector range could not fully express decisive
    lexical-match evidence (confirmed on real data).

    BM25 encodes term rarity (IDF), so we use that directly on an absolute
    scale:
        lex = 1 - exp(bm25_sqlite)      # sqlite's bm25() is negative, lower is better
    An exact match on a rare token (bm25 ~ -6) becomes 0.997, while a match on
    a common word (bm25 ~ -0.5) becomes 0.39 -- so only decisive lexical
    matches rise above the vector similarity ceiling (~0.87), while
    common-word hits still lose out to vector candidates.
    """
    relevances = {id_: sim for id_, sim in vec_results}
    for id_, bm25 in kw_results:
        lex = 1.0 - math.exp(bm25) if bm25 < 0 else 0.0
        relevances[id_] = max(relevances.get(id_, 0.0), lex)
    return relevances


def build_engine(settings: Settings | None = None, *, embedder: Embedder | None = None
                 ) -> MemoryEngine:
    """Factory that assembles an engine with the default configuration (used
    by server / cli). If embedder is not given, it is selected according to
    settings.embed_backend (embedder.make_embedder)."""
    if settings is None:
        settings = get_settings()

    if embedder is None:
        from .embedder import make_embedder
        embedder = make_embedder(settings)

    store = MarkdownStore(settings.memories_dir)
    # Always get the dimension from the actual model (if a guess is wrong,
    # the DB's vector table gets fixed at the wrong dimension and every
    # embedding from then on breaks). The ONNX path answers instantly from
    # meta.json, so no model load happens here. The torch path (RuriEmbedder)
    # does its first load here, but since the stdio server is long-running,
    # that only happens once.
    db = IndexDB(settings.db_path, embedder.dim)

    return MemoryEngine(settings=settings, store=store, db=db, embedder=embedder)


def _created_to_epoch(created: str) -> float:
    """Convert frontmatter's created (ISO 8601) to Unix seconds. Falls back to the current time if malformed."""
    from datetime import datetime

    try:
        return datetime.fromisoformat(created).timestamp()
    except (ValueError, TypeError):
        return time.time()


def _hit_to_dict(hit: RecallHit) -> dict:
    """Convert a RecallHit into a JSON-serializable dict."""
    return {
        "id": hit.id,
        "content": hit.content,
        "type": hit.type,
        "tags": hit.tags,
        "tier": hit.tier,
        "score": hit.score,
        "relevance": hit.relevance,
        "activation": hit.activation,
        "importance": hit.importance,
        "via": hit.via,
        "note": hit.note,
        "room": hit.room,
    }
