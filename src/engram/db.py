"""SQLite index layer (owner: Agent A).

The source of truth is Markdown (store.py). This DB is an index over vectors,
FTS, access logs, and links; everything except the access log can be rebuilt
from Markdown.

Implementation requirements:
- A sqlite-vec vec0 virtual table, created with
  `embedding float[{dim}] distance_metric=cosine`; similarity = 1 − distance.
  dim is saved to the meta table on first initialization, and any mismatch on
  later connections raises an exception.
- FTS5 uses `tokenize='trigram'` (Japanese can't be split on whitespace;
  requires SQLite>=3.34). Query terms are wrapped in double quotes to avoid
  syntax errors. bm25() — lower is better.
- vec0 KNN (`WHERE embedding MATCH ? AND k = ?`) can't apply a tier filter
  directly, so we overfetch k*4 rows and narrow via a JOIN with memories.
- Schema:
    memories(id TEXT PK, path TEXT, type TEXT, content_hash TEXT,
             created_at REAL, importance INTEGER, tier TEXT)
    access_events(id INTEGER PK AUTOINCREMENT, memory_id TEXT, ts REAL,
                  kind TEXT, weight REAL)  + INDEX(memory_id)
    links(src TEXT, dst TEXT, kind TEXT, weight REAL, PRIMARY KEY(src,dst,kind))
    meta(key TEXT PK, value TEXT)
- Connections use check_same_thread=False, WAL mode.
- The parent directory is created if it doesn't exist.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import sqlite_vec


class IndexDB:
    def __init__(self, path: Path, dim: int) -> None:
        """Open the DB (creating it if missing) and initialize the schema.

        Raises ValueError if the dim stored in meta doesn't match the dim
        argument.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._dim = dim
        self._conn = sqlite3.connect(
            str(path), check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row

        # Load sqlite-vec extension
        # The stock macOS Python and python.org Python builds are compiled
        # without SQLite extension-loading support, and fail with
        # AttributeError. Raise an error that explains the cause and the fix
        # (uv-managed Python does support it).
        if not hasattr(self._conn, "enable_load_extension"):
            self._conn.close()
            raise RuntimeError(
                "This Python's SQLite doesn't support extension loading, so "
                "vector search (sqlite-vec) can't be initialized. Please "
                "reinstall with a uv-managed Python: "
                "UV_PYTHON_PREFERENCE=only-managed uv tool install "
                "--python 3.12 --force git+https://github.com/ricoaiproject-cmd/engram.git"
            )
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        # WAL mode
        self._conn.execute("PRAGMA journal_mode=WAL")
        # In case the MCP server and the session-end hook write at the same
        # time, wait up to 5 seconds on lock contention instead of failing
        # immediately.
        self._conn.execute("PRAGMA busy_timeout=5000")

        self._init_schema()

    def _init_schema(self) -> None:
        dim = self._dim
        with self._conn:
            # meta table first (needed for dim check)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS meta "
                "(key TEXT PRIMARY KEY, value TEXT)"
            )
            # Check / store dim
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = 'dim'"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('dim', ?)",
                    (str(dim),),
                )
            else:
                stored = int(row["value"])
                if stored != dim:
                    raise ValueError(
                        f"DB dim mismatch: stored={stored}, requested={dim}"
                    )

            # memories table
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    path TEXT,
                    type TEXT,
                    content_hash TEXT,
                    created_at REAL,
                    importance INTEGER,
                    tier TEXT,
                    room TEXT DEFAULT 'common'
                )"""
            )
            # Migration from the old schema (v0.2 and earlier): add the room column if missing
            cols = {
                r["name"]
                for r in self._conn.execute("PRAGMA table_info(memories)")
            }
            if "room" not in cols:
                self._conn.execute(
                    "ALTER TABLE memories ADD COLUMN room TEXT DEFAULT 'common'"
                )

            # vec0 virtual table
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories "
                f"USING vec0(memory_id TEXT PRIMARY KEY, "
                f"embedding float[{dim}] distance_metric=cosine)"
            )

            # FTS5 virtual table
            self._conn.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS fts_memories
                USING fts5(memory_id UNINDEXED, content, tokenize='trigram')"""
            )

            # access_events table
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS access_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT,
                    ts REAL,
                    kind TEXT,
                    weight REAL
                )"""
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_memory_id "
                "ON access_events(memory_id)"
            )

            # links table
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS links (
                    src TEXT,
                    dst TEXT,
                    kind TEXT,
                    weight REAL,
                    PRIMARY KEY(src, dst, kind)
                )"""
            )

    def close(self) -> None:
        self._conn.close()

    # --- memories ---
    def upsert_memory(
        self,
        *,
        id: str,
        path: str,
        type: str,
        content_hash: str,
        created_at: float,
        importance: int,
        tier: str,
        content: str,
        embedding: np.ndarray,
        room: str = "common",
    ) -> None:
        """Upsert memories / vec_memories / fts_memories together (single transaction)."""
        emb = np.asarray(embedding, dtype=np.float32)
        emb_bytes = emb.tobytes()

        with self._conn:
            # memories: upsert
            self._conn.execute(
                """INSERT INTO memories(id, path, type, content_hash,
                    created_at, importance, tier, room)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                       path=excluded.path,
                       type=excluded.type,
                       content_hash=excluded.content_hash,
                       created_at=excluded.created_at,
                       importance=excluded.importance,
                       tier=excluded.tier,
                       room=excluded.room""",
                (id, path, type, content_hash, created_at, importance, tier,
                 room),
            )

            # vec_memories: vec0 doesn't support UPSERT reliably → DELETE+INSERT
            self._conn.execute(
                "DELETE FROM vec_memories WHERE memory_id = ?", (id,)
            )
            self._conn.execute(
                "INSERT INTO vec_memories(memory_id, embedding) VALUES (?, ?)",
                (id, emb_bytes),
            )

            # fts_memories: DELETE+INSERT
            self._conn.execute(
                "DELETE FROM fts_memories WHERE memory_id = ?", (id,)
            )
            self._conn.execute(
                "INSERT INTO fts_memories(memory_id, content) VALUES (?, ?)",
                (id, content),
            )

    def delete_memory(self, id: str) -> None:
        """Fully delete from memories / vec / fts / events / links (used to clean up orphans during reindex)."""
        with self._conn:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (id,))
            self._conn.execute(
                "DELETE FROM vec_memories WHERE memory_id = ?", (id,)
            )
            self._conn.execute(
                "DELETE FROM fts_memories WHERE memory_id = ?", (id,)
            )
            self._conn.execute(
                "DELETE FROM access_events WHERE memory_id = ?", (id,)
            )
            self._conn.execute(
                "DELETE FROM links WHERE src = ? OR dst = ?", (id, id)
            )

    def get_memory(self, id: str) -> dict | None:
        """Return a memories row as a dict (None if not found)."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def set_tier(self, id: str, tier: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE memories SET tier = ? WHERE id = ?", (tier, id)
            )

    def set_path(self, id: str, path: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE memories SET path = ? WHERE id = ?", (path, id)
            )

    def all_memories(
        self,
        *,
        tiers: list[str] | None = None,
        types: list[str] | None = None,
        rooms: list[str] | None = None,
    ) -> list[dict]:
        sql = "SELECT * FROM memories"
        params: list = []
        conditions: list[str] = []
        if tiers:
            placeholders = ",".join("?" * len(tiers))
            conditions.append(f"tier IN ({placeholders})")
            params.extend(tiers)
        if types:
            placeholders = ",".join("?" * len(types))
            conditions.append(f"type IN ({placeholders})")
            params.extend(types)
        if rooms:
            placeholders = ",".join("?" * len(rooms))
            conditions.append(f"room IN ({placeholders})")
            params.extend(rooms)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # --- access events ---
    def add_event(self, memory_id: str, kind: str, weight: float, ts: float) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO access_events(memory_id, ts, kind, weight) "
                "VALUES (?,?,?,?)",
                (memory_id, ts, kind, weight),
            )

    def get_events(
        self, memory_ids: list[str]
    ) -> dict[str, list[tuple[float, float]]]:
        """Return {memory_id: [(ts, weight), ...]} in a single query. Ids not found get an empty list."""
        result: dict[str, list[tuple[float, float]]] = {
            mid: [] for mid in memory_ids
        }
        if not memory_ids:
            return result
        placeholders = ",".join("?" * len(memory_ids))
        rows = self._conn.execute(
            f"SELECT memory_id, ts, weight FROM access_events "
            f"WHERE memory_id IN ({placeholders}) ORDER BY ts",
            memory_ids,
        ).fetchall()
        for row in rows:
            result[row["memory_id"]].append((row["ts"], row["weight"]))
        return result

    def export_events_jsonl(self, out_path: Path) -> int:
        """Write all access log entries out to JSONL (used for the nightly backup). Returns the row count."""
        rows = self._conn.execute(
            "SELECT memory_id, ts, kind, weight FROM access_events ORDER BY id"
        ).fetchall()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(
                    json.dumps(
                        {
                            "memory_id": row["memory_id"],
                            "ts": row["ts"],
                            "kind": row["kind"],
                            "weight": row["weight"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return len(rows)

    # --- links ---
    def add_link(
        self,
        src: str,
        dst: str,
        kind: str,
        *,
        increment: float = 1.0,
        max_weight: float = 1.0,
    ) -> None:
        """Create a link; if it already exists, weight += increment (clamped to max_weight)."""
        with self._conn:
            existing = self._conn.execute(
                "SELECT weight FROM links WHERE src=? AND dst=? AND kind=?",
                (src, dst, kind),
            ).fetchone()
            if existing is None:
                new_weight = min(increment, max_weight)
                self._conn.execute(
                    "INSERT INTO links(src, dst, kind, weight) VALUES (?,?,?,?)",
                    (src, dst, kind, new_weight),
                )
            else:
                new_weight = min(existing["weight"] + increment, max_weight)
                self._conn.execute(
                    "UPDATE links SET weight=? WHERE src=? AND dst=? AND kind=?",
                    (new_weight, src, dst, kind),
                )

    def get_links(
        self, ids: list[str], *, kinds: list[str] | None = None
    ) -> list[tuple[str, str, str, float]]:
        """Return edges (src, dst, kind, weight) where either src or dst is in ids."""
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        params: list = ids + ids
        sql = (
            f"SELECT src, dst, kind, weight FROM links "
            f"WHERE (src IN ({placeholders}) OR dst IN ({placeholders}))"
        )
        if kinds:
            kind_ph = ",".join("?" * len(kinds))
            sql += f" AND kind IN ({kind_ph})"
            params.extend(kinds)
        rows = self._conn.execute(sql, params).fetchall()
        return [(r["src"], r["dst"], r["kind"], r["weight"]) for r in rows]

    def get_embeddings(self, ids: list[str]) -> dict[str, np.ndarray]:
        """Fetch embeddings from vec_memories as {id: float32 ndarray}. Ids not found are omitted.
        (Used by deep recall to recompute relevance for nodes reached via association.)"""
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT memory_id, embedding FROM vec_memories "
            f"WHERE memory_id IN ({placeholders})",
            ids,
        ).fetchall()
        result: dict[str, np.ndarray] = {}
        for row in rows:
            vec = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            result[row["memory_id"]] = vec
        return result

    # --- search ---
    def vector_search(
        self,
        embedding: np.ndarray,
        k: int,
        *,
        tiers: list[str] | None = None,
        types: list[str] | None = None,
        rooms: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return k (id, cosine similarity) pairs sorted by similarity descending. Filters are applied via overfetch+JOIN."""
        emb = np.asarray(embedding, dtype=np.float32)
        emb_bytes = emb.tobytes()

        # Overfetch factor: larger when filters are active
        overfetch = k * 8 if (tiers or types or rooms) else k * 4
        overfetch = max(overfetch, k)

        try:
            knn_rows = self._conn.execute(
                "SELECT memory_id, distance FROM vec_memories "
                "WHERE embedding MATCH ? AND k = ?",
                (emb_bytes, overfetch),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        if not knn_rows:
            return []

        # Build allowed-id set from memories table with optional filters
        candidate_ids = [r["memory_id"] for r in knn_rows]
        placeholders = ",".join("?" * len(candidate_ids))
        filter_params: list = list(candidate_ids)
        filter_sql = f"SELECT id FROM memories WHERE id IN ({placeholders})"
        if tiers:
            tier_ph = ",".join("?" * len(tiers))
            filter_sql += f" AND tier IN ({tier_ph})"
            filter_params.extend(tiers)
        if types:
            type_ph = ",".join("?" * len(types))
            filter_sql += f" AND type IN ({type_ph})"
            filter_params.extend(types)
        if rooms:
            room_ph = ",".join("?" * len(rooms))
            filter_sql += f" AND room IN ({room_ph})"
            filter_params.extend(rooms)
        allowed = {
            r["id"]
            for r in self._conn.execute(filter_sql, filter_params).fetchall()
        }

        results: list[tuple[str, float]] = []
        for row in knn_rows:
            mid = row["memory_id"]
            if mid not in allowed:
                continue
            similarity = 1.0 - row["distance"]
            results.append((mid, similarity))
            if len(results) >= k:
                break

        return results

    def _allowed_ids(
        self,
        candidate_ids: list[str],
        tiers: list[str] | None,
        types: list[str] | None,
        rooms: list[str] | None,
    ) -> set[str] | None:
        """Return the set of ids passing the tier/type/room filters. Returns None if no filters are set."""
        if not (tiers or types or rooms):
            return None
        placeholders = ",".join("?" * len(candidate_ids))
        filter_params: list = list(candidate_ids)
        filter_sql = (
            f"SELECT id FROM memories WHERE id IN ({placeholders})"
        )
        if tiers:
            tier_ph = ",".join("?" * len(tiers))
            filter_sql += f" AND tier IN ({tier_ph})"
            filter_params.extend(tiers)
        if types:
            type_ph = ",".join("?" * len(types))
            filter_sql += f" AND type IN ({type_ph})"
            filter_params.extend(types)
        if rooms:
            room_ph = ",".join("?" * len(rooms))
            filter_sql += f" AND room IN ({room_ph})"
            filter_params.extend(rooms)
        return {
            r["id"]
            for r in self._conn.execute(filter_sql, filter_params).fetchall()
        }

    def keyword_search(
        self,
        query: str,
        k: int,
        *,
        tiers: list[str] | None = None,
        types: list[str] | None = None,
        rooms: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """FTS5 (trigram) BM25. Returns k (id, score) pairs in rank order.

        Trigram tokenization can't handle terms shorter than 3 characters.
        Mixing a short token into MATCH produces no token for it, which makes
        the AND condition match zero rows overall — so the MATCH expression
        is built only from tokens of 3+ characters. A short query with no
        token of 3+ characters (e.g. a 2-character Japanese word) falls back
        to a LIKE substring match and returns an IDF-based pseudo-score
        (_like_search).
        FTS syntax errors return an empty list (never raises).
        """
        stripped = query.strip()
        if not stripped:
            return []

        tokens = [t for t in stripped.split() if len(t) >= 3]
        if not tokens:
            return self._like_search(
                stripped.split(), k, tiers=tiers, types=types, rooms=rooms
            )

        # Escape and quote each term for FTS5
        # (internal double-quotes → doubled)
        fts_query = " ".join(
            '"' + tok.replace('"', '""') + '"' for tok in tokens
        )

        try:
            rows = self._conn.execute(
                """SELECT f.memory_id, bm25(fts_memories) AS score
                   FROM fts_memories f
                   WHERE f.content MATCH ?
                   ORDER BY score
                   LIMIT ?""",
                (fts_query, k * 4 if (tiers or types or rooms) else k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        if not rows:
            return []

        allowed = self._allowed_ids(
            [r["memory_id"] for r in rows], tiers, types, rooms
        )

        results: list[tuple[str, float]] = []
        for row in rows:
            mid = row["memory_id"]
            if allowed is not None and mid not in allowed:
                continue
            results.append((mid, row["score"]))
            if len(results) >= k:
                break

        return results

    def _like_search(
        self,
        tokens: list[str],
        k: int,
        *,
        tiers: list[str] | None = None,
        types: list[str] | None = None,
        rooms: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """LIKE substring match (AND) for short tokens that trigram can't handle.

        Instead of BM25, the score is an IDF-only pseudo-value:
            pseudo_bm25 = -ln(1 + N/df)   (N=total row count, df=match count)
        Combined with the engine-side mapping lex = 1 - exp(bm25), this gives
        lex = N/(N+df), so rarer-term hits score higher relevance (consistent
        in meaning with the BM25 path).
        Sort order is total token occurrence count descending (ties keep
        insertion order).
        """
        if not tokens:
            return []

        # Substring match condition (AND of tokens) with % _ \ escaped
        def esc(t: str) -> str:
            return (
                t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )

        where_sql = " AND ".join(
            "f.content LIKE ? ESCAPE '\\'" for _ in tokens
        )
        like_params = [f"%{esc(t)}%" for t in tokens]

        try:
            df_row = self._conn.execute(
                f"SELECT COUNT(*) AS df FROM fts_memories f WHERE {where_sql}",
                like_params,
            ).fetchone()
            df = int(df_row["df"])
            if df == 0:
                return []
            n_total = int(
                self._conn.execute(
                    "SELECT COUNT(*) AS n FROM fts_memories"
                ).fetchone()["n"]
            )

            # Order by total occurrence count (REPLACE trick)
            occ_terms = []
            occ_params: list = []
            for t in tokens:
                occ_terms.append(
                    "(LENGTH(f.content) - LENGTH(REPLACE(f.content, ?, '')))"
                    " / LENGTH(?)"
                )
                occ_params.extend([t, t])
            occ_sql = " + ".join(occ_terms)

            rows = self._conn.execute(
                f"""SELECT f.memory_id FROM fts_memories f
                    WHERE {where_sql}
                    ORDER BY ({occ_sql}) DESC
                    LIMIT ?""",
                like_params + occ_params
                + [k * 4 if (tiers or types or rooms) else k],
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        pseudo = -math.log(1.0 + n_total / df)
        allowed = self._allowed_ids(
            [r["memory_id"] for r in rows], tiers, types, rooms
        )

        results: list[tuple[str, float]] = []
        for row in rows:
            mid = row["memory_id"]
            if allowed is not None and mid not in allowed:
                continue
            results.append((mid, pseudo))
            if len(results) >= k:
                break

        return results

    # --- misc ---
    def stats(self) -> dict:
        """Counts (by type, by tier), event count, link count (by kind), etc."""
        total = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM memories"
        ).fetchone()["cnt"]

        by_type = {
            r["type"]: r["cnt"]
            for r in self._conn.execute(
                "SELECT type, COUNT(*) AS cnt FROM memories GROUP BY type"
            ).fetchall()
        }

        by_tier = {
            r["tier"]: r["cnt"]
            for r in self._conn.execute(
                "SELECT tier, COUNT(*) AS cnt FROM memories GROUP BY tier"
            ).fetchall()
        }

        by_room = {
            r["room"]: r["cnt"]
            for r in self._conn.execute(
                "SELECT room, COUNT(*) AS cnt FROM memories GROUP BY room"
            ).fetchall()
        }

        event_count = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM access_events"
        ).fetchone()["cnt"]

        links_by_kind = {
            r["kind"]: r["cnt"]
            for r in self._conn.execute(
                "SELECT kind, COUNT(*) AS cnt FROM links GROUP BY kind"
            ).fetchall()
        }

        return {
            "total_memories": total,
            "by_type": by_type,
            "by_tier": by_tier,
            "by_room": by_room,
            "event_count": event_count,
            "links_by_kind": links_by_kind,
        }
