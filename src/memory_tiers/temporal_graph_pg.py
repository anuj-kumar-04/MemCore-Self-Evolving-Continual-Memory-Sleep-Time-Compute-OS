"""
Tier 2: Episodic & Bi-Temporal Semantic Memory
PostgreSQL 16 (pgvector HNSW + Relational Graph Edges) with In-Memory fallback.
Time-stamped events (Episodic) and Subject-Predicate-Object triplets:
(valid_from, valid_until, observed_at, superseded_by)
"""

from __future__ import annotations
import os
import json
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
import structlog
import numpy as np

from src.api.schemas import GraphEdgeRecord, EpisodicEventRecord, TripletFact
from src.memory_tiers.embeddings import get_embedding, cosine_similarity, generate_local_embedding

logger = structlog.get_logger(__name__)

# Shared fallback state across all instances
_SHARED_EDGES: Dict[str, GraphEdgeRecord] = {}
_SHARED_EVENTS: Dict[str, EpisodicEventRecord] = {}
_PG_UNAVAILABLE = False

class BiTemporalGraphPG:
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL", "postgresql://memcore_user:memcore_secret@localhost:5432/memcore")
        self._pool = None
        self._is_mock = False

    async def _get_pool(self):
        global _PG_UNAVAILABLE
        if self._pool is not None:
            return self._pool
        if _PG_UNAVAILABLE or self._is_mock:
            self._is_mock = True
            return None
        try:
            import asyncpg
            pool = await asyncpg.create_pool(
                self.db_url,
                min_size=1,
                max_size=5,
                timeout=0.3,
                command_timeout=1.0,
            )
            # Test query
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            self._pool = pool
            self._is_mock = False
            logger.info("temporal_graph_pg_connected", db_url=self.db_url)
            return self._pool
        except Exception as e:
            logger.warning("postgres_pgvector_unavailable_using_fallback", error=str(e))
            _PG_UNAVAILABLE = True
            self._is_mock = True
            return None

    async def add_edge(self, edge: GraphEdgeRecord) -> GraphEdgeRecord:
        """
        Inserts a new bi-temporal graph edge with vector embedding.
        """
        if not edge.embedding:
            fact_text = f"{edge.subject} {edge.predicate} {edge.object}"
            edge.embedding = await get_embedding(fact_text)

        pool = await self._get_pool()
        if pool and not self._is_mock:
            try:
                import json
                async with pool.acquire() as conn:
                    # Format embedding for pgvector: '[0.1, 0.2, ...]'
                    emb_str = f"[{','.join(str(x) for x in edge.embedding)}]" if edge.embedding else None
                    query = """
                    INSERT INTO temporal_edges (
                        id, user_id, subject, predicate, object, confidence,
                        valid_from, valid_until, observed_at, superseded_by,
                        source_session_id, source_turn_ids, metadata, embedding
                    ) VALUES (
                        $1::uuid, $2, $3, $4, $5, $6,
                        $7, $8, $9, $10::uuid,
                        $11, $12::jsonb, $13::jsonb, $14::vector
                    ) RETURNING id;
                    """
                    await conn.execute(
                        query,
                        edge.id, edge.user_id, edge.subject, edge.predicate, edge.object, edge.confidence,
                        edge.valid_from, edge.valid_until, edge.observed_at,
                        edge.superseded_by, edge.source_session_id,
                        json.dumps(edge.source_turn_ids),
                        json.dumps(edge.metadata),
                        emb_str
                    )
                    return edge
            except Exception as e:
                logger.warning("pg_add_edge_failed_falling_back", error=str(e))
                self._is_mock = True

        # Fallback in-memory
        _SHARED_EDGES[edge.id] = edge
        return edge

    async def expire_edge(
        self,
        edge_id: str,
        superseded_by_id: str,
        expiration_time: Optional[datetime] = None
    ) -> bool:
        """
        Sets valid_until and superseded_by on an existing edge.
        Crucial: This invalidates current truth WITHOUT deleting history!
        """
        until_time = expiration_time or datetime.now(timezone.utc)
        pool = await self._get_pool()
        if pool and not self._is_mock:
            try:
                async with pool.acquire() as conn:
                    query = """
                    UPDATE temporal_edges
                    SET valid_until = $1,
                        superseded_by = $2::uuid
                    WHERE id = $3::uuid;
                    """
                    result = await conn.execute(query, until_time, superseded_by_id, edge_id)
                    return "UPDATE 1" in result
            except Exception as e:
                logger.warning("pg_expire_edge_failed_falling_back", error=str(e))
                self._is_mock = True

        # Fallback
        if edge_id in _SHARED_EDGES:
            record = _SHARED_EDGES[edge_id]
            record.valid_until = until_time
            record.superseded_by = superseded_by_id
            return True
        return False

    async def update_confidence(self, edge_id: str, boost: float = 0.05, new_turn_id: Optional[str] = None) -> bool:
        """
        NOOP update: Boosts confidence & last observed time when the exact same fact is reaffirmed.
        """
        now = datetime.now(timezone.utc)
        pool = await self._get_pool()
        if pool and not self._is_mock:
            try:
                async with pool.acquire() as conn:
                    query = """
                    UPDATE temporal_edges
                    SET confidence = LEAST(1.0, confidence + $1),
                        observed_at = $2,
                        source_turn_ids = CASE 
                            WHEN $3::text IS NOT NULL THEN source_turn_ids || jsonb_build_array($3::text)
                            ELSE source_turn_ids 
                        END
                    WHERE id = $4::uuid;
                    """
                    await conn.execute(query, boost, now, new_turn_id, edge_id)
                    return True
            except Exception as e:
                logger.warning("pg_update_confidence_failed", error=str(e))
                self._is_mock = True

        if edge_id in _SHARED_EDGES:
            rec = _SHARED_EDGES[edge_id]
            rec.confidence = min(1.0, rec.confidence + boost)
            rec.observed_at = now
            if new_turn_id and new_turn_id not in rec.source_turn_ids:
                rec.source_turn_ids.append(new_turn_id)
            return True
        return False

    async def get_active_facts(
        self,
        user_id: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None
    ) -> List[GraphEdgeRecord]:
        """
        Retrieves all currently active facts (where valid_until IS NULL).
        """
        pool = await self._get_pool()
        if pool and not self._is_mock:
            try:
                async with pool.acquire() as conn:
                    conditions = ["user_id = $1", "valid_until IS NULL"]
                    params: List[Any] = [user_id]
                    if subject:
                        params.append(subject.strip())
                        conditions.append(f"lower(subject) = lower(${len(params)})")
                    if predicate:
                        params.append(predicate.strip())
                        conditions.append(f"upper(predicate) = upper(${len(params)})")

                    where_clause = " AND ".join(conditions)
                    query = f"""
                    SELECT id, user_id, subject, predicate, object, confidence,
                           valid_from, valid_until, observed_at, superseded_by,
                           source_session_id, source_turn_ids, metadata
                    FROM temporal_edges
                    WHERE {where_clause}
                    ORDER BY confidence DESC, observed_at DESC;
                    """
                    rows = await conn.fetch(query, *params)
                    return [self._row_to_edge(row) for row in rows]
            except Exception as e:
                logger.warning("pg_get_active_facts_failed", error=str(e))
                self._is_mock = True

        results = []
        for edge in _SHARED_EDGES.values():
            if edge.user_id == user_id and edge.valid_until is None:
                if subject and edge.subject.strip().lower() != subject.strip().lower():
                    continue
                if predicate and edge.predicate.strip().upper() != predicate.strip().upper():
                    continue
                results.append(edge)
        results.sort(key=lambda e: (e.confidence, e.observed_at), reverse=True)
        return results

    async def get_active_fact_by_key(
        self,
        user_id: str,
        subject: str,
        predicate: str
    ) -> Optional[GraphEdgeRecord]:
        """
        Fast lookup of a specific active relationship, e.g. (backend, USES_DB).
        """
        facts = await self.get_active_facts(user_id=user_id, subject=subject, predicate=predicate)
        return facts[0] if facts else None

    async def query_as_of(
        self,
        user_id: str,
        as_of_time: datetime,
        subject: Optional[str] = None,
        predicate: Optional[str] = None
    ) -> List[GraphEdgeRecord]:
        """
        Time-Travel Query:
        Returns facts that were active at a specific point in history:
        valid_from <= as_of_time AND (valid_until IS NULL OR valid_until > as_of_time)
        """
        pool = await self._get_pool()
        if pool and not self._is_mock:
            try:
                async with pool.acquire() as conn:
                    conditions = [
                        "user_id = $1",
                        "valid_from <= $2",
                        "(valid_until IS NULL OR valid_until > $2)"
                    ]
                    params: List[Any] = [user_id, as_of_time]
                    if subject:
                        params.append(subject.strip())
                        conditions.append(f"lower(subject) = lower(${len(params)})")
                    if predicate:
                        params.append(predicate.strip())
                        conditions.append(f"upper(predicate) = upper(${len(params)})")

                    where_clause = " AND ".join(conditions)
                    query = f"""
                    SELECT id, user_id, subject, predicate, object, confidence,
                           valid_from, valid_until, observed_at, superseded_by,
                           source_session_id, source_turn_ids, metadata
                    FROM temporal_edges
                    WHERE {where_clause}
                    ORDER BY valid_from DESC;
                    """
                    rows = await conn.fetch(query, *params)
                    return [self._row_to_edge(row) for row in rows]
            except Exception as e:
                logger.warning("pg_query_as_of_failed", error=str(e))
                self._is_mock = True

        results = []
        for edge in _SHARED_EDGES.values():
            if edge.user_id != user_id:
                continue
            if subject and edge.subject.strip().lower() != subject.strip().lower():
                continue
            if predicate and edge.predicate.strip().upper() != predicate.strip().upper():
                continue
            # Check bi-temporal window
            is_valid_at_time = edge.valid_from <= as_of_time and (edge.valid_until is None or edge.valid_until > as_of_time)
            if is_valid_at_time:
                results.append(edge)
        results.sort(key=lambda e: e.valid_from, reverse=True)
        return results

    async def hybrid_search(
        self,
        user_id: str,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5
    ) -> List[GraphEdgeRecord]:
        """
        Combines pgvector HNSW cosine similarity search with active truth filtering.
        """
        if not query_embedding:
            query_embedding = await get_embedding(query)

        pool = await self._get_pool()
        if pool and not self._is_mock:
            try:
                async with pool.acquire() as conn:
                    emb_str = f"[{','.join(str(x) for x in query_embedding)}]"
                    query_sql = """
                    SELECT id, user_id, subject, predicate, object, confidence,
                           valid_from, valid_until, observed_at, superseded_by,
                           source_session_id, source_turn_ids, metadata,
                           1 - (embedding <=> $1::vector) AS similarity
                    FROM temporal_edges
                    WHERE user_id = $2 AND valid_until IS NULL
                    ORDER BY embedding <=> $1::vector ASC
                    LIMIT $3;
                    """
                    rows = await conn.fetch(query_sql, emb_str, user_id, top_k)
                    return [self._row_to_edge(row) for row in rows]
            except Exception as e:
                logger.warning("pg_hybrid_search_failed", error=str(e))
                self._is_mock = True

        # Fallback cosine similarity
        active = [e for e in _SHARED_EDGES.values() if e.user_id == user_id and e.valid_until is None]
        scored = []
        for e in active:
            emb = e.embedding or generate_local_embedding(f"{e.subject} {e.predicate} {e.object}")
            score = cosine_similarity(query_embedding, emb)
            scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    async def add_episodic_event(self, event: EpisodicEventRecord) -> EpisodicEventRecord:
        """
        Stores an episodic summary milestone.
        """
        if not event.embedding:
            event.embedding = await get_embedding(event.summary)

        pool = await self._get_pool()
        if pool and not self._is_mock:
            try:
                import json
                async with pool.acquire() as conn:
                    emb_str = f"[{','.join(str(x) for x in event.embedding)}]"
                    query = """
                    INSERT INTO episodic_events (
                        id, user_id, session_id, timestamp, summary, raw_context, tags, turn_range, embedding
                    ) VALUES (
                        $1::uuid, $2, $3, $4, $5, $6, $7, $8, $9::vector
                    );
                    """
                    await conn.execute(
                        query,
                        event.id, event.user_id, event.session_id, event.timestamp,
                        event.summary, event.raw_context, event.tags, event.turn_range,
                        emb_str
                    )
                    return event
            except Exception as e:
                logger.warning("pg_add_episodic_failed", error=str(e))
                self._is_mock = True

        _SHARED_EVENTS[event.id] = event
        return event

    async def search_episodic_events(
        self,
        user_id: str,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 3
    ) -> List[EpisodicEventRecord]:
        """
        Retrieves episodic events semantically matching query.
        """
        if not query_embedding:
            query_embedding = await get_embedding(query)

        pool = await self._get_pool()
        if pool and not self._is_mock:
            try:
                async with pool.acquire() as conn:
                    emb_str = f"[{','.join(str(x) for x in query_embedding)}]"
                    query_sql = """
                    SELECT id, user_id, session_id, timestamp, summary, raw_context, tags, turn_range
                    FROM episodic_events
                    WHERE user_id = $1
                    ORDER BY embedding <=> $2::vector ASC
                    LIMIT $3;
                    """
                    rows = await conn.fetch(query_sql, user_id, emb_str, top_k)
                    return [
                        EpisodicEventRecord(
                            id=str(r["id"]),
                            user_id=r["user_id"],
                            session_id=r["session_id"],
                            timestamp=r["timestamp"],
                            summary=r["summary"],
                            raw_context=r["raw_context"],
                            tags=list(r["tags"]) if r["tags"] else [],
                            turn_range=list(r["turn_range"]) if r["turn_range"] else None,
                        )
                        for r in rows
                    ]
            except Exception as e:
                logger.warning("pg_search_episodic_failed", error=str(e))
                self._is_mock = True

        user_events = [ev for ev in _SHARED_EVENTS.values() if ev.user_id == user_id]
        scored = []
        for ev in user_events:
            emb = ev.embedding or generate_local_embedding(ev.summary)
            score = cosine_similarity(query_embedding, emb)
            scored.append((score, ev))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    async def get_all_edges_history(self, user_id: str) -> List[GraphEdgeRecord]:
        """
        Returns full historical graph edges for audit and visualization.
        """
        pool = await self._get_pool()
        if pool and not self._is_mock:
            try:
                async with pool.acquire() as conn:
                    query = """
                    SELECT id, user_id, subject, predicate, object, confidence,
                           valid_from, valid_until, observed_at, superseded_by,
                           source_session_id, source_turn_ids, metadata
                    FROM temporal_edges
                    WHERE user_id = $1
                    ORDER BY observed_at DESC;
                    """
                    rows = await conn.fetch(query, user_id)
                    return [self._row_to_edge(row) for row in rows]
            except Exception as e:
                logger.warning("pg_get_history_failed", error=str(e))
                self._is_mock = True

        records = [e for e in _SHARED_EDGES.values() if e.user_id == user_id]
        records.sort(key=lambda e: e.observed_at, reverse=True)
        return records

    async def clear_all(self, user_id: str) -> None:
        global _SHARED_EDGES, _SHARED_EVENTS
        pool = await self._get_pool()
        if pool and not self._is_mock:
            try:
                async with pool.acquire() as conn:
                    await conn.execute("DELETE FROM temporal_edges WHERE user_id = $1;", user_id)
                    await conn.execute("DELETE FROM episodic_events WHERE user_id = $1;", user_id)
            except Exception:
                pass
        _SHARED_EDGES = {k: v for k, v in _SHARED_EDGES.items() if v.user_id != user_id}
        _SHARED_EVENTS = {k: v for k, v in _SHARED_EVENTS.items() if v.user_id != user_id}

    def _row_to_edge(self, row: Any) -> GraphEdgeRecord:
        import json
        source_turns = row["source_turn_ids"]
        if isinstance(source_turns, str):
            source_turns = json.loads(source_turns)
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return GraphEdgeRecord(
            id=str(row["id"]),
            user_id=row["user_id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            confidence=float(row["confidence"]),
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            observed_at=row["observed_at"],
            superseded_by=str(row["superseded_by"]) if row["superseded_by"] else None,
            source_session_id=row["source_session_id"],
            source_turn_ids=list(source_turns) if source_turns else [],
            metadata=dict(metadata) if metadata else {},
        )
