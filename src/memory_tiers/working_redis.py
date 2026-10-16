"""
Tier 1: Working Memory (Redis Sorted Sets + Inactivity Tracking)
Sliding window of the last N raw conversation turns + active scratchpad variables.
Synchronously appended during live chat (<5ms). Flushed/flagged after Sleep-Time consolidation.
"""

from __future__ import annotations
import json
import time
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import structlog

from src.api.schemas import TurnRecord, ScratchpadState

logger = structlog.get_logger(__name__)

# Shared fallback state across all instances
_SHARED_TURNS: Dict[str, List[TurnRecord]] = {}
_SHARED_SCRATCHPADS: Dict[str, Dict[str, Any]] = {}
_SHARED_UNCONSOLIDATED: Dict[str, int] = {}
_SHARED_LAST_ACTIVE: Dict[str, float] = {}
_REDIS_UNAVAILABLE = False

class WorkingMemoryRedis:
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client = None
        self._is_mock = False

    async def _get_client(self):
        global _REDIS_UNAVAILABLE
        if self._client is not None:
            return self._client
        if _REDIS_UNAVAILABLE or self._is_mock:
            self._is_mock = True
            return None
        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=0.3,
                socket_timeout=0.3,
            )
            await client.ping()
            self._client = client
            self._is_mock = False
            logger.info("working_memory_redis_connected", url=self.redis_url)
            return self._client
        except Exception as e:
            logger.warning("redis_connection_failed_falling_back_to_in_memory", error=str(e))
            _REDIS_UNAVAILABLE = True
            self._is_mock = True
            return None

    def _key_turns(self, user_id: str, session_id: str) -> str:
        return f"memcore:working:{user_id}:{session_id}:turns"

    def _key_unconsolidated(self, user_id: str, session_id: str) -> str:
        return f"memcore:working:{user_id}:{session_id}:unconsolidated"

    def _key_scratchpad(self, user_id: str, session_id: str) -> str:
        return f"memcore:working:{user_id}:{session_id}:scratchpad"

    def _key_last_active(self, user_id: str, session_id: str) -> str:
        return f"memcore:working:{user_id}:{session_id}:last_active"

    async def append_turn(self, turn: TurnRecord) -> TurnRecord:
        """
        Synchronously append turn to Redis sorted set (<5ms target latency).
        """
        t0 = time.perf_counter()
        session_key = f"{turn.user_id}:{turn.session_id}"
        turn_data = turn.model_dump_json()
        score = turn.timestamp.timestamp()

        client = await self._get_client()
        if client and not self._is_mock:
            try:
                pipe = client.pipeline(transaction=True)
                turns_key = self._key_turns(turn.user_id, turn.session_id)
                unconsolidated_key = self._key_unconsolidated(turn.user_id, turn.session_id)
                last_active_key = self._key_last_active(turn.user_id, turn.session_id)

                pipe.zadd(turns_key, {turn_data: score})
                pipe.incr(unconsolidated_key)
                pipe.set(last_active_key, score, ex=86400)
                pipe.expire(turns_key, 86400)
                await pipe.execute()
                latency_ms = (time.perf_counter() - t0) * 1000
                logger.debug("turn_appended_redis", latency_ms=latency_ms, turn_id=turn.id)
                return turn
            except Exception as e:
                logger.error("redis_append_error_using_fallback", error=str(e))
                self._is_mock = True

        # Fallback in-memory
        if session_key not in _SHARED_TURNS:
            _SHARED_TURNS[session_key] = []
        _SHARED_TURNS[session_key].append(turn)
        _SHARED_UNCONSOLIDATED[session_key] = _SHARED_UNCONSOLIDATED.get(session_key, 0) + 1
        _SHARED_LAST_ACTIVE[session_key] = score
        return turn

    async def get_recent_turns(self, user_id: str, session_id: str, limit: int = 6) -> List[TurnRecord]:
        """
        Retrieves the last N raw conversation turns in chronological order.
        """
        client = await self._get_client()
        if client and not self._is_mock:
            try:
                turns_key = self._key_turns(user_id, session_id)
                # Redis ZRANGE with REV to get latest, then reverse back to chronological
                raw_items = await client.zrange(turns_key, -limit, -1)
                records = [TurnRecord.model_validate_json(item) for item in raw_items]
                records.sort(key=lambda t: t.timestamp)
                return records
            except Exception as e:
                logger.warning("redis_get_recent_failed", error=str(e))
                self._is_mock = True

        session_key = f"{user_id}:{session_id}"
        all_turns = _SHARED_TURNS.get(session_key, [])
        return all_turns[-limit:] if all_turns else []

    async def get_unconsolidated_turns(self, user_id: str, session_id: str) -> List[TurnRecord]:
        """
        Retrieves all turns that have occurred since the last sleep-time consolidation.
        """
        client = await self._get_client()
        if client and not self._is_mock:
            try:
                unconsolidated_key = self._key_unconsolidated(user_id, session_id)
                count_str = await client.get(unconsolidated_key)
                count = int(count_str) if count_str else 0
                if count <= 0:
                    return []
                turns_key = self._key_turns(user_id, session_id)
                raw_items = await client.zrange(turns_key, -count, -1)
                records = [TurnRecord.model_validate_json(item) for item in raw_items]
                records.sort(key=lambda t: t.timestamp)
                return records
            except Exception as e:
                logger.warning("redis_get_unconsolidated_failed", error=str(e))
                self._is_mock = True

        session_key = f"{user_id}:{session_id}"
        all_turns = _SHARED_TURNS.get(session_key, [])
        count = _SHARED_UNCONSOLIDATED.get(session_key, 0)
        return all_turns[-count:] if count > 0 else []

    async def mark_consolidated(self, user_id: str, session_id: str, retain_window: int = 10) -> None:
        """
        Resets unconsolidated counter and trims working memory buffer to retain_window.
        """
        client = await self._get_client()
        if client and not self._is_mock:
            try:
                unconsolidated_key = self._key_unconsolidated(user_id, session_id)
                turns_key = self._key_turns(user_id, session_id)
                pipe = client.pipeline(transaction=True)
                pipe.set(unconsolidated_key, 0)
                # Trim sorted set to keep only the latest retain_window entries
                total_count = await client.zcard(turns_key)
                if total_count > retain_window:
                    pipe.zremrangebyrank(turns_key, 0, total_count - retain_window - 1)
                await pipe.execute()
                logger.info("working_memory_marked_consolidated", user_id=user_id, session_id=session_id)
                return
            except Exception as e:
                logger.warning("redis_mark_consolidated_failed", error=str(e))
                self._is_mock = True

        session_key = f"{user_id}:{session_id}"
        _SHARED_UNCONSOLIDATED[session_key] = 0
        if session_key in _SHARED_TURNS:
            _SHARED_TURNS[session_key] = _SHARED_TURNS[session_key][-retain_window:]

    async def get_session_stats(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Returns working memory diagnostic statistics.
        """
        client = await self._get_client()
        if client and not self._is_mock:
            try:
                turns_key = self._key_turns(user_id, session_id)
                unconsolidated_key = self._key_unconsolidated(user_id, session_id)
                last_active_key = self._key_last_active(user_id, session_id)

                total = await client.zcard(turns_key)
                unconsolidated = await client.get(unconsolidated_key)
                last_active = await client.get(last_active_key)
                return {
                    "total_turns_buffered": total or 0,
                    "unconsolidated_turns": int(unconsolidated) if unconsolidated else 0,
                    "last_active_epoch": float(last_active) if last_active else None,
                    "engine": "redis",
                }
            except Exception as e:
                logger.warning("redis_stats_failed", error=str(e))

        session_key = f"{user_id}:{session_id}"
        all_turns = _SHARED_TURNS.get(session_key, [])
        return {
            "total_turns_buffered": len(all_turns),
            "unconsolidated_turns": _SHARED_UNCONSOLIDATED.get(session_key, 0),
            "last_active_epoch": _SHARED_LAST_ACTIVE.get(session_key),
            "engine": "in-memory-fallback",
        }

    async def clear_all(self, user_id: str, session_id: str) -> None:
        client = await self._get_client()
        if client and not self._is_mock:
            try:
                await client.delete(
                    self._key_turns(user_id, session_id),
                    self._key_unconsolidated(user_id, session_id),
                    self._key_scratchpad(user_id, session_id),
                    self._key_last_active(user_id, session_id)
                )
            except Exception:
                pass
        session_key = f"{user_id}:{session_id}"
        _SHARED_TURNS.pop(session_key, None)
        _SHARED_UNCONSOLIDATED.pop(session_key, None)
        _SHARED_LAST_ACTIVE.pop(session_key, None)
        _SHARED_SCRATCHPADS.pop(session_key, None)
