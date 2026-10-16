"""
Tier 3: Procedural Memory (Self-Evolving Playbook)
PostgreSQL JSONB + Redis Cache.
Learned user rules, coding preferences, workflow corrections, and few-shot behavioral constraints.
Distilled during Sleep-Time when the Critic detects user corrections or negative feedback in chat logs.
"""

from __future__ import annotations
import os
import json
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
import structlog

from src.api.schemas import ProceduralPlaybookRecord, ProceduralRule

logger = structlog.get_logger(__name__)

# Shared fallback state across all instances
_SHARED_PLAYBOOKS: Dict[str, ProceduralPlaybookRecord] = {}
_PB_DB_UNAVAILABLE = False
_PB_REDIS_UNAVAILABLE = False

class ProceduralPlaybookManager:
    def __init__(self, db_url: Optional[str] = None, redis_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL", "postgresql://memcore_user:memcore_secret@localhost:5432/memcore")
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._db_pool = None
        self._redis_client = None
        self._is_db_mock = False
        self._is_redis_mock = False

    def _redis_cache_key(self, user_id: str) -> str:
        return f"memcore:playbook:{user_id}"

    async def _get_db_pool(self):
        global _PB_DB_UNAVAILABLE
        if self._db_pool is not None:
            return self._db_pool
        if _PB_DB_UNAVAILABLE or self._is_db_mock:
            self._is_db_mock = True
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
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            self._db_pool = pool
            self._is_db_mock = False
            return self._db_pool
        except Exception:
            _PB_DB_UNAVAILABLE = True
            self._is_db_mock = True
            return None

    async def _get_redis_client(self):
        global _PB_REDIS_UNAVAILABLE
        if self._redis_client is not None:
            return self._redis_client
        if _PB_REDIS_UNAVAILABLE or self._is_redis_mock:
            self._is_redis_mock = True
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
            self._redis_client = client
            self._is_redis_mock = False
            return self._redis_client
        except Exception:
            _PB_REDIS_UNAVAILABLE = True
            self._is_redis_mock = True
            return None
            self._is_redis_mock = True
            return None

    @staticmethod
    def compile_system_patch(rules: List[ProceduralRule]) -> str:
        """
        Compiles the active procedural rules into an authoritative system instruction patch.
        """
        if not rules:
            return ""

        sorted_rules = sorted(rules, key=lambda r: (r.occurrences, r.confidence), reverse=True)
        lines = [
            "### [MEMCORE CONTINUAL PROCEDURAL PLAYBOOK — STRICT ENFORCEMENT]:",
            "The following behavioral rules have been learned directly from the user's explicit corrections across past sessions. You MUST strictly adhere to them:",
        ]

        categories: Dict[str, List[ProceduralRule]] = {}
        for r in sorted_rules:
            cat = r.category.upper().replace("_", " ")
            categories.setdefault(cat, []).append(r)

        for cat, cat_rules in categories.items():
            lines.append(f"\n#### {cat}:")
            for r in cat_rules:
                trigger_txt = f" (Trigger when: {r.trigger_pattern})" if r.trigger_pattern else ""
                lines.append(f"- **{r.rule}**{trigger_txt} [Confidence: {r.confidence:.2f}, Reinforcements: {r.occurrences}]")

        return "\n".join(lines)

    async def get_playbook(self, user_id: str) -> ProceduralPlaybookRecord:
        """
        Retrieves user playbook, checking Redis cache first (<1ms Fast Path).
        """
        cache_key = self._redis_cache_key(user_id)
        redis_client = await self._get_redis_client()
        if redis_client and not self._is_redis_mock:
            try:
                cached_json = await redis_client.get(cache_key)
                if cached_json:
                    return ProceduralPlaybookRecord.model_validate_json(cached_json)
            except Exception:
                self._is_redis_mock = True

        # Check Postgres
        pool = await self._get_db_pool()
        if pool and not self._is_db_mock:
            try:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT user_id, rules, system_patch, version, updated_at FROM procedural_playbooks WHERE user_id = $1",
                        user_id
                    )
                    if row:
                        rules_raw = row["rules"]
                        if isinstance(rules_raw, str):
                            rules_raw = json.loads(rules_raw)
                        rules = [ProceduralRule.model_validate(r) for r in (rules_raw or [])]
                        pb = ProceduralPlaybookRecord(
                            user_id=row["user_id"],
                            rules=rules,
                            system_patch=row["system_patch"] or self.compile_system_patch(rules),
                            version=row["version"],
                            updated_at=row["updated_at"]
                        )
                        # Warm cache
                        if redis_client and not self._is_redis_mock:
                            await redis_client.set(cache_key, pb.model_dump_json(), ex=3600)
                        return pb
            except Exception as e:
                logger.warning("pg_get_playbook_failed", error=str(e))
                self._is_db_mock = True

        # Fallback in-memory
        if user_id in _SHARED_PLAYBOOKS:
            return _SHARED_PLAYBOOKS[user_id]

        empty_pb = ProceduralPlaybookRecord(user_id=user_id, rules=[], system_patch="")
        _SHARED_PLAYBOOKS[user_id] = empty_pb
        return empty_pb

    async def get_system_patch(self, user_id: str) -> str:
        """
        Ultra-fast retrieval of the pre-compiled system instruction patch (<1ms).
        """
        playbook = await self.get_playbook(user_id)
        return playbook.system_patch

    async def add_or_update_rule(
        self,
        user_id: str,
        candidate: ProceduralRule
    ) -> Tuple[ProceduralPlaybookRecord, bool]:
        """
        Adds a new learned rule or increments confidence/occurrences if a similar rule already exists.
        Returns (updated_playbook, is_new_rule).
        """
        playbook = await self.get_playbook(user_id)
        rules = list(playbook.rules)
        cand_norm = candidate.rule.strip().lower()

        is_new = True
        for idx, existing in enumerate(rules):
            exist_norm = existing.rule.strip().lower()
            # Check overlap or exact match
            if cand_norm == exist_norm or (len(cand_norm) > 10 and cand_norm in exist_norm):
                # Update existing rule
                existing.occurrences += 1
                existing.confidence = min(1.0, existing.confidence + 0.05)
                existing.last_applied_at = datetime.now(timezone.utc)
                if candidate.source_correction:
                    existing.source_correction = candidate.source_correction
                rules[idx] = existing
                is_new = False
                break

        if is_new:
            rules.append(candidate)

        new_patch = self.compile_system_patch(rules)
        updated_pb = ProceduralPlaybookRecord(
            user_id=user_id,
            rules=rules,
            system_patch=new_patch,
            version=playbook.version + 1,
            updated_at=datetime.now(timezone.utc)
        )

        # Save to DB
        pool = await self._get_db_pool()
        if pool and not self._is_db_mock:
            try:
                async with pool.acquire() as conn:
                    rules_json = json.dumps([r.model_dump(mode="json") for r in rules])
                    upsert_query = """
                    INSERT INTO procedural_playbooks (user_id, rules, system_patch, version, updated_at)
                    VALUES ($1, $2::jsonb, $3, $4, $5)
                    ON CONFLICT (user_id) DO UPDATE
                    SET rules = EXCLUDED.rules,
                        system_patch = EXCLUDED.system_patch,
                        version = procedural_playbooks.version + 1,
                        updated_at = EXCLUDED.updated_at;
                    """
                    await conn.execute(
                        upsert_query,
                        user_id, rules_json, new_patch, updated_pb.version, updated_pb.updated_at
                    )
            except Exception as e:
                logger.warning("pg_save_playbook_failed", error=str(e))
                self._is_db_mock = True

        # Invalidate / update Redis cache
        cache_key = self._redis_cache_key(user_id)
        redis_client = await self._get_redis_client()
        if redis_client and not self._is_redis_mock:
            try:
                await redis_client.set(cache_key, updated_pb.model_dump_json(), ex=3600)
            except Exception:
                pass

        # Update in-memory fallback
        _SHARED_PLAYBOOKS[user_id] = updated_pb
        return updated_pb, is_new

    async def clear_playbook(self, user_id: str) -> None:
        global _SHARED_PLAYBOOKS
        pool = await self._get_db_pool()
        if pool and not self._is_db_mock:
            try:
                async with pool.acquire() as conn:
                    await conn.execute("DELETE FROM procedural_playbooks WHERE user_id = $1;", user_id)
            except Exception:
                pass

        cache_key = self._redis_cache_key(user_id)
        redis_client = await self._get_redis_client()
        if redis_client and not self._is_redis_mock:
            try:
                await redis_client.delete(cache_key)
            except Exception:
                pass

        _SHARED_PLAYBOOKS.pop(user_id, None)
