"""
Sleep-Time Compute: Celery Background Daemon & Task Orchestrator
Processes session consolidation asynchronously when idle or triggered:
1. Episodic & Fact Extractor
2. Bi-Temporal Conflict Resolver (NOOP / CONTRADICTION / NEW_FACT)
3. Procedural Playbook Distiller
4. Working Memory Flush
"""

from __future__ import annotations
import os
import time
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from celery import Celery
import structlog

from src.api.schemas import ConsolidationSummary, EpisodicEventRecord
from src.memory_tiers.working_redis import WorkingMemoryRedis
from src.memory_tiers.temporal_graph_pg import BiTemporalGraphPG
from src.memory_tiers.procedural_playbook import ProceduralPlaybookManager
from src.sleep_engine.extractor import SleepExtractor
from src.sleep_engine.conflict_resolver import BiTemporalConflictResolver

logger = structlog.get_logger(__name__)

REDIS_BROKER = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
REDIS_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

celery_app = Celery(
    "memcore_sleep_os",
    broker=REDIS_BROKER,
    backend=REDIS_BACKEND
)

celery_app.conf.update(
    task_default_queue="memcore_sleep_queue",
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)

# Global memory instances for worker / direct execution
_working_memory: Optional[WorkingMemoryRedis] = None
_temporal_graph: Optional[BiTemporalGraphPG] = None
_playbook_manager: Optional[ProceduralPlaybookManager] = None
_extractor: Optional[SleepExtractor] = None
_conflict_resolver: Optional[BiTemporalConflictResolver] = None


def get_consolidation_pipeline():
    global _working_memory, _temporal_graph, _playbook_manager, _extractor, _conflict_resolver
    if _working_memory is None:
        _working_memory = WorkingMemoryRedis()
        _temporal_graph = BiTemporalGraphPG()
        _playbook_manager = ProceduralPlaybookManager()
        _extractor = SleepExtractor()
        _conflict_resolver = BiTemporalConflictResolver(_temporal_graph)
    return _working_memory, _temporal_graph, _playbook_manager, _extractor, _conflict_resolver


async def execute_sleep_consolidation(
    user_id: str,
    session_id: str,
    force: bool = False,
    timestamp_override: Optional[datetime] = None
) -> ConsolidationSummary:
    """
    Executes the 3-step Sleep-Time consolidation pipeline:
    1. Extracts facts & user corrections
    2. Mutates bi-temporal graph (resolving contradictions)
    3. Distills procedural playbook rules
    4. Trims working memory
    """
    t0 = time.perf_counter()
    working_mem, graph, playbook_mgr, extractor, resolver = get_consolidation_pipeline()

    # Step 1: Retrieve unconsolidated turns
    if force:
        turns = await working_mem.get_recent_turns(user_id, session_id, limit=50)
    else:
        turns = await working_mem.get_unconsolidated_turns(user_id, session_id)

    if not turns:
        logger.info("no_unconsolidated_turns_skipping_sleep", user_id=user_id, session_id=session_id)
        return ConsolidationSummary(
            user_id=user_id,
            session_id=session_id,
            turns_processed=0,
            facts_extracted=0,
            contradictions_resolved=0,
            noops=0,
            new_facts_added=0,
            rules_distilled=0,
            duration_ms=0.0,
            diffs=[],
            rules=[]
        )

    # Step 2: Run Extractor (facts, episodic summary, corrections)
    extracted = await extractor.extract_from_turns(turns)

    # Step 3: Run Bi-Temporal Conflict Resolver (Graph Diff Engine)
    diffs = await resolver.resolve_triplets(
        user_id=user_id,
        session_id=session_id,
        triplets=extracted.facts,
        timestamp_override=timestamp_override
    )

    contradictions = sum(1 for d in diffs if d.action == "CONTRADICTION")
    noops = sum(1 for d in diffs if d.action == "NOOP")
    new_facts = sum(1 for d in diffs if d.action == "NEW_FACT")

    # Step 4: Run Procedural Playbook Distiller
    distilled_rules = []
    for cand in extracted.rule_candidates:
        updated_pb, is_new = await playbook_mgr.add_or_update_rule(user_id, cand)
        distilled_rules.append(cand)
        logger.info("procedural_rule_distilled", rule=cand.rule, is_new=is_new, user_id=user_id)

    # Step 5: Save Episodic Summary Milestone
    if extracted.session_summary and len(turns) > 0:
        event = EpisodicEventRecord(
            user_id=user_id,
            session_id=session_id,
            timestamp=timestamp_override or datetime.now(timezone.utc),
            summary=extracted.session_summary,
            raw_context=f"{len(turns)} turns consolidated.",
            turn_range=[0, len(turns)]
        )
        await graph.add_episodic_event(event)

    # Step 6: Mark Working Memory Consolidated
    await working_mem.mark_consolidated(user_id, session_id)

    duration_ms = (time.perf_counter() - t0) * 1000

    summary = ConsolidationSummary(
        user_id=user_id,
        session_id=session_id,
        turns_processed=len(turns),
        facts_extracted=len(extracted.facts),
        contradictions_resolved=contradictions,
        noops=noops,
        new_facts_added=new_facts,
        rules_distilled=len(distilled_rules),
        duration_ms=round(duration_ms, 2),
        diffs=diffs,
        rules=distilled_rules
    )

    logger.info(
        "sleep_consolidation_completed",
        user_id=user_id,
        session_id=session_id,
        turns=len(turns),
        contradictions=contradictions,
        new_facts=new_facts,
        rules=len(distilled_rules),
        duration_ms=duration_ms
    )
    return summary


@celery_app.task(name="memcore.sleep_consolidation_task")
def sleep_consolidation_task(user_id: str, session_id: str, force: bool = False) -> Dict[str, Any]:
    """
    Celery background worker task for sleep-time compute.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        summary = loop.run_until_complete(
            execute_sleep_consolidation(user_id=user_id, session_id=session_id, force=force)
        )
        return summary.model_dump(mode="json")
    finally:
        loop.close()
