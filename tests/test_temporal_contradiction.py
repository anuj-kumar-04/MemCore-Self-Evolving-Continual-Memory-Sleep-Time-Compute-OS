"""
Unit & Integration Test: Temporal Contradiction & Fact Drift Resolution
Proves old facts expire when contradicted without losing history.
Verifies:
1. January assertion (MySQL) creates active edge (valid_until is NULL).
2. October assertion (PostgreSQL 16) expires MySQL edge (valid_until set, superseded_by linked).
3. Active truth retrieval ONLY returns PostgreSQL 16 (zero memory poisoning).
4. Historical time-travel query at January 2026 still accurately reconstructs MySQL.
"""

import pytest
from datetime import datetime, timezone
from src.api.schemas import TurnRecord, TripletFact
from src.memory_tiers.working_redis import WorkingMemoryRedis
from src.memory_tiers.temporal_graph_pg import BiTemporalGraphPG
from src.memory_tiers.procedural_playbook import ProceduralPlaybookManager
from src.sleep_engine.conflict_resolver import BiTemporalConflictResolver
from src.sleep_engine.context_compiler import ContextCompiler
from src.sleep_engine.celery_app import execute_sleep_consolidation


@pytest.mark.asyncio
async def test_temporal_contradiction_lifecycle():
    user_id = "test_user_temporal"
    session_id = "test_session_temporal"

    working_memory = WorkingMemoryRedis()
    temporal_graph = BiTemporalGraphPG()
    playbook_manager = ProceduralPlaybookManager()
    conflict_resolver = BiTemporalConflictResolver(temporal_graph)
    context_compiler = ContextCompiler(working_memory, temporal_graph, playbook_manager)

    # Clean test state
    await working_memory.clear_all(user_id, session_id)
    await temporal_graph.clear_all(user_id)
    await playbook_manager.clear_playbook(user_id)

    # --------------------------------------------------------------------------
    # Step 1: January 2026 Fact Assertion (MySQL)
    # --------------------------------------------------------------------------
    t_january = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    jan_turn = TurnRecord(
        session_id=session_id,
        user_id=user_id,
        role="user",
        content="Our backend runs on MySQL and AWS ECS.",
        timestamp=t_january
    )
    await working_memory.append_turn(jan_turn)

    # Execute sleep consolidation at January timestamp
    jan_summary = await execute_sleep_consolidation(
        user_id=user_id,
        session_id=session_id,
        force=True,
        timestamp_override=t_january
    )

    assert jan_summary.turns_processed >= 1
    assert jan_summary.new_facts_added >= 1

    # Verify January active edge
    jan_active = await temporal_graph.get_active_facts(user_id=user_id, subject="backend", predicate="USES_DB")
    assert len(jan_active) == 1
    assert jan_active[0].object == "MySQL"
    assert jan_active[0].valid_until is None
    assert jan_active[0].is_active is True
    original_edge_id = jan_active[0].id

    # --------------------------------------------------------------------------
    # Step 2: October 2026 Contradiction Fact (PostgreSQL 16)
    # --------------------------------------------------------------------------
    t_october = datetime(2026, 10, 16, 15, 0, 0, tzinfo=timezone.utc)
    oct_turn = TurnRecord(
        session_id=session_id,
        user_id=user_id,
        role="user",
        content="We migrated completely from MySQL to PostgreSQL 16 on Kubernetes.",
        timestamp=t_october
    )
    await working_memory.append_turn(oct_turn)

    # Execute sleep consolidation at October timestamp
    oct_summary = await execute_sleep_consolidation(
        user_id=user_id,
        session_id=session_id,
        force=True,
        timestamp_override=t_october
    )

    # Verify conflict resolver detected contradiction
    assert oct_summary.contradictions_resolved >= 1

    # --------------------------------------------------------------------------
    # Step 3: Verify Active Truth vs Expired History
    # --------------------------------------------------------------------------
    # Current active fact MUST be PostgreSQL 16
    current_active = await temporal_graph.get_active_facts(user_id=user_id, subject="backend", predicate="USES_DB")
    assert len(current_active) == 1
    assert current_active[0].object == "PostgreSQL 16"
    assert current_active[0].valid_until is None
    new_edge_id = current_active[0].id

    # Old fact (MySQL) MUST NOT be deleted; it MUST be marked expired with valid_until and superseded_by
    all_history = await temporal_graph.get_all_edges_history(user_id=user_id)
    mysql_edges = [e for e in all_history if e.id == original_edge_id]
    assert len(mysql_edges) == 1
    mysql_edge = mysql_edges[0]
    assert mysql_edge.valid_until is not None
    assert mysql_edge.is_active is False
    assert mysql_edge.superseded_by == new_edge_id

    # --------------------------------------------------------------------------
    # Step 4: Verify Fast-Path Zero Memory Poisoning
    # --------------------------------------------------------------------------
    messages, telemetry = await context_compiler.compile_context(
        user_id=user_id,
        session_id=session_id,
        user_message="Write a database migration script for our backend"
    )

    system_content = messages[0]["content"]
    # Verified active truth contains PostgreSQL 16
    assert "PostgreSQL 16" in system_content
    # Naive RAG failure (MySQL chunk in active context) is PREVENTED!
    assert "• [Active Fact]: backend USES_DB MySQL" not in system_content

    # --------------------------------------------------------------------------
    # Step 5: Verify Time-Travel Historical Auditability
    # --------------------------------------------------------------------------
    # Query what was true at January 20, 2026
    t_query_jan = datetime(2026, 1, 20, 0, 0, 0, tzinfo=timezone.utc)
    jan_historical = await temporal_graph.query_as_of(
        user_id=user_id,
        as_of_time=t_query_jan,
        subject="backend",
        predicate="USES_DB"
    )
    assert len(jan_historical) == 1
    assert jan_historical[0].object == "MySQL"

    # Query what is true at October 16, 2026
    t_query_oct = datetime(2026, 10, 16, 16, 0, 0, tzinfo=timezone.utc)
    oct_historical = await temporal_graph.query_as_of(
        user_id=user_id,
        as_of_time=t_query_oct,
        subject="backend",
        predicate="USES_DB"
    )
    assert len(oct_historical) == 1
    assert oct_historical[0].object == "PostgreSQL 16"
