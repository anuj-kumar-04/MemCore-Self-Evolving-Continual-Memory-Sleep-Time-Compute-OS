"""
Unit Test: Fast-Path Context Compiler SLA & Submodular Token Budget Enforcer
Verifies:
1. Context assembly executes in <50ms (zero LLM overhead during retrieval).
2. Strict 2,000-token budget is never exceeded, even with large conversation histories.
3. Submodular pruning maximizes relevant active facts while pruning bloated turns.
"""

import pytest
import time
from src.api.schemas import TurnRecord, GraphEdgeRecord, ProceduralRule
from src.memory_tiers.working_redis import WorkingMemoryRedis
from src.memory_tiers.temporal_graph_pg import BiTemporalGraphPG
from src.memory_tiers.procedural_playbook import ProceduralPlaybookManager
from src.sleep_engine.context_compiler import ContextCompiler


@pytest.mark.asyncio
async def test_fast_path_token_budget_and_latency():
    user_id = "test_user_budget"
    session_id = "test_session_budget"

    working_memory = WorkingMemoryRedis()
    temporal_graph = BiTemporalGraphPG()
    playbook_manager = ProceduralPlaybookManager()
    
    # 2,000 Token budget
    compiler = ContextCompiler(
        working_memory=working_memory,
        temporal_graph=temporal_graph,
        playbook_manager=playbook_manager,
        token_budget=2000
    )

    await working_memory.clear_all(user_id, session_id)
    await temporal_graph.clear_all(user_id)
    await playbook_manager.clear_playbook(user_id)

    # 1. Add Procedural Rule
    await playbook_manager.add_or_update_rule(
        user_id,
        ProceduralRule(
            rule="Always output SQL queries using Common Table Expressions (CTEs); never use raw subqueries.",
            category="coding_style",
            confidence=0.99
        )
    )

    # 2. Add 20 active facts
    for i in range(20):
        await temporal_graph.add_edge(
            GraphEdgeRecord(
                user_id=user_id,
                subject=f"service_{i}",
                predicate="DEPENDS_ON",
                object=f"database_cluster_{i % 3}",
                confidence=0.95
            )
        )

    # 3. Add 30 long turns in working memory buffer
    for i in range(30):
        turn = TurnRecord(
            session_id=session_id,
            user_id=user_id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"Turn {i}: Here is some detailed technical context discussing service architecture and load balancing parameters with lots of extra token bloat for testing submodular knapsack limits."
        )
        await working_memory.append_turn(turn)

    # Compile context and measure latency
    t0 = time.perf_counter()
    messages, telemetry = await compiler.compile_context(
        user_id=user_id,
        session_id=session_id,
        user_message="Show me the architecture dependencies for service_3 and generate a query"
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    # Strict SLA Assertions
    assert latency_ms < 50.0, f"Context assembly took {latency_ms:.2f}ms, exceeding <50ms SLA!"
    assert telemetry.tokens_used <= 2000, f"Tokens used {telemetry.tokens_used} exceeded 2,000 budget!"
    assert telemetry.breakdown["playbook_tokens"] > 0
    assert telemetry.breakdown["active_facts_tokens"] > 0
    assert telemetry.breakdown["working_memory_tokens"] > 0
    
    # Verify submodular priority: Playbook and service_3 fact are in prompt
    system_text = messages[0]["content"]
    assert "Common Table Expressions" in system_text
    assert "service_3" in system_text
