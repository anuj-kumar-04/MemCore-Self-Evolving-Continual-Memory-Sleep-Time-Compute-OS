"""
Unit & Integration Test: Procedural Memory & Self-Evolving Playbook
Proves user corrections auto-update the system instruction playbook.
Verifies:
1. User correction ("Stop using os.path, always use pathlib and ruff formatting")
   is detected by Extractor.
2. Sleep-Time distillation synthesizes an authoritative ProceduralRule.
3. Rule is stored in Playbook JSONB and cached in Redis.
4. Fast-Path Context Compiler embeds the compiled system patch in <50ms.
5. Repeated correction reinforces confidence and occurrence counter without duplication.
"""

import pytest
from datetime import datetime, timezone
from src.api.schemas import TurnRecord, ProceduralRule
from src.memory_tiers.working_redis import WorkingMemoryRedis
from src.memory_tiers.temporal_graph_pg import BiTemporalGraphPG
from src.memory_tiers.procedural_playbook import ProceduralPlaybookManager
from src.sleep_engine.context_compiler import ContextCompiler
from src.sleep_engine.celery_app import execute_sleep_consolidation


@pytest.mark.asyncio
async def test_procedural_evolution_lifecycle():
    user_id = "test_user_procedural"
    session_id = "test_session_procedural"

    working_memory = WorkingMemoryRedis()
    temporal_graph = BiTemporalGraphPG()
    playbook_manager = ProceduralPlaybookManager()
    context_compiler = ContextCompiler(working_memory, temporal_graph, playbook_manager)

    # Clean test state
    await working_memory.clear_all(user_id, session_id)
    await temporal_graph.clear_all(user_id)
    await playbook_manager.clear_playbook(user_id)

    # Verify initial playbook is empty
    initial_pb = await playbook_manager.get_playbook(user_id)
    assert len(initial_pb.rules) == 0
    assert initial_pb.system_patch == ""

    # --------------------------------------------------------------------------
    # Step 1: User Correction Turn
    # --------------------------------------------------------------------------
    correction_turn = TurnRecord(
        session_id=session_id,
        user_id=user_id,
        role="user",
        content="Stop using os.path, always use pathlib and ruff formatting in my repo."
    )
    await working_memory.append_turn(correction_turn)

    # Trigger Sleep Consolidation
    summary_1 = await execute_sleep_consolidation(user_id=user_id, session_id=session_id, force=True)
    assert summary_1.rules_distilled >= 1

    # --------------------------------------------------------------------------
    # Step 2: Verify Playbook State
    # --------------------------------------------------------------------------
    pb = await playbook_manager.get_playbook(user_id)
    assert len(pb.rules) >= 1
    rule = pb.rules[0]
    assert "pathlib" in rule.rule.lower()
    assert rule.category == "coding_style"
    assert rule.occurrences == 1
    assert rule.confidence >= 0.9

    # Verify Compiled System Instruction Patch
    patch = await playbook_manager.get_system_patch(user_id)
    assert "MEMCORE CONTINUAL PROCEDURAL PLAYBOOK" in patch
    assert "pathlib" in patch.lower()
    assert "CODING STYLE" in patch

    # --------------------------------------------------------------------------
    # Step 3: Verify Fast-Path Context Injection
    # --------------------------------------------------------------------------
    messages, telemetry = await context_compiler.compile_context(
        user_id=user_id,
        session_id=session_id,
        user_message="Write a python script to scan files in a folder"
    )

    system_prompt = messages[0]["content"]
    assert "MEMCORE CONTINUAL PROCEDURAL PLAYBOOK" in system_prompt
    assert "pathlib" in system_prompt.lower()
    assert telemetry.latency_ms < 50.0  # Strict SLA (<50ms)
    assert telemetry.breakdown["playbook_tokens"] > 0
    assert len(telemetry.playbook_rules_applied) >= 1

    # --------------------------------------------------------------------------
    # Step 4: Reinforcement Learning (Second Correction)
    # --------------------------------------------------------------------------
    second_correction = TurnRecord(
        session_id=session_id,
        user_id=user_id,
        role="user",
        content="Remember: stop using os.path, always use pathlib in all scripts."
    )
    await working_memory.append_turn(second_correction)

    summary_2 = await execute_sleep_consolidation(user_id=user_id, session_id=session_id, force=True)
    
    # Check playbook deduplication and reinforcement
    pb_reinforced = await playbook_manager.get_playbook(user_id)
    # Must NOT create a duplicate rule; it must increment occurrences
    assert len(pb_reinforced.rules) == 1
    assert pb_reinforced.rules[0].occurrences == 2
    assert pb_reinforced.rules[0].confidence >= rule.confidence
