"""
Fast-Path Context Compiler (<50ms Context Assembly — Zero LLM Overhead)
Enforces a strict 2,000-token budget via greedy submodular pruning.
Combines:
1. Procedural Playbook (Redis Cache)
2. Active Knowledge Graph Facts (valid_until IS NULL) + Episodic pgvector search
3. Last 6 raw conversation turns (Redis Working Memory)
"""

from __future__ import annotations
import time
import os
from typing import List, Optional, Dict, Any, Tuple
import structlog

from src.api.schemas import TurnRecord, GraphEdgeRecord, EpisodicEventRecord, ContextTelemetry
from src.memory_tiers.working_redis import WorkingMemoryRedis
from src.memory_tiers.temporal_graph_pg import BiTemporalGraphPG
from src.memory_tiers.procedural_playbook import ProceduralPlaybookManager

logger = structlog.get_logger(__name__)

# Lightweight token estimator
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Approximate 1 token ~ 3.8 characters for standard code and English
    return max(1, int(len(text) / 3.8))


class ContextCompiler:
    def __init__(
        self,
        working_memory: WorkingMemoryRedis,
        temporal_graph: BiTemporalGraphPG,
        playbook_manager: ProceduralPlaybookManager,
        token_budget: int = 2000
    ):
        self.working_memory = working_memory
        self.temporal_graph = temporal_graph
        self.playbook_manager = playbook_manager
        self.token_budget = int(os.getenv("FAST_PATH_TOKEN_BUDGET", token_budget))

    async def compile_context(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        as_of_time: Optional[Any] = None
    ) -> Tuple[List[Dict[str, str]], ContextTelemetry]:
        """
        Assembles LLM messages within the strict 2,000-token budget in <50ms.
        """
        t0 = time.perf_counter()

        # Step 1: Fetch Procedural Playbook (Tier 3 from Redis Cache)
        system_patch = await self.playbook_manager.get_system_patch(user_id)
        playbook_rec = await self.playbook_manager.get_playbook(user_id)

        # Step 2: Fetch Active Knowledge Graph Facts (Tier 2)
        # Note: If as_of_time is provided (Time-Travel mode), query historical state!
        if as_of_time:
            active_facts = await self.temporal_graph.query_as_of(user_id, as_of_time=as_of_time)
        else:
            # Active facts only (valid_until IS NULL)
            active_facts = await self.temporal_graph.get_active_facts(user_id)

        # Also retrieve relevant episodic events via hybrid search
        episodic_events = await self.temporal_graph.search_episodic_events(
            user_id=user_id,
            query=user_message,
            top_k=3
        )

        # Step 3: Fetch Last 6 Raw Turns from Working Memory (Tier 1 Redis)
        recent_turns = await self.working_memory.get_recent_turns(
            user_id=user_id,
            session_id=session_id,
            limit=6
        )

        # ----------------------------------------------------------------------
        # Submodular Token Budget Allocation & Pruning
        # ----------------------------------------------------------------------
        remaining_budget = self.token_budget
        breakdown = {
            "playbook_tokens": 0,
            "active_facts_tokens": 0,
            "episodic_tokens": 0,
            "working_memory_tokens": 0,
        }
        pruned_count = 0

        # Base System Instruction
        base_system = (
            "You are MemCore AI, a continuous memory operating assistant.\n"
            "You have access to verified, non-poisoned, bi-temporal facts and user procedural rules.\n"
            "Always state facts with precision and strictly follow the procedural rules below."
        )
        base_system_tokens = estimate_tokens(base_system)
        remaining_budget -= base_system_tokens

        # 1. Procedural Playbook Allocation (High priority)
        playbook_tokens = estimate_tokens(system_patch)
        if playbook_tokens <= remaining_budget:
            effective_patch = system_patch
            remaining_budget -= playbook_tokens
            breakdown["playbook_tokens"] = playbook_tokens
        else:
            # Truncate if extreme
            effective_patch = system_patch[: int(remaining_budget * 3.5)]
            breakdown["playbook_tokens"] = remaining_budget
            remaining_budget = 0
            pruned_count += 1

        # 2. Active Graph Facts Assembly (Highest factual relevance)
        # Sort facts by lexical or semantic overlap with user query
        query_words = set(user_message.lower().split())
        scored_facts: List[Tuple[float, GraphEdgeRecord]] = []
        for f in active_facts:
            fact_str = f"{f.subject} {f.predicate} {f.object}".lower()
            overlap = sum(1 for w in query_words if w in fact_str)
            score = f.confidence + (overlap * 2.0)
            scored_facts.append((score, f))
        scored_facts.sort(key=lambda x: x[0], reverse=True)

        selected_facts: List[str] = []
        for _, fact in scored_facts:
            f_str = f"• [Active Fact]: {fact.subject} {fact.predicate} {fact.object}"
            f_toks = estimate_tokens(f_str)
            if remaining_budget >= f_toks:
                selected_facts.append(f_str)
                remaining_budget -= f_toks
                breakdown["active_facts_tokens"] += f_toks
            else:
                pruned_count += 1

        # 3. Episodic Events Assembly
        selected_events: List[str] = []
        for ev in episodic_events:
            ev_str = f"• [Episodic Memory ({ev.timestamp.strftime('%Y-%m-%d')}]: {ev.summary}"
            ev_toks = estimate_tokens(ev_str)
            if remaining_budget >= ev_toks:
                selected_events.append(ev_str)
                remaining_budget -= ev_toks
                breakdown["episodic_tokens"] += ev_toks
            else:
                pruned_count += 1

        # Construct Unified System Prompt
        system_sections = [base_system]
        if effective_patch:
            system_sections.append(effective_patch)

        if selected_facts:
            system_sections.append("### [VERIFIED CURRENT BI-TEMPORAL KNOWLEDGE]:\n" + "\n".join(selected_facts))

        if selected_events:
            system_sections.append("### [HISTORICAL EPISODIC CONTEXT]:\n" + "\n".join(selected_events))

        full_system_message = "\n\n".join(system_sections)

        # 4. Working Memory (Last Conversation Turns)
        # Greedily include recent turns starting from the most recent backward
        included_turns: List[TurnRecord] = []
        for turn in reversed(recent_turns):
            turn_toks = estimate_tokens(turn.content) + 4
            if remaining_budget >= turn_toks:
                included_turns.append(turn)
                remaining_budget -= turn_toks
                breakdown["working_memory_tokens"] += turn_toks
            else:
                pruned_count += 1

        included_turns.reverse()  # Restore chronological order

        # Build final messages list for LLM API
        messages: List[Dict[str, str]] = [{"role": "system", "content": full_system_message}]
        for t in included_turns:
            messages.append({"role": t.role, "content": t.content})

        # Append current user prompt
        messages.append({"role": "user", "content": user_message})

        t_end = time.perf_counter()
        assembly_latency_ms = (t_end - t0) * 1000

        total_tokens_used = sum(breakdown.values()) + base_system_tokens + estimate_tokens(user_message)

        telemetry = ContextTelemetry(
            tokens_used=total_tokens_used,
            token_budget=self.token_budget,
            latency_ms=round(assembly_latency_ms, 2),
            breakdown=breakdown,
            pruned_items_count=pruned_count,
            active_facts_count=len(selected_facts),
            active_facts_summary=selected_facts,
            playbook_rules_applied=[r.rule for r in playbook_rec.rules],
        )

        logger.debug(
            "context_compiled_fast_path",
            latency_ms=assembly_latency_ms,
            tokens_used=total_tokens_used,
            budget=self.token_budget
        )
        return messages, telemetry
