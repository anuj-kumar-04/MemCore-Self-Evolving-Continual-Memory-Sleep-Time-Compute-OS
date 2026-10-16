"""
Sleep-Time Compute: Bi-Temporal Conflict Resolver (Graph Diff Engine)
Compares new extracted triplets against active Postgres Graph Edges:
• NOOP: Fact already exists -> Boost confidence & last_verified_at
• CONTRADICTION: Set old edge valid_until = NOW(), insert new edge
• NEW FACT: Insert new node/edge with pgvector embedding
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
import structlog

from src.api.schemas import TripletFact, GraphEdgeRecord, ConflictResolutionDiff
from src.memory_tiers.temporal_graph_pg import BiTemporalGraphPG

logger = structlog.get_logger(__name__)

class BiTemporalConflictResolver:
    def __init__(self, temporal_graph: BiTemporalGraphPG):
        self.graph = temporal_graph

    async def resolve_triplets(
        self,
        user_id: str,
        session_id: str,
        triplets: List[TripletFact],
        timestamp_override: Optional[datetime] = None
    ) -> List[ConflictResolutionDiff]:
        """
        Executes atomic bi-temporal diffing against the active knowledge graph.
        Returns detailed diff report for auditability and UI visualizer.
        """
        diffs: List[ConflictResolutionDiff] = []
        now = timestamp_override or datetime.now(timezone.utc)

        for triplet in triplets:
            subj_clean = triplet.subject.strip()
            pred_clean = triplet.predicate.strip().upper()
            obj_clean = triplet.object.strip()

            # Query currently active edge for (user_id, subject, predicate)
            active_edge = await self.graph.get_active_fact_by_key(
                user_id=user_id,
                subject=subj_clean,
                predicate=pred_clean
            )

            if active_edge is None:
                # -------------------------------------------------------------
                # CASE 1: NEW FACT
                # -------------------------------------------------------------
                new_edge = GraphEdgeRecord(
                    user_id=user_id,
                    subject=subj_clean,
                    predicate=pred_clean,
                    object=obj_clean,
                    confidence=triplet.confidence,
                    valid_from=now,
                    valid_until=None,
                    observed_at=now,
                    source_session_id=session_id,
                    metadata=triplet.metadata
                )
                saved_edge = await self.graph.add_edge(new_edge)
                diff = ConflictResolutionDiff(
                    action="NEW_FACT",
                    triplet=triplet,
                    previous_edge=None,
                    resulting_edge=saved_edge,
                    description=f"New fact established: {subj_clean} {pred_clean} {obj_clean}."
                )
                diffs.append(diff)
                logger.info("new_fact_inserted", subject=subj_clean, predicate=pred_clean, object=obj_clean)

            else:
                # Compare active object
                existing_obj_clean = active_edge.object.strip().lower()
                candidate_obj_clean = obj_clean.lower()

                if existing_obj_clean == candidate_obj_clean:
                    # ---------------------------------------------------------
                    # CASE 2: NOOP (Fact confirmed)
                    # ---------------------------------------------------------
                    await self.graph.update_confidence(active_edge.id, boost=0.05)
                    # Refresh object
                    active_edge.confidence = min(1.0, active_edge.confidence + 0.05)
                    diff = ConflictResolutionDiff(
                        action="NOOP",
                        triplet=triplet,
                        previous_edge=active_edge,
                        resulting_edge=active_edge,
                        description=f"Fact already established: {subj_clean} {pred_clean} {obj_clean}. Boosted confidence to {active_edge.confidence:.2f}."
                    )
                    diffs.append(diff)
                    logger.debug("fact_confirmed_noop", subject=subj_clean, predicate=pred_clean)

                else:
                    # ---------------------------------------------------------
                    # CASE 3: CONTRADICTION (Bi-temporal evolution!)
                    # Expire old edge without deleting history; create new edge.
                    # ---------------------------------------------------------
                    new_edge = GraphEdgeRecord(
                        user_id=user_id,
                        subject=subj_clean,
                        predicate=pred_clean,
                        object=obj_clean,
                        confidence=triplet.confidence,
                        valid_from=now,
                        valid_until=None,
                        observed_at=now,
                        source_session_id=session_id,
                        metadata={"supersedes_edge_id": active_edge.id}
                    )
                    saved_new_edge = await self.graph.add_edge(new_edge)
                    
                    # Mark previous edge as superseded and expired
                    await self.graph.expire_edge(
                        edge_id=active_edge.id,
                        superseded_by_id=saved_new_edge.id,
                        expiration_time=now
                    )
                    active_edge.valid_until = now
                    active_edge.superseded_by = saved_new_edge.id

                    diff = ConflictResolutionDiff(
                        action="CONTRADICTION",
                        triplet=triplet,
                        previous_edge=active_edge,
                        resulting_edge=saved_new_edge,
                        description=(
                            f"Contradiction resolved: {subj_clean} {pred_clean} transitioned from "
                            f"'{active_edge.object}' to '{obj_clean}'. Old edge expired at {now.isoformat()} "
                            "and archived for historical audit."
                        )
                    )
                    diffs.append(diff)
                    logger.info(
                        "temporal_contradiction_resolved",
                        subject=subj_clean,
                        predicate=pred_clean,
                        old_val=active_edge.object,
                        new_val=obj_clean,
                        expired_edge_id=active_edge.id,
                        new_edge_id=saved_new_edge.id
                    )

        return diffs
