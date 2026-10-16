"""
End-to-End Test for MemCore FastAPI Application Gateway
Verifies:
1. /health check
2. /api/v1/demo/seed-scenario (both temporal contradiction & procedural evolution)
3. /api/v1/chat (Fast Path streaming/JSON response with ContextTelemetry)
4. /api/v1/memory/graph/{user_id} and /api/v1/memory/time-travel
5. /api/v1/sleep/trigger
"""

import pytest
import httpx
from datetime import datetime, timezone
from src.api.main import app


@pytest.mark.asyncio
async def test_api_gateway_full_flow():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Health check
        resp = await client.get("/health")
        assert resp.status_code == 200
        health = resp.json()
        assert health["status"] == "healthy"
        assert "architecture" in health

        # 2. Seed scenario
        seed_resp = await client.post(
            "/api/v1/demo/seed-scenario",
            json={
                "scenario": "temporal_contradiction",
                "user_id": "api_test_user",
                "session_id": "api_test_session"
            }
        )
        assert seed_resp.status_code == 200
        seed_data = seed_resp.json()
        assert len(seed_data["steps"]) == 2

        # 3. Check graph edges
        graph_resp = await client.get("/api/v1/memory/graph/api_test_user")
        assert graph_resp.status_code == 200
        graph_data = graph_resp.json()
        assert graph_data["active_edges_count"] >= 1
        assert graph_data["expired_edges_count"] >= 1

        # 4. Fast-Path Chat query
        chat_resp = await client.post(
            "/api/v1/chat",
            json={
                "message": "Write a database migration script for our backend",
                "user_id": "api_test_user",
                "session_id": "api_test_session"
            }
        )
        assert chat_resp.status_code == 200
        chat_data = chat_resp.json()
        assert "response" in chat_data
        assert "PostgreSQL 16" in chat_data["response"]
        assert chat_data["context_telemetry"]["latency_ms"] < 50.0

        # 5. Time-Travel query at January 2026
        tt_resp = await client.post(
            "/api/v1/memory/time-travel",
            json={
                "user_id": "api_test_user",
                "as_of_time": "2026-01-20T00:00:00Z"
            }
        )
        assert tt_resp.status_code == 200
        tt_data = tt_resp.json()
        assert len(tt_data["active_facts"]) >= 1
        active_jan_db = [f for f in tt_data["active_facts"] if f["predicate"] == "USES_DB"]
        assert active_jan_db[0]["object"] == "MySQL"

        # 6. Sleep-Time manual trigger
        sleep_resp = await client.post(
            "/api/v1/sleep/trigger",
            json={
                "user_id": "api_test_user",
                "session_id": "api_test_session",
                "force": True
            }
        )
        assert sleep_resp.status_code == 200
        assert sleep_resp.json()["status"] == "success"
