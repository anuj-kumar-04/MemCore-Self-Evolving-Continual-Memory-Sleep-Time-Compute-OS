-- ==============================================================================
-- MemCore — Bi-Temporal Knowledge Graph & Cognitive Memory Schema
-- PostgreSQL 16 + pgvector (HNSW)
-- ==============================================================================

-- Enable UUID and pgvector extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- ------------------------------------------------------------------------------
-- 1. Entities Registry (Core Nodes)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(128) NOT NULL,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) DEFAULT 'general',
    properties JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_entity_user_name_cat UNIQUE (user_id, name, category)
);

CREATE INDEX IF NOT EXISTS idx_entities_user_cat ON entities (user_id, category);

-- ------------------------------------------------------------------------------
-- 2. Bi-Temporal Knowledge Graph Edges (Tier 2: Semantic Memory)
-- Supports valid_from, valid_until (reality time) and observed_at (audit time)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS temporal_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(128) NOT NULL,
    subject VARCHAR(255) NOT NULL,
    predicate VARCHAR(255) NOT NULL,
    object VARCHAR(255) NOT NULL,
    confidence FLOAT NOT NULL DEFAULT 1.0,
    valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMPTZ DEFAULT NULL,                 -- NULL = Active current truth!
    observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),        -- Transaction / observation time
    superseded_by UUID REFERENCES temporal_edges(id) ON DELETE SET NULL,
    source_session_id VARCHAR(128),
    source_turn_ids JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(384)
);

-- Fast lookup for query routing and active truth
CREATE INDEX IF NOT EXISTS idx_edges_lookup ON temporal_edges (user_id, subject, predicate);
CREATE INDEX IF NOT EXISTS idx_edges_active ON temporal_edges (user_id, subject, predicate) WHERE valid_until IS NULL;
CREATE INDEX IF NOT EXISTS idx_edges_temporal_window ON temporal_edges (user_id, valid_from, valid_until);
CREATE INDEX IF NOT EXISTS idx_edges_observed ON temporal_edges (user_id, observed_at DESC);

-- HNSW Cosine Distance Vector Index for Semantic Graph Retrieval
CREATE INDEX IF NOT EXISTS idx_edges_hnsw ON temporal_edges 
    USING hnsw (embedding vector_cosine_ops) 
    WITH (m = 16, ef_construction = 64);

-- ------------------------------------------------------------------------------
-- 3. Episodic Events Store (Tier 2: Episodic Milestones & Narratives)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS episodic_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(128) NOT NULL,
    session_id VARCHAR(128) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    summary TEXT NOT NULL,
    raw_context TEXT,
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    turn_range INT[],
    embedding vector(384)
);

CREATE INDEX IF NOT EXISTS idx_episodic_user_time ON episodic_events (user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_hnsw ON episodic_events 
    USING hnsw (embedding vector_cosine_ops) 
    WITH (m = 16, ef_construction = 64);

-- ------------------------------------------------------------------------------
-- 4. Procedural Playbooks (Tier 3: Self-Evolving User Behavioral Rulebook)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS procedural_playbooks (
    user_id VARCHAR(128) PRIMARY KEY,
    rules JSONB NOT NULL DEFAULT '[]'::jsonb,
    system_patch TEXT NOT NULL DEFAULT '',
    version INT NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------------------------
-- 5. Sleep-Time Compute Consolidation Logs (Audit & Observability)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sleep_consolidation_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(128) NOT NULL,
    session_id VARCHAR(128) NOT NULL,
    turns_processed INT NOT NULL DEFAULT 0,
    facts_extracted INT NOT NULL DEFAULT 0,
    contradictions_resolved INT NOT NULL DEFAULT 0,
    noops INT NOT NULL DEFAULT 0,
    rules_distilled INT NOT NULL DEFAULT 0,
    duration_ms FLOAT NOT NULL DEFAULT 0.0,
    diff_summary JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sleep_logs_user ON sleep_consolidation_logs (user_id, created_at DESC);
