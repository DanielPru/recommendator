-- ============================================================
-- CVIE Database Schema V2 - Supabase PostgreSQL
-- Uses asset_id as primary lifecycle identifier
-- Run this in the Supabase SQL Editor (fresh install only)
-- ============================================================

-- Enable pgcrypto extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- Table 1: decision_logs
-- Logs every structure recommendation decision
-- Uses asset_id (TEXT) as primary key - provided by caller
-- ============================================================
CREATE TABLE decision_logs (
    -- Primary key is asset_id (provided by upstream system)
    asset_id VARCHAR(255) PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Request context
    segment_strategy TEXT NOT NULL,
    channel VARCHAR(100) NOT NULL,
    traffic_type VARCHAR(20) NOT NULL,  -- 'organic' | 'paid'
    funnel_stage VARCHAR(10) NOT NULL,  -- 'TOFU' | 'MOFU' | 'BOFU'
    content_type VARCHAR(10) NOT NULL,  -- 'image' | 'video'
    
    -- Context hash for grouping similar requests
    context_hash VARCHAR(64) NOT NULL,
    
    -- Selected structure
    structure_hash VARCHAR(64) NOT NULL,
    structure_features JSONB NOT NULL,
    
    -- ML scoring results (NULL if no model available)
    p_attention FLOAT,
    p_persuasion FLOAT,
    p_final FLOAT,
    
    -- Decision metadata
    mode VARCHAR(20) NOT NULL DEFAULT 'exploration',  -- 'exploration' | 'exploitation'
    context_confidence FLOAT DEFAULT 0.0,
    exploration_weight FLOAT DEFAULT 0.0,
    model_version VARCHAR(50),
    candidates_count INTEGER NOT NULL,
    
    -- Schema version for future compatibility
    schema_version VARCHAR(10) DEFAULT 'v1' NOT NULL
);

-- Indexes for decision_logs
CREATE INDEX ix_decision_logs_context_hash ON decision_logs(context_hash);
CREATE INDEX ix_decision_logs_structure_hash ON decision_logs(structure_hash);
CREATE INDEX ix_decision_logs_created_at ON decision_logs(created_at);
CREATE INDEX ix_decision_logs_context_structure ON decision_logs(context_hash, structure_hash);

-- ============================================================
-- Table 2: performance_logs
-- Logs performance outcomes for structure decisions
-- Links to decision_logs via asset_id
-- ============================================================
CREATE TABLE performance_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Link to decision via asset_id
    asset_id VARCHAR(255) NOT NULL REFERENCES decision_logs(asset_id),
    
    -- Context identifiers (denormalized for query efficiency)
    context_hash VARCHAR(64) NOT NULL,
    structure_hash VARCHAR(64) NOT NULL,
    traffic_type VARCHAR(20) NOT NULL,
    
    -- Performance metrics (0.0 - 1.0)
    attention_score FLOAT NOT NULL,
    persuasion_score FLOAT NOT NULL,
    
    -- Optional raw metrics for debugging
    raw_metrics JSONB
);

-- Indexes for performance_logs
CREATE INDEX ix_performance_logs_asset_id ON performance_logs(asset_id);
CREATE INDEX ix_performance_logs_context_hash ON performance_logs(context_hash);
CREATE INDEX ix_performance_logs_structure_hash ON performance_logs(structure_hash);
CREATE INDEX ix_performance_logs_context ON performance_logs(context_hash, structure_hash);

-- ============================================================
-- Table 3: context_stats
-- Aggregated statistics per context hash
-- ============================================================
CREATE TABLE context_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    context_hash VARCHAR(64) UNIQUE NOT NULL,
    
    -- Counts
    total_decisions INTEGER DEFAULT 0,
    total_performances INTEGER DEFAULT 0,
    
    -- Aggregated scores
    avg_attention FLOAT,
    avg_persuasion FLOAT,
    std_attention FLOAT,
    std_persuasion FLOAT,
    
    -- Timestamps
    first_seen TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);

-- Index for context_stats
CREATE INDEX ix_context_stats_context_hash ON context_stats(context_hash);

-- ============================================================
-- Table 4: structure_stats
-- Aggregated statistics per structure hash
-- ============================================================
CREATE TABLE structure_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    structure_hash VARCHAR(64) UNIQUE NOT NULL,
    structure_features JSONB NOT NULL,
    
    -- Counts
    total_uses INTEGER DEFAULT 0,
    total_performances INTEGER DEFAULT 0,
    
    -- Aggregated scores
    avg_attention FLOAT,
    avg_persuasion FLOAT,
    
    -- Timestamps
    first_seen TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);

-- Index for structure_stats
CREATE INDEX ix_structure_stats_structure_hash ON structure_stats(structure_hash);

-- ============================================================
-- Table 5: model_registry
-- Registry of trained ML models
-- ============================================================
CREATE TABLE model_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Model identification
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,  -- 'organic_attention', 'organic_persuasion', 'paid_attention', 'paid_persuasion'
    
    -- Model metadata
    model_path VARCHAR(500) NOT NULL,
    training_samples INTEGER NOT NULL,
    feature_schema_version VARCHAR(10) DEFAULT 'v1',
    
    -- Performance metrics
    auc_score FLOAT,
    precision_at_k FLOAT,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE
);

-- Indexes for model_registry
CREATE INDEX ix_model_registry_version ON model_registry(version);
CREATE INDEX ix_model_registry_version_type ON model_registry(version, model_type);
CREATE INDEX ix_model_registry_active ON model_registry(is_active);

-- ============================================================
-- Verification: List all created tables
-- ============================================================
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('decision_logs', 'performance_logs', 'context_stats', 'structure_stats', 'model_registry')
ORDER BY table_name;
