-- Add performance indexes for dataflow persistence
-- This migration adds indexes to optimize query performance for the complete dataflow

-- Performance indexes for research_topics
CREATE INDEX IF NOT EXISTS idx_research_topics_user_status 
ON research_topics(user_id, status);

CREATE INDEX IF NOT EXISTS idx_research_topics_created_at_desc 
ON research_topics(created_at DESC);

-- Performance indexes for topic_decompositions
CREATE INDEX IF NOT EXISTS idx_topic_decompositions_user_research_topic 
ON topic_decompositions(user_id, research_topic_id);

-- Performance indexes for trend_analyses
CREATE INDEX IF NOT EXISTS idx_trend_analyses_topic_decomposition_status 
ON trend_analyses(topic_decomposition_id, status);

CREATE INDEX IF NOT EXISTS idx_trend_analyses_user_status 
ON trend_analyses(user_id, status);

-- Performance indexes for content_ideas
CREATE INDEX IF NOT EXISTS idx_content_ideas_trend_analysis_status 
ON content_ideas(trend_analysis_id, status);

CREATE INDEX IF NOT EXISTS idx_content_ideas_research_topic_status 
ON content_ideas(research_topic_id, status);

CREATE INDEX IF NOT EXISTS idx_content_ideas_user_status 
ON content_ideas(user_id, status);

-- JSONB GIN indexes for better JSON query performance
CREATE INDEX IF NOT EXISTS idx_topic_decompositions_subtopics_gin 
ON topic_decompositions USING GIN (subtopics);

CREATE INDEX IF NOT EXISTS idx_trend_analyses_keywords_gin 
ON trend_analyses USING GIN (keywords);

CREATE INDEX IF NOT EXISTS idx_trend_analyses_trend_data_gin 
ON trend_analyses USING GIN (trend_data);

CREATE INDEX IF NOT EXISTS idx_content_ideas_key_points_gin 
ON content_ideas USING GIN (key_points);

CREATE INDEX IF NOT EXISTS idx_content_ideas_secondary_keywords_gin 
ON content_ideas USING GIN (secondary_keywords);

CREATE INDEX IF NOT EXISTS idx_content_ideas_tags_gin 
ON content_ideas USING GIN (tags);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_research_topics_user_created_status 
ON research_topics(user_id, created_at DESC, status);

CREATE INDEX IF NOT EXISTS idx_trend_analyses_user_created_status 
ON trend_analyses(user_id, created_at DESC, status);

CREATE INDEX IF NOT EXISTS idx_content_ideas_user_created_status 
ON content_ideas(user_id, created_at DESC, status);
