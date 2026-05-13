-- Durable storage for released software ideas.
-- These rows intentionally do not depend on topic lifecycle and can survive
-- research-topic cleanup or content-idea deletion.

CREATE TABLE IF NOT EXISTS public.released_software_ideas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    source_idea_id UUID,
    topic_id UUID,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'saved',
    released_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published BOOLEAN NOT NULL DEFAULT TRUE,
    content_type TEXT NOT NULL DEFAULT 'software',
    subtopic TEXT,
    category TEXT,
    domain TEXT,
    keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    primary_keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    secondary_keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    search_phrase TEXT,
    total_search_volume NUMERIC,
    average_difficulty NUMERIC,
    average_cpc NUMERIC,
    affiliate_offer_count INTEGER,
    topic_rating NUMERIC NOT NULL DEFAULT 0,
    viability_score NUMERIC,
    trend_score NUMERIC,
    monetization_score NUMERIC,
    seo_ease_score NUMERIC,
    opportunity_score NUMERIC,
    product_type TEXT,
    user_job_to_be_done TEXT,
    key_inputs JSONB NOT NULL DEFAULT '[]'::jsonb,
    output_result TEXT,
    build_complexity TEXT,
    distribution_angle TEXT,
    target_intent TEXT,
    content_outline JSONB NOT NULL DEFAULT '[]'::jsonb,
    ranking_breakdown JSONB NOT NULL DEFAULT '{}'::jsonb,
    keyword_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    idea_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_dataforseo_output JSONB,
    raw_supabase_output JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT released_software_ideas_status_check
        CHECK (status IN ('saved', 'published', 'archived')),
    CONSTRAINT released_software_ideas_content_type_check
        CHECK (content_type = 'software')
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_released_software_ideas_user_source
    ON public.released_software_ideas (user_id, source_idea_id)
    WHERE source_idea_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_released_software_ideas_user_released_at
    ON public.released_software_ideas (user_id, released_at DESC);

CREATE INDEX IF NOT EXISTS idx_released_software_ideas_topic
    ON public.released_software_ideas (topic_id);

CREATE INDEX IF NOT EXISTS idx_released_software_ideas_product_type
    ON public.released_software_ideas (product_type);

ALTER TABLE public.released_software_ideas ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view their own released software ideas" ON public.released_software_ideas;
DROP POLICY IF EXISTS "Users can insert their own released software ideas" ON public.released_software_ideas;
DROP POLICY IF EXISTS "Users can update their own released software ideas" ON public.released_software_ideas;
DROP POLICY IF EXISTS "Users can delete their own released software ideas" ON public.released_software_ideas;

CREATE POLICY "Users can view their own released software ideas" ON public.released_software_ideas
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own released software ideas" ON public.released_software_ideas
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own released software ideas" ON public.released_software_ideas
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own released software ideas" ON public.released_software_ideas
    FOR DELETE USING (auth.uid() = user_id);

COMMENT ON TABLE public.released_software_ideas IS
    'Durable released software concepts that survive topic and content idea cleanup.';
