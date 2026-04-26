-- Normalized role mapping for task-based LLM routing.
-- This table is preferred over llm_providers.used_for when present.

CREATE TABLE IF NOT EXISTS llm_used_for (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    llm_provider_id UUID NOT NULL REFERENCES llm_providers(id) ON DELETE CASCADE,
    used_for TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (llm_provider_id, used_for)
);

CREATE INDEX IF NOT EXISTS idx_llm_used_for_used_for
    ON llm_used_for (used_for);

CREATE INDEX IF NOT EXISTS idx_llm_used_for_provider_id
    ON llm_used_for (llm_provider_id);

COMMENT ON TABLE llm_used_for IS
    'Normalized task-role assignments for llm_providers. Preferred over llm_providers.used_for.';

COMMENT ON COLUMN llm_used_for.used_for IS
    'Supported values: deep_research, svg, article_generation, final_review, toc.';

INSERT INTO llm_used_for (llm_provider_id, used_for)
SELECT
    p.id,
    CASE lower(trim(role_name))
        WHEN 'all_other' THEN 'article_generation'
        WHEN 'final review' THEN 'final_review'
        WHEN 'deep research' THEN 'deep_research'
        ELSE lower(trim(role_name))
    END AS normalized_role
FROM llm_providers p
CROSS JOIN LATERAL unnest(
    CASE
        WHEN p.used_for IS NULL THEN ARRAY[]::TEXT[]
        ELSE p.used_for
    END
) AS role_name
ON CONFLICT (llm_provider_id, used_for) DO NOTHING;
