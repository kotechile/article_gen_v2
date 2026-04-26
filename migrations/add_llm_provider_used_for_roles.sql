-- Add task-based routing metadata for LLM selection.
-- Safe to run on existing environments.

ALTER TABLE llm_providers
    ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;

ALTER TABLE llm_providers
    ADD COLUMN IF NOT EXISTS used_for TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[];

COMMENT ON COLUMN llm_providers.used_for IS
    'Task roles for backend model routing. Supported values: deep_research, svg, article_generation, final_review, toc.';

CREATE INDEX IF NOT EXISTS idx_llm_providers_is_active
    ON llm_providers (is_active);

-- Normalize any legacy comma-separated text values if the column was created manually
-- before this migration as TEXT rather than TEXT[].
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'llm_providers'
          AND column_name = 'used_for'
          AND data_type = 'text'
    ) THEN
        ALTER TABLE llm_providers
            ALTER COLUMN used_for DROP DEFAULT;

        ALTER TABLE llm_providers
            ALTER COLUMN used_for TYPE TEXT[]
            USING CASE
                WHEN used_for IS NULL OR btrim(used_for) = '' THEN ARRAY[]::TEXT[]
                ELSE regexp_split_to_array(used_for, '\s*,\s*')
            END;

        ALTER TABLE llm_providers
            ALTER COLUMN used_for SET DEFAULT ARRAY[]::TEXT[];
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_llm_providers_used_for
    ON llm_providers
    USING GIN (used_for);

-- Alias migration for any early role labels that made it into the database.
UPDATE llm_providers
SET used_for = ARRAY(
    SELECT DISTINCT
        CASE lower(trim(role_name))
            WHEN 'all_other' THEN 'article_generation'
            WHEN 'final review' THEN 'final_review'
            WHEN 'deep research' THEN 'deep_research'
            ELSE lower(trim(role_name))
        END
    FROM unnest(used_for) AS role_name
)
WHERE used_for IS NOT NULL;
