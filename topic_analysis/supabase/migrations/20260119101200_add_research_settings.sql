ALTER TABLE application_settings ADD COLUMN IF NOT EXISTS research_settings JSONB DEFAULT '{}'::jsonb;

COMMENT ON COLUMN application_settings.research_settings IS 'Stores user-defined thresholds for ProfitPath research (min_volume, max_kd, etc.)';
