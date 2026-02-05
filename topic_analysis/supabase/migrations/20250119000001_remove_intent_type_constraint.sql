-- Remove intent_type check constraint to allow any value
-- This allows DataForSEO API to return NAVIGATIONAL and other intent types

ALTER TABLE keyword_research_data DROP CONSTRAINT IF EXISTS keyword_research_data_intent_type_check;
