-- Migration to add 'deck' column to Titles table
-- This allows ensuring the 'deck' (teaser) is persisted as a dedicated column

ALTER TABLE "Titles" 
ADD COLUMN IF NOT EXISTS "deck" text;

COMMENT ON COLUMN "Titles"."deck" IS 'Short teaser content (15-25 words) displayed below the featured image.';
