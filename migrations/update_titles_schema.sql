-- Migration to add missing columns to Titles table
-- This adds support for citations persistence and WordPress settings persistence

-- 1. Citations persistence
ALTER TABLE "Titles" ADD COLUMN IF NOT EXISTS "citations" JSONB;
ALTER TABLE "Titles" ADD COLUMN IF NOT EXISTS "include_in_text_citations" BOOLEAN DEFAULT TRUE;
ALTER TABLE "Titles" ADD COLUMN IF NOT EXISTS "selected_citations" JSONB;

-- 2. WordPress settings persistence
ALTER TABLE "Titles" ADD COLUMN IF NOT EXISTS "last_wp_site_id" TEXT;
ALTER TABLE "Titles" ADD COLUMN IF NOT EXISTS "last_wp_post_status" TEXT;
ALTER TABLE "Titles" ADD COLUMN IF NOT EXISTS "last_wp_category_id" TEXT;

-- Verify columns
-- SELECT column_name, data_type 
-- FROM information_schema.columns 
-- WHERE table_name = 'Titles';
