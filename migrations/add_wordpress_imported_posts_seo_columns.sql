-- Add SEO metadata and rich article fields to wordpress_imported_posts table.
-- Safe to run multiple times.

ALTER TABLE IF EXISTS "wordpress_imported_posts"
    ADD COLUMN IF NOT EXISTS "slug" TEXT,
    ADD COLUMN IF NOT EXISTS "content_html" TEXT,
    ADD COLUMN IF NOT EXISTS "published_at" TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS "modified_at" TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS "featured_image_url" TEXT,
    ADD COLUMN IF NOT EXISTS "featured_image_alt" TEXT,
    ADD COLUMN IF NOT EXISTS "category_ids" JSONB,
    ADD COLUMN IF NOT EXISTS "category_names" JSONB,
    ADD COLUMN IF NOT EXISTS "tag_ids" JSONB,
    ADD COLUMN IF NOT EXISTS "tag_names" JSONB,
    ADD COLUMN IF NOT EXISTS "seo_title" TEXT,
    ADD COLUMN IF NOT EXISTS "seo_description" TEXT,
    ADD COLUMN IF NOT EXISTS "focus_keyword" TEXT,
    ADD COLUMN IF NOT EXISTS "primary_keyword" TEXT,
    ADD COLUMN IF NOT EXISTS "secondary_keywords" JSONB,
    ADD COLUMN IF NOT EXISTS "canonical_url" TEXT,
    ADD COLUMN IF NOT EXISTS "seo_metadata" JSONB,
    ADD COLUMN IF NOT EXISTS "raw_post_json" JSONB,
    ADD COLUMN IF NOT EXISTS "titles_record_id" TEXT;

COMMENT ON COLUMN "wordpress_imported_posts"."seo_title" IS 'SEO meta title from Yoast/RankMath or custom fields.';
COMMENT ON COLUMN "wordpress_imported_posts"."seo_description" IS 'SEO meta description from Yoast/RankMath or custom fields.';
COMMENT ON COLUMN "wordpress_imported_posts"."focus_keyword" IS 'Focus/primary keyword from Yoast/RankMath or custom fields.';
COMMENT ON COLUMN "wordpress_imported_posts"."seo_metadata" IS 'Comprehensive SEO metadata object including OpenGraph, Twitter, schema, and robots.';
COMMENT ON COLUMN "wordpress_imported_posts"."titles_record_id" IS 'Foreign key to Titles table if imported into Content Studio/Article Editor.';
