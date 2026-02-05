-- Make content_ideas table relational/hybrid by adding missing columns expected by the application
-- This allows the insert logic (which pushes title, description, etc.) to succeed.

ALTER TABLE "public"."content_ideas" 
ADD COLUMN IF NOT EXISTS "title" text,
ADD COLUMN IF NOT EXISTS "description" text,
ADD COLUMN IF NOT EXISTS "content_type" text,
ADD COLUMN IF NOT EXISTS "subtopic" text,
ADD COLUMN IF NOT EXISTS "category" text,
ADD COLUMN IF NOT EXISTS "status" text DEFAULT 'draft',
ADD COLUMN IF NOT EXISTS "keywords" text[], -- Array of strings
ADD COLUMN IF NOT EXISTS "content_outline" text[], -- Array of strings
ADD COLUMN IF NOT EXISTS "topic_id" text, -- Add topic_id alias if research_id is strictly enforcing legacy
ADD COLUMN IF NOT EXISTS "user_id" uuid;

-- Drop the JSONB default if we are moving to relational, or keep for legacy compatibility
-- ALTER TABLE "public"."content_ideas" ALTER COLUMN "ideas" DROP NOT NULL;
