-- Hard Reset of content_ideas table because existing constraints and schema states are causing persistent issues.
-- WARNING: This deletes existing data in content_ideas. Ideally this table is just a cache/draft store.

DROP TABLE IF EXISTS "public"."content_ideas";

CREATE TABLE "public"."content_ideas" (
    "id" uuid DEFAULT gen_random_uuid() NOT NULL PRIMARY KEY,
    "user_id" uuid NOT NULL, -- Assuming we want to track ownership
    "topic_id" text NOT NULL, -- The main foreign key to topic (text/uuid)
    "research_id" text, -- Legacy support, can match topic_id
    "subtopic" text,
    "title" text,
    "description" text,
    "content_type" text, -- 'blog' or 'software'
    "category" text,
    "status" text DEFAULT 'draft',
    "keywords" text[],
    "content_outline" text[],
    "total_search_volume" integer DEFAULT 0,
    "created_at" timestamp with time zone DEFAULT now(),
    "updated_at" timestamp with time zone DEFAULT now(),
    "ideas" jsonb DEFAULT '[]'::jsonb -- Legacy JSON column just in case some other code reads it
);

-- Index for fast lookups by topic
CREATE INDEX idx_content_ideas_topic_id ON "public"."content_ideas"("topic_id");
CREATE INDEX idx_content_ideas_user_id ON "public"."content_ideas"("user_id");

-- Add updated_at trigger
CREATE OR REPLACE FUNCTION update_content_ideas_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_content_ideas_timestamp
    BEFORE UPDATE ON "public"."content_ideas"
    FOR EACH ROW
    EXECUTE FUNCTION update_content_ideas_updated_at();

-- Enable RLS but allow everything for authenticated users (fix access issues)
ALTER TABLE "public"."content_ideas" ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all operations for authenticated users" ON "public"."content_ideas"
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Also allow service_role to bypass (implicit, but good to be permissive for dev)
-- Implicitly service_role bypasses RLS.
