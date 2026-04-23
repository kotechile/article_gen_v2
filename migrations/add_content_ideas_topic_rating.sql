-- User-settable 0-5 rating for content ideas.
ALTER TABLE content_ideas
ADD COLUMN IF NOT EXISTS topic_rating SMALLINT NOT NULL DEFAULT 0;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'content_ideas_topic_rating_check'
  ) THEN
    ALTER TABLE content_ideas
    ADD CONSTRAINT content_ideas_topic_rating_check
    CHECK (topic_rating >= 0 AND topic_rating <= 5);
  END IF;
END$$;

CREATE INDEX IF NOT EXISTS idx_content_ideas_topic_rating
ON content_ideas (topic_rating);
