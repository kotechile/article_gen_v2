-- User-settable 0-5 star rating for research topics.
ALTER TABLE research_topics
ADD COLUMN IF NOT EXISTS topic_rating SMALLINT NOT NULL DEFAULT 0;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'research_topics_topic_rating_check'
  ) THEN
    ALTER TABLE research_topics
    ADD CONSTRAINT research_topics_topic_rating_check
    CHECK (topic_rating >= 0 AND topic_rating <= 5);
  END IF;
END$$;
