-- Add archive/rating state to subtopics for Topic Detail workflow.
ALTER TABLE subtopics
ADD COLUMN IF NOT EXISTS is_archived BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE subtopics
ADD COLUMN IF NOT EXISTS topic_rating SMALLINT NOT NULL DEFAULT 0;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'subtopics_topic_rating_check'
  ) THEN
    ALTER TABLE subtopics
    ADD CONSTRAINT subtopics_topic_rating_check
    CHECK (topic_rating >= 0 AND topic_rating <= 5);
  END IF;
END$$;

CREATE INDEX IF NOT EXISTS idx_subtopics_is_archived ON subtopics (is_archived);
