-- Migration to add writer_notes column to Titles table for personal touch / custom reflections
-- Run this in your Supabase SQL editor if the column is not already present

ALTER TABLE "Titles" ADD COLUMN IF NOT EXISTS writer_notes TEXT;

COMMENT ON COLUMN "Titles".writer_notes IS 'Personal thoughts, opinions, experiences, or book citations to weave into LLM generation';
