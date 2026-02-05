-- Migration: Create Images table for storing image metadata
-- Created: 2026-01-06
-- Description: Stores metadata for images inserted into articles, including AI-generated, stock, uploaded, and infographic images

-- Images table for storing image metadata
CREATE TABLE IF NOT EXISTS Images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id) NOT NULL,
    ImageUrl TEXT NOT NULL,
    ImageAuthor TEXT,
    MediaAltText TEXT,
    mediaTitle TEXT,
    mediaCaption TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on user_id for faster queries
CREATE INDEX IF NOT EXISTS idx_images_user_id ON Images(user_id);

-- Enable Row Level Security
ALTER TABLE Images ENABLE ROW LEVEL SECURITY;

-- RLS policies for Images table
-- Users can only view their own images
CREATE POLICY "Users can view their own images" 
    ON Images FOR SELECT 
    USING (auth.uid() = user_id);

-- Users can only insert their own images
CREATE POLICY "Users can insert their own images" 
    ON Images FOR INSERT 
    WITH CHECK (auth.uid() = user_id);

-- Users can only update their own images
CREATE POLICY "Users can update their own images" 
    ON Images FOR UPDATE 
    USING (auth.uid() = user_id);

-- Users can only delete their own images
CREATE POLICY "Users can delete their own images" 
    ON Images FOR DELETE 
    USING (auth.uid() = user_id);
