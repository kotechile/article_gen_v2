-- Add missing columns to affiliate_programs
ALTER TABLE affiliate_programs 
ADD COLUMN IF NOT EXISTS target_audience JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS company_name TEXT,
ADD COLUMN IF NOT EXISTS program_url TEXT,
ADD COLUMN IF NOT EXISTS contact_email TEXT,
ADD COLUMN IF NOT EXISTS content_opportunities JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'unknown',
ADD COLUMN IF NOT EXISTS data_quality_score FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS research_score FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS verification_status TEXT DEFAULT 'unverified',
ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active';

-- Add index for status
CREATE INDEX IF NOT EXISTS idx_affiliate_programs_status ON affiliate_programs(status);

-- Add index for target_audience for faster JSONB searches
CREATE INDEX IF NOT EXISTS idx_affiliate_programs_target_audience ON affiliate_programs USING GIN(target_audience);
›