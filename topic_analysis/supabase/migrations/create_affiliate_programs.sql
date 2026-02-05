-- Create affiliate_programs table
CREATE TABLE IF NOT EXISTS affiliate_programs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  program_name TEXT NOT NULL,
  description TEXT,
  commission_rate TEXT,
  cookie_duration TEXT,
  website_url TEXT,
  categories TEXT[] DEFAULT '{}',
  network TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  
  CONSTRAINT affiliate_programs_name_unique UNIQUE(program_name)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_affiliate_programs_name ON affiliate_programs(program_name);
CREATE INDEX IF NOT EXISTS idx_affiliate_programs_categories ON affiliate_programs USING GIN(categories);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_affiliate_programs_modtime()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_affiliate_programs_timestamp
    BEFORE UPDATE ON affiliate_programs
    FOR EACH ROW
    EXECUTE PROCEDURE update_affiliate_programs_modtime();

-- RLS (Open for now, or restrictive?)
ALTER TABLE affiliate_programs ENABLE ROW LEVEL SECURITY;

-- Allow authenticated users to read/insert
CREATE POLICY "Authenticated users can read affiliate programs"
    ON affiliate_programs FOR SELECT
    USING (auth.role() = 'authenticated');

CREATE POLICY "Authenticated users can insert affiliate programs"
    ON affiliate_programs FOR INSERT
    WITH CHECK (auth.role() = 'authenticated');
