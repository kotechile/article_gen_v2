-- Create API Keys table for external service integration
-- This migration creates the api_keys table for storing encrypted external API keys

-- Create api_keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    service_name VARCHAR(50) NOT NULL CHECK (service_name IN ('linkup', 'semrush', 'ahrefs', 'google_trends')),
    api_key TEXT NOT NULL,
    encrypted BOOLEAN DEFAULT true,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_service_name ON api_keys(service_name);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_service ON api_keys(user_id, service_name);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active) WHERE is_active = true;

-- Create partial unique index for active API keys
CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_unique_active 
ON api_keys(user_id, service_name) 
WHERE is_active = true;

-- Enable Row Level Security
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
CREATE POLICY "Users can view their own API keys" ON api_keys
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own API keys" ON api_keys
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own API keys" ON api_keys
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own API keys" ON api_keys
    FOR DELETE USING (auth.uid() = user_id);

-- Add comments
COMMENT ON TABLE api_keys IS 'Stores encrypted external API keys for users';
COMMENT ON COLUMN api_keys.service_name IS 'Name of the external service (linkup, semrush, ahrefs, google_trends)';
COMMENT ON COLUMN api_keys.api_key IS 'Encrypted API key for the external service';
COMMENT ON COLUMN api_keys.encrypted IS 'Whether the API key is encrypted';
COMMENT ON COLUMN api_keys.is_active IS 'Whether the API key is currently active';
