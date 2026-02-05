-- Migration: Create API Keys table
-- Description: Creates table for storing API credentials
-- Created: 2024-01-15
-- Author: DataForSEO Integration

-- Create api_keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    base_url VARCHAR(500) NOT NULL,
    key_value TEXT NOT NULL,
    provider VARCHAR(50) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT api_keys_base_url_check CHECK (LENGTH(base_url) > 0),
    CONSTRAINT api_keys_key_value_check CHECK (LENGTH(key_value) > 0),
    CONSTRAINT api_keys_provider_check CHECK (LENGTH(provider) > 0)
);

-- Create index for active API keys
CREATE INDEX IF NOT EXISTS idx_api_keys_provider_active ON api_keys(provider, is_active);

-- Create trigger for updated_at
DROP TRIGGER IF EXISTS update_api_keys_updated_at ON api_keys;
CREATE TRIGGER update_api_keys_updated_at
    BEFORE UPDATE ON api_keys
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add comments
COMMENT ON TABLE api_keys IS 'Stores API credentials for external services';
COMMENT ON COLUMN api_keys.base_url IS 'Base URL for the API service';
COMMENT ON COLUMN api_keys.key_value IS 'API key or authentication token';
COMMENT ON COLUMN api_keys.provider IS 'Service provider name (e.g., dataforseo, openai)';
COMMENT ON COLUMN api_keys.is_active IS 'Whether this API key is currently active';
COMMENT ON COLUMN api_keys.description IS 'Optional description of the API key usage';
