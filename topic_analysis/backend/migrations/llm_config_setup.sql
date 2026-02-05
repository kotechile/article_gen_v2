-- LLM Configuration Tables Migration
-- Run this in your Supabase SQL editor or via psql

-- Create LLM Providers table
CREATE TABLE IF NOT EXISTS llm_providers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    provider_type VARCHAR(50) NOT NULL CHECK (provider_type IN ('openai', 'anthropic', 'google', 'local', 'custom')),
    model_name VARCHAR(100) NOT NULL,
    
    -- API Configuration
    api_key_env_var VARCHAR(100),
    base_url VARCHAR(500),
    api_version VARCHAR(50),
    
    -- Model Parameters
    max_tokens INTEGER DEFAULT 2000,
    temperature DECIMAL(3,2) DEFAULT 0.7,
    top_p DECIMAL(3,2) DEFAULT 1.0,
    frequency_penalty DECIMAL(3,2) DEFAULT 0.0,
    presence_penalty DECIMAL(3,2) DEFAULT 0.0,
    
    -- Cost and Performance
    cost_per_1k_tokens DECIMAL(10,6) DEFAULT 0.0,
    max_requests_per_minute INTEGER DEFAULT 60,
    average_response_time_ms INTEGER DEFAULT 2000,
    
    -- Status and Configuration
    is_active BOOLEAN DEFAULT true,
    is_default BOOLEAN DEFAULT false,
    priority INTEGER DEFAULT 0,
    
    -- Custom configuration (JSONB for flexibility)
    custom_config JSONB,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used TIMESTAMP WITH TIME ZONE,
    
    -- Usage statistics
    total_requests INTEGER DEFAULT 0,
    successful_requests INTEGER DEFAULT 0,
    failed_requests INTEGER DEFAULT 0,
    total_tokens_used INTEGER DEFAULT 0,
    total_cost DECIMAL(10,2) DEFAULT 0.0
);

-- Create LLM Configuration table
CREATE TABLE IF NOT EXISTS llm_configurations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Global Settings
    default_provider_id UUID REFERENCES llm_providers(id),
    fallback_provider_id UUID REFERENCES llm_providers(id),
    
    -- Feature Flags
    enable_llm_analysis BOOLEAN DEFAULT true,
    enable_auto_fallback BOOLEAN DEFAULT true,
    enable_cost_tracking BOOLEAN DEFAULT true,
    enable_usage_analytics BOOLEAN DEFAULT true,
    
    -- Rate Limiting
    global_rate_limit_per_minute INTEGER DEFAULT 100,
    user_rate_limit_per_minute INTEGER DEFAULT 10,
    
    -- Cost Management
    daily_cost_limit DECIMAL(10,2) DEFAULT 50.0,
    monthly_cost_limit DECIMAL(10,2) DEFAULT 1000.0,
    cost_alert_threshold DECIMAL(3,2) DEFAULT 0.8,
    
    -- Quality Control
    min_confidence_score DECIMAL(3,2) DEFAULT 0.7,
    enable_quality_checks BOOLEAN DEFAULT true,
    auto_retry_failed_requests BOOLEAN DEFAULT true,
    max_retry_attempts INTEGER DEFAULT 3,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_by UUID -- User who updated the config
);

-- Create LLM Usage Logs table
CREATE TABLE IF NOT EXISTS llm_usage_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider_id UUID NOT NULL REFERENCES llm_providers(id),
    user_id UUID, -- Optional user ID
    
    -- Request Details
    topic VARCHAR(255) NOT NULL,
    request_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    response_time_ms INTEGER NOT NULL,
    
    -- Token Usage
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    
    -- Cost and Quality
    cost DECIMAL(10,6) DEFAULT 0.0,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    quality_score DECIMAL(3,2), -- User rating or automated quality score
    
    -- Request Metadata
    request_metadata JSONB,
    response_metadata JSONB
);

-- Create LLM Provider Tests table
CREATE TABLE IF NOT EXISTS llm_provider_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider_id UUID NOT NULL REFERENCES llm_providers(id),
    
    -- Test Details
    test_topic VARCHAR(255) NOT NULL,
    test_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Test Results
    success BOOLEAN DEFAULT false,
    response_time_ms INTEGER,
    quality_score DECIMAL(3,2),
    error_message TEXT,
    
    -- Generated Content
    generated_related_areas JSONB,
    generated_affiliate_programs JSONB,
    
    -- Test Metadata
    test_parameters JSONB,
    test_notes TEXT
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_llm_providers_active ON llm_providers(is_active);
CREATE INDEX IF NOT EXISTS idx_llm_providers_default ON llm_providers(is_default);
CREATE INDEX IF NOT EXISTS idx_llm_providers_priority ON llm_providers(priority DESC);
CREATE INDEX IF NOT EXISTS idx_llm_usage_logs_provider ON llm_usage_logs(provider_id);
CREATE INDEX IF NOT EXISTS idx_llm_usage_logs_timestamp ON llm_usage_logs(request_timestamp);
CREATE INDEX IF NOT EXISTS idx_llm_usage_logs_user ON llm_usage_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_llm_provider_tests_provider ON llm_provider_tests(provider_id);

-- Insert default LLM providers
INSERT INTO llm_providers (
    name, provider_type, model_name, api_key_env_var, 
    max_tokens, temperature, cost_per_1k_tokens, 
    is_default, priority, is_active
) VALUES 
-- OpenAI Models
(
    'OpenAI GPT-5 Mini', 'openai', 'gpt-5-mini', 'OPENAI_API_KEY',
    4000, 0.7, 0.0015, true, 120, true
),
(
    'OpenAI GPT-4 Turbo', 'openai', 'gpt-4-turbo', 'OPENAI_API_KEY',
    4000, 0.7, 0.01, false, 110, true
),
(
    'OpenAI GPT-4', 'openai', 'gpt-4', 'OPENAI_API_KEY',
    2000, 0.7, 0.03, false, 100, true
),
(
    'OpenAI GPT-3.5 Turbo', 'openai', 'gpt-3.5-turbo', 'OPENAI_API_KEY',
    2000, 0.7, 0.002, false, 90, true
),
-- Anthropic Models
(
    'Anthropic Claude 3.5 Sonnet', 'anthropic', 'claude-3-5-sonnet-20241022', 'ANTHROPIC_API_KEY',
    4000, 0.7, 0.003, false, 95, true
),
(
    'Anthropic Claude 3 Sonnet', 'anthropic', 'claude-3-sonnet-20240229', 'ANTHROPIC_API_KEY',
    2000, 0.7, 0.015, false, 80, true
),
(
    'Anthropic Claude 3 Haiku', 'anthropic', 'claude-3-haiku-20240307', 'ANTHROPIC_API_KEY',
    2000, 0.7, 0.00025, false, 75, true
),
-- Google Models
(
    'Google Gemini 2.5 Flash', 'google', 'gemini-2.5-flash', 'GOOGLE_API_KEY',
    4000, 0.7, 0.0005, false, 85, true
),
(
    'Google Gemini 2.5 Flash Lite', 'google', 'gemini-2.5-flash-lite', 'GOOGLE_API_KEY',
    4000, 0.7, 0.0001, false, 70, true
),
(
    'Google Gemini Pro', 'google', 'gemini-pro', 'GOOGLE_API_KEY',
    2000, 0.7, 0.001, false, 65, true
),
-- Local Models
(
    'Local Ollama Llama3.1', 'local', 'llama3.1', NULL,
    2000, 0.7, 0.0, false, 60, false
),
(
    'Local Ollama Mistral', 'local', 'mistral', NULL,
    2000, 0.7, 0.0, false, 55, false
),
(
    'Local Ollama CodeLlama', 'local', 'codellama', NULL,
    2000, 0.7, 0.0, false, 50, false
) ON CONFLICT (name) DO NOTHING;

-- Insert default configuration
INSERT INTO llm_configurations (
    enable_llm_analysis, enable_auto_fallback, enable_cost_tracking,
    global_rate_limit_per_minute, user_rate_limit_per_minute,
    daily_cost_limit, monthly_cost_limit, cost_alert_threshold
) VALUES (
    true, true, true,
    100, 10,
    50.0, 1000.0, 0.8
) ON CONFLICT DO NOTHING;

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_llm_providers_updated_at 
    BEFORE UPDATE ON llm_providers 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_llm_configurations_updated_at 
    BEFORE UPDATE ON llm_configurations 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create RLS policies (if using Supabase RLS)
-- Note: Adjust these policies based on your authentication setup

-- Enable RLS on all tables
ALTER TABLE llm_providers ENABLE ROW LEVEL SECURITY;
ALTER TABLE llm_configurations ENABLE ROW LEVEL SECURITY;
ALTER TABLE llm_usage_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE llm_provider_tests ENABLE ROW LEVEL SECURITY;

-- Create policies for admin access
-- Note: Replace 'auth.uid()' with your actual user ID column if different
CREATE POLICY "Admin can manage LLM providers" ON llm_providers
    FOR ALL USING (auth.uid() IN (
        SELECT id FROM auth.users WHERE raw_user_meta_data->>'role' = 'admin'
    ));

CREATE POLICY "Admin can manage LLM config" ON llm_configurations
    FOR ALL USING (auth.uid() IN (
        SELECT id FROM auth.users WHERE raw_user_meta_data->>'role' = 'admin'
    ));

CREATE POLICY "Admin can view usage logs" ON llm_usage_logs
    FOR SELECT USING (auth.uid() IN (
        SELECT id FROM auth.users WHERE raw_user_meta_data->>'role' = 'admin'
    ));

CREATE POLICY "Admin can manage provider tests" ON llm_provider_tests
    FOR ALL USING (auth.uid() IN (
        SELECT id FROM auth.users WHERE raw_user_meta_data->>'role' = 'admin'
    ));

-- Grant necessary permissions
GRANT ALL ON llm_providers TO authenticated;
GRANT ALL ON llm_configurations TO authenticated;
GRANT ALL ON llm_usage_logs TO authenticated;
GRANT ALL ON llm_provider_tests TO authenticated;

-- Create a view for easy provider selection
CREATE OR REPLACE VIEW active_llm_providers AS
SELECT 
    id,
    name,
    provider_type,
    model_name,
    is_default,
    priority,
    total_requests,
    successful_requests,
    failed_requests,
    total_cost,
    last_used
FROM llm_providers 
WHERE is_active = true
ORDER BY priority DESC, total_requests DESC;

-- Create a view for usage analytics
CREATE OR REPLACE VIEW llm_usage_analytics AS
SELECT 
    p.name as provider_name,
    p.provider_type,
    COUNT(l.id) as total_requests,
    COUNT(CASE WHEN l.success THEN 1 END) as successful_requests,
    COUNT(CASE WHEN NOT l.success THEN 1 END) as failed_requests,
    ROUND(
        COUNT(CASE WHEN l.success THEN 1 END)::DECIMAL / NULLIF(COUNT(l.id), 0) * 100, 2
    ) as success_rate,
    SUM(l.total_tokens) as total_tokens,
    SUM(l.cost) as total_cost,
    ROUND(AVG(l.response_time_ms), 2) as avg_response_time_ms,
    MAX(l.request_timestamp) as last_used
FROM llm_providers p
LEFT JOIN llm_usage_logs l ON p.id = l.provider_id
WHERE p.is_active = true
GROUP BY p.id, p.name, p.provider_type
ORDER BY total_requests DESC;
