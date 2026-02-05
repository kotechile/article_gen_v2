-- Create basic tables for TrendTap
-- This migration creates the essential tables needed for the application

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('user', 'admin', 'moderator')),
    subscription_tier VARCHAR(20) DEFAULT 'free' CHECK (subscription_tier IN ('free', 'basic', 'premium', 'enterprise')),
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    preferences TEXT,
    timezone VARCHAR(50) DEFAULT 'UTC',
    language VARCHAR(10) DEFAULT 'en',
    subscription_start_date TIMESTAMP WITH TIME ZONE,
    subscription_end_date TIMESTAMP WITH TIME ZONE,
    subscription_status VARCHAR(50) DEFAULT 'active',
    api_calls_count INTEGER DEFAULT 0,
    last_api_call TIMESTAMP WITH TIME ZONE
);

-- Create workflow_sessions table
CREATE TABLE IF NOT EXISTS workflow_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'completed', 'failed', 'paused')),
    progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage >= 0 AND progress_percentage <= 100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create topic_decompositions table
CREATE TABLE IF NOT EXISTS topic_decompositions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    search_query TEXT NOT NULL,
    subtopics JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create affiliate_offers table
CREATE TABLE IF NOT EXISTS affiliate_offers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workflow_session_id UUID NOT NULL REFERENCES workflow_sessions(id) ON DELETE CASCADE,
    offer_name VARCHAR(255) NOT NULL,
    offer_description TEXT,
    commission_rate DECIMAL(5,2),
    access_instructions TEXT,
    subtopic_id UUID REFERENCES topic_decompositions(id) ON DELETE SET NULL,
    linkup_data JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'expired')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

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
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_subscription_tier ON users(subscription_tier);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);

CREATE INDEX IF NOT EXISTS idx_workflow_sessions_user_id ON workflow_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_workflow_sessions_status ON workflow_sessions(status);

CREATE INDEX IF NOT EXISTS idx_topic_decompositions_user_id ON topic_decompositions(user_id);

CREATE INDEX IF NOT EXISTS idx_affiliate_offers_user_id ON affiliate_offers(user_id);
CREATE INDEX IF NOT EXISTS idx_affiliate_offers_workflow_session_id ON affiliate_offers(workflow_session_id);
CREATE INDEX IF NOT EXISTS idx_affiliate_offers_status ON affiliate_offers(status);

CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_service_name ON api_keys(service_name);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_service ON api_keys(user_id, service_name);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active) WHERE is_active = true;

-- Create partial unique index for active API keys
CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_unique_active 
ON api_keys(user_id, service_name) 
WHERE is_active = true;

-- Enable Row Level Security
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE topic_decompositions ENABLE ROW LEVEL SECURITY;
ALTER TABLE affiliate_offers ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for users
CREATE POLICY "Users can view their own data" ON users
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update their own data" ON users
    FOR UPDATE USING (auth.uid() = id);

-- Create RLS policies for workflow_sessions
CREATE POLICY "Users can view their own workflow sessions" ON workflow_sessions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own workflow sessions" ON workflow_sessions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own workflow sessions" ON workflow_sessions
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own workflow sessions" ON workflow_sessions
    FOR DELETE USING (auth.uid() = user_id);

-- Create RLS policies for topic_decompositions
CREATE POLICY "Users can view their own topic decompositions" ON topic_decompositions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own topic decompositions" ON topic_decompositions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own topic decompositions" ON topic_decompositions
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own topic decompositions" ON topic_decompositions
    FOR DELETE USING (auth.uid() = user_id);

-- Create RLS policies for affiliate_offers
CREATE POLICY "Users can view their own affiliate offers" ON affiliate_offers
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own affiliate offers" ON affiliate_offers
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own affiliate offers" ON affiliate_offers
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own affiliate offers" ON affiliate_offers
    FOR DELETE USING (auth.uid() = user_id);

-- Create RLS policies for api_keys
CREATE POLICY "Users can view their own API keys" ON api_keys
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own API keys" ON api_keys
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own API keys" ON api_keys
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own API keys" ON api_keys
    FOR DELETE USING (auth.uid() = user_id);
