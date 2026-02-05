-- TrendTap Database Setup for Supabase SQL Editor
-- Run this script in your Supabase Dashboard → SQL Editor

-- =============================================================================
-- 1. CREATE TABLES
-- =============================================================================

-- Users table (core authentication)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('user', 'admin', 'moderator')),
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    email_verification_token VARCHAR(255),
    email_verification_expires TIMESTAMP WITH TIME ZONE,
    last_login TIMESTAMP WITH TIME ZONE,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_jti VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE NOT NULL,
    device_info JSONB,
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Authentication logs
CREATE TABLE IF NOT EXISTS authentication_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(50) NOT NULL,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    failure_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Password resets
CREATE TABLE IF NOT EXISTS password_resets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    used BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Affiliate research
CREATE TABLE IF NOT EXISTS affiliate_research (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    search_term VARCHAR(255) NOT NULL,
    niche VARCHAR(100),
    budget_range VARCHAR(50),
    results JSONB,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trend analysis
CREATE TABLE IF NOT EXISTS trend_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    keywords TEXT[] NOT NULL,
    analysis_data JSONB,
    time_period VARCHAR(50),
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Keyword data
CREATE TABLE IF NOT EXISTS keyword_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    keyword VARCHAR(255) NOT NULL,
    search_volume INTEGER,
    competition_level VARCHAR(20),
    cpc DECIMAL(10,2),
    trend_data JSONB,
    cluster_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Content ideas
CREATE TABLE IF NOT EXISTS content_ideas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    content_type VARCHAR(50),
    target_keywords TEXT[],
    estimated_traffic INTEGER,
    competition_level VARCHAR(20),
    status VARCHAR(20) DEFAULT 'draft',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Software solutions
CREATE TABLE IF NOT EXISTS software_solutions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    solution_type VARCHAR(50),
    target_audience VARCHAR(100),
    features JSONB,
    status VARCHAR(20) DEFAULT 'draft',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Content calendar
CREATE TABLE IF NOT EXISTS content_calendar (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    content_type VARCHAR(50),
    scheduled_date TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'scheduled',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Export templates
CREATE TABLE IF NOT EXISTS export_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    template_type VARCHAR(50),
    template_data JSONB,
    is_public BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Topic analysis
CREATE TABLE IF NOT EXISTS topic_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic VARCHAR(255) NOT NULL,
    analysis_data JSONB,
    related_topics TEXT[],
    affiliate_programs JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- 2. ENABLE ROW LEVEL SECURITY
-- =============================================================================

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE authentication_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE password_resets ENABLE ROW LEVEL SECURITY;
ALTER TABLE affiliate_research ENABLE ROW LEVEL SECURITY;
ALTER TABLE trend_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE keyword_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_ideas ENABLE ROW LEVEL SECURITY;
ALTER TABLE software_solutions ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_calendar ENABLE ROW LEVEL SECURITY;
ALTER TABLE export_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE topic_analysis ENABLE ROW LEVEL SECURITY;

-- =============================================================================
-- 3. CREATE HELPER FUNCTIONS
-- =============================================================================

CREATE OR REPLACE FUNCTION public.user_id()
RETURNS UUID
LANGUAGE SQL
SECURITY DEFINER
AS $$
  SELECT COALESCE(
    current_setting('request.jwt.claims', true)::json->>'sub',
    (current_setting('request.jwt.claims', true)::json->>'user_id')::text
  )::UUID;
$$;

CREATE OR REPLACE FUNCTION public.is_admin()
RETURNS BOOLEAN
LANGUAGE SQL
SECURITY DEFINER
AS $$
  SELECT EXISTS (
    SELECT 1 FROM users 
    WHERE id = public.user_id() 
    AND role = 'admin' 
    AND is_active = true
  );
$$;

CREATE OR REPLACE FUNCTION public.is_authenticated()
RETURNS BOOLEAN
LANGUAGE SQL
SECURITY DEFINER
AS $$
  SELECT public.user_id() IS NOT NULL;
$$;

-- =============================================================================
-- 4. CREATE RLS POLICIES
-- =============================================================================

-- Users table policies
CREATE POLICY "Users can view own profile" ON users
  FOR SELECT USING (id = public.user_id());

CREATE POLICY "Users can update own profile" ON users
  FOR UPDATE USING (id = public.user_id());

CREATE POLICY "Admins can view all users" ON users
  FOR SELECT USING (public.is_admin());

CREATE POLICY "Admins can update all users" ON users
  FOR UPDATE USING (public.is_admin());

CREATE POLICY "Admins can insert users" ON users
  FOR INSERT WITH CHECK (public.is_admin());

CREATE POLICY "Admins can delete users" ON users
  FOR DELETE USING (public.is_admin());

-- User sessions policies
CREATE POLICY "Users can view own sessions" ON user_sessions
  FOR SELECT USING (user_id = public.user_id());

CREATE POLICY "Users can insert own sessions" ON user_sessions
  FOR INSERT WITH CHECK (user_id = public.user_id());

CREATE POLICY "Users can update own sessions" ON user_sessions
  FOR UPDATE USING (user_id = public.user_id());

CREATE POLICY "Users can delete own sessions" ON user_sessions
  FOR DELETE USING (user_id = public.user_id());

CREATE POLICY "Admins can view all sessions" ON user_sessions
  FOR SELECT USING (public.is_admin());

-- Authentication logs policies
CREATE POLICY "Users can view own auth logs" ON authentication_logs
  FOR SELECT USING (user_id = public.user_id());

CREATE POLICY "System can insert auth logs" ON authentication_logs
  FOR INSERT WITH CHECK (true);

CREATE POLICY "Admins can view all auth logs" ON authentication_logs
  FOR SELECT USING (public.is_admin());

-- Password resets policies
CREATE POLICY "Users can view own password resets" ON password_resets
  FOR SELECT USING (user_id = public.user_id());

CREATE POLICY "Users can insert own password resets" ON password_resets
  FOR INSERT WITH CHECK (user_id = public.user_id());

CREATE POLICY "Users can update own password resets" ON password_resets
  FOR UPDATE USING (user_id = public.user_id());

CREATE POLICY "Users can delete own password resets" ON password_resets
  FOR DELETE USING (user_id = public.user_id());

-- Affiliate research policies
CREATE POLICY "Users can view own affiliate research" ON affiliate_research
  FOR SELECT USING (user_id = public.user_id());

CREATE POLICY "Users can insert own affiliate research" ON affiliate_research
  FOR INSERT WITH CHECK (user_id = public.user_id());

CREATE POLICY "Users can update own affiliate research" ON affiliate_research
  FOR UPDATE USING (user_id = public.user_id());

CREATE POLICY "Users can delete own affiliate research" ON affiliate_research
  FOR DELETE USING (user_id = public.user_id());

CREATE POLICY "Admins can view all affiliate research" ON affiliate_research
  FOR SELECT USING (public.is_admin());

-- Trend analysis policies
CREATE POLICY "Users can view own trend analysis" ON trend_analysis
  FOR SELECT USING (user_id = public.user_id());

CREATE POLICY "Users can insert own trend analysis" ON trend_analysis
  FOR INSERT WITH CHECK (user_id = public.user_id());

CREATE POLICY "Users can update own trend analysis" ON trend_analysis
  FOR UPDATE USING (user_id = public.user_id());

CREATE POLICY "Users can delete own trend analysis" ON trend_analysis
  FOR DELETE USING (user_id = public.user_id());

CREATE POLICY "Admins can view all trend analysis" ON trend_analysis
  FOR SELECT USING (public.is_admin());

-- Keyword data policies
CREATE POLICY "Users can view own keyword data" ON keyword_data
  FOR SELECT USING (user_id = public.user_id());

CREATE POLICY "Users can insert own keyword data" ON keyword_data
  FOR INSERT WITH CHECK (user_id = public.user_id());

CREATE POLICY "Users can update own keyword data" ON keyword_data
  FOR UPDATE USING (user_id = public.user_id());

CREATE POLICY "Users can delete own keyword data" ON keyword_data
  FOR DELETE USING (user_id = public.user_id());

CREATE POLICY "Admins can view all keyword data" ON keyword_data
  FOR SELECT USING (public.is_admin());

-- Content ideas policies
CREATE POLICY "Users can view own content ideas" ON content_ideas
  FOR SELECT USING (user_id = public.user_id());

CREATE POLICY "Users can insert own content ideas" ON content_ideas
  FOR INSERT WITH CHECK (user_id = public.user_id());

CREATE POLICY "Users can update own content ideas" ON content_ideas
  FOR UPDATE USING (user_id = public.user_id());

CREATE POLICY "Users can delete own content ideas" ON content_ideas
  FOR DELETE USING (user_id = public.user_id());

CREATE POLICY "Admins can view all content ideas" ON content_ideas
  FOR SELECT USING (public.is_admin());

-- Software solutions policies
CREATE POLICY "Users can view own software solutions" ON software_solutions
  FOR SELECT USING (user_id = public.user_id());

CREATE POLICY "Users can insert own software solutions" ON software_solutions
  FOR INSERT WITH CHECK (user_id = public.user_id());

CREATE POLICY "Users can update own software solutions" ON software_solutions
  FOR UPDATE USING (user_id = public.user_id());

CREATE POLICY "Users can delete own software solutions" ON software_solutions
  FOR DELETE USING (user_id = public.user_id());

CREATE POLICY "Admins can view all software solutions" ON software_solutions
  FOR SELECT USING (public.is_admin());

-- Content calendar policies
CREATE POLICY "Users can view own calendar" ON content_calendar
  FOR SELECT USING (user_id = public.user_id());

CREATE POLICY "Users can insert own calendar items" ON content_calendar
  FOR INSERT WITH CHECK (user_id = public.user_id());

CREATE POLICY "Users can update own calendar items" ON content_calendar
  FOR UPDATE USING (user_id = public.user_id());

CREATE POLICY "Users can delete own calendar items" ON content_calendar
  FOR DELETE USING (user_id = public.user_id());

CREATE POLICY "Admins can view all calendar items" ON content_calendar
  FOR SELECT USING (public.is_admin());

-- Export templates policies
CREATE POLICY "Users can view own export templates" ON export_templates
  FOR SELECT USING (user_id = public.user_id() OR is_public = true);

CREATE POLICY "Users can insert own export templates" ON export_templates
  FOR INSERT WITH CHECK (user_id = public.user_id());

CREATE POLICY "Users can update own export templates" ON export_templates
  FOR UPDATE USING (user_id = public.user_id());

CREATE POLICY "Users can delete own export templates" ON export_templates
  FOR DELETE USING (user_id = public.user_id());

CREATE POLICY "Admins can view all export templates" ON export_templates
  FOR SELECT USING (public.is_admin());

-- Topic analysis policies
CREATE POLICY "Users can view own topic analysis" ON topic_analysis
  FOR SELECT USING (user_id = public.user_id());

CREATE POLICY "Users can insert own topic analysis" ON topic_analysis
  FOR INSERT WITH CHECK (user_id = public.user_id());

CREATE POLICY "Users can update own topic analysis" ON topic_analysis
  FOR UPDATE USING (user_id = public.user_id());

CREATE POLICY "Users can delete own topic analysis" ON topic_analysis
  FOR DELETE USING (user_id = public.user_id());

CREATE POLICY "Admins can view all topic analysis" ON topic_analysis
  FOR SELECT USING (public.is_admin());

-- =============================================================================
-- 5. CREATE INDEXES
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token_jti ON user_sessions(token_jti);
CREATE INDEX IF NOT EXISTS idx_auth_logs_user_id ON authentication_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_auth_logs_action ON authentication_logs(action);
CREATE INDEX IF NOT EXISTS idx_password_resets_user_id ON password_resets(user_id);
CREATE INDEX IF NOT EXISTS idx_password_resets_token ON password_resets(token);
CREATE INDEX IF NOT EXISTS idx_affiliate_research_user_id ON affiliate_research(user_id);
CREATE INDEX IF NOT EXISTS idx_trend_analysis_user_id ON trend_analysis(user_id);
CREATE INDEX IF NOT EXISTS idx_keyword_data_user_id ON keyword_data(user_id);
CREATE INDEX IF NOT EXISTS idx_content_ideas_user_id ON content_ideas(user_id);
CREATE INDEX IF NOT EXISTS idx_software_solutions_user_id ON software_solutions(user_id);
CREATE INDEX IF NOT EXISTS idx_content_calendar_user_id ON content_calendar(user_id);
CREATE INDEX IF NOT EXISTS idx_export_templates_user_id ON export_templates(user_id);
CREATE INDEX IF NOT EXISTS idx_topic_analysis_user_id ON topic_analysis(user_id);

-- =============================================================================
-- 6. CREATE UPDATED_AT TRIGGERS
-- =============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_affiliate_research_updated_at BEFORE UPDATE ON affiliate_research
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trend_analysis_updated_at BEFORE UPDATE ON trend_analysis
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_keyword_data_updated_at BEFORE UPDATE ON keyword_data
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_content_ideas_updated_at BEFORE UPDATE ON content_ideas
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_software_solutions_updated_at BEFORE UPDATE ON software_solutions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_content_calendar_updated_at BEFORE UPDATE ON content_calendar
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_export_templates_updated_at BEFORE UPDATE ON export_templates
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_topic_analysis_updated_at BEFORE UPDATE ON topic_analysis
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- 7. GRANT PERMISSIONS
-- =============================================================================

GRANT USAGE ON SCHEMA public TO authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO authenticated;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO authenticated;

-- =============================================================================
-- COMPLETION MESSAGE
-- =============================================================================

-- This script creates:
-- ✅ 12 tables with proper user_id foreign keys
-- ✅ Row Level Security on all tables
-- ✅ Helper functions in public schema
-- ✅ Comprehensive RLS policies for user data isolation
-- ✅ Performance indexes
-- ✅ Updated_at triggers
-- ✅ Proper permissions for authenticated users

-- Your TrendTap database is now ready! 🎉


