-- Supabase Row Level Security (RLS) Setup for TrendTap
-- This script sets up comprehensive RLS policies for all tables

-- Enable RLS on all tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE affiliate_research ENABLE ROW LEVEL SECURITY;
ALTER TABLE trend_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE keyword_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_ideas ENABLE ROW LEVEL SECURITY;
ALTER TABLE software_solutions ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_calendar ENABLE ROW LEVEL SECURITY;
ALTER TABLE export_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE llm_providers ENABLE ROW LEVEL SECURITY;
ALTER TABLE llm_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE topic_analysis ENABLE ROW LEVEL SECURITY;

-- Create user roles enum
CREATE TYPE user_role AS ENUM ('user', 'admin', 'moderator');

-- Add role column to users table if it doesn't exist
ALTER TABLE users ADD COLUMN IF NOT EXISTS role user_role DEFAULT 'user';
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT true;
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login TIMESTAMP WITH TIME ZONE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
ALTER TABLE users ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

-- Create function to get current user ID
CREATE OR REPLACE FUNCTION auth.user_id()
RETURNS UUID
LANGUAGE SQL
SECURITY DEFINER
AS $$
  SELECT COALESCE(
    current_setting('request.jwt.claims', true)::json->>'sub',
    (current_setting('request.jwt.claims', true)::json->>'user_id')::text
  )::UUID;
$$;

-- Create function to check if user is admin
CREATE OR REPLACE FUNCTION auth.is_admin()
RETURNS BOOLEAN
LANGUAGE SQL
SECURITY DEFINER
AS $$
  SELECT EXISTS (
    SELECT 1 FROM users 
    WHERE id = auth.user_id() 
    AND role = 'admin' 
    AND is_active = true
  );
$$;

-- Create function to check if user is moderator or admin
CREATE OR REPLACE FUNCTION auth.is_moderator()
RETURNS BOOLEAN
LANGUAGE SQL
SECURITY DEFINER
AS $$
  SELECT EXISTS (
    SELECT 1 FROM users 
    WHERE id = auth.user_id() 
    AND role IN ('admin', 'moderator') 
    AND is_active = true
  );
$$;

-- Users table policies
-- Users can view their own profile
CREATE POLICY "Users can view own profile" ON users
  FOR SELECT USING (id = auth.user_id());

-- Users can update their own profile
CREATE POLICY "Users can update own profile" ON users
  FOR UPDATE USING (id = auth.user_id());

-- Admins can view all users
CREATE POLICY "Admins can view all users" ON users
  FOR SELECT USING (auth.is_admin());

-- Admins can update all users
CREATE POLICY "Admins can update all users" ON users
  FOR UPDATE USING (auth.is_admin());

-- Admins can insert new users
CREATE POLICY "Admins can insert users" ON users
  FOR INSERT WITH CHECK (auth.is_admin());

-- Admins can delete users
CREATE POLICY "Admins can delete users" ON users
  FOR DELETE USING (auth.is_admin());

-- Affiliate Research policies
-- Users can view their own research
CREATE POLICY "Users can view own affiliate research" ON affiliate_research
  FOR SELECT USING (user_id = auth.user_id());

-- Users can insert their own research
CREATE POLICY "Users can insert own affiliate research" ON affiliate_research
  FOR INSERT WITH CHECK (user_id = auth.user_id());

-- Users can update their own research
CREATE POLICY "Users can update own affiliate research" ON affiliate_research
  FOR UPDATE USING (user_id = auth.user_id());

-- Users can delete their own research
CREATE POLICY "Users can delete own affiliate research" ON affiliate_research
  FOR DELETE USING (user_id = auth.user_id());

-- Admins can view all research
CREATE POLICY "Admins can view all affiliate research" ON affiliate_research
  FOR SELECT USING (auth.is_admin());

-- Trend Analysis policies
CREATE POLICY "Users can view own trend analysis" ON trend_analysis
  FOR SELECT USING (user_id = auth.user_id());

CREATE POLICY "Users can insert own trend analysis" ON trend_analysis
  FOR INSERT WITH CHECK (user_id = auth.user_id());

CREATE POLICY "Users can update own trend analysis" ON trend_analysis
  FOR UPDATE USING (user_id = auth.user_id());

CREATE POLICY "Users can delete own trend analysis" ON trend_analysis
  FOR DELETE USING (user_id = auth.user_id());

CREATE POLICY "Admins can view all trend analysis" ON trend_analysis
  FOR SELECT USING (auth.is_admin());

-- Keyword Data policies
CREATE POLICY "Users can view own keyword data" ON keyword_data
  FOR SELECT USING (user_id = auth.user_id());

CREATE POLICY "Users can insert own keyword data" ON keyword_data
  FOR INSERT WITH CHECK (user_id = auth.user_id());

CREATE POLICY "Users can update own keyword data" ON keyword_data
  FOR UPDATE USING (user_id = auth.user_id());

CREATE POLICY "Users can delete own keyword data" ON keyword_data
  FOR DELETE USING (user_id = auth.user_id());

CREATE POLICY "Admins can view all keyword data" ON keyword_data
  FOR SELECT USING (auth.is_admin());

-- Content Ideas policies
CREATE POLICY "Users can view own content ideas" ON content_ideas
  FOR SELECT USING (user_id = auth.user_id());

CREATE POLICY "Users can insert own content ideas" ON content_ideas
  FOR INSERT WITH CHECK (user_id = auth.user_id());

CREATE POLICY "Users can update own content ideas" ON content_ideas
  FOR UPDATE USING (user_id = auth.user_id());

CREATE POLICY "Users can delete own content ideas" ON content_ideas
  FOR DELETE USING (user_id = auth.user_id());

CREATE POLICY "Admins can view all content ideas" ON content_ideas
  FOR SELECT USING (auth.is_admin());

-- Software Solutions policies
CREATE POLICY "Users can view own software solutions" ON software_solutions
  FOR SELECT USING (user_id = auth.user_id());

CREATE POLICY "Users can insert own software solutions" ON software_solutions
  FOR INSERT WITH CHECK (user_id = auth.user_id());

CREATE POLICY "Users can update own software solutions" ON software_solutions
  FOR UPDATE USING (user_id = auth.user_id());

CREATE POLICY "Users can delete own software solutions" ON software_solutions
  FOR DELETE USING (user_id = auth.user_id());

CREATE POLICY "Admins can view all software solutions" ON software_solutions
  FOR SELECT USING (auth.is_admin());

-- Content Calendar policies
CREATE POLICY "Users can view own calendar" ON content_calendar
  FOR SELECT USING (user_id = auth.user_id());

CREATE POLICY "Users can insert own calendar items" ON content_calendar
  FOR INSERT WITH CHECK (user_id = auth.user_id());

CREATE POLICY "Users can update own calendar items" ON content_calendar
  FOR UPDATE USING (user_id = auth.user_id());

CREATE POLICY "Users can delete own calendar items" ON content_calendar
  FOR DELETE USING (user_id = auth.user_id());

CREATE POLICY "Admins can view all calendar items" ON content_calendar
  FOR SELECT USING (auth.is_admin());

-- Export Templates policies
CREATE POLICY "Users can view own export templates" ON export_templates
  FOR SELECT USING (user_id = auth.user_id());

CREATE POLICY "Users can insert own export templates" ON export_templates
  FOR INSERT WITH CHECK (user_id = auth.user_id());

CREATE POLICY "Users can update own export templates" ON export_templates
  FOR UPDATE USING (user_id = auth.user_id());

CREATE POLICY "Users can delete own export templates" ON export_templates
  FOR DELETE USING (user_id = auth.user_id());

CREATE POLICY "Admins can view all export templates" ON export_templates
  FOR SELECT USING (auth.is_admin());

-- LLM Providers policies (admin only)
CREATE POLICY "Admins can view LLM providers" ON llm_providers
  FOR SELECT USING (auth.is_admin());

CREATE POLICY "Admins can insert LLM providers" ON llm_providers
  FOR INSERT WITH CHECK (auth.is_admin());

CREATE POLICY "Admins can update LLM providers" ON llm_providers
  FOR UPDATE USING (auth.is_admin());

CREATE POLICY "Admins can delete LLM providers" ON llm_providers
  FOR DELETE USING (auth.is_admin());

-- LLM Configs policies (admin only)
CREATE POLICY "Admins can view LLM configs" ON llm_configs
  FOR SELECT USING (auth.is_admin());

CREATE POLICY "Admins can insert LLM configs" ON llm_configs
  FOR INSERT WITH CHECK (auth.is_admin());

CREATE POLICY "Admins can update LLM configs" ON llm_configs
  FOR UPDATE USING (auth.is_admin());

CREATE POLICY "Admins can delete LLM configs" ON llm_configs
  FOR DELETE USING (auth.is_admin());

-- Topic Analysis policies
CREATE POLICY "Users can view own topic analysis" ON topic_analysis
  FOR SELECT USING (user_id = auth.user_id());

CREATE POLICY "Users can insert own topic analysis" ON topic_analysis
  FOR INSERT WITH CHECK (user_id = auth.user_id());

CREATE POLICY "Users can update own topic analysis" ON topic_analysis
  FOR UPDATE USING (user_id = auth.user_id());

CREATE POLICY "Users can delete own topic analysis" ON topic_analysis
  FOR DELETE USING (user_id = auth.user_id());

CREATE POLICY "Admins can view all topic analysis" ON topic_analysis
  FOR SELECT USING (auth.is_admin());

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);
CREATE INDEX IF NOT EXISTS idx_affiliate_research_user_id ON affiliate_research(user_id);
CREATE INDEX IF NOT EXISTS idx_trend_analysis_user_id ON trend_analysis(user_id);
CREATE INDEX IF NOT EXISTS idx_keyword_data_user_id ON keyword_data(user_id);
CREATE INDEX IF NOT EXISTS idx_content_ideas_user_id ON content_ideas(user_id);
CREATE INDEX IF NOT EXISTS idx_software_solutions_user_id ON software_solutions(user_id);
CREATE INDEX IF NOT EXISTS idx_content_calendar_user_id ON content_calendar(user_id);
CREATE INDEX IF NOT EXISTS idx_export_templates_user_id ON export_templates(user_id);
CREATE INDEX IF NOT EXISTS idx_topic_analysis_user_id ON topic_analysis(user_id);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at triggers to all tables
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

-- Insert default admin user (you'll need to set the password)
INSERT INTO users (id, email, username, role, is_active, created_at) 
VALUES (
  gen_random_uuid(),
  'admin@trendtap.com',
  'admin',
  'admin',
  true,
  NOW()
) ON CONFLICT (email) DO NOTHING;

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO authenticated;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO authenticated;


