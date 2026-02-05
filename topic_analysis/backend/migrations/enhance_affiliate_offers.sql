-- Enhanced Affiliate Offers System
-- This migration enhances the existing affiliate offers system with better research capabilities

-- 1. Enhanced affiliate_offers table with more research data
ALTER TABLE affiliate_offers ADD COLUMN IF NOT EXISTS research_score DECIMAL(3,2) DEFAULT 0.0;
ALTER TABLE affiliate_offers ADD COLUMN IF NOT EXISTS relevance_score DECIMAL(3,2) DEFAULT 0.0;
ALTER TABLE affiliate_offers ADD COLUMN IF NOT EXISTS subtopics TEXT[] DEFAULT '{}';
ALTER TABLE affiliate_offers ADD COLUMN IF NOT EXISTS commission_range VARCHAR(20);
ALTER TABLE affiliate_offers ADD COLUMN IF NOT EXISTS network_name VARCHAR(100);
ALTER TABLE affiliate_offers ADD COLUMN IF NOT EXISTS program_url TEXT;
ALTER TABLE affiliate_offers ADD COLUMN IF NOT EXISTS application_requirements TEXT;
ALTER TABLE affiliate_offers ADD COLUMN IF NOT EXISTS payment_terms TEXT;
ALTER TABLE affiliate_offers ADD COLUMN IF NOT EXISTS cookie_duration INTEGER; -- in days
ALTER TABLE affiliate_offers ADD COLUMN IF NOT EXISTS last_verified TIMESTAMP WITH TIME ZONE;
ALTER TABLE affiliate_offers ADD COLUMN IF NOT EXISTS verification_status VARCHAR(20) DEFAULT 'unverified';
ALTER TABLE affiliate_offers ADD COLUMN IF NOT EXISTS popularity_score INTEGER DEFAULT 0;
ALTER TABLE affiliate_offers ADD COLUMN IF NOT EXISTS conversion_rate DECIMAL(5,2);
ALTER TABLE affiliate_offers ADD COLUMN IF NOT EXISTS avg_order_value DECIMAL(10,2);
ALTER TABLE affiliate_offers ADD COLUMN IF NOT EXISTS target_audience TEXT[];
ALTER TABLE affiliate_offers ADD COLUMN IF NOT EXISTS content_opportunities JSONB DEFAULT '[]';
ALTER TABLE affiliate_offers ADD COLUMN IF NOT EXISTS seasonal_trends JSONB DEFAULT '{}';
ALTER TABLE affiliate_offers ADD COLUMN IF NOT EXISTS competitor_analysis JSONB DEFAULT '{}';

-- 2. Create affiliate_programs table for storing discovered programs
CREATE TABLE IF NOT EXISTS affiliate_programs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    program_name VARCHAR(255) NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    description TEXT,
    website_url TEXT,
    network_name VARCHAR(100),
    commission_rate DECIMAL(5,2),
    commission_type VARCHAR(50), -- percentage, fixed, tiered
    cookie_duration INTEGER, -- in days
    payment_terms TEXT,
    application_requirements TEXT,
    program_url TEXT,
    contact_email VARCHAR(255),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended', 'closed')),
    verification_status VARCHAR(20) DEFAULT 'unverified' CHECK (verification_status IN ('verified', 'unverified', 'pending', 'failed')),
    last_verified TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Research data
    research_score DECIMAL(3,2) DEFAULT 0.0,
    popularity_score INTEGER DEFAULT 0,
    conversion_rate DECIMAL(5,2),
    avg_order_value DECIMAL(10,2),
    target_audience TEXT[],
    content_opportunities JSONB DEFAULT '[]',
    seasonal_trends JSONB DEFAULT '{}',
    competitor_analysis JSONB DEFAULT '{}',
    
    -- Metadata
    source VARCHAR(50) DEFAULT 'manual', -- manual, linkup, llm_discovered, web_scraped
    data_quality_score DECIMAL(3,2) DEFAULT 0.0,
    last_researched TIMESTAMP WITH TIME ZONE,
    research_count INTEGER DEFAULT 0
);

-- 3. Create offer_research_sessions table for tracking research sessions
CREATE TABLE IF NOT EXISTS offer_research_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_name VARCHAR(255),
    search_terms TEXT[] NOT NULL,
    research_scope VARCHAR(50) DEFAULT 'comprehensive', -- quick, comprehensive, deep
    llm_analysis JSONB,
    discovered_programs UUID[] DEFAULT '{}',
    selected_offers UUID[] DEFAULT '{}',
    research_quality_score DECIMAL(3,2),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'completed', 'archived')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- 4. Create offer_analytics table for tracking offer performance
CREATE TABLE IF NOT EXISTS offer_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    offer_id UUID NOT NULL REFERENCES affiliate_offers(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    research_session_id UUID REFERENCES offer_research_sessions(id) ON DELETE SET NULL,
    
    -- Performance metrics
    click_count INTEGER DEFAULT 0,
    conversion_count INTEGER DEFAULT 0,
    revenue_generated DECIMAL(10,2) DEFAULT 0.0,
    commission_earned DECIMAL(10,2) DEFAULT 0.0,
    
    -- User interaction data
    time_spent_seconds INTEGER DEFAULT 0,
    selection_count INTEGER DEFAULT 0,
    content_ideas_generated INTEGER DEFAULT 0,
    
    -- Timestamps
    first_viewed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_viewed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 5. Create user_offer_preferences table for learning user preferences
CREATE TABLE IF NOT EXISTS user_offer_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Preference data
    preferred_networks TEXT[] DEFAULT '{}',
    preferred_commission_ranges TEXT[] DEFAULT '{}',
    preferred_categories TEXT[] DEFAULT '{}',
    preferred_difficulty_levels TEXT[] DEFAULT '{}',
    
    -- Learning data
    successful_offers UUID[] DEFAULT '{}',
    rejected_offers UUID[] DEFAULT '{}',
    content_preferences JSONB DEFAULT '{}',
    
    -- Metadata
    learning_enabled BOOLEAN DEFAULT true,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 6. Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_affiliate_offers_research_score ON affiliate_offers(research_score DESC);
CREATE INDEX IF NOT EXISTS idx_affiliate_offers_verification_status ON affiliate_offers(verification_status);
CREATE INDEX IF NOT EXISTS idx_affiliate_offers_last_verified ON affiliate_offers(last_verified);
CREATE INDEX IF NOT EXISTS idx_affiliate_offers_popularity_score ON affiliate_offers(popularity_score DESC);

CREATE INDEX IF NOT EXISTS idx_affiliate_programs_status ON affiliate_programs(status);
CREATE INDEX IF NOT EXISTS idx_affiliate_programs_verification_status ON affiliate_programs(verification_status);
CREATE INDEX IF NOT EXISTS idx_affiliate_programs_research_score ON affiliate_programs(research_score DESC);
CREATE INDEX IF NOT EXISTS idx_affiliate_programs_network_name ON affiliate_programs(network_name);
CREATE INDEX IF NOT EXISTS idx_affiliate_programs_commission_rate ON affiliate_programs(commission_rate DESC);

CREATE INDEX IF NOT EXISTS idx_offer_research_sessions_user_id ON offer_research_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_offer_research_sessions_status ON offer_research_sessions(status);
CREATE INDEX IF NOT EXISTS idx_offer_research_sessions_created_at ON offer_research_sessions(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_offer_analytics_offer_id ON offer_analytics(offer_id);
CREATE INDEX IF NOT EXISTS idx_offer_analytics_user_id ON offer_analytics(user_id);
CREATE INDEX IF NOT EXISTS idx_offer_analytics_revenue_generated ON offer_analytics(revenue_generated DESC);

CREATE INDEX IF NOT EXISTS idx_user_offer_preferences_user_id ON user_offer_preferences(user_id);

-- 7. Enable RLS on new tables
ALTER TABLE affiliate_programs ENABLE ROW LEVEL SECURITY;
ALTER TABLE offer_research_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE offer_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_offer_preferences ENABLE ROW LEVEL SECURITY;

-- 8. Create RLS policies for new tables
CREATE POLICY "Users can view affiliate programs" ON affiliate_programs
    FOR SELECT USING (true); -- Public read access for discovery

CREATE POLICY "Admins can manage affiliate programs" ON affiliate_programs
    FOR ALL USING (public.is_admin());

CREATE POLICY "Users can view own research sessions" ON offer_research_sessions
    FOR SELECT USING (user_id = public.user_id());

CREATE POLICY "Users can manage own research sessions" ON offer_research_sessions
    FOR ALL USING (user_id = public.user_id());

CREATE POLICY "Users can view own offer analytics" ON offer_analytics
    FOR SELECT USING (user_id = public.user_id());

CREATE POLICY "Users can manage own offer analytics" ON offer_analytics
    FOR ALL USING (user_id = public.user_id());

CREATE POLICY "Users can view own offer preferences" ON user_offer_preferences
    FOR SELECT USING (user_id = public.user_id());

CREATE POLICY "Users can manage own offer preferences" ON user_offer_preferences
    FOR ALL USING (user_id = public.user_id());

-- 9. Add updated_at triggers
CREATE TRIGGER update_affiliate_programs_updated_at BEFORE UPDATE ON affiliate_programs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_offer_research_sessions_updated_at BEFORE UPDATE ON offer_research_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_offer_analytics_updated_at BEFORE UPDATE ON offer_analytics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_offer_preferences_updated_at BEFORE UPDATE ON user_offer_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
