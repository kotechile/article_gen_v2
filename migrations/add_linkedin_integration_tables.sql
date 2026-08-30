-- LinkedIn Accounts table for personal LinkedIn OAuth connection
CREATE TABLE IF NOT EXISTS public.linkedin_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    account_type TEXT NOT NULL DEFAULT 'personal',
    linkedin_urn TEXT NOT NULL,
    account_name TEXT NOT NULL,
    profile_picture_url TEXT,
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Index for lookup by user_id
CREATE INDEX IF NOT EXISTS idx_linkedin_accounts_user_id ON public.linkedin_accounts(user_id);

-- Enable RLS
ALTER TABLE public.linkedin_accounts ENABLE ROW LEVEL SECURITY;

-- RLS Policies
DROP POLICY IF EXISTS "Users can view own linkedin accounts" ON public.linkedin_accounts;
CREATE POLICY "Users can view own linkedin accounts"
    ON public.linkedin_accounts FOR SELECT
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own linkedin accounts" ON public.linkedin_accounts;
CREATE POLICY "Users can insert own linkedin accounts"
    ON public.linkedin_accounts FOR INSERT
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update own linkedin accounts" ON public.linkedin_accounts;
CREATE POLICY "Users can update own linkedin accounts"
    ON public.linkedin_accounts FOR UPDATE
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete own linkedin accounts" ON public.linkedin_accounts;
CREATE POLICY "Users can delete own linkedin accounts"
    ON public.linkedin_accounts FOR DELETE
    USING (auth.uid() = user_id);

-- Add LinkedIn publishing tracking columns to Titles table if not present
ALTER TABLE public."Titles"
    ADD COLUMN IF NOT EXISTS linkedin_post_content TEXT,
    ADD COLUMN IF NOT EXISTS last_linkedin_post_id TEXT,
    ADD COLUMN IF NOT EXISTS last_linkedin_post_urn TEXT,
    ADD COLUMN IF NOT EXISTS last_linkedin_post_url TEXT,
    ADD COLUMN IF NOT EXISTS last_linkedin_status TEXT,
    ADD COLUMN IF NOT EXISTS last_linkedin_published_at TIMESTAMPTZ;
