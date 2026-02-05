-- Migration: Secure objects with simple approach
-- Description: Try to enable RLS on objects, handle errors gracefully
-- Created: 2024-01-15
-- Author: DataForSEO Integration
-- Target: Remote Supabase database (dgcsqiaciyqvprtpopxg)

-- Try to enable RLS on high_value_keywords
DO $$
BEGIN
    BEGIN
        ALTER TABLE high_value_keywords ENABLE ROW LEVEL SECURITY;
        RAISE NOTICE 'Successfully enabled RLS on high_value_keywords';
    EXCEPTION
        WHEN OTHERS THEN
            RAISE NOTICE 'Could not enable RLS on high_value_keywords: %', SQLERRM;
    END;
END $$;

-- Try to enable RLS on trending_subtopics
DO $$
BEGIN
    BEGIN
        ALTER TABLE trending_subtopics ENABLE ROW LEVEL SECURITY;
        RAISE NOTICE 'Successfully enabled RLS on trending_subtopics';
    EXCEPTION
        WHEN OTHERS THEN
            RAISE NOTICE 'Could not enable RLS on trending_subtopics: %', SQLERRM;
    END;
END $$;

-- Try to enable RLS on api_performance_metrics
DO $$
BEGIN
    BEGIN
        ALTER TABLE api_performance_metrics ENABLE ROW LEVEL SECURITY;
        RAISE NOTICE 'Successfully enabled RLS on api_performance_metrics';
    EXCEPTION
        WHEN OTHERS THEN
            RAISE NOTICE 'Could not enable RLS on api_performance_metrics: %', SQLERRM;
    END;
END $$;

-- Create RLS policies for high_value_keywords (if RLS was enabled)
DO $$
BEGIN
    -- Check if RLS is enabled on high_value_keywords
    IF EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
        AND c.relname = 'high_value_keywords'
        AND c.relrowsecurity = true
    ) THEN
        -- Drop existing policies if they exist
        DROP POLICY IF EXISTS "Allow authenticated users to read high value keywords" ON high_value_keywords;
        DROP POLICY IF EXISTS "Allow authenticated users to insert high value keywords" ON high_value_keywords;
        DROP POLICY IF EXISTS "Allow authenticated users to update high value keywords" ON high_value_keywords;
        DROP POLICY IF EXISTS "Allow authenticated users to delete high value keywords" ON high_value_keywords;

        -- Create new policies
        CREATE POLICY "Allow authenticated users to read high value keywords" ON high_value_keywords
            FOR SELECT
            TO authenticated
            USING (true);

        CREATE POLICY "Allow authenticated users to insert high value keywords" ON high_value_keywords
            FOR INSERT
            TO authenticated
            WITH CHECK (true);

        CREATE POLICY "Allow authenticated users to update high value keywords" ON high_value_keywords
            FOR UPDATE
            TO authenticated
            USING (true)
            WITH CHECK (true);

        CREATE POLICY "Allow authenticated users to delete high value keywords" ON high_value_keywords
            FOR DELETE
            TO authenticated
            USING (true);

        RAISE NOTICE 'Created RLS policies for high_value_keywords';
    ELSE
        RAISE NOTICE 'RLS not enabled on high_value_keywords, skipping policies';
    END IF;
END $$;

-- Create RLS policies for trending_subtopics (if RLS was enabled)
DO $$
BEGIN
    -- Check if RLS is enabled on trending_subtopics
    IF EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
        AND c.relname = 'trending_subtopics'
        AND c.relrowsecurity = true
    ) THEN
        -- Drop existing policies if they exist
        DROP POLICY IF EXISTS "Allow authenticated users to read trending subtopics" ON trending_subtopics;
        DROP POLICY IF EXISTS "Allow authenticated users to insert trending subtopics" ON trending_subtopics;
        DROP POLICY IF EXISTS "Allow authenticated users to update trending subtopics" ON trending_subtopics;
        DROP POLICY IF EXISTS "Allow authenticated users to delete trending subtopics" ON trending_subtopics;

        -- Create new policies
        CREATE POLICY "Allow authenticated users to read trending subtopics" ON trending_subtopics
            FOR SELECT
            TO authenticated
            USING (true);

        CREATE POLICY "Allow authenticated users to insert trending subtopics" ON trending_subtopics
            FOR INSERT
            TO authenticated
            WITH CHECK (true);

        CREATE POLICY "Allow authenticated users to update trending subtopics" ON trending_subtopics
            FOR UPDATE
            TO authenticated
            USING (true)
            WITH CHECK (true);

        CREATE POLICY "Allow authenticated users to delete trending subtopics" ON trending_subtopics
            FOR DELETE
            TO authenticated
            USING (true);

        RAISE NOTICE 'Created RLS policies for trending_subtopics';
    ELSE
        RAISE NOTICE 'RLS not enabled on trending_subtopics, skipping policies';
    END IF;
END $$;

-- Create RLS policies for api_performance_metrics (if RLS was enabled)
DO $$
BEGIN
    -- Check if RLS is enabled on api_performance_metrics
    IF EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
        AND c.relname = 'api_performance_metrics'
        AND c.relrowsecurity = true
    ) THEN
        -- Drop existing policies if they exist
        DROP POLICY IF EXISTS "Allow authenticated users to read API performance metrics" ON api_performance_metrics;
        DROP POLICY IF EXISTS "Allow authenticated users to insert API performance metrics" ON api_performance_metrics;
        DROP POLICY IF EXISTS "Allow authenticated users to update API performance metrics" ON api_performance_metrics;
        DROP POLICY IF EXISTS "Allow authenticated users to delete API performance metrics" ON api_performance_metrics;

        -- Create new policies
        CREATE POLICY "Allow authenticated users to read API performance metrics" ON api_performance_metrics
            FOR SELECT
            TO authenticated
            USING (true);

        CREATE POLICY "Allow authenticated users to insert API performance metrics" ON api_performance_metrics
            FOR INSERT
            TO authenticated
            WITH CHECK (true);

        CREATE POLICY "Allow authenticated users to update API performance metrics" ON api_performance_metrics
            FOR UPDATE
            TO authenticated
            USING (true)
            WITH CHECK (true);

        CREATE POLICY "Allow authenticated users to delete API performance metrics" ON api_performance_metrics
            FOR DELETE
            TO authenticated
            USING (true);

        RAISE NOTICE 'Created RLS policies for api_performance_metrics';
    ELSE
        RAISE NOTICE 'RLS not enabled on api_performance_metrics, skipping policies';
    END IF;
END $$;
