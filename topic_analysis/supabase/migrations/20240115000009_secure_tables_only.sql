-- Migration: Secure actual tables only
-- Description: Enables RLS only on actual tables, not views
-- Created: 2024-01-15
-- Author: DataForSEO Integration
-- Target: Remote Supabase database (dgcsqiaciyqvprtpopxg)

-- First, let's identify what we're working with
-- We'll only enable RLS on actual tables, not views

-- Check and enable RLS on high_value_keywords (if it's a table)
DO $$
BEGIN
    -- Check if high_value_keywords is a table
    IF EXISTS (
        SELECT 1 FROM pg_tables 
        WHERE schemaname = 'public' 
        AND tablename = 'high_value_keywords'
        AND tabletype = 'r'
    ) THEN
        ALTER TABLE high_value_keywords ENABLE ROW LEVEL SECURITY;
        RAISE NOTICE 'Enabled RLS on high_value_keywords table';
    ELSE
        RAISE NOTICE 'high_value_keywords is not a table, skipping RLS';
    END IF;
END $$;

-- Check and enable RLS on trending_subtopics (if it's a table)
DO $$
BEGIN
    -- Check if trending_subtopics is a table
    IF EXISTS (
        SELECT 1 FROM pg_tables 
        WHERE schemaname = 'public' 
        AND tablename = 'trending_subtopics'
        AND tabletype = 'r'
    ) THEN
        ALTER TABLE trending_subtopics ENABLE ROW LEVEL SECURITY;
        RAISE NOTICE 'Enabled RLS on trending_subtopics table';
    ELSE
        RAISE NOTICE 'trending_subtopics is not a table, skipping RLS';
    END IF;
END $$;

-- Create RLS policies for high_value_keywords (if it's a table)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_tables 
        WHERE schemaname = 'public' 
        AND tablename = 'high_value_keywords'
        AND tabletype = 'r'
    ) THEN
        -- Drop existing policies if they exist
        DROP POLICY IF EXISTS "Allow authenticated users to read high value keywords" ON high_value_keywords;
        DROP POLICY IF EXISTS "Allow authenticated users to insert high value keywords" ON high_value_keywords;
        DROP POLICY IF EXISTS "Allow authenticated users to update high value keywords" ON high_value_keywords;
        DROP POLICY IF EXISTS "Allow authenticated users to delete high value keywords" ON high_value_keywords;

        -- Allow authenticated users to read all high value keywords
        CREATE POLICY "Allow authenticated users to read high value keywords" ON high_value_keywords
            FOR SELECT
            TO authenticated
            USING (true);

        -- Allow authenticated users to insert high value keywords
        CREATE POLICY "Allow authenticated users to insert high value keywords" ON high_value_keywords
            FOR INSERT
            TO authenticated
            WITH CHECK (true);

        -- Allow authenticated users to update high value keywords
        CREATE POLICY "Allow authenticated users to update high value keywords" ON high_value_keywords
            FOR UPDATE
            TO authenticated
            USING (true)
            WITH CHECK (true);

        -- Allow authenticated users to delete high value keywords
        CREATE POLICY "Allow authenticated users to delete high value keywords" ON high_value_keywords
            FOR DELETE
            TO authenticated
            USING (true);

        RAISE NOTICE 'Created RLS policies for high_value_keywords table';
    END IF;
END $$;

-- Create RLS policies for trending_subtopics (if it's a table)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_tables 
        WHERE schemaname = 'public' 
        AND tablename = 'trending_subtopics'
        AND tabletype = 'r'
    ) THEN
        -- Drop existing policies if they exist
        DROP POLICY IF EXISTS "Allow authenticated users to read trending subtopics" ON trending_subtopics;
        DROP POLICY IF EXISTS "Allow authenticated users to insert trending subtopics" ON trending_subtopics;
        DROP POLICY IF EXISTS "Allow authenticated users to update trending subtopics" ON trending_subtopics;
        DROP POLICY IF EXISTS "Allow authenticated users to delete trending subtopics" ON trending_subtopics;

        -- Allow authenticated users to read all trending subtopics
        CREATE POLICY "Allow authenticated users to read trending subtopics" ON trending_subtopics
            FOR SELECT
            TO authenticated
            USING (true);

        -- Allow authenticated users to insert trending subtopics
        CREATE POLICY "Allow authenticated users to insert trending subtopics" ON trending_subtopics
            FOR INSERT
            TO authenticated
            WITH CHECK (true);

        -- Allow authenticated users to update trending subtopics
        CREATE POLICY "Allow authenticated users to update trending subtopics" ON trending_subtopics
            FOR UPDATE
            TO authenticated
            USING (true)
            WITH CHECK (true);

        -- Allow authenticated users to delete trending subtopics
        CREATE POLICY "Allow authenticated users to delete trending subtopics" ON trending_subtopics
            FOR DELETE
            TO authenticated
            USING (true);

        RAISE NOTICE 'Created RLS policies for trending_subtopics table';
    END IF;
END $$;

-- For views, we need to identify and secure the underlying tables
-- Let's create a function to help identify view dependencies
CREATE OR REPLACE FUNCTION get_view_dependencies(view_name TEXT)
RETURNS TABLE(dependent_table TEXT, table_type TEXT) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.tablename::TEXT as dependent_table,
        CASE 
            WHEN t.tabletype = 'r' THEN 'TABLE'
            WHEN t.tabletype = 'v' THEN 'VIEW'
            WHEN t.tabletype = 'm' THEN 'MATERIALIZED VIEW'
            ELSE 'OTHER'
        END as table_type
    FROM pg_tables t
    WHERE t.schemaname = 'public'
    AND (
        -- This is a simplified check - in practice, you'd parse the view definition
        t.tablename LIKE '%' || view_name || '%'
        OR view_name LIKE '%' || t.tablename || '%'
    )
    UNION
    SELECT 
        v.viewname::TEXT as dependent_table,
        'VIEW' as table_type
    FROM pg_views v
    WHERE v.schemaname = 'public'
    AND (
        v.viewname LIKE '%' || view_name || '%'
        OR view_name LIKE '%' || v.viewname || '%'
    );
END;
$$ LANGUAGE plpgsql;

-- Log information about what we found
DO $$
DECLARE
    obj_name TEXT;
    obj_type TEXT;
BEGIN
    RAISE NOTICE 'Checking object types for security...';
    
    -- Check each object
    FOR obj_name IN ('api_performance_metrics', 'high_value_keywords', 'trending_subtopics')
    LOOP
        -- Check if it's a table
        IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = obj_name AND tabletype = 'r') THEN
            obj_type := 'TABLE';
        -- Check if it's a view
        ELSIF EXISTS (SELECT 1 FROM pg_views WHERE schemaname = 'public' AND viewname = obj_name) THEN
            obj_type := 'VIEW';
        ELSE
            obj_type := 'NOT FOUND';
        END IF;
        
        RAISE NOTICE 'Object % is a %', obj_name, obj_type;
    END LOOP;
END $$;
