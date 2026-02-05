#!/bin/bash
# Apply database migrations to Supabase
# This script applies the new keywords and content_topics table migrations

set -e

echo "🚀 Applying database migrations..."

# Check if supabase CLI is installed
if ! command -v supabase &> /dev/null; then
    echo "❌ Error: Supabase CLI is not installed"
    echo "Install it with: brew install supabase/tap/supabase"
    exit 1
fi

# Check if we're in the right directory
if [ ! -d "supabase/migrations" ]; then
    echo "❌ Error: supabase/migrations directory not found"
    echo "Please run this script from the project root"
    exit 1
fi

echo "📋 Migration files to apply:"
echo "  - 20260115000001_create_keywords_table.sql"
echo "  - 20260115000002_create_content_topics_table.sql"
echo ""

# Apply migrations
echo "⏳ Applying migrations to local Supabase..."
supabase db reset

echo ""
echo "✅ Migrations applied successfully!"
echo ""
echo "📊 Verifying tables..."
supabase db dump --schema public --data-only=false | grep -E "CREATE TABLE.*(keywords|content_topics)" || echo "⚠️  Could not verify tables"

echo ""
echo "🎉 Database migration complete!"
echo ""
echo "Next steps:"
echo "  1. Verify tables in Supabase Studio"
echo "  2. Test RLS policies"
echo "  3. Proceed with backend service implementation"
