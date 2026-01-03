# .env Configuration Checklist

Use this checklist to verify your `.env` file is configured correctly.

## Required Variables for Supabase Integration

### ✅ Supabase Credentials (REQUIRED)
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
```
**OR** (alternative name):
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-supabase-anon-key
```

**Notes:**
- `SUPABASE_URL`: Your Supabase project URL (found in Supabase dashboard → Settings → API)
- `SUPABASE_KEY` or `SUPABASE_ANON_KEY`: Your Supabase anonymous/public key (also in Settings → API)
- The system will use either `SUPABASE_KEY` or `SUPABASE_ANON_KEY` (checks both)

### ✅ Supabase Database Setup
Make sure your Supabase `api_keys` table has a record:
```sql
SELECT key_value FROM api_keys WHERE provider = 'linkup';
```

If it doesn't exist, insert it:
```sql
INSERT INTO api_keys (provider, key_value) 
VALUES ('linkup', 'your-linkup-api-key-here');
```

## Important: API Keys Storage

**All API keys are stored in Supabase, NOT in .env file.**

The `.env` file should **ONLY** contain:
- `SUPABASE_URL`
- `SUPABASE_KEY` (or `SUPABASE_ANON_KEY`)

All other API keys (Linkup, OpenAI, Gemini, Anthropic, etc.) should be stored in the Supabase `api_keys` table with the `provider` column indicating which service they belong to.

### RAG Configuration (OPTIONAL)
```
RAG_ENDPOINT=http://localhost:8080/query_simple
RAG_COLLECTION=career_and_personal_finance
```

### Celery Configuration (OPTIONAL - has defaults)
```
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

## Verification Steps

1. **Check Supabase URL format:**
   - Should start with `https://`
   - Should end with `.supabase.co`
   - Example: `https://abcdefghijklmnop.supabase.co`

2. **Check Supabase Key:**
   - Should be a long string (typically starts with `eyJ...`)
   - This is the "anon" or "public" key from Supabase dashboard

3. **Verify API keys in Supabase:**
   - Run this query in Supabase SQL editor to see all stored API keys:
     ```sql
     SELECT provider, 
            CASE 
              WHEN LENGTH(key_value) > 0 THEN 'Key exists (length: ' || LENGTH(key_value) || ')'
              ELSE 'Key is empty'
            END as status
     FROM api_keys
     ORDER BY provider;
     ```
   - Specifically check for Linkup:
     ```sql
     SELECT key_value FROM api_keys WHERE provider = 'linkup';
     ```

4. **Test the connection:**
   - After setting up, restart your Celery worker
   - Check the logs for: `"Supabase client initialized successfully"`
   - Check for: `"Successfully fetched linkup API key from Supabase"`

## Common Issues

### Issue: "Supabase credentials not found"
**Solution:** Make sure both `SUPABASE_URL` and `SUPABASE_KEY` (or `SUPABASE_ANON_KEY`) are set in your `.env` file.

### Issue: "Linkup API key not found in Supabase"
**Solution:** 
1. Verify the `api_keys` table exists in Supabase
2. Check that there's a record with `provider = 'linkup'`
3. Verify the `key_value` column is not empty

### Issue: "Failed to initialize Supabase client"
**Solution:**
1. Check that `SUPABASE_URL` is correct (no trailing slash)
2. Verify `SUPABASE_KEY` is the correct anon/public key
3. Check your Supabase project is active and accessible

## Quick Test

After configuring, you can test by running a research task. Check the logs for:
- ✅ `"Supabase client initialized successfully"`
- ✅ `"Successfully fetched linkup API key from Supabase"`
- ✅ `"Using Linkup API key: ..."` (in evidence collection logs)

If you see warnings like:
- ⚠ `"Linkup API key not found in Supabase api_keys table (provider='linkup')"`
- ⚠ `"LINKUP_API_KEY not found in Supabase or environment, skipping web search"`

Then you need to:
1. Add the Linkup API key to Supabase `api_keys` table:
   ```sql
   INSERT INTO api_keys (provider, key_value) 
   VALUES ('linkup', 'your-linkup-api-key-here')
   ON CONFLICT (provider) DO UPDATE SET key_value = EXCLUDED.key_value;
   ```

