# Supabase API Keys Integration

## Overview

All API keys are stored in Supabase `api_keys` table, **not** in environment variables. Only Supabase credentials should be in the `.env` file.

## .env File Configuration

Your `.env` file should **ONLY** contain Supabase credentials:

```env
# Supabase Configuration (REQUIRED)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
```

**OR** (alternative variable name):
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-supabase-anon-key
```

## Supabase Database Setup

All API keys must be stored in the `api_keys` table with the following structure:

```sql
CREATE TABLE IF NOT EXISTS api_keys (
    provider VARCHAR(50) PRIMARY KEY,
    key_value TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Adding API Keys

Insert API keys into Supabase:

```sql
-- Linkup API key
INSERT INTO api_keys (provider, key_value) 
VALUES ('linkup', 'your-linkup-api-key-here')
ON CONFLICT (provider) DO UPDATE SET key_value = EXCLUDED.key_value;

-- OpenAI API key (if needed)
INSERT INTO api_keys (provider, key_value) 
VALUES ('openai', 'your-openai-api-key-here')
ON CONFLICT (provider) DO UPDATE SET key_value = EXCLUDED.key_value;

-- Gemini API key (if needed)
INSERT INTO api_keys (provider, key_value) 
VALUES ('gemini', 'your-gemini-api-key-here')
ON CONFLICT (provider) DO UPDATE SET key_value = EXCLUDED.key_value;
```

### Verifying API Keys

Check which API keys are stored:

```sql
SELECT provider, 
       CASE 
         WHEN LENGTH(key_value) > 0 THEN 'Key exists (length: ' || LENGTH(key_value) || ')'
         ELSE 'Key is empty'
       END as status
FROM api_keys
ORDER BY provider;
```

## Code Usage

The system automatically fetches API keys from Supabase:

```python
from supabase_client import get_linkup_api_key, get_api_key

# Get Linkup API key
linkup_key = get_linkup_api_key()

# Get any API key by provider name
openai_key = get_api_key('openai')
gemini_key = get_api_key('gemini')
anthropic_key = get_api_key('anthropic')
```

## How It Works

1. **Supabase Client Initialization**: 
   - Reads `SUPABASE_URL` and `SUPABASE_KEY` from environment
   - Creates a cached Supabase client instance

2. **API Key Retrieval**:
   - Queries `api_keys` table: `SELECT key_value FROM api_keys WHERE provider = 'linkup'`
   - Returns the key value if found
   - Logs warnings if not found

3. **No Environment Variable Fallback**:
   - The system **only** reads from Supabase
   - No fallback to environment variables for API keys
   - This ensures all keys are centrally managed in Supabase

## Supported Providers

The following provider names can be used in the `api_keys` table:

- `linkup` - Linkup web search API
- `openai` - OpenAI API
- `gemini` - Google Gemini API
- `anthropic` - Anthropic Claude API
- `cohere` - Cohere API
- `mistral` - Mistral AI API
- `kimi` - Kimi API
- `moonshot` - Moonshot AI API

Add more providers as needed by inserting records with the appropriate `provider` name.

## Troubleshooting

### Issue: "Supabase credentials not found"
**Solution**: Make sure `SUPABASE_URL` and `SUPABASE_KEY` are set in `.env`

### Issue: "Linkup API key not found in Supabase api_keys table"
**Solution**: 
1. Verify the key exists: `SELECT * FROM api_keys WHERE provider = 'linkup';`
2. If missing, insert it using the SQL above
3. Check that `key_value` is not empty

### Issue: "Failed to initialize Supabase client"
**Solution**:
1. Verify `SUPABASE_URL` format (should be `https://xxx.supabase.co`)
2. Verify `SUPABASE_KEY` is the correct anon/public key
3. Check Supabase project is active

## Benefits

1. **Centralized Management**: All API keys in one place (Supabase)
2. **Security**: Keys not stored in `.env` files that might be committed to git
3. **Easy Updates**: Update keys in Supabase without changing code
4. **Audit Trail**: Track when keys were added/updated via `created_at`/`updated_at`
5. **Multi-Environment**: Different keys per environment by using different Supabase projects

