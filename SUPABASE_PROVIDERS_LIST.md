# Complete List of API Key Providers for Supabase

This document lists all providers that require API keys to be stored in the Supabase `api_keys` table.

## Required API Keys

### 1. **linkup** (REQUIRED for web search)
- **Purpose**: Web search and research via Linkup API
- **Used in**: Evidence collection, web search fallback when RAG is insufficient
- **Priority**: High - needed for comprehensive research
- **SQL**:
  ```sql
  INSERT INTO api_keys (provider, key_value) 
  VALUES ('linkup', 'your-linkup-api-key-here')
  ON CONFLICT (provider) DO UPDATE SET key_value = EXCLUDED.key_value;
  ```

## LLM Provider API Keys

These are used for article generation. You only need to store keys for the providers you plan to use.

### 2. **openai**
- **Purpose**: OpenAI models (GPT-4, GPT-3.5, etc.)
- **Used in**: Article generation, content creation
- **Models**: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`, etc.
- **SQL**:
  ```sql
  INSERT INTO api_keys (provider, key_value) 
  VALUES ('openai', 'sk-your-openai-api-key')
  ON CONFLICT (provider) DO UPDATE SET key_value = EXCLUDED.key_value;
  ```

### 3. **gemini**
- **Purpose**: Google Gemini models
- **Used in**: Article generation, content creation
- **Models**: `gemini-2.5-flash`, `gemini-2.5`, `gemini-1.5-pro`, `gemini-1.5-flash`
- **SQL**:
  ```sql
  INSERT INTO api_keys (provider, key_value) 
  VALUES ('gemini', 'your-gemini-api-key')
  ON CONFLICT (provider) DO UPDATE SET key_value = EXCLUDED.key_value;
  ```

### 4. **anthropic**
- **Purpose**: Anthropic Claude models
- **Used in**: Article generation, content creation
- **Models**: `claude-3-5-sonnet-20241022`, `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- **SQL**:
  ```sql
  INSERT INTO api_keys (provider, key_value) 
  VALUES ('anthropic', 'sk-ant-your-anthropic-api-key')
  ON CONFLICT (provider) DO UPDATE SET key_value = EXCLUDED.key_value;
  ```

### 5. **deepseek**
- **Purpose**: DeepSeek models (OpenAI-compatible)
- **Used in**: Article generation, content creation
- **Models**: `deepseek-chat`, `deepseek-coder`, `deepseek-reasoner`
- **SQL**:
  ```sql
  INSERT INTO api_keys (provider, key_value) 
  VALUES ('deepseek', 'your-deepseek-api-key')
  ON CONFLICT (provider) DO UPDATE SET key_value = EXCLUDED.key_value;
  ```

### 6. **moonshot**
- **Purpose**: Moonshot AI models (OpenAI-compatible)
- **Used in**: Article generation, content creation
- **Models**: `moonshot-v1-8k`, `moonshot-v1-32k`, `moonshot-v1-128k`
- **SQL**:
  ```sql
  INSERT INTO api_keys (provider, key_value) 
  VALUES ('moonshot', 'your-moonshot-api-key')
  ON CONFLICT (provider) DO UPDATE SET key_value = EXCLUDED.key_value;
  ```

### 7. **kimi**
- **Purpose**: Kimi AI models (OpenAI-compatible, same as Moonshot)
- **Used in**: Article generation, content creation
- **Models**: `kimi-k2-0711-preview`, `kimi-k2-moonshine`
- **Note**: Uses same API as Moonshot, can share the same key
- **SQL**:
  ```sql
  INSERT INTO api_keys (provider, key_value) 
  VALUES ('kimi', 'your-kimi-api-key')
  ON CONFLICT (provider) DO UPDATE SET key_value = EXCLUDED.key_value;
  ```

### 8. **cohere**
- **Purpose**: Cohere models
- **Used in**: Article generation, content creation
- **SQL**:
  ```sql
  INSERT INTO api_keys (provider, key_value) 
  VALUES ('cohere', 'your-cohere-api-key')
  ON CONFLICT (provider) DO UPDATE SET key_value = EXCLUDED.key_value;
  ```

### 9. **mistral**
- **Purpose**: Mistral AI models
- **Used in**: Article generation, content creation
- **SQL**:
  ```sql
  INSERT INTO api_keys (provider, key_value) 
  VALUES ('mistral', 'your-mistral-api-key')
  ON CONFLICT (provider) DO UPDATE SET key_value = EXCLUDED.key_value;
  ```

## Optional API Keys

### 10. **rag** (Optional - if RAG service requires authentication)
- **Purpose**: RAG (Retrieval Augmented Generation) service authentication
- **Used in**: RAG queries for knowledge base search
- **Note**: Only needed if your RAG endpoint requires API key authentication
- **SQL**:
  ```sql
  INSERT INTO api_keys (provider, key_value) 
  VALUES ('rag', 'your-rag-api-key')
  ON CONFLICT (provider) DO UPDATE SET key_value = EXCLUDED.key_value;
  ```

## Quick Setup Script

Run this SQL in your Supabase SQL editor to set up all providers at once (replace with your actual keys):

```sql
-- Required: Linkup (for web search)
INSERT INTO api_keys (provider, key_value) 
VALUES ('linkup', 'your-linkup-api-key-here')
ON CONFLICT (provider) DO UPDATE SET key_value = EXCLUDED.key_value;

-- LLM Providers (add only the ones you use)
INSERT INTO api_keys (provider, key_value) 
VALUES 
  ('openai', 'sk-your-openai-key'),
  ('gemini', 'your-gemini-key'),
  ('anthropic', 'sk-ant-your-anthropic-key'),
  ('deepseek', 'your-deepseek-key'),
  ('moonshot', 'your-moonshot-key'),
  ('kimi', 'your-kimi-key'),
  ('cohere', 'your-cohere-key'),
  ('mistral', 'your-mistral-key')
ON CONFLICT (provider) DO UPDATE SET key_value = EXCLUDED.key_value;
```

## Verification Query

Check which API keys you have stored:

```sql
SELECT 
  provider,
  CASE 
    WHEN LENGTH(key_value) > 0 THEN '✓ Key exists (' || LENGTH(key_value) || ' chars)'
    ELSE '✗ Key is empty'
  END as status,
  created_at,
  updated_at
FROM api_keys
ORDER BY 
  CASE provider
    WHEN 'linkup' THEN 1  -- Required
    ELSE 2  -- Optional
  END,
  provider;
```

## Minimum Required Setup

**At minimum, you need:**
1. ✅ **linkup** - For web search and research (REQUIRED)

**Plus at least one LLM provider:**
2. ✅ **openai** OR **gemini** OR **anthropic** (or any other LLM provider you prefer)

## Provider Usage Priority

1. **linkup** - Always used when RAG is insufficient or disabled
2. **LLM Provider** - Used based on what's specified in the research request (`provider` field)
3. **RAG** - Optional, only if your RAG service requires authentication

## Notes

- **Kimi and Moonshot**: These use the same API endpoint, so you can use the same API key for both if you have a Moonshot account
- **Provider Names**: Must match exactly (lowercase) as shown above
- **Key Storage**: All keys are encrypted at rest in Supabase
- **Access Control**: Use Supabase Row Level Security (RLS) to control who can read/write API keys

