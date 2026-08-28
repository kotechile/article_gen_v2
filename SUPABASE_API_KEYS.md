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
- `flux` / `fluxapi` / `kie` - Flux image generation API
- `stability` / `stable_diffusion` - Stability AI (SD3) image generation API

Add more providers as needed by inserting records with the appropriate `provider` name.

---

## Image Generation Models & Application Routing (`used_for`)

### Overview

Image generation models (Flux, Stable Diffusion, Google Imagen) and their application assignments are managed dynamically in Supabase:

1. **`used_for`**: Stores application mappings. It contains the image applications:
   - `'article_image'` — Model for generating article illustrations / featured images.
   - `'infographics'` — Model for generating infographic images.
   - **`llm_image_id`**: Foreign key pointing to `llm_providers_image.id`.
2. **`llm_providers_image`**: Stores image models and provider configurations:
   - `id`: Primary key (UUID).
   - `model_name`: Technical model identifier (e.g. `flux-kontext-pro`, `sd3`, `imagen-4.0-generate-001`, `flux-2/flex-image-to-image`).
   - `provider`: Provider family (e.g. `flux`, `stability`, `google`, `kie.ai`).
   - `display_name`: Human-readable label (e.g. `Flux Kontext Pro`).
   - `api_keys_id`: Foreign key pointing to `api_keys.id`.
   - `is_active`: Boolean flag indicating if the model is enabled.
3. **`api_keys`**: Central secrets store:
   - `id`: Primary key (UUID).
   - `provider`: Provider name.
   - `key_value`: The actual API key value.

### Resolution Architecture

```text
[used_for] (application = 'article_image' or 'infographics')
     │
     └──> llm_image_id
               │
               ▼
[llm_providers_image] (id = llm_image_id)
     │   • model_name  --> Used in generation request
     │   • provider    --> Routes to provider function (Flux, SD3, Imagen)
     └──> api_keys_id
               │
               ▼
[api_keys] (id = api_keys_id)
         • key_value   --> Decrypted API key for provider authentication
```

### SQL Table Setup & Configuration

```sql
-- 1. Insert Image Provider API Key
INSERT INTO api_keys (id, provider, key_value)
VALUES ('00000000-0000-0000-0000-000000000001', 'flux', 'your-flux-api-key-here')
ON CONFLICT (id) DO UPDATE SET key_value = EXCLUDED.key_value;

-- 2. Insert Image Model into llm_providers_image
INSERT INTO llm_providers_image (id, model_name, display_name, provider, api_keys_id, is_active)
VALUES (
    '11111111-1111-1111-1111-111111111111',
    'flux-kontext-pro',
    'Flux Kontext Pro',
    'flux',
    '00000000-0000-0000-0000-000000000001',
    TRUE
)
ON CONFLICT (id) DO UPDATE SET
    model_name = EXCLUDED.model_name,
    api_keys_id = EXCLUDED.api_keys_id;

-- 3. Assign Model to Application in used_for
INSERT INTO used_for (application, llm_image_id)
VALUES 
    ('article_image', '11111111-1111-1111-1111-111111111111'),
    ('infographics', '11111111-1111-1111-1111-111111111111')
ON CONFLICT (application) DO UPDATE SET llm_image_id = EXCLUDED.llm_image_id;
```

### Python Code Usage

The backend utility module [`supabase_client.py`](file:///Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/supabase_client.py) provides helpers to resolve image models and API keys:

```python
from supabase_client import (
    resolve_image_provider,
    get_image_provider_for_application,
    get_image_applications_config,
    IMAGE_APP_ARTICLE_IMAGE,
    IMAGE_APP_INFOGRAPHICS,
)

# 1. Resolve image provider for article images
resolved = resolve_image_provider(application="article_image")
# Returns:
# {
#     "provider": "flux",
#     "model": "flux-kontext-pro",
#     "api_key": "your-flux-api-key",
#     "display_name": "Flux Kontext Pro",
#     "llm_image_id": "11111111-1111-1111-1111-111111111111",
#     "application": "article_image",
#     "source": "used_for"
# }

# 2. Get provider, model, and key as a tuple
provider, model_name, api_key = get_image_provider_for_application("infographics")

# 3. Get application mapping configuration (safe for UI, no secret keys exposed)
app_config = get_image_applications_config()
# {
#     "article_image": {"model_name": "flux-kontext-pro", "provider": "flux", "has_api_key": True, ...},
#     "infographics": {"model_name": "sd3", "provider": "stability", "has_api_key": True, ...}
# }
```

### Image Generation API Endpoints

- **`GET /api/v1/images/application-config`**: Returns the active image models configured for each application (`article_image`, `infographics`).
- **`POST /api/v1/images/generate-ai`**: Generates an image. Accepts:
  ```json
  {
      "prompt": "Modern office workspace with natural light",
      "application": "article_image",
      "aspectRatio": "16:9",
      "user_id": "your-user-uuid"
  }
  ```
  If `model` is omitted, the endpoint automatically resolves the model and API key configured in `used_for` for the given `application` (defaults to `'article_image'`).

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

