# LLM Service & Endpoint Documentation

## Overview
The `LLMService` provides a centralized, unified interface for interacting with multiple Large Language Model (LLM) providers (Gemini, OpenAI, Anthropic, DeepSeek, Kimi). It is designed to reliably handle provider configuration, API key management, and default fallback logic, allowing the application to switch models dynamically without code changes.

## Architecture

### 1. `LLMService`
- **Location**: `backend/src/services/llm/llm_service.py`
- **Responsibility**: 
  - Fetches provider configuration from Supabase.
  - Instantiates the correct `LLMProvider` subclass.
  - Handles "Default" provider selection if no specific model is requested.
  - Manages API key retrieval (via robust two-step fetch).

### 2. `LLMProvider` (Abstract Base Class)
- **Location**: `backend/src/services/llm/llm_provider.py`
- **Responsibility**: Defines the standard interface (`generate`) and response structure (`LLMResponse`) that all concrete providers must implement.

### 3. Concrete Providers
- **Location**: `backend/src/services/llm/providers.py`
- **Implementations**:
  - `GeminiProvider`: Connects to Google's Generative Language API.
  - `OpenAIProvider`: Connects to OpenAI API (and compatible APIs).
  - `AnthropicProvider`: Connects to Anthropic Claude API.
  - `DeepSeekProvider`: OpenAI-compatible wrapper for DeepSeek.
  - `KimiProvider`: OpenAI-compatible wrapper for Moonshot AI (Kimi).

### 4. Database Integration (Supabase)
- **Table `llm_providers`**: Stores provider metadata (`name`, `provider` type, `model_name`, `base_url`, `is_active`, `is_default`, `api_key_id`).
- **Table `api_keys`**: Stores the actual sensitive API keys.

---

## 1. Using the Python API
The service is available as a global singleton instance `llm_service`.

```python
from src.services.llm.llm_service import llm_service

# 1. Use the Default Provider (Recommended)
response = await llm_service.generate_text(
    prompt="Explain quantum computing in 5 words."
)

# 2. Use a Specific Provider by Name (e.g. "Gemini 3 Flash", "Deep Seek 3.2")
response = await llm_service.generate_text(
    prompt="Generate code for...",
    provider="Gemini 3 Flash", # Matches 'name' or 'provider' column in DB
    temperature=0.7,
    max_tokens=1000
)

print(response.content)      # The generated text
print(response.provider)     # e.g., 'gemini', 'openai'
print(response.model_name)   # e.g., 'gemini-3-flash-preview'
print(response.usage)        # Token usage stats
```

## 2. Using the REST API
A dedicated endpoint allows frontend or external services to generate text.

- **Endpoint**: `POST /api/llm/generate`
- **Headers**: `Content-Type: application/json`

### Request Body
```json
{
  "prompt": "What are 3 niche ideas for 2026?",
  "provider": "Gemini 3 Flash",  // Optional. Omit to use default.
  "max_tokens": 500,             // Optional
  "temperature": 0.8             // Optional
}
```

### Success Response (200 OK)
```json
{
  "content": "1. AI Ethics Consulting...",
  "provider": "gemini",
  "model": "gemini-3-flash-preview",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```

---

## Configuration & Management

To add or modify providers, you work directly with Supabase tables. No code deployment is needed.

### Adding a New Provider
1.  **Add API Key**: Insert a record into `api_keys` table.
    - `service`: "google" (or "openai", etc.)
    - `key_value`: "YOUR_ACTUAL_API_KEY"
    - `is_active`: `TRUE`
2.  **Add Provider Config**: Insert a record into `llm_providers` table.
    - `name`: "Gemini Pro 1.5" (Display name/Identifier)
    - `provider`: "google" (Must match supported types in `providers.py`)
    - `model_name`: "gemini-1.5-pro" (Actual model ID sent to API)
    - `api_key_id`: (UUID of the key added above)
    - `is_active`: `TRUE`
    - `is_default`: `FALSE` (Set to `TRUE` to make this the system default)

---

## Troubleshooting

### `LLM Provider 'X' not found or not active`
- Check `llm_providers` in Supabase.
- Ensure `is_active` is set to `TRUE`.
- Verify the `name` or `provider` column matches your request string.
- Note: The search is case-insensitive.

### `No API key linked` or `No API key found`
- Ensure the `api_key_id` in `llm_providers` is valid and exists in `api_keys`.
- Ensure the key record in `api_keys` has `is_active: TRUE`.

### `OpenAI API Error: 404`
- If using a custom `base_url`, ensure it does NOT end with a slash if the code appends endpoints, OR ensure the provider class handles it correctly.
- **Fix Applied**: `OpenAIProvider` now automatically appends `/chat/completions` if missing from `base_url`.

### `PGRST200` / Relationship Errors
- The service uses a **two-step fetch** (Provider -> then API Key) to avoid Postgrest relationship mapping errors. If you see DB errors, verify the table schemas and foreign keys exist, but the service logic is designed to be resilient to this.
