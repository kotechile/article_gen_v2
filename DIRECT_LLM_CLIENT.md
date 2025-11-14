# Direct LLM Client Implementation

## Overview

The system has been migrated from LiteLLM to direct provider SDKs for better reliability, control, and compatibility with Supabase-based configuration.

## Why Direct SDKs?

1. **No LiteLLM Dependency**: Removes the LiteLLM abstraction layer that was causing authentication issues
2. **Better Control**: Direct access to provider-specific features and configurations
3. **Supabase Integration**: Easier to pass provider/model/API key from Supabase per request
4. **More Reliable**: Native SDKs are more stable and better maintained
5. **Clearer Errors**: Provider-specific error messages are easier to debug

## Supported Providers

### OpenAI
- Uses: `openai` Python SDK
- Models: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`, etc.
- API Key: Passed directly to `OpenAI(api_key=...)`
- Base URL: `https://api.openai.com/v1` (default)

### Gemini (Google)
- Uses: `google-generativeai` Python SDK
- Models: `gemini-2.5-flash`, `gemini-2.5`, `gemini-1.5-pro`, `gemini-1.5-flash`
- API Key: Configured via `genai.configure(api_key=...)`

### Anthropic (Claude)
- Uses: `anthropic` Python SDK
- Models: `claude-3-5-sonnet-20241022`, `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- API Key: Passed directly to `Anthropic(api_key=...)`

### DeepSeek
- Uses: `openai` Python SDK (OpenAI-compatible API)
- Models: `deepseek-chat`, `deepseek-coder`, `deepseek-reasoner`
- API Key: Passed to `OpenAI(api_key=..., base_url="https://api.deepseek.com")`
- Base URL: `https://api.deepseek.com`

### Moonshot/Kimi
- Uses: `openai` Python SDK (OpenAI-compatible API)
- Models: `moonshot-v1-8k`, `moonshot-v1-32k`, `moonshot-v1-128k`, `kimi-k2-0711-preview`, `kimi-k2-moonshine`
- API Key: Passed to `OpenAI(api_key=..., base_url="https://api.moonshot.cn/v1")`
- Base URL: `https://api.moonshot.cn/v1`

## Usage

### From Tasks (Current Implementation)

```python
from llm_client_direct import create_llm_client

# Create client with provider/model/api_key from request
llm_client = create_llm_client(
    provider=research_data.get('provider', 'openai'),
    model=research_data.get('model', 'gpt-4'),
    api_key=research_data.get('api_key'),  # From Supabase/request
    temperature=0.7,
    max_tokens=2000
)

# Generate content
response = llm_client.generate(messages)
content = response.content
```

### From Supabase (Production)

```python
# In your API endpoint, fetch from Supabase:
user_config = supabase.table('user_llm_config').select('*').eq('user_id', user_id).execute()

# Use in request
llm_client = create_llm_client(
    provider=user_config['provider'],
    model=user_config['model'],
    api_key=user_config['api_key'],
    temperature=user_config.get('temperature', 0.7)
)
```

## API Compatibility

The direct client maintains the same interface as the old LiteLLM client:

- `create_llm_client(provider, model, api_key, **kwargs)` - Factory function
- `llm_client.generate(messages)` - Generate method
- Returns `LLMResponse` with `content`, `usage`, `cost`, etc.

No changes needed to:
- `article_structure_generator.py`
- `content_generator.py`
- `citation_generator.py`
- `tasks.py` (already updated)

## Migration Notes

1. **Old Client**: `llm_client.py` (LiteLLM-based) - kept for reference
2. **New Client**: `llm_client_direct.py` (Direct SDKs) - active
3. **Tasks Updated**: `tasks.py` now imports from `llm_client_direct`

## Benefits for Supabase Integration

1. **Per-Request Configuration**: Each request can use different provider/model/key
2. **User-Specific Settings**: Store LLM preferences per user in Supabase
3. **No Global State**: No environment variable conflicts
4. **Easy Extension**: Add new providers by adding SDK integration

## Adding New Providers

To add a new provider (e.g., Cohere, Mistral):

1. Add provider to `LLMProvider` enum
2. Add SDK import (with try/except for ImportError)
3. Implement `_generate_<provider>()` method
4. Add to `_init_provider_client()` method
5. Add cost estimation in `_estimate_cost_<provider>()`

## Testing

Test with a simple call:

```python
from llm_client_direct import create_llm_client

client = create_llm_client(
    provider='gemini',
    model='gemini-2.5-flash',
    api_key='YOUR_API_KEY'
)

response = client.generate([
    {'role': 'user', 'content': 'Say hello'}
])

print(response.content)
```

## Troubleshooting

### Import Errors
- Install missing SDKs: `pip install openai google-generativeai anthropic`

### Authentication Errors
- Verify API key is correct and active
- Check provider-specific API key format
- Ensure API key has proper permissions

### Message Format Issues
- Each provider has different message formats
- System messages handled differently per provider
- Check provider SDK documentation for format requirements

