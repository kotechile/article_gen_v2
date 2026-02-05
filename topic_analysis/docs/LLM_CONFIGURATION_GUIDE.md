# LLM Configuration Guide

## Overview

The TrendTap system supports multiple LLM providers with a flexible configuration system. You can manage LLM providers through:

1. **Admin Interface** - Web-based configuration panel
2. **Supabase Database** - Direct database management
3. **Environment Variables** - API keys and settings
4. **API Endpoints** - Programmatic configuration

## Supported LLM Providers

### 1. OpenAI (Latest Models)
- **GPT-5 Mini** (Latest) - $0.0015/1K tokens - Ultra-fast, cost-effective
- **GPT-4 Turbo** - $0.01/1K tokens - High performance, large context
- **GPT-4** - $0.03/1K tokens - Excellent quality, proven reliability
- **GPT-3.5 Turbo** - $0.002/1K tokens - Fast and cost-effective
- **API**: OpenAI API
- **Quality**: Excellent for affiliate marketing analysis

### 2. Anthropic Claude (Latest Models)
- **Claude 3.5 Sonnet** (Latest) - $0.003/1K tokens - Best reasoning, code generation
- **Claude 3 Sonnet** - $0.015/1K tokens - Great for structured analysis
- **Claude 3 Haiku** - $0.00025/1K tokens - Ultra-fast, cost-effective
- **Claude 3 Opus** - $0.015/1K tokens - Highest quality, complex tasks
- **API**: Anthropic API
- **Quality**: Excellent for structured analysis and reasoning

### 3. Google Gemini (Latest Models)
- **Gemini 2.5 Flash** (Latest) - $0.0005/1K tokens - Ultra-fast, multimodal
- **Gemini 2.5 Flash Lite** - $0.0001/1K tokens - Lightweight, very fast
- **Gemini Pro** - $0.001/1K tokens - Good for general analysis
- **Gemini Pro Vision** - $0.001/1K tokens - Multimodal capabilities
- **API**: Google AI Studio API
- **Quality**: Good for general analysis, excellent for multimodal

### 4. Local Models (Ollama)
- **Llama 3.1** (Latest) - Free - Meta's latest open model
- **Llama 3** - Free - Meta's flagship open model
- **Mistral** - Free - High-quality European model
- **CodeLlama** - Free - Specialized for code generation
- **Phi-3** - Free - Microsoft's efficient model
- **API**: Local Ollama server
- **Cost**: Free (hardware costs only)
- **Quality**: Varies by model, good for privacy

### 5. Custom APIs
- **Models**: Any compatible API
- **API**: Custom endpoints
- **Cost**: Variable
- **Quality**: Depends on provider

## Setup Methods

### Method 1: Admin Interface (Recommended)

1. **Access Admin Panel**
   ```
   http://localhost:3000/admin/llm
   ```

2. **Add New Provider**
   - Click "Add Provider"
   - Fill in provider details
   - Set API key environment variable
   - Test the provider
   - Set as default if desired

3. **Configure Global Settings**
   - Set rate limits
   - Configure cost limits
   - Enable/disable features

### Method 2: Supabase Database

1. **Run Migration Script**
   ```sql
   -- Run the migration script in Supabase SQL editor
   \i backend/migrations/llm_config_setup.sql
   ```

2. **Add Provider via SQL**
   ```sql
   INSERT INTO llm_providers (
       name, provider_type, model_name, api_key_env_var,
       max_tokens, temperature, cost_per_1k_tokens,
       is_default, priority, is_active
   ) VALUES (
       'My Custom Provider', 'openai', 'gpt-4',
       'MY_OPENAI_KEY', 2000, 0.7, 0.03,
       true, 100, true
   );
   ```

3. **Update Configuration**
   ```sql
   UPDATE llm_configurations 
   SET daily_cost_limit = 100.0,
       monthly_cost_limit = 2000.0
   WHERE id = (SELECT id FROM llm_configurations LIMIT 1);
   ```

### Method 3: Environment Variables

1. **Set API Keys**
   ```bash
   # OpenAI
   export OPENAI_API_KEY="your-openai-key"
   
   # Anthropic
   export ANTHROPIC_API_KEY="your-anthropic-key"
   
   # Google
   export GOOGLE_API_KEY="your-google-key"
   ```

2. **Configure Provider**
   ```bash
   # Set default provider
   export DEFAULT_LLM_PROVIDER="openai"
   
   # Set cost limits
   export DAILY_COST_LIMIT="50.0"
   export MONTHLY_COST_LIMIT="1000.0"
   ```

### Method 4: API Endpoints

1. **Create Provider**
   ```bash
   curl -X POST http://localhost:8000/api/admin/llm/providers \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-admin-token" \
     -d '{
       "name": "OpenAI GPT-4",
       "provider_type": "openai",
       "model_name": "gpt-4",
       "api_key_env_var": "OPENAI_API_KEY",
       "max_tokens": 2000,
       "temperature": 0.7,
       "cost_per_1k_tokens": 0.03,
       "priority": 100
     }'
   ```

2. **Set Default Provider**
   ```bash
   curl -X POST http://localhost:8000/api/admin/llm/providers/{provider_id}/set-default \
     -H "Authorization: Bearer your-admin-token"
   ```

3. **Test Provider**
   ```bash
   curl -X POST http://localhost:8000/api/admin/llm/providers/{provider_id}/test \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-admin-token" \
     -d '{"test_topic": "best wireless headphones"}'
   ```

## Configuration Options

### Provider Configuration

| Setting | Description | Example |
|---------|-------------|---------|
| `name` | Display name | "OpenAI GPT-4" |
| `provider_type` | Provider type | "openai", "anthropic", "google", "local", "custom" |
| `model_name` | Model identifier | "gpt-4", "claude-3-sonnet" |
| `api_key_env_var` | Environment variable for API key | "OPENAI_API_KEY" |
| `base_url` | Custom API endpoint | "http://localhost:11434" |
| `max_tokens` | Maximum tokens per request | 2000 |
| `temperature` | Response randomness (0-2) | 0.7 |
| `cost_per_1k_tokens` | Cost per 1000 tokens | 0.03 |
| `priority` | Selection priority | 100 (higher = preferred) |

### Global Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `enable_llm_analysis` | Enable LLM-powered analysis | true |
| `enable_auto_fallback` | Fallback to other providers on failure | true |
| `enable_cost_tracking` | Track usage costs | true |
| `global_rate_limit_per_minute` | Global rate limit | 100 |
| `user_rate_limit_per_minute` | Per-user rate limit | 10 |
| `daily_cost_limit` | Daily spending limit | $50.00 |
| `monthly_cost_limit` | Monthly spending limit | $1000.00 |

## Usage Examples

### 1. Switch Between Providers

```python
# Use specific provider
analysis = await llm_service.analyze_topic_with_llm(
    topic="best gaming laptops",
    provider_id="openai-gpt-4-id"
)

# Use default provider
analysis = await llm_service.analyze_topic_with_llm(
    topic="best gaming laptops"
)
```

### 2. Monitor Usage

```python
# Get usage statistics
stats = llm_service.get_usage_stats(days=30)
print(f"Total requests: {stats['total_requests']}")
print(f"Total cost: ${stats['total_cost']:.2f}")
```

### 3. Test Provider

```python
# Test a provider
result = await llm_service.test_provider(
    provider_id="openai-gpt-4-id",
    test_topic="best wireless headphones"
)
print(f"Success: {result['success']}")
print(f"Response time: {result['response_time_ms']}ms")
```

## Cost Management

### 1. Set Cost Limits

```sql
-- Set daily limit
UPDATE llm_configurations 
SET daily_cost_limit = 25.0 
WHERE id = (SELECT id FROM llm_configurations LIMIT 1);

-- Set monthly limit
UPDATE llm_configurations 
SET monthly_cost_limit = 500.0 
WHERE id = (SELECT id FROM llm_configurations LIMIT 1);
```

### 2. Monitor Spending

```sql
-- Check daily spending
SELECT 
    DATE(request_timestamp) as date,
    SUM(cost) as daily_cost
FROM llm_usage_logs 
WHERE request_timestamp >= CURRENT_DATE
GROUP BY DATE(request_timestamp)
ORDER BY date DESC;

-- Check provider costs
SELECT 
    p.name,
    SUM(l.cost) as total_cost,
    COUNT(l.id) as total_requests
FROM llm_providers p
LEFT JOIN llm_usage_logs l ON p.id = l.provider_id
WHERE l.request_timestamp >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY p.id, p.name
ORDER BY total_cost DESC;
```

### 3. Optimize Costs

- Use cheaper models for simple tasks
- Implement caching for repeated queries
- Set appropriate rate limits
- Monitor and adjust cost limits

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   Error: API key not found for OpenAI GPT-4
   ```
   **Solution**: Set the environment variable specified in `api_key_env_var`

2. **Rate Limit Exceeded**
   ```
   Error: Rate limit exceeded
   ```
   **Solution**: Increase rate limits or wait before retrying

3. **Cost Limit Exceeded**
   ```
   Error: Daily cost limit exceeded
   ```
   **Solution**: Increase cost limits or wait until next day

4. **Provider Test Failed**
   ```
   Error: Provider test failed: Connection timeout
   ```
   **Solution**: Check network connectivity and API endpoint

### Debug Mode

Enable debug logging to see detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

```bash
# Check provider status
curl http://localhost:8000/api/admin/llm/providers

# Check configuration
curl http://localhost:8000/api/admin/llm/config

# Check usage stats
curl http://localhost:8000/api/admin/llm/usage-stats
```

## Best Practices

### 1. Provider Selection
- Use GPT-4 for highest quality analysis
- Use GPT-3.5 for cost-effective analysis
- Use local models for privacy-sensitive topics
- Use Claude for structured data extraction

### 2. Cost Optimization
- Set appropriate cost limits
- Use caching for repeated queries
- Monitor usage regularly
- Switch providers based on cost/quality needs

### 3. Performance
- Set appropriate rate limits
- Use connection pooling
- Implement retry logic
- Monitor response times

### 4. Security
- Store API keys securely
- Use environment variables
- Implement proper authentication
- Monitor for unusual usage patterns

## Migration Guide

### From Hardcoded to Database

1. **Export current configuration**
2. **Run migration script**
3. **Import configuration**
4. **Test all providers**
5. **Update application code**

### Adding New Provider

1. **Add provider to database**
2. **Set up API key**
3. **Test provider**
4. **Update application code if needed**
5. **Set as default if desired**

This configuration system provides maximum flexibility while maintaining ease of use and cost control! 🚀
