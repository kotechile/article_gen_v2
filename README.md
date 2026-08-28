# Content Generator V2

A clean, reliable, and well-tested content generation system that transforms research briefs into comprehensive, well-structured articles with proper citations and SEO optimization.

## Features

- **Multi-stage Pipeline**: Asynchronous processing with Celery for scalable article generation
- **LLM Integration**: Support for multiple LLM providers (OpenAI, Anthropic, Google, etc.) via LiteLLM
- **RAG Integration**: Retrieval-Augmented Generation for knowledge base queries
- **Web Search**: Real-time information gathering via LinkUp service
- **Evidence Ranking**: Intelligent ranking of evidence based on relevance and credibility
- **Article Generation**: Structured article creation with proper citations
- **Refinement Pipeline**: Multiple specialized agents for fact-checking, SEO, and content optimization
- **REST API**: Clean REST endpoints with proper error handling and validation
- **Comprehensive Testing**: Unit tests, integration tests, and end-to-end testing
- **Monitoring**: Health checks, logging, and performance monitoring

## Architecture

```
content_generator_v2/
├── src/
│   ├── core/                 # Core business logic
│   │   ├── models/          # Data models and schemas
│   │   ├── services/        # Business services
│   │   └── pipeline/        # Article generation pipeline
│   ├── api/                 # REST API layer
│   │   ├── endpoints/       # API endpoints
│   │   ├── middleware/      # Authentication, rate limiting
│   │   └── schemas/         # Request/response schemas
│   ├── integrations/        # External service integrations
│   │   ├── llm/            # LLM client (LiteLLM)
│   │   ├── rag/            # RAG client
│   │   └── search/         # Web search client
│   ├── tasks/              # Celery tasks
│   └── utils/              # Utilities and helpers
├── tests/                  # Test suite
├── config/                # Configuration files
├── docs/                  # Documentation
└── scripts/               # Deployment and utility scripts
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```
   Set `RAG_API_URL=http://localhost:8081` for same-server RAG access, or use `https://rag.buildomain.com` if you need the public domain.

3. **Run Tests**:
   ```bash
   pytest tests/
   ```

4. **Start Services**:
   ```bash
   # Start Redis (required for Celery)
   redis-server
   
   # Start Celery worker
   python run_celery.py
   
   # Start Flask API (runs on port 5001 to avoid macOS conflicts)
   python run_app.py
   ```

## API Usage

### Create Research Task

```bash
curl -X POST http://localhost:5001/api/v1/research \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "brief": "The impact of artificial intelligence on modern healthcare",
    "keywords": "AI, healthcare, technology, medical diagnosis",
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "your-openai-key",
    "depth": "comprehensive",
    "tone": "journalistic",
    "target_word_count": 2000,
    "rag_enabled": true,
    "rag_collection": "medical_knowledge",
    "rag_endpoint": "http://localhost:8081/query_hybrid_enhanced",
    "rag_llm_provider": "openai"
  }'
```

### Check Task Status

```bash
curl -X GET http://localhost:5001/api/v1/research/{task_id} \
  -H "X-API-Key: your-api-key"
```

### Get Results

```bash
curl -X GET http://localhost:5001/api/v1/research/{task_id}/result \
  -H "X-API-Key: your-api-key"
```

### Image Generation & Application Routing

Image generation models (Flux, Stable Diffusion, Google Imagen) are managed dynamically via Supabase:
- **`used_for`**: Assigns models to applications (`article_image` for article featured/section visuals, `infographics` for infographics). Contains `llm_image_id` linking to `llm_providers_image.id`.
- **`llm_providers_image`**: Stores model identifiers (`model_name`, `provider`, `api_keys_id`).
- **`api_keys`**: Stores encrypted API keys linked via `api_keys_id`.

See [SUPABASE_API_KEYS.md](file:///Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/SUPABASE_API_KEYS.md) and [SUPABASE_PROVIDERS_LIST.md](file:///Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/SUPABASE_PROVIDERS_LIST.md) for full configuration details.

```bash
# Check configured models for applications
curl http://localhost:5001/api/v1/images/application-config

# Generate AI image using the application's configured model
curl -X POST http://localhost:5001/api/v1/images/generate-ai \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Modern architectural building with lush gardens",
    "application": "article_image",
    "aspectRatio": "16:9",
    "user_id": "user-uuid"
  }'
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Configuration

See `config/settings.py` for all configuration options.

## License

MIT License
