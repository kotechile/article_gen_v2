# Topic Analysis Architecture

## Overview

The TrendTap system uses a **hybrid approach** for topic analysis that combines:
1. **Pre-built database** for common topics (fast, reliable)
2. **LLM-powered analysis** for new topics (intelligent, scalable)
3. **Pattern matching** for smart categorization
4. **Caching** for performance optimization

## Architecture Components

### 1. Frontend Service (`topicAnalysisService.js`)

```javascript
// Hybrid service that tries multiple approaches
class TopicAnalysisService {
  // 1. Check pre-built database (fastest)
  // 2. Check cache (fast)
  // 3. Try pattern matching (smart)
  // 4. Use LLM API (intelligent)
  // 5. Fallback to generic analysis
}
```

**Features:**
- ✅ Fast lookup for common topics
- ✅ Smart keyword pattern matching
- ✅ LLM integration for new topics
- ✅ Caching for performance
- ✅ Graceful fallbacks

### 2. Backend API (`topic_analysis_routes.py`)

**Endpoints:**
- `POST /api/topic-analysis/analyze` - Analyze any topic
- `GET /api/topic-analysis/topics/database` - List all topics in DB
- `POST /api/topic-analysis/topics/add` - Add new topic manually
- `GET /api/topic-analysis/topics/search` - Search topics

**Features:**
- ✅ LLM integration (OpenAI GPT-4)
- ✅ Database storage
- ✅ Error handling
- ✅ Response caching

### 3. Database Models (`topic_analysis.py`)

**Tables:**
- `topic_analyses` - Main analysis storage
- `related_areas` - Normalized related areas
- `affiliate_programs` - Normalized affiliate programs
- `topic_search_logs` - Analytics and usage tracking

**Features:**
- ✅ Structured data storage
- ✅ Usage analytics
- ✅ Performance tracking
- ✅ User feedback system

### 4. Service Layer (`topic_analysis_service.py`)

**Core Functions:**
- `analyze_topic()` - Main analysis orchestrator
- `_get_existing_analysis()` - Database lookup
- `_find_pattern_match()` - Smart pattern matching
- `_analyze_with_llm()` - LLM integration
- `_save_analysis()` - Database persistence

## How New Topics Are Added

### Automatic Addition (LLM-Powered)

1. **User searches** for a new topic
2. **System checks** database → not found
3. **Pattern matching** → no match
4. **LLM analyzes** the topic using GPT-4
5. **Results saved** to database automatically
6. **Future searches** use cached results

### Manual Addition (Admin/Curator)

1. **Admin identifies** valuable topic
2. **Creates analysis** manually or via API
3. **Adds to database** with `add_manual_topic()`
4. **Immediately available** to all users

### Pattern-Based Addition

1. **System detects** keyword patterns
2. **Maps to existing** analysis categories
3. **Saves as new** topic variant
4. **Improves matching** for similar topics

## Topic Analysis Flow

```mermaid
graph TD
    A[User Searches Topic] --> B{In Database?}
    B -->|Yes| C[Return Cached Analysis]
    B -->|No| D{Pattern Match?}
    D -->|Yes| E[Return Pattern Analysis]
    D -->|No| F{LLM Available?}
    F -->|Yes| G[Generate with LLM]
    F -->|No| H[Generate Fallback]
    G --> I[Save to Database]
    H --> I
    E --> I
    I --> J[Return Analysis]
    C --> J
```

## LLM Integration

### OpenAI GPT-4 Integration

```python
# Backend API call to OpenAI
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an expert affiliate marketing analyst."},
        {"role": "user", "content": f"Analyze topic: {topic}"}
    ],
    max_tokens=2000,
    temperature=0.7
)
```

### Prompt Engineering

The system uses carefully crafted prompts to ensure:
- ✅ Relevant related areas
- ✅ Real affiliate programs
- ✅ Accurate commission rates
- ✅ Proper difficulty levels
- ✅ Structured JSON output

## Performance Optimization

### Caching Strategy

1. **Database Cache** - Pre-built common topics
2. **Memory Cache** - Recently analyzed topics
3. **API Cache** - LLM response caching
4. **CDN Cache** - Static analysis data

### Response Times

- **Database lookup**: ~10ms
- **Pattern matching**: ~50ms
- **LLM analysis**: ~2-5 seconds
- **Cached LLM**: ~100ms

## Cost Management

### LLM Usage Optimization

1. **Database First** - Check existing analyses
2. **Pattern Matching** - Avoid LLM for known categories
3. **Smart Caching** - Reuse LLM results
4. **Rate Limiting** - Prevent excessive API calls
5. **Batch Processing** - Analyze multiple topics together

### Cost Estimates

- **GPT-4**: ~$0.03 per analysis
- **Database**: ~$0.001 per lookup
- **Pattern Match**: ~$0.0001 per analysis

## Scaling Strategy

### Horizontal Scaling

1. **Multiple API instances** for LLM calls
2. **Database read replicas** for lookups
3. **Redis caching** for shared state
4. **CDN distribution** for static data

### Vertical Scaling

1. **Larger database** for more topics
2. **Faster LLM models** (GPT-4 Turbo)
3. **More memory** for caching
4. **Better hardware** for processing

## Monitoring & Analytics

### Key Metrics

- **Analysis success rate** (database vs LLM vs fallback)
- **Response times** by analysis type
- **LLM usage** and costs
- **Popular topics** and trends
- **User satisfaction** ratings

### Alerts

- **High LLM usage** (cost spikes)
- **Slow response times** (>5 seconds)
- **Analysis failures** (>5% error rate)
- **Database issues** (connection problems)

## Future Enhancements

### Planned Features

1. **Multi-LLM Support** - Anthropic, Google AI, etc.
2. **Real-time Updates** - Live topic trending
3. **User Feedback** - Rating system for analyses
4. **A/B Testing** - Compare analysis quality
5. **Custom Models** - Fine-tuned for affiliate marketing

### Advanced Analytics

1. **Topic Clustering** - Group similar topics
2. **Trend Prediction** - Forecast popular topics
3. **Competition Analysis** - Track affiliate program changes
4. **ROI Tracking** - Measure analysis effectiveness

## Configuration

### Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=2000

# Database Configuration
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://host:port

# Performance Tuning
CACHE_TTL=3600
MAX_CONCURRENT_LLM_CALLS=10
RATE_LIMIT_PER_MINUTE=100
```

### Feature Flags

```python
# Enable/disable features
ENABLE_LLM_ANALYSIS=true
ENABLE_PATTERN_MATCHING=true
ENABLE_CACHING=true
ENABLE_ANALYTICS=true
```

This hybrid architecture provides the best of both worlds: **fast performance for common topics** and **intelligent analysis for new topics**, while maintaining cost efficiency and scalability.


