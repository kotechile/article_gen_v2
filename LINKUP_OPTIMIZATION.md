# Linkup Search Optimization Guide

This document describes the optimizations implemented to reduce Linkup deep search costs by 70-90% while maintaining research quality.

## Overview

Linkup deep search costs approximately **10x more** than standard search. The optimization strategy implements:

1. **RAG-First Strategy**: Check RAG coverage before calling Linkup
2. **Explicit Depth Control**: Avoid accidental deep searches
3. **Intelligent Depth Selection**: Use deep search only when necessary
4. **Result Caching**: Cache search results to avoid duplicate API calls

## Key Features

### 1. RAG Coverage Assessment

Before making expensive Linkup API calls, the system assesses RAG coverage quality:

- **Minimum Sources**: Requires at least 3 unique sources (configurable)
- **Relevance Threshold**: Average relevance score ≥ 0.6 (configurable)
- **Keyword Coverage**: Ensures 50%+ keyword coverage in RAG results

If RAG coverage is sufficient, Linkup search is **skipped entirely** (unless depth='deep').

### 2. Intelligent Depth Selection

The system automatically determines optimal search depth:

- **Standard Search** (fast, cost-effective):
  - Used when RAG coverage is sufficient
  - Used for simple factual queries
  - Default for 'standard' depth requests

- **Deep Search** (slow, expensive):
  - Only used when:
    - User explicitly requests 'deep'
    - RAG coverage is insufficient AND query complexity requires it
    - Complex multi-part queries with insufficient RAG

- **Auto-Downgrade**: 'comprehensive' depth automatically downgrades to 'standard' when RAG is sufficient (configurable)

### 3. Result Caching

All Linkup search results are cached to avoid duplicate API calls:

- **Standard Search**: Cached for 6 hours (configurable)
- **Deep Search**: Cached for 24 hours (configurable)
- **Cache Key**: Based on query + depth + date_range + site + max_results
- **Query Normalization**: Queries are normalized (lowercase, trimmed) for consistent caching

### 4. Explicit Depth Parameter

The async Linkup client (`app/research_core/linkup_client.py`) now requires explicit depth:

- **Old Behavior**: `depth = 'deep' if max_results > 10 else 'standard'` (risky, could trigger accidental deep searches)
- **New Behavior**: `depth` parameter must be explicitly provided (defaults to 'standard')

## Configuration

All optimization settings can be configured via environment variables:

```bash
# RAG Coverage Thresholds
LINKUP_OPT_RAG_MIN_SOURCES=3          # Minimum number of sources required
LINKUP_OPT_RAG_MIN_RELEVANCE=0.6      # Minimum average relevance score

# Caching
LINKUP_OPT_CACHE_ENABLED=true         # Enable/disable caching
LINKUP_OPT_CACHE_TTL_STANDARD=21600   # 6 hours (standard search TTL)
LINKUP_OPT_CACHE_TTL_DEEP=86400       # 24 hours (deep search TTL)

# Auto-Downgrade
LINKUP_OPT_AUTO_DOWNGRADE=true        # Auto-downgrade comprehensive to standard when RAG sufficient
```

## Usage Examples

### Example 1: RAG Coverage Sufficient - Linkup Skipped

```python
# RAG returns 5 sources with avg relevance 0.75
# → Linkup search is SKIPPED
# → Cost savings: 100% (no API call)
```

### Example 2: RAG Insufficient - Standard Search Used

```python
# RAG returns 1 source with avg relevance 0.4
# Depth request: 'comprehensive'
# → RAG insufficient → Auto-downgrade 'comprehensive' to 'standard'
# → Standard Linkup search performed
# → Cost savings: ~90% vs deep search
```

### Example 3: RAG Insufficient + Complex Query - Deep Search Used

```python
# RAG returns 1 source
# Query: "Compare X vs Y and analyze pros/cons?"
# Depth request: 'comprehensive'
# → RAG insufficient + complex query → Deep search performed
# → Justified use of expensive deep search
```

### Example 4: Cached Result - No API Call

```python
# Same query searched 2 hours ago
# → Cache HIT
# → No API call, result returned from cache
# → Cost savings: 100% (cached), Performance: <1ms
```

## Expected Impact

- **Cost Reduction**: 70-90% fewer deep search calls
- **Performance**: 0.5-2s faster responses via caching
- **Quality**: Maintained research quality via RAG-first strategy
- **Reliability**: Explicit depth control prevents accidental expensive searches

## Monitoring

The system logs all optimization decisions:

```
📊 RAG Coverage Assessment:
  - Sources: 5 (min: 3)
  - Avg Relevance: 0.75 (min: 0.6)
  - Keyword Coverage: 0.85
  - Assessment: sufficient_5_sources
  - Sufficient: True

⏭️  Skipping Linkup search - RAG coverage is sufficient (5 sources, relevance: 0.75)
```

Or:

```
📊 RAG Coverage Assessment:
  - Sources: 1 (min: 3)
  - Avg Relevance: 0.4 (min: 0.6)
  - Assessment: insufficient_sources_1_needed_3

🎯 Selected Linkup search depth: 'standard' (requested: 'comprehensive')
```

## Architecture

### Files Modified

1. **`app/research_core/linkup_client.py`**
   - Added explicit `depth` parameter to `search()` and `batch_search()`
   - Removed implicit depth logic based on `max_results`

2. **`content_generator_v2/linkup_client.py`**
   - Added caching via `SearchCache` wrapper
   - Cache check before API call, cache store after success

3. **`content_generator_v2/tasks.py`**
   - Added `_assess_rag_coverage()` function
   - Added `_determine_optimal_search_depth()` function
   - Modified `_collect_evidence()` to implement RAG-first strategy

4. **`content_generator_v2/config.py`**
   - Added `LinkupOptimizationConfig` dataclass
   - Environment variable loading

5. **`content_generator_v2/utils/search_cache.py`** (NEW)
   - `SearchCache` class wrapping `CacheManager`
   - Query normalization and cache key generation

## Testing Recommendations

1. **RAG Coverage Assessment**
   - Test with varying source counts (0, 1, 3, 5+)
   - Test with varying relevance scores (0.3, 0.6, 0.9)
   - Verify keyword coverage calculation

2. **Depth Selection**
   - Test auto-downgrade from 'comprehensive' to 'standard'
   - Test complex query detection
   - Verify 'deep' requests are honored

3. **Caching**
   - Test cache hits with identical queries
   - Test cache misses with different queries
   - Verify TTL expiration

4. **Integration**
   - Monitor actual Linkup API call counts
   - Compare costs before/after implementation
   - Verify research quality maintained

## Troubleshooting

### Issue: Deep searches still being triggered unexpectedly

**Solution**: Check logs for depth selection decision. Verify `LINKUP_OPT_AUTO_DOWNGRADE=true` and RAG coverage thresholds.

### Issue: Cache not working

**Solution**: 
1. Verify Redis is running and accessible
2. Check `LINKUP_OPT_CACHE_ENABLED=true`
3. Check cache key generation (query normalization)

### Issue: RAG coverage always insufficient

**Solution**: 
1. Adjust thresholds: `LINKUP_OPT_RAG_MIN_SOURCES` and `LINKUP_OPT_RAG_MIN_RELEVANCE`
2. Improve RAG query quality
3. Check RAG endpoint health

## Related Documentation

- [Main README](../README.md)
- [API Usage Guide](../docs/API_USAGE.md)
- [Celery Task Documentation](../README_CELERY.md)


