"""
Unit tests for cache service

Tests the cache service functionality including
Redis operations, TTL management, and error handling.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.dataforseo.cache_service import CacheService
from src.models.trend_data import TrendData, TimelinePoint
from src.models.keyword_data import KeywordData, SearchVolumeTrend
from src.models.subtopic_data import SubtopicData

class TestCacheService:
    """Test cases for CacheService"""
    
    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client for testing"""
        client = AsyncMock()
        client.get.return_value = None
        client.set.return_value = True
        client.setex.return_value = True
        client.delete.return_value = True
        client.keys.return_value = []
        client.delete.return_value = 0
        return client
    
    @pytest.fixture
    def cache_service(self, mock_redis_client):
        """Create cache service with mocked Redis client"""
        service = CacheService()
        service.redis_client = mock_redis_client
        return service
    
    @pytest.mark.asyncio
    async def test_get_success(self, cache_service, mock_redis_client):
        """Test successful cache get operation"""
        # Arrange
        key = "test_key"
        cached_value = {"data": "test_value"}
        mock_redis_client.get.return_value = json.dumps(cached_value)
        
        # Act
        result = await cache_service.get(key)
        
        # Assert
        assert result == cached_value
        mock_redis_client.get.assert_called_once_with(key)
    
    @pytest.mark.asyncio
    async def test_get_not_found(self, cache_service, mock_redis_client):
        """Test cache get when key not found"""
        # Arrange
        key = "nonexistent_key"
        mock_redis_client.get.return_value = None
        
        # Act
        result = await cache_service.get(key)
        
        # Assert
        assert result is None
        mock_redis_client.get.assert_called_once_with(key)
    
    @pytest.mark.asyncio
    async def test_get_invalid_json(self, cache_service, mock_redis_client):
        """Test cache get with invalid JSON"""
        # Arrange
        key = "invalid_json_key"
        mock_redis_client.get.return_value = "invalid json"
        
        # Act
        result = await cache_service.get(key)
        
        # Assert
        assert result is None
        mock_redis_client.get.assert_called_once_with(key)
    
    @pytest.mark.asyncio
    async def test_set_success(self, cache_service, mock_redis_client):
        """Test successful cache set operation"""
        # Arrange
        key = "test_key"
        value = {"data": "test_value"}
        ttl = 3600
        
        # Act
        result = await cache_service.set(key, value, ttl)
        
        # Assert
        assert result is True
        mock_redis_client.setex.assert_called_once_with(key, ttl, json.dumps(value, default=str))
    
    @pytest.mark.asyncio
    async def test_set_no_ttl(self, cache_service, mock_redis_client):
        """Test cache set without TTL"""
        # Arrange
        key = "test_key"
        value = {"data": "test_value"}
        
        # Act
        result = await cache_service.set(key, value)
        
        # Assert
        assert result is True
        mock_redis_client.set.assert_called_once_with(key, json.dumps(value, default=str))
    
    @pytest.mark.asyncio
    async def test_set_redis_error(self, cache_service, mock_redis_client):
        """Test cache set with Redis error"""
        # Arrange
        key = "test_key"
        value = {"data": "test_value"}
        mock_redis_client.setex.side_effect = Exception("Redis Error")
        
        # Act
        result = await cache_service.set(key, value, 3600)
        
        # Assert
        assert result is False
    
    @pytest.mark.asyncio
    async def test_delete_success(self, cache_service, mock_redis_client):
        """Test successful cache delete operation"""
        # Arrange
        key = "test_key"
        mock_redis_client.delete.return_value = True
        
        # Act
        result = await cache_service.delete(key)
        
        # Assert
        assert result is True
        mock_redis_client.delete.assert_called_once_with(key)
    
    @pytest.mark.asyncio
    async def test_delete_redis_error(self, cache_service, mock_redis_client):
        """Test cache delete with Redis error"""
        # Arrange
        key = "test_key"
        mock_redis_client.delete.side_effect = Exception("Redis Error")
        
        # Act
        result = await cache_service.delete(key)
        
        # Assert
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_trend_data_success(self, cache_service, mock_redis_client):
        """Test successful trend data retrieval from cache"""
        # Arrange
        subtopics = ["topic 1", "topic 2"]
        location = "United States"
        time_range = "12m"
        
        cached_data = [
            {
                "subtopic": "topic 1",
                "location": "United States",
                "time_range": "12m",
                "average_interest": 50.0,
                "peak_interest": 75.0,
                "timeline_data": [
                    {"date": "2024-01-01", "value": 50},
                    {"date": "2024-01-02", "value": 60}
                ],
                "related_queries": ["query 1", "query 2"],
                "demographic_data": None
            }
        ]
        
        mock_redis_client.get.return_value = json.dumps(cached_data)
        
        # Act
        result = await cache_service.get_trend_data(subtopics, location, time_range)
        
        # Assert
        assert result is not None
        assert len(result) == 1
        assert result[0].subtopic == "topic 1"
        assert result[0].average_interest == 50.0
    
    @pytest.mark.asyncio
    async def test_set_trend_data_success(self, cache_service, mock_redis_client):
        """Test successful trend data caching"""
        # Arrange
        subtopics = ["topic 1"]
        location = "United States"
        time_range = "12m"
        
        trend_data = [
            TrendData(
                subtopic="topic 1",
                location="United States",
                time_range="12m",
                average_interest=50.0,
                peak_interest=75.0,
                timeline_data=[
                    TimelinePoint(date="2024-01-01", value=50),
                    TimelinePoint(date="2024-01-02", value=60)
                ],
                related_queries=["query 1", "query 2"],
                demographic_data=None
            )
        ]
        
        # Act
        result = await cache_service.set_trend_data(subtopics, location, time_range, trend_data)
        
        # Assert
        assert result is True
        mock_redis_client.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_keyword_data_success(self, cache_service, mock_redis_client):
        """Test successful keyword data retrieval from cache"""
        # Arrange
        seed_keywords = ["keyword 1", "keyword 2"]
        filters = {"max_difficulty": 70, "min_volume": 100}
        
        cached_data = [
            {
                "keyword": "keyword 1",
                "search_volume": 1000,
                "keyword_difficulty": 50,
                "cpc": 1.5,
                "competition_value": 60,
                "trend_percentage": 10.0,
                "intent_type": "COMMERCIAL",
                "related_keywords": ["related 1"],
                "search_volume_trend": [
                    {"month": "2024-01", "volume": 900},
                    {"month": "2024-02", "volume": 1000}
                ]
            }
        ]
        
        mock_redis_client.get.return_value = json.dumps(cached_data)
        
        # Act
        result = await cache_service.get_keyword_data(seed_keywords, filters)
        
        # Assert
        assert result is not None
        assert len(result) == 1
        assert result[0].keyword == "keyword 1"
        assert result[0].search_volume == 1000
    
    @pytest.mark.asyncio
    async def test_set_keyword_data_success(self, cache_service, mock_redis_client):
        """Test successful keyword data caching"""
        # Arrange
        seed_keywords = ["keyword 1"]
        filters = {"max_difficulty": 70, "min_volume": 100}
        
        keyword_data = [
            KeywordData(
                keyword="keyword 1",
                search_volume=1000,
                keyword_difficulty=50,
                cpc=1.5,
                competition_value=60,
                trend_percentage=10.0,
                intent_type="COMMERCIAL",
                related_keywords=["related 1"],
                search_volume_trend=[
                    SearchVolumeTrend(month="2024-01", volume=900),
                    SearchVolumeTrend(month="2024-02", volume=1000)
                ]
            )
        ]
        
        # Act
        result = await cache_service.set_keyword_data(seed_keywords, filters, keyword_data)
        
        # Assert
        assert result is True
        mock_redis_client.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_suggestions_success(self, cache_service, mock_redis_client):
        """Test successful suggestions retrieval from cache"""
        # Arrange
        base_subtopics = ["base topic"]
        location = "United States"
        
        cached_data = [
            {
                "topic": "suggested topic",
                "trending_status": "TRENDING",
                "growth_potential": 25.0,
                "search_volume": 1000,
                "related_queries": ["query 1", "query 2"],
                "competition_level": "LOW",
                "commercial_intent": 50.0
            }
        ]
        
        mock_redis_client.get.return_value = json.dumps(cached_data)
        
        # Act
        result = await cache_service.get_suggestions(base_subtopics, location)
        
        # Assert
        assert result is not None
        assert len(result) == 1
        assert result[0].topic == "suggested topic"
        assert result[0].trending_status == "TRENDING"
    
    @pytest.mark.asyncio
    async def test_set_suggestions_success(self, cache_service, mock_redis_client):
        """Test successful suggestions caching"""
        # Arrange
        base_subtopics = ["base topic"]
        location = "United States"
        
        suggestions = [
            SubtopicData(
                topic="suggested topic",
                trending_status="TRENDING",
                growth_potential=25.0,
                search_volume=1000,
                related_queries=["query 1", "query 2"],
                competition_level="LOW",
                commercial_intent=50.0
            )
        ]
        
        # Act
        result = await cache_service.set_suggestions(base_subtopics, location, suggestions)
        
        # Assert
        assert result is True
        mock_redis_client.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_invalidate_pattern_success(self, cache_service, mock_redis_client):
        """Test successful cache pattern invalidation"""
        # Arrange
        pattern = "trend_data:*"
        mock_redis_client.keys.return_value = ["dataforseo:trend_data:key1", "dataforseo:trend_data:key2"]
        mock_redis_client.delete.return_value = 2
        
        # Act
        result = await cache_service.invalidate_pattern(pattern)
        
        # Assert
        assert result == 2
        mock_redis_client.keys.assert_called_once_with(f"dataforseo:{pattern}")
        mock_redis_client.delete.assert_called_once_with("dataforseo:trend_data:key1", "dataforseo:trend_data:key2")
    
    @pytest.mark.asyncio
    async def test_invalidate_pattern_no_keys(self, cache_service, mock_redis_client):
        """Test cache pattern invalidation with no matching keys"""
        # Arrange
        pattern = "nonexistent:*"
        mock_redis_client.keys.return_value = []
        
        # Act
        result = await cache_service.invalidate_pattern(pattern)
        
        # Assert
        assert result == 0
        mock_redis_client.keys.assert_called_once_with(f"dataforseo:{pattern}")
        mock_redis_client.delete.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_cache_stats_success(self, cache_service, mock_redis_client):
        """Test successful cache statistics retrieval"""
        # Arrange
        mock_redis_client.info.return_value = {
            "used_memory_human": "1.2M",
            "connected_clients": 5,
            "total_commands_processed": 1000,
            "keyspace_hits": 800,
            "keyspace_misses": 200
        }
        
        # Act
        result = await cache_service.get_cache_stats()
        
        # Assert
        assert result["status"] == "active"
        assert result["used_memory"] == "1.2M"
        assert result["connected_clients"] == 5
        assert result["total_commands_processed"] == 1000
        assert result["keyspace_hits"] == 800
        assert result["keyspace_misses"] == 200
    
    @pytest.mark.asyncio
    async def test_get_cache_stats_redis_error(self, cache_service, mock_redis_client):
        """Test cache statistics retrieval with Redis error"""
        # Arrange
        mock_redis_client.info.side_effect = Exception("Redis Error")
        
        # Act
        result = await cache_service.get_cache_stats()
        
        # Assert
        assert result["status"] == "error"
        assert "Redis Error" in result["error"]
    
    @pytest.mark.asyncio
    async def test_generate_cache_key(self, cache_service):
        """Test cache key generation"""
        # Arrange
        prefix = "test_prefix"
        params = {"param1": "value1", "param2": "value2"}
        
        # Act
        key = cache_service._generate_cache_key(prefix, params)
        
        # Assert
        assert key.startswith("dataforseo:test_prefix:")
        assert "param1" in key
        assert "param2" in key
        assert "value1" in key
        assert "value2" in key
    
    @pytest.mark.asyncio
    async def test_generate_cache_key_long_params(self, cache_service):
        """Test cache key generation with long parameters"""
        # Arrange
        prefix = "test_prefix"
        params = {"long_param": "x" * 200}  # Very long parameter
        
        # Act
        key = cache_service._generate_cache_key(prefix, params)
        
        # Assert
        assert key.startswith("dataforseo:test_prefix:")
        assert len(key) < 200  # Should be hashed for long params
    
    @pytest.mark.asyncio
    async def test_redis_client_disabled(self, cache_service):
        """Test behavior when Redis client is disabled"""
        # Arrange
        cache_service.redis_client = None
        
        # Act & Assert
        assert await cache_service.get("test_key") is None
        assert await cache_service.set("test_key", "test_value") is False
        assert await cache_service.delete("test_key") is False
        assert await cache_service.get_cache_stats() == {"status": "disabled"}
