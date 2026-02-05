"""
Unit tests for trend analysis service

Tests the trend analysis service functionality including
API integration, data processing, and error handling.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.dataforseo.trend_service import TrendAnalysisService
from src.models.trend_data import TrendData, TimelinePoint, DemographicData, AgeGroupData, GenderData
from src.models.api_credentials import APICredentials

class TestTrendAnalysisService:
    """Test cases for TrendAnalysisService"""
    
    @pytest.fixture
    def mock_api_client(self):
        """Mock API client for testing"""
        client = AsyncMock()
        client.get_trend_data.return_value = [
            TrendData(
                subtopic="test topic",
                location="United States",
                time_range="12m",
                average_interest=50.0,
                peak_interest=75.0,
                timeline_data=[
                    TimelinePoint(date="2024-01-01", value=50),
                    TimelinePoint(date="2024-01-02", value=60),
                    TimelinePoint(date="2024-01-03", value=75)
                ],
                related_queries=["related query 1", "related query 2"],
                demographic_data=DemographicData(
                    age_groups=[
                        AgeGroupData(age_range="18-24", percentage=25),
                        AgeGroupData(age_range="25-34", percentage=35)
                    ],
                    gender_distribution=[
                        GenderData(gender="Male", percentage=55),
                        GenderData(gender="Female", percentage=45)
                    ],
                    interests=["health", "fitness"]
                )
            )
        ]
        return client
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager for testing"""
        cache = AsyncMock()
        cache.get_trend_data.return_value = None
        cache.set_trend_data.return_value = True
        return cache
    
    @pytest.fixture
    def mock_repository(self):
        """Mock repository for testing"""
        repo = AsyncMock()
        repo.get_trend_data.return_value = None
        repo.save_trend_data.return_value = True
        return repo
    
    @pytest.fixture
    def trend_service(self, mock_api_client, mock_cache_manager, mock_repository):
        """Create trend service with mocked dependencies"""
        service = TrendAnalysisService()
        service.api_client = mock_api_client
        service.cache_manager = mock_cache_manager
        service.repository = mock_repository
        return service
    
    @pytest.mark.asyncio
    async def test_get_trend_data_success(self, trend_service, mock_api_client, mock_cache_manager):
        """Test successful trend data retrieval"""
        # Arrange
        subtopics = ["test topic"]
        location = "United States"
        time_range = "12m"
        
        # Act
        result = await trend_service.get_trend_data(subtopics, location, time_range)
        
        # Assert
        assert len(result) == 1
        assert result[0].subtopic == "test topic"
        assert result[0].location == "United States"
        assert result[0].time_range == "12m"
        assert result[0].average_interest == 50.0
        assert result[0].peak_interest == 75.0
        assert len(result[0].timeline_data) == 3
        assert len(result[0].related_queries) == 2
        
        # Verify cache was checked
        mock_cache_manager.get_trend_data.assert_called_once_with(subtopics, location, time_range)
        
        # Verify API was called
        mock_api_client.get_trend_data.assert_called_once_with(subtopics, location, time_range)
        
        # Verify cache was set
        mock_cache_manager.set_trend_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_trend_data_from_cache(self, trend_service, mock_cache_manager):
        """Test trend data retrieval from cache"""
        # Arrange
        cached_data = [
            TrendData(
                subtopic="cached topic",
                location="United States",
                time_range="12m",
                average_interest=40.0,
                peak_interest=60.0,
                timeline_data=[],
                related_queries=[],
                demographic_data=None
            )
        ]
        mock_cache_manager.get_trend_data.return_value = cached_data
        
        # Act
        result = await trend_service.get_trend_data(["cached topic"], "United States", "12m")
        
        # Assert
        assert result == cached_data
        assert len(result) == 1
        assert result[0].subtopic == "cached topic"
        
        # Verify API was not called
        trend_service.api_client.get_trend_data.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_trend_data_api_error(self, trend_service, mock_api_client, mock_cache_manager):
        """Test trend data retrieval with API error"""
        # Arrange
        mock_api_client.get_trend_data.side_effect = Exception("API Error")
        
        # Act & Assert
        with pytest.raises(Exception, match="API Error"):
            await trend_service.get_trend_data(["test topic"], "United States", "12m")
    
    @pytest.mark.asyncio
    async def test_compare_trends_success(self, trend_service, mock_api_client):
        """Test successful trend comparison"""
        # Arrange
        subtopics = ["topic 1", "topic 2"]
        location = "United States"
        time_range = "12m"
        
        # Act
        result = await trend_service.compare_trends(subtopics, location, time_range)
        
        # Assert
        assert result is not None
        mock_api_client.get_trend_data.assert_called_once_with(subtopics, location, time_range)
    
    @pytest.mark.asyncio
    async def test_get_suggestions_success(self, trend_service, mock_api_client, mock_cache_manager):
        """Test successful suggestions retrieval"""
        # Arrange
        base_subtopics = ["base topic"]
        location = "United States"
        max_suggestions = 5
        
        # Act
        result = await trend_service.get_suggestions(base_subtopics, location, max_suggestions)
        
        # Assert
        assert result is not None
        mock_api_client.get_suggestions.assert_called_once_with(base_subtopics, location, max_suggestions)
    
    @pytest.mark.asyncio
    async def test_get_suggestions_from_cache(self, trend_service, mock_cache_manager):
        """Test suggestions retrieval from cache"""
        # Arrange
        cached_suggestions = [
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
        mock_cache_manager.get_suggestions.return_value = cached_suggestions
        
        # Act
        result = await trend_service.get_suggestions(["base topic"], "United States", 5)
        
        # Assert
        assert result == cached_suggestions
        assert len(result) == 1
        assert result[0]["topic"] == "suggested topic"
    
    @pytest.mark.asyncio
    async def test_validate_credentials_success(self, trend_service):
        """Test successful credentials validation"""
        # Arrange
        credentials = APICredentials(
            base_url="https://api.dataforseo.com",
            key_value="test_key",
            provider="dataforseo",
            is_active=True
        )
        
        # Act
        result = await trend_service.validate_credentials(credentials)
        
        # Assert
        assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_credentials_failure(self, trend_service):
        """Test credentials validation failure"""
        # Arrange
        credentials = APICredentials(
            base_url="https://api.dataforseo.com",
            key_value="invalid_key",
            provider="dataforseo",
            is_active=True
        )
        
        # Act
        result = await trend_service.validate_credentials(credentials)
        
        # Assert
        assert result is False
    
    @pytest.mark.asyncio
    async def test_process_trend_data_success(self, trend_service):
        """Test successful trend data processing"""
        # Arrange
        raw_data = {
            "keyword_info": {"keyword": "test topic"},
            "items": [
                {"date": "2024-01-01", "value": 50},
                {"date": "2024-01-02", "value": 60},
                {"date": "2024-01-03", "value": 75}
            ],
            "related_queries": [
                {"query": "related query 1"},
                {"query": "related query 2"}
            ]
        }
        
        # Act
        result = trend_service._process_trend_data(raw_data, "United States", "12m")
        
        # Assert
        assert result is not None
        assert result.subtopic == "test topic"
        assert result.location == "United States"
        assert result.time_range == "12m"
        assert result.average_interest == 61.67  # (50 + 60 + 75) / 3
        assert result.peak_interest == 75.0
        assert len(result.timeline_data) == 3
        assert len(result.related_queries) == 2
    
    @pytest.mark.asyncio
    async def test_process_trend_data_invalid(self, trend_service):
        """Test trend data processing with invalid data"""
        # Arrange
        raw_data = {
            "keyword_info": {},  # Missing keyword
            "items": [],
            "related_queries": []
        }
        
        # Act
        result = trend_service._process_trend_data(raw_data, "United States", "12m")
        
        # Assert
        assert result is None
    
    @pytest.mark.asyncio
    async def test_calculate_metrics_success(self, trend_service):
        """Test metrics calculation"""
        # Arrange
        timeline_data = [
            TimelinePoint(date="2024-01-01", value=50),
            TimelinePoint(date="2024-01-02", value=60),
            TimelinePoint(date="2024-01-03", value=75)
        ]
        
        # Act
        avg_interest, peak_interest = trend_service._calculate_metrics(timeline_data)
        
        # Assert
        assert avg_interest == 61.67  # (50 + 60 + 75) / 3
        assert peak_interest == 75.0
    
    @pytest.mark.asyncio
    async def test_calculate_metrics_empty(self, trend_service):
        """Test metrics calculation with empty data"""
        # Arrange
        timeline_data = []
        
        # Act
        avg_interest, peak_interest = trend_service._calculate_metrics(timeline_data)
        
        # Assert
        assert avg_interest == 0.0
        assert peak_interest == 0.0
    
    @pytest.mark.asyncio
    async def test_error_handling_graceful_degradation(self, trend_service, mock_api_client):
        """Test graceful degradation on API errors"""
        # Arrange
        mock_api_client.get_trend_data.side_effect = Exception("API Error")
        
        # Act & Assert
        with pytest.raises(Exception):
            await trend_service.get_trend_data(["test topic"], "United States", "12m")
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, trend_service, mock_api_client, mock_cache_manager):
        """Test handling of concurrent requests"""
        # Arrange
        subtopics = ["topic 1", "topic 2"]
        location = "United States"
        time_range = "12m"
        
        # Act
        tasks = [
            trend_service.get_trend_data(subtopics, location, time_range)
            for _ in range(5)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assert
        assert len(results) == 5
        # All should succeed or all should fail consistently
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        assert success_count in [0, 5]  # Either all succeed or all fail
