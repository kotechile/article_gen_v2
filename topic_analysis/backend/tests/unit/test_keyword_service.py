"""
Unit tests for keyword research service

Tests the keyword research service functionality including
API integration, data processing, and prioritization.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.dataforseo.keyword_service import KeywordResearchService
from src.models.keyword_data import KeywordData, SearchVolumeTrend
from src.models.api_credentials import APICredentials

class TestKeywordResearchService:
    """Test cases for KeywordResearchService"""
    
    @pytest.fixture
    def mock_api_client(self):
        """Mock API client for testing"""
        client = AsyncMock()
        client.get_keyword_data.return_value = [
            KeywordData(
                keyword="test keyword",
                search_volume=1000,
                keyword_difficulty=50,
                cpc=1.5,
                competition_value=60,
                trend_percentage=10.0,
                intent_type="COMMERCIAL",
                related_keywords=["related 1", "related 2"],
                search_volume_trend=[
                    SearchVolumeTrend(month="2024-01", volume=900),
                    SearchVolumeTrend(month="2024-02", volume=1000),
                    SearchVolumeTrend(month="2024-03", volume=1100)
                ]
            )
        ]
        return client
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager for testing"""
        cache = AsyncMock()
        cache.get_keyword_data.return_value = None
        cache.set_keyword_data.return_value = True
        return cache
    
    @pytest.fixture
    def mock_repository(self):
        """Mock repository for testing"""
        repo = AsyncMock()
        repo.get_keyword_data.return_value = []
        repo.save_keyword_data.return_value = True
        return repo
    
    @pytest.fixture
    def keyword_service(self, mock_api_client, mock_cache_manager, mock_repository):
        """Create keyword service with mocked dependencies"""
        service = KeywordResearchService()
        service.api_client = mock_api_client
        service.cache_manager = mock_cache_manager
        service.repository = mock_repository
        return service
    
    @pytest.mark.asyncio
    async def test_get_keywords_success(self, keyword_service, mock_api_client, mock_cache_manager):
        """Test successful keyword data retrieval"""
        # Arrange
        seed_keywords = ["test keyword"]
        max_difficulty = 70
        min_volume = 100
        intent_types = ["COMMERCIAL"]
        max_results = 100
        
        # Act
        result = await keyword_service.get_keywords(
            seed_keywords, max_difficulty, min_volume, intent_types, max_results
        )
        
        # Assert
        assert len(result) == 1
        assert result[0].keyword == "test keyword"
        assert result[0].search_volume == 1000
        assert result[0].keyword_difficulty == 50
        assert result[0].cpc == 1.5
        assert result[0].intent_type == "COMMERCIAL"
        
        # Verify cache was checked
        mock_cache_manager.get_keyword_data.assert_called_once()
        
        # Verify API was called
        mock_api_client.get_keyword_data.assert_called_once_with(
            seed_keywords, max_difficulty, min_volume, intent_types, max_results
        )
        
        # Verify cache was set
        mock_cache_manager.set_keyword_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_keywords_from_cache(self, keyword_service, mock_cache_manager):
        """Test keyword data retrieval from cache"""
        # Arrange
        cached_data = [
            KeywordData(
                keyword="cached keyword",
                search_volume=500,
                keyword_difficulty=30,
                cpc=0.8,
                competition_value=40,
                trend_percentage=5.0,
                intent_type="INFORMATIONAL",
                related_keywords=[],
                search_volume_trend=[]
            )
        ]
        mock_cache_manager.get_keyword_data.return_value = cached_data
        
        # Act
        result = await keyword_service.get_keywords(
            ["cached keyword"], 70, 100, ["INFORMATIONAL"], 100
        )
        
        # Assert
        assert result == cached_data
        assert len(result) == 1
        assert result[0].keyword == "cached keyword"
        
        # Verify API was not called
        keyword_service.api_client.get_keyword_data.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_keywords_api_error(self, keyword_service, mock_api_client):
        """Test keyword data retrieval with API error"""
        # Arrange
        mock_api_client.get_keyword_data.side_effect = Exception("API Error")
        
        # Act & Assert
        with pytest.raises(Exception, match="API Error"):
            await keyword_service.get_keywords(
                ["test keyword"], 70, 100, ["COMMERCIAL"], 100
            )
    
    @pytest.mark.asyncio
    async def test_prioritize_keywords_success(self, keyword_service):
        """Test successful keyword prioritization"""
        # Arrange
        keywords = [
            KeywordData(
                keyword="high priority",
                search_volume=5000,
                keyword_difficulty=30,
                cpc=2.0,
                competition_value=40,
                trend_percentage=20.0,
                intent_type="COMMERCIAL",
                related_keywords=[],
                search_volume_trend=[]
            ),
            KeywordData(
                keyword="low priority",
                search_volume=100,
                keyword_difficulty=80,
                cpc=0.5,
                competition_value=90,
                trend_percentage=-10.0,
                intent_type="INFORMATIONAL",
                related_keywords=[],
                search_volume_trend=[]
            )
        ]
        
        priority_factors = {
            "cpcWeight": 0.3,
            "volumeWeight": 0.4,
            "trendWeight": 0.3
        }
        
        # Act
        result = await keyword_service.prioritize_keywords(keywords, priority_factors)
        
        # Assert
        assert len(result) == 2
        assert result[0].keyword == "high priority"  # Should be first
        assert result[0].priority_score > result[1].priority_score
    
    @pytest.mark.asyncio
    async def test_prioritize_keywords_empty(self, keyword_service):
        """Test keyword prioritization with empty list"""
        # Arrange
        keywords = []
        priority_factors = {
            "cpcWeight": 0.3,
            "volumeWeight": 0.4,
            "trendWeight": 0.3
        }
        
        # Act
        result = await keyword_service.prioritize_keywords(keywords, priority_factors)
        
        # Assert
        assert result == []
    
    @pytest.mark.asyncio
    async def test_calculate_priority_score_success(self, keyword_service):
        """Test priority score calculation"""
        # Arrange
        keyword = KeywordData(
            keyword="test keyword",
            search_volume=1000,
            keyword_difficulty=50,
            cpc=1.0,
            competition_value=60,
            trend_percentage=10.0,
            intent_type="COMMERCIAL",
            related_keywords=[],
            search_volume_trend=[]
        )
        
        priority_factors = {
            "cpcWeight": 0.3,
            "volumeWeight": 0.4,
            "trendWeight": 0.3
        }
        
        # Act
        score = keyword_service._calculate_priority_score(keyword, priority_factors)
        
        # Assert
        assert isinstance(score, float)
        assert 0 <= score <= 100
    
    @pytest.mark.asyncio
    async def test_filter_keywords_by_intent(self, keyword_service):
        """Test keyword filtering by intent type"""
        # Arrange
        keywords = [
            KeywordData(
                keyword="commercial keyword",
                search_volume=1000,
                keyword_difficulty=50,
                cpc=1.0,
                competition_value=60,
                trend_percentage=10.0,
                intent_type="COMMERCIAL",
                related_keywords=[],
                search_volume_trend=[]
            ),
            KeywordData(
                keyword="informational keyword",
                search_volume=500,
                keyword_difficulty=30,
                cpc=0.5,
                competition_value=40,
                trend_percentage=5.0,
                intent_type="INFORMATIONAL",
                related_keywords=[],
                search_volume_trend=[]
            )
        ]
        
        intent_types = ["COMMERCIAL"]
        
        # Act
        result = keyword_service._filter_keywords_by_intent(keywords, intent_types)
        
        # Assert
        assert len(result) == 1
        assert result[0].keyword == "commercial keyword"
        assert result[0].intent_type == "COMMERCIAL"
    
    @pytest.mark.asyncio
    async def test_filter_keywords_by_difficulty(self, keyword_service):
        """Test keyword filtering by difficulty"""
        # Arrange
        keywords = [
            KeywordData(
                keyword="easy keyword",
                search_volume=1000,
                keyword_difficulty=30,
                cpc=1.0,
                competition_value=60,
                trend_percentage=10.0,
                intent_type="COMMERCIAL",
                related_keywords=[],
                search_volume_trend=[]
            ),
            KeywordData(
                keyword="hard keyword",
                search_volume=500,
                keyword_difficulty=80,
                cpc=0.5,
                competition_value=40,
                trend_percentage=5.0,
                intent_type="COMMERCIAL",
                related_keywords=[],
                search_volume_trend=[]
            )
        ]
        
        max_difficulty = 50
        
        # Act
        result = keyword_service._filter_keywords_by_difficulty(keywords, max_difficulty)
        
        # Assert
        assert len(result) == 1
        assert result[0].keyword == "easy keyword"
        assert result[0].keyword_difficulty <= max_difficulty
    
    @pytest.mark.asyncio
    async def test_filter_keywords_by_volume(self, keyword_service):
        """Test keyword filtering by search volume"""
        # Arrange
        keywords = [
            KeywordData(
                keyword="high volume keyword",
                search_volume=5000,
                keyword_difficulty=50,
                cpc=1.0,
                competition_value=60,
                trend_percentage=10.0,
                intent_type="COMMERCIAL",
                related_keywords=[],
                search_volume_trend=[]
            ),
            KeywordData(
                keyword="low volume keyword",
                search_volume=50,
                keyword_difficulty=30,
                cpc=0.5,
                competition_value=40,
                trend_percentage=5.0,
                intent_type="COMMERCIAL",
                related_keywords=[],
                search_volume_trend=[]
            )
        ]
        
        min_volume = 100
        
        # Act
        result = keyword_service._filter_keywords_by_volume(keywords, min_volume)
        
        # Assert
        assert len(result) == 1
        assert result[0].keyword == "high volume keyword"
        assert result[0].search_volume >= min_volume
    
    @pytest.mark.asyncio
    async def test_process_keyword_data_success(self, keyword_service):
        """Test successful keyword data processing"""
        # Arrange
        raw_data = {
            "keyword": "test keyword",
            "search_volume": 1000,
            "keyword_difficulty": 50,
            "cpc": 1.5,
            "competition_value": 60,
            "trend_percentage": 10.0,
            "related_keywords": ["related 1", "related 2"],
            "search_volume_trend": [
                {"month": "2024-01", "volume": 900},
                {"month": "2024-02", "volume": 1000},
                {"month": "2024-03", "volume": 1100}
            ]
        }
        
        intent_types = ["COMMERCIAL"]
        
        # Act
        result = keyword_service._process_keyword_data(raw_data, intent_types)
        
        # Assert
        assert result is not None
        assert result.keyword == "test keyword"
        assert result.search_volume == 1000
        assert result.keyword_difficulty == 50
        assert result.cpc == 1.5
        assert result.intent_type == "COMMERCIAL"
        assert len(result.related_keywords) == 2
        assert len(result.search_volume_trend) == 3
    
    @pytest.mark.asyncio
    async def test_process_keyword_data_invalid(self, keyword_service):
        """Test keyword data processing with invalid data"""
        # Arrange
        raw_data = {
            "keyword": "",  # Empty keyword
            "search_volume": 1000,
            "keyword_difficulty": 50,
            "cpc": 1.5,
            "competition_value": 60,
            "trend_percentage": 10.0,
            "related_keywords": [],
            "search_volume_trend": []
        }
        
        intent_types = ["COMMERCIAL"]
        
        # Act
        result = keyword_service._process_keyword_data(raw_data, intent_types)
        
        # Assert
        assert result is None
    
    @pytest.mark.asyncio
    async def test_determine_intent_type_success(self, keyword_service):
        """Test intent type determination"""
        # Test cases
        test_cases = [
            ("buy shoes", "TRANSACTIONAL"),
            ("best running shoes", "COMMERCIAL"),
            ("what is running", "INFORMATIONAL"),
            ("running techniques", "INFORMATIONAL"),
            ("order online", "TRANSACTIONAL"),
            ("compare prices", "COMMERCIAL")
        ]
        
        for keyword, expected_intent in test_cases:
            # Act
            result = keyword_service._determine_intent_type(keyword)
            
            # Assert
            assert result == expected_intent
    
    @pytest.mark.asyncio
    async def test_error_handling_graceful_degradation(self, keyword_service, mock_api_client):
        """Test graceful degradation on API errors"""
        # Arrange
        mock_api_client.get_keyword_data.side_effect = Exception("API Error")
        
        # Act & Assert
        with pytest.raises(Exception):
            await keyword_service.get_keywords(
                ["test keyword"], 70, 100, ["COMMERCIAL"], 100
            )
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, keyword_service, mock_api_client, mock_cache_manager):
        """Test handling of concurrent requests"""
        # Arrange
        seed_keywords = ["keyword 1", "keyword 2"]
        max_difficulty = 70
        min_volume = 100
        intent_types = ["COMMERCIAL"]
        max_results = 100
        
        # Act
        tasks = [
            keyword_service.get_keywords(seed_keywords, max_difficulty, min_volume, intent_types, max_results)
            for _ in range(5)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assert
        assert len(results) == 5
        # All should succeed or all should fail consistently
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        assert success_count in [0, 5]  # Either all succeed or all fail
