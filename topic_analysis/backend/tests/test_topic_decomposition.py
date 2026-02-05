"""
Tests for Topic Decomposition functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session
from src.services.topic_decomposition_service import TopicDecompositionService
from src.models.topic_decomposition import TopicDecomposition


class TestTopicDecompositionService:
    """Test cases for TopicDecompositionService"""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database session"""
        return Mock(spec=Session)
    
    @pytest.fixture
    def service(self, mock_db):
        """Create service instance with mocked database"""
        return TopicDecompositionService(mock_db)
    
    @pytest.mark.asyncio
    async def test_decompose_topic_success(self, service, mock_db):
        """Test successful topic decomposition"""
        # Mock LLM response
        mock_llm_response = [
            {
                "name": "Electric cars in California",
                "description": "EV market trends and opportunities in California",
                "relevance_score": 0.95,
                "category": "automotive"
            },
            {
                "name": "Car dealers",
                "description": "Automotive dealership opportunities and trends",
                "relevance_score": 0.88,
                "category": "automotive"
            }
        ]
        
        # Mock database operations
        mock_decomposition = Mock()
        mock_decomposition.id = "test_id"
        mock_decomposition.search_query = "Cars for the east coast"
        mock_decomposition.subtopics = mock_llm_response
        mock_decomposition.created_at = "2024-12-19T10:00:00Z"
        
        mock_db.add.return_value = None
        mock_db.commit.return_value = None
        mock_db.refresh.return_value = None
        mock_db.refresh.side_effect = lambda obj: setattr(obj, 'id', 'test_id')
        
        # Mock cache operations
        with patch.object(service, '_get_cached_result', return_value=None), \
             patch.object(service, '_set_cached_result', return_value=None), \
             patch.object(service, '_call_llm', return_value=mock_llm_response):
            
            result = await service.decompose_topic(
                search_query="Cars for the east coast",
                user_id="user_123",
                max_subtopics=10
            )
        
        # Assertions
        assert result["search_query"] == "Cars for the east coast"
        assert len(result["subtopics"]) == 2
        assert result["subtopics"][0]["name"] == "Electric cars in California"
        assert result["subtopics"][0]["relevance_score"] == 0.95
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_decompose_topic_with_cache(self, service, mock_db):
        """Test topic decomposition with cache hit"""
        cached_result = {
            "id": "cached_id",
            "search_query": "Cars for the east coast",
            "subtopics": [{"name": "Cached topic", "relevance_score": 0.8}],
            "created_at": "2024-12-19T10:00:00Z"
        }
        
        with patch.object(service, '_get_cached_result', return_value=cached_result):
            result = await service.decompose_topic(
                search_query="Cars for the east coast",
                user_id="user_123",
                max_subtopics=10
            )
        
        assert result == cached_result
        mock_db.add.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_decompose_topic_llm_failure_fallback(self, service, mock_db):
        """Test topic decomposition with LLM failure and fallback"""
        # Mock database operations
        mock_decomposition = Mock()
        mock_decomposition.id = "test_id"
        mock_decomposition.search_query = "Cars for the east coast"
        mock_decomposition.subtopics = []
        mock_decomposition.created_at = "2024-12-19T10:00:00Z"
        
        mock_db.add.return_value = None
        mock_db.commit.return_value = None
        mock_db.refresh.return_value = None
        mock_db.refresh.side_effect = lambda obj: setattr(obj, 'id', 'test_id')
        
        # Mock LLM failure and fallback
        with patch.object(service, '_get_cached_result', return_value=None), \
             patch.object(service, '_set_cached_result', return_value=None), \
             patch.object(service, '_call_llm', side_effect=Exception("LLM Error")), \
             patch.object(service, '_get_fallback_subtopics', return_value=[
                 {"name": "Fallback topic", "relevance_score": 0.8, "category": "general"}
             ]):
            
            result = await service.decompose_topic(
                search_query="Cars for the east coast",
                user_id="user_123",
                max_subtopics=10
            )
        
        assert result["search_query"] == "Cars for the east coast"
        assert len(result["subtopics"]) == 1
        assert result["subtopics"][0]["name"] == "Fallback topic"
    
    def test_validate_and_score_subtopics(self, service):
        """Test subtopic validation and scoring"""
        subtopics = [
            {
                "name": "Valid topic",
                "description": "A valid topic description",
                "relevance_score": 0.8,
                "category": "automotive"
            },
            {
                "name": "Invalid topic",
                "description": "",
                "relevance_score": 1.5,  # Invalid score
                "category": "automotive"
            }
        ]
        
        result = service._validate_and_score_subtopics(subtopics, "Cars for the east coast")
        
        assert len(result) == 1
        assert result[0]["name"] == "Valid topic"
        assert result[0]["relevance_score"] >= 0.8  # Should be enhanced
    
    def test_is_valid_subtopic(self, service):
        """Test subtopic validation"""
        valid_subtopic = {
            "name": "Valid topic",
            "description": "A valid description",
            "relevance_score": 0.8,
            "category": "automotive"
        }
        
        invalid_subtopic = {
            "name": "",
            "description": "Missing name",
            "relevance_score": 0.8,
            "category": "automotive"
        }
        
        assert service._is_valid_subtopic(valid_subtopic) == True
        assert service._is_valid_subtopic(invalid_subtopic) == False
    
    def test_calculate_enhanced_relevance_score(self, service):
        """Test enhanced relevance score calculation"""
        subtopic = {
            "name": "Electric cars in California",
            "relevance_score": 0.8
        }
        
        score = service._calculate_enhanced_relevance_score(subtopic, "Cars for the east coast")
        
        assert score >= 0.8
        assert score <= 1.0
    
    @pytest.mark.asyncio
    async def test_get_decomposition_success(self, service, mock_db):
        """Test getting topic decomposition by ID"""
        mock_decomposition = Mock()
        mock_decomposition.id = "test_id"
        mock_decomposition.search_query = "Cars for the east coast"
        mock_decomposition.subtopics = [{"name": "Test topic"}]
        mock_decomposition.created_at = "2024-12-19T10:00:00Z"
        mock_decomposition.updated_at = "2024-12-19T10:00:00Z"
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_decomposition
        
        result = await service.get_decomposition("test_id")
        
        assert result["id"] == "test_id"
        assert result["search_query"] == "Cars for the east coast"
        mock_db.query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_decomposition_not_found(self, service, mock_db):
        """Test getting non-existent topic decomposition"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        result = await service.get_decomposition("non_existent_id")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_decomposition_success(self, service, mock_db):
        """Test successful topic decomposition deletion"""
        mock_decomposition = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_decomposition
        mock_db.delete.return_value = None
        mock_db.commit.return_value = None
        
        result = await service.delete_decomposition("test_id")
        
        assert result == True
        mock_db.delete.assert_called_once_with(mock_decomposition)
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_decomposition_not_found(self, service, mock_db):
        """Test deleting non-existent topic decomposition"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        result = await service.delete_decomposition("non_existent_id")
        
        assert result == False
        mock_db.delete.assert_not_called()


class TestTopicDecompositionModel:
    """Test cases for TopicDecomposition model"""
    
    def test_topic_decomposition_creation(self):
        """Test creating a TopicDecomposition instance"""
        decomposition = TopicDecomposition(
            user_id="user_123",
            search_query="Cars for the east coast",
            subtopics=[{"name": "Test topic", "relevance_score": 0.8}]
        )
        
        assert decomposition.user_id == "user_123"
        assert decomposition.search_query == "Cars for the east coast"
        assert len(decomposition.subtopics) == 1
    
    def test_topic_decomposition_to_dict(self):
        """Test converting TopicDecomposition to dictionary"""
        decomposition = TopicDecomposition(
            id="test_id",
            user_id="user_123",
            search_query="Cars for the east coast",
            subtopics=[{"name": "Test topic", "relevance_score": 0.8}]
        )
        
        result = decomposition.to_dict()
        
        assert result["id"] == "test_id"
        assert result["user_id"] == "user_123"
        assert result["search_query"] == "Cars for the east coast"
        assert "subtopics" in result
        assert "created_at" in result
        assert "updated_at" in result
