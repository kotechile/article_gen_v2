"""
Backend Unit Tests for Services
T110: Backend unit tests for all services
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone
import sys
from pathlib import Path

# Add backend src to path
backend_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(backend_src))

from services.affiliate_service import AffiliateService
from services.trend_service import TrendService
from services.keyword_service import KeywordService
from services.content_service import ContentService
from services.software_service import SoftwareService
from services.export_service import ExportService
from services.calendar_service import CalendarService
from services.user_service import UserService


class TestAffiliateService:
    """Test AffiliateService"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()
    
    @pytest.fixture
    def affiliate_service(self, mock_db):
        return AffiliateService(mock_db)
    
    def test_init(self, affiliate_service, mock_db):
        """Test service initialization"""
        assert affiliate_service.db == mock_db
    
    @pytest.mark.asyncio
    async def test_search_programs(self, affiliate_service):
        """Test search_programs method"""
        # Mock database query
        mock_query = Mock()
        mock_query.filter.return_value.offset.return_value.limit.return_value.all.return_value = []
        affiliate_service.db.query.return_value = mock_query
        
        result = await affiliate_service.search_programs(
            query="test",
            category="technology",
            user_id="user123"
        )
        
        assert result == []
        affiliate_service.db.query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_program_details(self, affiliate_service):
        """Test get_program_details method"""
        # Mock database query
        mock_program = Mock()
        mock_program.id = "prog123"
        mock_program.name = "Test Program"
        affiliate_service.db.query.return_value.filter.return_value.first.return_value = mock_program
        
        result = await affiliate_service.get_program_details("prog123")
        
        assert result is not None
        assert result.id == "prog123"
        assert result.name == "Test Program"
    
    @pytest.mark.asyncio
    async def test_create_research(self, affiliate_service):
        """Test create_research method"""
        # Mock database operations
        affiliate_service.db.add = Mock()
        affiliate_service.db.commit = Mock()
        affiliate_service.db.refresh = Mock()
        
        research_data = {
            "query": "test query",
            "category": "technology",
            "user_id": "user123"
        }
        
        result = await affiliate_service.create_research(research_data)
        
        assert result is not None
        affiliate_service.db.add.assert_called_once()
        affiliate_service.db.commit.assert_called_once()


class TestTrendService:
    """Test TrendService"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()
    
    @pytest.fixture
    def trend_service(self, mock_db):
        return TrendService(mock_db)
    
    def test_init(self, trend_service, mock_db):
        """Test service initialization"""
        assert trend_service.db == mock_db
    
    @pytest.mark.asyncio
    async def test_analyze_trends(self, trend_service):
        """Test analyze_trends method"""
        # Mock database query
        mock_query = Mock()
        mock_query.filter.return_value.offset.return_value.limit.return_value.all.return_value = []
        trend_service.db.query.return_value = mock_query
        
        result = await trend_service.analyze_trends(
            keywords=["test", "trend"],
            geo="US",
            timeframe="today 5-y",
            user_id="user123"
        )
        
        assert result == []
        trend_service.db.query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_trend_data(self, trend_service):
        """Test get_trend_data method"""
        # Mock database query
        mock_trend = Mock()
        mock_trend.id = "trend123"
        mock_trend.keywords = ["test", "trend"]
        trend_service.db.query.return_value.filter.return_value.first.return_value = mock_trend
        
        result = await trend_service.get_trend_data("trend123")
        
        assert result is not None
        assert result.id == "trend123"
        assert result.keywords == ["test", "trend"]


class TestKeywordService:
    """Test KeywordService"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()
    
    @pytest.fixture
    def keyword_service(self, mock_db):
        return KeywordService(mock_db)
    
    def test_init(self, keyword_service, mock_db):
        """Test service initialization"""
        assert keyword_service.db == mock_db
    
    @pytest.mark.asyncio
    async def test_upload_keywords(self, keyword_service):
        """Test upload_keywords method"""
        # Mock database operations
        keyword_service.db.add = Mock()
        keyword_service.db.commit = Mock()
        keyword_service.db.refresh = Mock()
        
        keywords_data = [
            {"keyword": "test1", "search_volume": 1000},
            {"keyword": "test2", "search_volume": 2000}
        ]
        
        result = await keyword_service.upload_keywords(keywords_data, "user123")
        
        assert result is not None
        assert len(result) == 2
        keyword_service.db.add.assert_called()
        keyword_service.db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_keywords(self, keyword_service):
        """Test get_keywords method"""
        # Mock database query
        mock_query = Mock()
        mock_query.filter.return_value.offset.return_value.limit.return_value.all.return_value = []
        keyword_service.db.query.return_value = mock_query
        
        result = await keyword_service.get_keywords("user123")
        
        assert result == []
        keyword_service.db.query.assert_called_once()


class TestContentService:
    """Test ContentService"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()
    
    @pytest.fixture
    def content_service(self, mock_db):
        return ContentService(mock_db)
    
    def test_init(self, content_service, mock_db):
        """Test service initialization"""
        assert content_service.db == mock_db
    
    @pytest.mark.asyncio
    async def test_generate_content_ideas(self, content_service):
        """Test generate_content_ideas method"""
        # Mock database operations
        content_service.db.add = Mock()
        content_service.db.commit = Mock()
        content_service.db.refresh = Mock()
        
        idea_data = {
            "title": "Test Idea",
            "description": "Test Description",
            "keywords": ["test", "content"],
            "user_id": "user123"
        }
        
        result = await content_service.generate_content_ideas(idea_data)
        
        assert result is not None
        content_service.db.add.assert_called_once()
        content_service.db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_content_ideas(self, content_service):
        """Test get_content_ideas method"""
        # Mock database query
        mock_query = Mock()
        mock_query.filter.return_value.offset.return_value.limit.return_value.all.return_value = []
        content_service.db.query.return_value = mock_query
        
        result = await content_service.get_content_ideas("user123")
        
        assert result == []
        content_service.db.query.assert_called_once()


class TestSoftwareService:
    """Test SoftwareService"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()
    
    @pytest.fixture
    def software_service(self, mock_db):
        return SoftwareService(mock_db)
    
    def test_init(self, software_service, mock_db):
        """Test service initialization"""
        assert software_service.db == mock_db
    
    @pytest.mark.asyncio
    async def test_generate_software_ideas(self, software_service):
        """Test generate_software_ideas method"""
        # Mock database operations
        software_service.db.add = Mock()
        software_service.db.commit = Mock()
        software_service.db.refresh = Mock()
        
        software_data = {
            "name": "Test Software",
            "description": "Test Description",
            "complexity_score": 5,
            "user_id": "user123"
        }
        
        result = await software_service.generate_software_ideas(software_data)
        
        assert result is not None
        software_service.db.add.assert_called_once()
        software_service.db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_software_ideas(self, software_service):
        """Test get_software_ideas method"""
        # Mock database query
        mock_query = Mock()
        mock_query.filter.return_value.offset.return_value.limit.return_value.all.return_value = []
        software_service.db.query.return_value = mock_query
        
        result = await software_service.get_software_ideas("user123")
        
        assert result == []
        software_service.db.query.assert_called_once()


class TestExportService:
    """Test ExportService"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()
    
    @pytest.fixture
    def export_service(self, mock_db):
        return ExportService(mock_db)
    
    def test_init(self, export_service, mock_db):
        """Test service initialization"""
        assert export_service.db == mock_db
    
    @pytest.mark.asyncio
    async def test_export_data(self, export_service):
        """Test export_data method"""
        # Mock database operations
        export_service.db.add = Mock()
        export_service.db.commit = Mock()
        export_service.db.refresh = Mock()
        
        export_data = {
            "format": "csv",
            "data": [{"test": "data"}],
            "user_id": "user123"
        }
        
        result = await export_service.export_data(export_data)
        
        assert result is not None
        export_service.db.add.assert_called_once()
        export_service.db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_export_history(self, export_service):
        """Test get_export_history method"""
        # Mock database query
        mock_query = Mock()
        mock_query.filter.return_value.offset.return_value.limit.return_value.all.return_value = []
        export_service.db.query.return_value = mock_query
        
        result = await export_service.get_export_history("user123")
        
        assert result == []
        export_service.db.query.assert_called_once()


class TestCalendarService:
    """Test CalendarService"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()
    
    @pytest.fixture
    def calendar_service(self, mock_db):
        return CalendarService(mock_db)
    
    def test_init(self, calendar_service, mock_db):
        """Test service initialization"""
        assert calendar_service.db == mock_db
    
    @pytest.mark.asyncio
    async def test_create_calendar_entry(self, calendar_service):
        """Test create_calendar_entry method"""
        # Mock database operations
        calendar_service.db.add = Mock()
        calendar_service.db.commit = Mock()
        calendar_service.db.refresh = Mock()
        
        entry_data = {
            "title": "Test Entry",
            "date": datetime.now(timezone.utc),
            "user_id": "user123"
        }
        
        result = await calendar_service.create_calendar_entry(entry_data)
        
        assert result is not None
        calendar_service.db.add.assert_called_once()
        calendar_service.db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_calendar_entries(self, calendar_service):
        """Test get_calendar_entries method"""
        # Mock database query
        mock_query = Mock()
        mock_query.filter.return_value.offset.return_value.limit.return_value.all.return_value = []
        calendar_service.db.query.return_value = mock_query
        
        result = await calendar_service.get_calendar_entries("user123")
        
        assert result == []
        calendar_service.db.query.assert_called_once()


class TestUserService:
    """Test UserService"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()
    
    @pytest.fixture
    def user_service(self, mock_db):
        return UserService(mock_db)
    
    def test_init(self, user_service, mock_db):
        """Test service initialization"""
        assert user_service.db == mock_db
    
    @pytest.mark.asyncio
    async def test_get_user(self, user_service):
        """Test get_user method"""
        # Mock database query
        mock_user = Mock()
        mock_user.id = "user123"
        mock_user.email = "test@example.com"
        user_service.db.query.return_value.filter.return_value.first.return_value = mock_user
        
        result = await user_service.get_user("user123")
        
        assert result is not None
        assert result.id == "user123"
        assert result.email == "test@example.com"
    
    @pytest.mark.asyncio
    async def test_update_user(self, user_service):
        """Test update_user method"""
        # Mock database operations
        mock_user = Mock()
        mock_user.id = "user123"
        user_service.db.query.return_value.filter.return_value.first.return_value = mock_user
        user_service.db.commit = Mock()
        user_service.db.refresh = Mock()
        
        update_data = {"name": "Updated Name"}
        
        result = await user_service.update_user("user123", update_data)
        
        assert result is not None
        user_service.db.commit.assert_called_once()
        user_service.db.refresh.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])