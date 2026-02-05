"""
Tests for Trend Analysis functionality
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from uuid import uuid4

from ..services.trend_analysis_service import TrendAnalysisService
from ..services.google_trends_service import GoogleTrendsService
from ..services.csv_upload_service import CSVUploadService
from ..models.trend_analysis import TrendAnalysis, TrendAnalysisStatus, TrendAnalysisSource
from ..models.user import User

# Mock GoogleTrendsService for testing
class MockGoogleTrendsService:
    async def get_trend_data(self, **kwargs):
        return {
            "trends": [
                {
                    "keyword": "test keyword",
                    "search_volume": 1000,
                    "trend_score": 75,
                    "growth_rate": 15.5,
                    "seasonality": "stable",
                    "related_queries": ["test query 1", "test query 2"],
                    "time_series": [
                        {"date": "2024-01-01", "value": 70},
                        {"date": "2024-01-02", "value": 80}
                    ],
                    "source": "google_trends"
                }
            ],
            "source": "google_trends",
            "retrieved_at": datetime.utcnow().isoformat()
        }

# Mock CSVUploadService for testing
class MockCSVUploadService:
    async def get_cached_trend_data(self, user_id, workflow_session_id):
        return {
            "trend_data": {
                "trends": [
                    {
                        "keyword": "csv keyword",
                        "search_volume": 500,
                        "trend_score": 60,
                        "source": "csv_upload"
                    }
                ]
            }
        }

@pytest.fixture
def mock_db_session():
    """Mock SQLAlchemy session."""
    session = MagicMock(spec=Session)
    session.query.return_value.filter.return_value.first.return_value = None
    session.query.return_value.filter.return_value.all.return_value = []
    session.add.return_value = None
    session.commit.return_value = None
    session.refresh.return_value = None
    session.rollback.return_value = None
    return session

@pytest.fixture
def mock_google_trends_service():
    """Mock GoogleTrendsService."""
    return MockGoogleTrendsService()

@pytest.fixture
def mock_csv_upload_service():
    """Mock CSVUploadService."""
    return MockCSVUploadService()

@pytest.fixture
def trend_analysis_service(mock_db_session):
    """Create TrendAnalysisService with mocked dependencies."""
    service = TrendAnalysisService(mock_db_session)
    service.google_trends_service = MockGoogleTrendsService()
    service.csv_upload_service = MockCSVUploadService()
    return service

@pytest.mark.asyncio
async def test_create_trend_analysis_success(trend_analysis_service, mock_db_session):
    """Test successful trend analysis creation."""
    user_id = str(uuid4())
    workflow_session_id = str(uuid4())
    analysis_name = "Test Analysis"
    keywords = ["test keyword 1", "test keyword 2"]
    
    # Mock the database response
    mock_analysis = TrendAnalysis(
        id=str(uuid4()),
        user_id=user_id,
        workflow_session_id=workflow_session_id,
        analysis_name=analysis_name,
        keywords=keywords,
        timeframe="12m",
        geo="US",
        status=TrendAnalysisStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    mock_db_session.add.return_value = None
    mock_db_session.refresh.return_value = None
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_analysis
    
    result = await trend_analysis_service.create_trend_analysis(
        user_id=user_id,
        workflow_session_id=workflow_session_id,
        analysis_name=analysis_name,
        keywords=keywords
    )
    
    assert result is not None
    assert result["analysis_name"] == analysis_name
    assert result["keywords"] == keywords
    assert result["status"] == TrendAnalysisStatus.PENDING.value
    assert result["timeframe"] == "12m"
    assert result["geo"] == "US"
    
    # Verify DB interaction
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once()

@pytest.mark.asyncio
async def test_create_trend_analysis_validation_error(trend_analysis_service, mock_db_session):
    """Test trend analysis creation with validation error."""
    user_id = str(uuid4())
    workflow_session_id = str(uuid4())
    
    # Test with empty keywords
    with pytest.raises(ValueError, match="At least one keyword is required"):
        await trend_analysis_service.create_trend_analysis(
            user_id=user_id,
            workflow_session_id=workflow_session_id,
            analysis_name="Test Analysis",
            keywords=[]
        )

@pytest.mark.asyncio
async def test_start_trend_analysis_google_trends(trend_analysis_service, mock_db_session):
    """Test starting trend analysis with Google Trends source."""
    analysis_id = str(uuid4())
    user_id = str(uuid4())
    
    # Mock the database response
    mock_analysis = TrendAnalysis(
        id=analysis_id,
        user_id=user_id,
        workflow_session_id=str(uuid4()),
        analysis_name="Test Analysis",
        keywords=["test keyword"],
        timeframe="12m",
        geo="US",
        status=TrendAnalysisStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_analysis
    
    result = await trend_analysis_service.start_trend_analysis(
        analysis_id=analysis_id,
        user_id=user_id,
        source=TrendAnalysisSource.GOOGLE_TRENDS
    )
    
    assert result is not None
    assert result["status"] == TrendAnalysisStatus.COMPLETED.value
    assert "trend_data" in result
    assert "analysis_results" in result
    assert "insights" in result
    assert result["source"] == TrendAnalysisSource.GOOGLE_TRENDS.value

@pytest.mark.asyncio
async def test_start_trend_analysis_csv_upload(trend_analysis_service, mock_db_session):
    """Test starting trend analysis with CSV upload source."""
    analysis_id = str(uuid4())
    user_id = str(uuid4())
    
    # Mock the database response
    mock_analysis = TrendAnalysis(
        id=analysis_id,
        user_id=user_id,
        workflow_session_id=str(uuid4()),
        analysis_name="Test Analysis",
        keywords=["csv keyword"],
        timeframe="12m",
        geo="US",
        status=TrendAnalysisStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_analysis
    
    result = await trend_analysis_service.start_trend_analysis(
        analysis_id=analysis_id,
        user_id=user_id,
        source=TrendAnalysisSource.CSV_UPLOAD
    )
    
    assert result is not None
    assert result["status"] == TrendAnalysisStatus.COMPLETED.value
    assert "trend_data" in result
    assert "analysis_results" in result
    assert "insights" in result
    assert result["source"] == TrendAnalysisSource.CSV_UPLOAD.value

@pytest.mark.asyncio
async def test_start_trend_analysis_not_found(trend_analysis_service, mock_db_session):
    """Test starting trend analysis when analysis not found."""
    analysis_id = str(uuid4())
    user_id = str(uuid4())
    
    # Mock no analysis found
    mock_db_session.query.return_value.filter.return_value.first.return_value = None
    
    with pytest.raises(ValueError, match="Trend analysis not found"):
        await trend_analysis_service.start_trend_analysis(
            analysis_id=analysis_id,
            user_id=user_id,
            source=TrendAnalysisSource.GOOGLE_TRENDS
        )

@pytest.mark.asyncio
async def test_start_trend_analysis_wrong_status(trend_analysis_service, mock_db_session):
    """Test starting trend analysis when analysis is not in pending status."""
    analysis_id = str(uuid4())
    user_id = str(uuid4())
    
    # Mock analysis in wrong status
    mock_analysis = TrendAnalysis(
        id=analysis_id,
        user_id=user_id,
        workflow_session_id=str(uuid4()),
        analysis_name="Test Analysis",
        keywords=["test keyword"],
        timeframe="12m",
        geo="US",
        status=TrendAnalysisStatus.COMPLETED,  # Already completed
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_analysis
    
    with pytest.raises(ValueError, match="Analysis is not in pending status"):
        await trend_analysis_service.start_trend_analysis(
            analysis_id=analysis_id,
            user_id=user_id,
            source=TrendAnalysisSource.GOOGLE_TRENDS
        )

@pytest.mark.asyncio
async def test_get_trend_analysis_success(trend_analysis_service, mock_db_session):
    """Test getting trend analysis by ID."""
    analysis_id = str(uuid4())
    user_id = str(uuid4())
    
    # Mock the database response
    mock_analysis = TrendAnalysis(
        id=analysis_id,
        user_id=user_id,
        workflow_session_id=str(uuid4()),
        analysis_name="Test Analysis",
        keywords=["test keyword"],
        timeframe="12m",
        geo="US",
        status=TrendAnalysisStatus.COMPLETED,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_analysis
    
    result = await trend_analysis_service.get_trend_analysis(analysis_id, user_id)
    
    assert result is not None
    assert result["id"] == analysis_id
    assert result["analysis_name"] == "Test Analysis"
    assert result["status"] == TrendAnalysisStatus.COMPLETED.value

@pytest.mark.asyncio
async def test_get_trend_analysis_not_found(trend_analysis_service, mock_db_session):
    """Test getting trend analysis when not found."""
    analysis_id = str(uuid4())
    user_id = str(uuid4())
    
    # Mock no analysis found
    mock_db_session.query.return_value.filter.return_value.first.return_value = None
    
    result = await trend_analysis_service.get_trend_analysis(analysis_id, user_id)
    
    assert result is None

@pytest.mark.asyncio
async def test_list_trend_analyses(trend_analysis_service, mock_db_session):
    """Test listing trend analyses for a user."""
    user_id = str(uuid4())
    
    # Mock the database response
    mock_analyses = [
        TrendAnalysis(
            id=str(uuid4()),
            user_id=user_id,
            workflow_session_id=str(uuid4()),
            analysis_name="Analysis 1",
            keywords=["keyword 1"],
            timeframe="12m",
            geo="US",
            status=TrendAnalysisStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        ),
        TrendAnalysis(
            id=str(uuid4()),
            user_id=user_id,
            workflow_session_id=str(uuid4()),
            analysis_name="Analysis 2",
            keywords=["keyword 2"],
            timeframe="30d",
            geo="GB",
            status=TrendAnalysisStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
    ]
    mock_db_session.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = mock_analyses
    
    result = await trend_analysis_service.list_trend_analyses(user_id)
    
    assert len(result) == 2
    assert result[0]["analysis_name"] == "Analysis 1"
    assert result[1]["analysis_name"] == "Analysis 2"

@pytest.mark.asyncio
async def test_update_trend_analysis_success(trend_analysis_service, mock_db_session):
    """Test updating trend analysis."""
    analysis_id = str(uuid4())
    user_id = str(uuid4())
    
    # Mock the database response
    mock_analysis = TrendAnalysis(
        id=analysis_id,
        user_id=user_id,
        workflow_session_id=str(uuid4()),
        analysis_name="Test Analysis",
        keywords=["test keyword"],
        timeframe="12m",
        geo="US",
        status=TrendAnalysisStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_analysis
    
    updates = {
        "analysis_name": "Updated Analysis",
        "description": "Updated description",
        "timeframe": "30d"
    }
    
    result = await trend_analysis_service.update_trend_analysis(analysis_id, user_id, updates)
    
    assert result is not None
    assert result["analysis_name"] == "Updated Analysis"
    assert result["description"] == "Updated description"
    assert result["timeframe"] == "30d"
    
    # Verify DB interaction
    mock_db_session.commit.assert_called_once()

@pytest.mark.asyncio
async def test_update_trend_analysis_not_found(trend_analysis_service, mock_db_session):
    """Test updating trend analysis when not found."""
    analysis_id = str(uuid4())
    user_id = str(uuid4())
    
    # Mock no analysis found
    mock_db_session.query.return_value.filter.return_value.first.return_value = None
    
    updates = {"analysis_name": "Updated Analysis"}
    
    result = await trend_analysis_service.update_trend_analysis(analysis_id, user_id, updates)
    
    assert result is None

@pytest.mark.asyncio
async def test_delete_trend_analysis_success(trend_analysis_service, mock_db_session):
    """Test deleting trend analysis."""
    analysis_id = str(uuid4())
    user_id = str(uuid4())
    
    # Mock the database response
    mock_analysis = TrendAnalysis(
        id=analysis_id,
        user_id=user_id,
        workflow_session_id=str(uuid4()),
        analysis_name="Test Analysis",
        keywords=["test keyword"],
        timeframe="12m",
        geo="US",
        status=TrendAnalysisStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_analysis
    
    result = await trend_analysis_service.delete_trend_analysis(analysis_id, user_id)
    
    assert result is True
    
    # Verify DB interaction
    mock_db_session.delete.assert_called_once_with(mock_analysis)
    mock_db_session.commit.assert_called_once()

@pytest.mark.asyncio
async def test_delete_trend_analysis_not_found(trend_analysis_service, mock_db_session):
    """Test deleting trend analysis when not found."""
    analysis_id = str(uuid4())
    user_id = str(uuid4())
    
    # Mock no analysis found
    mock_db_session.query.return_value.filter.return_value.first.return_value = None
    
    result = await trend_analysis_service.delete_trend_analysis(analysis_id, user_id)
    
    assert result is False

@pytest.mark.asyncio
async def test_process_analysis_results(trend_analysis_service):
    """Test processing analysis results."""
    trend_data = {
        "trends": [
            {
                "keyword": "test keyword 1",
                "search_volume": 1000,
                "trend_score": 75,
                "growth_rate": 15.5,
                "seasonality": "stable",
                "related_queries": ["query 1", "query 2"],
                "time_series": [
                    {"date": "2024-01-01", "value": 70},
                    {"date": "2024-01-02", "value": 80}
                ],
                "source": "google_trends"
            },
            {
                "keyword": "test keyword 2",
                "search_volume": 500,
                "trend_score": 60,
                "growth_rate": -5.0,
                "seasonality": "high",
                "related_queries": ["query 3"],
                "time_series": [
                    {"date": "2024-01-01", "value": 60},
                    {"date": "2024-01-02", "value": 55}
                ],
                "source": "google_trends"
            }
        ]
    }
    
    mock_analysis = TrendAnalysis(
        id=str(uuid4()),
        user_id=str(uuid4()),
        workflow_session_id=str(uuid4()),
        analysis_name="Test Analysis",
        keywords=["test keyword 1", "test keyword 2"],
        timeframe="12m",
        geo="US",
        status=TrendAnalysisStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    
    result = await trend_analysis_service._process_analysis_results(trend_data, mock_analysis)
    
    assert result is not None
    assert "summary" in result
    assert "top_keywords" in result
    assert "growing_keywords" in result
    assert "declining_keywords" in result
    assert "time_series_analysis" in result
    
    # Check summary
    summary = result["summary"]
    assert summary["total_keywords"] == 2
    assert summary["total_volume"] == 1500
    assert summary["average_volume"] == 750
    assert summary["growing_keywords"] == 1
    assert summary["declining_keywords"] == 1
    
    # Check top keywords
    assert len(result["top_keywords"]) == 2
    assert result["top_keywords"][0]["keyword"] == "test keyword 1"  # Higher volume first

@pytest.mark.asyncio
async def test_generate_insights(trend_analysis_service):
    """Test generating insights from analysis results."""
    analysis_results = {
        "summary": {
            "total_keywords": 2,
            "total_volume": 1500,
            "average_volume": 750,
            "growing_keywords": 1,
            "declining_keywords": 1,
            "competition_levels": {"high": 1, "medium": 1}
        },
        "top_keywords": [
            {
                "keyword": "test keyword 1",
                "search_volume": 1000,
                "trend_score": 75,
                "source": "google_trends"
            }
        ],
        "growing_keywords": [
            {
                "keyword": "test keyword 1",
                "search_volume": 1000,
                "growth_rate": 15.5,
                "source": "google_trends"
            }
        ],
        "declining_keywords": [
            {
                "keyword": "test keyword 2",
                "search_volume": 500,
                "growth_rate": -5.0,
                "source": "google_trends"
            }
        ],
        "time_series_analysis": {
            "seasonal_keywords": [
                {
                    "keyword": "test keyword 2",
                    "pattern": "seasonal",
                    "peak_months": [6, 7, 8],
                    "variance": 100
                }
            ],
            "stable_keywords": [
                {
                    "keyword": "test keyword 1",
                    "pattern": "stable",
                    "variance": 25
                }
            ]
        }
    }
    
    mock_analysis = TrendAnalysis(
        id=str(uuid4()),
        user_id=str(uuid4()),
        workflow_session_id=str(uuid4()),
        analysis_name="Test Analysis",
        keywords=["test keyword 1", "test keyword 2"],
        timeframe="12m",
        geo="US",
        status=TrendAnalysisStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    
    result = await trend_analysis_service._generate_insights(analysis_results, mock_analysis)
    
    assert result is not None
    assert "insights" in result
    assert "top_trending" in result
    assert "growth_opportunities" in result
    assert "seasonal_patterns" in result
    assert "competition_level" in result
    assert "generated_at" in result
    
    # Check insights
    assert len(result["insights"]) > 0
    assert any("positive growth" in insight.lower() or "stable" in insight.lower() for insight in result["insights"])
    
    # Check top trending
    assert len(result["top_trending"]) == 1
    assert result["top_trending"][0]["keyword"] == "test keyword 1"
    
    # Check growth opportunities
    assert len(result["growth_opportunities"]) == 1
    assert result["growth_opportunities"][0]["keyword"] == "test keyword 1"
    
    # Check seasonal patterns
    assert len(result["seasonal_patterns"]) == 1
    assert result["seasonal_patterns"][0]["keyword"] == "test keyword 2"
