"""
Basic tests for Content Generator V2.

This module contains basic tests to verify the system structure
and basic functionality.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.models.research import ResearchRequest, ResearchDepth, ResearchTone
from src.core.models.errors import ValidationError
from src.utils.config import get_config


def test_research_request_creation():
    """Test creating a research request."""
    request = ResearchRequest(
        brief="Test research brief",
        keywords="test, keywords",
        provider="openai",
        model="gpt-4",
        api_key="test-key",
        depth=ResearchDepth.STANDARD,
        tone=ResearchTone.JOURNALISTIC,
        target_word_count=1000
    )
    
    assert request.brief == "Test research brief"
    assert request.keywords == "test, keywords"
    assert request.provider == "openai"
    assert request.model == "gpt-4"
    assert request.depth == ResearchDepth.STANDARD
    assert request.tone == ResearchTone.JOURNALISTIC
    assert request.target_word_count == 1000


def test_research_request_validation():
    """Test research request validation."""
    # Test valid request
    request = ResearchRequest(
        brief="Valid brief",
        keywords="valid, keywords",
        provider="openai",
        model="gpt-4",
        api_key="valid-key"
    )
    
    # Should not raise any validation errors
    assert request.brief == "Valid brief"
    
    # Test invalid request
    with pytest.raises(Exception):  # Pydantic validation error
        ResearchRequest(
            brief="",  # Empty brief should fail
            keywords="test",
            provider="openai",
            model="gpt-4",
            api_key="test-key"
        )


def test_config_loading():
    """Test configuration loading."""
    config = get_config('testing')
    
    assert config.TESTING is True
    assert config.DEBUG is True
    assert config.LOG_LEVEL == 'CRITICAL'


def test_imports():
    """Test that all modules can be imported."""
    # Test core models
    from src.core.models.research import ResearchRequest, ResearchResponse
    from src.core.models.article import Article, ArticleSection
    from src.core.models.evidence import Evidence, Claim
    from src.core.models.llm import LLMConfig, LLMResponse
    from src.core.models.errors import ContentGeneratorError
    
    # Test integrations
    from src.integrations.llm import LLMClient
    
    # Test API
    from src.api.app import create_app
    
    # Test tasks
    from src.tasks.celery_app import celery_app
    
    # Test utils
    from src.utils.config import get_config
    from src.utils.logging import setup_logging
    from src.utils.health import HealthChecker
    
    # If we get here, all imports succeeded
    assert True


def test_flask_app_creation():
    """Test Flask app creation."""
    from src.api.app import create_app
    
    app = create_app('testing')
    
    assert app is not None
    assert app.config['TESTING'] is True
    assert app.config['DEBUG'] is True


def test_celery_app_creation():
    """Test Celery app creation."""
    from src.tasks.celery_app import celery_app
    
    assert celery_app is not None
    assert celery_app.main == 'content_generator_v2'


if __name__ == '__main__':
    pytest.main([__file__])
