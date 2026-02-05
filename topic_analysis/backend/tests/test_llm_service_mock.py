import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.services.llm.llm_service import LLMService

@pytest.mark.asyncio
async def test_get_provider_default():
    """Test getting default provider when no name specified"""
    
    # Mock supabase client
    mock_supabase = MagicMock()
    mock_query = MagicMock()
    mock_supabase.table.return_value.select.return_value = mock_query
    
    # Mock response for default provider
    mock_response = MagicMock()
    mock_response.data = [{
        "provider": "openai",
        "name": "OpenAI GPT-4",
        "model_name": "gpt-4",
        "base_url": None,
        "api_keys": {
            "key_value": "sk-test-key",
            "base_url": None
        }
    }]
    
    # Setup chain for default query
    # eq("is_default", True).eq("is_active", True).execute()
    mock_query.eq.return_value.eq.return_value.execute.return_value = mock_response

    # Patch the global get_supabase_client to return our mock
    with patch("src.services.llm.llm_service.get_supabase_client", return_value=mock_supabase):
        service = LLMService()
        # Re-assign mock because __init__ called the real one before patch if not careful, 
        # but here we patch the import so __init__ will use it if we instantiate inside patch, 
        # or we just set it manually
        service.supabase = mock_supabase
        
        provider = await service.get_provider()
        
        assert provider.api_key == "sk-test-key"
        assert provider.model_name == "gpt-4"
        assert provider.__class__.__name__ == "OpenAIProvider"

@pytest.mark.asyncio
async def test_get_provider_specific():
    """Test getting specific provider by name"""
    
    mock_supabase = MagicMock()
    mock_query = MagicMock()
    mock_supabase.table.return_value.select.return_value = mock_query
    
    # Mock response for specific provider
    mock_response = MagicMock()
    mock_response.data = [{
        "provider": "google",
        "name": "Gemini Pro",
        "model_name": "gemini-pro",
        "base_url": None,
        "api_keys": {
            "key_value": "google-key",
            "base_url": None
        }
    }]
    
    # Setup chain: eq("provider", "google").eq("is_active", True).execute()
    mock_query.eq.return_value.eq.return_value.execute.return_value = mock_response

    with patch("src.services.llm.llm_service.get_supabase_client", return_value=mock_supabase):
        service = LLMService()
        service.supabase = mock_supabase
        
        provider = await service.get_provider("google")
        
        assert provider.api_key == "google-key"
        assert provider.model_name == "gemini-pro"
        assert provider.__class__.__name__ == "GeminiProvider"

@pytest.mark.asyncio
async def test_get_provider_not_found():
    """Test error when provider not found"""
    
    mock_supabase = MagicMock()
    mock_query = MagicMock()
    mock_supabase.table.return_value.select.return_value = mock_query
    
    # Empty data
    mock_response = MagicMock()
    mock_response.data = []
    
    mock_query.eq.return_value.eq.return_value.execute.return_value = mock_response

    with patch("src.services.llm.llm_service.get_supabase_client", return_value=mock_supabase):
        service = LLMService()
        service.supabase = mock_supabase
        
        with pytest.raises(ValueError) as excinfo:
            await service.get_provider("nonexistent")
        
        assert "not found" in str(excinfo.value)
