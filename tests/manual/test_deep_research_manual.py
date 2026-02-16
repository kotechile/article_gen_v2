import asyncio
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

# Add project root to path
# We are in content_generator_v2/tests/manual
# Root is content_generator_v2
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

# Import from topic_analysis.backend.src.services.research.deep_research_service
from topic_analysis.backend.src.services.research.deep_research_service import DeepResearchService

async def test_deep_research_service():
    print("🧪 Testing DeepResearchService Initialization...")
    
    # Mock environment variables
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key", "OPENAI_API_KEY": "test-key"}):
        service = DeepResearchService()
        print("✅ Service initialized.")
        
        # Test perform_deep_research with mocks
        with patch.object(service, '_tavily_search', return_value="Mocked Search Results") as mock_search:
            # Patch the ReActAgent class where it is imported in the service module
            with patch('topic_analysis.backend.src.services.research.deep_research_service.ReActAgent') as mock_agent_cls:
                mock_agent = MagicMock()
                # achat must be async, so use AsyncMock or set return_value to a future
                mock_agent.achat = AsyncMock(return_value="## Deep Research Report\n\nMocked report content.")
                mock_agent_cls.return_value = mock_agent
                
                with patch.object(service, '_upload_to_rag', return_value=True) as mock_upload:
                    print("🧪 Executing perform_deep_research...")
                    result = await service.perform_deep_research(
                        title_id="test-title-123",
                        outline="Section 1: Introduction\nSection 2: Analysis",
                        user_id="test-user",
                        collection_name="test-collection"
                    )
                    
                    print(f"📊 Result: {result}")
                    
                    if result['success'] and result['doc_id'].startswith("deep_research_"):
                        print("✅ perform_deep_research succeeded.")
                    else:
                        print("❌ perform_deep_research failed.")

if __name__ == "__main__":
    asyncio.run(test_deep_research_service())
