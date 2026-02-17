import asyncio
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

# Add project root to path
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
        
        # Mock TavilyClient in the service module
        with patch("topic_analysis.backend.src.services.research.deep_research_service.TavilyClient") as mock_tavily_cls:
            mock_tavily_instance = MagicMock()
            mock_tavily_instance.search.return_value = {
                "results": [
                    {"title": "Source 1", "url": "http://example.com/1", "content": "Content 1"},
                    {"title": "Source 2", "url": "http://example.com/2", "content": "Content 2"}
                ]
            }
            mock_tavily_cls.return_value = mock_tavily_instance

            # Patch ReActAgent INSIDE the TavilyClient mock scope
            with patch('topic_analysis.backend.src.services.research.deep_research_service.ReActAgent') as mock_agent_cls:
                mock_agent = MagicMock()
                
                # Define a side effect for achat that calls the tool wrapper
                async def mock_achat_side_effect(*args, **kwargs):
                    try:
                        # Try to get tools from kwargs
                        tools = mock_agent_cls.call_args.kwargs.get('tools')
                        if not tools and mock_agent_cls.call_args.args:
                                tools = mock_agent_cls.call_args.args[0]
                        
                        if tools:
                            search_tool = tools[0]
                            # Simulate agent calling the tool
                            search_tool.fn("test query")
                    except Exception as e:
                        print(f"⚠️ Failed to invoke tool in mock: {e}")
                        import traceback
                        traceback.print_exc()

                    return "## Deep Research Report\n\nMocked report content."

                mock_agent.achat = AsyncMock(side_effect=mock_achat_side_effect)
                mock_agent_cls.return_value = mock_agent

                # Patch aiohttp.ClientSession to capture the upload request
                with patch("aiohttp.ClientSession") as mock_session_cls:
                    mock_session = MagicMock()
                    mock_post = AsyncMock()
                    mock_post.__aenter__.return_value.status = 200
                    mock_session.post.return_value = mock_post
                    mock_session.__aenter__.return_value = mock_session
                    mock_session_cls.return_value = mock_session

                    print("🧪 Executing perform_deep_research...")
                    result = await service.perform_deep_research(
                        title_id="test-title-123",
                        outline="Section 1: Introduction",
                        user_id="test-user",
                        collection_name="test-collection"
                    )
                    
                    print(f"📊 Result: {result}")
                    
                    # Verify aiohttp post was called
                    if mock_session.post.called:
                        call_args = mock_session.post.call_args
                        # args: (url,), kwargs: data=...
                        # data is aiohttp.FormData
                        form_data = call_args.kwargs.get('data')
                        print(f"📦 Uploaded Data Type: {type(form_data)}")
                        # We assume if it reached here and status is True (from mock), it passed data.
                    else:
                        print("❌ Upload was NOT called.")

                    if result['success'] and result['citations_count'] == 2:
                         print("✅ Citations correctly captured.")
                         if result['upload_status']:
                             print("✅ Upload simulation successful.")
                    else:
                        print(f"❌ Verification failed. Citations: {result.get('citations_count')}")

if __name__ == "__main__":
    asyncio.run(test_deep_research_service())
