from typing import List, Dict, Any, Optional
import os
import logging
import asyncio
from datetime import datetime
from uuid import uuid4

# LlamaIndex & Tavily Imports
try:
    from llama_index.core import VectorStoreIndex, Document
    from llama_index.core.tools import FunctionTool
    from llama_index.core import VectorStoreIndex, Document
    from llama_index.core.tools import FunctionTool
    from llama_index.core.agent import ReActAgent
    from llama_index.llms.openai import OpenAI
    from tavily import TavilyClient
except ImportError:
    logging.warning("Deep Research dependencies not installed. Please install llama-index and tavily-python.")

from src.core.supabase_singleton import get_supabase_client
from ..rag_service import RAGService # Assuming RAGService exists for upload
from ...core.config import get_settings

logger = logging.getLogger(__name__)

class DeepResearchService:
    """
    Service for performing Deep Research using Agentic workflows (Tavily + LlamaIndex).
    Generates comprehensive reports to fill knowledge gaps.
    """

    def __init__(self):
        self.settings = get_settings()
        self.supabase = get_supabase_client()
        self.tavily_api_key = os.getenv("TAVILY_API_KEY") or self._get_api_key_from_db("tavily")
        self.openai_api_key = os.getenv("OPENAI_API_KEY") or self._get_api_key_from_db("openai")
        
        # Initialize RAG Service for uploading final reports
        # We need to instantiate it or get it from a singleton if available
        # checking previous file reads, it seems RAGService is in topic_analysis/backend/src/services/rag_service.py
        # but pure `rag_client.py` is in the root. 
        # based on existing code, we might use requests directly as user approved "external RAG service"
        self.rag_api_url = os.getenv("RAG_API_URL") or "http://localhost:8081" 

    def _get_api_key_from_db(self, provider: str) -> Optional[str]:
        """Fetch API key from Supabase if not in env"""
        try:
            response = self.supabase.table('api_keys').select('key_value').eq('provider', provider).eq('is_active', True).execute()
            if response.data:
                return response.data[0]['key_value']
        except Exception as e:
            logger.error(f"Error fetching {provider} key from DB: {e}")
        return None

    async def perform_deep_research(self, title_id: str, outline: str, user_id: str, collection_name: str) -> Dict[str, Any]:
        """
        Executes the Deep Research workflow for a single title.
        1. Plan research based on outline.
        2. Execute agentic search (Tavily).
        3. Synthesize report.
        4. Upload to RAG with citations.
        """
        if not self.tavily_api_key:
            raise ValueError("Tavily API Key is missing. Please add it to Settings -> API Keys.")

        logger.info(f"🚀 Starting Deep Research for Title ID: {title_id}")
        
        # Container to capture unique citations during the session
        collected_citations = []
        seen_urls = set()

        def tavily_search_wrapper(query: str) -> str:
            """Wrapper to capture headers/urls from search"""
            client = TavilyClient(api_key=self.tavily_api_key)
            response = client.search(query, search_depth="advanced", max_results=5)
            
            results_text = []
            for r in response.get('results', []):
                title = r.get('title', 'Unknown Source')
                url = r.get('url', '')
                content = r.get('content', '')
                
                # Capture citation if new
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    collected_citations.append({
                        "source": title,
                        "url": url,
                        "title": title
                    })
                
                results_text.append(f"- {title}: {content} ({url})")
            
            return "\n".join(results_text)

        # 1. Setup Agent
        tavily_tool = FunctionTool.from_defaults(
            fn=tavily_search_wrapper,
            name="web_search",
            description="Useful for searching the web for specific details, facts, and recent information."
        )
        
        llm = OpenAI(model="gpt-4o", api_key=self.openai_api_key) if self.openai_api_key else None
        if not llm:
             # Fallback if no OpenAI key, though ReActAgent usually needs a strong model
             logger.warning("No OpenAI key found for Deep Research Agent. It might fail.")

        agent = ReActAgent(tools=[tavily_tool], llm=llm, verbose=True)

        try:
            # 2. Execute Research
            prompt = f"""
            You are a Deep Research Expert. Your goal is to gather comprehensive information for an article section.
            
            CONTENT OUTLINE:
            {outline}
            
            TASK:
            1. Analyze the outline to identify key topics requiring external evidence, statistics, or examples.
            2. Use the web_search tool to find high-quality sources, data, and recent developments (2024-2025).
            3. Synthesize a "Deep Research Report" in Markdown format.
            4. The report MUST include citations or links to sources where possible.
            
            OUTPUT:
            Return ONLY the final Markdown report.
            """
            
            response = await agent.achat(prompt)
            report_content = str(response)

            # 3. Upload to RAG with Citations
            doc_id = f"deep_research_{title_id}_{uuid4().hex[:8]}"
            
            # Prepare metadata
            metadata = {
                "source_type": "deep_research",
                "citations": collected_citations,
                "title_id": title_id,
                "research_date": datetime.now().isoformat()
            }
            
            upload_success = await self._upload_to_rag(
                content=report_content,
                doc_id=doc_id,
                collection_name=collection_name,
                user_id=user_id,
                filename=f"Deep_Research_{title_id}.md",
                metadata=metadata
            )

            # 4. Update Title Status/Metadata? 
            # (Optional: Update title to say "Research Complete" or similar, handled by caller?)

            return {
                "success": True,
                "doc_id": doc_id,
                "report_preview": report_content[:200] + "...",
                "upload_status": upload_success,
                "citations_count": len(collected_citations)
            }

        except Exception as e:
            logger.error(f"Deep Research failed: {e}")
            return {"success": False, "error": str(e)}

    # Removed _tavily_search method as it is now an inner wrapper

    async def _upload_to_rag(self, content: str, doc_id: str, collection_name: str, user_id: str, filename: str, metadata: Dict[str, Any] = None) -> bool:
        """Uploads the synthesized report to the external RAG system"""
        import aiohttp
        import io
        import json
        
        file_obj = io.BytesIO(content.encode('utf-8'))
        file_obj.name = filename
        
        data = aiohttp.FormData()
        data.add_field('file', file_obj, filename=filename, content_type='text/markdown')
        data.add_field('docid', doc_id)
        data.add_field('collection_name', collection_name)
        data.add_field('user_id', user_id)
        
        if metadata:
            data.add_field('metadata', json.dumps(metadata))
            
        async with aiohttp.ClientSession() as session:
            try:
                # Assuming RAG endpoint is /upload
                url = f"{self.rag_api_url}/upload"
                async with session.post(url, data=data) as resp:
                    if resp.status == 200:
                        logger.info(f"✅ Automatically indexed Deep Research Report: {doc_id} with {len(metadata.get('citations', []))} citations")
                        return True
                    else:
                        logger.error(f"❌ Failed to upload Deep Research Report: {await resp.text()}")
                        return False
            except Exception as e:
                 logger.error(f"❌ Upload Connection Failed: {e}")
                 return False
