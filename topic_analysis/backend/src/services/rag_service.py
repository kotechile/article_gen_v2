"""
RAG Service for Article Generation
Handles RAG endpoint calls and integrates retrieved knowledge into article generation
"""

import httpx
import structlog
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = structlog.get_logger()

class RAGService:
    """Service for interacting with RAG endpoints"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def query_rag(
        self,
        query: str,
        rag_endpoint: str,
        collection_name: Optional[str] = None,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query RAG endpoint for relevant information
        
        Args:
            query: The search query
            rag_endpoint: The RAG endpoint URL (e.g., http://localhost:8080/query_simple)
            collection_name: Optional collection name to filter results
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant documents/chunks from RAG system
        """
        try:
            logger.info("Querying RAG endpoint", 
                       endpoint=rag_endpoint, 
                       query=query[:100],
                       collection=collection_name)
            
            # Prepare request payload
            payload = {
                "query": query,
                "max_results": max_results
            }
            
            if collection_name:
                payload["collection_name"] = collection_name
            
            # Make request to RAG endpoint
            response = await self.client.post(
                rag_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract relevant documents/chunks
            documents = []
            if isinstance(result, dict):
                # Handle different response formats
                if "results" in result:
                    documents = result["results"]
                elif "documents" in result:
                    documents = result["documents"]
                elif "chunks" in result:
                    documents = result["chunks"]
                elif "data" in result:
                    documents = result["data"] if isinstance(result["data"], list) else [result["data"]]
                else:
                    # If the response itself is a list or contains the data directly
                    documents = [result] if not isinstance(result, list) else result
            elif isinstance(result, list):
                documents = result
            
            logger.info("RAG query successful", 
                       endpoint=rag_endpoint,
                       documents_found=len(documents))
            
            return documents[:max_results]
            
        except httpx.HTTPStatusError as e:
            logger.error("RAG endpoint HTTP error", 
                        endpoint=rag_endpoint,
                        status_code=e.response.status_code,
                        error=str(e))
            return []
        except httpx.RequestError as e:
            logger.error("RAG endpoint request error", 
                        endpoint=rag_endpoint,
                        error=str(e))
            return []
        except Exception as e:
            logger.error("RAG query failed", 
                        endpoint=rag_endpoint,
                        error=str(e),
                        exc_info=True)
            return []
    
    async def query_multiple(
        self,
        queries: List[str],
        rag_endpoint: str,
        collection_name: Optional[str] = None,
        max_results_per_query: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query RAG endpoint with multiple queries
        
        Args:
            queries: List of search queries
            rag_endpoint: The RAG endpoint URL
            collection_name: Optional collection name
            max_results_per_query: Max results per query
            
        Returns:
            Dictionary mapping queries to their results
        """
        results = {}
        
        for query in queries:
            documents = await self.query_rag(
                query=query,
                rag_endpoint=rag_endpoint,
                collection_name=collection_name,
                max_results=max_results_per_query
            )
            results[query] = documents
        
        return results
    
    def format_rag_context(
        self,
        documents: List[Dict[str, Any]],
        include_sources: bool = True
    ) -> str:
        """
        Format RAG documents into context string for LLM prompt
        
        Args:
            documents: List of RAG documents
            include_sources: Whether to include source citations
            
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        context_parts = []
        context_parts.append("## Relevant Knowledge Base Information:\n")
        
        for i, doc in enumerate(documents, 1):
            # Handle different document formats
            content = ""
            source = ""
            
            if isinstance(doc, dict):
                content = doc.get("content", doc.get("text", doc.get("chunk", "")))
                source = doc.get("source", doc.get("url", doc.get("metadata", {}).get("source", "")))
            elif isinstance(doc, str):
                content = doc
            else:
                content = str(doc)
            
            if content:
                context_parts.append(f"\n### Source {i}:\n{content}")
                
                if include_sources and source:
                    context_parts.append(f"\n*Source: {source}*")
        
        return "\n".join(context_parts)
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()



