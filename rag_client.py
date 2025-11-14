"""
RAG (Retrieval-Augmented Generation) client for Content Generator V2.

This module provides integration with knowledge bases for evidence collection
and retrieval-augmented generation capabilities.
"""

import os
import time
import logging
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import json

# Configure logging
logger = logging.getLogger(__name__)

class RAGProvider(Enum):
    """Supported RAG providers."""
    CUSTOM = "custom"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    CHROMA = "chroma"
    QDRANT = "qdrant"

@dataclass
class RAGConfig:
    """Configuration for RAG requests."""
    endpoint: str
    api_key: Optional[str] = None
    collection: Optional[str] = None
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    max_results: int = 10  # Maps to top_k in API
    similarity_threshold: float = 0.7
    llm_provider: str = "deepseek"  # Default LLM provider

@dataclass
class RAGQuery:
    """RAG query parameters."""
    query: str
    collection: Optional[str] = None
    max_results: Optional[int] = None
    similarity_threshold: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    balance_emphasis: Optional[str] = None

@dataclass
class RAGResult:
    """RAG query result."""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    source: str
    source_type: str = "rag"
    relevance_score: float = 0.0
    credibility_score: float = 0.0

@dataclass
class RAGResponse:
    """RAG query response."""
    results: List[RAGResult]
    total_results: int
    query_time: float
    provider: str
    collection: str
    success: bool = True
    error: Optional[str] = None

class RAGClient:
    """
    RAG client for knowledge base integration.
    """
    
    def __init__(self, config: RAGConfig):
        """
        Initialize the RAG client.
        
        Args:
            config: RAG configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    async def query_async(self, rag_query: RAGQuery) -> RAGResponse:
        """
        Query the RAG system asynchronously.
        
        Args:
            rag_query: RAG query parameters
            
        Returns:
            RAGResponse with results
        """
        start_time = time.time()
        
        try:
            # Prepare query parameters according to API spec
            query_params = {
                "query": rag_query.query,
                "collection_name": rag_query.collection or self.config.collection or "default",
                "top_k": rag_query.max_results or self.config.max_results,  # Fixed: use top_k instead of max_results
                "llm": self.config.llm_provider or "deepseek"  # Added: LLM provider parameter
            }
            
            # Add optional parameters if provided
            if rag_query.similarity_threshold is not None:
                query_params["similarity_threshold"] = rag_query.similarity_threshold
            if rag_query.include_metadata is not None:
                query_params["include_metadata"] = rag_query.include_metadata
            if rag_query.balance_emphasis is not None:
                query_params["balance_emphasis"] = rag_query.balance_emphasis
            
            # Add filters if provided
            if rag_query.filters:
                query_params["filters"] = rag_query.filters
            
            # Make the request
            self.logger.info(f"Making RAG request to {self.config.endpoint} with timeout {self.config.timeout}s")
            self.logger.info(f"RAG Query Details:")
            self.logger.info(f"  - Query: '{rag_query.query}'")
            self.logger.info(f"  - Collection: '{query_params.get('collection_name', 'default')}'")
            self.logger.info(f"  - Top K: {query_params.get('top_k', 10)}")
            self.logger.info(f"  - Similarity Threshold: {query_params.get('similarity_threshold', 'default')}")
            self.logger.info(f"  - Full Query Params: {query_params}")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "ContentGeneratorV2/1.0"
                }
                
                if self.config.api_key:
                    headers["Authorization"] = f"Bearer {self.config.api_key}"
                
                async with session.post(
                    self.config.endpoint,
                    json=query_params,
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        query_time = time.time() - start_time
                        
                        # Log RAG response details
                        self.logger.info(f"RAG Response Details:")
                        self.logger.info(f"  - Status: {data.get('status', 'unknown')}")
                        self.logger.info(f"  - Documents Used: {len(data.get('documents_used', []))}")
                        self.logger.info(f"  - Chunks Used: {data.get('chunks_used', 0)}")
                        self.logger.info(f"  - Documents Searched: {data.get('documents_searched', 0)}")
                        self.logger.info(f"  - Search Quality: {data.get('searchQuality', 'unknown')}")
                        self.logger.info(f"  - Time Seconds: {data.get('time_seconds', 0)}")
                        
                        if data.get('documents_used'):
                            self.logger.info(f"  - Document Details:")
                            for i, doc in enumerate(data.get('documents_used', [])[:3]):  # Log first 3 docs
                                self.logger.info(f"    Doc {i+1}: ID={doc.get('document_id')}, Title='{doc.get('title')}', Author='{doc.get('author')}'")
                        
                        # Parse results
                        results = self._parse_results(data)
                        
                        self.logger.info(f"RAG query successful: {len(results)} results in {query_time:.2f}s")
                        
                        return RAGResponse(
                            results=results,
                            total_results=len(results),
                            query_time=query_time,
                            provider="custom",
                            collection=query_params["collection_name"],  # Fixed: use collection_name
                            success=True
                        )
                    else:
                        error_text = await response.text()
                        self.logger.error(f"RAG query failed: {response.status} - {error_text}")
                        
                        return RAGResponse(
                            results=[],
                            total_results=0,
                            query_time=time.time() - start_time,
                            provider="custom",
                            collection=query_params["collection_name"],  # Fixed: use collection_name
                            success=False,
                            error=f"HTTP {response.status}: {error_text}"
                        )
        
        except asyncio.TimeoutError:
            self.logger.error("RAG query timeout")
            return RAGResponse(
                results=[],
                total_results=0,
                query_time=time.time() - start_time,
                provider="custom",
                collection=rag_query.collection or self.config.collection,
                success=False,
                error="Query timeout"
            )
        
        except Exception as e:
            self.logger.error(f"RAG query error: {str(e)}")
            return RAGResponse(
                results=[],
                total_results=0,
                query_time=time.time() - start_time,
                provider="custom",
                collection=rag_query.collection or self.config.collection,
                success=False,
                error=str(e)
            )
    
    def query(self, rag_query: RAGQuery) -> RAGResponse:
        """
        Query the RAG system synchronously.
        
        Args:
            rag_query: RAG query parameters
            
        Returns:
            RAGResponse with results
        """
        return asyncio.run(self.query_async(rag_query))
    
    def _parse_results(self, data: Dict[str, Any]) -> List[RAGResult]:
        """
        Parse RAG query results.
        
        Args:
            data: Raw response data
            
        Returns:
            List of RAGResult objects
        """
        results = []
        
        try:
            # Handle the specific response format from your RAG system
            if data.get("status") == "success":
                # Get the main response content
                response_text = data.get("response", "")
                documents_used = data.get("documents_used", [])
                source_attribution = data.get("source_attribution", [])
                
                if response_text and response_text != "No relevant information found for your query.":
                    # Process each document used to create separate results
                    if documents_used:
                        for i, doc_info in enumerate(documents_used):
                            # Create metadata for this document
                            metadata = {
                                "query": data.get("query", ""),
                                "method": data.get("method", ""),
                                "search_quality": data.get("searchQuality", ""),
                                "time_seconds": data.get("time_seconds", 0),
                                "document_id": doc_info.get("document_id", ""),
                                "title": doc_info.get("title", ""),
                                "characters_retrieved": doc_info.get("characters_retrieved", 0),
                                "chunks_contributed": doc_info.get("chunks_contributed", 0),
                                "url": doc_info.get("url", ""),
                                "author": doc_info.get("author", ""),
                                "publication_date": doc_info.get("publish_date", doc_info.get("publication_date", "")),
                                "publisher": doc_info.get("publisher", "")
                            }
                            
                            # Create a meaningful source description using the actual title
                            doc_title = doc_info.get("title", "")
                            doc_author = doc_info.get("author", "")
                            doc_url = doc_info.get("url", "")
                            
                            # Use the actual title from the RAG response
                            if doc_title:
                                source = doc_title
                                # Add author if available
                                if doc_author:
                                    source = f"{doc_title} by {doc_author}"
                            else:
                                # Fallback to document ID if no title
                                source = f"Knowledge Base Document {doc_info.get('document_id', 'Unknown')}"
                            
                            # Add URL if available
                            if doc_url:
                                source = f"{source} ({doc_url})"
                            
                            # Calculate scores based on search quality and document relevance
                            search_quality = data.get("searchQuality", "basic")
                            base_score = 0.8 if search_quality == "high" else 0.6 if search_quality == "medium" else 0.4
                            similarity_score = base_score - (i * 0.1)  # Slight decrease for subsequent documents
                            
                            result = RAGResult(
                                content=response_text,  # Use the main response for all documents
                                metadata=metadata,
                                similarity_score=similarity_score,
                                source=source,
                                source_type="rag",
                                relevance_score=similarity_score,
                                credibility_score=0.8
                            )
                            results.append(result)
                    else:
                        # Fallback: create a single result from the main response
                        metadata = {
                            "query": data.get("query", ""),
                            "method": data.get("method", ""),
                            "search_quality": data.get("searchQuality", ""),
                            "time_seconds": data.get("time_seconds", 0)
                        }
                        
                        source = "RAG System Response"
                        search_quality = data.get("searchQuality", "basic")
                        similarity_score = 0.8 if search_quality == "high" else 0.6 if search_quality == "medium" else 0.4
                        
                        result = RAGResult(
                            content=response_text,
                            metadata=metadata,
                            similarity_score=similarity_score,
                            source=source,
                            source_type="rag",
                            relevance_score=similarity_score,
                            credibility_score=0.8
                        )
                        results.append(result)
            
            # Fallback to generic parsing for other formats
            elif "results" in data:
                items = data["results"]
                for item in items:
                    content = item.get("content", item.get("text", item.get("document", "")))
                    metadata = item.get("metadata", {})
                    similarity_score = item.get("similarity_score", item.get("score", 0.0))
                    source = item.get("source", item.get("url", item.get("id", "Unknown")))
                    
                    relevance_score = self._calculate_relevance_score(content, similarity_score)
                    credibility_score = self._calculate_credibility_score(metadata, source)
                    
                    result = RAGResult(
                        content=content,
                        metadata=metadata,
                        similarity_score=similarity_score,
                        source=source,
                        source_type="rag",
                        relevance_score=relevance_score,
                        credibility_score=credibility_score
                    )
                    
                    results.append(result)
        
        except Exception as e:
            self.logger.error(f"Error parsing RAG results: {str(e)}")
        
        return results
    
    def _calculate_relevance_score(self, content: str, similarity_score: float) -> float:
        """
        Calculate relevance score for a result.
        
        Args:
            content: Result content
            similarity_score: Similarity score from RAG system
            
        Returns:
            Relevance score (0-1)
        """
        # Base relevance on similarity score
        relevance = similarity_score
        
        # Boost for longer, more detailed content
        if len(content) > 500:
            relevance += 0.1
        elif len(content) > 200:
            relevance += 0.05
        
        # Boost for content with citations or references
        if any(indicator in content.lower() for indicator in ["source:", "reference:", "citation:", "doi:"]):
            relevance += 0.1
        
        return min(1.0, relevance)
    
    def _calculate_credibility_score(self, metadata: Dict[str, Any], source: str) -> float:
        """
        Calculate credibility score for a result.
        
        Args:
            metadata: Result metadata
            source: Source information
            
        Returns:
            Credibility score (0-1)
        """
        credibility = 0.5  # Base credibility
        
        # Boost for academic sources
        if any(domain in source.lower() for domain in [".edu", ".gov", ".org", "arxiv", "pubmed"]):
            credibility += 0.3
        
        # Boost for peer-reviewed sources
        if any(indicator in str(metadata).lower() for indicator in ["peer-reviewed", "journal", "conference", "proceedings"]):
            credibility += 0.2
        
        # Boost for recent content
        if "date" in metadata or "year" in metadata:
            try:
                date_str = metadata.get("date", metadata.get("year", ""))
                if date_str:
                    # Simple year extraction
                    year = int(str(date_str)[:4])
                    if year >= 2020:
                        credibility += 0.1
                    elif year >= 2015:
                        credibility += 0.05
            except:
                pass
        
        # Boost for authoritative sources
        if any(authority in source.lower() for authority in ["nature", "science", "ieee", "acm", "springer", "elsevier"]):
            credibility += 0.2
        
        return min(1.0, credibility)
    
    async def batch_query_async(self, queries: List[RAGQuery]) -> List[RAGResponse]:
        """
        Execute multiple RAG queries asynchronously.
        
        Args:
            queries: List of RAG queries
            
        Returns:
            List of RAGResponse objects
        """
        tasks = [self.query_async(query) for query in queries]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def batch_query(self, queries: List[RAGQuery]) -> List[RAGResponse]:
        """
        Execute multiple RAG queries synchronously.
        
        Args:
            queries: List of RAG queries
            
        Returns:
            List of RAGResponse objects
        """
        return asyncio.run(self.batch_query_async(queries))
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the RAG collection.
        
        Returns:
            Collection information
        """
        return {
            "endpoint": self.config.endpoint,
            "collection": self.config.collection,
            "max_results": self.config.max_results,
            "similarity_threshold": self.config.similarity_threshold,
            "timeout": self.config.timeout
        }

# Factory function for creating RAG clients
def create_rag_client(
    endpoint: str,
    api_key: Optional[str] = None,
    collection: Optional[str] = None,
    llm_provider: str = "deepseek",
    **kwargs
) -> RAGClient:
    """
    Create a RAG client with the specified configuration.
    
    Args:
        endpoint: RAG endpoint URL
        api_key: Optional API key
        collection: Optional collection name
        llm_provider: LLM provider to use (default: "deepseek")
        **kwargs: Additional configuration options
        
    Returns:
        Configured RAGClient instance
    """
    config = RAGConfig(
        endpoint=endpoint,
        api_key=api_key,
        collection=collection,
        llm_provider=llm_provider,
        **kwargs
    )
    
    return RAGClient(config)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    client = create_rag_client(
        endpoint="http://localhost:8080/query_hybrid_enhanced",
        collection="research_documents",
        max_results=5,
        similarity_threshold=0.7
    )
    
    query = RAGQuery(
        query="sustainable energy solutions",
        max_results=3
    )
    
    try:
        response = client.query(query)
        print(f"Found {response.total_results} results")
        for result in response.results:
            print(f"- {result.content[:100]}... (score: {result.similarity_score:.2f})")
    except Exception as e:
        print(f"Error: {e}")
