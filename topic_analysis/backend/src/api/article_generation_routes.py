"""
Article Generation API Routes with RAG Integration
Handles article generation requests with RAG support
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import structlog

from ..services.rag_service import RAGService
from ..services.enhanced_content_generator import EnhancedContentGenerator
from ..integrations.llm_providers import generate_content as llm_generate_content, llm_providers_manager

logger = structlog.get_logger()
router = APIRouter(prefix="/api/article", tags=["article-generation"])

# Also create a research endpoint that matches the expected payload format
research_router = APIRouter(prefix="/api/research", tags=["research"])

class ArticleGenerationRequest(BaseModel):
    """Request schema for article generation with RAG"""
    brief: str = Field(..., description="Article brief/description")
    keywords: str = Field(..., description="Comma-separated keywords")
    target_word_count: int = Field(2000, ge=500, le=10000, description="Target word count")
    tone: str = Field("professional", description="Writing tone")
    depth: str = Field("comprehensive", description="Research depth")
    
    # RAG Configuration
    rag_enabled: bool = Field(False, description="Enable RAG integration")
    rag_endpoint: Optional[str] = Field(None, description="RAG endpoint URL")
    rag_collection_name: Optional[str] = Field(None, description="RAG collection name")
    rag_balance_emphasis: str = Field("auto", description="RAG balance emphasis")
    include_in_text_citations: bool = Field(False, description="Include in-text citations")
    
    # LLM Configuration
    llm_model: Optional[str] = Field(None, description="LLM model to use")
    llm_key: Optional[str] = Field(None, description="LLM API key")
    llm_provider: Optional[str] = Field(None, description="LLM provider")
    
    # Additional options
    use_verbalized_sampling: bool = Field(False, description="Use verbalized sampling")
    claims_research_enabled: bool = Field(False, description="Enable claims research")

class ArticleGenerationResponse(BaseModel):
    """Response schema for article generation"""
    success: bool
    article: str
    word_count: int
    sections: List[Dict[str, Any]]
    rag_sources: Optional[List[Dict[str, Any]]] = None
    message: str

@router.post("/generate", response_model=ArticleGenerationResponse)
async def generate_article(request: ArticleGenerationRequest):
    """
    Generate article from brief with optional RAG integration
    
    This endpoint:
    1. Queries RAG system if enabled
    2. Generates article using LLM with RAG context
    3. Returns complete article with sources
    """
    try:
        logger.info("Starting article generation", 
                   brief_length=len(request.brief),
                   rag_enabled=request.rag_enabled,
                   target_words=request.target_word_count)
        
        # Initialize services
        rag_service = RAGService()
        
        # Prepare keywords list
        keywords_list = [k.strip() for k in request.keywords.split(",") if k.strip()]
        
        # Step 1: Query RAG if enabled
        rag_context = ""
        rag_sources = []
        
        if request.rag_enabled and request.rag_endpoint:
            logger.info("RAG enabled, querying RAG endpoint", 
                       endpoint=request.rag_endpoint,
                       collection=request.rag_collection_name)
            
            try:
                # Create queries from brief and keywords
                queries = [request.brief] + keywords_list[:3]  # Use brief + top 3 keywords
                
                # Query RAG for each query
                rag_results = await rag_service.query_multiple(
                    queries=queries,
                    rag_endpoint=request.rag_endpoint,
                    collection_name=request.rag_collection_name,
                    max_results_per_query=3
                )
                
                # Collect all documents
                all_documents = []
                for query, docs in rag_results.items():
                    all_documents.extend(docs)
                
                # Format RAG context for LLM
                if all_documents:
                    rag_context = rag_service.format_rag_context(
                        documents=all_documents,
                        include_sources=request.include_in_text_citations
                    )
                    rag_sources = all_documents
                    
                    logger.info("RAG context retrieved", 
                               documents_found=len(all_documents),
                               context_length=len(rag_context))
                else:
                    logger.warning("RAG query returned no results")
                    
            except Exception as rag_error:
                logger.error("RAG query failed, continuing without RAG", 
                           error=str(rag_error))
                # Continue without RAG if it fails
        
        # Step 2: Generate article using LLM
        article_prompt = _build_article_prompt(
            brief=request.brief,
            keywords=keywords_list,
            target_word_count=request.target_word_count,
            tone=request.tone,
            depth=request.depth,
            rag_context=rag_context,
            include_citations=request.include_in_text_citations
        )
        
        logger.info("Generating article with LLM", 
                   prompt_length=len(article_prompt),
                   has_rag_context=bool(rag_context))
        
        # Generate article content using LLM
        # Determine provider (use request provider or default to first available)
        provider = request.llm_provider or "openai"
        if provider not in llm_providers_manager.providers:
            # Fallback to first available provider
            available = list(llm_providers_manager.providers.keys())
            provider = available[0] if available else "openai"
            logger.warning(f"Requested provider not available, using {provider}")
        
        max_tokens = request.target_word_count * 2  # Rough estimate: 2 tokens per word
        
        logger.info("Calling LLM for article generation", 
                   provider=provider,
                   max_tokens=max_tokens,
                   has_rag=bool(rag_context))
        
        llm_result = await llm_generate_content(
            prompt=article_prompt,
            provider=provider,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        # Extract content from LLM result
        if isinstance(llm_result, dict):
            article_content = llm_result.get("content", llm_result.get("text", ""))
            if not article_content and "error" in llm_result:
                raise HTTPException(
                    status_code=500,
                    detail=f"LLM generation failed: {llm_result.get('error')}"
                )
        else:
            article_content = str(llm_result) if llm_result else ""
        
        if not article_content or len(article_content.strip()) < 100:
            raise HTTPException(
                status_code=500,
                detail="Article generation failed: Generated content is too short or empty"
            )
        
        # Step 3: Parse article into sections
        sections = _parse_article_sections(article_content)
        
        # Step 4: Calculate word count
        word_count = len(article_content.split())
        
        logger.info("Article generation completed", 
                   word_count=word_count,
                   sections=len(sections),
                   rag_sources_count=len(rag_sources))
        
        # Clean up
        await rag_service.close()
        
        return ArticleGenerationResponse(
            success=True,
            article=article_content,
            word_count=word_count,
            sections=sections,
            rag_sources=rag_sources if request.rag_enabled else None,
            message=f"Article generated successfully with {word_count} words"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Article generation failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Article generation failed: {str(e)}"
        )

def _build_article_prompt(
    brief: str,
    keywords: List[str],
    target_word_count: int,
    tone: str,
    depth: str,
    rag_context: str = "",
    include_citations: bool = False
) -> str:
    """Build comprehensive article generation prompt"""
    
    prompt_parts = [
        f"# Article Generation Request\n",
        f"## Brief:\n{brief}\n",
        f"## Requirements:\n",
        f"- Target word count: {target_word_count} words",
        f"- Writing tone: {tone}",
        f"- Research depth: {depth}",
        f"- Keywords to include: {', '.join(keywords)}",
    ]
    
    if rag_context:
        prompt_parts.append(f"\n## Knowledge Base Context:\n{rag_context}\n")
        prompt_parts.append(
            "\n## Instructions:\n"
            "Use the knowledge base context above to inform your article. "
            "Integrate the information naturally into your writing. "
        )
        if include_citations:
            prompt_parts.append(
                "Include in-text citations where you use information from the knowledge base. "
            )
    else:
        prompt_parts.append(
            "\n## Instructions:\n"
            "Write a comprehensive, well-researched article based on the brief above. "
            "Use your knowledge to provide accurate, valuable information. "
        )
    
    prompt_parts.extend([
        "\n## Article Structure:\n",
        "1. **Introduction**: Engaging opening that hooks the reader and introduces the topic",
        "2. **Main Content**: Detailed sections covering all aspects of the topic",
        "3. **Practical Examples**: Real-world examples, case studies, or actionable advice",
        "4. **Conclusion**: Summary and key takeaways",
        "\n## Writing Guidelines:\n",
        "- Write in a clear, engaging style",
        "- Use proper heading hierarchy (H2, H3)",
        "- Include specific examples and data",
        "- Make it actionable and valuable",
        "- Ensure natural keyword integration",
        "- Maintain consistent tone throughout",
        "\nGenerate the complete article now:"
    ])
    
    return "\n".join(prompt_parts)

def _parse_article_sections(article: str) -> List[Dict[str, Any]]:
    """Parse article into sections"""
    import re
    
    sections = []
    lines = article.split('\n')
    
    current_section = None
    current_content = []
    
    for line in lines:
        # Check for headings (markdown format)
        heading_match = re.match(r'^#+\s+(.+)$', line.strip())
        if heading_match:
            # Save previous section
            if current_section:
                sections.append({
                    'heading': current_section,
                    'content': '\n'.join(current_content).strip(),
                    'word_count': len('\n'.join(current_content).split())
                })
            
            # Start new section
            current_section = heading_match.group(1)
            current_content = []
        else:
            if line.strip():
                current_content.append(line)
    
    # Add last section
    if current_section:
        sections.append({
            'heading': current_section,
            'content': '\n'.join(current_content).strip(),
            'word_count': len('\n'.join(current_content).split())
        })
    
    # If no sections found, create a single section
    if not sections:
        sections.append({
            'heading': 'Article',
            'content': article,
            'word_count': len(article.split())
        })
    
    return sections

@research_router.post("/generate")
async def research_and_generate(request: ArticleGenerationRequest):
    """
    Research and generate article endpoint
    This endpoint logs "Starting research with payload" and handles the research request
    """
    logger.info("Starting research with payload", payload=request.dict())
    
    # Delegate to the article generation endpoint
    return await generate_article(request)

