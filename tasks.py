"""
Celery tasks for Content Generator V2.

This module contains all the asynchronous tasks for the article generation pipeline.
"""

import logging
import os
import re
import html
from datetime import datetime
from typing import Dict, Any, Optional, List
from celery import current_task
from celery_config import celery_app
from llm_client_direct import create_llm_client, LLMResponse
from rag_client import create_rag_client, RAGQuery
from linkup_client import create_linkup_client, SearchQuery
from article_structure_generator import create_article_structure_generator, ArticleStructureGenerator
from content_generator import create_content_generator, get_tone_specific_instructions
from citation_generator import create_citation_generator, CitationStyle
# Evidence ranking will be done inline
from config import get_config
from supabase_client import get_linkup_api_key

# Configure logging
logger = logging.getLogger(__name__)


def _assess_rag_coverage(
    rag_evidence: List[Dict[str, Any]],
    keywords: str = "",
    min_sources: int = 3,
    min_relevance: float = 0.6
) -> Dict[str, Any]:
    """
    Assess RAG coverage quality to determine if Linkup search is needed.
    
    Args:
        rag_evidence: List of evidence items from RAG
        keywords: Keywords to check coverage for
        min_sources: Minimum number of sources required (default: 3)
        min_relevance: Minimum average relevance score (default: 0.6)
        
    Returns:
        Dict with coverage assessment:
        - sufficient: bool - Whether RAG coverage is sufficient
        - source_count: int - Number of unique sources
        - avg_relevance: float - Average relevance score
        - keyword_coverage: float - Keyword coverage ratio (0-1)
        - assessment: str - Coverage assessment description
    """
    if not rag_evidence:
        return {
            'sufficient': False,
            'source_count': 0,
            'avg_relevance': 0.0,
            'keyword_coverage': 0.0,
            'assessment': 'insufficient_no_sources'
        }
    
    # Count unique sources
    unique_sources = set()
    relevance_scores = []
    keyword_matches = 0
    keyword_count = len(keywords.split()) if keywords else 0
    
    for evidence in rag_evidence:
        source = evidence.get('source', '')
        if source:
            unique_sources.add(source)
        
        relevance = evidence.get('relevance_score', 0.0) or evidence.get('similarity_score', 0.0)
        if relevance > 0:
            relevance_scores.append(relevance)
        
        # Check keyword coverage
        content = str(evidence.get('content', '')) + ' ' + str(evidence.get('source', ''))
        content_lower = content.lower()
        if keywords:
            matched_keywords = sum(1 for kw in keywords.lower().split() if kw in content_lower)
            keyword_matches += matched_keywords
    
    source_count = len(unique_sources)
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    keyword_coverage = (keyword_matches / keyword_count) if keyword_count > 0 else 0.0
    
    # Determine if sufficient
    sufficient = (
        source_count >= min_sources and
        avg_relevance >= min_relevance and
        (keyword_coverage >= 0.5 or keyword_count == 0)  # Require 50% keyword coverage if keywords provided
    )
    
    # Generate assessment description
    if sufficient:
        assessment = f'sufficient_{source_count}_sources'
    elif source_count < min_sources:
        assessment = f'insufficient_sources_{source_count}_needed_{min_sources}'
    elif avg_relevance < min_relevance:
        assessment = f'insufficient_relevance_{avg_relevance:.2f}_needed_{min_relevance:.2f}'
    else:
        assessment = f'insufficient_keyword_coverage_{keyword_coverage:.2f}'
    
    return {
        'sufficient': sufficient,
        'source_count': source_count,
        'avg_relevance': avg_relevance,
        'keyword_coverage': keyword_coverage,
        'assessment': assessment
    }


def _determine_optimal_search_depth(
    request_depth: str,
    rag_coverage: Dict[str, Any],
    query: str,
    auto_downgrade: bool = True
) -> str:
    """
    Determine optimal Linkup search depth based on RAG coverage and query complexity.
    
    Args:
        request_depth: User-requested depth ('standard', 'comprehensive', 'deep')
        rag_coverage: RAG coverage assessment result
        query: Search query string
        auto_downgrade: Whether to auto-downgrade comprehensive to standard if RAG sufficient
        
    Returns:
        Optimal depth: 'standard' or 'deep'
    """
    # If user explicitly requested 'deep', honor it
    if request_depth == 'deep':
        return 'deep'
    
    # If RAG coverage is sufficient and auto-downgrade is enabled, use standard
    if rag_coverage.get('sufficient', False) and auto_downgrade:
        if request_depth == 'comprehensive':
            logger.info(f"RAG coverage sufficient ({rag_coverage['source_count']} sources, "
                       f"relevance: {rag_coverage['avg_relevance']:.2f}), "
                       f"downgrading 'comprehensive' to 'standard' search")
            return 'standard'
        elif request_depth == 'standard':
            return 'standard'
    
    # Assess query complexity
    query_lower = query.lower()
    word_count = len(query.split())
    has_multiple_questions = query_lower.count('?') > 1
    has_complex_indicators = any(indicator in query_lower for indicator in [
        'compare', 'analyze', 'review', 'evaluate', 'assess', 'difference',
        'pros and cons', 'advantages disadvantages'
    ])
    
    # Complex queries need deep search if RAG insufficient
    if has_complex_indicators or (has_multiple_questions and word_count > 10):
        if not rag_coverage.get('sufficient', False):
            logger.info(f"Complex query detected with insufficient RAG coverage, using 'deep' search")
            return 'deep'
    
    # Map comprehensive to deep only if RAG insufficient
    if request_depth == 'comprehensive':
        if rag_coverage.get('sufficient', False):
            return 'standard'
        else:
            return 'deep'
    
    # Default to standard
    return 'standard'

# Task status constants
TASK_STATUS = {
    'PENDING': 'PENDING',
    'PROGRESS': 'PROGRESS', 
    'SUCCESS': 'SUCCESS',
    'FAILURE': 'FAILURE',
    'CANCELLED': 'CANCELLED'
}

def _generate_wp_slug(title: str) -> str:
    """Generate WordPress-friendly slug from title (max 90 characters for SEO)."""
    # Convert to lowercase
    slug = title.lower()
    # Replace spaces and special characters with hyphens
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)
    # Remove leading/trailing hyphens
    slug = slug.strip('-')
    # Limit length to 90 characters for optimal SEO and WordPress URL length
    if len(slug) > 90:
        # Truncate at word boundary if possible
        truncated = slug[:90].rsplit('-', 1)[0]
        if len(truncated) < 70:  # If truncation removed too much, just cut at 90
            slug = slug[:90].rstrip('-')
        else:
            slug = truncated
    return slug

def _extract_plain_text(html_content: str) -> str:
    """Extract plain text from HTML content."""
    if not html_content:
        return ''
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', html_content)
    # Decode HTML entities
    text = html.unescape(text)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def _truncate_seo_title(title: str, max_length: int = 60, focus_keyword: str = '') -> str:
    """
    Truncate title to max 60 characters for SEO meta title, preserving focus keyword.
    
    Args:
        title: The full title to truncate
        max_length: Maximum length (default 60 for SEO)
        focus_keyword: Primary keyword that must be preserved in the truncated title
    
    Returns:
        Truncated title that includes the focus keyword if provided
    """
    if len(title) <= max_length:
        return title
    
    # If no keyword provided, use standard truncation
    if not focus_keyword:
        truncated = title[:max_length].rsplit(' ', 1)[0]
        if len(truncated) < max_length * 0.8:
            truncated = title[:max_length-3] + '...'
        else:
            truncated = truncated.rstrip('.,;:')
        return truncated
    
    # Normalize keyword for matching (case-insensitive, handle variations)
    focus_keyword_lower = focus_keyword.lower().strip()
    title_lower = title.lower()
    
    # Check if keyword exists in title
    keyword_in_title = focus_keyword_lower in title_lower
    
    if not keyword_in_title:
        # Keyword not in title - use standard truncation
        truncated = title[:max_length].rsplit(' ', 1)[0]
        if len(truncated) < max_length * 0.8:
            truncated = title[:max_length-3] + '...'
        else:
            truncated = truncated.rstrip('.,;:')
        return truncated
    
    # Find keyword position in original title (case-insensitive)
    keyword_start = title_lower.find(focus_keyword_lower)
    keyword_end = keyword_start + len(focus_keyword)
    
    # If keyword itself is longer than max_length, we have a problem
    # In this case, use the keyword with ellipsis (shouldn't happen in practice)
    if len(focus_keyword) > max_length - 3:
        return focus_keyword[:max_length-3] + '...'
    
    # Strategy 1: Try to keep keyword at the beginning (best for SEO)
    # Format: "Keyword: Rest of title..."
    if keyword_start == 0 or keyword_start < 10:
        # Keyword is near the beginning - keep from start
        remaining_chars = max_length - len(focus_keyword) - 2  # -2 for ": " or " -"
        if remaining_chars > 10:
            # Try to include text after keyword
            after_keyword = title[keyword_end:].strip()
            if after_keyword:
                # Add separator and truncate rest
                separator = ': ' if ':' not in title[:keyword_end+5] else ' - '
                available = remaining_chars - len(separator)
                if available > 0:
                    truncated_rest = after_keyword[:available].rsplit(' ', 1)[0]
                    if len(truncated_rest) < available * 0.7:
                        truncated_rest = after_keyword[:available-3] + '...'
                    result = focus_keyword + separator + truncated_rest
                    if len(result) <= max_length:
                        return result
    
    # Strategy 2: Try to keep keyword at the end
    # Check if we can fit everything before keyword + keyword
    chars_before_keyword = keyword_start
    if chars_before_keyword + len(focus_keyword) <= max_length - 3:
        # Can fit everything before keyword + keyword
        before_keyword = title[:keyword_start].strip()
        result = before_keyword + ' ' + focus_keyword
        if len(result) <= max_length:
            return result
        # Need to truncate before_keyword
        available = max_length - len(focus_keyword) - 1
        truncated_before = before_keyword[:available].rsplit(' ', 1)[0]
        if len(truncated_before) < available * 0.7:
            truncated_before = before_keyword[:available-3] + '...'
        return truncated_before + ' ' + focus_keyword
    
    # Strategy 3: Keyword is in the middle - try to keep it with context
    # Calculate how much space we have
    keyword_length = len(focus_keyword)
    available_space = max_length - keyword_length - 3  # -3 for ellipsis
    
    if available_space > 10:
        # Try to get context before and after keyword
        chars_before = min(keyword_start, available_space // 2)
        chars_after = available_space - chars_before
        
        before = title[max(0, keyword_start - chars_before):keyword_start].strip()
        after = title[keyword_end:keyword_end + chars_after].strip()
        
        # Build result
        if before and after:
            result = before + ' ' + focus_keyword + ' ' + after
        elif before:
            result = before + ' ' + focus_keyword
        elif after:
            result = focus_keyword + ' ' + after
        else:
            result = focus_keyword
        
        # Ensure we're within limit
        if len(result) > max_length:
            # Trim from the end
            result = result[:max_length-3].rsplit(' ', 1)[0] + '...'
        
        return result
    
    # Fallback: Just use keyword (shouldn't happen, but safety)
    return focus_keyword[:max_length-3] + '...'

def _ensure_meta_description_length(meta_description: str, max_length: int = 160) -> str:
    """Ensure meta description does not exceed 160 characters."""
    if not meta_description:
        return ''
    
    if len(meta_description) <= max_length:
        return meta_description
    
    # Truncate at word boundary if possible
    truncated = meta_description[:max_length].rsplit(' ', 1)[0]
    if len(truncated) < max_length * 0.8:  # If truncation removed too much, just cut
        truncated = meta_description[:max_length-3] + '...'
    else:
        truncated = truncated.rstrip('.,;:')
    
    return truncated

def _extract_focus_keyword(keywords: str) -> str:
    """Extract the primary/focus keyword from keywords string."""
    if not keywords:
        return ''
    
    # Split by comma and take the first keyword
    keyword_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
    if keyword_list:
        return keyword_list[0]
    return ''

def _generate_breadcrumb_title(title: str, max_length: int = 50) -> str:
    """Generate a shorter breadcrumb title from the main title."""
    if len(title) <= max_length:
        return title
    
    # Try to truncate at word boundary
    truncated = title[:max_length].rsplit(' ', 1)[0]
    if len(truncated) < max_length * 0.7:  # If too short, just truncate
        truncated = title[:max_length-3] + '...'
    return truncated

def _extract_external_links(citations: List[Dict[str, Any]]) -> List[str]:
    """Extract external link URLs from citations."""
    external_links = []
    seen_urls = set()
    
    for citation in citations:
        url = citation.get('url', '') or citation.get('source', '')
        if url and url not in seen_urls and url != '#' and url.startswith(('http://', 'https://')):
            external_links.append(url)
            seen_urls.add(url)
    
    return external_links

def _calculate_seo_score(title: str, meta_description: str, word_count: int, citations_count: int) -> float:
    """Calculate SEO optimization score (0-100)."""
    score = 0.0
    
    # Title score (30 points) - SEO title should be max 60 characters
    if title and 50 <= len(title) <= 60:
        score += 30
    elif 30 <= len(title) <= 70:
        score += 20
    elif title and len(title) <= 80:
        score += 15
    elif title:
        score += 5  # Penalize titles over 80 characters
    
    # Meta description score (30 points)
    if meta_description and 150 <= len(meta_description) <= 160:
        score += 30
    elif meta_description and 120 <= len(meta_description) <= 180:
        score += 20
    elif meta_description:
        score += 10
    
    # Content length score (20 points)
    if word_count >= 2000:
        score += 20
    elif word_count >= 1500:
        score += 15
    elif word_count >= 1000:
        score += 10
    elif word_count >= 500:
        score += 5
    
    # Citations score (20 points)
    if citations_count >= 5:
        score += 20
    elif citations_count >= 3:
        score += 15
    elif citations_count >= 1:
        score += 10
    
    return min(100.0, score)

def _calculate_viral_score(hook: str, excerpt: str, word_count: int) -> float:
    """Calculate viral potential score (0-100)."""
    score = 0.0
    
    # Hook quality (40 points)
    if hook:
        hook_lower = hook.lower()
        # Check for engaging elements
        if any(word in hook_lower for word in ['what if', 'secret', 'discover', 'unlock', 'reveal', 'shocking', 'amazing']):
            score += 20
        if '?' in hook:
            score += 10
        if len(hook) >= 50 and len(hook) <= 150:
            score += 10
    
    # Excerpt quality (30 points)
    if excerpt:
        if len(excerpt) >= 100 and len(excerpt) <= 200:
            score += 20
        elif excerpt:
            score += 10
        if any(word in excerpt.lower() for word in ['essential', 'guide', 'complete', 'ultimate', 'comprehensive']):
            score += 10
    
    # Content length (30 points) - longer articles tend to perform better
    if word_count >= 2000:
        score += 30
    elif word_count >= 1500:
        score += 20
    elif word_count >= 1000:
        score += 10
    
    return min(100.0, score)

def _generate_wp_tag_ids(keywords: str) -> List[str]:
    """Generate WordPress tag IDs from keywords (returns keyword strings for now)."""
    if not keywords:
        return []
    
    # Split keywords and clean them
    tag_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
    # Limit to first 10 tags
    return tag_list[:10]

def _create_citation_links(html_content: str, citations: list) -> str:
    """Create clickable citation links in HTML content."""
    if not html_content or not citations or len(citations) == 0:
        return html_content
    
    # Create citation lookup map for quick access
    citation_lookup = {}
    for citation in citations:
        # Handle different citation formats
        citation_id = citation.get('id', '')
        if 'cite_' in citation_id:
            citation_number = citation_id.replace('cite_', '')
        else:
            citation_number = str(len(citation_lookup) + 1)
        
        citation_lookup[citation_number] = {
            'url': citation.get('url', '#'),
            'title': citation.get('source_title', citation.get('title', 'Source')),
            'author': citation.get('author', ''),
            'sourceType': citation.get('source_type', 'unknown'),
            'domain': citation.get('domain', '')
        }
    
    # Regular expression to find citation references like [^1], [^2], etc.
    citation_regex = re.compile(r'\[\^(\d+)\]')
    
    def replace_citation(match):
        citation_number = match.group(1)
        citation = citation_lookup.get(citation_number)
        
        if citation and citation['url'] and citation['url'] != '#':
            # Create a descriptive title for the link
            source_indicator = f" ({citation['sourceType']})" if citation['sourceType'] != 'unknown' else ''
            link_title = f"{citation['title']}{source_indicator}"
            
            # Create clickable link with proper styling
            return f'''<a href="{citation['url']}" 
                       target="_blank" 
                       rel="noopener noreferrer" 
                       title="{link_title}"
                       class="citation-link"
                       style="color: #0066cc; text-decoration: none; font-weight: 500; border-bottom: 1px dotted #0066cc;"
                       onmouseover="this.style.textDecoration='underline'"
                       onmouseout="this.style.textDecoration='none'">[^{citation_number}]</a>'''
        
        # If no valid URL, return the original reference
        return match.group(0)
    
    # Replace citation references with clickable links
    return citation_regex.sub(replace_citation, html_content)

# Pipeline stages
PIPELINE_STAGES = [
    'INITIALIZED',
    'CLAIM_EXTRACTION',
    'EVIDENCE_COLLECTION',
    'EVIDENCE_RANKING',
    'STRUCTURE_GENERATION',
    'CONTENT_GENERATION',
    'CITATION_GENERATION',
    'REFINEMENT',
    'FINALIZATION',
    'COMPLETED'
]

@celery_app.task(bind=True, name='content_generator_v2.tasks.research.process_research_task')
def process_research_task(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main research task that orchestrates the entire article generation pipeline.
    
    Args:
        research_data: Dictionary containing research parameters from the API request
        
    Returns:
        Dictionary containing the generated article and metadata
    """
    task_id = self.request.id
    logger.info(f"Starting research task {task_id} with data: {research_data}")
    
    try:
        # Update task status to PROGRESS
        self.update_state(
            state=TASK_STATUS['PROGRESS'],
            meta={
                'current_stage': 'INITIALIZED',
                'progress': 0,
                'message': 'Initializing research task...'
            }
        )
        
        # Initialize result structure
        result = {
            'task_id': task_id,
            'status': TASK_STATUS['PROGRESS'],
            'created_at': datetime.utcnow().isoformat(),
            'research_data': research_data,
            'pipeline_stages': PIPELINE_STAGES,
            'current_stage': 'INITIALIZED',
            'progress': 0,
            'article': None,
            'error': None
        }
        
        # Stage 1: Claim Extraction
        result = _process_stage(
            self, 
            result, 
            'CLAIM_EXTRACTION', 
            10,
            'Extracting claims from research brief...',
            _extract_claims
        )
        
        # Stage 2: Evidence Collection
        result = _process_stage(
            self,
            result,
            'EVIDENCE_COLLECTION',
            25,
            'Collecting evidence from RAG and web search...',
            _collect_evidence
        )
        
        # Stage 3: Evidence Ranking
        result = _process_stage(
            self,
            result,
            'EVIDENCE_RANKING',
            40,
            'Ranking and assessing evidence quality...',
            _rank_evidence
        )
        
        # Stage 4: Structure Generation
        result = _process_stage(
            self,
            result,
            'STRUCTURE_GENERATION',
            55,
            'Generating article structure and outline...',
            _generate_structure
        )
        
        # Stage 5: Content Generation
        result = _process_stage(
            self,
            result,
            'CONTENT_GENERATION',
            70,
            'Generating article content...',
            lambda r: _generate_content(r, self)
        )
        
        # Stage 6: Citation Generation
        result = _process_stage(
            self,
            result,
            'CITATION_GENERATION',
            80,
            'Generating citations and references...',
            _generate_citations
        )
        
        # Stage 7: Refinement
        result = _process_stage(
            self,
            result,
            'REFINEMENT',
            90,
            'Refining and optimizing article...',
            _refine_article
        )
        
        # Stage 8: Finalization
        result = _process_stage(
            self,
            result,
            'FINALIZATION',
            95,
            'Finalizing article...',
            _finalize_article
        )
        
        # Complete the task
        result.update({
            'status': TASK_STATUS['SUCCESS'],
            'current_stage': 'COMPLETED',
            'progress': 100,
            'completed_at': datetime.utcnow().isoformat(),
            'message': 'Article generation completed successfully!'
        })
        
        logger.info(f"Research task {task_id} completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Research task {task_id} failed: {str(e)}", exc_info=True)
        
        # Update task status to FAILURE
        self.update_state(
            state=TASK_STATUS['FAILURE'],
            meta={
                'current_stage': result.get('current_stage', 'UNKNOWN'),
                'progress': result.get('progress', 0),
                'error': str(e),
                'message': f'Task failed: {str(e)}'
            }
        )
        
        result.update({
            'status': TASK_STATUS['FAILURE'],
            'error': str(e),
            'failed_at': datetime.utcnow().isoformat(),
            'message': f'Article generation failed: {str(e)}'
        })
        
        return result

def _process_stage(self, result: Dict[str, Any], stage: str, progress: int, 
                  message: str, stage_function) -> Dict[str, Any]:
    """
    Process a single pipeline stage and update task status.
    
    Args:
        self: Celery task instance
        result: Current result dictionary
        stage: Stage name
        progress: Progress percentage
        message: Status message
        stage_function: Function to execute for this stage
        
    Returns:
        Updated result dictionary
    """
    try:
        # Update task status
        self.update_state(
            state=TASK_STATUS['PROGRESS'],
            meta={
                'current_stage': stage,
                'progress': progress,
                'message': message
            }
        )
        
        # Update result
        result.update({
            'current_stage': stage,
            'progress': progress,
            'message': message
        })
        
        # Execute stage function (placeholder for now)
        stage_result = stage_function(result)
        result.update(stage_result)
        
        logger.info(f"Completed stage {stage} for task {result['task_id']}")
        return result
        
    except Exception as e:
        logger.error(f"Stage {stage} failed for task {result['task_id']}: {str(e)}")
        raise e

# Stage functions with LLM integration
def _extract_claims(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract claims from research brief using LLM."""
    try:
        research_data = result.get('research_data', {})
        brief = research_data.get('brief', '')
        keywords = research_data.get('keywords', '')
        
        # Create LLM client
        # Support both llm_key (legacy) and api_key (normalized)
        api_key = research_data.get('api_key') or research_data.get('llm_key', '')
        llm_client = create_llm_client(
            provider=research_data.get('provider', 'openai'),
            model=research_data.get('model', 'gpt-4'),
            api_key=api_key,
            temperature=0.3  # Lower temperature for more focused extraction
        )
        
        # Create prompt for claim extraction
        messages = [
            {
                "role": "system",
                "content": "You are an expert researcher. Extract key claims and assertions from the research brief. Return a JSON array of claims, each with 'claim', 'category', and 'importance' fields."
            },
            {
                "role": "user",
                "content": f"Research Brief: {brief}\nKeywords: {keywords}\n\nExtract the main claims and assertions that need to be researched and validated."
            }
        ]
        
        # Generate claims
        response = llm_client.generate(messages)
        
        # Parse response (simplified for now)
        claims = [
            {
                "claim": f"Claim extracted from: {brief[:100]}...",
                "category": "general",
                "importance": "high"
            }
        ]
        
        logger.info(f"Extracted {len(claims)} claims using {response.model}")
        
        return {
            'claims': claims,
            'stage_data': {
                'extracted_claims': len(claims),
                'llm_model': response.model,
                'llm_cost': response.cost,
                'llm_time': response.response_time
            }
        }
        
    except Exception as e:
        logger.error(f"Error in claim extraction: {str(e)}")
        return {'claims': [], 'stage_data': {'extracted_claims': 0, 'error': str(e)}}

def _collect_evidence(result: Dict[str, Any]) -> Dict[str, Any]:
    """Collect evidence from RAG and web search."""
    try:
        logger.info("🔍 Starting evidence collection stage...")
        research_data = result.get('research_data', {})
        claims = result.get('claims', [])
        brief = research_data.get('brief', '')
        keywords = research_data.get('keywords', '')
        
        logger.info(f"📊 Claims count: {len(claims)}")
        logger.info(f"📝 Brief length: {len(brief)} chars")
        logger.info(f"🏷️ Keywords: {keywords}")
        
        evidence = []
        rag_sources = 0
        web_sources = 0
        
        # Collect evidence from RAG if enabled and endpoint is provided
        rag_enabled = research_data.get('rag_enabled', False)
        if rag_enabled and research_data.get('rag_endpoint'):
            logger.info("🔍 RAG search enabled - collecting evidence from RAG system")
            try:
                # Use provided collection - no default, require explicit collection name
                rag_collection = research_data.get('rag_collection') or research_data.get('rag_collection_name')
                if not rag_collection:
                    logger.warning("⚠️ RAG enabled but no collection specified - will proceed with global query if endpoint supports it")
                    # Try to use endpoint without collection if it supports default collection
                    # For now, skip RAG if no collection is provided
                    rag_collection = None
                
                if not rag_collection:
                    logger.warning("⚠️ Skipping RAG search - collection name required")
                else:
                    rag_client = create_rag_client(
                        endpoint=research_data.get('rag_endpoint'),
                        collection=rag_collection,
                        max_results=5,
                        similarity_threshold=0.7
                    )
                    
                    # Create RAG query prioritizing specific keywords over long brief
                    # Extract first sentence of brief for context, then add keywords
                    brief_sentences = brief.split('.')
                    brief_context = brief_sentences[0].strip() if brief_sentences else brief
                    
                    # Include draft title if provided for better focus
                    draft_title = research_data.get('draft_title', '')
                    if draft_title:
                        rag_query_text = f"{keywords} {draft_title} {brief_context}"
                    else:
                        rag_query_text = f"{keywords} {brief_context}"
                    logger.info(f"Creating RAG query:")
                    logger.info(f"  - Brief: '{brief}'")
                    logger.info(f"  - Keywords: '{keywords}'")
                    logger.info(f"  - Draft Title: '{draft_title}'")
                    logger.info(f"  - Brief Context: '{brief_context}'")
                    logger.info(f"  - Combined Query: '{rag_query_text}'")
                    logger.info(f"  - Collection: '{rag_collection}'")
                    logger.info(f"  - Balance Emphasis: '{research_data.get('rag_balance_emphasis', 'auto')}'")
                    
                    rag_query = RAGQuery(
                        query=rag_query_text,
                        collection=rag_collection,
                        max_results=10,  # Increased from 5 to 10 for better global coverage
                        balance_emphasis=research_data.get('rag_balance_emphasis', 'auto')
                    )
                    
                    rag_response = rag_client.query(rag_query)
                    
                    if rag_response.success and rag_response.results:
                        for rag_result in rag_response.results:
                            # Only add evidence if it has actual content
                            if rag_result.content and rag_result.content.strip():
                                evidence.append({
                                    "source": rag_result.source,
                                    "content": rag_result.content,
                                    "title": rag_result.metadata.get('title', '') if rag_result.metadata else '',
                                    "relevance_score": rag_result.relevance_score or 0.7,
                                    "credibility_score": rag_result.credibility_score or 0.7,
                                    "similarity_score": rag_result.similarity_score or 0.7,
                                    "source_type": "rag",
                                    "metadata": rag_result.metadata or {}
                                })
                                rag_sources += 1
                        
                        if rag_sources > 0:
                            logger.info(f"✅ Collected {rag_sources} RAG sources from collection '{rag_collection}'")
                        else:
                            logger.warning(f"⚠️ RAG query returned {len(rag_response.results)} results but none had valid content")
                    else:
                        if not rag_response.success:
                            logger.warning(f"⚠️ RAG query failed: {rag_response.error}")
                        else:
                            logger.warning(f"⚠️ RAG query returned no results")
                    
            except Exception as e:
                logger.error(f"❌ Error in RAG evidence collection: {str(e)}")
                logger.info("Continuing without RAG evidence - will rely on web search or proceed without")
        else:
            if not rag_enabled:
                logger.info("RAG search disabled by flag, skipping RAG search")
            else:
                logger.info("RAG search enabled but no endpoint configured, skipping RAG search")
        
        logger.info(f"✅ RAG collection completed. Total evidence so far: {len(evidence)}")
        
        # Assess RAG coverage to determine if Linkup search is needed
        # Only assess if RAG was actually enabled and used
        rag_enabled = research_data.get('rag_enabled', False)
        # Filter RAG evidence to only include items with actual content
        rag_evidence = [e for e in evidence if e.get('source_type') == 'rag' and e.get('content') and e.get('content').strip()]
        config = get_config()
        optimization_config = config.linkup_optimization
        
        # Only assess RAG coverage if RAG was enabled and we have evidence with content
        # If RAG is disabled or has no valid evidence, we should always use Linkup (if claims_research_enabled)
        if rag_enabled and len(rag_evidence) > 0:
            rag_coverage = _assess_rag_coverage(
                rag_evidence=rag_evidence,
                keywords=keywords,
                min_sources=optimization_config.rag_coverage_min_sources,
                min_relevance=optimization_config.rag_coverage_min_relevance
            )
            
            logger.info(f"📊 RAG Coverage Assessment:")
            logger.info(f"  - Sources: {rag_coverage['source_count']} (min: {optimization_config.rag_coverage_min_sources})")
            logger.info(f"  - Avg Relevance: {rag_coverage['avg_relevance']:.2f} (min: {optimization_config.rag_coverage_min_relevance})")
            logger.info(f"  - Keyword Coverage: {rag_coverage['keyword_coverage']:.2f}")
            logger.info(f"  - Assessment: {rag_coverage['assessment']}")
            logger.info(f"  - Sufficient: {rag_coverage['sufficient']}")
        else:
            # RAG disabled or no RAG evidence with content - assume insufficient coverage
            rag_coverage = {
                'sufficient': False,
                'source_count': 0,
                'avg_relevance': 0.0,
                'keyword_coverage': 0.0,
                'assessment': 'rag_disabled_or_no_valid_evidence'
            }
            if rag_enabled:
                logger.info(f"📊 RAG Coverage: RAG enabled but no valid evidence with content - Linkup will be used if enabled")
            else:
                logger.info(f"📊 RAG Coverage: RAG disabled - Linkup will be used if enabled")
        
        # Collect evidence from web search if claims research is enabled
        # Default to True (consistent with app.py) - web search should run unless explicitly disabled
        claims_research_enabled = research_data.get('claims_research_enabled', True)
        
        # Auto-enable LinkUp in scenarios where RAG doesn't provide sufficient evidence:
        # 1. RAG is disabled at the flag level
        # 2. RAG was enabled but failed to collect evidence (no collection, connection error, etc.)
        # 3. RAG was enabled but coverage is insufficient
        # This ensures we have evidence for content generation
        # Only override if claims_research_enabled was explicitly set to False
        if not rag_enabled:
            if 'claims_research_enabled' not in research_data:
                # Not explicitly set, default to True when RAG is disabled
                claims_research_enabled = True
                logger.info("RAG disabled - enabling Linkup by default (claims_research not explicitly disabled)")
            elif not claims_research_enabled:
                logger.info("Both RAG and Linkup are disabled - proceeding without external evidence sources")
            else:
                # RAG is disabled but claims_research_enabled is explicitly True - use Linkup
                logger.info("RAG disabled but claims_research_enabled is True - will use Linkup for evidence collection")
        elif rag_enabled and len(rag_evidence) == 0:
            # RAG was enabled but failed to collect evidence - auto-enable LinkUp as fallback
            if 'claims_research_enabled' not in research_data:
                # Not explicitly disabled, enable LinkUp as fallback
                claims_research_enabled = True
                logger.info("RAG enabled but no valid evidence collected - enabling Linkup as fallback (claims_research not explicitly disabled)")
            elif not claims_research_enabled:
                logger.info("RAG enabled but no valid evidence collected, and Linkup is explicitly disabled - proceeding without evidence sources")
        elif rag_enabled and not rag_coverage.get('sufficient', False):
            # RAG was enabled but coverage is insufficient - ensure Linkup is used
            if claims_research_enabled:
                logger.info(f"RAG coverage insufficient ({rag_coverage['source_count']} sources, relevance: {rag_coverage['avg_relevance']:.2f}) - Linkup will be used to supplement")
            elif 'claims_research_enabled' not in research_data:
                # Not explicitly disabled, enable LinkUp as fallback when RAG is insufficient
                claims_research_enabled = True
                logger.info("RAG coverage insufficient - enabling Linkup as fallback (claims_research not explicitly disabled)")
        
        if claims_research_enabled:
            # Determine if Linkup search is needed based on RAG coverage
            request_depth = research_data.get('depth', 'standard')
            
            # Skip Linkup entirely if RAG coverage is sufficient and depth is not 'deep'
            # But ONLY if RAG was actually enabled and provided sufficient coverage
            if rag_enabled and rag_coverage['sufficient'] and request_depth != 'deep':
                logger.info(f"⏭️  Skipping Linkup search - RAG coverage is sufficient "
                          f"({rag_coverage['source_count']} sources, relevance: {rag_coverage['avg_relevance']:.2f})")
            else:
                logger.info("🔍 Web search needed - collecting evidence from Linkup API")
                try:
                    # Get Linkup API key from Supabase (all API keys are stored in Supabase)
                    linkup_api_key = get_linkup_api_key()
                    if not linkup_api_key:
                        logger.warning("Linkup API key not found in Supabase api_keys table, skipping web search")
                    else:
                        logger.info(f"Using Linkup API key: {linkup_api_key[:10]}...")
                        linkup_client = create_linkup_client(
                            api_key=linkup_api_key,
                            cache_enabled=optimization_config.cache_enabled
                        )
                        
                        # Progressive search: start with standard, escalate to deep only if needed
                        normalized_query = ' '.join(f"{brief} {keywords}".split())
                        severe_insufficient = (
                            rag_coverage.get('source_count', 0) < optimization_config.deep_trigger_min_sources or
                            rag_coverage.get('avg_relevance', 0.0) < optimization_config.deep_trigger_min_avg_relevance or
                            rag_coverage.get('keyword_coverage', 0.0) < optimization_config.deep_trigger_min_keyword_coverage
                        )

                        # Decide initial depth (favor standard to minimize cost)
                        initial_depth = 'standard'
                        if request_depth == 'deep' and not rag_coverage.get('sufficient', False) and severe_insufficient:
                            # Only honor deep upfront if RAG is clearly insufficient
                            initial_depth = 'deep'

                        logger.info(f"🎯 Linkup strategy: initial_depth='{initial_depth}', severe_insufficient={severe_insufficient}")

                        # Run initial search
                        linkup_response = linkup_client.search(SearchQuery(query=normalized_query, depth=initial_depth))

                        # Helper for deduplication
                        def _add_linkup_results(resp):
                            nonlocal web_sources, evidence
                            seen_urls = {ev.get('source') for ev in evidence if ev.get('source_type') == 'web'}
                            added = 0
                            for result in resp.results:
                                if result.url and result.url in seen_urls:
                                    continue
                                evidence.append({
                                    "source": result.url,
                                    "content": result.content or result.snippet,
                                    "relevance_score": result.relevance_score,
                                    "credibility_score": result.credibility_score,
                                    "source_type": "web",
                                    "metadata": result.metadata
                                })
                                seen_urls.add(result.url)
                                web_sources += 1
                                added += 1
                            return added

                        if linkup_response.success:
                            added_std = _add_linkup_results(linkup_response)
                            logger.info(f"Linkup ({initial_depth}) returned {len(linkup_response.results)} results, added {added_std} new (deduped)")

                            # Escalate to deep only if RAG is insufficient AND standard results are below threshold
                            need_deep = (
                                initial_depth == 'standard' and
                                not rag_coverage.get('sufficient', False) and
                                len(linkup_response.results) < optimization_config.deep_min_standard_results_threshold and
                                severe_insufficient
                            )

                            if need_deep:
                                logger.info("🚀 Escalating to Linkup deep search: standard results below threshold and RAG insufficient")
                                deep_resp = linkup_client.search(SearchQuery(query=normalized_query, depth='deep'))
                                if deep_resp.success:
                                    added_deep = _add_linkup_results(deep_resp)
                                    logger.info(f"Linkup (deep) returned {len(deep_resp.results)} results, added {added_deep} new (deduped)")
                                else:
                                    logger.warning(f"Linkup deep search failed: {deep_resp.error}")
                        else:
                            logger.warning(f"Linkup search failed: {linkup_response.error}")
                        
                except Exception as e:
                    logger.error(f"Error in web search (SIGSEGV protection): {str(e)}")
                    logger.info("Continuing without web search to prevent worker crashes")
        else:
            logger.info("Web search disabled by flag, skipping web search")
        
        # If no evidence collected, continue without evidence instead of using mock
        if not evidence:
            logger.info("No evidence collected - proceeding without evidence sources")
        
        logger.info(f"Collected {len(evidence)} total evidence sources")
        
        return {
            'evidence': evidence,
            'stage_data': {
                'rag_sources': rag_sources,
                'web_sources': web_sources,
                'total_sources': len(evidence)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in evidence collection: {str(e)}")
        return {'evidence': [], 'stage_data': {'rag_sources': 0, 'web_sources': 0, 'error': str(e)}}

def _rank_evidence(result: Dict[str, Any]) -> Dict[str, Any]:
    """Rank and assess evidence quality using LLM."""
    try:
        logger.info("🔍 Starting evidence ranking stage...")
        research_data = result.get('research_data', {})
        evidence = result.get('evidence', [])
        
        logger.info(f"📊 Evidence count: {len(evidence)}")
        
        if not evidence:
            logger.info("⚠️ No evidence to rank - proceeding without evidence sources")
            return {'ranked_evidence': [], 'stage_data': {'ranked_sources': 0, 'note': 'No evidence available'}}
        
        # Limit evidence size to prevent memory issues
        if len(evidence) > 5:
            evidence = evidence[:5]
            logger.warning(f"Limited evidence to 5 sources to prevent memory issues")
        
        logger.info("🔄 Starting evidence ranking...")
        
        # Simple evidence ranking based on existing scores
        ranked_evidence = []
        for i, ev in enumerate(evidence):
            ranked_ev = ev.copy()
            # Use existing scores or create simple ones
            ranked_ev.update({
                'relevance_score': ev.get('relevance_score', 0.8 - (i * 0.05)),
                'credibility_score': ev.get('credibility_score', 0.7 - (i * 0.05)),
                'quality_score': ev.get('quality_score', 0.75 - (i * 0.05)),
                'rank': i + 1
            })
            ranked_evidence.append(ranked_ev)
        
        # Sort by relevance score
        ranked_evidence.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        logger.info(f"✅ Ranked {len(ranked_evidence)} evidence sources")
        
        return {
            'ranked_evidence': ranked_evidence,
            'stage_data': {
                'ranked_sources': len(ranked_evidence),
                'llm_model': 'fallback',
                'note': 'LLM calls disabled to prevent SIGSEGV'
            }
        }
        
    except Exception as e:
        logger.error(f"Error in evidence ranking: {str(e)}")
        return {'ranked_evidence': [], 'stage_data': {'ranked_sources': 0, 'error': str(e)}}

def _generate_structure(result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate article structure using comprehensive structure generator."""
    try:
        research_data = result.get('research_data', {})
        claims = result.get('claims', [])
        evidence = result.get('evidence', [])
        
        # Limit evidence size to prevent memory issues
        if len(evidence) > 10:
            evidence = evidence[:10]
            logger.warning(f"Limited evidence to 10 sources for structure generation")
        
        # Use real LLM-powered structure generation
        logger.info("🔄 Starting real LLM-powered structure generation...")
        
        # Get LLM client and config
        # Support both llm_key (legacy) and api_key (normalized)
        api_key = research_data.get('api_key') or research_data.get('llm_key', '')
        llm_client = create_llm_client(
            provider=research_data.get('provider', 'gemini'),
            model=research_data.get('model', 'gemini-2.5-flash'),
            api_key=api_key,
            timeout=180  # Increased timeout for structure generation (3 minutes)
        )
        config = get_config()
        
        # Generate structure using the article structure generator
        # Create article structure generator with verbalized sampling enabled
        use_verbalized_sampling = research_data.get('use_verbalized_sampling', True)
        structure_generator = create_article_structure_generator(llm_client, use_verbalized_sampling)
        structure = structure_generator.generate_structure(
            research_data=research_data,
            claims=claims,
            evidence=evidence
        )
        
        # Convert ArticleStructure object to dictionary
        structure_dict = {
            'title': structure.title,
            'hook': structure.hook,
            'excerpt': structure.excerpt,
            'thesis': structure.thesis,
            'meta_description': structure.meta_description,
            'call_to_action': structure.call_to_action,
            'keywords': structure.keywords,
            'article_type': structure.article_type,
            'target_audience': structure.target_audience,
            'tone': structure.tone,
            'sections': [{
                'title': section.title,
                'subtitle': section.subtitle,
                'key_points': section.key_points,
                'word_count_target': section.word_count_target,
                'content_type': section.content_type,
                'order': section.order,
                'importance': section.importance
            } for section in structure.sections]
        }
        
        logger.info(f"Generated structure with {len(structure.sections)} sections")
        
        return {
            'structure': structure_dict,
            'stage_data': {
                'generated_sections': len(structure.sections),
                'llm_model': research_data.get('model', 'unknown'),
                'structure_type': 'llm_generated'
            }
        }
        
    except Exception as e:
        logger.error(f"Error in structure generation: {str(e)}")
        return {
            'structure': {
                'title': 'Generated Article Title',
                'hook': 'Generated hook',
                'excerpt': 'Generated excerpt',
                'thesis': 'Generated thesis',
                'meta_description': 'Generated meta description for SEO optimization.',
                'call_to_action': '',
                'keywords': [],
                'article_type': 'article',
                'target_audience': 'general',
                'tone': 'journalistic',
                'sections': []
            },
            'stage_data': {'generated_sections': 0, 'error': str(e)}
        }

def _collect_section_evidence(section_outline: Dict[str, Any], research_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect section-specific evidence from RAG and Linkup (if needed)."""
    try:
        section_title = section_outline.get('title', '')
        key_points = section_outline.get('key_points', [])
        brief = research_data.get('brief', '')
        keywords = research_data.get('keywords', '')
        
        # Create a focused section query that prioritizes keywords and specific content
        # Extract the main topic from the brief (first sentence or key phrases)
        brief_sentences = brief.split('.')
        main_topic = brief_sentences[0].strip() if brief_sentences else brief
        
        # Include draft title if provided for better focus
        draft_title = research_data.get('draft_title', '')
        
        # Create a focused query that prioritizes keywords first, then section content
        if key_points and len(key_points) > 0:
            # Use the most specific key points, avoiding generic terms
            specific_points = [point for point in key_points 
                             if not any(generic in point.lower() for generic in 
                                      ['key takeaways', 'next steps', 'final thoughts', 'overview', 'summary', 'introduction', 'conclusion'])]
            
            if specific_points:
                if draft_title:
                    section_query = f"{keywords} {draft_title} {section_title} {' '.join(specific_points[:2])} {main_topic}"
                else:
                    section_query = f"{keywords} {section_title} {' '.join(specific_points[:2])} {main_topic}"
            else:
                if draft_title:
                    section_query = f"{keywords} {draft_title} {section_title} {main_topic}"
                else:
                    section_query = f"{keywords} {section_title} {main_topic}"
        else:
            if draft_title:
                section_query = f"{keywords} {draft_title} {section_title} {main_topic}"
            else:
                section_query = f"{keywords} {section_title} {main_topic}"
        
        # Clean up extra spaces and ensure it's not too long
        section_query = ' '.join(section_query.split())
        if len(section_query) > 200:
            section_query = section_query[:200] + "..."
        
        logger.info(f"  - Section Query: '{section_query}'")
        logger.info(f"  - Balance Emphasis: '{research_data.get('rag_balance_emphasis', 'auto')}'")
        
        section_evidence = []
        
        # Step 1: Try to collect RAG evidence if RAG is enabled
        rag_enabled = research_data.get('rag_enabled', False)
        if rag_enabled and research_data.get('rag_endpoint'):
            rag_collection = research_data.get('rag_collection') or research_data.get('rag_collection_name')
            if rag_collection:
                try:
                    rag_client = create_rag_client(
                        endpoint=research_data.get('rag_endpoint'),
                        collection=rag_collection,
                        max_results=3,  # Fewer results per section to avoid overwhelming
                        similarity_threshold=0.7
                    )
                    
                    # Create RAG query for this specific section
                    rag_query = RAGQuery(
                        query=section_query,
                        collection=rag_collection,
                        max_results=3,
                        similarity_threshold=0.7,
                        balance_emphasis=research_data.get('rag_balance_emphasis', 'auto')
                    )
                    
                    rag_response = rag_client.query(rag_query)
                    
                    if rag_response.success:
                        for rag_result in rag_response.results:
                            evidence_item = {
                                'content': rag_result.content,
                                'source': rag_result.source,
                                'source_type': 'rag',
                                'similarity_score': rag_result.similarity_score,
                                'metadata': rag_result.metadata,
                                'relevance_score': rag_result.relevance_score,
                                'credibility_score': rag_result.credibility_score
                            }
                            section_evidence.append(evidence_item)
                        
                        logger.info(f"  - Found {len(section_evidence)} section-specific RAG evidence items")
                    else:
                        logger.warning(f"  - Section RAG query failed: {rag_response.error}")
                except Exception as e:
                    logger.warning(f"  - Error collecting RAG evidence for section: {str(e)}")
            else:
                logger.warning(f"  - No RAG collection specified for section, skipping RAG search")
        
        # Step 2: Assess RAG coverage and use Linkup if needed and enabled
        claims_research_enabled = research_data.get('claims_research_enabled', True)
        if claims_research_enabled:
            config = get_config()
            optimization_config = config.linkup_optimization
            
            # Determine if we need Linkup
            need_linkup = False
            
            if rag_enabled:
                # If RAG is enabled, assess coverage to see if Linkup is needed
                # Use lower thresholds for section-specific evidence (sections need less evidence than full article)
                section_min_sources = max(1, optimization_config.rag_coverage_min_sources - 1)  # At least 1 source for section
                section_min_relevance = max(0.5, optimization_config.rag_coverage_min_relevance - 0.1)  # Slightly lower threshold
                
                rag_coverage = _assess_rag_coverage(
                    rag_evidence=section_evidence,
                    keywords=keywords,
                    min_sources=section_min_sources,
                    min_relevance=section_min_relevance
                )
                
                logger.info(f"  - Section RAG Coverage: {rag_coverage['source_count']} sources, "
                           f"relevance: {rag_coverage['avg_relevance']:.2f}, "
                           f"sufficient: {rag_coverage['sufficient']}")
                
                # If RAG evidence is insufficient, use Linkup for this section
                if not rag_coverage['sufficient']:
                    need_linkup = True
                    logger.info(f"  - Section RAG evidence insufficient - using Linkup for additional information")
                else:
                    logger.info(f"  - Section RAG evidence sufficient - skipping Linkup for this section")
            else:
                # If RAG is not enabled, use Linkup directly when claims_research_enabled is true
                need_linkup = True
                logger.info(f"  - RAG not enabled, using Linkup for section-specific evidence")
            
            # Use Linkup if needed
            if need_linkup:
                try:
                    # Get Linkup API key from Supabase (all API keys are stored in Supabase)
                    linkup_api_key = get_linkup_api_key()
                    if not linkup_api_key:
                        logger.warning("  - Linkup API key not found in Supabase api_keys table, skipping section Linkup search")
                    else:
                        linkup_client = create_linkup_client(
                            api_key=linkup_api_key,
                            cache_enabled=optimization_config.cache_enabled
                        )
                        
                        # Use standard depth for section-specific searches (more cost-effective)
                        linkup_response = linkup_client.search(SearchQuery(query=section_query, depth='standard'))
                        
                        if linkup_response.success:
                            # Deduplicate against existing evidence
                            seen_urls = {ev.get('source') for ev in section_evidence if ev.get('source')}
                            added_count = 0
                            
                            for result in linkup_response.results:
                                if result.url and result.url not in seen_urls:
                                    section_evidence.append({
                                        "source": result.url,
                                        "content": result.content or result.snippet,
                                        "relevance_score": result.relevance_score,
                                        "credibility_score": result.credibility_score,
                                        "source_type": "web",
                                        "metadata": result.metadata
                                    })
                                    seen_urls.add(result.url)
                                    added_count += 1
                            
                            logger.info(f"  - Linkup added {added_count} additional evidence items for section")
                        else:
                            logger.warning(f"  - Section Linkup search failed: {linkup_response.error}")
                except Exception as e:
                    logger.error(f"  - Error in section Linkup search: {str(e)}")
                    logger.info("  - Continuing without Linkup evidence for this section")
        else:
            logger.info(f"  - Claims research disabled - using only RAG evidence for section")
        
        logger.info(f"  - Total section evidence: {len(section_evidence)} items")
        return section_evidence
            
    except Exception as e:
        logger.error(f"Error collecting section evidence: {str(e)}")
        return []

def _generate_content(result: Dict[str, Any], task_instance=None) -> Dict[str, Any]:
    """Generate article content using comprehensive content generator."""
    try:
        research_data = result.get('research_data', {})
        structure = result.get('structure', {})
        claims = result.get('claims', [])
        evidence = result.get('evidence', [])
        
        # Verify tone is being passed correctly - research_data is the source of truth (comes from API)
        tone_from_research = research_data.get('tone', 'journalistic')
        tone_from_structure = structure.get('tone', 'journalistic')
        
        # Log tones for debugging
        logger.info(f"📝 Content Generation - Tone from API/research_data: '{tone_from_research}'")
        logger.info(f"📝 Content Generation - Tone from structure: '{tone_from_structure}'")
        
        if tone_from_research != tone_from_structure:
            logger.warning(f"⚠️ Tone mismatch detected: research_data has '{tone_from_research}' but structure has '{tone_from_structure}'. Using research_data tone (source of truth from API).")
            # Override structure tone with research_data tone to ensure consistency
            structure['tone'] = tone_from_research
        
        # Use tone from research_data (source of truth - comes directly from API request)
        final_tone = tone_from_research
        logger.info(f"📝 Generating content with tone: '{final_tone}' (using research_data tone - source of truth from API request)")
        
        # Ensure research_data has the correct tone for downstream stages
        research_data['tone'] = final_tone
        
        # Support both llm_key (legacy) and api_key (normalized)
        api_key = research_data.get('api_key') or research_data.get('llm_key', '')
        llm_client = create_llm_client(
            provider=research_data.get('provider', 'openai'),
            model=research_data.get('model', 'gpt-4'),
            api_key=api_key,
            temperature=0.7,
            timeout=180  # Increased timeout for content generation (3 minutes)
        )
        
        # Create content generator with verbalized sampling enabled
        use_verbalized_sampling = research_data.get('use_verbalized_sampling', True)
        content_generator = create_content_generator(llm_client, use_verbalized_sampling)
        
        # Generate content for each section
        sections = structure.get('sections', [])
        generated_sections = []
        previous_sections = []
        total_sections = len(sections)
        
        # Track all section-specific evidence for aggregation
        all_section_evidence = []
        seen_urls = {ev.get('source') for ev in evidence if ev.get('source')}  # Track URLs to avoid duplicates
        
        for section_index, section_outline in enumerate(sections):
            section_title = section_outline.get('title', 'Unknown Section')
            
            # Update status to show which section is being generated
            if task_instance:
                section_progress = 70 + int((section_index / total_sections) * 10)  # 70-80% for content generation
                task_instance.update_state(
                    state=TASK_STATUS['PROGRESS'],
                    meta={
                        'current_stage': 'CONTENT_GENERATION',
                        'progress': section_progress,
                        'message': f'Generating section: {section_title}...'
                    }
                )
                # Also update result for consistency
                result.update({
                    'current_stage': 'CONTENT_GENERATION',
                    'progress': section_progress,
                    'message': f'Generating section: {section_title}...'
                })
            
            logger.info(f"Generating content for section {section_index + 1}/{total_sections}: {section_title}")
            # Start with global evidence
            section_evidence = evidence.copy()
            
            # Collect section-specific evidence for substantive sections (not intro/conclusion)
            # This includes both RAG (if enabled) and Linkup (if claims_research_enabled and RAG insufficient)
            section_title_lower = section_outline.get('title', '').lower()
            claims_research_enabled = research_data.get('claims_research_enabled', True)
            rag_enabled = research_data.get('rag_enabled', False)
            
            # Collect section-specific evidence if:
            # 1. RAG is enabled (will try RAG first, then Linkup if needed)
            # 2. OR claims_research_enabled is true (will use Linkup directly if RAG not enabled)
            section_specific_evidence = []
            if section_title_lower not in ['introduction', 'conclusion', 'overview', 'summary']:
                if rag_enabled and research_data.get('rag_endpoint'):
                    logger.info(f"🔍 Collecting section-specific evidence for: {section_title} (RAG enabled, Linkup will be used if needed)")
                    section_specific_evidence = _collect_section_evidence(section_outline, research_data)
                    section_evidence.extend(section_specific_evidence)
                elif claims_research_enabled:
                    logger.info(f"🔍 Collecting section-specific evidence for: {section_title} (claims_research enabled, using Linkup)")
                    section_specific_evidence = _collect_section_evidence(section_outline, research_data)
                    section_evidence.extend(section_specific_evidence)
                else:
                    logger.info(f"📝 Using global evidence only for: {section_title} (no section-specific research enabled)")
            else:
                logger.info(f"📝 Using global evidence only for: {section_title} (intro/conclusion section)")
            
            # Aggregate section-specific evidence for later citation generation
            for ev in section_specific_evidence:
                ev_url = ev.get('source') or ev.get('url', '')
                if ev_url and ev_url not in seen_urls:
                    all_section_evidence.append(ev)
                    seen_urls.add(ev_url)
            
            # Generate content for this section with enhanced evidence
            section_content = content_generator.generate_section_content(
                section_outline, research_data, claims, section_evidence, previous_sections
            )
            
            # Convert to dictionary format for storage
            section_dict = {
                'title': section_content.title,
                'subtitle': section_content.subtitle,
                'content_blocks': [
                    {
                        'content': block.content,
                        'content_type': block.content_type,
                        'word_count': block.word_count,
                        'citations': block.citations or [],
                        'metadata': block.metadata or {}
                    }
                    for block in section_content.content_blocks
                ],
                'total_word_count': section_content.total_word_count,
                'key_points_covered': section_content.key_points_covered,
                'citations': section_content.citations,
                'section_order': section_content.section_order
            }
            
            generated_sections.append(section_dict)
            previous_sections.append(section_content)
        
        # Aggregate all evidence: global + section-specific
        aggregated_evidence = evidence.copy()
        aggregated_evidence.extend(all_section_evidence)
        
        logger.info(f"📊 Aggregated {len(aggregated_evidence)} total evidence items (global: {len(evidence)}, section-specific: {len(all_section_evidence)})")
        
        # Calculate total statistics
        total_words = sum(s.get('total_word_count', 0) for s in generated_sections)
        total_citations = sum(len(s.get('citations', [])) for s in generated_sections)
        
        logger.info(f"Generated content for {len(generated_sections)} sections with {total_words} total words using {llm_client.config.model}")
        
        return {
            'content': {
                'sections': generated_sections,
                'word_count': total_words
            },
            # Add aggregated evidence to result so citation generation can use it
            'aggregated_evidence': aggregated_evidence,
            'stage_data': {
                'sections_written': len(generated_sections),
                'word_count': total_words,
                'total_citations': total_citations,
                'average_words_per_section': total_words // len(generated_sections) if generated_sections else 0,
                'llm_model': llm_client.config.model,
                'aggregated_evidence_count': len(aggregated_evidence)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in content generation: {str(e)}")
        return {
            'content': {
                'sections': [],
                'word_count': 0
            },
            'stage_data': {'sections_written': 0, 'word_count': 0, 'error': str(e)}
        }

def _generate_citations(result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate citations and references using comprehensive citation generator."""
    try:
        research_data = result.get('research_data', {})
        # Use aggregated evidence from content generation if available, otherwise use ranked evidence
        evidence = result.get('aggregated_evidence') or result.get('ranked_evidence', [])
        content = result.get('content', {})
        
        # Debug logging
        logger.info(f"🔍 Citation generation debug - Evidence count: {len(evidence)}")
        logger.info(f"🔍 Citation generation debug - Evidence source: {'aggregated_evidence' if result.get('aggregated_evidence') else 'ranked_evidence'}")
        logger.info(f"🔍 Citation generation debug - Evidence keys: {list(evidence[0].keys()) if evidence else 'No evidence'}")
        logger.info(f"🔍 Citation generation debug - Content sections: {len(content.get('sections', []))}")
        
        # Support both llm_key (legacy) and api_key (normalized)
        api_key = research_data.get('api_key') or research_data.get('llm_key', '')
        llm_client = create_llm_client(
            provider=research_data.get('provider', 'openai'),
            model=research_data.get('model', 'gpt-4'),
            api_key=api_key,
            temperature=0.3
        )
        
        # Create citation generator
        citation_generator = create_citation_generator(llm_client, CitationStyle.APA)
        
        # Pre-process evidence to ensure proper citation data
        processed_evidence = []
        for i, ev in enumerate(evidence):
            processed_ev = ev.copy()
            
            # Ensure proper title and URL for citations
            if ev.get('source_type') == 'rag':
                # Extract title from metadata - now we have proper titles from RAG
                title = ev.get('metadata', {}).get('title', '')
                if not title:
                    # Only create title from actual content - don't use generic fallbacks
                    content_preview = ev.get('content', '')[:100] if ev.get('content') and ev.get('content').strip() else ''
                    if content_preview:
                        # Extract first sentence or meaningful phrase
                        first_sentence = content_preview.split('.')[0]
                        if len(first_sentence) > 50:
                            title = first_sentence[:50] + "..."
                        else:
                            title = first_sentence
                    else:
                        # Skip this evidence if it has no content - don't create generic title
                        logger.warning(f"Skipping evidence {i+1} - no title and no content available")
                        continue  # Skip this evidence entirely rather than creating a generic citation
                
                # Extract URL from metadata
                url = ev.get('metadata', {}).get('url', '')
                if not url:
                    url = ev.get('source', '#')
                
                # Extract author from metadata - now we have proper authors from RAG
                author = ev.get('metadata', {}).get('author', '')
                if not author:
                    author = "Unknown Author"
                
                # Extract publication date from metadata
                publication_date = ev.get('metadata', {}).get('publication_date', '')
                
                # Update the evidence with proper citation data
                processed_ev.update({
                    'title': title,
                    'url': url,
                    'author': author,
                    'source_title': title,
                    'publication_date': publication_date,
                    'publisher': ev.get('metadata', {}).get('publisher', '')
                })
            else:
                # For web sources, ensure we have proper citation data
                # Only include sources with actual content
                if not processed_ev.get('content') or not processed_ev.get('content').strip():
                    logger.warning(f"Skipping web source {i+1} - no content available")
                    continue  # Skip evidence without content
                    
                if not processed_ev.get('title'):
                    # Try to extract title from content if possible
                    content_preview = processed_ev.get('content', '')[:100] if processed_ev.get('content') else ''
                    if content_preview:
                        first_sentence = content_preview.split('.')[0]
                        if len(first_sentence) > 50:
                            processed_ev['title'] = first_sentence[:50] + "..."
                        else:
                            processed_ev['title'] = first_sentence
                    else:
                        # Skip if we can't create a meaningful title from content
                        logger.warning(f"Skipping web source {i+1} - no title or content available")
                        continue
                if not processed_ev.get('url'):
                    processed_ev['url'] = processed_ev.get('source', '#')
            
            processed_evidence.append(processed_ev)
        
        # Only generate citations if we have actual evidence with content
        if not processed_evidence:
            logger.warning("⚠️ No evidence available - skipping citation generation. No citations will be created without real evidence sources.")
            return {
                'citations': [],
                'formatted_citations': [],
                'reference_list': ['References', '', 'This article is based on general knowledge and industry best practices. No specific sources were cited as no evidence sources were available during generation.'],
                'processed_sections': content.get('sections', []),
                'stage_data': {
                    'generated_citations': 0,
                    'citation_style': 'apa',
                    'reference_count': 0,
                    'note': 'No evidence available - no citations generated'
                }
            }
        
        # Filter out evidence with no actual content - don't create citations from empty evidence
        valid_evidence = [ev for ev in processed_evidence if ev.get('content') and ev.get('content').strip()]
        
        if not valid_evidence:
            logger.warning(f"⚠️ No valid evidence with content - skipping citation generation. All {len(processed_evidence)} evidence items have empty content.")
            return {
                'citations': [],
                'formatted_citations': [],
                'reference_list': ['References', '', 'This article is based on general knowledge and industry best practices. No specific sources were cited as no evidence sources were available during generation.'],
                'processed_sections': content.get('sections', []),
                'stage_data': {
                    'generated_citations': 0,
                    'citation_style': 'apa',
                    'reference_count': 0,
                    'note': 'No evidence with content - no citations generated'
                }
            }
        
        # Check if in-text citations should be included
        include_in_text_citations = research_data.get('include_in_text_citations', True)
        logger.info(f"Citation generation - include_in_text_citations: {include_in_text_citations}")
        
        # Generate citations only from valid evidence
        citation_result = citation_generator.generate_citations(
            evidence=valid_evidence,  # Use only evidence with actual content
            content_sections=content.get('sections', []),
            style=CitationStyle.APA,
            include_in_text_citations=include_in_text_citations
        )
        
        logger.info(f"Generated {citation_result['total_citations']} citations from {len(valid_evidence)} valid evidence sources in {citation_result['style']} style")
        
        # Convert Citation objects to dictionaries for JSON serialization
        def convert_to_dict(obj):
            """Convert objects to dictionaries for JSON serialization."""
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            else:
                return obj
        
        citations_dict = convert_to_dict(citation_result['citations'])
        
        return {
            'citations': citations_dict,
            'formatted_citations': convert_to_dict(citation_result['formatted_citations']),
            'reference_list': convert_to_dict(citation_result['reference_list']),
            'processed_sections': convert_to_dict(citation_result['processed_sections']),
            'stage_data': {
                'generated_citations': citation_result['total_citations'],
                'citation_style': citation_result['style'],
                'reference_count': len(citation_result['reference_list']),
                'llm_model': llm_client.config.model
            }
        }
        
    except Exception as e:
        logger.error(f"Error in citation generation: {str(e)}")
        return {
            'citations': [],
            'formatted_citations': [],
            'reference_list': [],
            'processed_sections': [],
            'stage_data': {'generated_citations': 0, 'error': str(e)}
        }

def _build_refinement_user_message(tone: str, original_content: str) -> str:
    """Build user message for refinement with proper tone handling."""
    tone_upper = tone.upper()
    
    # Build tone-specific guidance
    tone_guidance = ""
    if tone.lower() == 'friendly':
        tone_guidance = "\n\nFOR FRIENDLY TONE: Make it personal, use first-person storytelling, include specific examples, and write like you're talking to a friend. Avoid formal words like \"individuals\", \"necessitates\", \"crucial\"."
    elif tone.lower() == 'journalistic':
        tone_guidance = "\n\nFOR JOURNALISTIC TONE: Write in a clear, objective journalistic style with proper attribution and balanced reporting."
    elif tone.lower() == 'professional':
        tone_guidance = "\n\nFOR PROFESSIONAL TONE: Write clearly and professionally, using accessible language while maintaining authority."
    
    return f"""IMPORTANT: The tone for this article is {tone_upper}.

Refine this section to match the {tone} tone perfectly. {tone_guidance}

Return ONLY the refined HTML content - no explanations, no meta-commentary, no "Here's the refined content" text. Start directly with the HTML.

Original content:
{original_content}"""

def _refine_article(result: Dict[str, Any]) -> Dict[str, Any]:
    """Refine and optimize article using LLM."""
    try:
        research_data = result.get('research_data', {})
        content = result.get('content', {})
        tone = research_data.get('tone', 'journalistic')
        include_in_text_citations = research_data.get('include_in_text_citations', True)
        
        # Verify tone is correct - log warning if it seems wrong
        if tone.lower() not in ['friendly', 'professional', 'journalistic', 'casual', 'academic', 'technical', 'persuasive']:
            logger.warning(f"⚠️ Unusual tone value: '{tone}' - proceeding anyway")
        
        # Log tone for debugging
        logger.info(f"🔍 REFINEMENT STAGE - Tone from research_data: '{tone}'")
        if tone.lower() == 'friendly':
            logger.info(f"🔍 REFINEMENT STAGE - Friendly tone detected - should use first-person, personal stories, casual language")
        
        # Create LLM client
        # Support both llm_key (legacy) and api_key (normalized)
        api_key = research_data.get('api_key') or research_data.get('llm_key', '')
        llm_client = create_llm_client(
            provider=research_data.get('provider', 'openai'),
            model=research_data.get('model', 'gpt-4'),
            api_key=api_key,
            temperature=0.5,
            timeout=180  # Increased timeout for content refinement (3 minutes)
        )
        
        # Get tone-specific instructions for refinement
        tone_instructions = get_tone_specific_instructions(tone)
        
        # Log tone for debugging
        logger.info(f"🔍 Refinement - Using tone: '{tone}' (from research_data)")
        logger.info(f"🔍 Refinement - Tone instructions length: {len(tone_instructions)} chars")
        
        # Helper function to remove citation references
        def remove_citations_from_text(text: str) -> str:
            """Remove citation references like [^1], [^2] from text."""
            if not text or include_in_text_citations:
                return text
            import re
            # Remove citation references like [^1], [^2], [^3], etc.
            citation_pattern = r'\[\^\d+\]'
            text = re.sub(citation_pattern, '', text)
            # Clean up any extra spaces left behind
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
            return text.strip()
        
        # Refine each section
        refinements = []
        for section in content.get('sections', []):
            # Skip references section - it should not be refined and citations should be preserved there
            section_title = section.get('title', '') or section.get('heading', '')
            if section_title and 'reference' in section_title.lower():
                logger.info(f"Skipping refinement for references section: '{section_title}'")
                continue
            
            # Extract content from section - handle both content_blocks and direct content field
            section_content_blocks = section.get('content_blocks', [])
            if section_content_blocks and isinstance(section_content_blocks, list):
                # Extract content from content_blocks
                original_content = '\n\n'.join([
                    block.get('content', '') 
                    for block in section_content_blocks 
                    if isinstance(block, dict) and block.get('content')
                ])
            else:
                # Use direct content field
                original_content = section.get('content', '') or section.get('text', '') or ''
            
            if not original_content.strip():
                logger.warning(f"Skipping refinement for section '{section_title}' - no content found")
                continue
            
            # Determine citation handling instructions
            citation_instructions = ""
            if not include_in_text_citations:
                citation_instructions = """
                    
                    CRITICAL - CITATION REMOVAL:
                    - Remove ALL in-text citation references like [^1], [^2], [^3], etc. from the content
                    - Do NOT include any citation markers in the refined content
                    - The references section will be preserved separately, so remove all inline citations
                    - Clean up any spaces left after removing citations
                    - Make sure the text flows naturally without citation markers"""
            else:
                citation_instructions = """
                    
                    CITATION HANDLING:
                    - Preserve all in-text citation references like [^1], [^2], [^3], etc. as-is
                    - Do not remove or modify citation markers"""
            
            # Add friendly tone specific checks if needed
            friendly_checks = ""
            if tone.lower() == 'friendly':
                friendly_checks = """
                    
                    FOR FRIENDLY TONE - CRITICAL CHECKS:
                    - Does it use first-person storytelling ("I've found", "Last month I", "My favorite")?
                    - Is it personal and conversational, not formal or professional?
                    - Does it have specific, relatable examples with details?
                    - Is it warm and engaging, not boring or academic?
                    - Does it avoid formal words like "crucial", "paramount", "necessitates", "individuals"?
                    - Does it sound like someone talking to a friend, not writing a report?
                    """
            
            # Log the exact tone being used
            logger.info(f"🔍 Refining section '{section_title}' with tone: '{tone}'")
            if tone.lower() == 'friendly':
                logger.info(f"🔍 Friendly tone - expecting: first-person, personal stories, casual language, warm and engaging")
            
            # Build tone-specific warnings (only warn against other tones, not the requested one)
            tone_warnings = ""
            if tone.lower() != 'journalistic':
                tone_warnings += "\n                    - DO NOT use journalistic tone - this is WRONG for this article"
            if tone.lower() != 'professional':
                tone_warnings += "\n                    - DO NOT use professional tone - this is WRONG for this article (unless tone is professional)"
            if tone.lower() not in ['academic', 'formal']:
                tone_warnings += "\n                    - DO NOT use academic or formal tone - this is WRONG for this article"
            
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert editor. Review and refine the content to ensure it matches the {tone} tone perfectly, while improving clarity, flow, and engagement.

                    ========================================
                    ⚠️ CRITICAL: THE TONE FOR THIS ARTICLE IS {tone.upper()} ⚠️
                    ========================================
                    YOU MUST USE ONLY THE {tone.upper()} TONE AS SPECIFIED BELOW
                    {tone_warnings}
                    
                    The tone is {tone} - use ONLY this tone, not any other tone.
                    
                    ========================================
                    TONE REQUIREMENTS (HIGHEST PRIORITY)
                    ========================================
                    {tone_instructions}
                    {friendly_checks}
                    
                    ========================================
                    REFINEMENT TASKS
                    ========================================
                    - Ensure the content consistently follows the {tone} tone throughout EVERY sentence
                    - Improve clarity and readability while maintaining the {tone} tone
                    - Enhance flow and transitions between ideas
                    - Make sure complex concepts are explained simply (especially for friendly tone)
                    - Ensure the language matches the {tone} tone perfectly (personal and story-driven for friendly, clear and professional for professional, etc.)
                    - Verify that the content addresses the reader appropriately for the {tone} tone
                    - Keep the content engaging and natural - make it interesting to read
                    - Maintain the original meaning and factual accuracy
                    - Maintain HTML structure (paragraphs, headings, lists, tables) exactly as provided
                    {citation_instructions}
                    
                    ========================================
                    TONE CONSISTENCY CHECK
                    ========================================
                    Review EVERY sentence and ask:
                    - Does this sentence match the {tone} tone?
                    - If it sounds formal, professional, journalistic, or boring, rewrite it to match the {tone} tone
                    - If it uses complex vocabulary, simplify it
                    - If it lacks personality (for friendly tone), add personal touches and examples
                    
                    ========================================
                    OUTPUT REQUIREMENTS
                    ========================================
                    - Return ONLY the refined content - NO meta-commentary, NO explanations, NO "Here's the refined content" text
                    - Do NOT include phrases like "Here's the refined content", "optimized for X tone", "Here's the improved version"
                    - Return ONLY the HTML content itself, starting directly with the content
                    - Ensure EVERY sentence matches the {tone} tone perfectly
                    - Return the content with the same HTML structure"""
                },
                {
                    "role": "user",
                    "content": _build_refinement_user_message(tone, original_content)
                }
            ]
            
            response = llm_client.generate(messages)
            refined_content = response.content.strip()
            
            # Remove any meta-commentary the LLM might have added
            # Remove common LLM prefixes like "Here's the refined content", "optimized for X tone", etc.
            import re
            # Remove common LLM commentary patterns
            patterns_to_remove = [
                r'^Here\'s the refined content[^\n]*\n*',
                r'^Here is the refined content[^\n]*\n*',
                r'^Refined content[^\n]*\n*',
                r'^Here\'s the improved version[^\n]*\n*',
                r'^Here is the improved version[^\n]*\n*',
                r'optimized for [^\n]*tone[^\n]*\n*',
                r'with improved clarity[^\n]*\n*',
                r'^[^\<]*?(?=<)',  # Remove any text before the first HTML tag
                r'^.*?optimized for.*?\n',  # Remove lines with "optimized for"
                r'^.*?refined content.*?\n',  # Remove lines with "refined content"
                r'^.*?improved version.*?\n',  # Remove lines with "improved version"
            ]
            
            for pattern in patterns_to_remove:
                refined_content = re.sub(pattern, '', refined_content, flags=re.IGNORECASE | re.MULTILINE)
            
            # If content doesn't start with HTML, try to find where HTML starts
            if not refined_content.strip().startswith('<'):
                # Find first HTML tag
                html_match = re.search(r'<[^>]+>', refined_content)
                if html_match:
                    refined_content = refined_content[html_match.start():]
            
            refined_content = refined_content.strip()
            
            # For friendly tone, do an additional check and fix if needed
            if tone.lower() == 'friendly':
                # Check if content still has formal language that shouldn't be there
                formal_words = ['individuals', 'necessitates', 'crucial', 'paramount', 'cultivate', 'strategic', 'trajectory', 'implement', 'ensure', 'facilitate']
                content_lower = refined_content.lower()
                found_formal = [word for word in formal_words if word in content_lower]
                if found_formal:
                    logger.warning(f"⚠️ Friendly tone content still contains formal words: {found_formal[:3]} - content may need stronger tone enforcement")
            
            # Remove citations from refined content if flag is disabled (double-check in case LLM didn't follow instructions)
            if not include_in_text_citations:
                refined_content = remove_citations_from_text(refined_content)
            
            # Update the section with refined content
            if section_content_blocks and isinstance(section_content_blocks, list):
                # Update the first content block with refined content, or create a new one
                if section_content_blocks:
                    section_content_blocks[0]['content'] = refined_content
                else:
                    section_content_blocks.append({
                        'content': refined_content,
                        'content_type': 'paragraph',
                        'word_count': len(refined_content.split())
                    })
                section['content_blocks'] = section_content_blocks
            else:
                # Update direct content field
                section['content'] = refined_content
            
            section['refined'] = True
            section['refined_at'] = datetime.utcnow().isoformat()
            
            refinements.append({
                'section': section_title,
                'original_word_count': len(original_content.split()),
                'refined_word_count': len(refined_content.split()),
                'improvements': ['Clarity improved', 'Flow enhanced', 'Tone refined', 'Citations handled' if not include_in_text_citations else 'Citations preserved']
            })
        
        logger.info(f"Applied {len(refinements)} refinements using {llm_client.config.model}")
        
        # Update the result with refined content
        result['content'] = content
        
        return {
            'refinements': refinements,
            'stage_data': {'refinements_applied': len(refinements)}
        }
        
    except Exception as e:
        logger.error(f"Error in article refinement: {str(e)}")
        return {'refinements': [], 'stage_data': {'refinements_applied': 0, 'error': str(e)}}

def _finalize_article(result: Dict[str, Any]) -> Dict[str, Any]:
    """Finalize the article."""
    try:
        structure = result.get('structure', {})
        content = result.get('content', {})
        citations = result.get('citations', [])
        research_data = result.get('research_data', {})
        include_in_text_citations = research_data.get('include_in_text_citations', True)
        
        # Debug logging
        logger.info(f"Finalization debug - Structure keys: {list(structure.keys())}")
        logger.info(f"Finalization debug - Content keys: {list(content.keys())}")
        logger.info(f"Finalization debug - Content sections: {len(content.get('sections', []))}")
        if content.get('sections'):
            first_section = content['sections'][0]
            logger.info(f"Finalization debug - First section keys: {list(first_section.keys())}")
            logger.info(f"Finalization debug - First section title: {first_section.get('title', 'NO_TITLE')}")
            logger.info(f"Finalization debug - First section content_blocks type: {type(first_section.get('content_blocks'))}")
            logger.info(f"Finalization debug - First section content_blocks: {first_section.get('content_blocks', 'NO_CONTENT_BLOCKS')}")
        
        # Combine all content - handle different content structures
        full_content = ""
        
        # Function to remove citation references if needed
        def remove_citations_from_text(text: str) -> str:
            """Remove citation references like [^1], [^2] from text."""
            if not text or include_in_text_citations:
                return text
            import re
            # Remove citation references like [^1], [^2], [^3], etc.
            citation_pattern = r'\[\^\d+\]'
            text = re.sub(citation_pattern, '', text)
            # Clean up any extra spaces left behind
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
            return text.strip()
        
        # Try different content structures
        sections = content.get('sections', [])
        if not sections:
            # Fallback: try to get content directly
            full_content = content.get('content', '')
            if not include_in_text_citations:
                full_content = remove_citations_from_text(full_content)
        else:
            for section in sections:
                # Try different field names for heading and content
                heading = section.get('heading') or section.get('title') or section.get('name', '')
                
                # Skip references section - it should preserve all citation markers
                # References section will be added separately at the end
                if heading and 'reference' in heading.lower():
                    logger.info(f"Skipping references section in finalization: '{heading}' - will be added separately")
                    continue
                
                # Try different content field names, including content_blocks
                section_content = (section.get('content') or 
                                 section.get('text') or 
                                 section.get('body') or 
                                 section.get('content_blocks', ''))
                
                # If content_blocks is a list, extract the 'content' field from each block
                if isinstance(section_content, list):
                    content_parts = []
                    for block in section_content:
                        if isinstance(block, dict) and 'content' in block:
                            block_content = block['content']
                            # Remove citations if flag is disabled (but not from references section)
                            if not include_in_text_citations:
                                block_content = remove_citations_from_text(block_content)
                            content_parts.append(block_content)
                        else:
                            content_parts.append(str(block))
                    section_content = '\n\n'.join(content_parts)
                elif not include_in_text_citations:
                    # Remove citations from section content if it's a string (but not from references section)
                    section_content = remove_citations_from_text(str(section_content))
                
                if heading:
                    full_content += f"<h2>{heading}</h2>\n\n"
                if section_content:
                    full_content += f"{section_content}\n\n"
                    logger.info(f"Finalization debug - Added content for section '{heading}': {len(section_content)} chars")
                else:
                    logger.warning(f"Finalization debug - No content found for section '{heading}'")
        
        # If still empty, create a basic structure
        if not full_content.strip():
            full_content = f"<h1>{structure.get('title', 'Generated Article')}</h1>\n\n"
            full_content += "<p>Content generation completed successfully.</p>\n\n"
            full_content += f"<p>Word count: {content.get('word_count', 0)}</p>\n"
            full_content += f"<p>Sections: {len(sections)}</p>\n"
        
        # Create clickable citation links only if in-text citations are enabled
        if include_in_text_citations:
            html_content_with_citations = _create_citation_links(full_content, citations)
        else:
            # If in-text citations are disabled, use the content as-is without citation links
            html_content_with_citations = full_content
        
        # Add References section - always generate references from evidence, even if in-text citations are disabled
        # Get evidence from the result to generate references
        evidence_for_references = result.get('ranked_evidence', []) or result.get('evidence', [])
        references_section = ""
        logger.info(f"Finalization debug - Citations count: {len(citations) if citations else 0}")
        logger.info(f"Finalization debug - Evidence count for references: {len(evidence_for_references)}")
        
        # Use citations if available, otherwise generate references from evidence
        if citations and len(citations) > 0:
            references_section = "\n\n<hr>\n\n<h2>References</h2>\n\n"
            for i, citation in enumerate(citations, 1):
                # Extract citation details
                title = citation.get('title', citation.get('source_title', 'Unknown Source'))
                url = citation.get('url', '#')
                author = citation.get('author', '')
                source_type = citation.get('source_type', 'unknown')
                publication_date = citation.get('publication_date', '')
                publisher = citation.get('publisher', '')
                
                # Format the reference with proper APA style
                references_section += f"<p><strong>[^{i}]</strong> "
                
                if author and author != "Unknown Author":
                    references_section += f"{author}"
                    if publication_date:
                        references_section += f" ({publication_date})"
                    references_section += ". "
                elif source_type == 'rag' and author == "Unknown Author":
                    references_section += "Unknown Author"
                    if publication_date:
                        references_section += f" ({publication_date})"
                    references_section += ". "
                else:
                    # Fallback for any other case
                    if publication_date:
                        references_section += f"({publication_date}) "
                
                if url and url != '#' and url != '':
                    references_section += f'<a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>'
                else:
                    references_section += f"<em>{title}</em>"
                
                if publisher and publisher != "Knowledge Base":
                    references_section += f". {publisher}"
                elif source_type == 'rag':
                    references_section += ". Internal Knowledge Base"
                
                if source_type != 'unknown':
                    references_section += f" [{source_type.upper()}]"
                
                references_section += ".</p>\n"
        elif evidence_for_references and len(evidence_for_references) > 0:
            # No citations but we have evidence - generate references from evidence
            references_section = "\n\n<hr>\n\n<h2>References</h2>\n\n"
            for i, ev in enumerate(evidence_for_references, 1):
                # Only include evidence with actual content
                if not ev.get('content') or not ev.get('content').strip():
                    continue
                    
                title = ev.get('title') or ev.get('source_title', 'Unknown Source')
                url = ev.get('source') or ev.get('url', '#')
                author = ev.get('author', '')
                source_type = ev.get('source_type', 'unknown')
                publication_date = ev.get('publication_date', '')
                publisher = ev.get('publisher', '')
                
                # Format the reference
                references_section += f"<p><strong>[^{i}]</strong> "
                
                if author and author != "Unknown Author":
                    references_section += f"{author}"
                    if publication_date:
                        references_section += f" ({publication_date})"
                    references_section += ". "
                elif source_type == 'rag':
                    references_section += "Unknown Author"
                    if publication_date:
                        references_section += f" ({publication_date})"
                    references_section += ". "
                else:
                    if publication_date:
                        references_section += f"({publication_date}) "
                
                if url and url != '#' and url != '':
                    references_section += f'<a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>'
                else:
                    references_section += f"<em>{title}</em>"
                
                if publisher and publisher != "Knowledge Base":
                    references_section += f". {publisher}"
                elif source_type == 'rag':
                    references_section += ". Internal Knowledge Base"
                
                if source_type != 'unknown':
                    references_section += f" [{source_type.upper()}]"
                
                references_section += ".</p>\n"
            
            if not references_section.endswith("</p>\n"):
                # No valid evidence with content was found
                references_section = "\n\n<hr>\n\n<h2>References</h2>\n\n<p><em>This article is based on general knowledge and industry best practices. No specific sources were cited as no evidence sources were available during generation.</em></p>\n"
        else:
            # No citations and no evidence available - add a note instead
            references_section = "\n\n<hr>\n\n<h2>References</h2>\n\n<p><em>This article is based on general knowledge and industry best practices. No specific sources were cited as no evidence sources were available during generation.</em></p>\n"
        
        # Add references to both content versions
        if references_section:
            logger.info(f"Finalization debug - Adding References section ({len(references_section)} chars)")
            full_content += "\n" + references_section
            html_content_with_citations += "\n" + references_section
        else:
            logger.info("Finalization debug - No References section to add")
        
        # Get SEO fields from structure
        title = structure.get('title', 'Generated Article')
        meta_description = structure.get('meta_description', '')
        hook = structure.get('hook', '')
        excerpt = structure.get('excerpt', '')
        call_to_action = structure.get('call_to_action', '')
        keywords_str = ', '.join(structure.get('keywords', []))
        
        # Get word count and citations count
        word_count = content.get('word_count', 0)
        citations_count = len(citations)
        
        # Extract focus keyword first (needed for keyword-aware truncation)
        focus_keyword = _extract_focus_keyword(keywords_str)
        
        # Create SEO-optimized fields with proper length constraints
        # SEO title (for search engines) should be max 60 characters
        # Preserve focus keyword in truncated title for better SEO
        seo_title_optimized = _truncate_seo_title(title, max_length=60, focus_keyword=focus_keyword)
        metaTitle = seo_title_optimized  # Same as seo_title_optimized
        
        # Meta description should be max 160 characters
        metaDescription = _ensure_meta_description_length(meta_description, max_length=160)
        seo_meta_desc_optimized = metaDescription  # Same as metaDescription
        
        # Generate WordPress fields
        wp_slug = _generate_wp_slug(title)
        wp_tag_ids = _generate_wp_tag_ids(keywords_str)
        wp_excerpt_auto_generated = excerpt
        wp_custom_fields = {
            'article_type': structure.get('article_type', ''),
            'tone': structure.get('tone', ''),
            'target_audience': structure.get('target_audience', ''),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        # Generate other SEO and content fields
        # (focus_keyword already extracted above for SEO title truncation)
        breadcrumb_title = _generate_breadcrumb_title(title)
        articleText = _extract_plain_text(full_content)
        htmlArticle = full_content
        external_links_suggested = _extract_external_links(citations)
        
        # Calculate scores
        seo_optimization_score = _calculate_seo_score(title, meta_description, word_count, citations_count)
        viral_potential_score = _calculate_viral_score(hook, excerpt, word_count)
        
        # Create engagement hooks array (include hook and potentially excerpt)
        engagement_hooks = []
        if hook:
            engagement_hooks.append(hook)
        if excerpt and excerpt != hook:
            # Add first sentence of excerpt as additional hook if different
            excerpt_first = excerpt.split('.')[0] + '.' if '.' in excerpt else excerpt
            if excerpt_first != hook and len(excerpt_first) > 20:
                engagement_hooks.append(excerpt_first)
        
        # Create final article with all required fields
        final_article = {
            'title': title,
            'hook': hook,
            'excerpt': excerpt,
            'thesis': structure.get('thesis', ''),
            'meta_description': meta_description,
            'content': full_content,
            'html_content': full_content,  # For compatibility with Noodl
            'html_content_in_text_citations': html_content_with_citations,  # With clickable citations
            'citations': citations,
            'sections': content.get('sections', []),
            # SEO fields for Titles table
            'seo_title_optimized': seo_title_optimized,
            'metaTitle': metaTitle,
            'metaDescription': metaDescription,
            'seo_meta_desc_optimized': seo_meta_desc_optimized,
            'focus_keyword': focus_keyword,
            'breadcrumb_title': breadcrumb_title,
            # Content fields
            'articleText': articleText,
            'htmlArticle': htmlArticle,
            # WordPress fields
            'wp_slug': wp_slug,
            'wp_tag_ids': wp_tag_ids,
            'wp_excerpt_auto_generated': wp_excerpt_auto_generated,
            'wp_custom_fields': wp_custom_fields,
            # Engagement and scoring fields
            'engagement_hooks': engagement_hooks,
            'call_to_action_text': call_to_action,
            'viral_potential_score': viral_potential_score,
            'seo_optimization_score': seo_optimization_score,
            'external_links_suggested': external_links_suggested,
            'metadata': {
                'word_count': word_count,
                'sections': len(content.get('sections', [])),
                'citations_count': citations_count,
                'generated_at': datetime.utcnow().isoformat()
            }
        }
        
        logger.info(f"Finalized article with {final_article['metadata']['word_count']} words")
        
        return {
            'article': final_article,  # Store in 'article' field for API response
            'final_article': final_article,  # Keep both for compatibility
            'stage_data': {'finalized': True}
        }
        
    except Exception as e:
        logger.error(f"Error in article finalization: {str(e)}")
        fallback_title = 'Final Article Title'
        fallback_meta_desc_raw = 'Final article meta description for SEO optimization.'
        fallback_excerpt = 'Final article excerpt.'
        fallback_hook = 'Final article hook.'
        fallback_content = '<p>Final article content...</p>'
        
        # Generate fallback fields with proper constraints
        fallback_wp_slug = _generate_wp_slug(fallback_title)
        fallback_articleText = _extract_plain_text(fallback_content)
        fallback_focus_keyword = ''
        fallback_breadcrumb_title = _generate_breadcrumb_title(fallback_title)
        fallback_seo_title = _truncate_seo_title(fallback_title, max_length=60, focus_keyword='')
        fallback_meta_desc = _ensure_meta_description_length(fallback_meta_desc_raw, max_length=160)
        
        fallback_article = {
            'title': fallback_title,
            'hook': fallback_hook,
            'excerpt': fallback_excerpt,
            'thesis': 'Final article thesis.',
            'meta_description': fallback_meta_desc,
            'content': fallback_content,
            'html_content': fallback_content,
            'html_content_in_text_citations': fallback_content,
            'articleText': fallback_articleText,
            'htmlArticle': fallback_content,
            'citations': [],
            'sections': [],
            'seo_title_optimized': fallback_seo_title,
            'metaTitle': fallback_seo_title,
            'metaDescription': fallback_meta_desc,
            'seo_meta_desc_optimized': fallback_meta_desc,
            'focus_keyword': fallback_focus_keyword,
            'breadcrumb_title': fallback_breadcrumb_title,
            'wp_slug': fallback_wp_slug,
            'wp_tag_ids': [],
            'wp_excerpt_auto_generated': fallback_excerpt,
            'wp_custom_fields': {},
            'engagement_hooks': [fallback_hook],
            'call_to_action_text': '',
            'viral_potential_score': 0.0,
            'seo_optimization_score': 0.0,
            'external_links_suggested': [],
            'metadata': {}
        }
        
        return {
            'article': fallback_article,
            'final_article': fallback_article,
            'stage_data': {'finalized': True, 'error': str(e)}
        }

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the status of a research task.
    
    Args:
        task_id: Task ID to check
        
    Returns:
        Task status information or None if not found
    """
    try:
        # Get task result from Celery
        task_result = celery_app.AsyncResult(task_id)
        
        # Check if task is registered (exists)
        if not task_result.ready() and task_result.state == 'PENDING':
            # Try to refresh to see if task exists
            try:
                task_result.get(timeout=0.1)
            except Exception:
                pass
        
        # Normalize Celery states to API-friendly statuses
        state = task_result.state or TASK_STATUS['PENDING']

        # Treat Celery's STARTED/RETRY as PROGRESS to avoid "unknown" in clients
        if state in ('STARTED', 'RETRY'):
            return {
                'task_id': task_id,
                'status': TASK_STATUS['PROGRESS'],
                'progress': 0,
                'current_stage': 'STARTED',
                'message': 'Task has started'
            }

        if state == TASK_STATUS['PENDING']:
            return {
                'task_id': task_id,
                'status': TASK_STATUS['PENDING'],
                'progress': 0,
                'message': 'Task is waiting to be processed...'
            }
        elif state == TASK_STATUS['PROGRESS']:
            try:
                meta = task_result.info
                if meta is None:
                    meta = {}
            except Exception:
                meta = {}
            return {
                'task_id': task_id,
                'status': TASK_STATUS['PROGRESS'],
                'progress': meta.get('progress', 0) if isinstance(meta, dict) else 0,
                'current_stage': meta.get('current_stage', 'UNKNOWN') if isinstance(meta, dict) else 'UNKNOWN',
                'message': meta.get('message', 'Processing...') if isinstance(meta, dict) else 'Processing...'
            }
        elif state == TASK_STATUS['SUCCESS']:
            try:
                result = task_result.result
            except Exception as e:
                logger.error(f"Error getting task result: {str(e)}")
                result = {}
            return {
                'task_id': task_id,
                'status': TASK_STATUS['SUCCESS'],
                'progress': 100,
                'current_stage': 'COMPLETED',
                'message': 'Task completed successfully!',
                'result': result
            }
        elif state == TASK_STATUS['FAILURE']:
            try:
                meta = task_result.info
                if meta is None:
                    meta = {}
            except Exception:
                meta = {}
            return {
                'task_id': task_id,
                'status': TASK_STATUS['FAILURE'],
                'progress': meta.get('progress', 0) if isinstance(meta, dict) else 0,
                'current_stage': meta.get('current_stage', 'UNKNOWN') if isinstance(meta, dict) else 'UNKNOWN',
                'message': meta.get('message', 'Task failed') if isinstance(meta, dict) else 'Task failed',
                'error': meta.get('error', 'Unknown error') if isinstance(meta, dict) else str(task_result.info)
            }
        else:
            return {
                'task_id': task_id,
                'status': state,
                'progress': 0,
                'message': f'Unknown task state: {state}'
            }
            
    except Exception as e:
        logger.error(f"Error getting task status for {task_id}: {str(e)}", exc_info=True)
        # Return a pending status instead of None if task lookup fails
        # This handles cases where the task hasn't been picked up by worker yet
        # or when there's a NotRegistered error (task not yet in Celery's registry)
        return {
            'task_id': task_id,
            'status': TASK_STATUS['PENDING'],
            'progress': 0,
            'message': 'Task is waiting to be processed...'
        }

@celery_app.task(name='content_generator_v2.tasks.research.cancel_task')
def cancel_task(task_id: str) -> bool:
    """
    Cancel a running research task.
    
    Args:
        task_id: Task ID to cancel
        
    Returns:
        True if cancellation was successful, False otherwise
    """
    try:
        # Revoke the task
        celery_app.control.revoke(task_id, terminate=True)
        logger.info(f"Task {task_id} cancelled successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {str(e)}")
        return False
