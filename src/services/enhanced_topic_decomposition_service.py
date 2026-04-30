"""
Enhanced topic decomposition service
Combines Google Autocomplete with LLM processing for optimal topic research
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from uuid import uuid4
import re

from ..core.models.enhanced_subtopic import EnhancedSubtopic, SubtopicSource
from ..core.models.autocomplete_result import AutocompleteResult
from ..core.models.method_comparison import MethodComparison, MethodResult
from ..core.models.comparison_metrics import ComparisonMetrics
from ..core.models.search_volume_indicator import SearchVolumeIndicator, IndicatorType
from ..integrations.google_autocomplete import GoogleAutocompleteService
from .llm.llm_service import llm_service
from .topic_brief_builder_service import topic_brief_builder_service
from .editorial_subtopic_service import editorial_subtopic_service
from .subtopic_scoring_service import subtopic_scoring_service

logger = logging.getLogger(__name__)

EDITORIAL_DECOMPOSITION_TIMEOUT_SECONDS = 45.0
FALLBACK_LLM_TIMEOUT_SECONDS = 45.0
AUTOCOMPLETE_TIMEOUT_SECONDS = 20.0

class EnhancedTopicDecompositionService:
    """
    Service for enhanced topic decomposition using Google Autocomplete + LLM
    
    Features:
    - Hybrid approach combining autocomplete and LLM
    - Method comparison and analysis
    - Relevance scoring and ranking
    - Fallback mechanisms
    - Performance optimization
    """
    
    def __init__(self, 
                 google_autocomplete_service: Optional[GoogleAutocompleteService] = None):
        """
        Initialize enhanced topic decomposition service
        
        Args:
            google_autocomplete_service: Google Autocomplete service instance
        """
        self.google_autocomplete_service = google_autocomplete_service or GoogleAutocompleteService()
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 3600  # 1 hour in seconds
    
    async def decompose_topic_enhanced(self, 
                                     query: str,
                                     user_id: str,
                                     max_subtopics: int = 6,
                                     use_autocomplete: bool = True,
                                     use_llm: bool = True,
                                     decomposition_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Decompose topic using enhanced approach (autocomplete + LLM)
        
        Args:
            query: Topic to decompose
            user_id: User identifier
            max_subtopics: Maximum number of subtopics to return
            use_autocomplete: Whether to use Google Autocomplete
            use_llm: Whether to use LLM processing
            
        Returns:
            Dictionary with decomposition results
        """
        start_time = time.time()
        editorial_debug: Dict[str, Any] = {}
        
        async def _build_fallback_result(reason: Exception) -> Dict[str, Any]:
            logger.warning(
                "Entering fallback decomposition query=%r reason=%s",
                query,
                reason,
            )
            fallback_started = time.perf_counter()
            fallback = await self._run_hybrid_method(query, max_subtopics)
            fallback_titles = fallback.subtopics or []
            if not fallback_titles:
                llm_fallback = await self._run_llm_only_method(query, max_subtopics)
                fallback_titles = llm_fallback.subtopics or []
            logger.info(
                "Fallback decomposition completed query=%r subtopic_count=%s elapsed_ms=%.1f",
                query,
                len(fallback_titles),
                (time.perf_counter() - fallback_started) * 1000,
            )

            if not fallback_titles:
                logger.warning(
                    "Fallback decomposition produced no usable subtopics query=%r reason=%s",
                    query,
                    reason,
                )
                return {
                    "success": False,
                    "message": "Subtopic generation could not produce distinct results for this topic with the current LLM output.",
                    "original_query": query,
                    "decomposition_context": decomposition_context or {},
                    "subtopics": [],
                    "autocomplete_data": None,
                    "processing_time": time.time() - start_time,
                    "enhancement_methods": ["fallback_hybrid_or_llm"],
                    "warnings": [str(reason), "No fallback subtopics were produced."],
                    "debug": {
                        "editorial": editorial_debug,
                    },
                }

            fallback_subtopics = []
            for title in fallback_titles[:max_subtopics]:
                title_text = (title or "").strip()
                if not title_text:
                    continue
                seed_tokens = [token for token in re.split(r"[^a-zA-Z0-9]+", title_text.lower()) if len(token) > 2][:4]
                fallback_subtopics.append({
                    "id": str(uuid4()),
                    "title": title_text,
                    "search_volume_indicators": ["Fallback decomposition"],
                    "autocomplete_suggestions": [],
                    "relevance_score": 0.7,
                    "source": "llm",
                    "rationale": "Generated via fallback decomposition path.",
                    "seed_keywords": seed_tokens or [title_text.lower()],
                    "target_audience": "General Audience",
                    "search_volume": 0,
                    "cpc": 0.0,
                    "keyword_difficulty": 0,
                    "trend_analysis": None,
                    "monetization_data": None,
                    "intent_bucket": (decomposition_context or {}).get("intent_bucket"),
                    "decision_focus": (decomposition_context or {}).get("decision_focus"),
                    "angle_question": (decomposition_context or {}).get("angle_question"),
                    "value_layer_tags": (decomposition_context or {}).get("value_layer_tags") or [],
                    "cluster_type": "fallback",
                    "primary_user_outcome": "Explore adjacent content opportunities",
                    "serp_intent_match": "medium",
                    "tool_potential_score": 0,
                })

            if not fallback_subtopics:
                logger.warning(
                    "Fallback titles existed but normalization removed all subtopics query=%r",
                    query,
                )
                return {
                    "success": False,
                    "message": "Subtopic generation returned only unusable fallback results.",
                    "original_query": query,
                    "decomposition_context": decomposition_context or {},
                    "subtopics": [],
                    "autocomplete_data": None,
                    "processing_time": time.time() - start_time,
                    "enhancement_methods": ["fallback_hybrid_or_llm"],
                    "warnings": [str(reason), "Fallback titles normalized to zero usable subtopics."],
                    "debug": {
                        "editorial": editorial_debug,
                    },
                }

            return {
                "success": True,
                "message": f"Fallback decomposition produced {len(fallback_subtopics)} subtopics.",
                "original_query": query,
                "decomposition_context": decomposition_context or {},
                "subtopics": fallback_subtopics,
                "autocomplete_data": None,
                "processing_time": time.time() - start_time,
                "enhancement_methods": ["fallback_hybrid_or_llm"],
                "warnings": [str(reason)],
                "debug": {
                    "editorial": editorial_debug,
                },
            }

        try:
            # Validate inputs
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            query = query.strip()
            
            # Check cache
            cache_key = f"{user_id}:{query}:{max_subtopics}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Returning cached result for query: {query}")
                return cached_result
            
            # Subtopics-first flow (DataForSEO-free at subtopic stage):
            # 1) build editorial subtopics, 2) score editorial evidence only.
            logger.info(
                "Enhanced decomposition started query=%r user_id=%s max_subtopics=%s context_keys=%s",
                query,
                user_id,
                max_subtopics,
                sorted((decomposition_context or {}).keys()),
            )
            brief = topic_brief_builder_service.build(
                topic={"title": query, **(decomposition_context or {})},
                project={},
                decomposition_context=decomposition_context or {},
            )
            editorial_started = time.perf_counter()
            editorial_result = await asyncio.wait_for(
                editorial_subtopic_service.generate_with_debug(
                    brief=brief,
                    max_subtopics=max_subtopics,
                ),
                timeout=EDITORIAL_DECOMPOSITION_TIMEOUT_SECONDS,
            )
            editorial_subtopics = editorial_result.get("subtopics") or []
            editorial_debug = editorial_result.get("debug") or {}
            logger.info(
                "Editorial decomposition stage finished query=%r subtopic_count=%s elapsed_ms=%.1f",
                query,
                len(editorial_subtopics),
                (time.perf_counter() - editorial_started) * 1000,
            )

            enhanced_subtopics: List[EnhancedSubtopic] = []
            for item in editorial_subtopics[:max_subtopics]:
                selected_keywords: List[Dict[str, Any]] = []
                score = subtopic_scoring_service.score(item, selected_keywords)

                keyword_strings = [k.strip() for k in (item.get("seed_phrases") or []) if isinstance(k, str) and k.strip()]
                vol = 0
                cpc = 0.0
                kd = 0

                indicators = [
                    "Editorial candidate",
                    f"State: {score.get('validation_state')}",
                ]

                trend_analysis = {
                    "state": score.get("validation_state"),
                    "editorial_value_score": score.get("editorial_value_score"),
                    "seo_support_score": score.get("seo_support_score"),
                    "geo_readiness_score": score.get("geo_readiness_score"),
                    "final_subtopic_score": score.get("final_subtopic_score"),
                    "keywords_mined": 0,
                    "variants_tried": [],
                }
                monetization_data = {
                    "status": "pending",
                    "offers": [],
                    "keyword_evidence": selected_keywords,
                    "primary_keyword": None,
                    "commercial_paths": item.get("commercial_paths", []),
                }

                subtopic = EnhancedSubtopic(
                    id=str(uuid4()),
                    title=item.get("title", "Untitled Subtopic"),
                    search_volume_indicators=indicators if indicators else ["Editorial candidate"],
                    autocomplete_suggestions=keyword_strings,
                    relevance_score=float(score.get("final_subtopic_score") or 0.5),
                    source=SubtopicSource.HYBRID,
                    rationale=item.get("summary") or item.get("user_problem") or "Generated from editorial subtopic pipeline.",
                    seed_keywords=keyword_strings or (item.get("seed_phrases") or []),
                    target_audience=item.get("target_audience") or brief.get("target_audience") or "Niche Audience",
                    search_volume=vol,
                    cpc=cpc,
                    keyword_difficulty=kd,
                    trend_analysis=trend_analysis,
                    monetization_data=monetization_data,
                    intent_bucket=brief.get("intent_bucket"),
                    decision_focus=brief.get("decision_focus"),
                    angle_question=brief.get("angle_question"),
                    value_layer_tags=brief.get("value_layer_tags") or [],
                    cluster_type=item.get("decision_type") or "decision",
                    primary_user_outcome=item.get("user_problem") or item.get("summary") or "",
                    serp_intent_match="high" if score.get("seo_support_score", 0) >= 0.45 else "medium",
                    tool_potential_score=int((score.get("seo_support_score", 0) * 100)),
                    validation_state=score.get("validation_state"),
                    seo_readiness_score=score.get("seo_support_score"),
                    geo_readiness_score=score.get("geo_readiness_score"),
                    editorial_value_score=score.get("editorial_value_score"),
                    keyword_evidence=selected_keywords,
                )
                enhanced_subtopics.append(subtopic)

            if not enhanced_subtopics:
                raise ValueError("Editorial subtopic generation produced no results.")
            
            # Prepare response
            processing_time = time.time() - start_time
            enhancement_methods = ["semantic_expansion", "profit_verification"]
            
            message = f"Topic decomposed into {len(enhanced_subtopics)} editorial clusters"
            
            result = {
                "success": True,
                "message": message,
                "original_query": query,
                "decomposition_context": decomposition_context or {},
                "subtopics": [subtopic.to_dict() for subtopic in enhanced_subtopics],
                "autocomplete_data": None, # Deprecated in this view
                "processing_time": processing_time,
                "enhancement_methods": enhancement_methods,
                "warnings": [],
                "debug": {
                    "editorial": editorial_debug,
                },
            }
            
            # Cache result
            self._cache_result(cache_key, result)
            logger.info(
                "Enhanced decomposition succeeded query=%r subtopic_count=%s elapsed_ms=%.1f",
                query,
                len(enhanced_subtopics),
                (time.time() - start_time) * 1000,
            )
            
            return result
            
        except ValueError as e:
            # Graceful fallback: avoid hard 500s when strict semantic pipeline yields no clusters.
            logger.warning(f"Strict semantic pipeline failed, switching to fallback decomposition: {e}")
            try:
                return await _build_fallback_result(e)
            except Exception:
                raise
            
        except Exception as e:
            logger.error(f"Error in enhanced topic decomposition: {str(e)}", exc_info=True)
            try:
                logger.warning("Unexpected decomposition error, attempting resilient fallback path")
                return await _build_fallback_result(e)
            except Exception as fallback_error:
                logger.error(f"Fallback decomposition also failed: {fallback_error}", exc_info=True)
                return {
                    "success": False,
                    "message": f"Quality Control Error: {str(e)}",
                    "original_query": query,
                    "subtopics": [],
                    "autocomplete_data": None,
                    "processing_time": time.time() - start_time,
                    "enhancement_methods": [],
                    "debug": {},
                }
    
    async def compare_methods(self, 
                            query: str,
                            user_id: str,
                            max_subtopics: int = 6) -> Dict[str, Any]:
        """
        Compare different decomposition methods side-by-side
        
        Args:
            query: Topic to analyze
            user_id: User identifier
            max_subtopics: Maximum number of subtopics per method
            
        Returns:
            Dictionary with method comparison results
        """
        start_time = time.time()
        
        try:
            # Run all methods in parallel
            llm_task = self._run_llm_only_method(query, max_subtopics)
            autocomplete_task = self._run_autocomplete_only_method(query, max_subtopics)
            hybrid_task = self._run_hybrid_method(query, max_subtopics)
            
            # Wait for all methods to complete
            llm_result, autocomplete_result, hybrid_result = await asyncio.gather(
                llm_task, autocomplete_task, hybrid_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(llm_result, Exception):
                logger.error(f"LLM method failed: {str(llm_result)}")
                llm_result = MethodResult(subtopics=[], processing_time=0.0, method="LLM Only")
            
            if isinstance(autocomplete_result, Exception):
                logger.error(f"Autocomplete method failed: {str(autocomplete_result)}")
                autocomplete_result = MethodResult(subtopics=[], processing_time=0.0, method="Autocomplete Only")
            
            if isinstance(hybrid_result, Exception):
                logger.error(f"Hybrid method failed: {str(hybrid_result)}")
                hybrid_result = MethodResult(subtopics=[], processing_time=0.0, method="Hybrid (LLM + Autocomplete)")
            
            # Create method comparison
            comparison = MethodComparison(
                id=str(uuid4()),
                original_query=query,
                llm_only_results=llm_result,
                autocomplete_only_results=autocomplete_result,
                hybrid_results=hybrid_result
            )
            
            # Update metrics
            comparison.update_metrics()
            
            # Get recommendation
            recommendation = comparison.get_recommendation()
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "original_query": query,
                "comparison": {
                    "llm_only": comparison.llm_only_results.to_dict(),
                    "autocomplete_only": comparison.autocomplete_only_results.to_dict(),
                    "hybrid": comparison.hybrid_results.to_dict()
                },
                "recommendation": recommendation,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in method comparison: {str(e)}")
            return {
                "success": False,
                "original_query": query,
                "comparison": None,
                "recommendation": f"Error comparing methods: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    async def _run_llm_only_method(self, query: str, max_subtopics: int) -> MethodResult:
        """Run LLM-only decomposition method"""
        start_time = time.time()
        
        try:
            # Always try to use LLM first
            subtopics = await asyncio.wait_for(
                self._get_llm_subtopics(query, None),
                timeout=FALLBACK_LLM_TIMEOUT_SECONDS,
            )
            
            processing_time = time.time() - start_time
            logger.info(
                "LLM-only fallback finished query=%r subtopic_count=%s elapsed_ms=%.1f",
                query,
                len(subtopics),
                processing_time * 1000,
            )
            
            return MethodResult(
                subtopics=subtopics[:max_subtopics],
                processing_time=processing_time,
                method="LLM Only"
            )
            
        except Exception as e:
            logger.error(f"LLM-only method failed: {str(e)}")
            return MethodResult(
                subtopics=[],
                processing_time=time.time() - start_time,
                method="LLM Only"
            )
    
    async def _run_autocomplete_only_method(self, query: str, max_subtopics: int) -> MethodResult:
        """Run autocomplete-only decomposition method"""
        start_time = time.time()
        
        try:
            autocomplete_result = await asyncio.wait_for(
                self.google_autocomplete_service.get_suggestions(query),
                timeout=AUTOCOMPLETE_TIMEOUT_SECONDS,
            )
            
            if autocomplete_result.success:
                subtopics = autocomplete_result.suggestions[:max_subtopics]
            else:
                subtopics = []
            
            processing_time = time.time() - start_time
            logger.info(
                "Autocomplete-only fallback finished query=%r success=%s suggestion_count=%s elapsed_ms=%.1f",
                query,
                autocomplete_result.success,
                len(subtopics),
                processing_time * 1000,
            )
            
            return MethodResult(
                subtopics=subtopics,
                processing_time=processing_time,
                method="Autocomplete Only"
            )
            
        except Exception as e:
            logger.error(f"Autocomplete-only method failed: {str(e)}")
            return MethodResult(
                subtopics=[],
                processing_time=time.time() - start_time,
                method="Autocomplete Only"
            )
    
    async def _run_hybrid_method(self, query: str, max_subtopics: int) -> MethodResult:
        """Run hybrid decomposition method"""
        start_time = time.time()
        
        try:
            # Get autocomplete data
            autocomplete_started = time.perf_counter()
            autocomplete_result = await asyncio.wait_for(
                self.google_autocomplete_service.get_suggestions(query),
                timeout=AUTOCOMPLETE_TIMEOUT_SECONDS,
            )
            logger.info(
                "Hybrid fallback autocomplete stage finished query=%r success=%s suggestion_count=%s elapsed_ms=%.1f",
                query,
                autocomplete_result.success,
                len(autocomplete_result.suggestions or []),
                (time.perf_counter() - autocomplete_started) * 1000,
            )
            
            # Get LLM subtopics with autocomplete context
            llm_started = time.perf_counter()
            llm_subtopics = await asyncio.wait_for(
                self._get_llm_subtopics(query, autocomplete_result),
                timeout=FALLBACK_LLM_TIMEOUT_SECONDS,
            )
            logger.info(
                "Hybrid fallback LLM stage finished query=%r subtopic_count=%s elapsed_ms=%.1f",
                query,
                len(llm_subtopics),
                (time.perf_counter() - llm_started) * 1000,
            )
            
            # Combine and enhance subtopics
            enhanced_subtopics = await self._create_enhanced_subtopics(
                query, llm_subtopics, autocomplete_result, max_subtopics
            )
            
            # Extract subtopic titles
            subtopics = [subtopic.title for subtopic in enhanced_subtopics]
            
            processing_time = time.time() - start_time
            logger.info(
                "Hybrid fallback finished query=%r subtopic_count=%s elapsed_ms=%.1f",
                query,
                len(subtopics),
                processing_time * 1000,
            )
            
            return MethodResult(
                subtopics=subtopics,
                processing_time=processing_time,
                method="Hybrid (LLM + Autocomplete)"
            )
            
        except Exception as e:
            logger.error(f"Hybrid method failed: {str(e)}")
            return MethodResult(
                subtopics=[],
                processing_time=time.time() - start_time,
                method="Hybrid (LLM + Autocomplete)"
            )
    
    async def _get_llm_subtopics(self, query: str, autocomplete_data: Optional[AutocompleteResult]) -> List[Dict[str, Any]]:
        """Get subtopics from LLM service"""
        try:
            # Create enhanced prompt with autocomplete context
            prompt = self._create_enhanced_prompt(query, autocomplete_data)
            
            # Call LLM service
            response = await llm_service.generate_text(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.7
            )
            
            content = response.content
            
            # Parse subtopics from LLM response
            subtopics = self._parse_llm_subtopics(content)
            
            return subtopics
            
        except Exception as e:
            logger.error(f"Error getting LLM subtopics: {str(e)}")
            return []

    def _create_enhanced_prompt(self, query: str, autocomplete_data: Optional[AutocompleteResult]) -> str:
        """Create enhanced prompt with autocomplete context"""
        base_prompt = f"""
        ### ROLE
        You are a Senior Niche Strategist and Affiliate Marketing Expert. Your goal is to "explode" a broad seed topic into high-value, specific sub-niches that are primed for revenue generation via affiliate programs and low-competition SEO.

        ### TASK
        Decompose the provided "{query}" into exactly 10-12 subtopics. Each subtopic must be a specific "micro-niche" where users are likely to spend money or seek specific software solutions.

        ### GUIDELINES FOR SUBTOPICS
        1. COMMERCIAL INTENT: Prioritize subtopics where a user is looking for a "solution," "tool," or "product."
        2. SPECIFICITY: Avoid broad terms. (e.g., instead of "Investing," use "Micro-investing for College Students").
        3. TREND POTENTIAL: Focus on "evergreen" topics or rising trends in the current year (2026).
        4. AFFILIATE FEASIBILITY: Ensure the niche typically has products like SaaS, courses, or physical gear associated with it.

        ### SEED KEYWORD GENERATION (PRE-SEO)
        For EACH subtopic, generate 3 "Seed Keywords."
        - LENGTH: Keywords must be short-tail (3-4 words maximum).
        - INTENT: Must be "Commercial" (e.g., "best budget apps") or "Informational" (e.g., "how to save for retirement").

        ### OUTPUT FORMAT
        You must verify the response is strictly in the following TEXT DELIMITED format. Do not use JSON.

        [SUBTOPIC]
        Name: <name of subtopic>
        Rationale: <one sentence rationale>
        Seed Keywords: <keyword 1>, <keyword 2>, <keyword 3>
        Target Audience: <specific persona>
        [END]
        
        Repeat this block for each subtopic.
        """
        
        if autocomplete_data and autocomplete_data.success:
            autocomplete_context = f"""
            
            ### REAL-TIME SEARCH DATA
            Based on real-time search data, here are related search suggestions you should consider integrating:
            {', '.join(autocomplete_data.suggestions[:10])}
            """
            base_prompt += autocomplete_context
        
        return base_prompt
    
    def _parse_llm_subtopics(self, content: str) -> List[Dict[str, Any]]:
        """Parse subtopics from LLM response content using flexible text delimiters"""
        try:
            subtopics = []
            
            # Normalize content
            clean_content = content.strip()
            
            # Log raw content for debugging
            logger.info("DEBUG - RAW LLM OUPUT START")
            logger.info(clean_content)
            logger.info("DEBUG - RAW LLM OUPUT END")
            
            # Split by known block headers instead of just [SUBTOPIC]
            # We look for "Name:" as the start of a block
            # This handles cases where [SUBTOPIC] is missing
            
            # Pattern: (Start of String or Newline) followed by optional list markers (1., -, *, #), then "Name:"
            # Matches: "Name:", "1. Name:", "- Name:", "**Name**:", "## 1. Name:"
            block_pattern = r'(?:^|\n)(?:[\s\d\.\*\-#]*?)Name(?:[\*\-#]*\s*):'
            
            # Find all start indices
            starts = [m.start() for m in re.finditer(block_pattern, clean_content, re.IGNORECASE)]
            
            if not starts:
                logger.warning("No 'Name:' fields found in LLM output")
                return []
                
            blocks = []
            for i in range(len(starts)):
                start_idx = starts[i]
                end_idx = starts[i+1] if i < len(starts) - 1 else len(clean_content)
                blocks.append(clean_content[start_idx:end_idx])
            
            for block in blocks:
                if not block.strip():
                    continue
                    
                # Clean block
                if "[END]" in block:
                    block = block.split("[END]")[0]
                
                # Parse fields using robust multiline regex with Markdown support
                # ... (Matches stay the same, but now applied to reliably split blocks) ...
                
                # RE-USE REGEX FROM BEFORE
                name_match = re.search(r'(?:[\*\-#]+\s*)?Name(?:[\*\-#]+)?:\s*(.*?)(?=\n\s*(?:(?:[\*\-#]+\s*)?Rationale|(?:[\*\-#]+\s*)?(?:Seed\s+)?Keywords|(?:[\*\-#]+\s*)?Target Audience|\[END\])|$)', block, re.IGNORECASE | re.DOTALL)
                
                rationale_match = re.search(r'(?:[\*\-#]+\s*)?Rationale(?:[\*\-#]+)?:\s*(.*?)(?=\n\s*(?:(?:[\*\-#]+\s*)?(?:Seed\s+)?Keywords|(?:[\*\-#]+\s*)?Target Audience|\[END\])|$)', block, re.IGNORECASE | re.DOTALL)
                
                keywords_match = re.search(r'(?:[\*\-#]+\s*)?(?:Seed\s+)?Keywords(?:[\*\-#]+)?:\s*(.*?)(?=\n\s*(?:(?:[\*\-#]+\s*)?Target Audience|\[END\])|$)', block, re.IGNORECASE | re.DOTALL)
                
                audience_match = re.search(r'(?:[\*\-#]+\s*)?Target Audience(?:[\*\-#]+)?:\s*(.*?)(?=\n\s*(?:\[END\])|$)', block, re.IGNORECASE | re.DOTALL)
                
                if name_match:
                    name = name_match.group(1).strip()
                    rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided"
                    keywords_str = keywords_match.group(1).strip() if keywords_match else ""
                    audience = audience_match.group(1).strip() if audience_match else "General Audience"
                    
                    # Split keywords via comma or newline to handle bullet points
                    # Replace newlines with commas, then split
                    clean_kw_str = re.sub(r'[\r\n]+', ',', keywords_str)
                    clean_kw_str = re.sub(r'[•\-\*]', '', clean_kw_str)
                    
                    keywords = [k.strip() for k in clean_kw_str.split(',') if k.strip()]
                    
                    subtopics.append({
                        "subtopic_name": name,
                        "rationale": rationale,
                        "seed_keywords": keywords,
                        "target_audience": audience
                    })
            
            logger.info(f"Successfully parsed {len(subtopics)} subtopics from LLM response")
            return subtopics
            
            logger.info(f"Successfully parsed {len(subtopics)} subtopics from LLM response")
            return subtopics
            
        except Exception as e:
            logger.error(f"Error parsing LLM subtopics: {str(e)}")
            return []

    async def _create_enhanced_subtopics(self, 
                                       query: str,
                                       llm_subtopics: List[Dict[str, Any]],
                                       autocomplete_data: Optional[AutocompleteResult],
                                       max_subtopics: int) -> List[EnhancedSubtopic]:
        """Create enhanced subtopics with relevance scoring"""
        enhanced_subtopics = []
        
        # Helper to find if a title exists in LLM subtopics
        def find_in_llm(title: str) -> Optional[Dict[str, Any]]:
            for s in llm_subtopics:
                if s["subtopic_name"].lower() == title.lower():
                    return s
            return None

        # Combine both autocomplete and LLM suggestions
        all_suggestions = set()
        
        # Add LLM suggestions first (they have rich data)
        for s in llm_subtopics:
            all_suggestions.add(s["subtopic_name"])
            
        # Add autocomplete suggestions 
        if autocomplete_data and autocomplete_data.success and autocomplete_data.suggestions:
            for s in autocomplete_data.suggestions:
                all_suggestions.add(s)
        
        logger.info(f"Combined suggestions ({len(all_suggestions)} total)")
        
        # Create enhanced subtopics
        # Priority to LLM subtopics as they come with seed keywords etc.
        unique_titles = list(all_suggestions)
        try:
            for i, subtopic_title in enumerate(unique_titles):
                # Find associated LLM data if available
                llm_data = find_in_llm(subtopic_title)
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(
                    subtopic_title, query, autocomplete_data
                )
                
                # Determine source
                is_in_llm = llm_data is not None
                is_in_autocomplete = autocomplete_data and autocomplete_data.success and subtopic_title in autocomplete_data.suggestions
                
                if is_in_llm and is_in_autocomplete:
                    source = SubtopicSource.HYBRID
                    relevance_score += 0.2  # Bonus for hybrid
                elif is_in_llm:
                    source = SubtopicSource.LLM
                    relevance_score += 0.1  # Bonus for LLM over raw autocomplete
                elif is_in_autocomplete:
                    source = SubtopicSource.AUTOCOMPLETE
                    # No bonus for pure autocomplete, especially fallbacks
                else:
                    source = SubtopicSource.LLM
                
                # Create search volume indicators
                search_volume_indicators = self._create_search_volume_indicators(
                    subtopic_title, autocomplete_data
                )
                
                # Get autocomplete suggestions for this subtopic
                autocomplete_suggestions = self._get_autocomplete_suggestions_for_subtopic(
                    subtopic_title, autocomplete_data
                )
                
                # Create enhanced subtopic
                enhanced_subtopic = EnhancedSubtopic(
                    id=str(uuid4()),
                    title=subtopic_title,
                    search_volume_indicators=search_volume_indicators,
                    autocomplete_suggestions=autocomplete_suggestions,
                    relevance_score=relevance_score,
                    source=source,
                    rationale=llm_data.get("rationale") if llm_data else None,
                    seed_keywords=llm_data.get("seed_keywords", []) if llm_data else [],
                    target_audience=llm_data.get("target_audience") if llm_data else None
                )
                
                enhanced_subtopics.append(enhanced_subtopic)
        except Exception as e:
            logger.error(f"Error creating enhanced subtopics loop: {e}")
            raise
        
        # Sort by relevance score
        enhanced_subtopics.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # QUALITY FILTER: Only keep subtopics that have a rationale (from LLM)
        # unless it's a very high confidence hybrid.
        filtered_subtopics = [s for s in enhanced_subtopics if s.rationale is not None or s.source == SubtopicSource.HYBRID]
        
        return filtered_subtopics[:max_subtopics]
    
    def _calculate_relevance_score(self, 
                                 subtopic: str, 
                                 query: str, 
                                 autocomplete_data: Optional[AutocompleteResult]) -> float:
        """Calculate relevance score for a subtopic"""
        base_score = 0.5
        
        # Boost score if subtopic contains query terms
        query_terms = query.lower().split()
        subtopic_lower = subtopic.lower()
        
        for term in query_terms:
            if term in subtopic_lower:
                base_score += 0.1
        
        # Boost score if subtopic appears in autocomplete suggestions
        if autocomplete_data and autocomplete_data.success:
            # ONLY boost if it's NOT fallback data. Fallback suggestions are too generic.
            if not getattr(autocomplete_data, 'is_fallback', False):
                if subtopic in autocomplete_data.suggestions:
                    base_score += 0.3
        
        # Boost score for commercial keywords
        commercial_keywords = ['best', 'review', 'buy', 'price', 'compare', 'top', 'guide']
        for keyword in commercial_keywords:
            if keyword in subtopic_lower:
                base_score += 0.05
        
        # Boost score for trending indicators
        trending_indicators = ['2024', 'new', 'latest', 'trending', 'popular']
        for indicator in trending_indicators:
            if indicator in subtopic_lower:
                base_score += 0.05
        
        return min(1.0, max(0.0, base_score))
    
    def _determine_source(self, 
                         subtopic: str, 
                         llm_subtopics: List[str], 
                         autocomplete_data: Optional[AutocompleteResult]) -> SubtopicSource:
        """Determine the source of a subtopic"""
        in_llm = subtopic in llm_subtopics
        in_autocomplete = (autocomplete_data and 
                          autocomplete_data.success and 
                          subtopic in autocomplete_data.suggestions)
        
        if in_llm and in_autocomplete:
            return SubtopicSource.HYBRID
        elif in_llm:
            return SubtopicSource.LLM
        elif in_autocomplete:
            return SubtopicSource.AUTOCOMPLETE
        else:
            return SubtopicSource.LLM  # Default to LLM
    
    def _create_search_volume_indicators(self, 
                                       subtopic: str, 
                                       autocomplete_data: Optional[AutocompleteResult]) -> List[str]:
        """Create search volume indicators for a subtopic"""
        indicators = []
        
        if autocomplete_data and autocomplete_data.success:
            if subtopic in autocomplete_data.suggestions:
                indicators.append("Found in autocomplete suggestions")
            
            if len(autocomplete_data.suggestions) > 5:
                indicators.append("High search volume from autocomplete")
        
        # Add generic indicators based on subtopic content
        if 'best' in subtopic.lower():
            indicators.append("High commercial intent")
        
        if '2024' in subtopic.lower():
            indicators.append("Trending topic")
        
        if 'review' in subtopic.lower():
            indicators.append("Review-focused search")
        
        return indicators if indicators else ["Standard search volume"]
    
    def _get_autocomplete_suggestions_for_subtopic(self, 
                                                 subtopic: str, 
                                                 autocomplete_data: Optional[AutocompleteResult]) -> List[str]:
        """Get autocomplete suggestions related to a subtopic"""
        if not autocomplete_data or not autocomplete_data.success:
            return []
        
        # Filter suggestions that are related to the subtopic
        related_suggestions = []
        subtopic_terms = subtopic.lower().split()
        
        for suggestion in autocomplete_data.suggestions:
            suggestion_lower = suggestion.lower()
            if any(term in suggestion_lower for term in subtopic_terms):
                related_suggestions.append(suggestion)
        
        return related_suggestions[:3]  # Limit to 3 related suggestions
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired"""
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                return cached_data['result']
            else:
                del self.cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache result with timestamp"""
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def clear_cache(self) -> None:
        """Clear all cached results"""
        self.cache.clear()
        logger.info("Cleared enhanced topic decomposition cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.cache),
            'cache_ttl_seconds': self.cache_ttl,
            'cached_queries': list(self.cache.keys())
        }
