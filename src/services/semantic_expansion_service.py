"""
Semantic Expansion Service
Implements the "Bucket of Seeds" approach for keyword expansion and profitability verification.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
import math
import re
from datetime import datetime, timedelta

from ..integrations.dataforseo import dataforseo_api
from .llm.llm_service import llm_service
from ..core.models.enhanced_subtopic import EnhancedSubtopic, SubtopicSource

logger = logging.getLogger(__name__)

class SemanticExpansionService:
    """
    Service for semantic keyword expansion, filtering, clustering, and verification.
    """

    def __init__(self):
        pass

    def _sanitize_keyword_text(self, text: Any) -> str:
        """Normalize noisy LLM keyword strings into clean query text."""
        if not isinstance(text, str):
            return ""
        cleaned = text.strip()
        if not cleaned:
            return ""
        # Handle annotation-heavy patterns like: "Pivot.* (4 words) -> *Wait vs buy before pivot.* (5 words) - *"
        if "->" in cleaned:
            cleaned = cleaned.split("->")[-1]
        # Remove markdown bullets/emphasis and noisy suffixes
        cleaned = re.sub(r"[*`_#•]+", " ", cleaned)
        cleaned = re.sub(r"\(\s*\d+\s+words?\s*\)", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bwords?\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+-\s*$", " ", cleaned)
        cleaned = re.sub(r"[^a-zA-Z0-9&'/%\-\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
        return cleaned

    def _normalize_keyword_key(self, text: Any) -> str:
        return self._sanitize_keyword_text(text).lower().strip()

    def _compact_keyword_for_metrics(self, text: str) -> str:
        """Create a shorter lookup variant to improve metric hit-rate on very long tails."""
        cleaned = self._sanitize_keyword_text(text)
        if not cleaned:
            return ""
        words = [w for w in cleaned.split() if w]
        if len(words) <= 5:
            return cleaned

        leading_noise = {
            "how", "what", "why", "when", "where", "should", "can", "is", "are",
            "the", "a", "an", "to", "for", "in", "on", "of", "with"
        }
        removable_words = {
            "the", "a", "an", "to", "for", "in", "on", "of", "with", "and", "or", "by"
        }

        # Strip question boilerplate so compaction keeps the decision/topic core.
        trimmed = list(words)
        while trimmed and trimmed[0].lower() in leading_noise:
            trimmed.pop(0)
        if not trimmed:
            trimmed = list(words)

        lowered = [w.lower() for w in trimmed]

        # Preserve comparison intent for DataForSEO (e.g., "x vs y").
        if "vs" in lowered or "versus" in lowered:
            idx = lowered.index("vs") if "vs" in lowered else lowered.index("versus")
            left = [w for w in trimmed[:idx] if w.lower() not in removable_words]
            right = [w for w in trimmed[idx + 1:] if w.lower() not in removable_words]
            compact_parts: List[str] = []
            compact_parts.extend(left[-2:])
            compact_parts.append("vs")
            compact_parts.extend(right[:2])
            compact = " ".join(compact_parts).strip()
            if compact and len(compact.split()) >= 2:
                return compact

        core = [w for w in trimmed if w.lower() not in removable_words]
        if len(core) <= 5:
            return " ".join(core) if core else " ".join(trimmed[:5])

        # Use front-loaded intent plus one tail discriminator instead of blunt truncation.
        compact_parts = core[:3] + [core[-1]]
        deduped_parts: List[str] = []
        seen = set()
        for part in compact_parts:
            key = part.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped_parts.append(part)

        compact = " ".join(deduped_parts[:5]).strip()
        if compact and len(compact.split()) >= 2:
            return compact
        return " ".join(core[:5]) if core else " ".join(trimmed[:5])

    def _head_keyword_for_metrics(self, text: str) -> str:
        """Create a short head-term fallback when long-tail phrases return no metrics."""
        cleaned = self._sanitize_keyword_text(text)
        if not cleaned:
            return ""
        stopwords = {
            "the", "a", "an", "and", "or", "to", "for", "of", "with", "without", "in",
            "on", "at", "vs", "versus", "how", "what", "when", "where", "best"
        }
        words = [w for w in cleaned.split() if len(w) > 2 and w.lower() not in stopwords]
        if not words:
            return ""
        return " ".join(words[:3])

    async def expand_and_verify(
        self,
        topic: str,
        user_id: str,
        decomposition_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Main entry point: Expands a central topic into verified, profitable clusters.
        """
        logger.info(f"Starting semantic expansion for topic: {topic}")

        # Step 1: Semantic Explosion (LLM)
        seed_bundle = await self.generate_seeds(topic, decomposition_context=decomposition_context)
        seeds = seed_bundle.get("seeds", [])
        seed_intent_map = seed_bundle.get("seed_intent_map", {})
        if not seeds:
            logger.warning("No seeds generated. Aborting.")
            return []
        
        # Step 2: Bulk Data Retrieval (DataForSEO)
        raw_keywords = await self.fetch_bulk_keyword_data(seeds, seed_intent_map=seed_intent_map)
        if not raw_keywords:
             logger.warning("No keyword data found. Aborting.")
             return []

        # Step 3: Profit Filtering (Math)
        filtered_keywords = await self.filter_profitable_keywords(raw_keywords)
        if not filtered_keywords:
            logger.warning("No profitable keywords found after filtering. Aborting.")
            return []
            
        if filtered_keywords:
            await self.enrich_keywords_with_difficulty(filtered_keywords)
            
            # Re-sort after getting true SEO difficulty (KD might have changed from 0 to 80!)
            filtered_keywords.sort(key=lambda x: x.get('profit_score', 0), reverse=True)
            
            # Step 3.6: Post-Enrichment Safety Filter
            # Now that we know the TRUTH, discard anything that is too hard (KD > 85),
            # even if it has massive volume.
            filtered_keywords = [k for k in filtered_keywords if k.get('keyword_difficulty', 0) <= 85]
            
            # Keep top 75 for clustering (Optimal per LLM context window)
            filtered_keywords = filtered_keywords[:75]
            
            logger.info(f"Enrichment complete. Proceeding with {len(filtered_keywords)} validated keywords.")

        # Step 4: Semantic Clustering (LLM)
        clusters = await self.cluster_keywords(
            filtered_keywords,
            decomposition_context=decomposition_context,
        )
        if not clusters:
             logger.warning("No clusters generated. Aborting.")
             return []

        # Step 5: Profitability Verification (Trends + LLM)
        verified_clusters = await self.verify_clusters(clusters)
        
        return verified_clusters

    def _format_decomposition_context(self, topic: str, decomposition_context: Optional[Dict[str, Any]]) -> str:
        """Render a compact context packet for prompts."""
        if not decomposition_context:
            return ""

        context_lines: List[str] = []

        project_name = decomposition_context.get("project_name")
        if project_name:
            context_lines.append(f"Website/Project: {project_name}")

        project_description = decomposition_context.get("project_description")
        if project_description:
            context_lines.append(f"Website Description: {project_description}")

        topic_description = decomposition_context.get("topic_description")
        if topic_description:
            context_lines.append(f"Topic Description: {topic_description}")

        category_path = decomposition_context.get("category_path")
        if category_path:
            context_lines.append(f"Selected Category Lens: {category_path}")

        decision_focus = decomposition_context.get("decision_focus")
        if decision_focus:
            context_lines.append(f"Decision Focus: {decision_focus}")

        intent_bucket = decomposition_context.get("intent_bucket")
        if intent_bucket:
            context_lines.append(f"Intent Bucket: {intent_bucket}")

        angle_question = decomposition_context.get("angle_question")
        if angle_question:
            context_lines.append(f"Angle Question: {angle_question}")

        value_layer_tags = decomposition_context.get("value_layer_tags") or []
        if value_layer_tags:
            context_lines.append(f"Value Layer Tags: {', '.join(value_layer_tags[:8])}")

        audience = decomposition_context.get("target_audience")
        if audience:
            context_lines.append(f"Target Audience: {audience}")

        evidence_sources = decomposition_context.get("evidence_sources") or []
        if evidence_sources:
            context_lines.append(f"Evidence Sources: {', '.join(evidence_sources[:8])}")

        signal_terms = decomposition_context.get("signal_terms") or []
        if signal_terms:
            context_lines.append(f"Relevant Signal Terms: {', '.join(signal_terms[:10])}")

        trend_titles = decomposition_context.get("trend_titles") or []
        if trend_titles:
            context_lines.append(f"Recent Trend Themes: {', '.join(trend_titles[:6])}")

        constraints = decomposition_context.get("decomposition_constraints") or []
        if constraints:
            context_lines.append("Subtopic Constraints:")
            for item in constraints[:8]:
                context_lines.append(f"- {item}")

        if not context_lines:
            return ""

        joined_context = "\n".join(context_lines)
        return f"""
        ADDITIONAL DECOMPOSITION CONTEXT:
        {joined_context}

        Use this context to interpret the seed topic "{topic}".
        Prefer subtopics that help the user make a concrete decision, compare options, quantify tradeoffs, or surface hidden costs.
        """

    async def generate_seeds(self, topic: str, decomposition_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Step 1: Ask LLM for intent-aware seed keywords anchored to the topic angle.
        """
        preferred_intent = (decomposition_context or {}).get("intent_bucket") or "informational_decision"
        context_block = self._format_decomposition_context(topic, decomposition_context)
        prompt = f"""
        You are an expert SEO strategist specializing in search intent, keyword clustering, and content positioning.
        I have a central topic: "{topic}".
        {context_block}
        Preferred Intent Bucket: {preferred_intent}

        Goal:
        Produce seed keywords that can succeed in DataForSEO lookups and map to real Google behavior.
        Translate technical/abstract ideas into practical phrases people actually search.

        Required intent groups and counts (exactly 20 total):
        - QUESTION_INTENT: 5 seeds (why/how/should/can phrasing)
        - COMPARISON_INTENT: 5 seeds (X vs Y, option tradeoffs, alternative evaluation)
        - ROI_INTENT: 5 seeds (returns, cost, upside/downside, optimization, framework outcomes)
        - TOOL_INTENT: 5 seeds (calculator, checklist, model, scorecard, worksheet, framework utility)

        Quality rules:
        - Each seed must be 3-5 words and readable as a normal search query.
        - Avoid punctuation-heavy or symbolic phrases. No wildcards, regex-like patterns, arrows, or markdown.
        - Avoid generic filler such as "best topic", "guide", "tips", "overview" without a concrete qualifier.
        - Keep one clear concept per seed; avoid run-on clauses.
        - Include practical, human wording alongside technical meaning (bridging terms).
        - Keep seeds tightly tied to topic decision-focus, selected category lens, and intended audience.
        - Favor the preferred intent bucket tone where reasonable while still covering all 4 groups.

        Output contract (strict):
        Return ONLY lines in this exact format, one seed per line:
        QUESTION_INTENT | keyword text
        COMPARISON_INTENT | keyword text
        ROI_INTENT | keyword text
        TOOL_INTENT | keyword text

        Do not include numbering, bullets, headings, JSON, or commentary.
        """
        try:
            response = await asyncio.wait_for(
                llm_service.generate_text(prompt=prompt, max_tokens=500),
                timeout=30.0  # 30 second timeout for seed generation
            )
            text = response.content.strip()
            seeds = []
            seed_intent_map: Dict[str, str] = {}
            for line in text.split('\n'):
                clean_line = line.strip().lstrip('- ').strip()
                if not clean_line:
                    continue
                if '|' in clean_line:
                    intent_label, keyword_text = clean_line.split('|', 1)
                    intent_label = intent_label.strip().upper()
                    keyword_text = keyword_text.strip()
                else:
                    intent_label = "GENERAL_INTENT"
                    keyword_text = clean_line
                if keyword_text:
                    normalized_seed = self._compact_keyword_for_metrics(keyword_text)
                    logger.debug(
                        "Seed compaction: intent=%s raw=%r compact=%r",
                        intent_label,
                        keyword_text,
                        normalized_seed,
                    )
                    # DataForSEO performs better with concise, query-like seeds.
                    if len(normalized_seed.split()) < 2:
                        logger.debug(
                            "Seed dropped after compaction (too short): intent=%s raw=%r compact=%r",
                            intent_label,
                            keyword_text,
                            normalized_seed,
                        )
                        continue
                    seeds.append(normalized_seed)
                    seed_intent_map[normalized_seed.lower()] = intent_label

            # Deduplicate while preserving order
            deduped_seeds = []
            seen = set()
            for seed in seeds:
                normalized = seed.lower()
                if normalized in seen:
                    logger.debug("Seed deduplicated: %r", seed)
                    continue
                seen.add(normalized)
                deduped_seeds.append(seed)
            seeds = deduped_seeds

            logger.info(
                "Generated %s unique compact seeds for topic %r. Sample=%s",
                len(seeds),
                topic,
                seeds[:8]
            )
            return {
                "seeds": seeds,
                "seed_intent_map": {seed.lower(): seed_intent_map.get(seed.lower(), "GENERAL_INTENT") for seed in seeds}
            }
        except Exception as e:
            logger.error(f"Error generating seeds: {e}")
            fallback_seed = [topic] if topic else []
            return {
                "seeds": fallback_seed,
                "seed_intent_map": {topic.lower(): "GENERAL_INTENT"} if topic else {}
            }

    async def fetch_bulk_keyword_data(self, seeds: List[str], seed_intent_map: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Step 2: Fetch related keywords for all seeds from DataForSEO.
        NOTE: This mimics bulk retrieval by making parallel calls for batches of seeds.
        """
        all_keywords = []
        seed_intent_map = seed_intent_map or {}
        
        # Limit seeds to avoid excessive API usage if LLM returns too many
        seeds_to_process = seeds[:50] 
        
        # Execute using Standard API (Queue-based, high volume)
        if seeds_to_process:
            logger.info(
                "DataForSEO expansion request: seed_count=%s sample=%s",
                len(seeds_to_process),
                seeds_to_process[:8],
            )
            # 1. Try to get Related Keywords (Expansion)
            # Note: This endpoint often returns empty for specific long-tails.
            results = await dataforseo_api.get_related_keywords_standard(seeds_to_process, limit_per_seed=20)
            
            if results:
                all_keywords.extend(results)
                logger.info(f"Standard API expansion returned {len(results)} keywords.")
            else:
                logger.warning("Standard API expansion returned no results. Proceeding with seeds.")

        # Deduplicate by keyword text
        unique_keywords = {}
        for kw in all_keywords:
            if kw['keyword'] not in unique_keywords:
                normalized_kw = kw['keyword'].lower()
                kw['seed_intent_group'] = seed_intent_map.get(normalized_kw, "")
                unique_keywords[kw['keyword']] = kw
        
        # 2. Start with what we have (Expanded or Empty)
        final_list = list(unique_keywords.values())

        # 3. IF we have no expanded keywords (or very few), add the SEEDS to the list
        # We want to verify the seeds themselves regardless of expansion success.
        expanded_kw_text = set(k['keyword'].lower() for k in final_list)
        seeds_added = 0
        for seed in seeds:
            s_clean = seed.strip()
            if s_clean.lower() not in expanded_kw_text:
                final_list.append({
                    'keyword': s_clean,
                    'search_volume': 0, # Will enrich below
                    'cpc': 0,
                    'keyword_difficulty': 0,
                    'is_fallback': True,
                    'seed_intent_group': seed_intent_map.get(s_clean.lower(), "GENERAL_INTENT"),
                })
                seeds_added += 1
        
        if seeds_added > 0:
            logger.info(f"Added {seeds_added} seeds to candidate list for enrichment.")

        # 4. CRITICAL: Enrich ALL candidates with Volume/CPC using Schema Endpoint
        # 'get_related_keywords' gives volume for the *related* terms, but if we used seeds (fallback), 
        # they have 0. We must fetch their volume.
        # Also, sometimes 'related' endpoint metrics are stale.
        # We will batch-fetch Volume/CPC for the final list.
        
        candidates_to_enrich = [k['keyword'] for k in final_list if k.get('search_volume', 0) == 0]
        
        logger.info(f"DEBUG: Checking candidates for enrichment. Total: {len(final_list)}. Need Enrichment: {len(candidates_to_enrich)}")
        # Debug the first candidate's logic
        if final_list:
            logger.info(f"DEBUG First Candidate Vol: {final_list[0].get('search_volume')} (Type: {type(final_list[0].get('search_volume'))})")

        if candidates_to_enrich:
            logger.info(f"Enriching Volume/CPC for {len(candidates_to_enrich)} keywords: {candidates_to_enrich[:5]}...")
            lookup_terms: List[str] = []
            seen_terms = set()
            for original in candidates_to_enrich:
                compact = self._compact_keyword_for_metrics(original)
                head = self._head_keyword_for_metrics(original)
                if original not in seen_terms:
                    lookup_terms.append(original)
                    seen_terms.add(original)
                if compact and compact not in seen_terms:
                    lookup_terms.append(compact)
                    seen_terms.add(compact)
                if head and head not in seen_terms:
                    lookup_terms.append(head)
                    seen_terms.add(head)

            # Use robust standard endpoint with both original and compact fallback terms.
            logger.info(
                "DataForSEO bulk metrics lookup: candidate_count=%s lookup_term_count=%s sample=%s",
                len(candidates_to_enrich),
                len(lookup_terms),
                lookup_terms[:10],
            )
            bulk_metrics = await dataforseo_api.get_bulk_metrics_standard(lookup_terms)
            logger.info(
                "DataForSEO bulk metrics response: metrics_count=%s sample=%s",
                len(bulk_metrics or []),
                [m.get('keyword') for m in (bulk_metrics or [])[:10]],
            )
            
            # Map metrics back
            metrics_map = {m['keyword'].lower(): m for m in bulk_metrics if m.get('keyword')}
            updated_vol_count = 0
            
            for k in final_list:
                k_norm = k['keyword'].lower()
                direct_metric = metrics_map.get(k_norm)
                compact_metric = metrics_map.get(self._compact_keyword_for_metrics(k['keyword']).lower())
                head_metric = metrics_map.get(self._head_keyword_for_metrics(k['keyword']).lower())
                m = direct_metric or compact_metric or head_metric
                if m:
                    k['search_volume'] = m.get('search_volume', 0)
                    k['cpc'] = m.get('cpc', 0)
                    if not k.get('seed_intent_group'):
                        k['seed_intent_group'] = seed_intent_map.get(k_norm, "")
                    if not k.get('competition') or k.get('competition') == 'UNKNOWN':
                        k['competition'] = m.get('competition')
                    updated_vol_count += 1
            
            logger.info(f"Enriched Volume/CPC for {updated_vol_count} keywords.")

        non_zero_volume = sum(1 for k in final_list if (k.get('search_volume') or 0) > 0)
        non_zero_cpc = sum(1 for k in final_list if (k.get('cpc') or 0) > 0)
        non_zero_kd = sum(1 for k in final_list if (k.get('keyword_difficulty') or 0) > 0)
        unresolved_keywords = [k.get('keyword') for k in final_list if (k.get('search_volume') or 0) == 0][:10]
        logger.info(
            "Keyword enrichment summary: total=%s non_zero_volume=%s non_zero_cpc=%s non_zero_kd=%s unresolved_sample=%s",
            len(final_list),
            non_zero_volume,
            non_zero_cpc,
            non_zero_kd,
            unresolved_keywords,
        )

        return final_list

    async def _get_research_settings(self) -> Dict[str, Any]:
        """Fetch research settings from database or return defaults"""
        try:
            # Import here to avoid circular deps; get_supabase_client is a top-level helper
            try:
                from supabase_client import get_supabase_client
            except ImportError:
                import sys, os
                sys.path.append(os.getcwd())
                from supabase_client import get_supabase_client
            supabase = get_supabase_client()
            response = supabase.table('application_settings').select('research_settings').limit(1).execute()
            if response.data:
                return response.data[0].get('research_settings') or {}
        except Exception as e:
            logger.warning(f"Failed to fetch research settings: {e}")
        
        # Defaults
        return {
            "min_volume": 50,
            "max_difficulty": 50,
            "min_cpc": 0.5,
            "strict_mode": True
        }

    async def filter_profitable_keywords(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 3: The Filter (Kill Switch).
        Uses configurable thresholds from settings.
        Metric Thresholds: Vol < min_vol OR KD > max_kd => Kill (if Strict).
        Profit Score: (Vol * CPC) / KD.
        """
        scored_keywords = []
        settings = await self._get_research_settings()
        
        min_vol = settings.get("min_volume", 50)
        max_kd = settings.get("max_difficulty", 50) # Tighter default as per spec
        strict_mode = settings.get("strict_mode", True)
        
        logger.info(f"Filtering with thresholds: MinVol={min_vol}, MaxKD={max_kd}, Strict={strict_mode}")

        # Adaptive relaxation: if most candidates have zero volume, avoid over-pruning on volume.
        positive_volume_count = sum(1 for kw in keywords if (kw.get('search_volume') or 0) > 0)
        if positive_volume_count < 10:
            logger.warning(
                "Low-volume candidate set detected (positive_volume=%s of %s). Relaxing volume gate.",
                positive_volume_count,
                len(keywords)
            )
            min_vol = 0
            strict_mode = False

        for kw in keywords:
            vol = kw.get('search_volume', 0) or 0
            kd = kw.get('keyword_difficulty', 0) or 0
            cpc = kw.get('cpc', 0) or 0
            
            # Helper: sometimes API returns None
            if vol is None: vol = 0
            if kd is None: kd = 0
            if cpc is None: cpc = 0

            # Filter Logic
            is_profitable = True
            
            if vol < min_vol:
                is_profitable = False
            if kd > max_kd:
                is_profitable = False
                
            # If Strict Mode is ON, we kill unprofitable keywords
            if strict_mode and not is_profitable:
                continue
            
            # Score Calculation
            safe_kd = max(kd, 1) # Avoid division by zero
            score = (vol * cpc) / safe_kd
            
            # If not profitable but kept (Strict=False), maybe penalize score?
            if not is_profitable:
                score = score * 0.1 # Penalty for failing criteria
            
            kw['profitability_score'] = score
            scored_keywords.append(kw)
            
        if not scored_keywords:
            logger.warning("Strict filtering removed all keywords. Engaging SAFE MODE (Visualizing raw data).")
            # SAFE MODE: If strict filter killed everything, we bring back valid keywords (vol > 0)
            # regardless of KD or CPC, just to show SOMETHING.
            for kw in keywords:
                if kw.get('search_volume', 0) > 0:
                    scored_keywords.append(kw)

            # If still nothing, just take top 10 raw
            if not scored_keywords:
                 scored_keywords = keywords[:10]

        # ADDITIONAL SAFE MODE: If we have very few keywords after filtering,
        # add more from the original list to ensure good clustering
        if len(scored_keywords) < 20:
            logger.warning(f"Only {len(scored_keywords)} keywords passed strict filter. Adding more candidates for clustering.")
            existing_kws = {k['keyword'] for k in scored_keywords}
            for kw in keywords:
                if kw['keyword'] not in existing_kws and kw.get('search_volume', 0) > 0:
                    kw['profitability_score'] = 0.1  # Low score but keep it
                    scored_keywords.append(kw)
                if len(scored_keywords) >= 30:
                    break

        # Sort by Score descending (or volume if score missing)
        scored_keywords.sort(key=lambda x: x.get('profitability_score', x.get('search_volume', 0)), reverse=True)
        
        # Keep top 250 (Wider net for enrichment)
        top_keywords = scored_keywords[:250]
        
        if not top_keywords:
             logger.warning("Strict filtering removed all keywords. Engaging SAFE MODE.")
             # SAFE MODE: Use raw keywords if filter killed everything
             top_keywords = keywords[:100]

        logger.info(f"Filtered down to {len(top_keywords)} candidates for enrichment.")
        return top_keywords

    async def enrich_keywords_with_difficulty(self, keywords: List[Dict[str, Any]]) -> None:
        """
        Fetch real Organic Keyword Difficulty for a batch of keywords.
        Modifies the dictionary objects in-place with 'keyword_difficulty'.
        """
        try:
             # Extract plain valid keywords
            kw_list = [k['keyword'] for k in keywords if k.get('keyword')]
            if not kw_list: return
            
            # Batch in chunks of 500 (DataForSEO limit is 1000, keep safety margin)
            chunk_size = 500
            for i in range(0, len(kw_list), chunk_size):
                batch = kw_list[i:i + chunk_size]
                
                # Call DataForSEO Live Endpoint
                logger.info(f"Enriching KD for batch of {len(batch)} keywords...")
                kd_data = await dataforseo_api.get_keyword_difficulty(batch)
                
                if not kd_data:
                    logger.warning("No KD data returned from DataForSEO (Empty List).")
                    continue
                
                logger.info(f"DEBUG: KD Data Sample: {kd_data[0] if kd_data else 'None'}")
                    
                # Create map
                kd_map = {item['keyword'].lower(): item['keyword_difficulty'] for item in kd_data if item.get('keyword')}
                logger.info(f"DEBUG: KD Map Keys Sample: {list(kd_map.keys())[:5]}")
                
                # Update original objects
                updated_count = 0
                for k in keywords:
                    k_text = k['keyword'].lower()
                    if k_text in kd_map:
                        # Update KD (Handle None explicitly)
                        new_kd = kd_map.get(k_text)
                        if new_kd is None: new_kd = 0
                        
                        k['keyword_difficulty'] = int(new_kd)
                        
                        # Recalculate profit score with REAL KD
                        vol = k.get('search_volume') or 0
                        cpc = k.get('cpc') or 0
                        safe_kd = max(new_kd, 1)
                        k['profitability_score'] = (vol * cpc) / safe_kd
                        updated_count += 1
                        
                        if updated_count == 1:
                            logger.info(f"DEBUG: Updated First Keyword '{k_text}' with KD: {new_kd}")
                
                logger.info(f"Updated KD for {updated_count} keywords.")

        except Exception as e:
            logger.error(f"Failed to enrich KD: {e}")


    async def cluster_keywords(
        self,
        keywords: List[Dict[str, Any]],
        decomposition_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Step 4: Group keywords into concepts.
        Uses delimited text format instead of JSON for better reliability.
        """
        # Prepare list for prompt
        kw_list_str = "\n".join([f"- {k['keyword']} (Vol: {k['search_volume']}, KD: {k['keyword_difficulty']})" for k in keywords])

        # Count keywords to determine appropriate number of clusters
        kw_count = len(keywords)
        target_clusters = min(max(kw_count // 8, 3), 12)  # Aim for 3-12 clusters, roughly 8 keywords per cluster

        context_block = self._format_decomposition_context("keyword clustering", decomposition_context)

        prompt = f"""
        I have a list of {kw_count} high-potential keywords:
        {kw_list_str}
        {context_block}

        Task:
        Group these keywords into {target_clusters} distinct "Subtopics" or "Clusters".
        Each cluster should represent a specific content theme or micro-niche.

        IMPORTANT:
        - Create EXACTLY {target_clusters} separate clusters (not 1, not 2, but {target_clusters})
        - Each cluster should be distinct and non-overlapping
        - Distribute keywords across all clusters (don't put everything in one cluster)
        - Focus on specific micro-niches, not broad categories
        - Name each cluster so it is obviously useful for downstream article ideation
        - Prefer cluster names that imply a user problem, decision, comparison, framework, checklist, audit, or scenario
        - Keep the website/category lens in mind when naming and grouping

        Output Format (use EXACTLY this format):

        CLUSTER: Subtopic Name Here
        OUTCOME: One clear user outcome this cluster helps achieve
        TYPE: one of [problem, decision, comparison, checklist, framework, audit, calculator, scenario]
        INTENT_MATCH: one of [high, medium, low]
        TOOL_POTENTIAL: integer 0-100
        KEYWORDS: keyword1, keyword2, keyword3

        CLUSTER: Another Subtopic Name
        OUTCOME: ...
        TYPE: ...
        INTENT_MATCH: ...
        TOOL_POTENTIAL: ...
        KEYWORDS: keyword4, keyword5, keyword6

        Rules:
        - Start each cluster with "CLUSTER: " followed by the name
        - Include OUTCOME/TYPE/INTENT_MATCH/TOOL_POTENTIAL fields before KEYWORDS
        - List keywords on the final line starting with "KEYWORDS: "
        - Separate keywords with commas
        - Do not use markdown code blocks
        - Return ONLY the cluster definitions, no other text
        - Create {target_clusters} clusters minimum
        """

        try:
            response = await asyncio.wait_for(
                llm_service.generate_text(prompt=prompt, max_tokens=1500),
                timeout=45.0  # 45 second timeout for clustering
            )

            # Parse delimited text response
            text = response.content.strip()
            subtopics = self._parse_cluster_response(text)

            if not subtopics:
                logger.warning(f"No clusters parsed from LLM response. Raw text: {text[:500]}")
                return []

            # Create lookup map for keyword data
            kw_map = {}
            for kw in keywords:
                key = self._normalize_keyword_key(kw.get("keyword"))
                if key:
                    kw_map[key] = kw

            enriched_clusters = []
            for cluster in subtopics:
                if not cluster: continue
                # Calculate metrics for the cluster
                total_vol = 0
                total_cpc = 0.0
                max_kd = 0
                count = 0
                
                # Parse title from different LLM label formats
                title = cluster.get('subtopic_name', cluster.get('cluster_title', 'Unknown Cluster'))
                
                # Get raw keyword strings from LLM
                raw_kws = cluster.get('seed_keywords', cluster.get('keywords', []))
                
                # Normalize and find matches
                matched_kw_objects = []
                logger.info(f"DEBUG Cluster: {title}")
                logger.info(f"DEBUG Raw KWs from LLM: {raw_kws}")
                # logger.info(f"DEBUG KW Map Keys: {list(kw_map.keys())[:10]}") # Sample keys

                for k_str in raw_kws:
                    k_clean = self._normalize_keyword_key(k_str)
                    if not k_clean:
                        continue
                    if k_clean in kw_map:
                        kw_data = kw_map[k_clean]
                        vol = kw_data.get('search_volume') or 0
                        cpc = kw_data.get('cpc') or 0
                        kd = kw_data.get('keyword_difficulty') or 0
                        
                        total_vol += vol
                        total_cpc += cpc
                        max_kd = max(max_kd, kd)
                        count += 1
                        
                        logger.info(f"DEBUG Match Found: {k_clean} | Vol: {vol}")

                        # Add full object to list
                        matched_kw_objects.append({
                            "keyword": kw_data.get('keyword') or self._sanitize_keyword_text(k_str),
                            "search_volume": vol,
                            "cpc": cpc,
                            "keyword_difficulty": kd,
                            "competition": kw_data.get('competition'),
                            "main_intent": kw_data.get('main_intent') or kw_data.get('intent', 'commercial'),
                            "profitability_score": kw_data.get('profitability_score'),
                            "seed_intent_group": kw_data.get('seed_intent_group', ''),
                        })
                # Fallback: If strict matching failed (LLM hallucinated new words), 
                # try to find ANY keywords from our Golden List that contain the cluster title words.
                if not matched_kw_objects:
                    logger.warning(f"No exact keyword matches for cluster '{title}'. Attempting semantic fallback...")
                    title_words = title.lower().split()
                    short_title_words = [w for w in title_words if len(w) > 3] # meaningful words
                    
                    if short_title_words:
                        for kw_obj in keywords: # Iterate over ALL valid keywords
                             # If we already matched this checking somewhere? No easy state here.
                             # Just check overlap
                             kw_text = kw_obj['keyword'].lower()
                             if any(w in kw_text for w in short_title_words):
                                 # We found a relevant keyword!
                                 # Re-create object structure to match
                                 matched_kw_objects.append({
                                    "keyword": kw_obj['keyword'],
                                    "search_volume": kw_obj.get('search_volume') or 0,
                                    "cpc": kw_obj.get('cpc') or 0,
                                    "keyword_difficulty": kw_obj.get('keyword_difficulty') or 0,
                                    "competition": kw_obj.get('competition'),
                                    "main_intent": kw_obj.get('main_intent') or kw_obj.get('intent', 'commercial'),
                                    "profitability_score": kw_obj.get('profitability_score'),
                                    "seed_intent_group": kw_obj.get('seed_intent_group', ''),
                                 })
                                 
                                 total_vol += kw_obj.get('search_volume') or 0
                                 total_cpc += kw_obj.get('cpc') or 0
                                 max_kd = max(max_kd, kw_obj.get('keyword_difficulty') or 0)
                                 count += 1
                                 
                                 if len(matched_kw_objects) >= 5: # Limit fallback to 5
                                     break

                # Ultimate Fallback: Just grab the top 3 unassigned high-volume keywords 
                # to ensure the cluster isn't empty of metrics.
                if not matched_kw_objects and keywords:
                     logger.warning(f"Semantic fallback failed for '{title}'. Assigning top generic keywords.")
                     for i in range(min(3, len(keywords))):
                         kw_obj = keywords[i]
                         matched_kw_objects.append({
                            "keyword": kw_obj['keyword'],
                            "search_volume": kw_obj.get('search_volume') or 0,
                            "cpc": kw_obj.get('cpc') or 0,
                            "keyword_difficulty": kw_obj.get('keyword_difficulty') or 0,
                            "profitability_score": kw_obj.get('profitability_score'),
                            "seed_intent_group": kw_obj.get('seed_intent_group', ''),
                         })
                         total_vol += kw_obj.get('search_volume') or 0
                         count += 1

                # Average CPC
                avg_cpc = total_cpc / count if count > 0 else 0.0
                
                # Enrich cluster object
                cluster['cluster_title'] = title
                # CRITICAL: Store objects in BOTH keys for compatibility
                cluster['keywords'] = matched_kw_objects if matched_kw_objects else raw_kws 
                cluster['seed_keywords'] = matched_kw_objects if matched_kw_objects else raw_kws
                if matched_kw_objects:
                    cluster['primary_keyword'] = self._sanitize_keyword_text(matched_kw_objects[0].get('keyword'))
                else:
                    fallback_kw = self._sanitize_keyword_text(raw_kws[0]) if raw_kws else ""
                    cluster['primary_keyword'] = fallback_kw or self._sanitize_keyword_text(title)
                cluster['search_volume'] = total_vol
                cluster['cpc'] = round(avg_cpc, 2)
                cluster['keyword_difficulty'] = max_kd
                cluster['cluster_type'] = cluster.get('cluster_type') or "decision"
                cluster['primary_user_outcome'] = cluster.get('primary_user_outcome') or f"Evaluate {title} and choose a practical next step"
                cluster['serp_intent_match'] = (cluster.get('serp_intent_match') or "medium").lower()
                cluster['tool_potential_score'] = int(cluster.get('tool_potential_score') or 50)
                cluster['intent_bucket'] = (
                    cluster.get('intent_bucket')
                    or (decomposition_context or {}).get('intent_bucket')
                    or "informational_decision"
                )
                cluster['decision_focus'] = (
                    cluster.get('decision_focus')
                    or (decomposition_context or {}).get('decision_focus')
                    or f"Help users make a better decision about {title}"
                )
                cluster['angle_question'] = (
                    cluster.get('angle_question')
                    or (decomposition_context or {}).get('angle_question')
                    or f"How should users evaluate options in {title}?"
                )
                cluster['value_layer_tags'] = (
                    cluster.get('value_layer_tags')
                    or (decomposition_context or {}).get('value_layer_tags')
                    or ["decision-support"]
                )
                intent_coverage: Dict[str, int] = {}
                for kw in matched_kw_objects:
                    intent_group = (kw.get('seed_intent_group') or "GENERAL_INTENT").strip().upper()
                    intent_coverage[intent_group] = intent_coverage.get(intent_group, 0) + 1
                cluster['intent_coverage'] = intent_coverage
                if intent_coverage:
                    ordered_groups = sorted(intent_coverage.items(), key=lambda item: item[1], reverse=True)
                    cluster['cluster_rationale'] = (
                        f"Primary intent coverage: {', '.join([f'{group}:{count}' for group, count in ordered_groups])}."
                    )
                if cluster['tool_potential_score'] < 0:
                    cluster['tool_potential_score'] = 0
                if cluster['tool_potential_score'] > 100:
                    cluster['tool_potential_score'] = 100
                
                # Ensure primary keyword is set validly
                if not cluster.get('primary_keyword'):
                     cluster['primary_keyword'] = self._sanitize_keyword_text(title)

                enriched_clusters.append(cluster)

            if not enriched_clusters:
                logger.warning("No valid clusters formed after enrichment.")
                # Fallback
                if keywords:
                     return [{
                         "cluster_title": "General Ideas",
                         "primary_keyword": keywords[0]['keyword'],
                         "keywords": [k['keyword'] for k in keywords[:15]],
                         "search_volume": sum(k.get('search_volume', 0) for k in keywords[:15]),
                         "cpc": 0.5,
                         "keyword_difficulty": 50,
                         "cluster_type": "decision",
                         "primary_user_outcome": "Evaluate broad options and choose next steps",
                         "serp_intent_match": "medium",
                         "tool_potential_score": 50,
                     }]
                return []

            logger.info(f"Formed {len(enriched_clusters)} enriched clusters with metrics.")
            return enriched_clusters

        except Exception as e:
            logger.error(f"Error clustering keywords: {e}")
            # Fallback to single cluster
            if keywords:
                return [{
                     "cluster_title": "General Ideas",
                     "primary_keyword": keywords[0]['keyword'],
                     "keywords": [k['keyword'] for k in keywords[:15]],
                     "search_volume": sum(k.get('search_volume', 0) for k in keywords[:15]),
                     "cpc": 0.0,
                     "keyword_difficulty": 50,
                     "cluster_type": "decision",
                     "primary_user_outcome": "Evaluate broad options and choose next steps",
                     "serp_intent_match": "medium",
                     "tool_potential_score": 50,
                }]
            return []

    async def verify_clusters(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 5: Verify Profitability (Trends + Monetization).
        """
        verified_clusters = []
        
        # Create verification task for a single cluster
        # Limit concurrency to 10 to avoid "Too many open files" or API rate limits
        semaphore = asyncio.Semaphore(10)

        async def verify_single_cluster(cluster):
            async with semaphore:
                primary_kw = cluster.get('primary_keyword')
                primary_kw = self._sanitize_keyword_text(primary_kw)
                if not primary_kw:
                    primary_kw = self._sanitize_keyword_text(cluster.get('cluster_title'))
                if not primary_kw:
                    return None
                logger.info(
                    "Cluster verification keyword selected cluster=%r primary_keyword=%r",
                    cluster.get('cluster_title'),
                    primary_kw
                )
                    
                # Run Trend & Monetization in parallel for this cluster
                trend_task = self.analyze_trend(primary_kw)
                monetization_task = self.check_monetization(primary_kw, cluster.get('cluster_title'))
                
                results = await asyncio.gather(trend_task, monetization_task, return_exceptions=True)
                
                trend_data = results[0]
                monetization = results[1]
                
                # Handle Trend Exceptions
                if isinstance(trend_data, Exception):
                    logger.error(f"Trend check error for {primary_kw}: {trend_data}")
                    trend_data = {"status": "FAIL", "reason": "Error checking trend"}
                
                # Handle Monetization Exceptions    
                if isinstance(monetization, Exception):
                    logger.error(f"Monetization check error for {primary_kw}: {monetization}")
                    monetization = {"status": "FAIL", "reason": "Error checking monetization"}

                # A. Trend Analysis Logic
                if trend_data['status'] == 'FAIL':
                    logger.info(f"Cluster '{cluster.get('cluster_title')}' failed trend check: {trend_data['reason']}")
                    # SAFE MODE CHANGE: Don't discard, just mark as warning
                    trend_data['label'] = f"⚠️ {trend_data['reason']}"
                
                # B. Monetization Check Logic
                if monetization['status'] == 'FAIL':
                     logger.info(f"Cluster '{cluster.get('cluster_title')}' failed monetization check: {monetization['reason']}")
                     # SAFE MODE CHANGE: Don't discard
                     if 'details' not in monetization: monetization['details'] = {}
                     monetization['details']['intent'] = f"⚠️ {monetization['reason']}"
                     
                # Enrich cluster
                cluster['trend_analysis'] = trend_data
                cluster['monetization'] = monetization
                return cluster

        # Create tasks for all clusters
        tasks = [verify_single_cluster(c) for c in clusters if c]
        
        # Run all cluster verifications in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for res in results:
            if isinstance(res, dict):
                verified_clusters.append(res)
            elif isinstance(res, Exception):
                logger.error(f"Cluster verification task failed: {res}")
            
        logger.info(f"Verified {len(verified_clusters)} clusters out of {len(clusters)} candidates.")
        return verified_clusters

    async def analyze_trend(self, keyword: str) -> Dict[str, Any]:
        """
        Check 12-month trend slope.
        """
        # Fetch trend data with timeout protection
        # Note: DataForSEO trends API might require dates. Let's assume default (last 12 mo) or specified.
        # We need past 12 months.
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        try:
            data = await asyncio.wait_for(
                dataforseo_api.get_keyword_trends(
                    [keyword],
                    date_from=start_date.strftime('%Y-%m-%d'),
                    date_to=end_date.strftime('%Y-%m-%d')
                ),
                timeout=15.0  # 15 second timeout for trend API call
            )
        except asyncio.TimeoutError:
            logger.warning(f"Trend analysis timed out for keyword '{keyword}'")
            return {
                "status": "PASS",
                "reason": "Trend analysis timeout",
                "slope": 0.0,
                "label": "Neutral (Timeout)",
                "historical_data": []
            }
        
        # Calculate slope
        slope = 0.0
        
        try:
            if data and isinstance(data, list) and len(data) > 0:
                trend_entry = data[0]
                items = trend_entry.get('items', [])
                
                if items:
                    values = []
                    for item in items:
                        # DataForSEO structure varies: 'value', 'interest', 'values'
                        val = item.get('value') or item.get('interest') or item.get('values')
                        
                        if isinstance(val, list) and len(val) > 0:
                            values.append(float(val[0]))
                        elif isinstance(val, (int, float)):
                             values.append(float(val))
                        
                        # Debug structure if extraction fails
                        if val is None:
                             logger.debug(f"Trend item extraction failed. Keys: {list(item.keys())}")
                    
                    if len(values) > 1:
                        slope = self._calculate_slope(values)
                        logger.info(f"Calculated trend slope for '{keyword}': {slope:.4f} (Points: {len(values)})")
                    else:
                        logger.warning(f"Not enough data points to calculate slope for '{keyword}'. Using default 0.0")
                else:
                    logger.warning(f"No trend items returned for '{keyword}'.")
            else:
                 logger.warning(f"Empty trend data for '{keyword}'.")
                 
        except Exception as e:
            logger.error(f"Error calculating slope for '{keyword}': {e}")
            slope = 0.0 # Default to neutral if calc fails
            
        # Decision Logic
        # Decision Logic - INFORMATIONAL ONLY (User Request)
        # We do not fail/kill based on trends anymore, just label them.
        status = "PASS"
        label = "Neutral"
        
        if slope < -0.2:
             label = "Downtrend"
        elif slope > 0.1:
             label = "Uptrend"
             
        # Add warning emoji for visual pop if downtrend, but keeps status PASS
        if label == "Downtrend":
            label = "📉 Downtrend"
        elif label == "Uptrend":
            label = "📈 Uptrend"

        return { 
            "status": status, 
            "reason": "Trend Analysis Complete", 
            "slope": slope, 
            "label": label, 
            "historical_data": values if 'values' in locals() else [] 
        }

    def _parse_cluster_response(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse delimited cluster text into structured format.
        Handles variations in LLM output formatting.
        """
        clusters = []

        # Normalize line endings and clean up
        text = text.replace('\r\n', '\n').strip()

        # Split by CLUSTER: marker
        # Pattern matches "CLUSTER:" followed by name, then "KEYWORDS:" followed by comma-separated list
        pattern = (
            r'CLUSTER:\s*(.+?)\n'
            r'OUTCOME:\s*(.+?)\n'
            r'TYPE:\s*(.+?)\n'
            r'INTENT_MATCH:\s*(.+?)\n'
            r'TOOL_POTENTIAL:\s*(.+?)\n'
            r'KEYWORDS:\s*(.+?)(?=\nCLUSTER:|\Z)'
        )
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        if matches:
            for name, outcome, cluster_type, intent_match, tool_potential, keywords_str in matches:
                # Clean up the keywords
                keywords = [k.strip().strip('- ') for k in keywords_str.split(',') if k.strip()]
                if keywords:  # Only add if we have keywords
                    tool_score = 50
                    try:
                        tool_score = int(re.findall(r"-?\d+", str(tool_potential))[0])
                    except Exception:
                        tool_score = 50
                    clusters.append({
                        'subtopic_name': name.strip(),
                        'seed_keywords': keywords,
                        'primary_user_outcome': outcome.strip(),
                        'cluster_type': cluster_type.strip().lower(),
                        'serp_intent_match': intent_match.strip().lower(),
                        'tool_potential_score': max(0, min(100, tool_score)),
                    })
        else:
            # Fallback: try simpler parsing if regex fails
            lines = text.split('\n')
            current_cluster = None
            current_keywords = []
            current_outcome = ""
            current_type = "decision"
            current_intent_match = "medium"
            current_tool_score = 50

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check for cluster header (various formats)
                if line.upper().startswith('CLUSTER:') or line.startswith('**') or ':' in line:
                    # Save previous cluster if exists
                    if current_cluster and current_keywords:
                        clusters.append({
                            'subtopic_name': current_cluster,
                            'seed_keywords': current_keywords,
                            'primary_user_outcome': current_outcome or f"Evaluate {current_cluster} and choose next steps",
                            'cluster_type': current_type,
                            'serp_intent_match': current_intent_match,
                            'tool_potential_score': current_tool_score,
                        })

                    # Extract new cluster name
                    if ':' in line:
                        current_cluster = line.split(':', 1)[1].strip().strip('* ')
                    else:
                        current_cluster = line.strip('* ')
                    current_keywords = []
                    current_outcome = ""
                    current_type = "decision"
                    current_intent_match = "medium"
                    current_tool_score = 50

                elif line.upper().startswith('OUTCOME:'):
                    current_outcome = line.split(':', 1)[1].strip()
                elif line.upper().startswith('TYPE:'):
                    current_type = line.split(':', 1)[1].strip().lower() or "decision"
                elif line.upper().startswith('INTENT_MATCH:'):
                    current_intent_match = line.split(':', 1)[1].strip().lower() or "medium"
                elif line.upper().startswith('TOOL_POTENTIAL:'):
                    potential = line.split(':', 1)[1].strip()
                    try:
                        current_tool_score = int(re.findall(r"-?\d+", potential)[0])
                    except Exception:
                        current_tool_score = 50
                    current_tool_score = max(0, min(100, current_tool_score))

                # Check for keywords line
                elif line.upper().startswith('KEYWORDS:') or line.startswith('-'):
                    # Extract keywords
                    if ':' in line:
                        kw_part = line.split(':', 1)[1]
                    else:
                        kw_part = line.lstrip('- ')

                    keywords = [k.strip() for k in kw_part.split(',') if k.strip()]
                    current_keywords.extend(keywords)

            # Don't forget the last cluster
            if current_cluster and current_keywords:
                clusters.append({
                    'subtopic_name': current_cluster,
                    'seed_keywords': current_keywords,
                    'primary_user_outcome': current_outcome or f"Evaluate {current_cluster} and choose next steps",
                    'cluster_type': current_type,
                    'serp_intent_match': current_intent_match,
                    'tool_potential_score': current_tool_score,
                })

        return clusters

    def _parse_monetization_response(self, text: str) -> Dict[str, Any]:
        """
        Parse delimited monetization text into structured format.
        Handles variations in LLM output formatting.
        """
        result = {
            "intent": "Commercial",  # Default
            "price_range": "Mid",    # Default
            "affiliate_categories": []
        }

        # Normalize line endings
        text = text.replace('\r\n', '\n').strip()
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse INTENT line
            if line.upper().startswith('INTENT:'):
                intent_val = line.split(':', 1)[1].strip()
                # Normalize to valid values
                intent_lower = intent_val.lower()
                if 'transactional' in intent_lower:
                    result['intent'] = 'Transactional'
                elif 'informational' in intent_lower:
                    result['intent'] = 'Informational'
                else:
                    result['intent'] = 'Commercial'  # Default/fallback

            # Parse PRICE_RANGE line
            elif line.upper().startswith('PRICE_RANGE:') or line.upper().startswith('PRICE RANGE:'):
                if ':' in line:
                    price_val = line.split(':', 1)[1].strip()
                else:
                    price_val = line.split(None, 1)[1].strip() if ' ' in line else line

                # Normalize to valid values
                price_lower = price_val.lower()
                if 'low' in price_lower:
                    result['price_range'] = 'Low'
                elif 'high' in price_lower:
                    result['price_range'] = 'High'
                else:
                    result['price_range'] = 'Mid'

            # Parse AFFILIATE_CATEGORIES line
            elif line.upper().startswith('AFFILIATE_CATEGORIES:') or line.upper().startswith('AFFILIATE CATEGORIES:'):
                if ':' in line:
                    cats_val = line.split(':', 1)[1].strip()
                else:
                    cats_val = line.split(None, 1)[1].strip() if ' ' in line else ''

                # Split by comma and clean
                if cats_val:
                    categories = [c.strip() for c in cats_val.split(',') if c.strip()]
                    result['affiliate_categories'] = categories

        # If no categories were found, add a default
        if not result['affiliate_categories']:
            result['affiliate_categories'] = ['General']

        return result

    def _calculate_slope(self, values: List[float]) -> float:
        """
        Simple linear regression slope (y = mx + b).
        Returns 'm'. normalized to range typically -1 to 1 for trend analysis checks.
        """
        n = len(values)
        if n < 2: return 0.0
        
        # X axis is just index 0..n-1
        xs = range(n)
        ys = values
        
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        denominator = sum((x - mean_x) ** 2 for x in xs)
        
        if denominator == 0: return 0.0
        
        m = numerator / denominator
        
        # Normalize: 'm' is change per unit time (per week/semimonth).
        # A slope of +2 means +2 interest points per week.
        # Over 52 weeks, that's huge. 
        # Let's keep raw 'm' but logic expects small numbers like 0.1.
        # If interest values are 0-100, a steep rise might be m=5.
        # Let's normalize by dividing by 100? Or just return raw m.
        # Step logic uses < -0.2. 
        # If m = -1 (losses 1 point per week), that's definitely failing.
        
        return m


    async def check_monetization(self, keyword: str, topic: str) -> Dict[str, Any]:
        """
        Ask LLM for intent AND fetch real affiliate programs.
        Uses delimited text format instead of JSON for better reliability.
        """
        # 1. LLM Analysis for Intent & Price
        prompt = f"""
        Analyze the keyword: '{keyword}' for the topic '{topic}'.

        1. Is the intent Transactional, Commercial, or Informational?
        2. If a user buys a product related to this, what is the estimated price range (Low: <$20, Mid: $20-$100, High: >$100)?
        3. List 2 potential affiliate categories (e.g., Amazon Home, ClickBank Crypto).

        Output Format (use EXACTLY this format, one per line):
        INTENT: Commercial
        PRICE_RANGE: Mid
        AFFILIATE_CATEGORIES: Category1, Category2

        Rules:
        - INTENT must be exactly: Transactional, Commercial, or Informational
        - PRICE_RANGE must be exactly: Low, Mid, or High
        - Return ONLY these three lines, no other text
        """
        monetization_result = { "status": "PASS", "details": {}, "offers": [] }

        try:
            # Run LLM analysis with timeout to prevent hanging
            response = await asyncio.wait_for(
                llm_service.generate_text(prompt, max_tokens=300),
                timeout=30.0  # 30 second timeout for LLM call
            )

            # Parse delimited text response
            analysis = self._parse_monetization_response(response.content)
            monetization_result['details'] = analysis

        except asyncio.TimeoutError:
            logger.warning(f"Monetization LLM call timed out for keyword '{keyword}'")
            monetization_result['details'] = {
                "intent": "Unknown (Timeout)",
                "price_range": "Unknown",
                "affiliate_categories": []
            }
        except Exception as e:
            logger.error(f"Monetization LLM analysis error for '{keyword}': {e}")
            # Fallback: provide default structure so rest of pipeline continues
            monetization_result['details'] = {
                "intent": "Commercial",  # Default assumption
                "price_range": "Mid",
                "affiliate_categories": ["General"],
                "parse_error": str(e)
            }
            # Fallback: provide default structure so rest of pipeline continues
            monetization_result['details'] = {
                "intent": "Commercial",  # Default assumption
                "price_range": "Mid",
                "affiliate_categories": ["General"],
                "parse_error": str(e)
            }

        # 2. Real Affiliate Search (New: Fix for 0 offers) - run with timeout
        try:
            from .affiliate_research_service import AffiliateResearchService
            app_affiliate_service = AffiliateResearchService()

            # Simple search to get count. limit to 5 to be fast.
            search_res = await asyncio.wait_for(
                app_affiliate_service.search_affiliate_programs(
                    search_term=keyword,
                    niche=None, # Auto-detect
                    ignore_cache=False
                ),
                timeout=10.0  # 10 second timeout for affiliate search
            )

            programs = search_res.get('programs', [])
            monetization_result['offers'] = programs
            monetization_result['offer_count'] = len(programs)
        except asyncio.TimeoutError:
            logger.warning(f"Affiliate search timed out for keyword '{keyword}'")
            monetization_result['offers'] = []
            monetization_result['offer_count'] = 0
        except Exception as e:
            logger.error(f"Affiliate search error for '{keyword}': {e}")
            monetization_result['offers'] = []
            monetization_result['offer_count'] = 0

        return monetization_result

# Global instance
semantic_expansion_service = SemanticExpansionService()
