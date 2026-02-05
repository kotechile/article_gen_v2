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
from ..models.enhanced_subtopic import EnhancedSubtopic, SubtopicSource

logger = logging.getLogger(__name__)

class SemanticExpansionService:
    """
    Service for semantic keyword expansion, filtering, clustering, and verification.
    """

    def __init__(self):
        pass

    async def expand_and_verify(self, topic: str, user_id: str) -> List[Dict[str, Any]]:
        """
        Main entry point: Expands a central topic into verified, profitable clusters.
        """
        logger.info(f"Starting semantic expansion for topic: {topic}")

        # Step 1: Semantic Explosion (LLM)
        seeds = await self.generate_seeds(topic)
        if not seeds:
            logger.warning("No seeds generated. Aborting.")
            return []
        
        # Step 2: Bulk Data Retrieval (DataForSEO)
        raw_keywords = await self.fetch_bulk_keyword_data(seeds)
        if not raw_keywords:
             logger.warning("No keyword data found. Aborting.")
             return []

        # Step 3: Profit Filtering (Math)
        filtered_keywords = self.filter_profitable_keywords(raw_keywords)
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
        clusters = await self.cluster_keywords(filtered_keywords)
        if not clusters:
             logger.warning("No clusters generated. Aborting.")
             return []

        # Step 5: Profitability Verification (Trends + LLM)
        verified_clusters = await self.verify_clusters(clusters)
        
        return verified_clusters

    async def generate_seeds(self, topic: str) -> List[str]:
        """
        Step 1: Ask LLM for 10 distinct sub-niches and 3-5 search terms for each.
        """
        prompt = f"""
        You are a Keyword Research Expert.
        I have a central topic: "{topic}".
        
        Task:
        1. Identify 10 distinct sub-niches related to this topic.
        2. For EACH sub-niche, generate 3-5 specific, long-tail search terms (seeds).
        
        Constraints:
        - DO NOT generate generic terms like "Best [Topic]" (e.g., "Best Gardening" is bad).
        - Focus on specific problems, questions, or product comparison queries.
        - Keywords must be 3-5 words long.
        
        Output Format:
        Return ONLY a flat list of these search terms, one per line. No other text.
        """
        try:
            response = await llm_service.generate_text(prompt=prompt, max_tokens=500)
            text = response.content.strip()
            # Split by newlines and clean
            seeds = [line.strip().lstrip('- ').strip() for line in text.split('\n') if line.strip()]
            # Deduplicate
            seeds = list(set(seeds))
            logger.info(f"Generated {len(seeds)} unique seeds for topic '{topic}'")
            return seeds
        except Exception as e:
            logger.error(f"Error generating seeds: {e}")
            return [topic] if topic else []

    async def fetch_bulk_keyword_data(self, seeds: List[str]) -> List[Dict[str, Any]]:
        """
        Step 2: Fetch related keywords for all seeds from DataForSEO.
        NOTE: This mimics bulk retrieval by making parallel calls for batches of seeds.
        """
        all_keywords = []
        
        # Limit seeds to avoid excessive API usage if LLM returns too many
        seeds_to_process = seeds[:50] 
        
        # Execute using Standard API (Queue-based, high volume)
        if seeds_to_process:
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
                    'is_fallback': True
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
            # Use our new robust endpoint
            bulk_metrics = await dataforseo_api.get_bulk_metrics_standard(candidates_to_enrich)
            
            # Map metrics back
            metrics_map = {m['keyword'].lower(): m for m in bulk_metrics if m.get('keyword')}
            updated_vol_count = 0
            
            for k in final_list:
                k_norm = k['keyword'].lower()
                if k_norm in metrics_map:
                    m = metrics_map[k_norm]
                    k['search_volume'] = m.get('search_volume', 0)
                    k['cpc'] = m.get('cpc', 0)
                    if not k.get('competition') or k.get('competition') == 'UNKNOWN':
                        k['competition'] = m.get('competition')
                    updated_vol_count += 1
            
            logger.info(f"Enriched Volume/CPC for {updated_vol_count} keywords.")

        return final_list

    async def _get_research_settings(self) -> Dict[str, Any]:
        """Fetch research settings from database or return defaults"""
        try:
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
                        vol = k.get('search_volume', 0)
                        cpc = k.get('cpc', 0)
                        safe_kd = max(new_kd, 1)
                        k['profitability_score'] = (vol * cpc) / safe_kd
                        updated_count += 1
                        
                        if updated_count == 1:
                            logger.info(f"DEBUG: Updated First Keyword '{k_text}' with KD: {new_kd}")
                
                logger.info(f"Updated KD for {updated_count} keywords.")

        except Exception as e:
            logger.error(f"Failed to enrich KD: {e}")


    async def cluster_keywords(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 4: Group keywords into concepts.
        """
        # Prepare list for prompt
        kw_list_str = "\n".join([f"- {k['keyword']} (Vol: {k['search_volume']}, KD: {k['keyword_difficulty']})" for k in keywords])
        
        prompt = f"""
        I have a list of high-potential keywords:
        {kw_list_str}
        
        Task:
        1. Group these keywords into specific "Article Concepts" or "Clusters".
        1. Group these keywords into specific "Subtopics" or "Clusters".
        2. For each subtopic, provide a list of "seed_keywords" that belong to it.
        
        Output Format (JSON List):
        [
            {{
                "subtopic_name": "Subtopic Name",
                "seed_keywords": ["kw1", "kw2", "kw3"]
            }},
            ...
        ]
        Return ONLY valid JSON.
        """
        
        try:
            response = await llm_service.generate_json(prompt=prompt, max_tokens=1500)
            
            # generate_json usually returns a dict or list. Assume list of clusters.
            subtopics = []
            if isinstance(response, list):
                subtopics = response
            elif isinstance(response, dict) and 'clusters' in response:
                 subtopics = response['clusters']
            elif isinstance(response, dict) and 'subtopics' in response:
                 subtopics = response['subtopics']
            else:
                 # Fallback if structure is weird
                 logger.warning(f"Unexpected JSON structure from clustering: {response}")
                 return []

            # Create lookup map for keyword data
            kw_map = {k['keyword'].lower(): k for k in keywords}

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
                    k_clean = k_str.lower().strip()
                    if k_clean in kw_map:
                        kw_data = kw_map[k_clean]
                        vol = kw_data.get('search_volume', 0)
                        cpc = kw_data.get('cpc', 0) or 0
                        kd = kw_data.get('keyword_difficulty', 0) or 0
                        
                        total_vol += vol
                        total_cpc += cpc
                        max_kd = max(max_kd, kd)
                        count += 1
                        
                        logger.info(f"DEBUG Match Found: {k_clean} | Vol: {vol}")

                        # Add full object to list
                        matched_kw_objects.append({
                            "keyword": k_str, 
                            "search_volume": vol,
                            "cpc": cpc,
                            "keyword_difficulty": kd,
                            "competition": kw_data.get('competition'),
                            "main_intent": kw_data.get('main_intent') or kw_data.get('intent', 'commercial'),
                            "profitability_score": kw_data.get('profitability_score')
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
                                    "search_volume": kw_obj.get('search_volume', 0),
                                    "cpc": kw_obj.get('cpc', 0),
                                    "keyword_difficulty": kw_obj.get('keyword_difficulty', 0),
                                    "competition": kw_obj.get('competition'),
                                    "main_intent": kw_obj.get('main_intent') or kw_obj.get('intent', 'commercial'),
                                    "profitability_score": kw_obj.get('profitability_score')
                                 })
                                 
                                 total_vol += kw_obj.get('search_volume', 0)
                                 total_cpc += kw_obj.get('cpc', 0)
                                 max_kd = max(max_kd, kw_obj.get('keyword_difficulty', 0))
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
                            "search_volume": kw_obj.get('search_volume', 0),
                            "cpc": kw_obj.get('cpc', 0),
                            "keyword_difficulty": kw_obj.get('keyword_difficulty', 0),
                            "profitability_score": kw_obj.get('profitability_score')
                         })
                         total_vol += kw_obj.get('search_volume', 0)
                         count += 1

                # Average CPC
                avg_cpc = total_cpc / count if count > 0 else 0.0
                
                # Enrich cluster object
                cluster['cluster_title'] = title
                # CRITICAL: Store objects in BOTH keys for compatibility
                cluster['keywords'] = matched_kw_objects if matched_kw_objects else raw_kws 
                cluster['seed_keywords'] = matched_kw_objects if matched_kw_objects else raw_kws
                cluster['primary_keyword'] = matched_kw_objects[0]['keyword'] if matched_kw_objects else (raw_kws[0] if raw_kws else title)
                cluster['search_volume'] = total_vol
                cluster['cpc'] = round(avg_cpc, 2)
                cluster['keyword_difficulty'] = max_kd
                
                # Ensure primary keyword is set validly
                if not cluster.get('primary_keyword'):
                     cluster['primary_keyword'] = title

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
                         "keyword_difficulty": 50
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
                     "keyword_difficulty": 50
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
                if not primary_kw:
                    return None
                    
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
        # Fetch trend data
        # Note: DataForSEO trends API might require dates. Let's assume default (last 12 mo) or specified.
        # We need past 12 months.
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        data = await dataforseo_api.get_keyword_trends(
            [keyword], 
            date_from=start_date.strftime('%Y-%m-%d'),
            date_to=end_date.strftime('%Y-%m-%d')
        )
        
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
        """
        # 1. LLM Analysis for Intent & Price
        prompt = f"""
        Analyze the keyword: '{keyword}' for the topic '{topic}'.
        1. Is the intent Transactional, Commercial, or Informational?
        2. If a user buys a product related to this, what is the estimated price range (Low: <$20, Mid: $20-$100, High: >$100)?
        3. List 2 potential affiliate categories (e.g., Amazon Home, ClickBank Crypto).
        
        Output JSON:
        {{
            "intent": "Commercial",
            "price_range": "Mid",
            "affiliate_categories": ["Cat1", "Cat2"]
        }}
        """
        monetization_result = { "status": "PASS", "details": {}, "offers": [] }
        
        try:
            # Run LLM analysis
            analysis = await llm_service.generate_json(prompt, max_tokens=300)
            monetization_result['details'] = analysis
            
            # 2. Real Affiliate Search (New: Fix for 0 offers)
            from .affiliate_research_service import AffiliateResearchService
            app_affiliate_service = AffiliateResearchService()
            
            # Simple search to get count. limit to 5 to be fast.
            search_res = await app_affiliate_service.search_affiliate_programs(
                search_term=keyword,
                niche=None, # Auto-detect
                ignore_cache=False
            )
            
            programs = search_res.get('programs', [])
            monetization_result['offers'] = programs
            monetization_result['offer_count'] = len(programs)
            
            return monetization_result
            
        except Exception as e:
            logger.error(f"Monetization check error: {e}")
            return { "status": "PASS", "details": {"error": str(e)}, "offers": [], "offer_count": 0 }

# Global instance
semantic_expansion_service = SemanticExpansionService()
