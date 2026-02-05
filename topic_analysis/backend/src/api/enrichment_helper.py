import logging
import asyncio
from uuid import UUID
from ..services.trend_service import TrendService
from ..services.enhanced_affiliate_research_service import EnhancedAffiliateResearchService
from ..services.subtopics_service import SubtopicsService
from ..services.llm.llm_service import llm_service

logger = logging.getLogger(__name__)

# Instantiate services
trend_service = TrendService()
enhanced_affiliate_service = EnhancedAffiliateResearchService()
subtopics_service = SubtopicsService()

async def _process_single_subtopic_enrichment(sub_id: str, sub_name: str, sub_data: dict, user_id: UUID) -> bool:
    """
    Helper function to process enrichment for a single subtopic.
    Returns True if successful, False otherwise.
    """
    try:
        # Execute Trend and Affiliate research in parallel to reduce total wait time
        # Both independent, so we can run them concurrently
        
        # Shared: Try to get the "Top Keyword" for this subtopic from the DB
        # Users search for keywords, not cluster names (descriptions)
        top_keyword = await subtopics_service.get_top_keyword(UUID(sub_id))
        
        # Determine Broad Topic: Prefer Top Keyword, else Subtopic Name
        source_term = top_keyword if top_keyword else sub_name
        broad_topic = source_term

        # Enhaced Logic: If the term is long (>2 words), ask LLM to canonicalize it for better Trend data
        # E.g. "homemade to kill weeds" (from DB) -> "Homemade Weed Killer"
        if len(source_term.split()) > 2:
             try:
                 prompt = f"Convert '{source_term}' into the shortest, most popular 2-4 word Google search term. Example: 'Homemade Natural Weed Killers' -> 'Homemade Weed Killer'. Return ONLY the term."
                 response = await llm_service.generate_text(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=100
                 )
                 if response and hasattr(response, 'content'):
                     candidate = response.content.strip()
                     if candidate:
                        broad_topic = candidate
                 elif isinstance(response, str) and response.strip():
                     broad_topic = response.strip()
             except Exception as e:
                 logger.warning(f"LLM Broad Topic extraction failed for {source_term}: {e}")
                 # Fallback to source_term (no change)
        
        # Clean broad_topic (remove quotes if any)
        if broad_topic:
            broad_topic = broad_topic.replace('"', '').replace("'", "").strip()
            
        logger.info(f"Using Broad Topic '{broad_topic}' for enrichment (Original: '{sub_name}')")


        async def fetch_affiliate():
            try:
                # Use Broad Topic unless Top Keyword is better?
                # Broad topic is derived from top keyword if present.
                search_term = broad_topic if broad_topic else sub_name
                    
                logger.info(f"Using search term '{search_term}' for affiliate discovery")
                
                # 120s timeout for Affiliate Discovery
                return await asyncio.wait_for(
                    enhanced_affiliate_service.intelligent_offer_discovery(
                        search_terms=[search_term],
                        user_id=str(user_id),
                        research_scope="comprehensive",
                        max_offers=10,
                        ignore_cache=True
                    ),
                    timeout=120.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"Affiliate enrichment timed out for {sub_name}")
                return {}
            except Exception as e:
                logger.warning(f"Affiliate enrichment failed for {sub_name}: {e}")
                return {}

        # Launch ONLY affiliate task
        logger.info(f"DEBUG: Starting Affiliate Enrichment for {sub_name}...")
        discovery_result = await fetch_affiliate()
        
        # --- Process Affiliate Results ---
        offer_count = discovery_result.get("discovered_programs", 0)
        logger.info(f"DEBUG: Affiliate Discovery Result for {sub_name}: Found {offer_count} offers. Keys: {list(discovery_result.keys())}")
        sub_data["affiliate_offer_count"] = offer_count
        
        # Extract recommended offers
        offers = discovery_result.get("recommended_offers", [])
        
        # Initialize or update monetization_data
        if "monetization_data" not in sub_data or not sub_data["monetization_data"]:
            sub_data["monetization_data"] = {}
            
        # Add offers to monetization_data
        sub_data["monetization_data"]["offers"] = offers
        
        if offer_count > 0:
            current_viability = sub_data.get("viability_score", 0)
            if current_viability is None:
                current_viability = 0
            boost = min(offer_count * 5, 30)
            sub_data["viability_score"] = min(current_viability + boost, 100)

        # --- C. Save Updates ---
        await subtopics_service.update(
            subtopic_id=UUID(sub_id),
            update_data={
                "search_volume": sub_data.get("search_volume"),
                "affiliate_offer_count": sub_data.get("affiliate_offer_count"),
                "viability_score": sub_data.get("viability_score"),
                "monetization_data": sub_data.get("monetization_data") # Persist the offers
            },
            user_id=user_id
        )
        return True
        
    except Exception as e:
        logger.error(f"Enrichment worker failed for {sub_name}: {e}")
        return False
