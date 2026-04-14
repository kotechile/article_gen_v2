
import logging
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.core.supabase_singleton import get_supabase_client
from src.integrations.dataforseo import DataForSEOAPI
from src.integrations.apify import ApifyClient
from src.integrations.llm.client import LLMClient

logger = logging.getLogger(__name__)

class TrendEngine:
    def __init__(self):
        self.supabase = get_supabase_client()
        self.dfs = DataForSEOAPI()
        self.apify = ApifyClient()
        
        # Initialize LLM with default from DB
        provider, model = self._get_default_llm_config()
        self.llm = LLMClient(default_provider=provider, default_model=model)

    async def get_whats_trending(self, site_id: str) -> Dict[str, Any]:
        """
        Generates a "Trend Report" for a specific site ID.
        """
        logger.info(f"Generating trend report for site_id: {site_id}")
        
        # 1. Database Extraction
        site = self._get_site_details(site_id)
        if not site:
            logger.error(f"Site ID {site_id} not found in wordPress_details")
            raise ValueError(f"Site ID {site_id} not found")

        # 'categories' might be a JSON array or text list. Assuming JSON array or comma-separated string.
        raw_categories = site.get('categories')
        categories = []
        if isinstance(raw_categories, list):
            categories = raw_categories
        elif isinstance(raw_categories, str):
            try:
                categories = json.loads(raw_categories)
            except:
                categories = [c.strip() for c in raw_categories.split(',')]
        
        site_description = site.get('site_description') or "A general interest website."
        
        # 2. DataForSEO Integration (Keyword Ideas - Standard Method)
        # Use first category as seed, or site description keywords
        seed_keyword = categories[0] if categories else "trends"
        
        logger.info(f"Fetching keyword ideas (Standard/Queued) for seed: {seed_keyword}")
        
        # Fetch broader set to filter for growth
        # Use Standard method for cost savings
        keyword_ideas = await self.dfs.get_keyword_ideas_standard(
            seed_keyword=seed_keyword,
            limit=100,
            filters=[["keyword_info.search_volume", ">", 100]], # Basic volume filter
            order_by=["keyword_info.search_volume,desc"]
        )
        
        # Filter for YoY Growth and Competition
        # User Logic: ["keyword_info.monthly_searches.11.search_volume", ">", "keyword_info.monthly_searches.0.search_volume"] (YoY growth)
        # Sort: Order by keyword_info.competition (low) and growth (high).
        
        growing_keywords = self._process_keywords_for_growth(keyword_ideas)
        top_growing = growing_keywords[:3]
        
        logger.info(f"Found {len(top_growing)} growing keywords.")

        # 3. News Aggregation (Standard Method)
        # Query: site's categories
        news_query = " ".join(categories[:3]) if categories else seed_keyword
        logger.info(f"Fetching news (Standard/Queued) for query: {news_query}")
        
        news_articles = await self.dfs.get_news_search_standard(keyword=news_query, limit=5)
        
        # 4. Pinterest Context (via Apify)
        pinterest_trends = []
        if top_growing:
            search_terms = [k['keyword'] for k in top_growing]
            logger.info(f"Fetching Pinterest trends for: {search_terms}")
            pinterest_trends = await self.apify.get_pinterest_trends(search_terms)

        # 5. Social Pulse (Reddit/Quora/LinkedIn)
        social_pulse = []
        
        # Reddit & Quora using DataForSEO (Standard SERP)
        # Search for: site:reddit.com "category" or site:quora.com "category"
        # Since categories is a list, let's use the first one + seed
        social_query_reddit = f'site:reddit.com "{seed_keyword}"'
        social_query_quora = f'site:quora.com "{seed_keyword}"'
        
        logger.info(f"Fetching Social Pulse (SERP Standard): {social_query_reddit}")
        reddit_results = await self.dfs.get_serp_standard(social_query_reddit, depth=10, time_period="last_month")
        for r in reddit_results: r['source'] = 'Reddit'
        
        logger.info(f"Fetching Social Pulse (SERP Standard): {social_query_quora}")
        quora_results = await self.dfs.get_serp_standard(social_query_quora, depth=10, time_period="last_month")
        for r in quora_results: r['source'] = 'Quora'
        
        social_pulse.extend(reddit_results)
        social_pulse.extend(quora_results)
        
        # LinkedIn using Apify
        linkedin_posts = []
        if top_growing:
             # Use top 1 keyword to save credits/time
             li_keyword = top_growing[0]['keyword']
             logger.info(f"Fetching LinkedIn posts (Apify) for: {li_keyword}")
             linkedin_posts = await self.apify.get_linkedin_posts([li_keyword], max_results=5)
             social_pulse.extend(linkedin_posts)

        # 6. Gemini 2.0 Synthesis (Updated for Pain Points)
        # Fetch recent articles for context to avoid duplication
        recent_posts = []
        try:
             # Fetch last 10 posts for this site
             rp_resp = self.supabase.table('wordpress_imported_posts').select('title').eq('wordpress_detail_id', site_id).order('created_at', desc=True).limit(10).execute()
             if rp_resp.data:
                 recent_posts = [p['title'] for p in rp_resp.data]
        except Exception as e:
            logger.warning(f"Failed to fetch recent posts for context: {e}")

        logger.info("Synthesizing report with Gemini...")
        trend_report_content = await self._generate_synthesis(
            site_description=site_description,
            categories=categories,
            keywords=top_growing,
            news=news_articles,
            pinterest=pinterest_trends,
            social_pulse=social_pulse,
            recent_posts=recent_posts
        )
        
        full_report = {
            "generated_at": datetime.utcnow().isoformat(),
            "site_id": site_id,
            "report_content": trend_report_content, 
            "raw_data": {
                "keywords": top_growing,
                "news": news_articles,
                "pinterest": pinterest_trends,
                "social_pulse": social_pulse
            }
        }
        
        # 7. Database Update
        logger.info("Saving report to DB...")
        self._save_report(site_id, full_report)
        
        return full_report
    
    def _get_site_details(self, site_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a site's configuration from `wordPress_details`.
        """
        try:
            if not self.supabase:
                logger.error("Supabase client is not initialized")
                return None

            response = (
                self.supabase
                .table('wordPress_details')
                .select('*')
                .eq('id', site_id)
                .limit(1)
                .execute()
            )

            if response.data:
                return response.data[0]

            return None
        except Exception as e:
            logger.error(f"Failed to fetch site details for {site_id}: {e}")
            return None

    def _process_keywords_for_growth(self, keyword_ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank keyword ideas by year-over-year growth while favoring lower competition.
        """
        processed_keywords = []

        competition_rank = {
            "LOW": 0,
            "MEDIUM": 1,
            "HIGH": 2,
            "UNKNOWN": 3,
        }

        for item in keyword_ideas or []:
            monthly_searches = item.get("monthly_searches") or []
            current_volume = item.get("search_volume") or 0
            prior_volume = 0

            if isinstance(monthly_searches, list) and len(monthly_searches) >= 12:
                latest = monthly_searches[0] or {}
                year_ago = monthly_searches[11] or {}
                current_volume = latest.get("search_volume", current_volume) or current_volume
                prior_volume = year_ago.get("search_volume", 0) or 0
            elif isinstance(monthly_searches, list) and len(monthly_searches) >= 2:
                latest = monthly_searches[0] or {}
                previous = monthly_searches[-1] or {}
                current_volume = latest.get("search_volume", current_volume) or current_volume
                prior_volume = previous.get("search_volume", 0) or 0

            if prior_volume > 0:
                growth_pct = ((current_volume - prior_volume) / prior_volume) * 100
            elif current_volume > 0:
                growth_pct = 100.0
            else:
                growth_pct = 0.0

            processed = {
                **item,
                "growth_pct": round(growth_pct, 1),
                "growth_formatted": f"{growth_pct:+.1f}%",
            }
            processed_keywords.append(processed)

        processed_keywords.sort(
            key=lambda kw: (
                competition_rank.get((kw.get("competition") or "UNKNOWN").upper(), 3),
                -(kw.get("growth_pct") or 0),
                -(kw.get("search_volume") or 0),
            )
        )

        return [kw for kw in processed_keywords if (kw.get("growth_pct") or 0) >= 0]
    
    # ... (helper methods) ...

    async def _generate_synthesis(self, site_description: str, categories: List[str], keywords, news, pinterest, social_pulse, recent_posts: List[str] = []) -> Any:
        
        prompt = f"""
        Act as a content director for the following website:
        Description: {site_description}
        Categories: {', '.join(categories)}

        We want to create NEW, TRENDING content.
        
        CONTEXT:
        We have already published articles on the following topics (DO NOT suggest duplicates):
        {json.dumps(recent_posts, indent=2) if recent_posts else "No recent articles found."}

        Based on the following raw trend data, suggest 3 high-impact blog post topics.
        
        GROWING KEYWORDS (High Growth, Low Competition):
        {json.dumps(keywords, indent=2)}

        RECENT NEWS HEADLINES:
        {json.dumps(news, indent=2)}

        PINTEREST TRENDS (Visual Context):
        {json.dumps(pinterest, indent=2)}
        
        SOCIAL PULSE (Reddit/Quora/LinkedIn):
        {json.dumps(social_pulse, indent=2)}

        Instructions:
        1. Check the Reddit/Quora/LinkedIn discussions. What specific problems or "pain points" are people complaining about?
        2. Suggest 3 high-impact blog post topics.
        3. For each, include a 'Why it's trending' section explaining the data.
        4. Include a 'Pain Point' section identifying the specific user frustration (e.g., "Why is X so hard?").
        5. Provide the output in strictly valid JSON format with the following structure:
        {{
            "topics": [
                {{
                    "title": "Topic Title",
                    "rationale": "Why it's trending...",
                    "suggested_angle": "How to approach it...",
                    "pain_point": "Specific user problem found in social data",
                    "source_signal": "Reddit/LinkedIn/News"
                }}
            ],
            "pain_points": [
                 {{
                    "source": "Reddit",
                    "question": "Example Question found",
                    "hook": "Proposed Content Hook"
                 }}
            ]
        }}
        """

        try:
            # Use defaults from LLMClient initialization
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                # Dynamic response format handling
            )
            
            # Parse JSON
            content = response.content
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "")
            
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {"error": "Failed to generate insights", "raw_content": str(e)}

    def _save_report(self, site_id: str, report: Dict[str, Any]):
        try:
             # Use jsonb column
             self.supabase.table('wordPress_details').update({'last_trend_report': report}).eq('id', site_id).execute()
        except Exception as e:
            logger.error(f"Failed to save report to DB: {e}")

    def _get_default_llm_config(self) -> tuple[str, str]:
        """Fetch default LLM provider and model from database"""
        default_provider = "gemini"
        default_model = "gemini-2.0-flash-exp"
        
        try:
            # Try to get default from llm_providers table
            resp = self.supabase.table('llm_providers').select('provider, model_name').eq('is_default', True).limit(1).execute()
            
            if resp.data and len(resp.data) > 0:
                settings = resp.data[0]
                if settings.get('provider') and settings.get('model_name'):
                    return settings['provider'], settings['model_name']
            
            # If no default set, try to find any active provider
            resp = self.supabase.table('llm_providers').select('provider, model_name').eq('is_active', True).limit(1).execute()
            if resp.data and len(resp.data) > 0:
                settings = resp.data[0]
                return settings['provider'], settings['model_name']
                
        except Exception as e:
            logger.warning(f"Failed to fetch LLM config from DB, using defaults: {e}")
            
        return default_provider, default_model
