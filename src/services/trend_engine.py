
import logging
import json
import asyncio
import re
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
            logger.error(f"Site/Project ID {site_id} not found in projects or wordPress_details")
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
        
        site_description = (
            site.get('site_description')
            or site.get('websiteDescription')
            or "A general interest website."
        )
        focus_topics = self._build_focus_topics(site, categories, site_description)
        
        # 2. DataForSEO Integration (Keyword Ideas - Standard Method)
        # Use niche-specific focus topics first, then categories as fallback.
        seed_keyword = focus_topics[0] if focus_topics else (categories[0] if categories else "trends")
        
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
        # Query should stay tightly aligned to the site's actual niche.
        news_query = " ".join(focus_topics[:3]) if focus_topics else (" ".join(categories[:3]) if categories else seed_keyword)
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
             # Fetch last 10 posts for this site/project.
             # Modern UI passes a `projects.id` UUID; legacy flows may still use `wordPress_details.id`.
             posts_query = self.supabase.table('wordpress_imported_posts').select('title').order('created_at', desc=True).limit(10)
             if site.get('_source_table') == 'projects' and site.get('user_id'):
                 posts_query = posts_query.eq('user_id', site['user_id'])
             else:
                 posts_query = posts_query.eq('wordpress_detail_id', site_id)
             rp_resp = posts_query.execute()
             if rp_resp.data:
                 recent_posts = [p['title'] for p in rp_resp.data]
        except Exception as e:
            logger.warning(f"Failed to fetch recent posts for context: {e}")

        logger.info("Synthesizing report with Gemini...")
        trend_report_content = await self._generate_synthesis(
            site_description=site_description,
            categories=categories,
            focus_topics=focus_topics,
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
        self._save_report(site, full_report)
        
        return full_report
    
    def _get_site_details(self, site_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a site's configuration from the modern `projects` table first,
        then fall back to legacy `wordPress_details`.
        """
        try:
            if not self.supabase:
                logger.error("Supabase client is not initialized")
                return None

            project_response = (
                self.supabase
                .table('projects')
                .select('*')
                .eq('id', site_id)
                .limit(1)
                .execute()
            )
            if project_response.data:
                return {
                    **project_response.data[0],
                    '_source_table': 'projects',
                }

            legacy_response = (
                self.supabase
                .table('wordPress_details')
                .select('*')
                .eq('id', site_id)
                .limit(1)
                .execute()
            )
            if legacy_response.data:
                return {
                    **legacy_response.data[0],
                    '_source_table': 'wordPress_details',
                }

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

    def _build_focus_topics(self, site: Dict[str, Any], categories: List[str], site_description: str) -> List[str]:
        """
        Build niche-specific topics from saved project metadata so external queries
        stay anchored to the actual site instead of drifting into generic trends.
        """
        candidates: List[str] = []

        target_keywords = site.get('target_keywords') or []
        if isinstance(target_keywords, str):
            try:
                target_keywords = json.loads(target_keywords)
            except Exception:
                target_keywords = [k.strip() for k in target_keywords.split(',') if k.strip()]

        if isinstance(target_keywords, list):
            candidates.extend([str(k).strip() for k in target_keywords if str(k).strip()])

        candidates.extend([c.strip() for c in categories if isinstance(c, str) and c.strip()])

        if site_description:
            cleaned = re.sub(r'\s+', ' ', site_description).strip()
            phrase_candidates = re.split(r'[.;]|\band\b', cleaned, flags=re.IGNORECASE)
            for phrase in phrase_candidates:
                phrase = phrase.strip(" ,:-")
                if not phrase:
                    continue

                lower_phrase = phrase.lower()
                skip_markers = (
                    "wellroost transforms",
                    "tailored to individual needs",
                    "their products and services",
                    "enjoy automation",
                    "lower utility bills",
                    "personalized home improvements",
                    "elevate your living experience",
                    "diy-friendly approach",
                    "creating your dream space",
                )
                if any(marker in lower_phrase for marker in skip_markers):
                    continue

                if 2 <= len(phrase.split()) <= 6:
                    candidates.append(phrase)

        seen = set()
        focus_topics = []
        for candidate in candidates:
            normalized = candidate.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            focus_topics.append(candidate.strip())
            if len(focus_topics) >= 6:
                break

        return focus_topics
    
    # ... (helper methods) ...

    async def _generate_synthesis(self, site_description: str, categories: List[str], focus_topics: List[str], keywords, news, pinterest, social_pulse, recent_posts: List[str] = []) -> Any:
        
        prompt = f"""
        Act as a content director and niche editor for the following website.

        SITE NICHE:
        Description: {site_description}
        Categories: {', '.join(categories)}
        Core Focus Topics: {', '.join(focus_topics)}

        We want to create NEW, TRENDING content that is tightly aligned to this site's niche.
        
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
        1. First infer the site's true niche from the description and core focus topics.
        2. Reject any keyword, headline, Pinterest idea, or social discussion that does NOT clearly relate to the site's niche.
        3. Do NOT suggest off-topic lifestyle content such as fashion, beauty, celebrity, dating, generic women's lifestyle, or unrelated wellness unless it is explicitly supported by the site's niche description.
        4. Favor topics connected to home improvement, smart home technology, energy management, home security, outdoor living, DIY upgrades, efficient home office setups, water systems, or waste solutions when the data supports them.
        5. Check the Reddit/Quora/LinkedIn discussions. What specific homeowner or DIY pain points are people complaining about?
        6. Suggest exactly 3 high-impact blog post topics that would make sense for this site to publish.
        7. For each topic, explain specifically why it fits THIS site, not just why it is trending generally.
        8. Include a 'Pain Point' section identifying the specific user frustration.
        9. If the source data is weak or partially irrelevant, still stay on-niche and use only the relevant fragments.
        10. Provide the output in strictly valid JSON format with the following structure:
        {{
            "topics": [
                {{
                    "title": "Topic Title",
                    "rationale": "Why it's trending and why it fits this site's niche...",
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

    def _save_report(self, site: Dict[str, Any], report: Dict[str, Any]):
        try:
             source_table = site.get('_source_table', 'wordPress_details')
             record_id = site.get('id')
             if not record_id:
                 logger.error("Cannot save trend report without a record id")
                 return

             self.supabase.table(source_table).update({'last_trend_report': report}).eq('id', record_id).execute()
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
