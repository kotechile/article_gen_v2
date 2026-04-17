
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
        
        # Initialize LLM using DB preferences + application_settings keys.
        self._llm_candidates = self._get_llm_runtime_candidates()
        if not self._llm_candidates:
            raise RuntimeError(
                "No LLM API key configured. Please add a key in Settings → Content Generation."
            )
        provider, model, api_key, base_url = self._llm_candidates[0]
        self.llm = LLMClient(
            default_provider=provider,
            default_model=model,
            api_key=api_key,
            base_url=base_url,
        )

    async def get_whats_trending(self, site_id: str, primary_category_id: Optional[str] = None, secondary_category_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates a "Trend Report" for a specific site ID.
        """
        logger.info(
            "trend_engine: generating site_id=%s primary_category_id=%s secondary_category_id=%s",
            site_id,
            primary_category_id,
            secondary_category_id,
        )

        # 1. Database Extraction
        site = self._get_site_details(site_id)
        if not site:
            logger.error(f"Site/Project ID {site_id} not found in projects or wordPress_details")
            raise ValueError(f"Site ID {site_id} not found")

        # Resolve selected category/subcategory names from project_categories table
        primary_category_name = None
        secondary_category_name = None
        if primary_category_id or secondary_category_id:
            resolved = self._resolve_category_names(primary_category_id, secondary_category_id)
            primary_category_name = resolved.get('primary_category_name')
            secondary_category_name = resolved.get('secondary_category_name')

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
        focus_topics = self._build_focus_topics(site, categories, site_description, primary_category_name=primary_category_name, secondary_category_name=secondary_category_name)
        logger.info(
            "trend_engine: scope primary_category=%s secondary_category=%s focus_topics=%s",
            primary_category_name,
            secondary_category_name,
            focus_topics,
        )
        
        # 2. DataForSEO Integration (Rising search signals - Standard Method)
        # Use niche-specific focus topics first, then categories as fallback.
        seed_keyword = focus_topics[0] if focus_topics else (categories[0] if categories else "trends")
        
        logger.info(f"Fetching keyword ideas (Standard/Queued) for seed: {seed_keyword}")
        
        # Fetch a broader set to derive "rising queries" signals.
        keyword_ideas = await self.dfs.get_keyword_ideas_standard(
            seed_keyword=seed_keyword,
            limit=100,
            filters=[["keyword_info.search_volume", ">", 50]],  # Keep broad, avoid tiny noise
            order_by=["keyword_info.search_volume,desc"],
        )
        
        # Rank for growth (do not treat as final SEO targets at this stage).
        growing_keywords = self._process_keywords_for_growth(keyword_ideas)
        top_growing = growing_keywords[:10]
        
        logger.info(f"Found {len(top_growing)} growing keywords.")

        # 3. News Aggregation (Standard Method)
        # Query should stay tightly aligned to the site's actual niche and selected category path.
        news_query = " ".join(focus_topics[:3]) if focus_topics else (" ".join(categories[:3]) if categories else seed_keyword)
        logger.info(f"Fetching news (Standard/Queued) for query: {news_query}")
        
        news_articles = await self.dfs.get_news_search_standard(keyword=news_query, limit=5)
        logger.info("trend_engine: news_results=%s query=%s", len(news_articles or []), news_query)
        
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

        # 6. LLM Synthesis: Seed themes for research (NOT SEO-optimized titles)
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
            recent_posts=recent_posts,
            primary_category_name=primary_category_name,
            secondary_category_name=secondary_category_name,
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

        try:
            topics = (trend_report_content or {}).get("topics") or []
            logger.info("trend_engine: synthesized_topics_count=%s", len(topics) if isinstance(topics, list) else "non_list")
            if isinstance(topics, list):
                preview = [str((t or {}).get("title") or "")[:80] for t in topics[:5]]
                logger.info("trend_engine: synthesized_topics_preview=%s", preview)
        except Exception:
            logger.warning("trend_engine: failed to log synthesized topics preview", exc_info=True)
        
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

    def _resolve_category_names(self, primary_category_id: Optional[str], secondary_category_id: Optional[str]) -> Dict[str, Optional[str]]:
        """Resolve category UUIDs to names from project_categories table."""
        result = {'primary_category_name': None, 'secondary_category_name': None}
        try:
            category_ids = [cid for cid in [primary_category_id, secondary_category_id] if cid]
            if not category_ids:
                return result

            resp = self.supabase.table('project_categories').select('id, name').in_('id', category_ids).execute()
            if resp.data:
                for cat in resp.data:
                    if cat['id'] == primary_category_id:
                        result['primary_category_name'] = cat['name']
                    if cat['id'] == secondary_category_id:
                        result['secondary_category_name'] = cat['name']
        except Exception as e:
            logger.warning(f"Failed to resolve category names: {e}")
        return result

    def _process_keywords_for_growth(self, keyword_ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank keyword ideas by year-over-year growth while lightly favoring lower competition.
        These are used only as directional "rising query" signals for trend synthesis.
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

    def _build_focus_topics(self, site: Dict[str, Any], categories: List[str], site_description: str, primary_category_name: Optional[str] = None, secondary_category_name: Optional[str] = None) -> List[str]:
        """
        Build niche-specific topics from saved project metadata so external queries
        stay anchored to the actual site instead of drifting into generic trends.
        """
        candidates: List[str] = []

        # Prioritize selected category/subcategory names first
        if secondary_category_name:
            candidates.append(secondary_category_name.strip())
        if primary_category_name:
            candidates.append(primary_category_name.strip())

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

    async def _generate_synthesis(
        self,
        site_description: str,
        categories: List[str],
        focus_topics: List[str],
        keywords,
        news,
        pinterest,
        social_pulse,
        recent_posts: List[str] = [],
        primary_category_name: Optional[str] = None,
        secondary_category_name: Optional[str] = None,
    ) -> Any:

        # Build the selected category line for the prompt
        selected_category_line = ""
        if primary_category_name:
            if secondary_category_name:
                selected_category_line = f"Selected Category: {primary_category_name} / {secondary_category_name}"
            else:
                selected_category_line = f"Selected Category: {primary_category_name}"

        # Build dynamic instruction #4 based on selected category
        category_instruction = ""
        if primary_category_name:
            if secondary_category_name:
                category_instruction = f"Favor topics tightly aligned with the selected category ({primary_category_name} / {secondary_category_name}) when the data supports them."
            else:
                category_instruction = f"Favor topics tightly aligned with the selected category ({primary_category_name}) when the data supports them."
        else:
            category_instruction = "Favor topics that are tightly aligned with the site's core niche and categories when the data supports them."

        prompt = f"""
Act as a content director and niche editor for the following website.

SITE NICHE:
Description: {site_description}
Categories: {', '.join(categories)}
Core Focus Topics: {', '.join(focus_topics)}
{selected_category_line}

We are in the FIRST stage of a research workflow. We need BROAD TREND THEMES (seed topics),
not SEO-optimized blog post titles. These seed topics will be expanded later into specific
article ideas and low-competition keywords.

CONTEXT:
We have already published articles on the following topics (avoid suggesting near-duplicates):
{json.dumps(recent_posts, indent=2) if recent_posts else "No recent articles found."}

RAW TREND SIGNALS (use as evidence, not as final titles):

RISING SEARCH QUERIES (directional signals, may include competitive terms):
{json.dumps(keywords, indent=2)}

RECENT NEWS RESULTS:
{json.dumps(news, indent=2)}

PININTEREST CONTEXT:
{json.dumps(pinterest, indent=2)}

SOCIAL PULSE (Reddit/Quora/LinkedIn):
{json.dumps(social_pulse, indent=2)}

Instructions:
1. Infer the site's niche from the description and focus topics.
2. Reject any signal that is not clearly related to the niche.
3. {category_instruction}
4. Do NOT output article headlines; do NOT output "how to ..." titles unless the theme truly demands it.
5. Synthesize 8–12 seed topics. Each seed topic should be a short 2–6 word theme label.
6. Each seed topic must be broad enough to generate many long-tail keywords later.
7. Include brief rationale and cite which sources contributed (News / Pinterest / Reddit / Quora / LinkedIn / Search).
8. If the signals are weak, stay on-niche and propose evergreen-but-timely themes that match the niche anyway.
9. Respond in strictly valid JSON with this structure:
{{
  "topics": [
    {{
      "title": "Seed theme label",
      "rationale": "1-2 sentences tying it to the niche and why now",
      "source_signals": ["News", "Reddit"],
      "related_terms": ["optional", "2-5 short terms"]
    }}
  ]
}}
"""

        try:
            response = None
            last_error: Optional[Exception] = None

            # Try provider chain (Gemini -> OpenAI -> Anthropic -> Perplexity) and fall back on auth/key failures.
            candidates = self._llm_candidates or self._get_llm_runtime_candidates()
            for provider, model, api_key, base_url in candidates:
                try:
                    if base_url:
                        # For OpenAI-compatible base URLs (e.g., Perplexity), create a one-off client.
                        tmp = LLMClient(
                            default_provider=provider,
                            default_model=model,
                            api_key=api_key,
                            base_url=base_url,
                        )
                        response = await tmp.generate(prompt=prompt, temperature=0.7)
                    else:
                        response = await self.llm.generate(
                            prompt=prompt,
                            temperature=0.7,
                            provider=provider,
                            model=model,
                            api_key=api_key,
                        )
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    if self._is_llm_auth_error(e):
                        logger.warning(
                            "LLM auth/key error for provider=%s model=%s; trying next provider. err=%s",
                            provider,
                            model,
                            str(e),
                        )
                        continue
                    raise

            if response is None:
                raise last_error or RuntimeError("LLM synthesis failed (no providers succeeded)")
            
            # Parse JSON
            content = response.content
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "")
            parsed = json.loads(content)

            topics = (parsed or {}).get("topics")
            if not isinstance(topics, list) or not topics:
                raise ValueError(f"LLM synthesis returned no topics (type={type(topics).__name__})")
            if not any(isinstance(t, dict) and str(t.get("title") or "").strip() for t in topics):
                raise ValueError("LLM synthesis topics missing required 'title' fields")

            return parsed
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}", exc_info=True)
            raise

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

    def _get_llm_runtime_candidates(self) -> list[tuple[str, str, str, Optional[str]]]:
        """
        Resolve a provider/model/API-key chain.

        Requirement (for every LLM API call):
        1) From `llm_providers` get the row where `is_default = true`
           and read `provider`, `model_name`, and `api_keys_id`
        2) From `api_keys` read `key_value` for that `api_keys_id`
        """
        preference = self._get_default_llm_preference_row()
        provider_preference = self._normalize_provider(preference.get("provider"))
        model_preference = preference.get("model_name") or ""

        candidates: list[tuple[str, str, str, Optional[str]]] = []

        # Default provider must have an api_keys_id.
        api_keys_id = preference.get("api_keys_id") or preference.get("api_key_id")
        if not api_keys_id:
            logger.warning("Default llm_providers row has no api_keys_id (provider=%s model=%s)", provider_preference, model_preference)
            return []

        key_value = self._get_api_key_value(api_keys_id)
        if not key_value:
            logger.warning("api_keys.key_value not found for default llm_providers.api_keys_id=%s", api_keys_id)
            return []

        if provider_preference == "perplexity":
            # Perplexity is OpenAI-compatible (OpenAI ChatCompletions) via a different base_url.
            candidates.append(("openai", model_preference or "llama-3.1-sonar-small-128k-online", key_value, "https://api.perplexity.ai"))
        else:
            candidates.append((provider_preference, model_preference, key_value, None))

        return candidates

    def _get_default_llm_preference_row(self) -> Dict[str, Any]:
        """Fetch preferred LLM provider/model from `llm_providers` (may include api_keys_id)."""
        try:
            resp = (
                self.supabase.table("llm_providers")
                .select("provider, model_name, api_keys_id, api_key_id")
                .eq("is_default", True)
                .limit(1)
                .execute()
            )
            if resp.data:
                row = resp.data[0] or {}
                if row.get("provider") and row.get("model_name"):
                    return row
        except Exception as e:
            logger.warning("Failed to fetch default LLM preference from DB: %s", e)

        return {}

    def _get_api_key_value(self, api_keys_id: Any) -> Optional[str]:
        """Fetch key_value from api_keys table by id."""
        try:
            res = (
                self.supabase.table("api_keys")
                .select("key_value")
                .eq("id", api_keys_id)
                .limit(1)
                .execute()
            )
            if res.data:
                key_value = (res.data[0] or {}).get("key_value")
                if isinstance(key_value, str):
                    key_value = key_value.strip().strip('"').strip("'")
                return key_value
        except Exception as e:
            logger.warning("Failed to load api_keys id=%s: %s", api_keys_id, e)
        return None

    def _normalize_provider(self, provider: Optional[str]) -> str:
        """Map legacy/alias provider names to the internal provider id."""
        p = (provider or "").strip().lower()
        if p in ("gemini", "google"):
            return "gemini"
        if p in ("openai", "anthropic", "perplexity"):
            return p
        # Default to gemini because it has broad availability in our settings table.
        return "gemini"

    def _get_application_settings(self) -> Dict[str, Any]:
        """Load application_settings row (id=1)."""
        try:
            res = (
                self.supabase.table("application_settings")
                .select(
                    "geminiKey, geminiModel, openAIKey, openAIModel, perplexityAI_key, perplexityModel, claudeKey"
                )
                .eq("id", 1)
                .single()
                .execute()
            )
            return res.data or {}
        except Exception as e:
            logger.warning("Failed to load application_settings for LLM config: %s", e)
            return {}

    def _resolve_llm_from_settings(
        self,
        *,
        settings: Dict[str, Any],
        preferred_provider: str,
        preferred_model: str,
    ) -> Optional[tuple[str, str, str, Optional[str]]]:
        """
        Return (provider, model, api_key, base_url) or None.

        Tries preferred provider first, then falls back in priority order.
        """
        def resolve(provider: str, model: str) -> Optional[tuple[str, str, str, Optional[str]]]:
            provider = self._normalize_provider(provider)

            if provider == "gemini":
                api_key = settings.get("geminiKey")
                if not api_key:
                    return None
                model_name = model or settings.get("geminiModel") or "gemini-1.5-flash"
                return ("gemini", model_name, api_key, None)

            if provider == "openai":
                api_key = settings.get("openAIKey")
                if not api_key:
                    return None
                model_name = model or settings.get("openAIModel") or "gpt-4o-mini"
                return ("openai", model_name, api_key, None)

            if provider == "anthropic":
                api_key = settings.get("claudeKey")
                if not api_key:
                    return None
                return ("anthropic", model or "claude-3-haiku-20240307", api_key, None)

            if provider == "perplexity":
                api_key = settings.get("perplexityAI_key")
                if not api_key:
                    return None
                model_name = model or settings.get("perplexityModel") or "llama-3.1-sonar-small-128k-online"
                # Perplexity is OpenAI-compatible.
                return ("openai", model_name, api_key, "https://api.perplexity.ai")

            return None

        # 1) Preferred provider/model from DB
        preferred = resolve(preferred_provider, preferred_model)
        if preferred:
            return preferred

        # 2) Fallback order (match /api/ai/propose-topics behavior)
        for provider, model in [
            ("gemini", settings.get("geminiModel") or "gemini-1.5-flash"),
            ("openai", settings.get("openAIModel") or "gpt-4o-mini"),
            ("anthropic", "claude-3-haiku-20240307"),
            ("perplexity", settings.get("perplexityModel") or "llama-3.1-sonar-small-128k-online"),
        ]:
            resolved = resolve(provider, model)
            if resolved:
                return resolved

        return None

    def _is_llm_auth_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(s in msg for s in [
            "api key expired",
            "api_key_invalid",
            "apikey_invalid",
            "invalid api key",
            "authentication failed",
            "unauthorized",
            "forbidden",
        ])
