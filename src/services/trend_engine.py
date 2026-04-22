
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
        
        # Initialize LLM using the default config from llm_providers/api_keys.
        self._llm_candidates = self._get_llm_runtime_candidates()
        if not self._llm_candidates:
            raise RuntimeError(
                "No default LLM API key configured. Set llm_providers.is_default=true and attach api_keys.key_value via llm_providers.api_keys_id."
            )
        provider, model, api_key, base_url = self._llm_candidates[0]
        self.llm = LLMClient(
            default_provider=provider,
            default_model=model,
            api_key=api_key,
            base_url=base_url,
        )

    async def get_whats_trending(
        self,
        site_id: str,
        primary_category_id: Optional[str] = None,
        secondary_category_id: Optional[str] = None,
        project_name: Optional[str] = None,
        project_description: Optional[str] = None,
        niche_description: Optional[str] = None,
        primary_category_name: Optional[str] = None,
        primary_category_description: Optional[str] = None,
        secondary_category_name: Optional[str] = None,
        secondary_category_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates a "Trend Report" for a specific site ID.
        """
        logger.info(
            "trend_engine: generating site_id=%s primary_category_id=%s secondary_category_id=%s primary_category_name=%s secondary_category_name=%s",
            site_id,
            primary_category_id,
            secondary_category_id,
            primary_category_name,
            secondary_category_name,
        )

        # 1. Database Extraction
        site = self._get_site_details(site_id)
        if not site:
            logger.error(f"Site/Project ID {site_id} not found in projects or wordPress_details")
            raise ValueError(f"Site ID {site_id} not found")

        # Resolve selected category/subcategory context from request payload and/or project_categories table
        category_context = self._resolve_category_context(
            primary_category_id=primary_category_id,
            secondary_category_id=secondary_category_id,
            primary_category_name=primary_category_name,
            primary_category_description=primary_category_description,
            secondary_category_name=secondary_category_name,
            secondary_category_description=secondary_category_description,
        )
        primary_category_name = category_context.get('primary_category_name')
        secondary_category_name = category_context.get('secondary_category_name')
        primary_category_description = category_context.get('primary_category_description')
        secondary_category_description = category_context.get('secondary_category_description')

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
            project_description
            or niche_description
            or site.get('site_description')
            or site.get('websiteDescription')
            or "A general interest website."
        )
        focus_topics = self._build_focus_topics(
            site,
            categories,
            site_description,
            primary_category_name=primary_category_name,
            secondary_category_name=secondary_category_name,
            primary_category_description=primary_category_description,
            secondary_category_description=secondary_category_description,
        )
        logger.info(
            "trend_engine: scope project=%s primary_category=%s secondary_category=%s focus_topics=%s",
            project_name or site.get('domain') or site.get('app_name'),
            primary_category_name,
            secondary_category_name,
            focus_topics,
        )
        
        # 2. Discovery terms only (no topic-stage keyword survey/validation).
        # Keep the trend path data-driven via news/social signals, but avoid
        # spending DataForSEO credits on keyword surveys that are not reused
        # downstream.
        discovery_terms = [term for term in (focus_topics[:4] or categories[:4]) if str(term).strip()]
        if not discovery_terms:
            discovery_terms = ["market trends"]
        seed_keyword = discovery_terms[0]

        # 3. News Aggregation (Standard Method)
        # Query should stay tightly aligned to the site's actual niche and selected category path.
        news_query = " ".join(focus_topics[:3]) if focus_topics else (" ".join(categories[:3]) if categories else seed_keyword)
        logger.info(f"Fetching news (Standard/Queued) for query: {news_query}")
        
        news_articles = await self.dfs.get_news_search_standard(keyword=news_query, limit=5)
        logger.info("trend_engine: news_results=%s query=%s", len(news_articles or []), news_query)
        
        # 4. Pinterest Context (via Apify)
        pinterest_trends = []
        if discovery_terms:
            search_terms = discovery_terms[:3]
            logger.info("Fetching Pinterest trends for discovery terms: %s", search_terms)
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
        if discovery_terms:
             # Use first discovery term to keep requests lightweight.
             li_keyword = discovery_terms[0]
             logger.info("Fetching LinkedIn posts (Apify) for discovery term: %s", li_keyword)
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
            discovery_terms=discovery_terms,
            news=news_articles,
            pinterest=pinterest_trends,
            social_pulse=social_pulse,
            recent_posts=recent_posts,
            primary_category_name=primary_category_name,
            secondary_category_name=secondary_category_name,
            primary_category_description=primary_category_description,
            secondary_category_description=secondary_category_description,
        )
        
        full_report = {
            "generated_at": datetime.utcnow().isoformat(),
            "site_id": site_id,
            "report_content": trend_report_content, 
            "raw_data": {
                "keywords": [],
                "discovery_terms": discovery_terms,
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

    def _resolve_category_context(
        self,
        primary_category_id: Optional[str],
        secondary_category_id: Optional[str],
        primary_category_name: Optional[str],
        primary_category_description: Optional[str],
        secondary_category_name: Optional[str],
        secondary_category_description: Optional[str],
    ) -> Dict[str, Optional[str]]:
        """Resolve category UUIDs to names/descriptions from project_categories table, while honoring request-provided context."""
        result = {
            'primary_category_name': primary_category_name.strip() if isinstance(primary_category_name, str) and primary_category_name.strip() else None,
            'secondary_category_name': secondary_category_name.strip() if isinstance(secondary_category_name, str) and secondary_category_name.strip() else None,
            'primary_category_description': primary_category_description.strip() if isinstance(primary_category_description, str) and primary_category_description.strip() else None,
            'secondary_category_description': secondary_category_description.strip() if isinstance(secondary_category_description, str) and secondary_category_description.strip() else None,
        }
        try:
            category_ids = [cid for cid in [primary_category_id, secondary_category_id] if cid]
            if not category_ids:
                return result

            resp = self.supabase.table('project_categories').select('id, name, description').in_('id', category_ids).execute()
            if resp.data:
                for cat in resp.data:
                    if cat['id'] == primary_category_id:
                        if not result.get('primary_category_name'):
                            result['primary_category_name'] = cat.get('name')
                        if not result.get('primary_category_description'):
                            result['primary_category_description'] = cat.get('description')
                    if cat['id'] == secondary_category_id:
                        if not result.get('secondary_category_name'):
                            result['secondary_category_name'] = cat.get('name')
                        if not result.get('secondary_category_description'):
                            result['secondary_category_description'] = cat.get('description')
        except Exception as e:
            logger.warning(f"Failed to resolve category context: {e}")
        return result

    def _build_focus_topics(
        self,
        site: Dict[str, Any],
        categories: List[str],
        site_description: str,
        primary_category_name: Optional[str] = None,
        secondary_category_name: Optional[str] = None,
        primary_category_description: Optional[str] = None,
        secondary_category_description: Optional[str] = None,
    ) -> List[str]:
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
        if secondary_category_description:
            candidates.extend(self._extract_simple_phrases(secondary_category_description))
        if primary_category_description:
            candidates.extend(self._extract_simple_phrases(primary_category_description))

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

    def _extract_simple_phrases(self, text: str, max_phrases: int = 4) -> List[str]:
        """Extract short, search-like phrases from descriptive text."""
        if not text:
            return []
        cleaned = re.sub(r'\s+', ' ', str(text)).strip()
        if not cleaned:
            return []
        chunks = re.split(r'[.;:|]|\band\b', cleaned, flags=re.IGNORECASE)
        phrases: List[str] = []
        for chunk in chunks:
            phrase = chunk.strip(" ,-")
            if not phrase:
                continue
            words = phrase.split()
            if 2 <= len(words) <= 6:
                phrases.append(" ".join(words))
            if len(phrases) >= max_phrases:
                break
        return phrases

    def _normalize_topic_title_plain_language(self, title: str) -> str:
        """Reduce consultant-speak in topic titles."""
        if not title:
            return ""
        cleaned = re.sub(r"\s+", " ", str(title).strip())
        replacements = {
            r"\bframework\b": "guide",
            r"\baudit\b": "checklist",
            r"\bscenario\b": "plan",
            r"\barbitrage\b": "cost gap",
            r"\boptimization\b": "improvements",
            r"\bplaybook\b": "step-by-step plan",
            r"\bsolvency\b": "financial stability",
        }
        normalized = cleaned
        for pattern, value in replacements.items():
            normalized = re.sub(pattern, value, normalized, flags=re.IGNORECASE)
        return re.sub(r"\s{2,}", " ", normalized).strip(" -:")
    
    # ... (helper methods) ...

    async def _generate_synthesis(
        self,
        site_description: str,
        categories: List[str],
        focus_topics: List[str],
        discovery_terms: List[str],
        news,
        pinterest,
        social_pulse,
        recent_posts: List[str] = [],
        primary_category_name: Optional[str] = None,
        secondary_category_name: Optional[str] = None,
        primary_category_description: Optional[str] = None,
        secondary_category_description: Optional[str] = None,
    ) -> Any:

        # Build the selected category line for the prompt
        selected_category_line = ""
        if primary_category_name:
            if secondary_category_name:
                selected_category_line = f"Selected Category: {primary_category_name} / {secondary_category_name}"
            else:
                selected_category_line = f"Selected Category: {primary_category_name}"
        selected_category_description_lines = []
        if primary_category_description:
            selected_category_description_lines.append(f"Primary Category Description: {primary_category_description}")
        if secondary_category_description:
            selected_category_description_lines.append(f"Sub-Category Description: {secondary_category_description}")

        # Build dynamic instruction #4 based on selected category
        category_instruction = ""
        if primary_category_name:
            if secondary_category_name:
                category_instruction = f"Favor topics tightly aligned with the selected category ({primary_category_name} / {secondary_category_name}) and descriptions when the data supports them."
            else:
                category_instruction = f"Favor topics tightly aligned with the selected category ({primary_category_name}) and description when the data supports them."
        else:
            category_instruction = "Favor topics that are tightly aligned with the site's core niche and categories when the data supports them."

        prompt = f"""
Act as a content director and niche editor for the following website.

SITE NICHE:
Description: {site_description}
Categories: {', '.join(categories)}
Core Focus Topics: {', '.join(focus_topics)}
{selected_category_line}
{chr(10).join(selected_category_description_lines)}

We are in the FIRST stage of a research workflow. We need BROAD TREND THEMES (seed topics),
not SEO-optimized blog post titles. These seed topics will be expanded later into specific
article ideas and low-competition keywords.

CONTEXT:
We have already published articles on the following topics (avoid suggesting near-duplicates):
{json.dumps(recent_posts, indent=2) if recent_posts else "No recent articles found."}

RAW TREND SIGNALS (use as evidence, not as final titles):

DISCOVERY TERMS (category-aligned directional seeds, not validated keyword targets):
{json.dumps(discovery_terms or [], indent=2)}

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
4. Hard scope guard: every topic MUST pass this test:
   - Explicitly relevant to the selected category/sub-category descriptions.
   - If unsure, discard and replace with a safer in-scope topic.
5. Do NOT output article headlines; do NOT output "how to ..." titles unless the theme truly demands it.
6. Synthesize 8–12 seed topics. Each seed topic should be a short 2–6 word theme label.
7. Each seed topic must be broad enough to generate many long-tail keywords later.
8. Use plain language. Avoid consultant-speak/corporate wording in titles (e.g., avoid "framework", "operating model", "optimization" unless absolutely necessary).
9. Include brief rationale and cite which sources contributed (News / Pinterest / Reddit / Quora / LinkedIn / Search).
10. If the signals are weak, stay on-niche and propose evergreen-but-timely themes that match the niche anyway.
11. Respond in strictly valid JSON with this structure:
{{
  "topics": [
    {{
      "title": "Seed theme label",
      "rationale": "1-2 sentences tying it to the niche and why now",
      "source_signals": ["News", "Reddit"],
      "related_terms": ["optional", "2-5 short terms"],
      "intent_bucket": "informational_decision|commercial_evaluation|decision_financial|solution_enablement",
      "decision_focus": "One sentence describing the user decision this theme supports",
      "angle_question": "A concrete question to answer in decomposition",
      "value_layer_tags": ["roi-focused", "cost-vs-value"]
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

            for topic in topics:
                if not isinstance(topic, dict):
                    continue
                topic["title"] = self._normalize_topic_title_plain_language(str(topic.get("title") or ""))

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
        api_keys_id = preference.get("api_keys_id")
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
                .select("provider, model_name, api_keys_id")
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
