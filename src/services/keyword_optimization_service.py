"""
Keyword Optimization Service for Content Generator V2.

Provides:
1. DataForSEO keyword discovery and competitiveness analysis for existing/imported articles.
2. Direct keyword lookup with search volume, KD, CPC, and intent.
3. Non-destructive AI keyword weaving to naturally integrate selected keywords into article HTML.
"""

from __future__ import annotations

import os
import re
import json
import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from src.integrations.dataforseo import dataforseo_api

logger = logging.getLogger(__name__)


def calculate_opportunity_score(search_volume: Optional[int], keyword_difficulty: Optional[float]) -> int:
    """Calculate an achievability/opportunity score (0-100) favoring low KD and high volume."""
    vol = search_volume if search_volume is not None else 0
    kd = keyword_difficulty if keyword_difficulty is not None else 50

    vol_score = min(vol / 40.0, 100.0) if vol > 0 else 0.0
    kd_ease = max(0.0, 100.0 - kd)

    # 60% ease (low KD) + 40% search volume
    raw_score = (kd_ease * 0.60) + (vol_score * 0.40)
    return int(max(0, min(100, round(raw_score))))


def count_keyword_occurrences(text: str, keyword: str) -> int:
    """Count exact/stemmed occurrences of a keyword in text or HTML."""
    if not text or not keyword:
        return 0
    # Strip HTML tags
    clean_text = re.sub(r"<[^>]+>", " ", text).lower()
    kw_lower = keyword.strip().lower()
    if not kw_lower:
        return 0
    pattern = r"\b" + re.escape(kw_lower) + r"\b"
    return len(re.findall(pattern, clean_text))


class KeywordOptimizationService:
    """Service for keyword discovery, metric retrieval, and content weaving."""

    def __init__(self):
        pass

    def _get_llm_client(self):
        """Create LLM client instance using configured provider and key."""
        try:
            from supabase_client import get_default_llm_provider
            from llm_client import create_llm_client
            provider, model, api_key = get_default_llm_provider()
            if provider and model and api_key:
                return create_llm_client(provider=provider, model=model, api_key=api_key)
        except Exception as err:
            logger.warning(f"[KeywordOptimizationService] LLM provider init failed: {err}")

        try:
            from llm_client import create_llm_client
            return create_llm_client()
        except Exception as err:
            logger.warning(f"[KeywordOptimizationService] LLM direct fallback failed: {err}")
            return None

    def _extract_heuristic_seeds(self, title: str, content: str = "", tags: Optional[List[str]] = None) -> List[str]:
        """Extract realistic 2-3 word search seeds from title/content without LLM."""
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could",
            "should", "would", "will", "just", "died", "your", "my", "our", "their", "this",
            "that", "these", "those", "how", "what", "why", "when", "where", "which", "who",
            "more", "most", "some", "such", "no", "not", "only", "own", "same", "so", "than",
            "too", "very", "s", "t", "swapping", "swap", "now", "vs", "versus", "article",
        }

        seeds: List[str] = []

        # 1. Tags if present
        if tags:
            for t in tags:
                clean_t = re.sub(r"[^a-zA-Z0-9\s-]", "", str(t)).strip().lower()
                if clean_t and 1 < len(clean_t.split()) <= 4:
                    seeds.append(clean_t)

        # 2. Extract key phrases from title
        clean_title = re.sub(r"[^a-zA-Z0-9\s]", " ", title or "").lower()
        words = [w for w in clean_title.split() if w]
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]

        if len(meaningful_words) >= 2:
            for i in range(min(4, len(meaningful_words) - 1)):
                seeds.append(f"{meaningful_words[i]} {meaningful_words[i+1]}")
        if len(meaningful_words) >= 3:
            for i in range(min(3, len(meaningful_words) - 2)):
                seeds.append(f"{meaningful_words[i]} {meaningful_words[i+1]} {meaningful_words[i+2]}")

        # Domain-specific smart anchors
        if "heat" in words and "pump" in words:
            seeds.append("heat pump rebate")
            seeds.append("heat pump tax credit")
            seeds.append("heat pump incentives")
        if "furnace" in words:
            seeds.append("gas furnace rebate")
            seeds.append("furnace replacement rebate")

        # Deduplicate
        deduped = []
        seen = set()
        for s in seeds:
            s_clean = s.strip().lower()
            if s_clean and s_clean not in seen and len(s_clean.split()) <= 4:
                seen.add(s_clean)
                deduped.append(s_clean)

        return deduped[:5] if deduped else ["heat pump rebate", "furnace rebate"]

    async def extract_seeds_from_content(self, title: str, content: str, tags: Optional[List[str]] = None) -> List[str]:
        """Extract 3-5 high-intent, short (2-4 words) search query seeds from title and content."""
        clean_content = re.sub(r"<[^>]+>", " ", content or "")[:1500].strip()
        tags_str = ", ".join(tags) if tags else ""

        llm = self._get_llm_client()
        if llm:
            prompt = f"""
You are an expert SEO researcher. Given the article title and content excerpt below, extract 3 to 5 realistic, high-intent Google search query phrases that real users type into Google search.

CRITICAL REQUIREMENT:
- Each query phrase MUST be short and concise (2 to 4 words max), e.g. "heat pump rebate", "gas furnace replacement", "heat pump tax credit".
- NEVER output full sentences or headlines.

Title: {title}
Tags: {tags_str}
Excerpt: {clean_content}

Output ONLY a JSON array of 3 to 5 lowercase strings. No explanations.
"""
            try:
                messages = [{"role": "user", "content": prompt}]
                res = llm.generate(messages=messages) if hasattr(llm, "generate") else llm.generate(prompt)
                raw_text = res.content if hasattr(res, "content") else str(res)
                # Parse JSON
                match = re.search(r"\[[\s\S]*\]", raw_text)
                if match:
                    seeds = json.loads(match.group(0))
                    if isinstance(seeds, list):
                        valid_seeds = [
                            str(s).strip().lower()
                            for s in seeds
                            if str(s).strip() and len(str(s).strip().split()) <= 4
                        ]
                        if valid_seeds:
                            return valid_seeds[:5]
            except Exception as err:
                logger.warning(f"[KeywordOptimizationService] LLM seed extraction failed: {err}")

        # Fallback to heuristic
        return self._extract_heuristic_seeds(title, content, tags)

    async def discover_keywords_for_article(
        self,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
        custom_seed: Optional[str] = None,
        location_code: int = 2840,
        language_code: str = "en",
    ) -> Dict[str, Any]:
        """
        Discover related keywords and retrieve DataForSEO metrics.
        Returns a dict with 'keywords' (sorted candidate list) and 'seeds' (the search seeds used).
        """
        seeds: List[str] = []
        if custom_seed and custom_seed.strip():
            raw_seed = custom_seed.strip().lower()
            # If custom_seed is long (e.g. user pasted headline), distill it
            if len(raw_seed.split()) > 4 or len(raw_seed) > 35:
                seeds = await self.extract_seeds_from_content(raw_seed, "", tags)
            else:
                seeds = [raw_seed]
                # Add 1-2 close heuristic seeds
                heuristics = self._extract_heuristic_seeds(raw_seed, "", tags)
                for h in heuristics:
                    if h not in seeds and len(seeds) < 3:
                        seeds.append(h)
        else:
            seeds = await self.extract_seeds_from_content(title, content, tags)

        if not seeds:
            seeds = self._extract_heuristic_seeds(title, content, tags)

        logger.info(f"[KeywordOptimizationService] Querying DataForSEO for seeds: {seeds}")

        # 1. Fetch related keywords from DataForSEO Labs live
        raw_items: List[Dict[str, Any]] = []
        try:
            related_items = await dataforseo_api.get_related_keywords_labs_live(
                seeds=seeds,
                location_code=location_code,
                limit_per_seed=25,
            )
            if isinstance(related_items, list):
                raw_items.extend(related_items)
        except Exception as err:
            logger.warning(f"[KeywordOptimizationService] DataForSEO Labs related keywords fetch failed: {err}")

        # 2. Also fetch keyword suggestions (without restrictive filters)
        try:
            suggestions = await dataforseo_api.get_keyword_suggestions_labs_live(
                seeds=seeds[:3],
                location_code=location_code,
                limit_per_seed=25,
                filters=[],  # No KD filters
            )
            if isinstance(suggestions, list):
                raw_items.extend(suggestions)
        except Exception as err:
            logger.warning(f"[KeywordOptimizationService] DataForSEO Labs suggestions fetch failed: {err}")

        # 3. Fetch metrics for seeds directly if not present
        seed_keywords_to_query = [s for s in seeds if not any(item.get("keyword") == s for item in raw_items)]
        if seed_keywords_to_query:
            try:
                seed_metrics = await dataforseo_api.get_bulk_metrics_standard(
                    keywords=seed_keywords_to_query,
                    location_code=location_code,
                    language_code=language_code,
                )
                if isinstance(seed_metrics, list):
                    raw_items.extend(seed_metrics)
            except Exception as err:
                logger.warning(f"[KeywordOptimizationService] Bulk metrics fetch failed: {err}")

        # 4. Format and deduplicate
        seen = set()
        formatted: List[Dict[str, Any]] = []

        full_text = f"{title}\n{content}"

        for item in raw_items:
            kw = str(item.get("keyword") or "").strip().lower()
            if not kw or kw in seen:
                continue
            seen.add(kw)

            vol = item.get("search_volume")
            kd = item.get("keyword_difficulty")
            cpc = item.get("cpc")
            intent = item.get("intent") or item.get("search_intent_info", {}).get("main_intent") or "informational"
            competition = item.get("competition") or "UNKNOWN"

            opp_score = calculate_opportunity_score(vol, kd)
            occurrences = count_keyword_occurrences(full_text, kw)

            formatted.append({
                "keyword": kw,
                "search_volume": vol if vol is not None else 0,
                "keyword_difficulty": round(kd, 1) if kd is not None else None,
                "cpc": round(cpc, 2) if cpc is not None else 0.0,
                "intent": intent,
                "competition": competition,
                "opportunity_score": opp_score,
                "in_text_count": occurrences,
                "is_seed": kw in seeds,
            })

        # Sort primarily by opportunity score descending, then search volume
        formatted.sort(key=lambda x: (x["opportunity_score"], x["search_volume"]), reverse=True)
        return {
            "keywords": formatted,
            "seeds": seeds,
        }

    async def search_single_keyword(
        self,
        keyword: str,
        location_code: int = 2840,
        language_code: str = "en",
    ) -> Dict[str, Any]:
        """Search DataForSEO for a specific keyword query and its immediate variants."""
        kw_clean = keyword.strip().lower()
        if not kw_clean:
            return {"keywords": [], "seeds": []}

        return await self.discover_keywords_for_article(
            title="",
            content="",
            custom_seed=kw_clean,
            location_code=location_code,
            language_code=language_code,
        )

    async def weave_keywords_into_content(
        self,
        html_content: str,
        primary_keyword: str,
        secondary_keywords: Optional[List[str]] = None,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Subtly modify the HTML article to weave in primary and secondary keywords naturally.
        Preserves all HTML structure, citations, links, tone, and facts.
        """
        secondaries = [s.strip() for s in (secondary_keywords or []) if s.strip()]
        primary = primary_keyword.strip()

        if not primary and not secondaries:
            return {
                "success": False,
                "error": "No primary or secondary keywords provided.",
                "html": html_content,
                "changes": [],
            }

        llm = self._get_llm_client()
        if not llm:
            return {
                "success": False,
                "error": "LLM service unavailable for keyword weaving.",
                "html": html_content,
                "changes": [],
            }

        prompt = f"""
You are an elite editorial SEO specialist. Your goal is to naturally weave target SEO keywords into an existing article HTML without disrupting its flow, voice, citations, or HTML structure.

Target Primary Keyword (Must appear naturally in H1/intro/H2):
- "{primary}"

Target Secondary Keywords (Integrate 1-2 times each in relevant subheadings or body text):
{chr(10).join(f'- "{sec}"' for sec in secondaries)}

STRICT RULES:
1. NON-DESTRUCTIVE: Do NOT remove or delete paragraphs. Only edit 2 to 5 sentences or 1-2 H2 headings to smoothly weave the keywords in.
2. CITATION PRESERVATION: Preserve EVERY citation marker (e.g. `[1]`, `[^1]`, `<a class="citation-link">...</a>`, `<section class="geo-key-takeaways">...`), blockquotes, and HTML tags exactly intact.
3. NO KEYWORD STUFFING: The text must read as if it were written by a top human journalist, completely natural and grammatically fluent.
4. Output MUST be a valid JSON object matching this schema:
{{
  "modified_html": "<complete modified HTML string>",
  "changes": [
    "Brief explanation of change 1 (e.g. Added primary keyword to Introduction)",
    "Brief explanation of change 2 (e.g. Adjusted second H2 heading to include secondary keyword)"
  ]
}}

Existing Article HTML:
```html
{html_content}
```
"""
        try:
            messages = [
                {"role": "system", "content": "You are an elite editorial SEO specialist that naturally weaves keywords into HTML articles preserving citations, facts, and structure. Output strictly valid JSON."},
                {"role": "user", "content": prompt}
            ]
            res = llm.generate(messages=messages)
            raw_text = res.content if hasattr(res, "content") else str(res)

            # Parse JSON
            match = re.search(r"\{[\s\S]*\}", raw_text)
            if match:
                parsed = json.loads(match.group(0))
                modified_html = parsed.get("modified_html") or html_content
                changes = parsed.get("changes") or []

                # Calculate placement summary
                placements = []
                if primary:
                    placements.append({
                        "keyword": primary,
                        "type": "primary",
                        "count": count_keyword_occurrences(modified_html, primary),
                    })
                for sec in secondaries:
                    placements.append({
                        "keyword": sec,
                        "type": "secondary",
                        "count": count_keyword_occurrences(modified_html, sec),
                    })

                return {
                    "success": True,
                    "html": modified_html,
                    "changes": changes,
                    "placements": placements,
                }
        except Exception as err:
            logger.error(f"[KeywordOptimizationService] Weaving failed: {err}", exc_info=True)
            return {
                "success": False,
                "error": str(err),
                "html": html_content,
                "changes": [],
            }

        return {
            "success": False,
            "error": "Failed to parse LLM weaving output.",
            "html": html_content,
            "changes": [],
        }


# Singleton export
keyword_optimization_service = KeywordOptimizationService()
