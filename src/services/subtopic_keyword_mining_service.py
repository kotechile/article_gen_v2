"""
Subtopic keyword mining service.
Mines DataForSEO evidence per editorial subtopic and selects best supporting keywords.
"""

import logging
import re
from typing import Any, Dict, List

from ..integrations.dataforseo import dataforseo_api

logger = logging.getLogger(__name__)


class SubtopicKeywordMiningService:
    """Mine keyword evidence per subtopic."""

    def _clean(self, text: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9&'/%\-\s]", " ", (text or "").strip())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _compact(self, text: str, n: int) -> str:
        cleaned = self._clean(text)
        words = [w for w in cleaned.split() if w]
        if len(words) <= n:
            return cleaned
        return " ".join(words[:n])

    def _build_variants(self, subtopic: Dict[str, Any], brief: Dict[str, Any]) -> List[str]:
        title = subtopic.get("title", "")
        summary = subtopic.get("summary", "")
        seeds = subtopic.get("seed_phrases") or []
        category = brief.get("category_path", "")
        variants = [
            title,
            self._compact(title, 4),
            self._compact(title, 3),
            summary,
            self._compact(summary, 4),
            category,
            self._compact(f"{title} {category}", 4),
        ]
        variants.extend(seeds)

        deduped: List[str] = []
        seen = set()
        for v in variants:
            c = self._clean(v)
            if not c:
                continue
            if len(c.split()) < 2:
                continue
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)
        return deduped[:8]

    def _competition_rank(self, competition: str) -> int:
        comp = (competition or "").upper()
        if comp == "LOW":
            return 1
        if comp == "MEDIUM":
            return 2
        if comp == "HIGH":
            return 3
        return 4

    def _select_best_keywords(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not keywords:
            return []
        normalized = []
        for kw in keywords:
            normalized.append(
                {
                    "keyword": kw.get("keyword"),
                    "search_volume": int(kw.get("search_volume") or 0),
                    "cpc": float(kw.get("cpc") or 0.0),
                    "keyword_difficulty": int(kw.get("keyword_difficulty") or 0),
                    "competition": (kw.get("competition") or "UNKNOWN").upper(),
                    "source": kw.get("source", "dataforseo"),
                }
            )
        normalized.sort(
            key=lambda x: (
                -x["search_volume"],
                self._competition_rank(x["competition"]),
                x["keyword_difficulty"],
                -x["cpc"],
            )
        )
        selected = normalized[:12]
        if selected:
            selected[0]["is_selected_primary"] = True
            selected[0]["selection_reason"] = "Highest volume with best competition/KD tradeoff."
        return selected

    async def mine_for_subtopic(self, subtopic: Dict[str, Any], brief: Dict[str, Any]) -> Dict[str, Any]:
        variants = self._build_variants(subtopic, brief)
        if not variants:
            return {"variants_tried": [], "keywords": [], "primary_keyword": None}

        direct_metrics = await dataforseo_api.get_bulk_metrics_standard(variants)
        direct_by_keyword = {
            (item.get("keyword") or "").lower(): item
            for item in (direct_metrics or [])
            if item.get("keyword")
        }

        # Expand from top-performing variants only to control cost/latency.
        expansion_seeds = variants[:3]
        expanded_keywords = await dataforseo_api.get_related_keywords_standard(expansion_seeds, limit_per_seed=15)

        combined: List[Dict[str, Any]] = []
        for kw in (expanded_keywords or []):
            if kw.get("keyword"):
                kw["source"] = "related_keywords"
                combined.append(kw)

        for term in variants:
            metric = direct_by_keyword.get(term.lower())
            if not metric:
                continue
            combined.append(
                {
                    "keyword": term,
                    "search_volume": metric.get("search_volume", 0),
                    "cpc": metric.get("cpc", 0),
                    "competition": metric.get("competition", "UNKNOWN"),
                    "keyword_difficulty": 0,
                    "source": "direct_metrics",
                }
            )

        selected = self._select_best_keywords(combined)
        primary = selected[0]["keyword"] if selected else None

        logger.info(
            "Keyword mining subtopic=%r variants=%s selected=%s primary=%r",
            subtopic.get("title"),
            len(variants),
            len(selected),
            primary,
        )
        return {
            "variants_tried": variants,
            "keywords": selected,
            "primary_keyword": primary,
        }


subtopic_keyword_mining_service = SubtopicKeywordMiningService()

