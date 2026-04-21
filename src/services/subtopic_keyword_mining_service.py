"""
Subtopic keyword mining service.
Mines DataForSEO evidence per editorial subtopic and selects best supporting keywords.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List

from ..integrations.dataforseo import dataforseo_api
from .llm.llm_service import llm_service

logger = logging.getLogger(__name__)


class SubtopicKeywordMiningService:
    """Mine keyword evidence per subtopic."""

    _NOISE_TOKENS = {
        "words", "word", "vs", "and", "the", "for", "with", "without", "using",
        "guide", "framework", "checklist", "audit", "calculator", "scenario",
        "comparison", "decision", "problem",
    }

    def _clean(self, text: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9&'/%\-\s]", " ", (text or "").strip())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _normalize_short_seed(self, text: str) -> str:
        cleaned = self._clean(text)
        if not cleaned:
            return ""
        words = [w for w in cleaned.split() if w]
        if len(words) < 2:
            return ""
        if len(words) > 3:
            words = words[:3]
        normalized = " ".join(words)
        if re.search(r"(framework|playbook|methodology|optimization|solutioning|enablement)", normalized.lower()):
            return ""
        return normalized

    def _compact(self, text: str, n: int) -> str:
        cleaned = self._clean(text)
        words = [w for w in cleaned.split() if w]
        if len(words) <= n:
            return cleaned
        return " ".join(words[:n])

    def _tokenize(self, text: str) -> List[str]:
        cleaned = self._clean(text).lower()
        tokens = [t for t in cleaned.split() if t and len(t) > 1 and t not in self._NOISE_TOKENS]
        return tokens

    def _build_anchor_phrases(self, brief: Dict[str, Any]) -> List[str]:
        primary = brief.get("primary_category_name") or ""
        secondary = brief.get("secondary_category_name") or ""
        category_path = brief.get("category_path") or ""
        path_tokens = self._tokenize(category_path)

        anchors = []
        if primary:
            anchors.append(self._compact(primary, 3))
        if secondary:
            anchors.append(self._compact(secondary, 3))
        if len(path_tokens) >= 2:
            anchors.append(" ".join(path_tokens[:3]))

        return [self._clean(a) for a in anchors if self._clean(a)]

    def _compress_title_seed(self, title: str) -> List[str]:
        tokens = self._tokenize(title)
        if not tokens:
            return []

        out = []
        if len(tokens) >= 2:
            out.append(" ".join(tokens[:2]))
        if len(tokens) >= 3:
            out.append(" ".join(tokens[:3]))
        if len(tokens) >= 4:
            out.append(" ".join(tokens[:4]))
        return out

    async def _generate_llm_short_seeds(self, subtopic: Dict[str, Any], brief: Dict[str, Any]) -> List[str]:
        topic = brief.get("topic_title") or ""
        category_path = brief.get("category_path") or ""
        summary = subtopic.get("summary") or ""
        title = subtopic.get("title") or ""
        decision_type = subtopic.get("decision_type") or ""
        seed_phrases = ", ".join((subtopic.get("seed_phrases") or [])[:8])
        prompt = f"""
You are a veteran SEO Researcher and Search Intent Specialist.
Take a technical concept and reverse-engineer it into simple phrases real people type into Google.

Core Topic: {topic}
Subtopic Title: {title}
Subtopic Summary: {summary}
Decision Type: {decision_type}
Category Context: {category_path}
Existing Seed Hints: {seed_phrases}

Objective:
- Output only realistic search terms.
- Avoid consultant-speak or marketing fluff.
- Keep each keyword 1-3 words, plain language.
- Prefer terms with commercial or decision intent.

Required reasoning constraints:
1) Deconstruction Rule:
- Do NOT optimize for the full technical phrase.
- Break the topic into component concepts and generate keywords per component.
- Prefer component terms that people actually search for over precise technical jargon.

2) "2 AM Stress" Rule:
- Assume the user is stressed and searching for immediate clarity.
- Favor pain/outcome language such as cost, risk, failure, repair, inspection, tax, legal, ROI.
- If a phrase sounds like a consultant report title instead of a search query, discard it.

Task:
1) Problem Keywords: 10
2) Object Keywords: 10
3) Action Keywords: 8
4) Comparison Keywords: 5
5) Semantic Cluster Seeds: choose best 12 short terms across all groups

Output format:
- Return ONLY one keyword per line.
- No numbering, no bullets, no extra text.
- Every line must be 1-3 words.
"""
        try:
            response = await asyncio.wait_for(
                llm_service.generate_text(prompt=prompt, max_tokens=700, temperature=0.2),
                timeout=25.0,
            )
            lines = (response.content or "").splitlines()
            seeds: List[str] = []
            seen = set()
            for raw in lines:
                candidate = raw.strip().lstrip("-").strip()
                normalized = self._normalize_short_seed(candidate)
                if not normalized:
                    continue
                key = normalized.lower()
                if key in seen:
                    continue
                seen.add(key)
                seeds.append(normalized)
            logger.info(
                "LLM short seed generation subtopic=%r generated=%s sample=%s",
                title,
                len(seeds),
                seeds[:8],
            )
            return seeds[:20]
        except Exception as e:
            logger.warning("LLM short seed generation failed subtopic=%r err=%s", title, e)
            return []

    async def _build_variants(self, subtopic: Dict[str, Any], brief: Dict[str, Any]) -> List[str]:
        title = subtopic.get("title", "")
        summary = subtopic.get("summary", "")
        seeds = subtopic.get("seed_phrases") or []
        category = brief.get("category_path", "")
        anchors = self._build_anchor_phrases(brief)
        short_title_seeds = self._compress_title_seed(title)
        llm_short_seeds = await self._generate_llm_short_seeds(subtopic, brief)

        variants = [
            title,
            self._compact(title, 4),
            self._compact(title, 3),
            summary,
            self._compact(summary, 4),
            category,
            self._compact(f"{title} {category}", 4),
        ]
        variants.extend(short_title_seeds)
        variants.extend(anchors)
        variants.extend(llm_short_seeds)

        for seed in seeds:
            variants.append(seed)
            variants.append(self._compact(seed, 4))
            variants.append(self._compact(seed, 3))
            variants.append(self._normalize_short_seed(seed))

        # Add short category-anchored mixes to improve DataForSEO recall.
        for base in short_title_seeds[:2]:
            for anchor in anchors[:2]:
                variants.append(self._compact(f"{base} {anchor}", 5))
                variants.append(self._compact(f"{anchor} {base}", 5))

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
        return deduped[:18]

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
        variants = await self._build_variants(subtopic, brief)
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

        idea_keywords: List[Dict[str, Any]] = []
        for seed in variants[:2]:
            try:
                ideas = await dataforseo_api.get_keyword_ideas(seed, limit=30, filters=[])
                for idea in (ideas or []):
                    if idea.get("keyword"):
                        idea["source"] = "keyword_ideas_relaxed"
                        idea_keywords.append(idea)
            except Exception as e:
                logger.debug("Keyword ideas lookup failed seed=%r err=%s", seed, e)

        combined: List[Dict[str, Any]] = []
        for kw in (expanded_keywords or []):
            if kw.get("keyword"):
                kw["source"] = "related_keywords"
                combined.append(kw)

        for kw in idea_keywords:
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
                    "keyword_difficulty": metric.get("keyword_difficulty", 0),
                    "source": "direct_metrics",
                }
            )

        # Enrich keyword difficulty and missing competition/cpc for top candidates.
        kd_seeds = [item.get("keyword") for item in combined[:8] if item.get("keyword")]
        if kd_seeds:
            kd_rows = await dataforseo_api.get_keyword_difficulty(kd_seeds)
            kd_by_keyword = {
                (row.get("keyword") or "").lower(): row
                for row in (kd_rows or [])
                if row.get("keyword")
            }
            for item in combined:
                key = (item.get("keyword") or "").lower()
                kd_row = kd_by_keyword.get(key)
                if not kd_row:
                    continue
                item["keyword_difficulty"] = kd_row.get("keyword_difficulty", item.get("keyword_difficulty", 0))
                if (item.get("competition") in [None, "", "UNKNOWN"]) and kd_row.get("competition"):
                    item["competition"] = kd_row.get("competition")
                if not item.get("search_volume") and kd_row.get("search_volume"):
                    item["search_volume"] = kd_row.get("search_volume")
                if not item.get("cpc") and kd_row.get("cpc"):
                    item["cpc"] = kd_row.get("cpc")

        selected = self._select_best_keywords(combined)
        primary = selected[0]["keyword"] if selected else None
        non_zero_count = len([k for k in selected if (k.get("search_volume") or 0) > 0 or (k.get("cpc") or 0) > 0])

        logger.info(
            "Keyword mining subtopic=%r variants=%s selected=%s non_zero=%s primary=%r",
            subtopic.get("title"),
            len(variants),
            len(selected),
            non_zero_count,
            primary,
        )
        return {
            "variants_tried": variants,
            "keywords": selected,
            "primary_keyword": primary,
        }


subtopic_keyword_mining_service = SubtopicKeywordMiningService()
