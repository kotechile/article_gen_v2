"""
Editorial subtopic generation service.
Generates subtopics from topic brief first, independent from DataForSEO metrics.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List

from .llm.llm_service import llm_service

logger = logging.getLogger(__name__)


class EditorialSubtopicService:
    """Generate structured editorial subtopics from a topic brief."""

    _TOKEN_SYNONYMS = {
        "affordable": "price",
        "budget": "price",
        "cost": "price",
        "costs": "price",
        "lowcost": "price",
        "overpaying": "price",
        "pricing": "price",
        "priced": "price",
        "provider": "platform",
        "providers": "platform",
        "select": "choose",
        "selected": "choose",
        "selecting": "choose",
        "selection": "choose",
        "software": "platform",
        "tool": "platform",
        "tools": "platform",
        "vendor": "platform",
        "vendors": "platform",
    }

    _GENERIC_CONCEPT_TOKENS = {
        "about",
        "against",
        "analysis",
        "approach",
        "audit",
        "best",
        "better",
        "checklist",
        "comparison",
        "comparisons",
        "complete",
        "decision",
        "decisions",
        "for",
        "framework",
        "frameworks",
        "from",
        "guide",
        "guides",
        "how",
        "into",
        "more",
        "overview",
        "playbook",
        "problem",
        "problems",
        "scenario",
        "scenarios",
        "strategic",
        "strategy",
        "strategies",
        "than",
        "that",
        "the",
        "their",
        "them",
        "these",
        "those",
        "using",
        "what",
        "when",
        "which",
        "with",
        "your",
    }

    def _build_prompt(self, brief: Dict[str, Any], max_subtopics: int) -> str:
        return f"""
You are a senior editorial strategist for SEO and GEO content planning.

TOPIC BRIEF
- Topic: {brief.get("topic_title")}
- Description: {brief.get("topic_description")}
- Project: {brief.get("project_name")}
- Project Description: {brief.get("project_description")}
- Category Lens: {brief.get("category_path")}
- Primary Category Description: {brief.get("primary_category_description")}
- Sub-Category Description: {brief.get("secondary_category_description")}
- Category Strategy Hint: {brief.get("category_strategy_hint")}
- Intent Bucket: {brief.get("intent_bucket")}
- Decision Focus: {brief.get("decision_focus")}
- Angle Question: {brief.get("angle_question")}
- Value Tags: {", ".join(brief.get("value_layer_tags") or [])}
- Audience: {brief.get("target_audience")}
- Signals: {", ".join((brief.get("signal_terms") or [])[:12])}

TASK
Generate exactly {max_subtopics} editorial subtopics. These are decision/problem frameworks, not keyword strings.
Use concrete types: comparison, framework, checklist, audit, calculator, scenario, decision, or problem.
Keep every idea tightly aligned with the category lens and sub-category strategy.
SEED_PHRASES must be short search-style phrases (2-5 words), plain language, without symbols or meta-text.

DIVERSITY RULES
- Every subtopic must represent a meaningfully different concept, not a paraphrase of another one.
- Do not produce multiple subtopics that cover the same decision through different wording, such as cost vs pricing, vendor selection vs choosing a provider, or setup checklist vs implementation checklist.
- Spread the list across different decision spaces when possible, such as comparison, budgeting, implementation, mistakes, measurement, use-case fit, risk, or migration.
- If two candidates would lead to mostly the same article outline, keep only the stronger one.
- Distinguish subtopics by the core question being answered, not by surface wording.
- Before finalizing, remove any near-duplicate or synonym-based variation.

OUTPUT FORMAT
Return only repeated blocks in this format:

[SUBTOPIC]
TITLE: <clear human-readable subtopic title>
SUMMARY: <one sentence summary>
DECISION_TYPE: <comparison|framework|checklist|audit|calculator|scenario|decision|problem>
USER_PROBLEM: <what user is trying to solve>
TARGET_AUDIENCE: <specific audience>
SEED_PHRASES: <phrase 1>, <phrase 2>, <phrase 3>, <phrase 4>
GEO_ENTITY_HINTS: <entity 1>, <entity 2>, <entity 3>
COMMERCIAL_PATHS: <path 1>, <path 2>
[END]
"""

    def _normalize_token(self, token: str) -> str:
        token = re.sub(r"[^a-z0-9]", "", token.lower())
        if len(token) <= 2:
            return ""
        if token.endswith("ies") and len(token) > 4:
            token = f"{token[:-3]}y"
        elif token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("es") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        token = self._TOKEN_SYNONYMS.get(token, token)
        return token

    def _concept_tokens(self, subtopic: Dict[str, Any]) -> List[str]:
        text = " ".join(
            [
                subtopic.get("title") or "",
                subtopic.get("summary") or "",
                subtopic.get("user_problem") or "",
            ]
        )
        tokens: List[str] = []
        seen = set()
        for raw in re.findall(r"[a-zA-Z0-9]+", text.lower()):
            token = self._normalize_token(raw)
            if not token or token in self._GENERIC_CONCEPT_TOKENS:
                continue
            if token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        return tokens

    def _is_near_duplicate(self, candidate: Dict[str, Any], existing: Dict[str, Any]) -> bool:
        candidate_title = re.sub(r"\s+", " ", (candidate.get("title") or "").strip().lower())
        existing_title = re.sub(r"\s+", " ", (existing.get("title") or "").strip().lower())
        if candidate_title and candidate_title == existing_title:
            return True

        candidate_tokens = set(self._concept_tokens(candidate))
        existing_tokens = set(self._concept_tokens(existing))
        if not candidate_tokens or not existing_tokens:
            return False

        overlap = len(candidate_tokens & existing_tokens)
        coverage = overlap / max(1, min(len(candidate_tokens), len(existing_tokens)))
        jaccard = overlap / max(1, len(candidate_tokens | existing_tokens))
        same_decision_type = (
            (candidate.get("decision_type") or "").lower()
            == (existing.get("decision_type") or "").lower()
        )

        return (
            coverage >= 0.8
            or (same_decision_type and jaccard >= 0.6)
            or (same_decision_type and overlap >= 3 and coverage >= 0.5)
        )

    def _dedupe_distinct_subtopics(
        self,
        subtopics: List[Dict[str, Any]],
        max_subtopics: int,
    ) -> List[Dict[str, Any]]:
        distinct: List[Dict[str, Any]] = []
        for subtopic in subtopics:
            if any(self._is_near_duplicate(subtopic, existing) for existing in distinct):
                logger.info("Dropping near-duplicate editorial subtopic title=%r", subtopic.get("title"))
                continue
            distinct.append(subtopic)
            if len(distinct) >= max_subtopics:
                break
        return distinct

    def _parse(self, text: str) -> List[Dict[str, Any]]:
        blocks = re.findall(r"\[SUBTOPIC\](.*?)\[END\]", text, flags=re.DOTALL | re.IGNORECASE)
        parsed: List[Dict[str, Any]] = []
        for block in blocks:
            fields: Dict[str, str] = {}
            for raw_line in block.splitlines():
                line = raw_line.strip()
                if not line or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                fields[key.strip().upper()] = val.strip()

            title = fields.get("TITLE", "").strip()
            if not title:
                continue
            seed_phrases = [p.strip() for p in fields.get("SEED_PHRASES", "").split(",") if p.strip()]
            geo_hints = [p.strip() for p in fields.get("GEO_ENTITY_HINTS", "").split(",") if p.strip()]
            commercial_paths = [p.strip() for p in fields.get("COMMERCIAL_PATHS", "").split(",") if p.strip()]
            parsed.append(
                {
                    "title": title,
                    "summary": fields.get("SUMMARY", ""),
                    "decision_type": (fields.get("DECISION_TYPE") or "decision").lower(),
                    "user_problem": fields.get("USER_PROBLEM", ""),
                    "target_audience": fields.get("TARGET_AUDIENCE", ""),
                    "seed_phrases": seed_phrases[:8],
                    "geo_entity_hints": geo_hints[:8],
                    "commercial_paths": commercial_paths[:6],
                }
            )
        return parsed

    async def generate(self, brief: Dict[str, Any], max_subtopics: int = 8) -> List[Dict[str, Any]]:
        prompt = self._build_prompt(brief, max_subtopics=max_subtopics)
        try:
            response = await asyncio.wait_for(
                llm_service.generate_text(prompt=prompt, max_tokens=1800),
                timeout=35.0,
            )
            parsed = self._dedupe_distinct_subtopics(
                self._parse(response.content or ""),
                max_subtopics=max_subtopics,
            )
            if parsed:
                logger.info("Editorial subtopics generated count=%s", len(parsed))
                return parsed[:max_subtopics]
        except Exception as e:
            logger.warning("Editorial subtopic generation failed: %s", e)

        # Safe fallback: create a few deterministic editorial subtopics.
        topic_title = (brief.get("topic_title") or "").strip()
        if not topic_title:
            return []
        fallbacks = [
            {
                "title": f"{topic_title} Comparison Framework",
                "summary": "Compare options with transparent tradeoffs and measurable outcomes.",
                "decision_type": "comparison",
                "user_problem": "Need to choose between multiple options confidently.",
                "target_audience": brief.get("target_audience") or "General Audience",
                "seed_phrases": [topic_title, f"{topic_title} comparison", f"{topic_title} checklist"],
                "geo_entity_hints": [],
                "commercial_paths": ["software", "services"],
            },
            {
                "title": f"{topic_title} ROI Audit",
                "summary": "Quantify upside, downside, and hidden costs before execution.",
                "decision_type": "audit",
                "user_problem": "Need numbers and risk visibility before taking action.",
                "target_audience": brief.get("target_audience") or "General Audience",
                "seed_phrases": [f"{topic_title} ROI", f"{topic_title} cost analysis", f"{topic_title} decision tool"],
                "geo_entity_hints": [],
                "commercial_paths": ["affiliate products", "consulting"],
            },
            {
                "title": f"{topic_title} Scenario Playbook",
                "summary": "Use scenario-based planning to choose next best actions.",
                "decision_type": "scenario",
                "user_problem": "Need actionable options for different market or personal scenarios.",
                "target_audience": brief.get("target_audience") or "General Audience",
                "seed_phrases": [f"{topic_title} strategy", f"{topic_title} scenarios", f"{topic_title} guide"],
                "geo_entity_hints": [],
                "commercial_paths": ["courses", "software"],
            },
        ]
        return fallbacks[:max_subtopics]


editorial_subtopic_service = EditorialSubtopicService()
