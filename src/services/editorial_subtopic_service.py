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
            parsed = self._parse(response.content or "")
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
