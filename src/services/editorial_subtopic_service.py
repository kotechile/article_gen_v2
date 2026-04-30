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

    _RAW_RESPONSE_LOG_LIMIT = 4000

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
Return only repeated blocks in this exact tagged format:

<<SUBTOPIC>>
<<TITLE>>
<clear human-readable subtopic title>
<</TITLE>>
<<SUMMARY>>
<one sentence summary>
<</SUMMARY>>
<<DECISION_TYPE>>
<comparison|framework|checklist|audit|calculator|scenario|decision|problem>
<</DECISION_TYPE>>
<<USER_PROBLEM>>
<what user is trying to solve>
<</USER_PROBLEM>>
<<TARGET_AUDIENCE>>
<specific audience>
<</TARGET_AUDIENCE>>
<<SEED_PHRASES>>
<phrase 1> | <phrase 2> | <phrase 3> | <phrase 4>
<</SEED_PHRASES>>
<<GEO_ENTITY_HINTS>>
<entity 1> | <entity 2> | <entity 3>
<</GEO_ENTITY_HINTS>>
<<COMMERCIAL_PATHS>>
<path 1> | <path 2>
<</COMMERCIAL_PATHS>>
<</SUBTOPIC>>

FORMAT RULES
- Use the tags exactly as written.
- Do not use JSON.
- Do not add numbering, commentary, markdown, or prose outside the tagged blocks.
"""

    def _split_multi_value_field(self, raw_value: str) -> List[str]:
        return [
            p.strip()
            for p in re.split(r"\s*\|\s*|\s*,\s*", raw_value or "")
            if p.strip()
        ]

    def _parse_tagged_block(self, block: str) -> Dict[str, Any]:
        def extract(tag: str) -> str:
            pattern = rf"<<{tag}>>\s*(.*?)\s*<</{tag}>>"
            match = re.search(pattern, block, flags=re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""

        title = extract("TITLE")
        if not title:
            return {}

        return {
            "title": title,
            "summary": extract("SUMMARY"),
            "decision_type": (extract("DECISION_TYPE") or "decision").lower(),
            "user_problem": extract("USER_PROBLEM"),
            "target_audience": extract("TARGET_AUDIENCE"),
            "seed_phrases": self._split_multi_value_field(extract("SEED_PHRASES"))[:8],
            "geo_entity_hints": self._split_multi_value_field(extract("GEO_ENTITY_HINTS"))[:8],
            "commercial_paths": self._split_multi_value_field(extract("COMMERCIAL_PATHS"))[:6],
        }

    def _parse_structured_block(self, block: str) -> Dict[str, Any]:
        fields: Dict[str, str] = {}
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            normalized_key = re.sub(r"[^A-Z_ ]", "", key.strip().upper()).replace(" ", "_")
            fields[normalized_key] = val.strip()

        title = fields.get("TITLE", "").strip()
        if not title:
            return {}

        seed_phrases = self._split_multi_value_field(fields.get("SEED_PHRASES", ""))
        geo_hints = self._split_multi_value_field(fields.get("GEO_ENTITY_HINTS", ""))
        commercial_paths = self._split_multi_value_field(fields.get("COMMERCIAL_PATHS", ""))
        return {
            "title": title,
            "summary": fields.get("SUMMARY", ""),
            "decision_type": (fields.get("DECISION_TYPE") or "decision").lower(),
            "user_problem": fields.get("USER_PROBLEM", ""),
            "target_audience": fields.get("TARGET_AUDIENCE", ""),
            "seed_phrases": seed_phrases[:8],
            "geo_entity_hints": geo_hints[:8],
            "commercial_paths": commercial_paths[:6],
        }

    def _parse(self, text: str) -> List[Dict[str, Any]]:
        tagged_blocks = re.findall(r"<<SUBTOPIC>>\s*(.*?)\s*<</SUBTOPIC>>", text, flags=re.DOTALL | re.IGNORECASE)
        parsed: List[Dict[str, Any]] = []
        for block in tagged_blocks:
            parsed_block = self._parse_tagged_block(block)
            if parsed_block:
                parsed.append(parsed_block)

        if parsed:
            return parsed

        blocks = re.findall(r"\[SUBTOPIC\](.*?)\[END\]", text, flags=re.DOTALL | re.IGNORECASE)
        for block in blocks:
            parsed_block = self._parse_structured_block(block)
            if parsed_block:
                parsed.append(parsed_block)

        if parsed:
            return parsed

        numbered_blocks = re.split(r"\n\s*(?=\d+\.\s+)", text.strip())
        for block in numbered_blocks:
            if "TITLE:" not in block.upper():
                continue
            parsed_block = self._parse_structured_block(block)
            if parsed_block:
                parsed.append(parsed_block)
        return parsed

    async def generate_with_debug(
        self,
        brief: Dict[str, Any],
        max_subtopics: int = 8,
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(brief, max_subtopics=max_subtopics)
        debug_info: Dict[str, Any] = {
            "topic_title": brief.get("topic_title"),
            "format_requested": "tagged_subtopic_blocks",
            "provider": None,
            "model": None,
            "raw_output_chars": 0,
            "raw_output_preview": "",
            "raw_output_truncated": False,
            "parser_match_mode": None,
        }
        try:
            response = await asyncio.wait_for(
                llm_service.generate_text(prompt=prompt, max_tokens=1800),
                timeout=35.0,
            )
            raw_content = response.content or ""
            debug_info.update(
                {
                    "provider": response.provider,
                    "model": response.model_name,
                    "raw_output_chars": len(raw_content),
                    "raw_output_preview": raw_content[:self._RAW_RESPONSE_LOG_LIMIT],
                    "raw_output_truncated": len(raw_content) > self._RAW_RESPONSE_LOG_LIMIT,
                }
            )
            logger.info(
                "Editorial subtopic LLM response provider=%s model=%s chars=%s topic=%r",
                response.provider,
                response.model_name,
                len(raw_content),
                brief.get("topic_title"),
            )
            logger.info(
                "Editorial subtopic raw LLM output topic=%r preview=%r truncated=%s",
                brief.get("topic_title"),
                raw_content[:self._RAW_RESPONSE_LOG_LIMIT],
                len(raw_content) > self._RAW_RESPONSE_LOG_LIMIT,
            )

            if re.search(r"<<SUBTOPIC>>", raw_content, flags=re.IGNORECASE):
                debug_info["parser_match_mode"] = "tagged"
            elif re.search(r"\[SUBTOPIC\]", raw_content, flags=re.IGNORECASE):
                debug_info["parser_match_mode"] = "legacy_block"
            elif "TITLE:" in raw_content.upper():
                debug_info["parser_match_mode"] = "numbered_structured"
            else:
                debug_info["parser_match_mode"] = "no_known_format_marker"

            parsed = self._parse(raw_content)
            if parsed:
                logger.info("Editorial subtopics generated count=%s", len(parsed))
                return {
                    "subtopics": parsed[:max_subtopics],
                    "debug": debug_info,
                }
        except Exception as e:
            debug_info["error"] = str(e)
            logger.warning("Editorial subtopic generation failed: %s", e)

        logger.warning(
            "Editorial subtopic generation produced no usable results topic=%r max_subtopics=%s parser_match_mode=%s",
            brief.get("topic_title"),
            max_subtopics,
            debug_info.get("parser_match_mode"),
        )
        return {
            "subtopics": [],
            "debug": debug_info,
        }

    async def generate(self, brief: Dict[str, Any], max_subtopics: int = 8) -> List[Dict[str, Any]]:
        result = await self.generate_with_debug(brief=brief, max_subtopics=max_subtopics)
        return result.get("subtopics") or []


editorial_subtopic_service = EditorialSubtopicService()
