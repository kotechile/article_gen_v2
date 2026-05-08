"""
Compatibility bridge for the research rebuild.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


class ResearchCompatibilityAdapterService:
    """
    Bridge the new research model into legacy downstream objects.

    This adapter exists so the rebuild can preserve the current contracts around
    `research_topics`, `content_ideas`, `Titles`, Content Studio, and WordPress
    export while the new model becomes the source of truth.
    """

    async def outcome_to_content_idea_payload(
        self,
        *,
        candidate: Dict[str, Any],
        generated_outcome: Dict[str, Any],
        category_context: Dict[str, Any],
        keyword_pack: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Map a research rebuild outcome into a content_ideas insert payload."""
        outcome_metadata = generated_outcome.get("outcome_metadata") or {}
        candidate_metadata = candidate.get("candidate_metadata") or {}
        primary_keyword = (keyword_pack or {}).get("primary_keyword")
        secondary_keywords = (keyword_pack or {}).get("secondary_keywords_json") or []

        title = (
            outcome_metadata.get("title")
            or outcome_metadata.get("headline")
            or candidate.get("candidate_text")
            or "Untitled Research Outcome"
        )
        description = (
            outcome_metadata.get("description")
            or outcome_metadata.get("summary")
            or f"Generated from research opportunity: {candidate.get('candidate_text') or title}"
        )
        content_type = "software" if generated_outcome.get("outcome_type") == "software" else "blog"

        metadata = dict(outcome_metadata.get("idea_metadata") or {})
        metadata["category_context"] = category_context or {}
        metadata["research_rebuild"] = {
            "candidate_id": candidate.get("id"),
            "generated_outcome_id": generated_outcome.get("id"),
            "validation_run_id": generated_outcome.get("validation_run_id"),
            "routing_decision_id": generated_outcome.get("routing_decision_id"),
        }

        keywords = []
        if primary_keyword:
            keywords.append(str(primary_keyword).strip())
        for value in secondary_keywords:
            cleaned = str(value or "").strip()
            if cleaned and cleaned not in keywords:
                keywords.append(cleaned)

        return {
            "title": title,
            "description": description,
            "content_type": content_type,
            "category": outcome_metadata.get("category")
            or candidate_metadata.get("category")
            or ("software_tool" if content_type == "software" else "seo_optimized"),
            "subtopic": outcome_metadata.get("subtopic")
            or candidate_metadata.get("subtopic")
            or candidate.get("candidate_text")
            or title,
            "topic_id": outcome_metadata.get("topic_id")
            or candidate_metadata.get("topic_id"),
            "keywords": keywords,
            "primary_keywords": [keywords[0]] if keywords else [],
            "secondary_keywords": secondary_keywords,
            "search_phrase": primary_keyword or candidate.get("candidate_text"),
            "target_intent": outcome_metadata.get("target_intent")
            or candidate_metadata.get("target_intent"),
            "product_type": outcome_metadata.get("product_type")
            or candidate_metadata.get("product_type"),
            "user_job_to_be_done": outcome_metadata.get("user_job_to_be_done")
            or candidate_metadata.get("user_job_to_be_done")
            or candidate_metadata.get("job_text"),
            "build_complexity": outcome_metadata.get("build_complexity")
            or candidate_metadata.get("build_complexity"),
            "distribution_angle": outcome_metadata.get("distribution_angle")
            or candidate_metadata.get("distribution_angle"),
            "keyword_metrics": (keyword_pack or {}).get("keyword_metrics_json") or {},
            "idea_metadata": metadata,
            "status": "draft",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

    async def outcome_to_titles_payload(
        self,
        *,
        content_idea: Dict[str, Any],
        category_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Map a published content idea into the existing Titles contract."""
        idea_metadata = dict(content_idea.get("idea_metadata") or {})
        idea_metadata["category_context"] = category_context or {}
        return {
            "source_idea_id": content_idea.get("id"),
            "topic_id": content_idea.get("topic_id"),
            "title": content_idea.get("title") or "Untitled Article",
            "description": content_idea.get("description") or "",
            "keywords": content_idea.get("primary_keywords") or content_idea.get("keywords") or [],
            "primary_keywords": content_idea.get("primary_keywords") or [],
            "secondary_keywords": content_idea.get("secondary_keywords") or [],
            "search_phrase": content_idea.get("search_phrase"),
            "wordpress_category_id": content_idea.get("wordpress_category_id"),
            "idea_metadata": idea_metadata,
        }

    async def outcome_to_released_software_payload(
        self,
        *,
        generated_outcome: Dict[str, Any],
        content_idea: Dict[str, Any] | None,
        user_id: str,
        released_at: str,
    ) -> Dict[str, Any]:
        """Map a generated outcome or software-flavored content idea into released_software_ideas."""
        source = content_idea or {}
        outcome_metadata = generated_outcome.get("outcome_metadata") or {}
        idea_metadata = dict(source.get("idea_metadata") or {})
        return {
            "user_id": user_id,
            "source_idea_id": source.get("id"),
            "topic_id": source.get("topic_id"),
            "title": source.get("title") or outcome_metadata.get("title") or "Untitled Software Idea",
            "description": source.get("description") or outcome_metadata.get("description") or "",
            "status": "saved",
            "released_at": released_at,
            "published": True,
            "content_type": "software",
            "subtopic": source.get("subtopic") or outcome_metadata.get("subtopic"),
            "category": source.get("category") or outcome_metadata.get("category") or "software_tool",
            "domain": source.get("domain"),
            "keywords": source.get("primary_keywords") or source.get("keywords") or [],
            "primary_keywords": source.get("primary_keywords") or source.get("keywords") or [],
            "secondary_keywords": source.get("secondary_keywords") or [],
            "search_phrase": source.get("search_phrase"),
            "total_search_volume": source.get("total_search_volume"),
            "average_difficulty": source.get("average_difficulty"),
            "average_cpc": source.get("average_cpc"),
            "affiliate_offer_count": source.get("affiliate_offer_count"),
            "topic_rating": source.get("topic_rating") or 0,
            "viability_score": source.get("viability_score"),
            "trend_score": source.get("trend_score"),
            "monetization_score": source.get("monetization_score"),
            "seo_ease_score": source.get("seo_ease_score"),
            "opportunity_score": source.get("opportunity_score"),
            "product_type": source.get("product_type") or outcome_metadata.get("product_type"),
            "user_job_to_be_done": source.get("user_job_to_be_done") or outcome_metadata.get("user_job_to_be_done"),
            "key_inputs": source.get("key_inputs") or idea_metadata.get("key_inputs") or [],
            "output_result": source.get("output_result") or outcome_metadata.get("output_result"),
            "build_complexity": source.get("build_complexity") or outcome_metadata.get("build_complexity"),
            "distribution_angle": source.get("distribution_angle") or outcome_metadata.get("distribution_angle"),
            "target_intent": source.get("target_intent") or outcome_metadata.get("target_intent"),
            "content_outline": source.get("content_outline") or [],
            "ranking_breakdown": source.get("ranking_breakdown") or {},
            "keyword_metrics": source.get("keyword_metrics") or {},
            "idea_metadata": idea_metadata,
            "raw_dataforseo_output": source.get("raw_dataforseo_output"),
            "raw_supabase_output": source.get("raw_supabase_output"),
            "updated_at": released_at,
        }
