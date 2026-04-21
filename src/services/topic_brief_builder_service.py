"""
Topic brief builder service.
Builds a normalized brief that downstream subtopic + SEO services can use consistently.
"""

from typing import Any, Dict, Optional


class TopicBriefBuilderService:
    """Build normalized topic briefs from request context."""

    def build(
        self,
        topic: Dict[str, Any],
        project: Optional[Dict[str, Any]] = None,
        decomposition_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        project = project or {}
        decomposition_context = decomposition_context or {}

        category_path = decomposition_context.get("category_path")
        if not category_path:
            primary = decomposition_context.get("primary_category_name")
            secondary = decomposition_context.get("secondary_category_name")
            if primary and secondary:
                category_path = f"{primary} / {secondary}"
            elif primary:
                category_path = primary
            else:
                category_path = ""

        return {
            "topic_title": topic.get("title") or "",
            "topic_description": topic.get("description") or "",
            "project_name": project.get("domain") or project.get("app_name") or decomposition_context.get("project_name") or "",
            "project_description": (
                project.get("site_description")
                or project.get("websitedescription")
                or decomposition_context.get("project_description")
                or ""
            ),
            "category_path": category_path or "",
            "intent_bucket": topic.get("intent_bucket") or decomposition_context.get("intent_bucket") or "informational_decision",
            "decision_focus": topic.get("decision_focus") or decomposition_context.get("decision_focus") or "",
            "angle_question": topic.get("angle_question") or decomposition_context.get("angle_question") or "",
            "value_layer_tags": topic.get("value_layer_tags") or decomposition_context.get("value_layer_tags") or [],
            "target_audience": topic.get("target_audience") or decomposition_context.get("target_audience") or "",
            "evidence_sources": topic.get("evidence_sources") or decomposition_context.get("evidence_sources") or [],
            "signal_terms": decomposition_context.get("signal_terms") or [],
            "trend_titles": decomposition_context.get("trend_titles") or [],
            "autocomplete_suggestions": decomposition_context.get("autocomplete_suggestions") or [],
        }


topic_brief_builder_service = TopicBriefBuilderService()

