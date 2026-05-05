"""
Topic generation brief builder.
Builds a normalized editorial brief for research topic generation.
"""

from typing import Any, Dict, List, Optional


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


class TopicGenerationBriefService:
    """Build a stable brief for editorial topic generation."""

    def build(
        self,
        project: Optional[Dict[str, Any]] = None,
        primary_category: Optional[Dict[str, Any]] = None,
        secondary_category: Optional[Dict[str, Any]] = None,
        trend_titles: Optional[List[str]] = None,
        fallback_niche_description: Optional[str] = None,
        count: int = 10,
    ) -> Dict[str, Any]:
        project = project or {}
        primary_category = primary_category or {}
        secondary_category = secondary_category or {}
        trend_titles = trend_titles or []

        project_name = (
            _safe_text(project.get("project_name"))
            or _safe_text(project.get("domain"))
            or _safe_text(project.get("app_name"))
            or "selected website"
        )
        project_description = (
            _safe_text(project.get("project_description"))
            or _safe_text(project.get("site_description"))
            or _safe_text(project.get("websiteDescription"))
            or _safe_text(project.get("websitedescription"))
            or _safe_text(project.get("targetAudienceDescription"))
            or _safe_text(project.get("targetaudiencedescription"))
            or _safe_text(fallback_niche_description)
        )
        target_audience = (
            _safe_text(project.get("targetAudienceDescription"))
            or _safe_text(project.get("targetaudiencedescription"))
        )

        primary_name = _safe_text(primary_category.get("name"))
        secondary_name = _safe_text(secondary_category.get("name"))
        primary_description = _safe_text(primary_category.get("description"))
        secondary_description = _safe_text(secondary_category.get("description"))
        category_path = " / ".join([part for part in [primary_name, secondary_name] if part])

        parts = []
        if primary_name:
            parts.append(f"Primary category: {primary_name}.")
        if primary_description:
            parts.append(f"Primary category context: {primary_description}")
        if secondary_name:
            parts.append(f"Sub-category: {secondary_name}.")
        if secondary_description:
            parts.append(f"Sub-category context: {secondary_description}")

        return {
            "project_name": project_name,
            "project_description": project_description,
            "target_audience": target_audience,
            "primary_category_name": primary_name,
            "primary_category_description": primary_description,
            "secondary_category_name": secondary_name,
            "secondary_category_description": secondary_description,
            "category_path": category_path,
            "category_strategy_hint": " ".join(parts).strip(),
            "trend_titles": [item.strip() for item in trend_titles if str(item).strip()][:8],
            "count": max(1, min(int(count or 10), 20)),
        }


topic_generation_brief_service = TopicGenerationBriefService()
