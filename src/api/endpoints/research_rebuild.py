"""
Research rebuild API endpoints.

This blueprint provides the first usable API surface for the job-first,
validation-first research workflow while keeping the legacy research pipeline
intact.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
import os
import re
from uuid import UUID, uuid4

from flask import Blueprint, jsonify, request

from ...api.middleware.auth import require_api_key
from ...services.research_candidate_service import ResearchCandidateService
from ...services.research_compatibility_adapter_service import ResearchCompatibilityAdapterService
from ...services.research_generation_service import ResearchGenerationService
from ...services.research_internal_link_fit_service import ResearchInternalLinkFitService
from ...services.research_job_service import ResearchJobService
from ...services.research_keyword_pack_service import ResearchKeywordPackService
from ...services.research_routing_service import ResearchRoutingService
from ...services.research_validation_service import ResearchValidationService

logger = logging.getLogger(__name__)

research_rebuild_bp = Blueprint("research_rebuild", __name__, url_prefix="/api/research-rebuild")

job_service = ResearchJobService()
candidate_service = ResearchCandidateService()
validation_service = ResearchValidationService()
routing_service = ResearchRoutingService()
keyword_pack_service = ResearchKeywordPackService()
internal_link_fit_service = ResearchInternalLinkFitService()
generation_service = ResearchGenerationService()
compatibility_adapter_service = ResearchCompatibilityAdapterService()

ALLOWED_CANDIDATE_TYPES = {"seo_article", "software", "editorial"}
ALLOWED_FRESHNESS_STATES = {"fresh", "stale", "expired"}
ALLOWED_ROUTES = {
    "article_ready",
    "software_ready",
    "article_plus_software",
    "editorial_only",
    "software_backlog_low_feasibility",
    "needs_more_keyword_validation",
    "rejected_low_achievability",
}
ALLOWED_KEYWORD_PACK_STATUSES = {"draft", "ready", "cluster_too_thin", "needs_more_keyword_validation"}
ALLOWED_LINK_ROLES = {"parent_candidate", "child_candidate", "sibling_candidate", "hub_candidate"}
ALLOWED_OUTCOME_TYPES = {"article", "software", "editorial"}
ALLOWED_OUTCOME_STATUSES = {"draft", "generated", "persisted", "published", "archived"}
PROMOTABLE_ROUTES = {"article_ready", "software_ready", "article_plus_software", "editorial_only"}


def _normalize_review_title(value: str | None) -> str:
    """Normalize titles so rebuild/legacy duplicates can be compared safely."""
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _get_user_id_from_request():
    """Extract and validate user_id from Bearer token."""
    try:
        from supabase_client import get_supabase_client
    except ImportError:
        import sys

        sys.path.append(os.getcwd())
        from supabase_client import get_supabase_client

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None

    token = auth_header.split("Bearer ")[1]
    supabase = get_supabase_client()
    try:
        user_response = supabase.auth.get_user(token)
        if user_response and user_response.user:
            return user_response.user.id
    except Exception as exc:
        logger.warning("research-rebuild token validation failed: %s", exc)
    return None


def _parse_uuid(value: str, field_name: str) -> UUID:
    """Parse a UUID and raise a ValueError with a field-specific message."""
    try:
        return UUID(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a valid UUID") from exc


def _get_admin_supabase_client():
    """Return the backend Supabase client used by existing endpoint modules."""
    try:
        from src.core.supabase_singleton import get_supabase_client
    except ImportError:
        from ...core.supabase_singleton import get_supabase_client
    return get_supabase_client()


async def _build_persisted_workflow_snapshot(
    *,
    user_id: UUID,
    project_id: UUID,
    primary_category_id: UUID | None = None,
    secondary_category_id: UUID | None = None,
    job_status: str | None = None,
    workflow_run_id: str | None = None,
    route: str | None = None,
    candidate_type: str | None = None,
    outcome_type: str | None = None,
    search: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Assemble persisted rebuild artifacts into workflow-shaped job groups."""
    jobs = await job_service.list_jobs(
        user_id=user_id,
        project_id=project_id,
        primary_category_id=primary_category_id,
        secondary_category_id=secondary_category_id,
        status=job_status,
        include_archived=False,
        active_only=False,
    )
    allowed_job_ids = {str(job.get("id")) for job in jobs}

    candidates = await candidate_service.list_candidates(user_id=user_id, project_id=project_id)
    validation_runs = await validation_service.list_validation_runs(user_id=user_id, project_id=project_id)
    routing_decisions = await routing_service.list_routing_decisions(user_id=user_id, project_id=project_id)
    keyword_packs = await keyword_pack_service.list_keyword_packs(user_id=user_id, project_id=project_id)
    internal_links = await internal_link_fit_service.list_internal_link_candidates(user_id=user_id, project_id=project_id)
    generated_outcomes = await generation_service.list_generated_outcomes(user_id=user_id, project_id=project_id)

    jobs_by_id = {str(job.get("id")): job for job in jobs}
    normalized_search = (search or "").strip().lower()

    def _first_by_candidate(rows: list[dict]) -> dict[str, dict]:
        grouped: dict[str, dict] = {}
        for row in rows:
            candidate_id = str(row.get("candidate_id") or "")
            if candidate_id and candidate_id not in grouped:
                grouped[candidate_id] = row
        return grouped

    validation_by_candidate = _first_by_candidate(validation_runs)
    routing_by_candidate = _first_by_candidate(routing_decisions)
    keyword_pack_by_candidate = _first_by_candidate(keyword_packs)
    generated_outcome_by_candidate = _first_by_candidate(generated_outcomes)

    internal_links_by_candidate: dict[str, list[dict]] = {}
    for row in internal_links:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            continue
        internal_links_by_candidate.setdefault(candidate_id, []).append(row)

    grouped_results: dict[str, dict] = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "")
        user_job_id = str(candidate.get("user_job_id") or "")
        if not candidate_id or not user_job_id or user_job_id not in allowed_job_ids:
            continue
        if str(candidate.get("status") or "").strip().lower() == "rejected":
            continue
        candidate_metadata = candidate.get("candidate_metadata") or {}
        candidate_workflow_run_id = ""
        if isinstance(candidate_metadata, dict):
            candidate_workflow_run_id = str(candidate_metadata.get("workflow_run_id") or "")
        if workflow_run_id and candidate_workflow_run_id != workflow_run_id:
            continue

        validation_row = validation_by_candidate.get(candidate_id)
        routing_row = routing_by_candidate.get(candidate_id)
        keyword_pack_row = keyword_pack_by_candidate.get(candidate_id)
        outcome_row = generated_outcome_by_candidate.get(candidate_id)
        if not validation_row or not routing_row or not keyword_pack_row or not outcome_row:
            continue

        if candidate_type and str(candidate.get("candidate_type") or "") != candidate_type:
            continue
        if route and str(routing_row.get("route") or "") != route:
            continue
        if outcome_type and str(outcome_row.get("outcome_type") or "") != outcome_type:
            continue

        job_row = jobs_by_id.get(user_job_id)
        if normalized_search:
            haystack = " ".join(
                [
                    str(candidate.get("candidate_text") or ""),
                    str((keyword_pack_row or {}).get("primary_keyword") or ""),
                    str((routing_row or {}).get("route") or ""),
                    str((outcome_row or {}).get("outcome_type") or ""),
                    str((job_row or {}).get("job_text") or ""),
                ]
            ).lower()
            if normalized_search not in haystack:
                continue

        job_group = grouped_results.setdefault(
            user_job_id,
            {
                "job_id": user_job_id,
                "job": job_row,
                "candidates": [],
            },
        )
        job_group["candidates"].append(
            {
                "candidate": candidate,
                "validation_run": validation_row,
                "routing_decision": routing_row,
                "keyword_pack": keyword_pack_row,
                "internal_link_candidates": internal_links_by_candidate.get(candidate_id, []),
                "generated_outcome": outcome_row,
            }
        )

    sorted_results = sorted(
        grouped_results.values(),
        key=lambda row: str((row.get("job") or {}).get("created_at") or ""),
        reverse=True,
    )
    total_jobs = len(sorted_results)
    paged_results = sorted_results[offset: offset + limit] if limit is not None else sorted_results[offset:]
    return paged_results, total_jobs


async def _list_persisted_workflow_runs(
    *,
    user_id: UUID,
    project_id: UUID,
    primary_category_id: UUID | None = None,
    secondary_category_id: UUID | None = None,
    job_status: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Summarize recent workflow runs from persisted candidate metadata."""
    jobs = await job_service.list_jobs(
        user_id=user_id,
        project_id=project_id,
        primary_category_id=primary_category_id,
        secondary_category_id=secondary_category_id,
        status=job_status,
        include_archived=False,
        active_only=False,
    )
    allowed_job_ids = {str(job.get("id")) for job in jobs}
    jobs_by_id = {str(job.get("id")): job for job in jobs}
    candidates = await candidate_service.list_candidates(user_id=user_id, project_id=project_id)
    routing_decisions = await routing_service.list_routing_decisions(user_id=user_id, project_id=project_id)
    generated_outcomes = await generation_service.list_generated_outcomes(user_id=user_id, project_id=project_id)

    routing_by_candidate = {str(row.get("candidate_id") or ""): row for row in routing_decisions}
    outcome_by_candidate = {str(row.get("candidate_id") or ""): row for row in generated_outcomes}

    runs: dict[str, dict] = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "")
        user_job_id = str(candidate.get("user_job_id") or "")
        if not candidate_id or user_job_id not in allowed_job_ids:
            continue
        if str(candidate.get("status") or "").strip().lower() == "rejected":
            continue
        metadata = candidate.get("candidate_metadata") or {}
        if not isinstance(metadata, dict):
            continue
        run_id = str(metadata.get("workflow_run_id") or "")
        if not run_id:
            continue

        run_row = runs.setdefault(
            run_id,
            {
                "workflow_run_id": run_id,
                "started_at": metadata.get("workflow_run_started_at") or candidate.get("created_at"),
                "candidate_count": 0,
                "job_ids": set(),
                "primary_category_ids": set(),
                "secondary_category_ids": set(),
                "route_counts": {},
                "outcome_counts": {},
            },
        )
        run_row["candidate_count"] += 1
        run_row["job_ids"].add(user_job_id)
        job_row = jobs_by_id.get(user_job_id)
        primary_category_value = str((job_row or {}).get("primary_category_id") or "")
        secondary_category_value = str((job_row or {}).get("secondary_category_id") or "")
        if primary_category_value:
            run_row["primary_category_ids"].add(primary_category_value)
        if secondary_category_value:
            run_row["secondary_category_ids"].add(secondary_category_value)

        route = str((routing_by_candidate.get(candidate_id) or {}).get("route") or "")
        if route:
            route_counts = run_row["route_counts"]
            route_counts[route] = route_counts.get(route, 0) + 1

        outcome_type_value = str((outcome_by_candidate.get(candidate_id) or {}).get("outcome_type") or "")
        if outcome_type_value:
            outcome_counts = run_row["outcome_counts"]
            outcome_counts[outcome_type_value] = outcome_counts.get(outcome_type_value, 0) + 1

    sorted_runs = sorted(
        runs.values(),
        key=lambda row: str(row.get("started_at") or ""),
        reverse=True,
    )
    normalized = []
    for row in sorted_runs[: max(1, min(limit, 100))]:
        normalized.append(
            {
                "workflow_run_id": row["workflow_run_id"],
                "started_at": row["started_at"],
                "candidate_count": row["candidate_count"],
                "job_count": len(row["job_ids"]),
                "primary_category_ids": sorted(row["primary_category_ids"]),
                "secondary_category_ids": sorted(row["secondary_category_ids"]),
                "route_counts": row["route_counts"],
                "outcome_counts": row["outcome_counts"],
            }
        )
    return normalized


async def _build_topic_review_items(
    *,
    user_id: UUID,
    topic_id: UUID,
    project_id: UUID,
    primary_category_id: UUID | None = None,
    secondary_category_id: UUID | None = None,
    workflow_run_id: str | None = None,
    source: str = "all",
    include_suppressed_legacy: bool = False,
    limit: int = 12,
) -> tuple[list[dict], int]:
    """Build a combined rebuild + legacy review queue for a topic detail page."""
    snapshot_items, _ = await _build_persisted_workflow_snapshot(
        user_id=user_id,
        project_id=project_id,
        primary_category_id=primary_category_id,
        secondary_category_id=secondary_category_id,
        workflow_run_id=workflow_run_id,
        limit=24,
        offset=0,
    )

    rebuild_items: list[dict] = []
    persisted_rebuild_titles: set[str] = set()
    for job_result in snapshot_items:
        for candidate_result in job_result.get("candidates") or []:
            candidate = candidate_result.get("candidate") or {}
            keyword_pack = candidate_result.get("keyword_pack") or {}
            routing_decision = candidate_result.get("routing_decision") or {}
            generated_outcome = candidate_result.get("generated_outcome") or {}
            outcome_metadata = generated_outcome.get("outcome_metadata") or {}
            title = (
                outcome_metadata.get("title")
                or outcome_metadata.get("name")
                or candidate.get("candidate_text")
                or "Untitled outcome"
            )
            normalized_title = _normalize_review_title(title)
            status = str(generated_outcome.get("status") or "")
            if status.lower() in {"persisted", "published"} and normalized_title:
                persisted_rebuild_titles.add(normalized_title)

            rebuild_items.append(
                {
                    "id": f"rebuild:{generated_outcome.get('id')}",
                    "source": "rebuild",
                    "source_id": generated_outcome.get("id"),
                    "title": title,
                    "description": (
                        outcome_metadata.get("description")
                        or outcome_metadata.get("summary")
                        or (job_result.get("job") or {}).get("job_text")
                        or ""
                    ),
                    "type": generated_outcome.get("outcome_type") or "article",
                    "status": status or "draft",
                    "score": float(candidate.get("candidate_metadata", {}).get("opportunity_score") or 0),
                    "route": routing_decision.get("route"),
                    "keyword": keyword_pack.get("primary_keyword"),
                    "content_idea_id": generated_outcome.get("content_idea_id"),
                    "normalized_title": normalized_title,
                }
            )

    supabase = _get_admin_supabase_client()
    legacy_response = (
        supabase.table("content_ideas")
        .select("*")
        .eq("user_id", str(user_id))
        .eq("topic_id", str(topic_id))
        .order("created_at", desc=True)
        .execute()
    )
    legacy_rows = legacy_response.data or []

    legacy_items: list[dict] = []
    suppressed_legacy_count = 0
    for row in legacy_rows:
        metadata = row.get("idea_metadata") or {}
        topic_keyword_research = metadata.get("topic_keyword_research") or {}
        topic_editorial_generation = metadata.get("topic_editorial_generation") or {}
        raw_output = row.get("raw_dataforseo_output") or {}
        is_topic_generated = bool(
            topic_keyword_research.get("generation_origin") == "topic_keyword_pipeline_v1"
            or topic_keyword_research.get("keyword_cluster_id")
            or topic_keyword_research.get("cluster_name")
            or topic_editorial_generation.get("generation_origin") == "topic_editorial_pipeline_v1"
            or raw_output.get("topic_keyword_research_run_id")
        )
        if not is_topic_generated:
            continue

        is_published = bool(
            row.get("published")
            or row.get("published_to_titles")
            or row.get("titles_record_id")
            or str(row.get("status") or "").lower() == "published"
        )
        if is_published:
            continue

        normalized_title = _normalize_review_title(row.get("title"))
        if normalized_title and normalized_title in persisted_rebuild_titles and not include_suppressed_legacy:
            suppressed_legacy_count += 1
            continue

        primary_keywords = row.get("primary_keywords") or []
        first_keyword = primary_keywords[0] if isinstance(primary_keywords, list) and primary_keywords else None
        legacy_items.append(
            {
                "id": f"legacy:{row.get('id')}",
                "source": "legacy",
                "source_id": row.get("id"),
                "title": row.get("title") or "Untitled idea",
                "description": row.get("description") or "",
                "type": "software" if row.get("content_type") == "software" else "article",
                "status": row.get("status") or "draft",
                "score": float(row.get("opportunity_score") or 0),
                "route": None,
                "keyword": first_keyword,
                "content_idea_id": row.get("id"),
                "normalized_title": normalized_title,
            }
        )

    combined = []
    if source in {"all", "rebuild"}:
        combined.extend(rebuild_items)
    if source in {"all", "legacy"}:
        combined.extend(legacy_items)

    combined.sort(
        key=lambda row: (
            0 if row.get("source") == "rebuild" else 1,
            -float(row.get("score") or 0),
            str(row.get("title") or ""),
        )
    )

    for row in combined:
        row.pop("normalized_title", None)

    return combined[: max(1, min(limit, 50))], suppressed_legacy_count


async def _build_topic_context(
    *,
    user_id: UUID,
    topic_id: UUID,
    project_id: UUID,
    primary_category_id: UUID | None = None,
    secondary_category_id: UUID | None = None,
    review_source: str = "all",
    include_suppressed_legacy: bool = False,
    run_limit: int = 10,
    preview_limit: int = 6,
    review_limit: int = 8,
) -> dict:
    """Build the consolidated rebuild context used by TopicDetail."""
    runs = await _list_persisted_workflow_runs(
        user_id=user_id,
        project_id=project_id,
        primary_category_id=primary_category_id,
        secondary_category_id=secondary_category_id,
        limit=run_limit,
    )
    latest_run = runs[0] if runs else None
    latest_run_id = str((latest_run or {}).get("workflow_run_id") or "") if latest_run else ""
    latest_route = None
    if latest_run:
        latest_route = next(
            (
                route_value
                for route_value, _count_value in sorted(
                    (latest_run.get("route_counts") or {}).items(),
                    key=lambda item: int(item[1] or 0),
                    reverse=True,
                )
            ),
            None,
        )
    latest_preview_items: list[dict] = []
    if latest_run_id:
        latest_preview_items, _ = await _build_persisted_workflow_snapshot(
            user_id=user_id,
            project_id=project_id,
            primary_category_id=primary_category_id,
            secondary_category_id=secondary_category_id,
            workflow_run_id=latest_run_id,
            limit=preview_limit,
            offset=0,
        )

    review_items, suppressed_legacy_count = await _build_topic_review_items(
        user_id=user_id,
        topic_id=topic_id,
        project_id=project_id,
        primary_category_id=primary_category_id,
        secondary_category_id=secondary_category_id,
        workflow_run_id=latest_run_id or None,
        source=review_source,
        include_suppressed_legacy=include_suppressed_legacy,
        limit=review_limit,
    )

    return {
        "runs": runs,
        "latest_run": latest_run,
        "latest_workflow_run_id": latest_run_id or None,
        "latest_route": latest_route,
        "latest_preview_items": latest_preview_items,
        "review_items": review_items,
        "suppressed_legacy_count": suppressed_legacy_count,
    }


async def _build_topic_scope_summaries(
    *,
    user_id: UUID,
    project_id: UUID,
    primary_category_id: UUID | None = None,
    secondary_category_id: UUID | None = None,
    job_status: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Build compact rebuild summaries grouped by category scope."""
    runs = await _list_persisted_workflow_runs(
        user_id=user_id,
        project_id=project_id,
        primary_category_id=primary_category_id,
        secondary_category_id=secondary_category_id,
        job_status=job_status,
        limit=limit,
    )

    summaries: dict[str, dict] = {}
    for run in runs:
        primary_ids = [str(value or "") for value in run.get("primary_category_ids") or [] if str(value or "")]
        secondary_ids = [str(value or "") for value in run.get("secondary_category_ids") or [] if str(value or "")]
        route_counts = run.get("route_counts") or {}

        for primary_id_value in primary_ids:
            scope_secondary_ids = secondary_ids or [None]
            for secondary_id_value in scope_secondary_ids:
                key = f"{primary_id_value}:{secondary_id_value or ''}"
                summary = summaries.setdefault(
                    key,
                    {
                        "primary_category_id": primary_id_value,
                        "secondary_category_id": secondary_id_value,
                        "run_count": 0,
                        "latest_run": None,
                        "route_counts": {},
                    },
                )
                summary["run_count"] += 1
                latest_run = summary["latest_run"]
                if latest_run is None or str(run.get("started_at") or "") > str(latest_run.get("started_at") or ""):
                    summary["latest_run"] = run

                aggregate_route_counts = summary["route_counts"]
                for route_value, count_value in route_counts.items():
                    aggregate_route_counts[route_value] = aggregate_route_counts.get(route_value, 0) + int(count_value or 0)

    items: list[dict] = []
    for summary in summaries.values():
        dominant_route = None
        dominant_count = 0
        for route_value, count_value in (summary.get("route_counts") or {}).items():
            if int(count_value or 0) > dominant_count:
                dominant_route = route_value
                dominant_count = int(count_value or 0)

        items.append(
            {
                "project_id": str(project_id),
                "primary_category_id": summary["primary_category_id"],
                "secondary_category_id": summary["secondary_category_id"],
                "run_count": summary["run_count"],
                "latest_run": summary["latest_run"],
                "dominant_route": dominant_route,
                "route_counts": summary.get("route_counts") or {},
            }
        )

    return sorted(
        items,
        key=lambda row: str((row.get("latest_run") or {}).get("started_at") or ""),
        reverse=True,
    )


async def _build_workflow_context(
    *,
    user_id: UUID,
    project_id: UUID,
    primary_category_id: UUID | None = None,
    secondary_category_id: UUID | None = None,
    job_status: str | None = None,
    workflow_run_id: str | None = None,
    route: str | None = None,
    candidate_type: str | None = None,
    outcome_type: str | None = None,
    search: str | None = None,
    limit: int = 10,
    offset: int = 0,
    run_limit: int = 25,
) -> dict:
    """Build the consolidated rebuild context used by the ResearchRebuild page."""
    runs = await _list_persisted_workflow_runs(
        user_id=user_id,
        project_id=project_id,
        primary_category_id=primary_category_id,
        secondary_category_id=secondary_category_id,
        job_status=job_status,
        limit=run_limit,
    )
    items, total_jobs = await _build_persisted_workflow_snapshot(
        user_id=user_id,
        project_id=project_id,
        primary_category_id=primary_category_id,
        secondary_category_id=secondary_category_id,
        job_status=job_status,
        workflow_run_id=workflow_run_id,
        route=route,
        candidate_type=candidate_type,
        outcome_type=outcome_type,
        search=search,
        limit=limit,
        offset=offset,
    )
    return {
        "runs": runs,
        "snapshot": {
            "items": items,
            "count": len(items),
            "total_jobs": total_jobs,
            "limit": limit,
            "offset": offset,
        },
    }


async def _build_page_context(
    *,
    user_id: UUID,
    project_id: UUID,
    primary_category_id: UUID | None = None,
    secondary_category_id: UUID | None = None,
    job_status: str | None = None,
    workflow_run_id: str | None = None,
    route: str | None = None,
    candidate_type: str | None = None,
    outcome_type: str | None = None,
    search: str | None = None,
    limit: int = 10,
    offset: int = 0,
    run_limit: int = 25,
) -> dict:
    """Build the consolidated page context for ResearchRebuild."""
    jobs = await job_service.list_jobs(
        user_id=user_id,
        project_id=project_id,
        primary_category_id=primary_category_id,
        secondary_category_id=secondary_category_id,
        status=job_status,
        active_only=True,
    )
    workflow_context = await _build_workflow_context(
        user_id=user_id,
        project_id=project_id,
        primary_category_id=primary_category_id,
        secondary_category_id=secondary_category_id,
        job_status=job_status,
        workflow_run_id=workflow_run_id,
        route=route,
        candidate_type=candidate_type,
        outcome_type=outcome_type,
        search=search,
        limit=limit,
        offset=offset,
        run_limit=run_limit,
    )
    return {
        "jobs": {
            "items": jobs,
            "count": len(jobs),
        },
        "workflow": workflow_context,
    }


def _insert_with_schema_fallback(supabase, table_name: str, row: dict) -> dict | None:
    """
    Insert a row while tolerating missing legacy columns during migration.

    This mirrors the compatibility style already used in the current research
    topic persistence path.
    """
    payload = dict(row)
    max_attempts = max(12, len(payload) + 4)
    last_error = None

    for _ in range(max_attempts):
        try:
            result = supabase.table(table_name).insert(payload).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as exc:
            last_error = exc
            missing_cols = re.findall(r"Could not find the '([^']+)' column", str(exc))
            if not missing_cols:
                raise
            removed_any = False
            for col in missing_cols:
                if col in payload:
                    payload.pop(col, None)
                    removed_any = True
            if not removed_any:
                break

    if last_error:
        raise last_error
    return None


async def _load_candidate_or_404(candidate_id: str, user_id: str):
    return await candidate_service.get_record(
        record_id=_parse_uuid(candidate_id, "candidate_id"),
        user_id=UUID(user_id),
    )


@research_rebuild_bp.route("/jobs", methods=["GET"])
@require_api_key
def list_research_jobs():
    """List persisted user jobs for a project."""
    project_id = request.args.get("project_id")
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        items = asyncio.run(
            job_service.list_jobs(
                user_id=UUID(user_id),
                project_id=UUID(project_id),
                primary_category_id=UUID(request.args["primary_category_id"]) if request.args.get("primary_category_id") else None,
                secondary_category_id=UUID(request.args["secondary_category_id"]) if request.args.get("secondary_category_id") else None,
                status=request.args.get("status"),
                active_only=(request.args.get("active_only", "true").lower() != "false"),
            )
        )
        return jsonify({"items": items, "count": len(items)}), 200
    except Exception as exc:
        logger.error("research-rebuild list jobs failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to list jobs: {exc}"}), 500


@research_rebuild_bp.route("/jobs", methods=["POST"])
@require_api_key
def create_research_job():
    """Create a single manual user job."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json() or {}
    project_id = data.get("project_id")
    job_text = str(data.get("job_text") or "").strip()

    if not project_id:
        return jsonify({"error": "project_id is required"}), 400
    if not job_text:
        return jsonify({"error": "job_text is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        item = asyncio.run(
            job_service.create_job(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                primary_category_id=_parse_uuid(data["primary_category_id"], "primary_category_id") if data.get("primary_category_id") else None,
                secondary_category_id=_parse_uuid(data["secondary_category_id"], "secondary_category_id") if data.get("secondary_category_id") else None,
                job_text=job_text,
                job_type_hint=data.get("job_type_hint"),
                job_source=data.get("job_source") or "manual",
                status=data.get("status") or "draft",
                website_context_snapshot=data.get("website_context_snapshot") or {},
                generation_metadata=data.get("generation_metadata") or {},
            )
        )
        return jsonify(item), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild create job failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to create job: {exc}"}), 500


@research_rebuild_bp.route("/jobs/generate", methods=["POST"])
@require_api_key
def generate_research_jobs():
    """Generate and persist user jobs from website/category context."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    data = request.get_json() or {}
    project_id = data.get("project_id")
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400
    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401
    website_context = {
        "project_name": data.get("project_name"),
        "website_description": data.get("website_description"),
        "primary_category_name": data.get("primary_category_name"),
        "primary_category_description": data.get("primary_category_description"),
        "secondary_category_name": data.get("secondary_category_name"),
        "secondary_category_description": data.get("secondary_category_description"),
        "target_audience": data.get("target_audience"),
        "focus_area": data.get("focus_area"),
        "avoid_guidance": data.get("avoid_guidance"),
        "trend_titles": data.get("trend_titles") or [],
    }
    try:
        negative_context = asyncio.run(
            job_service.build_negative_context(user_id=UUID(user_id), project_id=_parse_uuid(project_id, "project_id"))
        )
        generated = asyncio.run(
            job_service.generate_jobs(
                context=website_context,
                count=int(data.get("count") or 30),
                negative_context=negative_context,
            )
        )
        archive_existing_in_scope = bool(data.get("archive_existing_in_scope", True))
        batch_id = str(uuid4())
        archived_count = 0
        if archive_existing_in_scope:
            archived_count = asyncio.run(
                job_service.archive_active_jobs_in_scope(
                    user_id=UUID(user_id),
                    project_id=_parse_uuid(project_id, "project_id"),
                    primary_category_id=_parse_uuid(data["primary_category_id"], "primary_category_id") if data.get("primary_category_id") else None,
                    secondary_category_id=_parse_uuid(data["secondary_category_id"], "secondary_category_id") if data.get("secondary_category_id") else None,
                )
            )
        generated_with_batch = []
        for item in generated:
            payload = dict(item)
            generation_metadata = dict(payload.get("generation_metadata") or {})
            generation_metadata.update(
                {
                    "batch_id": batch_id,
                    "focus_area": data.get("focus_area"),
                    "avoid_guidance": data.get("avoid_guidance"),
                }
            )
            payload["generation_metadata"] = generation_metadata
            generated_with_batch.append(payload)
        saved = asyncio.run(
            job_service.save_jobs(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                primary_category_id=_parse_uuid(data["primary_category_id"], "primary_category_id") if data.get("primary_category_id") else None,
                secondary_category_id=_parse_uuid(data["secondary_category_id"], "secondary_category_id") if data.get("secondary_category_id") else None,
                website_context_snapshot=website_context,
                jobs=generated_with_batch,
            )
        )
        return jsonify({"items": saved, "count": len(saved), "batch_id": batch_id, "archived_count": archived_count}), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild generate jobs failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to generate jobs: {exc}"}), 500


@research_rebuild_bp.route("/jobs/<job_id>/approve", methods=["POST"])
@require_api_key
def approve_research_job(job_id: str):
    """Approve a persisted user job."""
    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        item = asyncio.run(job_service.approve_job(job_id=UUID(job_id), user_id=UUID(user_id)))
        if not item:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(item), 200
    except Exception as exc:
        logger.error("research-rebuild approve job failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to approve job: {exc}"}), 500


@research_rebuild_bp.route("/jobs/<job_id>/reject", methods=["POST"])
@require_api_key
def reject_research_job(job_id: str):
    """Reject a persisted user job and capture structured feedback."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json() or {}
    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        item = asyncio.run(
            job_service.reject_job(
                job_id=_parse_uuid(job_id, "job_id"),
                user_id=UUID(user_id),
                rejection_reason_tags=data.get("rejection_reason_tags") or [],
                rejection_reason_free_text=data.get("rejection_reason_free_text"),
            )
        )
        if not item:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(item), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild reject job failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to reject job: {exc}"}), 500


@research_rebuild_bp.route("/candidates", methods=["GET"])
@require_api_key
def list_research_candidates():
    """List persisted opportunity candidates for a project."""
    project_id = request.args.get("project_id")
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        items = asyncio.run(
            candidate_service.list_candidates(
                user_id=UUID(user_id),
                project_id=UUID(project_id),
                user_job_id=UUID(request.args["user_job_id"]) if request.args.get("user_job_id") else None,
                candidate_type=request.args.get("candidate_type"),
                status=request.args.get("status"),
            )
        )
        return jsonify({"items": items, "count": len(items)}), 200
    except Exception as exc:
        logger.error("research-rebuild list candidates failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to list candidates: {exc}"}), 500


@research_rebuild_bp.route("/candidates", methods=["POST"])
@require_api_key
def create_research_candidate():
    """Create a single manual opportunity candidate."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json() or {}
    project_id = data.get("project_id")
    user_job_id = data.get("user_job_id")
    candidate_type = str(data.get("candidate_type") or "").strip()
    candidate_text = str(data.get("candidate_text") or "").strip()

    if not project_id:
        return jsonify({"error": "project_id is required"}), 400
    if not user_job_id:
        return jsonify({"error": "user_job_id is required"}), 400
    if candidate_type not in ALLOWED_CANDIDATE_TYPES:
        return jsonify({"error": "candidate_type must be one of seo_article, software, editorial"}), 400
    if not candidate_text:
        return jsonify({"error": "candidate_text is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        item = asyncio.run(
            candidate_service.create_candidate(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                user_job_id=_parse_uuid(user_job_id, "user_job_id"),
                candidate_type=candidate_type,
                candidate_text=candidate_text,
                normalized_candidate_text=data.get("normalized_candidate_text"),
                status=data.get("status") or "draft",
                candidate_metadata=data.get("candidate_metadata") or {},
                source_keywords_json=data.get("source_keywords_json") or [],
            )
        )
        return jsonify(item), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild create candidate failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to create candidate: {exc}"}), 500


@research_rebuild_bp.route("/candidates/generate", methods=["POST"])
@require_api_key
def generate_research_candidates():
    """Derive and persist candidates from an approved job."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    data = request.get_json() or {}
    project_id = data.get("project_id")
    user_job_id = data.get("user_job_id")
    if not project_id or not user_job_id:
        return jsonify({"error": "project_id and user_job_id are required"}), 400
    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401
    try:
        job = asyncio.run(job_service.get_record(record_id=_parse_uuid(user_job_id, "user_job_id"), user_id=UUID(user_id)))
        if not job:
            return jsonify({"error": "Job not found"}), 404
        website_context = job.get("website_context_snapshot") or {}
        generated = asyncio.run(
            candidate_service.derive_candidates_from_job(
                job=job,
                website_context=website_context,
            )
        )
        for item in generated:
            metadata = dict(item.get("candidate_metadata") or {})
            metadata.setdefault("category_context", {
                "project_id": project_id,
                "primary_category_id": job.get("primary_category_id"),
                "secondary_category_id": job.get("secondary_category_id"),
            })
            metadata.setdefault("job_text", job.get("job_text"))
            item["candidate_metadata"] = metadata
        saved = asyncio.run(
            candidate_service.save_candidates(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                user_job_id=_parse_uuid(user_job_id, "user_job_id"),
                candidates=generated,
            )
        )
        return jsonify({"items": saved, "count": len(saved)}), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild generate candidates failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to generate candidates: {exc}"}), 500


@research_rebuild_bp.route("/candidates/<candidate_id>/reject", methods=["POST"])
@require_api_key
def reject_research_candidate(candidate_id: str):
    """Reject a persisted opportunity candidate and capture structured feedback."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json() or {}
    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        item = asyncio.run(
            candidate_service.reject_candidate(
                candidate_id=_parse_uuid(candidate_id, "candidate_id"),
                user_id=UUID(user_id),
                rejection_reason_tags=data.get("rejection_reason_tags") or [],
                rejection_reason_free_text=data.get("rejection_reason_free_text"),
            )
        )
        if not item:
            return jsonify({"error": "Candidate not found"}), 404
        return jsonify(item), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild reject candidate failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to reject candidate: {exc}"}), 500


@research_rebuild_bp.route("/validation-runs", methods=["GET"])
@require_api_key
def list_validation_runs():
    """List validation runs for a candidate or project."""
    project_id = request.args.get("project_id")
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        items = asyncio.run(
            validation_service.list_validation_runs(
                user_id=UUID(user_id),
                project_id=UUID(project_id),
                candidate_id=UUID(request.args["candidate_id"]) if request.args.get("candidate_id") else None,
                freshness_state=request.args.get("freshness_state"),
            )
        )
        return jsonify({"items": items, "count": len(items)}), 200
    except Exception as exc:
        logger.error("research-rebuild list validation runs failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to list validation runs: {exc}"}), 500


@research_rebuild_bp.route("/validation-runs", methods=["POST"])
@require_api_key
def create_validation_run():
    """Persist a manual validation run, optionally with a SERP snapshot."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json() or {}
    project_id = data.get("project_id")
    candidate_id = data.get("candidate_id")
    validation_version = str(data.get("validation_version") or "").strip() or "manual_v1"

    if not project_id:
        return jsonify({"error": "project_id is required"}), 400
    if not candidate_id:
        return jsonify({"error": "candidate_id is required"}), 400

    freshness_state = data.get("freshness_state") or "fresh"
    if freshness_state not in ALLOWED_FRESHNESS_STATES:
        return jsonify({"error": "freshness_state must be one of fresh, stale, expired"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        payload = {
            "validated_at": data.get("validated_at"),
            "expires_at": data.get("expires_at"),
            "freshness_state": freshness_state,
            "eligibility_passed": bool(data.get("eligibility_passed", False)),
            "intent_match_score": data.get("intent_match_score"),
            "serp_weakness_score": data.get("serp_weakness_score"),
            "serp_gap_score": data.get("serp_gap_score"),
            "software_pattern_score": data.get("software_pattern_score"),
            "feasibility_score": data.get("feasibility_score"),
            "monetization_fit_score": data.get("monetization_fit_score"),
            "volume_score": data.get("volume_score"),
            "kd_ease_score": data.get("kd_ease_score"),
            "niche_drift_score": data.get("niche_drift_score"),
            "achievability_score": data.get("achievability_score"),
            "validation_reason_codes": data.get("validation_reason_codes") or [],
            "validation_metadata": data.get("validation_metadata") or {},
        }
        item = asyncio.run(
            validation_service.create_manual_validation_run(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                candidate_id=_parse_uuid(candidate_id, "candidate_id"),
                validation_version=validation_version,
                ttl_days=int(data.get("ttl_days") or 14),
                payload=payload,
            )
        )
        if not item:
            return jsonify({"error": "Failed to create validation run"}), 500

        serp_snapshot = None
        if isinstance(data.get("serp_snapshot"), dict):
            serp_payload = dict(data["serp_snapshot"])
            serp_payload.setdefault("validated_at", item.get("validated_at"))
            serp_payload.setdefault("snapshot_source", "manual")
            serp_snapshot = asyncio.run(
                validation_service.save_serp_snapshot(
                    user_id=UUID(user_id),
                    project_id=_parse_uuid(project_id, "project_id"),
                    candidate_id=_parse_uuid(candidate_id, "candidate_id"),
                    validation_run_id=_parse_uuid(item["id"], "validation_run_id"),
                    payload=serp_payload,
                )
            )

        return jsonify({"validation_run": item, "serp_snapshot": serp_snapshot}), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild create validation run failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to create validation run: {exc}"}), 500


@research_rebuild_bp.route("/validation-runs/validate", methods=["POST"])
@require_api_key
def validate_candidates():
    """Run first-pass validation for one or more persisted candidates and save the results."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    data = request.get_json() or {}
    project_id = data.get("project_id")
    candidate_ids = data.get("candidate_ids") or []
    if not project_id or not candidate_ids:
        return jsonify({"error": "project_id and candidate_ids are required"}), 400
    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401
    try:
        saved_items = []
        for candidate_id in candidate_ids[:40]:
            candidate = asyncio.run(_load_candidate_or_404(str(candidate_id), user_id))
            if not candidate:
                continue
            metadata = candidate.get("candidate_metadata") or {}
            website_context = {
                "website_description": data.get("website_description") or "",
                "primary_category_name": (metadata.get("category_context") or {}).get("primary_category_name"),
                "primary_category_description": data.get("primary_category_description") or "",
                "secondary_category_name": (metadata.get("category_context") or {}).get("secondary_category_name"),
                "secondary_category_description": data.get("secondary_category_description") or "",
            }
            validation_result = asyncio.run(
                validation_service.validate_candidate(
                    candidate=candidate,
                    website_context=website_context,
                    force_refresh=bool(data.get("force_refresh", False)),
                )
            )
            validation_row = asyncio.run(
                validation_service.create_manual_validation_run(
                    user_id=UUID(user_id),
                    project_id=_parse_uuid(project_id, "project_id"),
                    candidate_id=_parse_uuid(str(candidate_id), "candidate_id"),
                    validation_version="heuristic_v2",
                    ttl_days=int(data.get("ttl_days") or 14),
                    payload=validation_result,
                )
            )
            serp_snapshot = asyncio.run(
                validation_service.save_serp_snapshot(
                    user_id=UUID(user_id),
                    project_id=_parse_uuid(project_id, "project_id"),
                    candidate_id=_parse_uuid(str(candidate_id), "candidate_id"),
                    validation_run_id=_parse_uuid(validation_row["id"], "validation_run_id"),
                    payload={
                        "query_text": validation_result.get("validation_metadata", {}).get("query") or candidate.get("candidate_text"),
                        "snapshot_source": "dataforseo_serp_standard",
                        "validated_at": validation_result.get("validated_at"),
                        "top_results_json": validation_result.get("validation_metadata", {}).get("serp_rows")
                        or [],
                        "serp_summary_json": {
                            "serp_weakness_score": validation_result.get("serp_weakness_score"),
                            "intent_match_score": validation_result.get("intent_match_score"),
                        },
                    },
                )
            )
            saved_items.append({"validation_run": validation_row, "serp_snapshot": serp_snapshot})
        return jsonify({"items": saved_items, "count": len(saved_items)}), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild validate candidates failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to validate candidates: {exc}"}), 500


@research_rebuild_bp.route("/serp-snapshots", methods=["GET"])
@require_api_key
def list_serp_snapshots():
    """List SERP snapshots for a project, candidate, or validation run."""
    project_id = request.args.get("project_id")
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        items = asyncio.run(
            validation_service.list_serp_snapshots(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                candidate_id=_parse_uuid(request.args["candidate_id"], "candidate_id") if request.args.get("candidate_id") else None,
                validation_run_id=_parse_uuid(request.args["validation_run_id"], "validation_run_id") if request.args.get("validation_run_id") else None,
            )
        )
        return jsonify({"items": items, "count": len(items)}), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild list serp snapshots failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to list SERP snapshots: {exc}"}), 500


@research_rebuild_bp.route("/validation/refresh", methods=["POST"])
@require_api_key
def refresh_validation():
    """Refresh freshness metadata for an existing validation run."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json() or {}
    validation_run_id = data.get("validation_run_id")
    if not validation_run_id:
        return jsonify({"error": "validation_run_id is required"}), 400

    freshness_state = data.get("freshness_state")
    if freshness_state and freshness_state not in ALLOWED_FRESHNESS_STATES:
        return jsonify({"error": "freshness_state must be one of fresh, stale, expired"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        item = asyncio.run(
            validation_service.refresh_validation_run(
                validation_run_id=_parse_uuid(validation_run_id, "validation_run_id"),
                user_id=UUID(user_id),
                ttl_days=int(data["ttl_days"]) if data.get("ttl_days") is not None else None,
                freshness_state=freshness_state,
            )
        )
        if not item:
            return jsonify({"error": "Validation run not found"}), 404
        return jsonify(item), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild refresh validation failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to refresh validation: {exc}"}), 500


@research_rebuild_bp.route("/routing-decisions", methods=["GET"])
@require_api_key
def list_routing_decisions():
    """List routing decisions for a project or candidate."""
    project_id = request.args.get("project_id")
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        items = asyncio.run(
            routing_service.list_routing_decisions(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                candidate_id=_parse_uuid(request.args["candidate_id"], "candidate_id") if request.args.get("candidate_id") else None,
                route=request.args.get("route"),
            )
        )
        return jsonify({"items": items, "count": len(items)}), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild list routing decisions failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to list routing decisions: {exc}"}), 500


@research_rebuild_bp.route("/routing-decisions", methods=["POST"])
@require_api_key
def create_routing_decision():
    """Create a routing decision for a validated candidate."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json() or {}
    project_id = data.get("project_id")
    candidate_id = data.get("candidate_id")
    validation_run_id = data.get("validation_run_id")
    route = str(data.get("route") or "").strip()

    if not project_id or not candidate_id or not validation_run_id:
        return jsonify({"error": "project_id, candidate_id, and validation_run_id are required"}), 400
    if route not in ALLOWED_ROUTES:
        return jsonify({"error": "route is invalid"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        item = asyncio.run(
            routing_service.save_routing_decision(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                candidate_id=_parse_uuid(candidate_id, "candidate_id"),
                validation_run_id=_parse_uuid(validation_run_id, "validation_run_id"),
                route=route,
                route_reason_codes=data.get("route_reason_codes") or [],
                route_metadata=data.get("route_metadata") or {},
            )
        )
        return jsonify(item), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild create routing decision failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to create routing decision: {exc}"}), 500


@research_rebuild_bp.route("/routing-decisions/decide", methods=["POST"])
@require_api_key
def decide_routing():
    """Compute and persist a route from an existing validation run."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    data = request.get_json() or {}
    project_id = data.get("project_id")
    candidate_id = data.get("candidate_id")
    validation_run_id = data.get("validation_run_id")
    if not project_id or not candidate_id or not validation_run_id:
        return jsonify({"error": "project_id, candidate_id, and validation_run_id are required"}), 400
    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401
    try:
        candidate = asyncio.run(_load_candidate_or_404(candidate_id, user_id))
        validation_run = asyncio.run(
            validation_service.get_record(
                record_id=_parse_uuid(validation_run_id, "validation_run_id"),
                user_id=UUID(user_id),
            )
        )
        if not candidate or not validation_run:
            return jsonify({"error": "Candidate or validation run not found"}), 404
        decision = asyncio.run(
            routing_service.decide_route(candidate=candidate, validation_result=validation_run)
        )
        saved = asyncio.run(
            routing_service.save_routing_decision(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                candidate_id=_parse_uuid(candidate_id, "candidate_id"),
                validation_run_id=_parse_uuid(validation_run_id, "validation_run_id"),
                route=decision["route"],
                route_reason_codes=decision.get("route_reason_codes") or [],
                route_metadata=decision.get("route_metadata") or {},
            )
        )
        return jsonify(saved), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild decide routing failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to decide route: {exc}"}), 500


@research_rebuild_bp.route("/keyword-packs", methods=["GET"])
@require_api_key
def list_keyword_packs():
    """List keyword packs for a project or candidate."""
    project_id = request.args.get("project_id")
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        items = asyncio.run(
            keyword_pack_service.list_keyword_packs(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                candidate_id=_parse_uuid(request.args["candidate_id"], "candidate_id") if request.args.get("candidate_id") else None,
                keyword_pack_status=request.args.get("keyword_pack_status"),
            )
        )
        return jsonify({"items": items, "count": len(items)}), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild list keyword packs failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to list keyword packs: {exc}"}), 500


@research_rebuild_bp.route("/keyword-packs", methods=["POST"])
@require_api_key
def create_keyword_pack():
    """Persist a keyword pack for a validated candidate."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json() or {}
    project_id = data.get("project_id")
    candidate_id = data.get("candidate_id")
    validation_run_id = data.get("validation_run_id")
    keyword_pack_status = data.get("keyword_pack_status") or "draft"

    if not project_id or not candidate_id or not validation_run_id:
        return jsonify({"error": "project_id, candidate_id, and validation_run_id are required"}), 400
    if keyword_pack_status not in ALLOWED_KEYWORD_PACK_STATUSES:
        return jsonify({"error": "keyword_pack_status is invalid"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        item = asyncio.run(
            keyword_pack_service.save_keyword_pack(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                candidate_id=_parse_uuid(candidate_id, "candidate_id"),
                validation_run_id=_parse_uuid(validation_run_id, "validation_run_id"),
                payload={
                    "primary_keyword": data.get("primary_keyword"),
                    "secondary_keywords_json": data.get("secondary_keywords_json") or [],
                    "keyword_metrics_json": data.get("keyword_metrics_json") or {},
                    "keyword_pack_status": keyword_pack_status,
                    "keyword_pack_reason_codes": data.get("keyword_pack_reason_codes") or [],
                },
            )
        )
        return jsonify(item), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild create keyword pack failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to create keyword pack: {exc}"}), 500


@research_rebuild_bp.route("/keyword-packs/build", methods=["POST"])
@require_api_key
def build_keyword_pack():
    """Build and persist a keyword pack from a candidate + validation run."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    data = request.get_json() or {}
    project_id = data.get("project_id")
    candidate_id = data.get("candidate_id")
    validation_run_id = data.get("validation_run_id")
    if not project_id or not candidate_id or not validation_run_id:
        return jsonify({"error": "project_id, candidate_id, and validation_run_id are required"}), 400
    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401
    try:
        candidate = asyncio.run(_load_candidate_or_404(candidate_id, user_id))
        validation_run = asyncio.run(
            validation_service.get_record(
                record_id=_parse_uuid(validation_run_id, "validation_run_id"),
                user_id=UUID(user_id),
            )
        )
        if not candidate or not validation_run:
            return jsonify({"error": "Candidate or validation run not found"}), 404
        built = asyncio.run(
            keyword_pack_service.build_keyword_pack(candidate=candidate, validation_result=validation_run)
        )
        saved = asyncio.run(
            keyword_pack_service.save_keyword_pack(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                candidate_id=_parse_uuid(candidate_id, "candidate_id"),
                validation_run_id=_parse_uuid(validation_run_id, "validation_run_id"),
                payload=built,
            )
        )
        return jsonify(saved), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild build keyword pack failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to build keyword pack: {exc}"}), 500


@research_rebuild_bp.route("/internal-link-candidates", methods=["GET"])
@require_api_key
def list_internal_link_candidates():
    """List internal-link candidates for a project or candidate."""
    project_id = request.args.get("project_id")
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        items = asyncio.run(
            internal_link_fit_service.list_internal_link_candidates(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                candidate_id=_parse_uuid(request.args["candidate_id"], "candidate_id") if request.args.get("candidate_id") else None,
                link_role=request.args.get("link_role"),
            )
        )
        return jsonify({"items": items, "count": len(items)}), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild list internal link candidates failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to list internal-link candidates: {exc}"}), 500


@research_rebuild_bp.route("/internal-link-candidates", methods=["POST"])
@require_api_key
def create_internal_link_candidate():
    """Persist a parent/child/sibling/hub link suggestion."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json() or {}
    project_id = data.get("project_id")
    candidate_id = data.get("candidate_id")
    link_role = str(data.get("link_role") or "").strip()

    if not project_id or not candidate_id:
        return jsonify({"error": "project_id and candidate_id are required"}), 400
    if link_role not in ALLOWED_LINK_ROLES:
        return jsonify({"error": "link_role is invalid"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        item = asyncio.run(
            internal_link_fit_service.save_internal_link_candidate(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                candidate_id=_parse_uuid(candidate_id, "candidate_id"),
                payload={
                    "validation_run_id": str(_parse_uuid(data["validation_run_id"], "validation_run_id")) if data.get("validation_run_id") else None,
                    "wordpress_imported_post_id": str(_parse_uuid(data["wordpress_imported_post_id"], "wordpress_imported_post_id")) if data.get("wordpress_imported_post_id") else None,
                    "link_role": link_role,
                    "match_score": data.get("match_score"),
                    "match_reason_codes": data.get("match_reason_codes") or [],
                    "match_metadata": data.get("match_metadata") or {},
                },
            )
        )
        return jsonify(item), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild create internal link candidate failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to create internal-link candidate: {exc}"}), 500


@research_rebuild_bp.route("/internal-link-candidates/compute", methods=["POST"])
@require_api_key
def compute_internal_link_candidates():
    """Compute and persist internal-link candidates for a research candidate."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    data = request.get_json() or {}
    project_id = data.get("project_id")
    candidate_id = data.get("candidate_id")
    if not project_id or not candidate_id:
        return jsonify({"error": "project_id and candidate_id are required"}), 400
    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401
    try:
        candidate = asyncio.run(_load_candidate_or_404(candidate_id, user_id))
        if not candidate:
            return jsonify({"error": "Candidate not found"}), 404
        validation_run = None
        if data.get("validation_run_id"):
            validation_run = asyncio.run(
                validation_service.get_record(
                    record_id=_parse_uuid(data["validation_run_id"], "validation_run_id"),
                    user_id=UUID(user_id),
                )
            )
        computed = asyncio.run(
            internal_link_fit_service.compute_internal_link_fit(
                candidate=candidate,
                validation_result=validation_run,
            )
        )
        saved = []
        for item in (computed.get("items") or [])[:10]:
            row = asyncio.run(
                internal_link_fit_service.save_internal_link_candidate(
                    user_id=UUID(user_id),
                    project_id=_parse_uuid(project_id, "project_id"),
                    candidate_id=_parse_uuid(candidate_id, "candidate_id"),
                    payload=item,
                )
            )
            if row:
                saved.append(row)
        return jsonify({"items": saved, "count": len(saved)}), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild compute internal links failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to compute internal-link candidates: {exc}"}), 500


@research_rebuild_bp.route("/generated-outcomes", methods=["GET"])
@require_api_key
def list_generated_outcomes():
    """List generated outcomes for a project or candidate."""
    project_id = request.args.get("project_id")
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        items = asyncio.run(
            generation_service.list_generated_outcomes(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                candidate_id=_parse_uuid(request.args["candidate_id"], "candidate_id") if request.args.get("candidate_id") else None,
                outcome_type=request.args.get("outcome_type"),
                status=request.args.get("status"),
            )
        )
        return jsonify({"items": items, "count": len(items)}), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild list generated outcomes failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to list generated outcomes: {exc}"}), 500


@research_rebuild_bp.route("/topic-review", methods=["GET"])
@require_api_key
def get_topic_review():
    """Return a topic-scoped combined review queue for rebuild + legacy ideas."""
    topic_id = request.args.get("topic_id")
    project_id = request.args.get("project_id")
    if not topic_id:
        return jsonify({"error": "topic_id is required"}), 400
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    source = str(request.args.get("source") or "all").strip().lower()
    if source not in {"all", "rebuild", "legacy"}:
        return jsonify({"error": "source must be one of: all, rebuild, legacy"}), 400

    include_suppressed_legacy = str(request.args.get("include_suppressed_legacy") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    limit = int(request.args.get("limit") or 12)

    try:
        items, suppressed_legacy_count = asyncio.run(
            _build_topic_review_items(
                user_id=UUID(user_id),
                topic_id=_parse_uuid(topic_id, "topic_id"),
                project_id=_parse_uuid(project_id, "project_id"),
                primary_category_id=_parse_uuid(request.args["primary_category_id"], "primary_category_id")
                if request.args.get("primary_category_id")
                else None,
                secondary_category_id=_parse_uuid(request.args["secondary_category_id"], "secondary_category_id")
                if request.args.get("secondary_category_id")
                else None,
                workflow_run_id=request.args.get("workflow_run_id"),
                source=source,
                include_suppressed_legacy=include_suppressed_legacy,
                limit=limit,
            )
        )
        return jsonify(
            {
                "items": items,
                "count": len(items),
                "suppressed_legacy_count": suppressed_legacy_count,
            }
        ), 200
    except Exception as exc:
        logger.error("research-rebuild topic review failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to load topic review: {exc}"}), 500


@research_rebuild_bp.route("/topic-context", methods=["GET"])
@require_api_key
def get_topic_context():
    """Return the consolidated rebuild context for a topic detail page."""
    topic_id = request.args.get("topic_id")
    project_id = request.args.get("project_id")
    if not topic_id:
        return jsonify({"error": "topic_id is required"}), 400
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    review_source = str(request.args.get("review_source") or "all").strip().lower()
    if review_source not in {"all", "rebuild", "legacy"}:
        return jsonify({"error": "review_source must be one of: all, rebuild, legacy"}), 400

    include_suppressed_legacy = str(request.args.get("include_suppressed_legacy") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    try:
        payload = asyncio.run(
            _build_topic_context(
                user_id=UUID(user_id),
                topic_id=_parse_uuid(topic_id, "topic_id"),
                project_id=_parse_uuid(project_id, "project_id"),
                primary_category_id=_parse_uuid(request.args["primary_category_id"], "primary_category_id")
                if request.args.get("primary_category_id")
                else None,
                secondary_category_id=_parse_uuid(request.args["secondary_category_id"], "secondary_category_id")
                if request.args.get("secondary_category_id")
                else None,
                review_source=review_source,
                include_suppressed_legacy=include_suppressed_legacy,
                run_limit=int(request.args.get("run_limit") or 10),
                preview_limit=int(request.args.get("preview_limit") or 6),
                review_limit=int(request.args.get("review_limit") or 8),
            )
        )
        return jsonify(payload), 200
    except Exception as exc:
        logger.error("research-rebuild topic context failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to load topic context: {exc}"}), 500


@research_rebuild_bp.route("/topic-summaries", methods=["GET"])
@require_api_key
def get_topic_summaries():
    """Return compact rebuild summaries grouped by category scope."""
    project_ids = [value for value in request.args.getlist("project_id") if value]
    if not project_ids:
        single_project_id = request.args.get("project_id")
        if single_project_id:
            project_ids = [single_project_id]
    if not project_ids:
        return jsonify({"error": "project_id is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        parsed_project_ids = [_parse_uuid(project_id_value, "project_id") for project_id_value in project_ids]
        primary_category_id = (
            _parse_uuid(request.args["primary_category_id"], "primary_category_id")
            if request.args.get("primary_category_id")
            else None
        )
        secondary_category_id = (
            _parse_uuid(request.args["secondary_category_id"], "secondary_category_id")
            if request.args.get("secondary_category_id")
            else None
        )
        job_status = request.args.get("job_status")
        limit = int(request.args.get("limit") or 100)

        async def _load_many() -> list[dict]:
            items: list[dict] = []
            for parsed_project_id in parsed_project_ids:
                items.extend(
                    await _build_topic_scope_summaries(
                        user_id=UUID(user_id),
                        project_id=parsed_project_id,
                        primary_category_id=primary_category_id,
                        secondary_category_id=secondary_category_id,
                        job_status=job_status,
                        limit=limit,
                    )
                )
            return items

        payload = asyncio.run(_load_many())
        return jsonify({"items": payload, "count": len(payload)}), 200
    except Exception as exc:
        logger.error("research-rebuild topic summaries failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to load topic summaries: {exc}"}), 500


@research_rebuild_bp.route("/workflow-context", methods=["GET"])
@require_api_key
def get_workflow_context():
    """Return the consolidated rebuild context for the ResearchRebuild page."""
    project_id = request.args.get("project_id")
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        limit = int(request.args.get("limit") or 10)
        offset = int(request.args.get("offset") or 0)
        run_limit = int(request.args.get("run_limit") or 25)
        payload = asyncio.run(
            _build_workflow_context(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                primary_category_id=_parse_uuid(request.args["primary_category_id"], "primary_category_id")
                if request.args.get("primary_category_id")
                else None,
                secondary_category_id=_parse_uuid(request.args["secondary_category_id"], "secondary_category_id")
                if request.args.get("secondary_category_id")
                else None,
                job_status=request.args.get("job_status"),
                workflow_run_id=request.args.get("workflow_run_id"),
                route=request.args.get("route"),
                candidate_type=request.args.get("candidate_type"),
                outcome_type=request.args.get("outcome_type"),
                search=request.args.get("search"),
                limit=limit,
                offset=offset,
                run_limit=run_limit,
            )
        )
        return jsonify(payload), 200
    except Exception as exc:
        logger.error("research-rebuild workflow context failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to load workflow context: {exc}"}), 500


@research_rebuild_bp.route("/page-context", methods=["GET"])
@require_api_key
def get_page_context():
    """Return the consolidated page context for ResearchRebuild."""
    project_id = request.args.get("project_id")
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        limit = int(request.args.get("limit") or 10)
        offset = int(request.args.get("offset") or 0)
        run_limit = int(request.args.get("run_limit") or 25)
        payload = asyncio.run(
            _build_page_context(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                primary_category_id=_parse_uuid(request.args["primary_category_id"], "primary_category_id")
                if request.args.get("primary_category_id")
                else None,
                secondary_category_id=_parse_uuid(request.args["secondary_category_id"], "secondary_category_id")
                if request.args.get("secondary_category_id")
                else None,
                job_status=request.args.get("job_status"),
                workflow_run_id=request.args.get("workflow_run_id"),
                route=request.args.get("route"),
                candidate_type=request.args.get("candidate_type"),
                outcome_type=request.args.get("outcome_type"),
                search=request.args.get("search"),
                limit=limit,
                offset=offset,
                run_limit=run_limit,
            )
        )
        return jsonify(payload), 200
    except Exception as exc:
        logger.error("research-rebuild page context failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to load page context: {exc}"}), 500


@research_rebuild_bp.route("/generated-outcomes", methods=["POST"])
@require_api_key
def create_generated_outcome():
    """Persist a generated article/software/editorial outcome."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json() or {}
    project_id = data.get("project_id")
    candidate_id = data.get("candidate_id")
    outcome_type = str(data.get("outcome_type") or "").strip()
    status = data.get("status") or "draft"

    if not project_id or not candidate_id:
        return jsonify({"error": "project_id and candidate_id are required"}), 400
    if outcome_type not in ALLOWED_OUTCOME_TYPES:
        return jsonify({"error": "outcome_type is invalid"}), 400
    if status not in ALLOWED_OUTCOME_STATUSES:
        return jsonify({"error": "status is invalid"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        item = asyncio.run(
            generation_service.save_generated_outcome(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                candidate_id=_parse_uuid(candidate_id, "candidate_id"),
                payload={
                    "validation_run_id": str(_parse_uuid(data["validation_run_id"], "validation_run_id")) if data.get("validation_run_id") else None,
                    "routing_decision_id": str(_parse_uuid(data["routing_decision_id"], "routing_decision_id")) if data.get("routing_decision_id") else None,
                    "content_idea_id": str(_parse_uuid(data["content_idea_id"], "content_idea_id")) if data.get("content_idea_id") else None,
                    "outcome_type": outcome_type,
                    "status": status,
                    "outcome_metadata": data.get("outcome_metadata") or {},
                },
            )
        )
        return jsonify(item), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild create generated outcome failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to create generated outcome: {exc}"}), 500


@research_rebuild_bp.route("/generated-outcomes/generate", methods=["POST"])
@require_api_key
def generate_outcome():
    """Generate and persist an outcome from a routed candidate."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    data = request.get_json() or {}
    project_id = data.get("project_id")
    candidate_id = data.get("candidate_id")
    routing_decision_id = data.get("routing_decision_id")
    if not project_id or not candidate_id or not routing_decision_id:
        return jsonify({"error": "project_id, candidate_id, and routing_decision_id are required"}), 400
    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401
    try:
        candidate = asyncio.run(_load_candidate_or_404(candidate_id, user_id))
        routing_decision = asyncio.run(
            routing_service.get_record(
                record_id=_parse_uuid(routing_decision_id, "routing_decision_id"),
                user_id=UUID(user_id),
            )
        )
        keyword_pack = None
        if data.get("keyword_pack_id"):
            keyword_pack = asyncio.run(
                keyword_pack_service.get_record(
                    record_id=_parse_uuid(data["keyword_pack_id"], "keyword_pack_id"),
                    user_id=UUID(user_id),
                )
            )
        if not candidate or not routing_decision:
            return jsonify({"error": "Candidate or routing decision not found"}), 404
        generated = asyncio.run(
            generation_service.generate_outcome(
                candidate=candidate,
                routing_decision=routing_decision,
                keyword_pack=keyword_pack,
            )
        )
        generated["routing_decision_id"] = routing_decision_id
        generated["validation_run_id"] = data.get("validation_run_id") or routing_decision.get("validation_run_id")
        saved = asyncio.run(
            generation_service.save_generated_outcome(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                candidate_id=_parse_uuid(candidate_id, "candidate_id"),
                payload=generated,
            )
        )
        return jsonify(saved), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild generate outcome failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to generate outcome: {exc}"}), 500


@research_rebuild_bp.route("/generated-outcomes/<outcome_id>/persist-content-idea", methods=["POST"])
@require_api_key
def persist_generated_outcome_to_content_idea(outcome_id: str):
    """Bridge a generated outcome into the legacy content_ideas table."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json() or {}
    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        from .content_ideas import (
            _get_admin_supabase_client as _get_content_ideas_admin_client,
            _mark_content_idea_published,
            _publish_article_content_idea_to_titles,
        )

        outcome_uuid = _parse_uuid(outcome_id, "outcome_id")
        outcome = asyncio.run(generation_service.get_record(record_id=outcome_uuid, user_id=UUID(user_id)))
        if not outcome:
            return jsonify({"error": "Generated outcome not found"}), 404

        candidate_id = outcome.get("candidate_id")
        if not candidate_id:
            return jsonify({"error": "Generated outcome is missing candidate_id"}), 400

        candidate = asyncio.run(
            candidate_service.get_record(
                record_id=_parse_uuid(candidate_id, "candidate_id"),
                user_id=UUID(user_id),
            )
        )
        if not candidate:
            return jsonify({"error": "Candidate not found"}), 404

        keyword_pack = None
        validation_run_id = outcome.get("validation_run_id")
        project_id = data.get("project_id") or outcome.get("project_id") or candidate.get("project_id")
        if not project_id:
            return jsonify({"error": "project_id is required"}), 400

        if validation_run_id:
            packs = asyncio.run(
                keyword_pack_service.list_keyword_packs(
                    user_id=UUID(user_id),
                    project_id=_parse_uuid(project_id, "project_id"),
                    candidate_id=_parse_uuid(candidate_id, "candidate_id"),
                )
            )
            keyword_pack = packs[0] if packs else None

        provided_category_context = data.get("category_context")
        candidate_metadata = candidate.get("candidate_metadata") or {}
        category_context = (
            provided_category_context
            or (candidate_metadata.get("category_context") if isinstance(candidate_metadata, dict) else None)
            or {}
        )

        payload = asyncio.run(
            compatibility_adapter_service.outcome_to_content_idea_payload(
                candidate=candidate,
                generated_outcome=outcome,
                category_context=category_context,
                keyword_pack=keyword_pack,
            )
        )

        payload["user_id"] = user_id

        supabase = _get_admin_supabase_client()
        content_idea = None
        if outcome.get("content_idea_id"):
            existing_response = (
                supabase
                .table("content_ideas")
                .select("*")
                .eq("id", outcome.get("content_idea_id"))
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            content_idea = (existing_response.data or [None])[0]

        if not content_idea:
            topic_id = data.get("topic_id") or payload.get("topic_id") or candidate_id
            payload["topic_id"] = topic_id
            content_idea = _insert_with_schema_fallback(supabase, "content_ideas", payload)
            if not content_idea:
                return jsonify({"error": "Failed to insert content idea"}), 500

        supabase_admin = _get_content_ideas_admin_client(supabase)
        published_to_titles, titles_record_id = _publish_article_content_idea_to_titles(
            supabase,
            supabase_admin,
            idea=content_idea,
            user_id=user_id,
            now=datetime.utcnow().isoformat(),
        )
        if not published_to_titles:
            return jsonify({"error": "Failed to promote idea to Content Studio"}), 500

        _mark_content_idea_published(
            supabase,
            idea_id=str(content_idea.get("id")),
            user_id=user_id,
            now=datetime.utcnow().isoformat(),
            titles_record_id=titles_record_id,
        )

        if titles_record_id:
            refreshed_response = (
                supabase
                .table("content_ideas")
                .select("*")
                .eq("id", content_idea.get("id"))
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            content_idea = (refreshed_response.data or [content_idea])[0]

        updated_outcome = asyncio.run(
            generation_service.update_record(
                record_id=outcome_uuid,
                user_id=UUID(user_id),
                data={
                    "content_idea_id": content_idea.get("id"),
                    "status": "published",
                },
            )
        )

        return jsonify(
            {
                "content_idea": content_idea,
                "titles_record_id": titles_record_id,
                "generated_outcome": updated_outcome,
            }
        ), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild persist content idea failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to persist generated outcome to content_ideas: {exc}"}), 500


@research_rebuild_bp.route("/generated-outcomes/<outcome_id>/release-software", methods=["POST"])
@require_api_key
def release_generated_software_outcome(outcome_id: str):
    """Persist a software rebuild outcome directly into released_software_ideas."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    data = request.get_json() or {}
    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401
    try:
        outcome_uuid = _parse_uuid(outcome_id, "outcome_id")
        outcome = asyncio.run(generation_service.get_record(record_id=outcome_uuid, user_id=UUID(user_id)))
        if not outcome:
            return jsonify({"error": "Generated outcome not found"}), 404
        if str(outcome.get("outcome_type") or "") != "software":
            return jsonify({"error": "Only software outcomes can be released here"}), 400
        content_idea = None
        if outcome.get("content_idea_id"):
            supabase = _get_admin_supabase_client()
            result = (
                supabase
                .table("content_ideas")
                .select("*")
                .eq("id", outcome.get("content_idea_id"))
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            content_idea = (result.data or [None])[0]
        payload = asyncio.run(
            compatibility_adapter_service.outcome_to_released_software_payload(
                generated_outcome=outcome,
                content_idea=content_idea,
                user_id=user_id,
                released_at=datetime.utcnow().isoformat(),
            )
        )
        inserted = _insert_with_schema_fallback(_get_admin_supabase_client(), "released_software_ideas", payload)
        updated = asyncio.run(
            generation_service.update_record(
                record_id=outcome_uuid,
                user_id=UUID(user_id),
                data={"status": "published"},
            )
        )
        return jsonify({"released_software_idea": inserted, "generated_outcome": updated}), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild release software outcome failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to release software outcome: {exc}"}), 500


@research_rebuild_bp.route("/workflow/snapshot", methods=["GET"])
@require_api_key
def get_research_rebuild_workflow_snapshot():
    """Return the persisted rebuild workflow grouped by job."""
    project_id = request.args.get("project_id")
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        limit = int(request.args.get("limit") or 20)
        offset = int(request.args.get("offset") or 0)
        limit = max(1, min(limit, 100))
        offset = max(0, offset)

        items, total_jobs = asyncio.run(
            _build_persisted_workflow_snapshot(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                primary_category_id=_parse_uuid(request.args["primary_category_id"], "primary_category_id") if request.args.get("primary_category_id") else None,
                secondary_category_id=_parse_uuid(request.args["secondary_category_id"], "secondary_category_id") if request.args.get("secondary_category_id") else None,
                job_status=request.args.get("job_status"),
                workflow_run_id=request.args.get("workflow_run_id"),
                route=request.args.get("route"),
                candidate_type=request.args.get("candidate_type"),
                outcome_type=request.args.get("outcome_type"),
                search=request.args.get("search"),
                limit=limit,
                offset=offset,
            )
        )
        return jsonify(
            {
                "items": items,
                "count": len(items),
                "total_jobs": total_jobs,
                "limit": limit,
                "offset": offset,
            }
        ), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild workflow snapshot failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to load research rebuild workflow snapshot: {exc}"}), 500


@research_rebuild_bp.route("/workflow/runs", methods=["GET"])
@require_api_key
def list_research_rebuild_workflow_runs():
    """List recent persisted workflow runs derived from rebuild metadata."""
    project_id = request.args.get("project_id")
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        limit = int(request.args.get("limit") or 20)
        items = asyncio.run(
            _list_persisted_workflow_runs(
                user_id=UUID(user_id),
                project_id=_parse_uuid(project_id, "project_id"),
                primary_category_id=_parse_uuid(request.args["primary_category_id"], "primary_category_id") if request.args.get("primary_category_id") else None,
                secondary_category_id=_parse_uuid(request.args["secondary_category_id"], "secondary_category_id") if request.args.get("secondary_category_id") else None,
                job_status=request.args.get("job_status"),
                limit=limit,
            )
        )
        return jsonify({"items": items, "count": len(items)}), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild workflow runs failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to list research rebuild workflow runs: {exc}"}), 500


@research_rebuild_bp.route("/workflow/run", methods=["POST"])
@require_api_key
def run_research_rebuild_workflow():
    """
    Run the first-pass rebuild workflow for selected jobs.

    Current scope:
    - derive candidates from approved jobs
    - validate candidates
    - route candidates
    - build keyword packs
    - generate outcomes
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    data = request.get_json() or {}
    project_id = data.get("project_id")
    user_job_ids = data.get("user_job_ids") or []
    if not project_id or not user_job_ids:
        return jsonify({"error": "project_id and user_job_ids are required"}), 400
    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        workflow_run_id = str(uuid4())
        workflow_run_started_at = datetime.utcnow().isoformat()
        results = []
        for user_job_id in user_job_ids[:20]:
            job = asyncio.run(job_service.get_record(record_id=_parse_uuid(str(user_job_id), "user_job_id"), user_id=UUID(user_id)))
            if not job:
                continue
            website_context = job.get("website_context_snapshot") or {}
            generated_candidates = asyncio.run(
                candidate_service.derive_candidates_from_job(job=job, website_context=website_context)
            )
            for item in generated_candidates:
                metadata = dict(item.get("candidate_metadata") or {})
                metadata.setdefault("category_context", {
                    "project_id": project_id,
                    "primary_category_id": job.get("primary_category_id"),
                    "secondary_category_id": job.get("secondary_category_id"),
                })
                metadata.setdefault("job_text", job.get("job_text"))
                metadata["workflow_run_id"] = workflow_run_id
                metadata["workflow_run_started_at"] = workflow_run_started_at
                item["candidate_metadata"] = metadata
            saved_candidates = asyncio.run(
                candidate_service.save_candidates(
                    user_id=UUID(user_id),
                    project_id=_parse_uuid(project_id, "project_id"),
                    user_job_id=_parse_uuid(str(user_job_id), "user_job_id"),
                    candidates=generated_candidates,
                )
            )

            job_result = {"job_id": user_job_id, "job": job, "candidates": []}
            for candidate in saved_candidates:
                validation_payload = asyncio.run(
                    validation_service.validate_candidate(
                        candidate=candidate,
                        website_context=website_context,
                        force_refresh=False,
                    )
                )
                validation_metadata = dict(validation_payload.get("validation_metadata") or {})
                validation_metadata["workflow_run_id"] = workflow_run_id
                validation_metadata["workflow_run_started_at"] = workflow_run_started_at
                validation_payload["validation_metadata"] = validation_metadata
                validation_row = asyncio.run(
                    validation_service.create_manual_validation_run(
                        user_id=UUID(user_id),
                        project_id=_parse_uuid(project_id, "project_id"),
                        candidate_id=_parse_uuid(candidate["id"], "candidate_id"),
                        validation_version="heuristic_v2",
                        ttl_days=int(data.get("ttl_days") or 14),
                        payload=validation_payload,
                    )
                )
                asyncio.run(
                    validation_service.save_serp_snapshot(
                        user_id=UUID(user_id),
                        project_id=_parse_uuid(project_id, "project_id"),
                        candidate_id=_parse_uuid(candidate["id"], "candidate_id"),
                        validation_run_id=_parse_uuid(validation_row["id"], "validation_run_id"),
                        payload={
                            "query_text": validation_payload.get("validation_metadata", {}).get("query") or candidate.get("candidate_text"),
                            "snapshot_source": "dataforseo_serp_standard",
                            "validated_at": validation_payload.get("validated_at"),
                            "top_results_json": validation_payload.get("validation_metadata", {}).get("serp_rows") or [],
                            "serp_summary_json": {
                                "serp_weakness_score": validation_payload.get("serp_weakness_score"),
                                "intent_match_score": validation_payload.get("intent_match_score"),
                            },
                        },
                    )
                )
                route = asyncio.run(routing_service.decide_route(candidate=candidate, validation_result=validation_row))
                route_metadata = dict(route.get("route_metadata") or {})
                route_metadata["workflow_run_id"] = workflow_run_id
                route["route_metadata"] = route_metadata
                routing_row = asyncio.run(
                    routing_service.save_routing_decision(
                        user_id=UUID(user_id),
                        project_id=_parse_uuid(project_id, "project_id"),
                        candidate_id=_parse_uuid(candidate["id"], "candidate_id"),
                        validation_run_id=_parse_uuid(validation_row["id"], "validation_run_id"),
                        route=route["route"],
                        route_reason_codes=route.get("route_reason_codes") or [],
                        route_metadata=route.get("route_metadata") or {},
                    )
                )
                keyword_pack_payload = asyncio.run(
                    keyword_pack_service.build_keyword_pack(candidate=candidate, validation_result=validation_row)
                )
                keyword_metrics_json = dict(keyword_pack_payload.get("keyword_metrics_json") or {})
                keyword_metrics_json["_workflow_run"] = {
                    "workflow_run_id": workflow_run_id,
                    "workflow_run_started_at": workflow_run_started_at,
                }
                keyword_pack_payload["keyword_metrics_json"] = keyword_metrics_json
                keyword_pack_row = asyncio.run(
                    keyword_pack_service.save_keyword_pack(
                        user_id=UUID(user_id),
                        project_id=_parse_uuid(project_id, "project_id"),
                        candidate_id=_parse_uuid(candidate["id"], "candidate_id"),
                        validation_run_id=_parse_uuid(validation_row["id"], "validation_run_id"),
                        payload=keyword_pack_payload,
                    )
                )
                internal_link_computed = asyncio.run(
                    internal_link_fit_service.compute_internal_link_fit(
                        candidate=candidate,
                        validation_result=validation_row,
                    )
                )
                internal_link_rows = []
                for item in (internal_link_computed.get("items") or [])[:10]:
                    match_metadata = dict(item.get("match_metadata") or {})
                    match_metadata["workflow_run_id"] = workflow_run_id
                    item["match_metadata"] = match_metadata
                    row = asyncio.run(
                        internal_link_fit_service.save_internal_link_candidate(
                            user_id=UUID(user_id),
                            project_id=_parse_uuid(project_id, "project_id"),
                            candidate_id=_parse_uuid(candidate["id"], "candidate_id"),
                            payload=item,
                        )
                    )
                    if row:
                        internal_link_rows.append(row)
                outcome_payload = asyncio.run(
                    generation_service.generate_outcome(
                        candidate=candidate,
                        routing_decision=routing_row,
                        keyword_pack=keyword_pack_row,
                    )
                )
                outcome_metadata = dict(outcome_payload.get("outcome_metadata") or {})
                outcome_metadata["workflow_run_id"] = workflow_run_id
                outcome_metadata["workflow_run_started_at"] = workflow_run_started_at
                outcome_payload["outcome_metadata"] = outcome_metadata
                outcome_payload["routing_decision_id"] = routing_row["id"]
                outcome_payload["validation_run_id"] = validation_row["id"]
                outcome_row = asyncio.run(
                    generation_service.save_generated_outcome(
                        user_id=UUID(user_id),
                        project_id=_parse_uuid(project_id, "project_id"),
                        candidate_id=_parse_uuid(candidate["id"], "candidate_id"),
                        payload=outcome_payload,
                    )
                )
                job_result["candidates"].append(
                    {
                        "candidate": candidate,
                        "validation_run": validation_row,
                        "routing_decision": routing_row,
                        "keyword_pack": keyword_pack_row,
                        "internal_link_candidates": internal_link_rows,
                        "generated_outcome": outcome_row,
                    }
                )
            route_values = [
                str((candidate_result.get("routing_decision") or {}).get("route") or "").strip()
                for candidate_result in job_result["candidates"]
            ]
            has_promotable_route = any(route_value in PROMOTABLE_ROUTES for route_value in route_values)
            if not has_promotable_route:
                job_metadata = dict(job.get("generation_metadata") or {})
                job_metadata["last_workflow_run_id"] = workflow_run_id
                job_metadata["last_workflow_run_started_at"] = workflow_run_started_at
                job_metadata["last_route_outcomes"] = route_values
                updated_job = asyncio.run(
                    job_service.update_record(
                        record_id=_parse_uuid(str(user_job_id), "user_job_id"),
                        user_id=UUID(user_id),
                        data={
                            "status": "validated_no_opportunity",
                            "generation_metadata": job_metadata,
                        },
                    )
                )
                if updated_job:
                    job = updated_job
                    job_result["job"] = updated_job
            results.append(job_result)

        return jsonify(
            {
                "workflow_run_id": workflow_run_id,
                "started_at": workflow_run_started_at,
                "items": results,
                "count": len(results),
            }
        ), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("research-rebuild workflow run failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to run research rebuild workflow: {exc}"}), 500
