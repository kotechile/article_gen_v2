"""
Content Ideas API endpoints.

Provides list, publish, and delete actions used by the frontend Idea Burst flow.
"""

import asyncio
import logging
import re
import time
from datetime import datetime
from uuid import uuid4
from flask import Blueprint, jsonify, request

from ...core.models.errors import ErrorResponse
from ...api.middleware.auth import require_api_key
from ...integrations.dataforseo import dataforseo_api
from ...services.affiliate_research_service import AffiliateResearchService

try:
    from supabase_client import get_supabase_client
except ImportError:
    import sys
    import os
    sys.path.append(os.getcwd())
    from supabase_client import get_supabase_client


logger = logging.getLogger(__name__)

content_ideas_bp = Blueprint("content_ideas", __name__, url_prefix="/api/content-ideas")
affiliate_research_service = AffiliateResearchService()

# Keep request latency bounded so reverse proxies don't return 504 on enrichment.
DATAFORSEO_BULK_TIMEOUT_SECONDS = 45
DATAFORSEO_KD_TIMEOUT_SECONDS = 20
AFFILIATE_ENRICH_TIMEOUT_SECONDS = 15
PER_IDEA_ENRICH_TIMEOUT_SECONDS = 70


def _resolve_user_id_from_request(supabase, data=None):
    auth_header = request.headers.get("Authorization")
    user_id = None

    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split("Bearer ")[1]
        try:
            user_response = supabase.auth.get_user(token)
            if user_response and user_response.user:
                user_id = user_response.user.id
        except Exception as auth_error:
            logger.warning(f"Failed to validate token or get user: {auth_error}")

    if not user_id and data and data.get("user_id"):
        user_id = data["user_id"]

    return user_id


def _extract_keywords_for_enrichment(idea: dict) -> list[str]:
    """Collect a normalized keyword list from idea payload fields."""
    candidates = []
    for field in ("primary_keywords", "keywords", "secondary_keywords"):
        value = idea.get(field)
        if isinstance(value, list):
            candidates.extend([str(item).strip() for item in value if str(item).strip()])
        elif isinstance(value, str) and value.strip():
            # Handle comma-separated fallback shapes.
            candidates.extend([part.strip() for part in value.split(",") if part.strip()])

    if not candidates:
        # Fallback: derive a few keyword-like tokens from title.
        title = str(idea.get("title") or "").strip().lower()
        tokens = re.findall(r"[a-z0-9]{3,}", title)
        candidates.extend(tokens[:8])

    seen = set()
    normalized = []
    for kw in candidates:
        key = kw.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(kw)
    return normalized[:20]


async def _compute_idea_enrichment(idea: dict) -> dict:
    """
    Compute SEO/offer enrichment for one idea.

    Returns aggregate metrics and a compact offers preview.
    """
    idea_id = idea.get("id")
    start_ts = time.perf_counter()
    keywords = _extract_keywords_for_enrichment(idea)
    logger.info(
        "Enrichment start for idea_id=%s title=%s keyword_count=%s",
        idea_id,
        (idea.get("title") or "")[:120],
        len(keywords),
    )
    if not keywords:
        logger.warning("Enrichment aborted for idea_id=%s: no keywords extracted", idea_id)
        return {
            "keywords_used": [],
            "total_search_volume": 0,
            "average_cpc": 0.0,
            "average_difficulty": 0.0,
            "affiliate_offer_count": 0,
            "affiliate_offers": [],
            "status": "failed",
            "reason": "No usable keywords found on idea",
        }

    total_search_volume = 0
    average_cpc = 0.0
    average_difficulty = 0.0
    cpc_count = 0
    kd_count = 0

    metrics_map = {}
    try:
        bulk_start = time.perf_counter()
        bulk_metrics = await asyncio.wait_for(
            dataforseo_api.get_bulk_metrics_standard(keywords),
            timeout=DATAFORSEO_BULK_TIMEOUT_SECONDS
        )
        logger.info(
            "DataForSEO bulk metrics completed for idea_id=%s in %.2fs rows=%s",
            idea_id,
            time.perf_counter() - bulk_start,
            len(bulk_metrics or []),
        )
        for item in (bulk_metrics or []):
            keyword = str(item.get("keyword") or "").strip().lower()
            if not keyword:
                continue
            metrics_map[keyword] = {
                "search_volume": item.get("search_volume") or 0,
                "cpc": item.get("cpc") or 0.0,
            }
    except Exception:
        logger.warning("DataForSEO bulk metrics failed for idea_id=%s", idea_id, exc_info=True)

    try:
        kd_start = time.perf_counter()
        kd_rows = await asyncio.wait_for(
            dataforseo_api.get_keyword_difficulty(keywords),
            timeout=DATAFORSEO_KD_TIMEOUT_SECONDS
        )
        logger.info(
            "DataForSEO keyword difficulty completed for idea_id=%s in %.2fs rows=%s",
            idea_id,
            time.perf_counter() - kd_start,
            len(kd_rows or []),
        )
        for item in (kd_rows or []):
            keyword = str(item.get("keyword") or "").strip().lower()
            if not keyword:
                continue
            existing = metrics_map.get(keyword, {})
            existing["keyword_difficulty"] = item.get("keyword_difficulty") or 0
            metrics_map[keyword] = existing
    except Exception:
        logger.warning("DataForSEO keyword difficulty failed for idea_id=%s", idea_id, exc_info=True)

    for keyword in keywords:
        row = metrics_map.get(keyword.lower(), {})
        search_volume = int(row.get("search_volume") or 0)
        cpc = float(row.get("cpc") or 0.0)
        difficulty = float(row.get("keyword_difficulty") or 0.0)

        total_search_volume += search_volume
        if cpc > 0:
            average_cpc += cpc
            cpc_count += 1
        if difficulty > 0:
            average_difficulty += difficulty
            kd_count += 1

    average_cpc = round((average_cpc / cpc_count) if cpc_count else 0.0, 2)
    average_difficulty = round((average_difficulty / kd_count) if kd_count else 0.0, 1)

    search_term = str(idea.get("title") or keywords[0]).strip()
    affiliate_offer_count = 0
    affiliate_offers_preview = []
    try:
        affiliate_start = time.perf_counter()
        affiliate_result = await asyncio.wait_for(
            affiliate_research_service.search_affiliate_programs(
                search_term=search_term,
                niche=str(idea.get("content_type") or "").strip() or None,
                user_id=idea.get("user_id"),
            ),
            timeout=AFFILIATE_ENRICH_TIMEOUT_SECONDS
        )
        logger.info(
            "Affiliate enrichment completed for idea_id=%s in %.2fs",
            idea_id,
            time.perf_counter() - affiliate_start,
        )
        programs = affiliate_result.get("programs") or []
        affiliate_offer_count = len(programs)
        affiliate_offers_preview = [
            {
                "name": program.get("name"),
                "network": program.get("network"),
                "commission_rate": program.get("commission_rate"),
            }
            for program in programs[:5]
        ]
    except Exception:
        logger.warning("Affiliate search failed for idea_id=%s", idea_id, exc_info=True)

    total_elapsed = time.perf_counter() - start_ts
    logger.info(
        "Enrichment complete for idea_id=%s in %.2fs volume=%s cpc=%s kd=%s offers=%s",
        idea_id,
        total_elapsed,
        int(total_search_volume),
        average_cpc,
        average_difficulty,
        affiliate_offer_count,
    )

    return {
        "keywords_used": keywords,
        "total_search_volume": int(total_search_volume),
        "average_cpc": average_cpc,
        "average_difficulty": average_difficulty,
        "affiliate_offer_count": affiliate_offer_count,
        "affiliate_offers": affiliate_offers_preview,
        "status": "enriched",
        "reason": None,
    }


@content_ideas_bp.route("/list", methods=["POST"])
@require_api_key
def list_content_ideas():
    try:
        if not request.is_json:
            return jsonify(ErrorResponse(
                error="invalid_content_type",
                message="Content-Type must be application/json",
                error_code="INVALID_CONTENT_TYPE",
                status=400,
            ).dict()), 400

        data = request.get_json() or {}
        supabase = get_supabase_client()
        request_user_id = _resolve_user_id_from_request(supabase, data)
        if not request_user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401,
            ).dict()), 401

        user_id = data.get("user_id") or request_user_id
        if user_id != request_user_id:
            return jsonify(ErrorResponse(
                error="forbidden",
                message="You do not have access to this user_id",
                error_code="FORBIDDEN",
                status=403,
            ).dict()), 403

        topic_id = data.get("topic_id")
        content_type = data.get("content_type")

        query = (
            supabase
            .table("content_ideas")
            .select("*")
            .eq("user_id", user_id)
        )
        if topic_id:
            query = query.eq("topic_id", topic_id)
        if content_type:
            query = query.eq("content_type", content_type)

        response = query.order("created_at", desc=True).execute()
        return jsonify(response.data or []), 200

    except Exception as e:
        logger.error(f"Error listing content ideas: {e}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500,
        ).dict()), 500


@content_ideas_bp.route("/publish", methods=["POST"])
@require_api_key
def publish_content_ideas():
    try:
        if not request.is_json:
            return jsonify(ErrorResponse(
                error="invalid_content_type",
                message="Content-Type must be application/json",
                error_code="INVALID_CONTENT_TYPE",
                status=400,
            ).dict()), 400

        data = request.get_json() or {}
        idea_ids = data.get("idea_ids") or []
        if not idea_ids:
            return jsonify(ErrorResponse(
                error="validation_error",
                message="idea_ids is required",
                error_code="VALIDATION_ERROR",
                status=400,
            ).dict()), 400

        supabase = get_supabase_client()
        request_user_id = _resolve_user_id_from_request(supabase, data)
        if not request_user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401,
            ).dict()), 401

        user_id = data.get("user_id") or request_user_id
        if user_id != request_user_id:
            return jsonify(ErrorResponse(
                error="forbidden",
                message="You do not have access to this user_id",
                error_code="FORBIDDEN",
                status=403,
            ).dict()), 403

        now = datetime.utcnow().isoformat()
        updated_count = 0
        published_to_titles_count = 0
        for idea_id in idea_ids:
            # 1) Fetch idea row first (works across old/new schemas).
            idea_resp = (
                supabase
                .table("content_ideas")
                .select("*")
                .eq("id", idea_id)
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            if not idea_resp.data:
                continue
            idea = idea_resp.data[0]

            # 2) For blog ideas, publish into Titles table directly.
            if (idea.get("content_type") or "").lower() != "software":
                primary_keywords = idea.get("primary_keywords") or idea.get("keywords") or []
                secondary_keywords = idea.get("secondary_keywords") or []
                if isinstance(primary_keywords, str):
                    primary_keywords = [k.strip() for k in primary_keywords.split(",") if k.strip()]
                if isinstance(secondary_keywords, str):
                    secondary_keywords = [k.strip() for k in secondary_keywords.split(",") if k.strip()]
                primary_keyword = primary_keywords[0] if primary_keywords else ""
                title_payload = {
                    "id": str(uuid4()),
                    "user_id": user_id,
                    "Title": idea.get("title") or "Untitled Article",
                    "userDescription": idea.get("description") or "",
                    "Keywords": ", ".join(primary_keywords),
                    # This record is queued for writing, not published to an external CMS.
                    "status": "New",
                    "published": False,
                    "dateCreatedOn": now,
                    "source_idea_id": idea.get("id"),
                    # Phase 1 keyword handoff defaults (Research -> Content Generation)
                    "keyword_candidates_json": primary_keywords + [k for k in secondary_keywords if k not in primary_keywords],
                    "keyword_clusters_json": [],
                    "keyword_research_status": "ready" if primary_keywords else "fallback",
                    "keyword_research_source": "dataforseo" if idea.get("total_search_volume") else "llm_fallback",
                    "keyword_research_confidence": 0.85 if idea.get("total_search_volume") else 0.35,
                    "keyword_research_generated_at": now,
                    "primary_keyword": primary_keyword,
                    "secondary_keywords_json": secondary_keywords,
                    "selected_keyword_search_volume": int(idea.get("total_search_volume") or 0),
                    "selected_keyword_difficulty": float(idea.get("average_difficulty") or 0.0),
                    "selected_keyword_intent": idea.get("target_intent") or "informational",
                    "keyword_selection_reason": "Initialized from research idea publish payload.",
                    "keyword_strategy_version": "phase1_v1",
                    "keyword_selection_source": "research_dossier_reused",
                }
                try:
                    supabase.table("Titles").insert(title_payload).execute()
                    published_to_titles_count += 1
                except Exception as insert_error:
                    err = str(insert_error)
                    missing_cols = re.findall(r"Could not find the '([^']+)' column", err)
                    if missing_cols:
                        fallback_payload = dict(title_payload)
                        for col in missing_cols:
                            fallback_payload.pop(col, None)
                        try:
                            supabase.table("Titles").insert(fallback_payload).execute()
                            published_to_titles_count += 1
                            logger.warning(
                                "Inserted Titles row for idea_id=%s after dropping missing columns: %s",
                                idea_id,
                                ", ".join(missing_cols),
                            )
                        except Exception:
                            logger.warning("Could not insert Titles row for idea_id=%s", idea_id, exc_info=True)
                    else:
                        logger.warning("Could not insert Titles row for idea_id=%s", idea_id, exc_info=True)

            # 3) Best-effort status update on content_ideas with progressive fallbacks.
            try:
                supabase.table("content_ideas").update({
                    "status": "published",
                    "published": True,
                    "published_to_titles": True,
                    "published_at": now,
                    "updated_at": now,
                }).eq("id", idea_id).eq("user_id", user_id).execute()
                updated_count += 1
                continue
            except Exception:
                pass

            try:
                supabase.table("content_ideas").update({
                    "status": "published",
                    "updated_at": now,
                }).eq("id", idea_id).eq("user_id", user_id).execute()
                updated_count += 1
                continue
            except Exception:
                pass

            try:
                supabase.table("content_ideas").update({
                    "updated_at": now,
                }).eq("id", idea_id).eq("user_id", user_id).execute()
                updated_count += 1
            except Exception:
                # Last fallback for minimal schemas without updated_at.
                result = (
                    supabase
                    .table("content_ideas")
                    .update({"description": idea.get("description") or ""})
                    .eq("id", idea_id)
                    .eq("user_id", user_id)
                    .execute()
                )
                if result.data:
                    updated_count += 1

        success = (updated_count > 0) or (published_to_titles_count > 0)
        status_code = 200 if success else 400
        return jsonify({
            "success": success,
            "published_count": updated_count,
            "published_to_titles_count": published_to_titles_count,
            "requested_count": len(idea_ids),
            "message": None if success else "No ideas were published. Verify idea IDs and schema.",
        }), status_code

    except Exception as e:
        logger.error(f"Error publishing content ideas: {e}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500,
        ).dict()), 500


@content_ideas_bp.route("/enrich", methods=["POST"])
@require_api_key
def enrich_content_ideas():
    """
    Enrich selected content ideas with SEO metrics and affiliate offer signals.
    """
    try:
        if not request.is_json:
            return jsonify(ErrorResponse(
                error="invalid_content_type",
                message="Content-Type must be application/json",
                error_code="INVALID_CONTENT_TYPE",
                status=400,
            ).dict()), 400

        data = request.get_json() or {}
        idea_ids = data.get("idea_ids") or []
        if not idea_ids:
            return jsonify(ErrorResponse(
                error="validation_error",
                message="idea_ids is required",
                error_code="VALIDATION_ERROR",
                status=400,
            ).dict()), 400

        supabase = get_supabase_client()
        request_user_id = _resolve_user_id_from_request(supabase, data)
        if not request_user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401,
            ).dict()), 401

        user_id = data.get("user_id") or request_user_id
        if user_id != request_user_id:
            return jsonify(ErrorResponse(
                error="forbidden",
                message="You do not have access to this user_id",
                error_code="FORBIDDEN",
                status=403,
            ).dict()), 403

        req_start = time.perf_counter()
        now = datetime.utcnow().isoformat()
        results = []
        enriched_count = 0
        logger.info(
            "Enrich request received user_id=%s idea_count=%s",
            user_id,
            len(idea_ids),
        )

        for idea_id in idea_ids:
            idea_start = time.perf_counter()
            fetch_resp = (
                supabase
                .table("content_ideas")
                .select("*")
                .eq("id", idea_id)
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            if not fetch_resp.data:
                logger.warning("Enrich idea fetch failed idea_id=%s user_id=%s", idea_id, user_id)
                results.append({
                    "idea_id": idea_id,
                    "status": "failed",
                    "reason": "Idea not found for user",
                })
                continue

            idea = fetch_resp.data[0]
            try:
                enrichment = asyncio.run(
                    asyncio.wait_for(
                        _compute_idea_enrichment(idea),
                        timeout=PER_IDEA_ENRICH_TIMEOUT_SECONDS
                    )
                )
            except TimeoutError:
                logger.warning("Per-idea enrichment timed out for idea_id=%s", idea_id)
                results.append({
                    "idea_id": idea_id,
                    "status": "failed",
                    "reason": "Enrichment timed out for this idea",
                })
                continue

            if enrichment.get("status") != "enriched":
                logger.warning(
                    "Enrichment failed for idea_id=%s reason=%s",
                    idea_id,
                    enrichment.get("reason"),
                )
                results.append({
                    "idea_id": idea_id,
                    "status": "failed",
                    "reason": enrichment.get("reason") or "Enrichment failed",
                })
                continue

            # If all enrichment metrics are zero, report this as a failed enrichment
            # so UI does not show a misleading "success" with no visible SEO stats.
            if (
                int(enrichment.get("total_search_volume") or 0) == 0
                and float(enrichment.get("average_cpc") or 0.0) == 0.0
                and float(enrichment.get("average_difficulty") or 0.0) == 0.0
                and int(enrichment.get("affiliate_offer_count") or 0) == 0
            ):
                logger.warning(
                    "Enrichment produced zero metrics for idea_id=%s keywords=%s",
                    idea_id,
                    enrichment.get("keywords_used") or [],
                )
                results.append({
                    "idea_id": idea_id,
                    "status": "failed",
                    "reason": "No SEO/offer metrics returned for this idea",
                })
                continue

            update_payload = {
                "total_search_volume": enrichment["total_search_volume"],
                "average_cpc": enrichment["average_cpc"],
                "average_difficulty": enrichment["average_difficulty"],
                "affiliate_offer_count": enrichment["affiliate_offer_count"],
                "status": "in_progress",
                "updated_at": now,
            }

            updated = False
            # Try richest payload first; gracefully degrade for older schemas.
            for payload in (
                {
                    **update_payload,
                    "idea_metadata": {
                        **(idea.get("idea_metadata") or {}),
                        "seo_offer_enrichment": {
                            "keywords_used": enrichment["keywords_used"],
                            "affiliate_offers_preview": enrichment["affiliate_offers"],
                            "enriched_at": now,
                        },
                    },
                },
                update_payload,
                {"updated_at": now},
            ):
                try:
                    supabase.table("content_ideas").update(payload).eq("id", idea_id).eq("user_id", user_id).execute()
                    updated = True
                    break
                except Exception:
                    continue

            if updated:
                enriched_count += 1
                logger.info(
                    "Enrichment persisted for idea_id=%s in %.2fs",
                    idea_id,
                    time.perf_counter() - idea_start,
                )
                results.append({
                    "idea_id": idea_id,
                    "status": "enriched",
                    "metrics": {
                        "total_search_volume": enrichment["total_search_volume"],
                        "average_cpc": enrichment["average_cpc"],
                        "average_difficulty": enrichment["average_difficulty"],
                        "affiliate_offer_count": enrichment["affiliate_offer_count"],
                    },
                    "keywords_used": enrichment["keywords_used"],
                    "offers_preview": enrichment["affiliate_offers"],
                })
            else:
                logger.error(
                    "Enrichment persistence failed for idea_id=%s after %.2fs",
                    idea_id,
                    time.perf_counter() - idea_start,
                )
                results.append({
                    "idea_id": idea_id,
                    "status": "failed",
                    "reason": "Could not persist enrichment values",
                })

        logger.info(
            "Enrich request finished user_id=%s requested=%s enriched=%s elapsed=%.2fs",
            user_id,
            len(idea_ids),
            enriched_count,
            time.perf_counter() - req_start,
        )
        return jsonify({
            "success": enriched_count > 0,
            "requested_count": len(idea_ids),
            "enriched_count": enriched_count,
            "results": results,
        }), 200 if enriched_count > 0 else 400

    except Exception as e:
        logger.error("Error enriching content ideas: %s", e, exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500,
        ).dict()), 500


@content_ideas_bp.route("/<idea_id>", methods=["DELETE"])
@require_api_key
def delete_content_idea(idea_id):
    try:
        supabase = get_supabase_client()
        request_user_id = _resolve_user_id_from_request(supabase, request.args)
        if not request_user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401,
            ).dict()), 401

        user_id = request.args.get("user_id") or request_user_id
        if user_id != request_user_id:
            return jsonify(ErrorResponse(
                error="forbidden",
                message="You do not have access to this user_id",
                error_code="FORBIDDEN",
                status=403,
            ).dict()), 403

        response = (
            supabase
            .table("content_ideas")
            .delete()
            .eq("id", idea_id)
            .eq("user_id", user_id)
            .execute()
        )

        if not response.data:
            return jsonify(ErrorResponse(
                error="not_found",
                message="Content idea not found",
                error_code="NOT_FOUND",
                status=404,
            ).dict()), 404

        return jsonify({"success": True, "id": idea_id}), 200

    except Exception as e:
        logger.error(f"Error deleting content idea {idea_id}: {e}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500,
        ).dict()), 500
