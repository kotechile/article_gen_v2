"""
Content Ideas API endpoints.

Provides list, publish, and delete actions used by the frontend Idea Burst flow.
"""

import logging
from datetime import datetime
from uuid import uuid4
from flask import Blueprint, jsonify, request

from ...core.models.errors import ErrorResponse
from ...api.middleware.auth import require_api_key

try:
    from supabase_client import get_supabase_client
except ImportError:
    import sys
    import os
    sys.path.append(os.getcwd())
    from supabase_client import get_supabase_client


logger = logging.getLogger(__name__)

content_ideas_bp = Blueprint("content_ideas", __name__, url_prefix="/api/content-ideas")


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
                title_payload = {
                    "id": str(uuid4()),
                    "user_id": user_id,
                    "Title": idea.get("title") or "Untitled Article",
                    "userDescription": idea.get("description") or "",
                    "Keywords": ", ".join(idea.get("keywords") or []),
                    "status": "Published",
                    "published": True,
                    "dateCreatedOn": now,
                    "source_idea_id": idea.get("id"),
                }
                try:
                    supabase.table("Titles").insert(title_payload).execute()
                    published_to_titles_count += 1
                except Exception:
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

        return jsonify({
            "success": True,
            "published_count": updated_count,
            "published_to_titles_count": published_to_titles_count,
            "requested_count": len(idea_ids),
        }), 200

    except Exception as e:
        logger.error(f"Error publishing content ideas: {e}", exc_info=True)
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
