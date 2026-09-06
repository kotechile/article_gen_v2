"""
Editorial Factory API Endpoints for Content Generator V2.

Provides endpoints to:
- List available articles from the Editorial Factory Supabase database (GET /api/v1/editorial-factory/articles)
- Import an article into local Titles and format for Content Editor (POST /api/v1/editorial-factory/import)
"""

from __future__ import annotations

import logging
from flask import Blueprint, request, jsonify

from ...services.editorial_factory_service import editorial_factory_service
from supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

editorial_factory_bp = Blueprint("editorial_factory", __name__, url_prefix="/api/v1/editorial-factory")


def _get_user_id_from_auth() -> str | None:
    """Extract authenticated user_id from Authorization Bearer token."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None

    token = auth_header.split("Bearer ")[1].strip()
    supabase = get_supabase_client()
    if not supabase:
        return None

    try:
        user_res = supabase.auth.get_user(token)
        if user_res and user_res.user:
            return user_res.user.id
    except Exception as err:
        logger.warning(f"[EditorialFactory] Could not extract user_id from Bearer token: {err}")
    return None


@editorial_factory_bp.route("/articles", methods=["GET"])
def list_articles():
    """
    List articles from Editorial Factory.
    Query parameters:
    - search: optional string search filter
    - limit: maximum number of items (default 50)
    - offset: pagination offset (default 0)
    """
    try:
        search = request.args.get("search", "").strip()
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))

        articles = editorial_factory_service.list_articles(
            search=search,
            limit=min(limit, 100),
            offset=max(offset, 0)
        )

        return jsonify({
            "success": True,
            "articles": articles,
            "count": len(articles)
        }), 200
    except Exception as err:
        logger.error(f"[EditorialFactory] Error listing articles: {err}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(err),
            "articles": []
        }), 500


@editorial_factory_bp.route("/import", methods=["POST"])
def import_article():
    """
    Import an article from Editorial Factory into local Titles.
    Expected JSON payload:
    {
        "article_id": "uuid-or-id",
        "domain": "optional-target-domain",
        "wordpress_category_id": 123,           // optional
        "wordpress_parent_category_id": 456      // optional
    }
    """
    try:
        user_id = _get_user_id_from_auth()
        if not user_id:
            # Fallback to user_id from body if provided (e.g. dev/admin mode)
            user_id = (request.json or {}).get("user_id")

        if not user_id:
            return jsonify({
                "success": False,
                "error": "Unauthorized: Authentication required."
            }), 401

        payload = request.json or {}
        article_id = payload.get("article_id")
        if not article_id:
            return jsonify({
                "success": False,
                "error": "Missing required 'article_id' parameter."
            }), 400

        target_domain = payload.get("domain")
        target_category_id = payload.get("wordpress_category_id")
        target_parent_category_id = payload.get("wordpress_parent_category_id")

        success, title_id, result = editorial_factory_service.import_article_to_titles(
            article_id=str(article_id),
            user_id=user_id,
            target_domain=target_domain,
            target_category_id=target_category_id,
            target_parent_category_id=target_parent_category_id,
        )

        if not success or not title_id:
            return jsonify({
                "success": False,
                "error": (result or {}).get("error", "Failed to import article.")
            }), 400

        return jsonify({
            "success": True,
            "title_id": title_id,
            "data": result
        }), 201
    except Exception as err:
        logger.error(f"[EditorialFactory] Error importing article: {err}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(err)
        }), 500
