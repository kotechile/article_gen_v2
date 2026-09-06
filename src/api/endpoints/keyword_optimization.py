"""
Keyword Optimization API Endpoints for Content Generator V2.

Provides endpoints for:
- Discovering DataForSEO keywords for an article
- Live search of keyword difficulty & volume
- AI natural keyword weaving
- Persisting selected keywords & metrics to Titles table
"""

from __future__ import annotations

import re
import asyncio
from typing import Optional, List, Dict, Any
import logging
from flask import Blueprint, request, jsonify

from src.services.keyword_optimization_service import keyword_optimization_service
from supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

keyword_optimization_bp = Blueprint("keyword_optimization", __name__, url_prefix="/api/v1/keywords")


def _get_user_id_from_auth() -> Optional[str]:
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
        logger.warning(f"[KeywordOptimization] Could not extract user_id from token: {err}")
    return None


@keyword_optimization_bp.route("/discover-for-article", methods=["POST"])
def discover_for_article():
    """Discover related keywords and fetch DataForSEO metrics."""
    try:
        payload = request.json or {}
        title = str(payload.get("title", "")).strip()
        content = str(payload.get("content", "")).strip()
        tags = payload.get("tags") or []
        custom_seed = payload.get("custom_seed")

        if not title and not content and not custom_seed:
            return jsonify({
                "success": False,
                "error": "At least title, content, or custom_seed must be provided.",
                "keywords": []
            }), 400

        keywords = asyncio.run(
            keyword_optimization_service.discover_keywords_for_article(
                title=title,
                content=content,
                tags=tags,
                custom_seed=custom_seed,
            )
        )

        return jsonify({
            "success": True,
            "keywords": keywords,
            "count": len(keywords),
        }), 200
    except Exception as err:
        logger.error(f"[KeywordOptimization] Discovery failed: {err}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(err),
            "keywords": [],
        }), 500


@keyword_optimization_bp.route("/search-dataforseo", methods=["POST"])
def search_dataforseo():
    """Search DataForSEO directly for a specific keyword phrase."""
    try:
        payload = request.json or {}
        query = str(payload.get("query", "")).strip()
        if not query:
            return jsonify({
                "success": False,
                "error": "Missing 'query' parameter.",
                "keywords": []
            }), 400

        keywords = asyncio.run(
            keyword_optimization_service.search_single_keyword(keyword=query)
        )

        return jsonify({
            "success": True,
            "keywords": keywords,
            "count": len(keywords),
        }), 200
    except Exception as err:
        logger.error(f"[KeywordOptimization] Search failed: {err}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(err),
            "keywords": [],
        }), 500


@keyword_optimization_bp.route("/weave-into-article", methods=["POST"])
def weave_into_article():
    """Naturally weave primary and secondary keywords into article HTML."""
    try:
        payload = request.json or {}
        html_content = payload.get("html", "")
        primary_keyword = str(payload.get("primary_keyword", "")).strip()
        secondary_keywords = payload.get("secondary_keywords") or []
        instructions = payload.get("instructions")

        if not html_content:
            return jsonify({
                "success": False,
                "error": "Missing 'html' content parameter."
            }), 400

        if not primary_keyword and not secondary_keywords:
            return jsonify({
                "success": False,
                "error": "Provide at least a primary_keyword or secondary_keywords."
            }), 400

        result = asyncio.run(
            keyword_optimization_service.weave_keywords_into_content(
                html_content=html_content,
                primary_keyword=primary_keyword,
                secondary_keywords=secondary_keywords,
                instructions=instructions,
            )
        )

        return jsonify(result), (200 if result.get("success") else 400)
    except Exception as err:
        logger.error(f"[KeywordOptimization] Weaving failed: {err}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(err)
        }), 500


@keyword_optimization_bp.route("/save-to-title", methods=["POST"])
def save_to_title():
    """Save selected keywords, metrics, and optionally updated HTML to the Titles record."""
    try:
        user_id = _get_user_id_from_auth()
        payload = request.json or {}
        if not user_id:
            user_id = payload.get("user_id")

        title_id = payload.get("title_id")
        if not title_id:
            return jsonify({
                "success": False,
                "error": "Missing required 'title_id' parameter."
            }), 400

        primary_kw = payload.get("primary_keyword")
        secondary_kws = payload.get("secondary_keywords") or []
        primary_metric = payload.get("primary_metric") or {}
        html_article = payload.get("html")

        # Build update payload
        update_data: dict = {
            "primary_keyword": primary_kw or None,
            "secondary_keywords_json": secondary_kws,
        }

        if primary_metric:
            vol = primary_metric.get("search_volume")
            kd = primary_metric.get("keyword_difficulty")
            intent = primary_metric.get("intent")

            if vol is not None:
                update_data["selected_keyword_search_volume"] = int(vol)
            if kd is not None:
                update_data["selected_keyword_difficulty"] = float(kd)
            if intent:
                update_data["selected_keyword_intent"] = str(intent)

            update_data["selected_keyword_metrics_json"] = {
                "primary": {
                    "keyword": primary_kw,
                    "search_volume": vol,
                    "keyword_difficulty": kd,
                    "cpc": primary_metric.get("cpc", 0.0),
                    "intent": intent,
                    "is_estimated": False,
                    "metric_source": "dataforseo_live_dossier",
                },
                "secondary": [
                    {
                        "keyword": kw,
                        "is_estimated": False,
                        "metric_source": "dataforseo_live_dossier",
                    }
                    for kw in secondary_kws
                ]
            }

        if html_article:
            update_data["htmlArticle"] = html_article
            update_data["articleText"] = re.sub(r"<[^>]+>", " ", html_article).strip()

        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"success": False, "error": "Database client unavailable."}), 500

        query = supabase.table("Titles").update(update_data).eq("id", str(title_id))
        if user_id:
            query = query.eq("user_id", user_id)

        res = query.execute()
        return jsonify({
            "success": True,
            "updated": res.data or [],
        }), 200

    except Exception as err:
        logger.error(f"[KeywordOptimization] Save to title failed: {err}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(err)
        }), 500
