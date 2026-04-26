"""
Settings API endpoints.

Provides read/write access to research filtering settings used by the UI.
"""

import logging
from typing import Optional
from flask import Blueprint, jsonify, request

from supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

settings_bp = Blueprint("settings", __name__, url_prefix="/api/settings")

DEFAULT_RESEARCH_SETTINGS = {
    "min_volume": 50,
    "max_difficulty": 50,
    "min_cpc": 0.5,
    "strict_mode": True,
}


def _normalize_llm_provider_rows(rows: Optional[list]) -> list[dict]:
    normalized = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        model_name = str(row.get("model_name") or "").strip()
        if not model_name:
            continue
        normalized.append({
            "id": str(row.get("id") or ""),
            "name": str(row.get("name") or model_name),
            "provider": str(row.get("provider") or ""),
            "model_name": model_name,
            "api_keys_id": row.get("api_keys_id"),
            "is_default": row.get("is_default") if isinstance(row.get("is_default"), bool) else None,
            "is_active": row.get("is_active") if isinstance(row.get("is_active"), bool) else None,
            "used_for": row.get("used_for"),
        })
    return normalized


def _fetch_llm_providers_with_fallbacks(supabase) -> list[dict]:
    attempts = [
        ("active-with-flags", "id,name,provider,model_name,api_keys_id,is_default,is_active,used_for", True),
        ("all-with-flags", "id,name,provider,model_name,api_keys_id,is_default,is_active,used_for", False),
        ("all-with-default", "id,name,provider,model_name,api_keys_id,is_default", False),
        ("all-core-fields", "id,name,provider,model_name,api_keys_id", False),
    ]

    for label, select_fields, active_only in attempts:
        try:
            query = supabase.table("llm_providers").select(select_fields)
            if active_only:
                query = query.eq("is_active", True)
            response = query.execute()
            rows = _normalize_llm_provider_rows(response.data)
            if rows:
                return rows
        except Exception:
            logger.warning("LLM providers query attempt failed: %s", label, exc_info=True)

    return []


def _normalize_research_settings(raw: Optional[dict]) -> dict:
    values = dict(DEFAULT_RESEARCH_SETTINGS)
    payload = raw or {}

    try:
        values["min_volume"] = int(payload.get("min_volume", values["min_volume"]))
    except (TypeError, ValueError):
        pass
    try:
        values["max_difficulty"] = int(payload.get("max_difficulty", values["max_difficulty"]))
    except (TypeError, ValueError):
        pass
    try:
        values["min_cpc"] = float(payload.get("min_cpc", values["min_cpc"]))
    except (TypeError, ValueError):
        pass
    values["strict_mode"] = bool(payload.get("strict_mode", values["strict_mode"]))

    # Keep values in expected ranges.
    values["min_volume"] = max(0, values["min_volume"])
    values["max_difficulty"] = min(100, max(0, values["max_difficulty"]))
    values["min_cpc"] = max(0.0, values["min_cpc"])

    return values


@settings_bp.route("/research", methods=["GET"])
def get_research_settings():
    """Return research settings from application_settings.research_settings (id=1)."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({
                "success": False,
                "message": "Database connection failed",
                "data": DEFAULT_RESEARCH_SETTINGS,
            }), 500

        response = (
            supabase
            .table("application_settings")
            .select("research_settings")
            .eq("id", 1)
            .limit(1)
            .execute()
        )

        raw = {}
        if response.data and len(response.data) > 0:
            raw = response.data[0].get("research_settings") or {}

        return jsonify({
            "success": True,
            "data": _normalize_research_settings(raw),
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch research settings", exc_info=True)
        return jsonify({
            "success": False,
            "message": "Failed to retrieve settings",
            "data": DEFAULT_RESEARCH_SETTINGS,
            "error": str(exc),
        }), 200


@settings_bp.route("/research", methods=["POST"])
def update_research_settings():
    """Update application_settings.research_settings (id=1)."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({
                "success": False,
                "message": "Database connection failed",
            }), 500

        payload = request.get_json(silent=True) or {}
        settings = _normalize_research_settings(payload)

        update_data = {"research_settings": settings}
        response = (
            supabase
            .table("application_settings")
            .update(update_data)
            .eq("id", 1)
            .execute()
        )

        # If row 1 does not exist, create it.
        if not response.data:
            supabase.table("application_settings").insert({"id": 1, **update_data}).execute()

        return jsonify({
            "success": True,
            "data": settings,
            "message": "Settings saved successfully",
        }), 200
    except Exception as exc:
        logger.error("Failed to update research settings", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Failed to save settings: {str(exc)}",
        }), 500


@settings_bp.route("/llm-providers", methods=["GET"])
def get_llm_providers():
    """Return LLM provider rows from Supabase using backend service-role credentials."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({
                "success": False,
                "message": "Database connection failed",
                "data": [],
            }), 500

        providers = _fetch_llm_providers_with_fallbacks(supabase)
        providers.sort(key=lambda row: (not bool(row.get("is_default")), (row.get("name") or row.get("model_name") or "").lower()))

        return jsonify({
            "success": True,
            "data": providers,
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch llm providers", exc_info=True)
        return jsonify({
            "success": False,
            "message": "Failed to fetch llm providers",
            "data": [],
            "error": str(exc),
        }), 500
