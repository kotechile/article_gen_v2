"""
Settings API endpoints.

Provides read/write access to research filtering settings used by the UI.
"""

import logging
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


def _normalize_research_settings(raw: dict | None) -> dict:
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
