"""
LinkedIn API endpoints for Content Generator V2.

Provides routes for:
- OAuth flow (/api/linkedin/auth-url, /api/linkedin/callback)
- Connection status and disconnection (/api/linkedin/account)
- Direct publishing to LinkedIn personal feeds (/api/linkedin/publish)
- AI-powered article repurposing for LinkedIn (/api/linkedin/repurpose)
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, redirect

from ...api.middleware.auth import require_api_key
from ...services.linkedin_service import linkedin_service
from supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

linkedin_bp = Blueprint('linkedin', __name__, url_prefix='/api/linkedin')


def _get_user_id_from_auth() -> str | None:
    """Extract authenticated user_id from Authorization Bearer token."""
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return None

    token = auth_header.split('Bearer ')[1].strip()
    supabase = get_supabase_client()
    if not supabase:
        return None

    try:
        user_res = supabase.auth.get_user(token)
        if user_res and user_res.user:
            return user_res.user.id
    except Exception as err:
        logger.warning(f"Could not extract user_id from Bearer token: {err}")
    return None


def _extract_linkedin_creds_from_dict(data: dict) -> tuple[str | None, str | None, str | None]:
    """Helper to extract LinkedIn client_id, client_secret, redirect_uri from any dict or row."""
    if not isinstance(data, dict):
        return None, None, None

    # Check client_id variations
    c_id = (
        data.get("LINKEDIN_CLIENT_ID")
        or data.get("linkedin_client_id")
        or data.get("linkedinClientId")
        or data.get("linkedinKey")
        or data.get("linkedin_key")
    )
    # Check client_secret variations
    c_sec = (
        data.get("LINKEDIN_CLIENT_SECRET")
        or data.get("linkedin_client_secret")
        or data.get("linkedinClientSecret")
        or data.get("linkedinSecret")
        or data.get("linkedin_secret")
    )
    # Check redirect_uri variations
    r_uri = (
        data.get("LINKEDIN_REDIRECT_URI")
        or data.get("linkedin_redirect_uri")
        or data.get("linkedinRedirectUri")
        or data.get("linkedinCallbackUrl")
    )

    # Also check if nested in a json field like research_settings or linkedin_settings
    for nested_field in ("linkedin_settings", "linkedin", "research_settings"):
        nested = data.get(nested_field)
        if isinstance(nested, dict):
            n_id, n_sec, n_uri = _extract_linkedin_creds_from_dict(nested)
            c_id = c_id or n_id
            c_sec = c_sec or n_sec
            r_uri = r_uri or n_uri

    return (
        str(c_id).strip() if c_id else None,
        str(c_sec).strip() if c_sec else None,
        str(r_uri).strip() if r_uri else None,
    )


def _get_linkedin_service_for_user(user_id: str | None = None) -> LinkedInService:
    """
    Returns a LinkedInService instance.
    Resolution priority:
    1. application_settings table in Supabase (global application API credentials).
    2. user_profile table in Supabase (per-user override if present).
    3. Environment variables.
    """
    client_id = os.getenv("LINKEDIN_CLIENT_ID", "").strip() or None
    client_secret = os.getenv("LINKEDIN_CLIENT_SECRET", "").strip() or None
    redirect_uri = os.getenv("LINKEDIN_REDIRECT_URI", "").strip() or None

    supabase = get_supabase_client()
    if supabase:
        # 1. Check application_settings table (primary for VPS)
        try:
            app_res = supabase.table("application_settings").select("*").limit(1).execute()
            if app_res.data and len(app_res.data) > 0:
                a_id, a_sec, a_uri = _extract_linkedin_creds_from_dict(app_res.data[0])
                if a_id:
                    client_id = a_id
                if a_sec:
                    client_secret = a_sec
                if a_uri:
                    redirect_uri = a_uri
        except Exception as e:
            logger.debug(f"Could not load LinkedIn credentials from application_settings: {e}")

        # 2. Check user_profile table (for per-user override if present)
        if user_id:
            try:
                user_res = supabase.table("user_profile").select("*").eq("id", user_id).limit(1).execute()
                if user_res.data and len(user_res.data) > 0:
                    u_id, u_sec, u_uri = _extract_linkedin_creds_from_dict(user_res.data[0])
                    if u_id:
                        client_id = u_id
                    if u_sec:
                        client_secret = u_sec
                    if u_uri:
                        redirect_uri = u_uri
            except Exception as e:
                logger.debug(f"Could not load LinkedIn credentials from user_profile: {e}")

    # Fallback redirect_uri default if still unset
    if not redirect_uri:
        redirect_uri = "https://content.buildomain.com/api/linkedin/callback"

    return LinkedInService(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
    )


@linkedin_bp.route('/auth-url', methods=['GET'])
def get_auth_url():
    """
    Return the LinkedIn OAuth 2.0 authorization URL.
    Passes user_id in the state parameter for correlation in callback.
    """
    user_id = _get_user_id_from_auth()
    state = json.dumps({"user_id": user_id or "anonymous", "timestamp": datetime.utcnow().isoformat()})
    
    try:
        service = _get_linkedin_service_for_user(user_id)
        url = service.get_authorization_url(state=state)
        return jsonify({"auth_url": url})
    except ValueError as e:
        return jsonify({"error": "configuration_error", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"Error generating LinkedIn auth URL: {e}", exc_info=True)
        return jsonify({"error": "server_error", "message": str(e)}), 500


@linkedin_bp.route('/callback', methods=['GET'])
def oauth_callback():
    """
    OAuth 2.0 redirect callback endpoint from LinkedIn.
    Exchanges code for access token, retrieves member profile, saves account to DB,
    and redirects user back to the application Settings tab.
    """
    frontend_base = os.getenv("FRONTEND_URL")
    if not frontend_base:
        forwarded_host = request.headers.get("X-Forwarded-Host") or request.host
        forwarded_proto = request.headers.get("X-Forwarded-Proto") or ("https" if request.is_secure else "http")
        if "localhost" in forwarded_host or "127.0.0.1" in forwarded_host:
            frontend_base = "http://localhost:5173"
        else:
            frontend_base = f"{forwarded_proto}://{forwarded_host}"
    frontend_base = frontend_base.rstrip("/")
    redirect_target = f"{frontend_base}/settings?tab=integrations"

    error = request.args.get('error')
    error_desc = request.args.get('error_description')
    if error:
        logger.warning(f"LinkedIn OAuth error: {error} - {error_desc}")
        return redirect(f"{redirect_target}&linkedin_error={error}")

    code = request.args.get('code')
    state_raw = request.args.get('state', '')

    if not code:
        return redirect(f"{redirect_target}&linkedin_error=missing_code")

    user_id = None
    try:
        state_data = json.loads(state_raw)
        user_id = state_data.get("user_id")
        if user_id == "anonymous":
            user_id = None
    except Exception:
        pass

    try:
        service = _get_linkedin_service_for_user(user_id)
        # Step 1: Exchange code for tokens
        token_data = service.exchange_code_for_token(code)
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        expires_at = token_data.get("expires_at")

        if not access_token:
            return redirect(f"{redirect_target}&linkedin_error=no_token")

        # Step 2: Fetch profile
        profile = service.get_member_profile(access_token)
        member_urn = profile.get("urn")
        member_name = profile.get("name")
        picture_url = profile.get("picture")

        # Step 3: Persist to Supabase linkedin_accounts
        supabase = get_supabase_client()
        if supabase:
            account_record = {
                "account_type": "personal",
                "linkedin_urn": member_urn,
                "account_name": member_name,
                "profile_picture_url": picture_url,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_at": expires_at,
                "updated_at": datetime.utcnow().isoformat(),
            }
            if user_id:
                account_record["user_id"] = user_id

            try:
                # Upsert by user_id or linkedin_urn
                existing = None
                if user_id:
                    check_q = supabase.table("linkedin_accounts").select("id").eq("user_id", user_id).limit(1).execute()
                    if check_q.data and len(check_q.data) > 0:
                        existing = check_q.data[0]["id"]

                if existing:
                    supabase.table("linkedin_accounts").update(account_record).eq("id", existing).execute()
                else:
                    supabase.table("linkedin_accounts").insert(account_record).execute()
            except Exception as db_err:
                logger.error(f"Error persisting LinkedIn account to database: {db_err}", exc_info=True)

        return redirect(f"{redirect_target}&linkedin_connected=true")

    except Exception as e:
        logger.error(f"Error handling LinkedIn OAuth callback: {e}", exc_info=True)
        return redirect(f"{redirect_target}&linkedin_error=callback_failed")


@linkedin_bp.route('/account', methods=['GET'])
def get_account():
    """
    Get the connected personal LinkedIn account details for the current user.
    """
    user_id = _get_user_id_from_auth()
    supabase = get_supabase_client()
    if not supabase:
        return jsonify({"connected": False, "message": "Database unavailable"}), 200

    try:
        query = supabase.table("linkedin_accounts").select(
            "id, account_type, linkedin_urn, account_name, profile_picture_url, expires_at, updated_at"
        )
        if user_id:
            query = query.eq("user_id", user_id)

        res = query.order("updated_at", desc=True).limit(1).execute()
        if res.data and len(res.data) > 0:
            row = res.data[0]
            # Check expiration
            expires_at = row.get("expires_at")
            is_expired = False
            if expires_at:
                try:
                    exp_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                    if exp_dt < datetime.now(exp_dt.tzinfo):
                        is_expired = True
                except Exception:
                    pass

            return jsonify({
                "connected": not is_expired,
                "is_expired": is_expired,
                "account": {
                    "id": row["id"],
                    "account_type": row["account_type"],
                    "linkedin_urn": row["linkedin_urn"],
                    "account_name": row["account_name"],
                    "profile_picture_url": row.get("profile_picture_url"),
                    "expires_at": expires_at,
                }
            })

        return jsonify({"connected": False, "account": None})
    except Exception as err:
        logger.warning(f"Failed to query linkedin_accounts: {err}")
        return jsonify({"connected": False, "account": None, "warning": str(err)})


@linkedin_bp.route('/account', methods=['DELETE'])
def disconnect_account():
    """
    Disconnect the active LinkedIn account.
    """
    user_id = _get_user_id_from_auth()
    supabase = get_supabase_client()
    if not supabase:
        return jsonify({"error": "database_error"}), 500

    try:
        query = supabase.table("linkedin_accounts").delete()
        if user_id:
            query = query.eq("user_id", user_id)
        else:
            query = query.neq("id", "00000000-0000-0000-0000-000000000000")

        query.execute()
        return jsonify({"success": True, "message": "LinkedIn account disconnected"})
    except Exception as e:
        logger.error(f"Error disconnecting LinkedIn account: {e}", exc_info=True)
        return jsonify({"error": "server_error", "message": str(e)}), 500


@linkedin_bp.route('/publish', methods=['POST'])
def publish():
    """
    Publish a post to the connected personal LinkedIn feed.

    Expected JSON body:
    {
        "article_id": "optional-uuid",
        "commentary": "Post text content...",
        "image_url": "https://... (optional)",
        "image_alt_text": "Alt description (optional)",
        "article_url": "https://... (optional)",
        "article_title": "Article title (optional)",
        "article_description": "Article summary (optional)"
    }
    """
    data = request.get_json() or {}
    commentary = str(data.get("commentary") or "").strip()
    if not commentary:
        return jsonify({"error": "validation_error", "message": "Commentary is required to publish a post."}), 400

    user_id = _get_user_id_from_auth()
    supabase = get_supabase_client()
    if not supabase:
        return jsonify({"error": "database_unavailable"}), 500

    # Retrieve valid token and author URN
    try:
        query = supabase.table("linkedin_accounts").select(
            "id, access_token, linkedin_urn, expires_at"
        )
        if user_id:
            query = query.eq("user_id", user_id)
        account_res = query.order("updated_at", desc=True).limit(1).execute()

        if not account_res.data or len(account_res.data) == 0:
            return jsonify({
                "error": "not_connected",
                "message": "No connected LinkedIn account found. Please connect your LinkedIn account in Settings first."
            }), 400

        account = account_res.data[0]
        access_token = account.get("access_token")
        author_urn = account.get("linkedin_urn")

        if not access_token or not author_urn:
            return jsonify({"error": "invalid_credentials", "message": "Stored LinkedIn credentials are missing."}), 400

        # Execute publish via service
        service = _get_linkedin_service_for_user(user_id)
        publish_result = service.publish_post(
            access_token=access_token,
            author_urn=author_urn,
            commentary=commentary,
            image_url=data.get("image_url"),
            image_alt_text=data.get("image_alt_text"),
            article_url=data.get("article_url"),
            article_title=data.get("article_title"),
            article_description=data.get("article_description"),
        )

        article_id = data.get("article_id")
        if article_id:
            # Update Titles record with LinkedIn post metadata
            try:
                update_fields = {
                    "last_linkedin_post_urn": publish_result.get("post_urn"),
                    "last_linkedin_post_url": publish_result.get("post_url"),
                    "last_linkedin_status": "published",
                    "last_linkedin_published_at": publish_result.get("published_at"),
                    "linkedin_post_content": commentary,
                }
                supabase.table("Titles").update(update_fields).eq("id", article_id).execute()
            except Exception as update_err:
                logger.warning(f"Could not update Titles record with LinkedIn metadata: {update_err}")

        return jsonify({
            "success": True,
            "message": "Post successfully published to LinkedIn!",
            "post_urn": publish_result.get("post_urn"),
            "post_url": publish_result.get("post_url"),
            "published_at": publish_result.get("published_at"),
        })

    except Exception as e:
        logger.error(f"Error publishing to LinkedIn: {e}", exc_info=True)
        return jsonify({"error": "publish_failed", "message": str(e)}), 500


@linkedin_bp.route('/repurpose', methods=['POST'])
def repurpose():
    """
    Repurpose an article into a viral, platform-optimized LinkedIn post using the LLM.

    Expected JSON body:
    {
        "title": "Article Title",
        "content": "Article HTML or plain text",
        "tone": "thought_leadership" (optional)
    }
    """
    data = request.get_json() or {}
    title = str(data.get("title") or "").strip()
    content = str(data.get("content") or "").strip()
    tone = str(data.get("tone") or "thought_leadership").strip()

    if not title or not content:
        return jsonify({"error": "validation_error", "message": "Both title and content are required."}), 400

    try:
        service = _get_linkedin_service_for_user()
        repurposed = service.repurpose_article_for_linkedin(
            article_title=title,
            article_content=content,
            tone=tone,
        )
        return jsonify({
            "success": True,
            "repurposed": repurposed
        })
    except Exception as e:
        logger.error(f"Error repurposing article for LinkedIn: {e}", exc_info=True)
        return jsonify({"error": "repurpose_failed", "message": str(e)}), 500
