"""
LinkedIn Service for Content Generator V2.

Handles:
1. LinkedIn OAuth 2.0 (authorization URL, token exchange, profile retrieval).
2. Image uploads to LinkedIn Media.
3. Publishing posts & article shares to personal LinkedIn feeds via LinkedIn REST API.
4. AI repurposing of articles into high-converting LinkedIn thought-leadership posts.
"""

from __future__ import annotations

import os
import re
import json
import logging
import urllib.parse
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

LINKEDIN_AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
LINKEDIN_TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
LINKEDIN_USERINFO_URL = "https://api.linkedin.com/v2/userinfo"
LINKEDIN_REST_API_BASE = "https://api.linkedin.com/rest"
LINKEDIN_API_VERSION = "202401"
LINKEDIN_DEFAULT_SCOPES = "openid profile email w_member_social"


class LinkedInService:
    """Service to handle LinkedIn OAuth, publishing, and content repurposing."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
    ):
        self.client_id = client_id or os.getenv("LINKEDIN_CLIENT_ID", "").strip()
        self.client_secret = client_secret or os.getenv("LINKEDIN_CLIENT_SECRET", "").strip()
        self.redirect_uri = redirect_uri or os.getenv(
            "LINKEDIN_REDIRECT_URI",
            "http://localhost:5001/api/linkedin/callback",
        ).strip()

    def get_authorization_url(
        self,
        state: str = "linkedin_oauth_state",
        scopes: Optional[str] = None,
        include_org: bool = False,
    ) -> str:
        """
        Generate LinkedIn OAuth 2.0 authorization URL.
        Uses OpenID Connect and member social posting scopes,
        and optionally organization social scope if Community Management API is enabled.
        """
        if not self.client_id:
            raise ValueError("LINKEDIN_CLIENT_ID is not configured in environment or settings.")

        target_scopes = scopes
        if not target_scopes:
            target_scopes = LINKEDIN_DEFAULT_SCOPES
            if include_org:
                target_scopes = f"{target_scopes} w_organization_social"

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "state": state,
            "scope": target_scopes,
        }
        return f"{LINKEDIN_AUTH_URL}?{urllib.parse.urlencode(params)}"

    def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for an OAuth access token.
        """
        if not self.client_id or not self.client_secret:
            raise ValueError("LINKEDIN_CLIENT_ID or LINKEDIN_CLIENT_SECRET is missing.")

        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = requests.post(LINKEDIN_TOKEN_URL, data=payload, headers=headers, timeout=30)
        if not response.ok:
            logger.error(f"LinkedIn token exchange failed: {response.status_code} {response.text}")
            raise RuntimeError(f"LinkedIn authentication failed: {response.text}")

        token_data = response.json()
        expires_in = token_data.get("expires_in", 5184000)  # default 60 days
        token_data["expires_at"] = (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat()
        return token_data

    def get_member_profile(self, access_token: str) -> Dict[str, Any]:
        """
        Fetch basic profile info via OpenID userinfo endpoint.
        Returns author URN (urn:li:person:<sub_id>), name, and picture.
        """
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(LINKEDIN_USERINFO_URL, headers=headers, timeout=20)
        if not response.ok:
            logger.error(f"LinkedIn profile fetch failed: {response.status_code} {response.text}")
            raise RuntimeError(f"Failed to fetch LinkedIn profile: {response.text}")

        data = response.json()
        person_sub = data.get("sub", "")
        if not person_sub:
            raise ValueError("No 'sub' user identifier found in LinkedIn userinfo response.")

        return {
            "urn": f"urn:li:person:{person_sub}",
            "sub": person_sub,
            "name": data.get("name") or f"{data.get('given_name', '')} {data.get('family_name', '')}".strip(),
            "email": data.get("email"),
            "picture": data.get("picture"),
        }

    def upload_image(self, access_token: str, author_urn: str, image_url: str) -> str:
        """
        Upload an image to LinkedIn using the modern REST Images API.
        Returns the image URN (e.g. 'urn:li:image:...').
        """
        headers = {
            "Authorization": f"Bearer {access_token}",
            "LinkedIn-Version": LINKEDIN_API_VERSION,
            "X-Restli-Protocol-Version": "2.0.0",
            "Content-Type": "application/json",
        }

        # Step 1: Initialize image upload
        init_payload = {
            "initializeUploadRequest": {
                "owner": author_urn,
            }
        }
        init_res = requests.post(
            f"{LINKEDIN_REST_API_BASE}/images?action=initializeUpload",
            headers=headers,
            json=init_payload,
            timeout=30,
        )
        if not init_res.ok:
            logger.error(f"Failed to initialize LinkedIn image upload: {init_res.status_code} {init_res.text}")
            raise RuntimeError(f"LinkedIn image upload initialization failed: {init_res.text}")

        init_data = init_res.json().get("value", {})
        upload_url = init_data.get("uploadUrl")
        image_urn = init_data.get("image")

        if not upload_url or not image_urn:
            raise RuntimeError("LinkedIn initializeUpload response missing uploadUrl or image URN")

        # Step 2: Fetch the source image
        img_res = requests.get(image_url, timeout=30)
        if not img_res.ok:
            raise RuntimeError(f"Failed to download image from {image_url}")

        content_type = img_res.headers.get("Content-Type", "image/jpeg")

        # Step 3: PUT raw bytes to LinkedIn uploadUrl
        put_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": content_type,
        }
        put_res = requests.put(upload_url, data=img_res.content, headers=put_headers, timeout=60)
        if not put_res.ok:
            logger.error(f"Failed to stream image to LinkedIn: {put_res.status_code} {put_res.text}")
            raise RuntimeError(f"Failed to upload image bytes to LinkedIn: {put_res.text}")

        logger.info(f"Successfully uploaded image to LinkedIn with URN: {image_urn}")
        return image_urn

    def publish_post(
        self,
        access_token: str,
        author_urn: str,
        commentary: str,
        image_url: Optional[str] = None,
        image_alt_text: Optional[str] = None,
        article_url: Optional[str] = None,
        article_title: Optional[str] = None,
        article_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Publish a post to the personal LinkedIn feed via POST /rest/posts.
        Supports:
        - Standalone commentary (thought-leadership post / micro-article).
        - Commentary + Image attachment.
        - Commentary + Article link preview card.
        """
        headers = {
            "Authorization": f"Bearer {access_token}",
            "LinkedIn-Version": LINKEDIN_API_VERSION,
            "X-Restli-Protocol-Version": "2.0.0",
            "Content-Type": "application/json",
        }

        clean_commentary = commentary.strip()
        if len(clean_commentary) > 3000:
            logger.warning(
                "Commentary exceeds 3000 characters (%d chars). Truncating for LinkedIn API limit.",
                len(clean_commentary),
            )
            clean_commentary = clean_commentary[:2990].rstrip() + "\n..."

        payload: Dict[str, Any] = {
            "author": author_urn,
            "commentary": clean_commentary,
            "visibility": "PUBLIC",
            "distribution": {
                "feedDistribution": "MAIN_FEED",
                "targetEntities": [],
                "thirdPartyDistributionChannels": [],
            },
            "lifecycleState": "PUBLISHED",
        }

        # Case A: Article link preview
        if article_url:
            article_content: Dict[str, Any] = {
                "source": article_url,
                "title": article_title or "Read Article",
                "description": article_description or "",
            }
            # If an image is also provided, we can upload and attach as thumbnail
            if image_url:
                try:
                    img_urn = self.upload_image(access_token, author_urn, image_url)
                    article_content["thumbnail"] = img_urn
                except Exception as img_err:
                    logger.warning(f"Could not attach thumbnail to article link preview: {img_err}")

            payload["content"] = {"article": article_content}

        # Case B: Image post (no article link)
        elif image_url:
            try:
                img_urn = self.upload_image(access_token, author_urn, image_url)
                payload["content"] = {
                    "media": {
                        "id": img_urn,
                        "altText": image_alt_text or "Post visual",
                    }
                }
            except Exception as img_err:
                logger.error(f"Image upload failed before posting to LinkedIn: {img_err}")
                raise

        logger.info(f"Submitting post to LinkedIn REST API for author {author_urn}...")
        response = requests.post(
            f"{LINKEDIN_REST_API_BASE}/posts",
            headers=headers,
            json=payload,
            timeout=30,
        )

        if not response.ok:
            logger.error(f"LinkedIn post creation failed: {response.status_code} {response.text}")
            err_msg = response.text
            if response.status_code == 403 and "organization" in author_urn:
                err_msg = (
                    "LinkedIn returned 403 Forbidden. Publishing to a Company Page requires the "
                    "'Community Management API' product enabled in your LinkedIn Developer App with the "
                    "'w_organization_social' scope, and your personal account must be an Admin of that page."
                )
            raise RuntimeError(f"LinkedIn publish error: {err_msg}")

        # The post URN is returned in the 'x-restli-id' response header
        post_urn = response.headers.get("x-restli-id") or ""
        if not post_urn and response.text:
            try:
                post_urn = response.json().get("id") or ""
            except Exception:
                pass

        live_url = f"https://www.linkedin.com/feed/update/{post_urn}/" if post_urn else "https://www.linkedin.com/feed/"

        return {
            "success": True,
            "post_urn": post_urn,
            "post_url": live_url,
            "published_at": datetime.utcnow().isoformat(),
        }

    def repurpose_article_for_linkedin(
        self,
        article_title: str,
        article_content: str,
        tone: str = "thought_leadership",
    ) -> Dict[str, Any]:
        """
        Use the default LLM to repurpose a full-length article into an engaging,
        high-converting LinkedIn post (under 3,000 characters).
        """
        from supabase_client import get_default_llm_provider
        from llm_client import create_llm_client

        provider, model, api_key = get_default_llm_provider()
        if not provider or not model or not api_key:
            raise RuntimeError("No default LLM provider configured for LinkedIn repurposing.")

        client = create_llm_client(provider=provider, model=model, api_key=api_key)

        # Strip HTML tags from article content for clean prompt context
        clean_text = re.sub(r"<[^>]+>", " ", article_content)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        context_slice = clean_text[:4000]

        system_prompt = """You are a seasoned industry practitioner and thoughtful professional writer on LinkedIn.
Your task is to transform a long-form article into a friendly, insightful LinkedIn post that sparks genuine peer-to-peer discussion among professionals.

CORE VOICE & POSTURE:
1. HUMBLE & COLLABORATIVE (NEVER DOGMATIC):
   - Never sound like an "owner of the truth", all-knowing guru, or preachy lecturer.
   - STRICTLY FORBIDDEN: Preachy phrases like "Most leadership teams fail to...", "You are doing it wrong...", "Here is why everyone misses the point...", "Stop blaming X...".
   - Write from the perspective of an experienced peer sharing real-world field observations and nuanced operational realities:
     e.g., "Teams often misdiagnose...", "Standard metrics rarely catch these in real time...", "Traditional reporting can easily obscure...", "A common blind spot across networks...".

2. CLEAN, LINE-SEPARATED TAKEAWAYS:
   - When listing causes, leaks, symptoms, or takeaways:
     DO NOT cram multiple points into a single long sentence connected with commas.
     Put each point on its own clear line or bullet point with generous spacing.

3. AUTHENTIC PEER DISCUSSION (ZERO ENGAGEMENT-BAIT CLICHÉS):
   - STRICTLY FORBIDDEN: Clichés like "What's your take on this?", "Drop your thoughts below 👇", "Comment below", "Agree or disagree?", "Thoughts?", "Hit like".
   - Conclude with 1 or 2 specific, curious peer-level questions that invite colleagues to share real-world experiences from their own organizations.

4. HASHTAGS:
   - 0 to 2 clean, highly relevant industry tags at most (e.g., #SupplyChain #Operations). Never include generic engagement tags.

5. LENGTH & PACING:
   - Total post should be concise: 800 to 1,800 characters (strictly below LinkedIn's 3,000 character limit).
   - Generous line breaks. Paragraphs should be 1-2 sentences maximum.

STYLE REFERENCE (AIM FOR THIS TONE AND STRUCTURE):
Three supply chain leaks quietly drain revenue across multi-echelon networks—and standard ERP metrics rarely catch them in real time:

• Winner-take-all root cause analysis that masks simultaneous upstream supply deficits.
• Overnight SAP rejection code overwrites that silently corrupt EDI and credit data.
• Reactive alerts that trigger only after orders have already drained distribution centers.

A single missing promotional SKU can drop a respectable 97% Case Fill Rate to a 0% Order Fill Rate overnight, triggering heavy OTIF penalties. Yet teams often misdiagnose these cuts because traditional reporting hides factory bottlenecks that formed 2 to 8 weeks upstream.

Curious how others are tackling this in your networks: Are you seeing similar blind spots between factory scheduling and retail compliance? How do you approach untangling multi-cause fulfillment failures?

Return your answer strictly in valid JSON with these keys:
{
  "hook": "The opening statement introducing the challenge or topic",
  "body": "The body with clean bullet points, context, and generous spacing",
  "cta": "The thoughtful, peer-to-peer discussion question (no emojis like 👇, no generic engagement bait)",
  "hashtags": ["#Tag1", "#Tag2"],
  "full_post": "The complete post ready to publish (hook + body + cta + hashtags joined with clean newlines)"
}"""

        user_prompt = f"""Article Title: {article_title}
Tone: {tone}

Article Excerpt:
{context_slice}

Please craft the high-performing LinkedIn post now."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = client.generate(messages=messages)
        content = response.content.strip()

        # Parse JSON response
        try:
            # Clean markdown fences if present
            if "```" in content:
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
                if match:
                    content = match.group(1).strip()
            result = json.loads(content)
        except Exception as parse_err:
            logger.warning(f"Failed to parse LLM JSON for LinkedIn repurpose: {parse_err}. Using raw text.")
            result = {
                "hook": article_title,
                "body": content,
                "cta": "Curious how others in your networks are approaching this challenge.",
                "hashtags": [],
                "full_post": content,
            }

        return result


linkedin_service = LinkedInService()
