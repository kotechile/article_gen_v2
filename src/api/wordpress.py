from __future__ import annotations

from flask import Blueprint, jsonify, request
from supabase_client import get_supabase_client
from src.utils.wordpress_client import WordPressClient
import logging
import re
import json
import os
import html
import requests
from datetime import datetime

wordpress_bp = Blueprint('wordpress', __name__)
logger = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return re.sub(r"(^-|-$)", "", value) or "category"


def _fallback_category_description(
    category_name: str,
    parent_name: str | None = None,
    site_domain: str | None = None,
) -> str:
    if parent_name:
        return f"Articles and guides about {category_name} under {parent_name} for {site_domain or 'this website'}."
    return f"Articles and guides about {category_name} for {site_domain or 'this website'}."


def _shorten_wp_title(raw_name: str, max_chars: int = 60) -> str:
    """
    Keep titles SEO-friendly for WordPress category names with a hard max length.
    Preserves whole words when possible.
    """
    name = re.sub(r"\s+", " ", (raw_name or "").strip())
    if len(name) <= max_chars:
        return name

    clipped = name[:max_chars].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0].rstrip()

    # If word-boundary clipping became too short, fallback to strict truncation.
    if len(clipped) < max(20, max_chars // 2):
        clipped = name[:max_chars].rstrip()
    return clipped or name[:max_chars]


def _generate_category_descriptions(
    domain: str,
    project_name: str | None,
    categories: list[dict],
) -> dict[str, str]:
    """
    Generate category descriptions with the default LLM.
    Falls back to deterministic template descriptions on any failure.
    """
    by_id = {str(c.get("id")): c for c in categories}
    parent_name_by_id = {}
    for c in categories:
        cid = str(c.get("id"))
        pid = str(c.get("parent_category_id") or "")
        if pid and pid in by_id:
            parent_name_by_id[cid] = (by_id[pid].get("name") or "").strip()

    fallback_map: dict[str, str] = {}
    for c in categories:
        cid = str(c.get("id"))
        manual_description = str(c.get("description") or "").strip()
        if manual_description:
            fallback_map[cid] = manual_description
        else:
            fallback_map[cid] = _fallback_category_description(
                (c.get("name") or "").strip(),
                parent_name_by_id.get(cid),
                domain,
            )

    # Only generate with LLM for entries that do not have a manual description.
    categories_missing_description = [
        c for c in categories if not str(c.get("description") or "").strip()
    ]
    if not categories_missing_description:
        return fallback_map

    try:
        from supabase_client import get_default_llm_provider as _get_default_llm_provider
    except Exception:
        logger.warning("Default LLM provider helper unavailable, using fallback descriptions.")
        return fallback_map

    provider, model, api_key = _get_default_llm_provider()
    if not provider or not model or not api_key:
        logger.info("No default LLM configured; using fallback category descriptions.")
        return fallback_map

    try:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from llm_client_direct import create_llm_client

        items = []
        for c in categories_missing_description:
            cid = str(c.get("id"))
            items.append({
                "id": cid,
                "name": (c.get("name") or "").strip(),
                "level": int(c.get("level") or 1),
                "parent_name": parent_name_by_id.get(cid),
            })

        system_prompt = (
            "You write concise WordPress category descriptions for SEO and readers. "
            "Return only JSON."
        )
        user_prompt = (
            "Create one plain-text description per category.\n"
            f"Site domain: {domain}\n"
            f"Project: {project_name or domain}\n"
            "Rules:\n"
            "- 1 sentence, 90-160 characters.\n"
            "- No markdown, no quotes, no hype.\n"
            "- Mention the category topic naturally.\n"
            "- For subcategories, reflect the parent context.\n\n"
            f"Categories JSON:\n{json.dumps(items, ensure_ascii=True)}\n\n"
            "Return ONLY a JSON array with this exact shape:\n"
            "[{\"id\":\"<id>\",\"description\":\"<text>\"}]"
        )

        llm = create_llm_client(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=0.3,
            max_tokens=1500,
            timeout=45,
            max_retries=0,
        )
        raw = llm.generate([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]).content

        cleaned = (raw or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            return fallback_map

        output = dict(fallback_map)
        for row in parsed:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("id") or "").strip()
            desc = str(row.get("description") or "").strip()
            if cid in output and desc:
                output[cid] = desc
        return output
    except Exception as e:
        logger.warning("Category description LLM generation failed, using fallback: %s", str(e))
        return fallback_map

def _clean_html_to_plain_text(html_text: str) -> str:
    """Strip HTML tags, unescape HTML entities, and collapse whitespace."""
    if not html_text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(html_text))
    text = html.unescape(text)
    text = re.sub(r"\s+([,.:;!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_wordpress_seo_metadata(post: dict, domain: str = "") -> dict:
    """
    Extract comprehensive SEO metadata, social cards, taxonomies, and rich fields
    from a WordPress REST API post response (Yoast SEO, RankMath SEO, meta, _embedded).
    """
    title_obj = post.get("title") or {}
    title_rendered = title_obj.get("rendered", "") if isinstance(title_obj, dict) else str(title_obj or "")
    plain_title = _clean_html_to_plain_text(title_rendered) or str(post.get("title") or "")
    
    excerpt_obj = post.get("excerpt") or {}
    excerpt_rendered = excerpt_obj.get("rendered", "") if isinstance(excerpt_obj, dict) else str(excerpt_obj or "")
    plain_excerpt = _clean_html_to_plain_text(excerpt_rendered)
    
    content_obj = post.get("content") or {}
    content_rendered = content_obj.get("rendered", "") if isinstance(content_obj, dict) else str(content_obj or "")
    
    slug = str(post.get("slug") or "").strip()
    link = str(post.get("link") or "").strip()
    if link:
        link = link.replace("://cms.", "://")
        link = re.sub(r"/(\d{4}/\d{2}/\d{2}/|\d{4}/\d{2}/)", "/", link)
    
    meta = post.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
        
    yoast_head = post.get("yoast_head_json") or {}
    if not isinstance(yoast_head, dict):
        yoast_head = {}
        
    rank_math = post.get("rank_math_seo") or {}
    if not isinstance(rank_math, dict):
        rank_math = {}
        
    embedded = post.get("_embedded") or {}
    if not isinstance(embedded, dict):
        embedded = {}
        
    # 1. Featured Image extraction
    featured_image_url = ""
    featured_image_alt = ""
    featured_image_title = ""
    featured_image_caption = ""
    
    featured_media_list = embedded.get("wp:featuredmedia") or []
    if isinstance(featured_media_list, list) and featured_media_list and isinstance(featured_media_list[0], dict):
        fm = featured_media_list[0]
        featured_image_url = fm.get("source_url") or ""
        featured_image_alt = fm.get("alt_text") or ""
        fm_title = fm.get("title") or {}
        featured_image_title = fm_title.get("rendered", "") if isinstance(fm_title, dict) else str(fm_title or "")
        fm_caption = fm.get("caption") or {}
        featured_image_caption = fm_caption.get("rendered", "") if isinstance(fm_caption, dict) else str(fm_caption or "")

    if not featured_image_url:
        og_images = yoast_head.get("og_image") or []
        if isinstance(og_images, list) and og_images and isinstance(og_images[0], dict):
            featured_image_url = og_images[0].get("url") or ""
        elif isinstance(og_images, str):
            featured_image_url = og_images
            
    if not featured_image_url:
        featured_image_url = (
            meta.get("rank_math_facebook_image")
            or meta.get("_yoast_wpseo_opengraph-image")
            or meta.get("seo_og_image")
            or ""
        )

    # 2. Categories & Tags extraction
    category_ids = post.get("categories") or []
    if not isinstance(category_ids, list):
        category_ids = [category_ids] if category_ids else []
    
    tag_ids = post.get("tags") or []
    if not isinstance(tag_ids, list):
        tag_ids = [tag_ids] if tag_ids else []
        
    category_names = []
    tag_names = []
    
    terms_list = embedded.get("wp:term") or []
    if isinstance(terms_list, list):
        for term_group in terms_list:
            if not isinstance(term_group, list):
                continue
            for term in term_group:
                if not isinstance(term, dict):
                    continue
                taxonomy = term.get("taxonomy")
                term_name = str(term.get("name") or "").strip()
                if taxonomy == "category" and term_name and term_name not in category_names:
                    category_names.append(term_name)
                elif taxonomy == "post_tag" and term_name and term_name not in tag_names:
                    tag_names.append(term_name)

    # 3. Focus / Primary Keyword extraction
    raw_focus = (
        meta.get("_yoast_wpseo_focuskw")
        or meta.get("rank_math_focus_keyword")
        or meta.get("seo_focus_keyword")
        or meta.get("focus_keyword")
        or rank_math.get("focus_keyword")
        or ""
    )
    
    focus_keyword = ""
    extra_focus_secondaries = []
    
    if isinstance(raw_focus, str) and raw_focus.strip():
        # Handle comma-separated RankMath keywords
        parts = [p.strip() for p in raw_focus.split(",") if p.strip()]
        if parts:
            focus_keyword = parts[0]
            extra_focus_secondaries = parts[1:]
    elif isinstance(raw_focus, list) and raw_focus:
        focus_keyword = str(raw_focus[0]).strip()
        extra_focus_secondaries = [str(p).strip() for p in raw_focus[1:] if str(p).strip()]
        
    if not focus_keyword and tag_names:
        focus_keyword = tag_names[0]
    if not focus_keyword and category_names:
        focus_keyword = category_names[0]
    if not focus_keyword and plain_title:
        words = [w for w in re.split(r"\s+", plain_title) if len(w) > 2]
        if words:
            focus_keyword = " ".join(words[:3]).lower()

    # 4. Secondary Keywords
    secondary_keywords = []
    for extra in extra_focus_secondaries:
        if extra and extra.lower() != focus_keyword.lower() and extra not in secondary_keywords:
            secondary_keywords.append(extra)
            
    # Check Yoast Premium secondary keywords (_yoast_wpseo_focuskeywords JSON)
    yoast_secondaries_raw = meta.get("_yoast_wpseo_focuskeywords")
    if yoast_secondaries_raw:
        try:
            parsed_yoast = json.loads(yoast_secondaries_raw) if isinstance(yoast_secondaries_raw, str) else yoast_secondaries_raw
            if isinstance(parsed_yoast, list):
                for item in parsed_yoast:
                    kw = item.get("keyword") if isinstance(item, dict) else str(item)
                    kw = str(kw or "").strip()
                    if kw and kw.lower() != focus_keyword.lower() and kw not in secondary_keywords:
                        secondary_keywords.append(kw)
        except Exception:
            pass

    raw_secondary = meta.get("seo_secondary_keywords") or meta.get("secondary_keywords")
    if isinstance(raw_secondary, str) and raw_secondary.strip():
        for k in raw_secondary.split(","):
            k = k.strip()
            if k and k.lower() != focus_keyword.lower() and k not in secondary_keywords:
                secondary_keywords.append(k)
    elif isinstance(raw_secondary, list):
        for k in raw_secondary:
            k = str(k).strip()
            if k and k.lower() != focus_keyword.lower() and k not in secondary_keywords:
                secondary_keywords.append(k)
        
    for tag in tag_names:
        if tag and tag.lower() != focus_keyword.lower() and tag not in secondary_keywords:
            secondary_keywords.append(tag)

    # 5. SEO Title & Description
    seo_title = str(
        yoast_head.get("title")
        or meta.get("_yoast_wpseo_title")
        or meta.get("rank_math_title")
        or meta.get("seo_meta_title")
        or rank_math.get("title")
        or plain_title
    ).strip()
    
    seo_description = str(
        yoast_head.get("description")
        or meta.get("_yoast_wpseo_metadesc")
        or meta.get("rank_math_description")
        or meta.get("seo_meta_description")
        or rank_math.get("description")
        or plain_excerpt
    ).strip()

    # 6. Canonical URL
    canonical_url = str(
        yoast_head.get("canonical")
        or meta.get("_yoast_wpseo_canonical")
        or meta.get("rank_math_canonical_url")
        or meta.get("seo_canonical_url")
        or rank_math.get("canonical_url")
        or link
    ).strip()

    # 7. Social & OpenGraph
    og_title = str(
        yoast_head.get("og_title")
        or meta.get("_yoast_wpseo_opengraph-title")
        or meta.get("rank_math_facebook_title")
        or seo_title
    )
    og_description = str(
        yoast_head.get("og_description")
        or meta.get("_yoast_wpseo_opengraph-description")
        or meta.get("rank_math_facebook_description")
        or seo_description
    )
    og_image = featured_image_url
    og_type = str(yoast_head.get("og_type") or meta.get("seo_og_type") or "article")
    
    twitter_title = str(
        yoast_head.get("twitter_title")
        or meta.get("_yoast_wpseo_twitter-title")
        or meta.get("rank_math_twitter_title")
        or og_title
    )
    twitter_description = str(
        yoast_head.get("twitter_description")
        or meta.get("_yoast_wpseo_twitter-description")
        or meta.get("rank_math_twitter_description")
        or og_description
    )
    twitter_image = str(
        yoast_head.get("twitter_image")
        or meta.get("_yoast_wpseo_twitter-image")
        or meta.get("rank_math_twitter_image")
        or featured_image_url
    )
    twitter_card = str(
        yoast_head.get("twitter_card")
        or meta.get("_yoast_wpseo_twitter-card")
        or meta.get("rank_math_twitter_card_type")
        or "summary_large_image"
    )

    # 8. Robots & Schema
    robots_meta = "index,follow"
    if yoast_head.get("robots"):
        rob = yoast_head.get("robots")
        if isinstance(rob, dict):
            parts = []
            if rob.get("index") == "noindex":
                parts.append("noindex")
            else:
                parts.append("index")
            if rob.get("follow") == "nofollow":
                parts.append("nofollow")
            else:
                parts.append("follow")
            robots_meta = ",".join(parts)
        elif isinstance(rob, str):
            robots_meta = rob
    elif meta.get("rank_math_robots"):
        rm_rob = meta.get("rank_math_robots")
        robots_meta = ",".join(rm_rob) if isinstance(rm_rob, list) else str(rm_rob)
    elif str(meta.get("_yoast_wpseo_meta_robots_noindex")) == "1":
        robots_meta = "noindex,follow"

    schema_type = str(meta.get("seo_schema_type") or "Article")
    
    # 9. SEO & Readability Scores
    readability_score = None
    try:
        if meta.get("seo_readability_score"):
            readability_score = float(meta.get("seo_readability_score"))
        elif meta.get("_yoast_wpseo_content_score"):
            readability_score = float(meta.get("_yoast_wpseo_content_score"))
        elif yoast_head.get("readability_score"):
            readability_score = float(yoast_head.get("readability_score"))
    except Exception:
        pass
    if readability_score is None:
        readability_score = 75.0 if len(content_rendered) > 500 else 60.0

    seo_optimization_score = None
    try:
        if meta.get("seo_optimization_score"):
            seo_optimization_score = float(meta.get("seo_optimization_score"))
        elif meta.get("_yoast_wpseo_linkdex"):
            seo_optimization_score = float(meta.get("_yoast_wpseo_linkdex"))
        elif rank_math.get("score"):
            seo_optimization_score = float(rank_math.get("score"))
        elif meta.get("rank_math_seo_score"):
            seo_optimization_score = float(meta.get("rank_math_seo_score"))
    except Exception:
        pass
    if seo_optimization_score is None:
        seo_optimization_score = 70.0
        if focus_keyword and seo_title and seo_description:
            seo_optimization_score = 85.0
            if len(content_rendered) > 1000:
                seo_optimization_score = 90.0
        elif focus_keyword or (seo_title and seo_description):
            seo_optimization_score = 75.0

    structured_seo_metadata = {
        "focusKeyword": focus_keyword,
        "primaryKeywords": [focus_keyword] if focus_keyword else [],
        "secondaryKeywords": secondary_keywords,
        "metaTitle": seo_title,
        "metaDescription": seo_description,
        "canonicalUrl": canonical_url,
        "robotsMeta": robots_meta,
        "schemaType": schema_type,
        "readabilityScore": readability_score,
        "ogTitle": og_title,
        "ogDescription": og_description,
        "ogImageUrl": og_image,
        "ogType": og_type,
        "twitterTitle": twitter_title,
        "twitterDescription": twitter_description,
        "twitterImageUrl": twitter_image,
        "twitterCardType": twitter_card,
        "categoryNames": category_names,
        "tagNames": tag_names,
        "featuredImageUrl": featured_image_url,
        "featuredImageAlt": featured_image_alt,
    }

    return {
        "title": plain_title or title_rendered,
        "slug": slug,
        "content_html": content_rendered,
        "excerpt": plain_excerpt or excerpt_rendered,
        "link": link,
        "published_at": post.get("date_gmt") or post.get("date"),
        "modified_at": post.get("modified_gmt") or post.get("modified"),
        "post_id": post.get("id"),
        "featured_image_url": featured_image_url,
        "featured_image_alt": featured_image_alt,
        "featured_image_title": featured_image_title,
        "featured_image_caption": featured_image_caption,
        "category_ids": category_ids,
        "category_names": category_names,
        "tag_ids": tag_ids,
        "tag_names": tag_names,
        "seo_title": seo_title,
        "seo_description": seo_description,
        "focus_keyword": focus_keyword,
        "primary_keyword": focus_keyword,
        "secondary_keywords": secondary_keywords,
        "canonical_url": canonical_url,
        "robots_meta": robots_meta,
        "schema_type": schema_type,
        "readability_score": readability_score,
        "seo_optimization_score": seo_optimization_score,
        "seo_metadata": structured_seo_metadata,
        "raw_post_json": post,
    }


def _build_titles_payload_from_imported_post(user_id: str, site_id: Any, domain: str, extracted: dict) -> dict:
    """Build a Titles table payload with all fields needed by Content Studio and Article Editor."""
    from uuid import uuid4
    now_iso = datetime.utcnow().isoformat()
    focus_kw = extracted.get("focus_keyword") or extracted.get("primary_keyword") or ""
    secondary_kws = extracted.get("secondary_keywords") or []
    all_kws = [focus_kw] if focus_kw else []
    for k in secondary_kws:
        if k and k not in all_kws:
            all_kws.append(k)
            
    cat_id = extracted.get("category_ids", [None])[0] if extracted.get("category_ids") else None
    cat_name = extracted.get("category_names", ["Uncategorized"])[0] if extracted.get("category_names") else "Uncategorized"
    
    selected_metrics = {
        "primary": {
            "keyword": focus_kw,
            "search_volume": None,
            "keyword_difficulty": None,
            "metric_source": "editorial_factory_wp_import",
            "is_estimated": False,
        },
        "secondary": [
            {
                "keyword": skw,
                "search_volume": None,
                "keyword_difficulty": None,
                "metric_source": "editorial_factory_wp_import",
                "is_estimated": False,
            }
            for skw in secondary_kws
        ],
    } if (focus_kw or secondary_kws) else {}
    
    raw_seo_meta = extracted.get("seo_metadata") or {}
    if not isinstance(raw_seo_meta, dict):
        raw_seo_meta = {}
        
    seo_meta = {
        "meta_title": raw_seo_meta.get("metaTitle") or raw_seo_meta.get("meta_title") or extracted.get("seo_title") or extracted.get("title") or "",
        "metaTitle": raw_seo_meta.get("metaTitle") or raw_seo_meta.get("meta_title") or extracted.get("seo_title") or extracted.get("title") or "",
        "meta_description": raw_seo_meta.get("metaDescription") or raw_seo_meta.get("meta_description") or extracted.get("seo_description") or extracted.get("excerpt") or "",
        "metaDescription": raw_seo_meta.get("metaDescription") or raw_seo_meta.get("meta_description") or extracted.get("seo_description") or extracted.get("excerpt") or "",
        "focus_keyword": raw_seo_meta.get("focusKeyword") or raw_seo_meta.get("focus_keyword") or focus_kw,
        "focusKeyword": raw_seo_meta.get("focusKeyword") or raw_seo_meta.get("focus_keyword") or focus_kw,
        "primary_keywords": raw_seo_meta.get("primaryKeywords") or raw_seo_meta.get("primary_keywords") or ([focus_kw] if focus_kw else []),
        "primaryKeywords": raw_seo_meta.get("primaryKeywords") or raw_seo_meta.get("primary_keywords") or ([focus_kw] if focus_kw else []),
        "secondary_keywords": raw_seo_meta.get("secondaryKeywords") or raw_seo_meta.get("secondary_keywords") or secondary_kws,
        "secondaryKeywords": raw_seo_meta.get("secondaryKeywords") or raw_seo_meta.get("secondary_keywords") or secondary_kws,
        "canonical_url": raw_seo_meta.get("canonicalUrl") or raw_seo_meta.get("canonical_url") or extracted.get("canonical_url") or "",
        "canonicalUrl": raw_seo_meta.get("canonicalUrl") or raw_seo_meta.get("canonical_url") or extracted.get("canonical_url") or "",
        "robots_meta": raw_seo_meta.get("robotsMeta") or raw_seo_meta.get("robots_meta") or extracted.get("robots_meta") or "index,follow",
        "robotsMeta": raw_seo_meta.get("robotsMeta") or raw_seo_meta.get("robots_meta") or extracted.get("robots_meta") or "index,follow",
        "readability_score": raw_seo_meta.get("readabilityScore") or raw_seo_meta.get("readability_score") or extracted.get("readability_score") or 75.0,
        "readabilityScore": raw_seo_meta.get("readabilityScore") or raw_seo_meta.get("readability_score") or extracted.get("readability_score") or 75.0,
        "categoryNames": extracted.get("category_names") or [],
        "tagNames": extracted.get("tag_names") or [],
        "featuredImageUrl": extracted.get("featured_image_url") or "",
        "featuredImageAlt": extracted.get("featured_image_alt") or "",
    }

    sec_cat_name = extracted.get("category_names", [""])[1] if len(extracted.get("category_names", [])) > 1 else ""

    idea_meta = {
        "imported_from": "editorial_factory_wordpress",
        "wp_post_id": extracted.get("post_id"),
        "wp_site_id": site_id,
        "domain": domain,
        "slug": extracted.get("slug"),
        "canonical_url": extracted.get("canonical_url"),
        "seo_metadata": seo_meta,
        "category_context": {
            "primary": cat_name,
            "primary_category_name": cat_name,
            "secondary": sec_cat_name,
            "secondary_category_name": sec_cat_name,
            "category_path": " / ".join(extracted.get("category_names") or [cat_name]),
            "primary_id": cat_id,
        }
    }
    
    payload = {
        "id": str(uuid4()),
        "user_id": user_id,
        "Title": extracted.get("title") or "Untitled Article",
        "userDescription": extracted.get("seo_description") or extracted.get("excerpt") or "",
        "deck": extracted.get("seo_description") or extracted.get("excerpt") or "",
        "htmlArticle": extracted.get("content_html") or "",
        "Keywords": ", ".join(all_kws),
        "primary_keyword": focus_kw,
        "primary_keywords": [focus_kw] if focus_kw else [],
        "secondary_keywords": secondary_kws,
        "secondary_keywords_json": secondary_kws,
        "search_phrase": focus_kw or extracted.get("title"),
        "seo_optimization_score": extracted.get("seo_optimization_score") or extracted.get("seo_score") or 75.0,
        "readability_score": extracted.get("readability_score") or 75.0,
        "featuredImageUrl": extracted.get("featured_image_url") or None,
        "featuredImageURL": extracted.get("featured_image_url") or None,
        "featuredImageAuthor": domain,
        "ImageAuthor": domain,
        "mediaAltText": extracted.get("featured_image_alt") or None,
        "MediaAltText": extracted.get("featured_image_alt") or None,
        "mediaTitle": extracted.get("featured_image_title") or None,
        "mediaCaption": extracted.get("featured_image_caption") or None,
        "domain": domain,
        "wordpress_category_id": cat_id,
        "category": cat_name,
        "status": "WP Published",
        "published": True,
        "last_wp_site_id": str(site_id) if site_id is not None else None,
        "last_wp_post_status": "publish",
        "last_wp_category_id": str(cat_id) if cat_id is not None else None,
        "dateCreatedOn": extracted.get("published_at") or now_iso,
        "idea_metadata": idea_meta,
        "selected_keyword_metrics_json": selected_metrics,
        "keyword_selection_source": "editorial_factory_wp_import",
        "keyword_research_source": "editorial_factory_wp_import",
        "keyword_research_confidence": 0.85,
    }
    return payload


def _insert_with_schema_fallback(supabase, table_name: str, records: list[dict], log_label: str = "Records") -> bool:
    """Insert rows into Supabase table, gracefully dropping columns if schema has not yet migrated."""
    if not records:
        return True
    try:
        supabase.table(table_name).insert(records).execute()
        return True
    except Exception as insert_err:
        err_str = str(insert_err)
        missing_cols = re.findall(r"Could not find the '([^']+)' column", err_str)
        if missing_cols:
            logger.warning(
                "Dropping missing columns for %s table %s: %s",
                log_label, table_name, missing_cols
            )
            fallback_records = []
            for r in records:
                copy_r = dict(r)
                for col in missing_cols:
                    copy_r.pop(col, None)
                fallback_records.append(copy_r)
            try:
                supabase.table(table_name).insert(fallback_records).execute()
                return True
            except Exception as fallback_err:
                logger.error("Fallback insert failed for %s on %s: %s", log_label, table_name, fallback_err)
                return False
        logger.error("Insert failed for %s on %s: %s", log_label, table_name, insert_err)
        return False


@wordpress_bp.route('/api/wordpress/sync-posts', methods=['POST'])
def sync_wordpress_posts():
    """
    Fetch posts from all configured WordPress sites for the user and save to DB
    including full SEO metadata (Yoast SEO, RankMath SEO, focus keywords, OpenGraph, etc.).
    Optional params:
      user_id (query param or body)
      import_to_titles (bool, default False) - whether to also populate Titles for Content Studio/Article Editor
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            logger.error("Supabase client not initialized")
            return jsonify({'error': 'Internal server error: Database connection failed'}), 500

        user_id = request.args.get('user_id')
        import_to_titles = request.args.get('import_to_titles', '').lower() in ('true', '1')
        
        data = request.get_json(silent=True) or {}
        if not user_id:
            user_id = data.get('user_id')
        if 'import_to_titles' in data:
            import_to_titles = bool(data.get('import_to_titles'))
        
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400

        # 1. Get WP Credentials for User
        response = supabase.table("wordPress_details").select("*").eq("user_id", user_id).execute()
        sites = response.data
        
        if not sites:
            return jsonify({'total_synced': 0, 'details': "No WordPress sites configured", 'logs': ["No sites found"]}), 200

        debug_logs = []
        debug_logs.append(f"Found {len(sites)} sites to sync")
        
        # Clear all existing imported posts for this user to avoid stale/orphaned posts
        try:
            supabase.table("wordpress_imported_posts").delete().eq("user_id", user_id).execute()
            debug_logs.append("Cleared old imported posts for user to ensure clean sync")
        except Exception as clear_err:
            logger.warning(f"Failed to clear old posts: {clear_err}")
            debug_logs.append(f"Warning: Could not clear old posts: {str(clear_err)}")

        total_posts_saved = 0
        titles_created_count = 0
        
        # 2. Iterate each site
        for i, site in enumerate(sites):
            domain = site.get('domain')
            api_domain = (site.get('cms') or site.get('cms_url') or domain or "").strip()
            debug_logs.append(f"Processing site {i+1}/{len(sites)}: {domain} (API domain: {api_domain})")
            try:
                username = site.get('wpUserName')
                password = site.get('wordpress_key')
                site_id = site.get('id')
                
                if not api_domain or not username or not password:
                    debug_logs.append(f"Skipping site {domain} due to missing credentials or domain")
                    continue
                    
                client = WordPressClient(api_domain, username, password)
                
                # Fetch Categories
                debug_logs.append(f"Fetching categories for {domain}...")
                try:
                    categories = client.get_categories()
                    if categories:
                         supabase.table("wordPress_details").update({"categories": categories}).eq("id", site_id).execute()
                         debug_logs.append(f"Synced {len(categories)} categories")
                except Exception as cat_err:
                     debug_logs.append(f"Error fetching categories: {cat_err}")
                
                # Fetch all posts with full SEO metadata (_embed=1)
                debug_logs.append(f"Fetching posts with full SEO metadata for {domain}...")
                posts = []
                page = 1
                try:
                    while True:
                        try:
                            page_posts = client.get_posts(page=page, per_page=100, embed=True)
                            if not page_posts:
                                break
                            posts.extend(page_posts)
                            if len(page_posts) < 100:
                                break
                            page += 1
                        except Exception as page_err:
                            if hasattr(page_err, 'response') and page_err.response is not None and page_err.response.status_code == 400:
                                break
                            raise page_err
                    debug_logs.append(f"Fetched {len(posts)} posts for {domain}")
                except Exception as fetch_err:
                    debug_logs.append(f"Error fetching posts for {domain}: {str(fetch_err)}")
                    if not posts:
                        continue
                
                if not posts:
                    debug_logs.append(f"No posts found for {domain}")
                    continue
                    
                # 3. Extract SEO metadata and save to Supabase
                records = []
                titles_payloads = []
                for post in posts:
                    extracted = extract_wordpress_seo_metadata(post, domain=domain)
                    
                    record = {
                        "user_id": user_id,
                        "wordpress_detail_id": site_id,
                        "post_id": extracted["post_id"],
                        "title": extracted["title"],
                        "link": extracted["link"],
                        "excerpt": extracted["excerpt"],
                        "slug": extracted["slug"],
                        "content_html": extracted["content_html"],
                        "published_at": extracted["published_at"],
                        "modified_at": extracted["modified_at"],
                        "featured_image_url": extracted["featured_image_url"],
                        "featured_image_alt": extracted["featured_image_alt"],
                        "category_ids": extracted["category_ids"],
                        "category_names": extracted["category_names"],
                        "tag_ids": extracted["tag_ids"],
                        "tag_names": extracted["tag_names"],
                        "seo_title": extracted["seo_title"],
                        "seo_description": extracted["seo_description"],
                        "focus_keyword": extracted["focus_keyword"],
                        "primary_keyword": extracted["primary_keyword"],
                        "secondary_keywords": extracted["secondary_keywords"],
                        "canonical_url": extracted["canonical_url"],
                        "seo_metadata": extracted["seo_metadata"],
                        "raw_post_json": extracted["raw_post_json"],
                    }
                    records.append(record)
                    
                    if import_to_titles:
                        t_payload = _build_titles_payload_from_imported_post(user_id, site_id, domain, extracted)
                        titles_payloads.append(t_payload)
                
                if records:
                    debug_logs.append(f"Saving {len(records)} records with SEO metadata for {domain}")
                    _insert_with_schema_fallback(supabase, "wordpress_imported_posts", records, log_label=f"WP Posts ({domain})")
                    total_posts_saved += len(records)
                    debug_logs.append(f"Saved {len(records)} records with SEO metadata for {domain}")
                    
                if titles_payloads:
                    debug_logs.append(f"Importing {len(titles_payloads)} posts into Titles for {domain}")
                    _insert_with_schema_fallback(supabase, "Titles", titles_payloads, log_label=f"Titles ({domain})")
                    titles_created_count += len(titles_payloads)
                    debug_logs.append(f"Imported {len(titles_payloads)} posts to Titles")
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Failed to sync site {site.get('domain')}: {str(e)}")
                debug_logs.append(f"Error syncing {site.get('domain')}: {str(e)}")
                continue

        return jsonify({
            'total_synced': total_posts_saved,
            'titles_created': titles_created_count,
            'details': f"Sync completed. Processed {len(sites)} sites with SEO metadata.",
            'logs': debug_logs
        }), 200

    except Exception as e:
        logger.error(f"Sync error: {str(e)}")
        return jsonify({'error': str(e), 'logs': [str(e)]}), 500


@wordpress_bp.route('/api/wordpress/import-to-titles', methods=['POST'])
def import_post_to_titles():
    """
    Import a specific imported WordPress post into the Titles table with full SEO metadata
    for editing in Content Studio and Article Editor.
    
    Expected body:
    {
        "user_id": "...",
        "imported_post_id": "..." (UUID from wordpress_imported_posts) OR "post_id": 123,
        "wordpress_detail_id": optional site id
    }
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({'error': 'Database connection failed'}), 500

        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id')
        imported_post_id = data.get('imported_post_id')
        post_id = data.get('post_id')
        site_id = data.get('wordpress_detail_id')

        if not user_id or (not imported_post_id and not post_id):
            return jsonify({'error': 'Missing user_id or imported_post_id/post_id'}), 400

        # Fetch imported post record
        query = supabase.table("wordpress_imported_posts").select("*").eq("user_id", user_id)
        if imported_post_id:
            query = query.eq("id", imported_post_id)
        elif post_id:
            query = query.eq("post_id", post_id)
            if site_id:
                query = query.eq("wordpress_detail_id", site_id)
        
        resp = query.limit(1).execute()
        if not resp.data:
            return jsonify({'error': 'Imported post not found'}), 404

        post_row = resp.data[0]
        
        # Get site domain if available
        site_id = post_row.get("wordpress_detail_id") or site_id
        domain = ""
        if site_id:
            site_resp = supabase.table("wordPress_details").select("domain").eq("id", site_id).limit(1).execute()
            if site_resp.data:
                domain = site_resp.data[0].get("domain") or ""

        # Check if already has a titles_record_id
        existing_title_id = post_row.get("titles_record_id")
        if existing_title_id:
            # Check if title still exists
            check = supabase.table("Titles").select("id").eq("id", existing_title_id).limit(1).execute()
            if check.data:
                return jsonify({
                    'success': True,
                    'title_id': existing_title_id,
                    'already_imported': True,
                    'message': 'Article already exists in Content Library'
                }), 200

        # Extract/prepare rich SEO metadata
        raw_post = post_row.get("raw_post_json")
        if isinstance(raw_post, dict) and raw_post:
            extracted = extract_wordpress_seo_metadata(raw_post, domain=domain)
        else:
            # Reconstruct from stored columns
            extracted = {
                "title": post_row.get("title") or "Untitled Article",
                "slug": post_row.get("slug") or "",
                "content_html": post_row.get("content_html") or "",
                "excerpt": post_row.get("excerpt") or "",
                "link": post_row.get("link") or "",
                "published_at": post_row.get("published_at"),
                "modified_at": post_row.get("modified_at"),
                "post_id": post_row.get("post_id"),
                "featured_image_url": post_row.get("featured_image_url") or "",
                "featured_image_alt": post_row.get("featured_image_alt") or "",
                "category_ids": post_row.get("category_ids") or [],
                "category_names": post_row.get("category_names") or [],
                "tag_ids": post_row.get("tag_ids") or [],
                "tag_names": post_row.get("tag_names") or [],
                "seo_title": post_row.get("seo_title") or post_row.get("title") or "",
                "seo_description": post_row.get("seo_description") or post_row.get("excerpt") or "",
                "focus_keyword": post_row.get("focus_keyword") or post_row.get("primary_keyword") or "",
                "primary_keyword": post_row.get("primary_keyword") or post_row.get("focus_keyword") or "",
                "secondary_keywords": post_row.get("secondary_keywords") or [],
                "canonical_url": post_row.get("canonical_url") or post_row.get("link") or "",
                "seo_metadata": post_row.get("seo_metadata") or {},
                "seo_optimization_score": 85.0,
                "readability_score": 75.0,
            }

        title_payload = _build_titles_payload_from_imported_post(user_id, site_id, domain, extracted)
        
        insert_ok = _insert_with_schema_fallback(supabase, "Titles", [title_payload], log_label="Import to Titles")
        if not insert_ok:
            return jsonify({'error': 'Failed to insert article into Titles table'}), 500

        title_id = title_payload["id"]

        # Stamp titles_record_id back to wordpress_imported_posts
        try:
            supabase.table("wordpress_imported_posts").update({"titles_record_id": title_id}).eq("id", post_row.get("id")).execute()
        except Exception as stamp_err:
            logger.warning(f"Could not stamp titles_record_id on imported post: {stamp_err}")

        return jsonify({
            'success': True,
            'title_id': title_id,
            'title': title_payload.get("Title"),
            'primary_keyword': title_payload.get("primary_keyword"),
            'secondary_keywords': title_payload.get("secondary_keywords"),
            'seo_metadata': extracted.get("seo_metadata"),
            'message': 'Article imported to Content Studio successfully with full SEO metadata'
        }), 200

    except Exception as e:
        logger.error(f"Error importing post to Titles: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@wordpress_bp.route('/api/wordpress/import-all-to-titles', methods=['POST'])
def import_all_posts_to_titles():
    """
    Batch import multiple imported posts into the Titles table with SEO metadata.
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({'error': 'Database connection failed'}), 500

        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id')
        imported_ids = data.get('imported_ids')  # list of UUIDs

        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400

        query = supabase.table("wordpress_imported_posts").select("*").eq("user_id", user_id)
        if imported_ids and isinstance(imported_ids, list):
            query = query.in_("id", imported_ids)

        resp = query.execute()
        posts = resp.data or []
        if not posts:
            return jsonify({'imported_count': 0, 'message': 'No posts to import'}), 200

        # Load sites for domain resolution
        site_resp = supabase.table("wordPress_details").select("id, domain").eq("user_id", user_id).execute()
        domain_by_site = {str(s["id"]): s.get("domain", "") for s in (site_resp.data or [])}

        titles_payloads = []
        created_ids = []
        for post_row in posts:
            site_id = post_row.get("wordpress_detail_id")
            domain = domain_by_site.get(str(site_id), "")
            
            raw_post = post_row.get("raw_post_json")
            if isinstance(raw_post, dict) and raw_post:
                extracted = extract_wordpress_seo_metadata(raw_post, domain=domain)
            else:
                extracted = {
                    "title": post_row.get("title") or "Untitled Article",
                    "slug": post_row.get("slug") or "",
                    "content_html": post_row.get("content_html") or "",
                    "excerpt": post_row.get("excerpt") or "",
                    "link": post_row.get("link") or "",
                    "published_at": post_row.get("published_at"),
                    "modified_at": post_row.get("modified_at"),
                    "post_id": post_row.get("post_id"),
                    "featured_image_url": post_row.get("featured_image_url") or "",
                    "featured_image_alt": post_row.get("featured_image_alt") or "",
                    "category_ids": post_row.get("category_ids") or [],
                    "category_names": post_row.get("category_names") or [],
                    "tag_ids": post_row.get("tag_ids") or [],
                    "tag_names": post_row.get("tag_names") or [],
                    "seo_title": post_row.get("seo_title") or post_row.get("title") or "",
                    "seo_description": post_row.get("seo_description") or post_row.get("excerpt") or "",
                    "focus_keyword": post_row.get("focus_keyword") or post_row.get("primary_keyword") or "",
                    "primary_keyword": post_row.get("primary_keyword") or post_row.get("focus_keyword") or "",
                    "secondary_keywords": post_row.get("secondary_keywords") or [],
                    "canonical_url": post_row.get("canonical_url") or post_row.get("link") or "",
                    "seo_metadata": post_row.get("seo_metadata") or {},
                    "seo_optimization_score": 85.0,
                    "readability_score": 75.0,
                }
            payload = _build_titles_payload_from_imported_post(user_id, site_id, domain, extracted)
            titles_payloads.append(payload)
            created_ids.append(payload["id"])

        if titles_payloads:
            _insert_with_schema_fallback(supabase, "Titles", titles_payloads, log_label="Batch Import to Titles")

        return jsonify({
            'success': True,
            'imported_count': len(titles_payloads),
            'title_ids': created_ids,
            'message': f'Successfully imported {len(titles_payloads)} articles into Content Library'
        }), 200

    except Exception as e:
        logger.error(f"Error in batch import to Titles: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@wordpress_bp.route('/api/wordpress/sync-project-categories', methods=['POST'])
def sync_project_categories_to_wordpress():
    """
    Synchronize a project's local categories/subcategories to its WordPress site.

    Expected JSON body:
    {
      "user_id": "<uuid>",
      "project_id": "<uuid>"
    }
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            logger.error("Supabase client not initialized")
            return jsonify({'error': 'Internal server error: Database connection failed'}), 500

        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id') or request.args.get('user_id')
        project_id = data.get('project_id') or request.args.get('project_id')

        if not user_id or not project_id:
            return jsonify({'error': 'Missing user_id or project_id'}), 400

        project_resp = (
            supabase
            .table("projects")
            .select("id, user_id, domain, wpusername, wordpress_key, app_name")
            .eq("id", project_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        project_rows = project_resp.data or []
        if not project_rows:
            return jsonify({'error': 'Project not found'}), 404
        project = project_rows[0]

        project_domain = (project.get("domain") or "").strip()
        username = (project.get("wpusername") or project.get("wpUserName") or "").strip()
        app_password = (project.get("wordpress_key") or "").strip()
        api_domain = project_domain
        
        # Load credentials and CMS domain from wordPress_details if available to override projects table
        if project_domain:
            try:
                wp_resp = (
                    supabase
                    .table("wordPress_details")
                    .select("wpUserName, wordpress_key, cms, cms_url")
                    .eq("user_id", user_id)
                    .eq("domain", project_domain)
                    .limit(1)
                    .execute()
                )
                if wp_resp.data:
                    wp_detail = wp_resp.data[0]
                    username = (wp_detail.get("wpUserName") or wp_detail.get("wpusername") or username).strip()
                    app_password = (wp_detail.get("wordpress_key") or app_password).strip()
                    
                    cms_domain = (wp_detail.get("cms") or wp_detail.get("cms_url") or "").strip()
                    if cms_domain:
                        api_domain = cms_domain
            except Exception as wp_err:
                logger.warning("Failed to fetch credentials from wordPress_details for domain %s: %s", project_domain, wp_err)

        if not api_domain or not username or not app_password:
            return jsonify({'error': 'WordPress credentials are incomplete for this project'}), 400

        local_categories = None
        category_select_attempts = [
            "id, name, description, slug, level, parent_category_id, sort_order, wordpress_category_id, wordpress_parent_category_id, wordpress_site_domain",
            "id, name, slug, level, parent_category_id, sort_order, wordpress_category_id, wordpress_parent_category_id, wordpress_site_domain",
            "id, name, description, slug, level, parent_category_id, sort_order",
            "id, name, slug, level, parent_category_id, sort_order",
        ]
        for select_fields in category_select_attempts:
            try:
                categories_resp = (
                    supabase
                    .table("project_categories")
                    .select(select_fields)
                    .eq("project_id", project_id)
                    .eq("user_id", user_id)
                    .order("level", desc=False)
                    .order("sort_order", desc=False)
                    .order("name", desc=False)
                    .execute()
                )
                local_categories = categories_resp.data or []
                break
            except Exception:
                continue

        if local_categories is None:
            raise Exception("Failed to load project categories for synchronization")

        for row in local_categories:
            row.setdefault("description", None)
            row.setdefault("wordpress_category_id", None)
            row.setdefault("wordpress_parent_category_id", None)
            row.setdefault("wordpress_site_domain", None)
        if not local_categories:
            return jsonify({
                "success": True,
                "synced": 0,
                "created": 0,
                "updated": 0,
                "details": "No local categories to sync."
            }), 200

        client = WordPressClient(api_domain, username, app_password)
        try:
            wp_categories = client.get_categories_detailed()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            reason = None
            if e.response is not None:
                try:
                    body = e.response.json() or {}
                    reason = body.get("message") or body.get("code")
                except Exception:
                    reason = e.response.text[:300]
            logger.warning(
                "WordPress category fetch failed for domain=%s project=%s status=%s reason=%s",
                api_domain,
                project_id,
                status,
                reason,
            )
            return jsonify({
                "success": False,
                "error": "WordPress API request failed while reading categories",
                "status_code": status,
                "details": reason or str(e),
            }), 400 if status in (400, 401, 403, 404) else 502
        except requests.exceptions.RequestException as e:
            logger.warning(
                "WordPress network error for domain=%s project=%s: %s",
                api_domain,
                project_id,
                str(e),
            )
            return jsonify({
                "success": False,
                "error": "Failed to connect to WordPress API",
                "details": str(e),
            }), 502

        # Build quick lookup indexes.
        by_slug_parent = {}
        by_name_parent = {}
        by_id = {}
        for cat in wp_categories:
            cat_id = int(cat.get("id"))
            parent = int(cat.get("parent") or 0)
            slug = (cat.get("slug") or "").strip().lower()
            name = (cat.get("name") or "").strip().lower()
            by_id[cat_id] = cat
            if slug:
                by_slug_parent[(slug, parent)] = cat
            if name:
                by_name_parent[(name, parent)] = cat

        created_count = 0
        updated_count = 0
        synced_count = 0
        local_to_wp: dict[str, int] = {}
        update_rows = []
        sync_errors = []
        sync_details = []

        level_1 = [c for c in local_categories if int(c.get("level") or 0) == 1]
        level_2 = [c for c in local_categories if int(c.get("level") or 0) == 2]

        by_slug_global = {}
        for cat in wp_categories:
            slug = (cat.get("slug") or "").strip().lower()
            if slug:
                by_slug_global[slug] = cat

        category_descriptions = _generate_category_descriptions(
            domain=api_domain,
            project_name=(project.get("app_name") or "").strip(),
            categories=local_categories,
        )

        def ensure_wp_category(local_cat, parent_wp_id: int = 0):
            nonlocal created_count, updated_count, synced_count
            local_id = str(local_cat.get("id") or "")
            app_name = (local_cat.get("name") or "").strip()
            wp_name = _shorten_wp_title(app_name, max_chars=60)
            slug = (local_cat.get("slug") or "").strip().lower() or _slugify(app_name)
            description = (category_descriptions.get(local_id) or "").strip()
            mapped_wp_id = local_cat.get("wordpress_category_id")
            if not app_name:
                return None

            existing = None
            mapped_wp_id_str = str(mapped_wp_id or "").strip()
            if mapped_wp_id_str:
                try:
                    mapped_wp_id = int(mapped_wp_id_str)
                except (TypeError, ValueError):
                    raise Exception(f"Invalid stored wordpress_category_id: {mapped_wp_id}")

                if mapped_wp_id <= 0:
                    mapped_wp_id = None

            else:
                mapped_wp_id = None

            if mapped_wp_id is not None:
                try:
                    mapped_wp_id = int(mapped_wp_id)
                except (TypeError, ValueError):
                    raise Exception(f"Invalid stored wordpress_category_id: {mapped_wp_id}")

                existing = by_id.get(mapped_wp_id)
                if existing is None:
                    try:
                        existing = client.get_category(mapped_wp_id)
                    except requests.exceptions.HTTPError as e:
                        status = e.response.status_code if e.response is not None else None
                        if status == 404:
                            existing = None
                        else:
                            raise
                    if existing is not None:
                        by_id[mapped_wp_id] = existing
                        existing_slug = (existing.get("slug") or "").strip().lower()
                        existing_name = (existing.get("name") or "").strip().lower()
                        existing_parent = int(existing.get("parent") or 0)
                        if existing_slug:
                            by_slug_parent[(existing_slug, existing_parent)] = existing
                            by_slug_global[existing_slug] = existing
                        if existing_name:
                            by_name_parent[(existing_name, existing_parent)] = existing
            if mapped_wp_id is None:
                existing = (
                    by_slug_parent.get((slug, parent_wp_id))
                    or by_name_parent.get((wp_name.lower(), parent_wp_id))
                    or by_slug_global.get(slug)
                )
            if existing:
                cat_id = int(existing.get("id"))
                needs_update = (
                    (existing.get("name") or "").strip() != wp_name
                    or (existing.get("slug") or "").strip().lower() != slug
                    or int(existing.get("parent") or 0) != int(parent_wp_id or 0)
                    or (existing.get("description") or "").strip() != description
                )
                if needs_update:
                    updated = client.update_category(
                        cat_id,
                        name=wp_name,
                        slug=slug,
                        parent=parent_wp_id,
                        description=description,
                    )
                    existing = updated
                    updated_count += 1
            else:
                created = client.create_category(
                    name=wp_name,
                    slug=slug,
                    parent=parent_wp_id,
                    description=description,
                )
                existing = created
                created_count += 1

            cat_id = int(existing.get("id"))
            # refresh indexes for child lookups
            by_id[cat_id] = existing
            by_slug_parent[(slug, int(parent_wp_id or 0))] = existing
            by_name_parent[(wp_name.lower(), int(parent_wp_id or 0))] = existing
            by_slug_global[slug] = existing
            synced_count += 1
            return cat_id

        # 1) Sync parent categories first.
        for cat in level_1:
            local_id = str(cat.get("id"))
            try:
                wp_id = ensure_wp_category(cat, parent_wp_id=0)
            except Exception as e:
                sync_errors.append({
                    "local_category_id": local_id,
                    "name": cat.get("name"),
                    "level": 1,
                    "error": str(e),
                })
                continue
            if wp_id:
                local_to_wp[local_id] = wp_id
                update_rows.append({
                    "id": local_id,
                    "wordpress_category_id": wp_id,
                    "wordpress_parent_category_id": None,
                    "wordpress_site_domain": project_domain,
                    "wordpress_last_synced_at": datetime.utcnow().isoformat(),
                })
                sync_details.append({
                    "local_category_id": local_id,
                    "name": cat.get("name"),
                    "wordpress_name": _shorten_wp_title((cat.get("name") or "").strip(), max_chars=60),
                    "level": 1,
                    "wordpress_category_id": wp_id,
                })

        # 2) Sync subcategories, linked to mapped parent.
        for cat in level_2:
            local_id = str(cat.get("id"))
            local_parent = str(cat.get("parent_category_id") or "")
            parent_wp_id = local_to_wp.get(local_parent, 0)
            try:
                wp_id = ensure_wp_category(cat, parent_wp_id=parent_wp_id)
            except Exception as e:
                sync_errors.append({
                    "local_category_id": local_id,
                    "name": cat.get("name"),
                    "level": 2,
                    "parent_local_id": local_parent or None,
                    "parent_wordpress_id": parent_wp_id or None,
                    "error": str(e),
                })
                continue
            if wp_id:
                local_to_wp[local_id] = wp_id
                update_rows.append({
                    "id": local_id,
                    "wordpress_category_id": wp_id,
                    "wordpress_parent_category_id": parent_wp_id or None,
                    "wordpress_site_domain": project_domain,
                    "wordpress_last_synced_at": datetime.utcnow().isoformat(),
                })
                sync_details.append({
                    "local_category_id": local_id,
                    "name": cat.get("name"),
                    "wordpress_name": _shorten_wp_title((cat.get("name") or "").strip(), max_chars=60),
                    "level": 2,
                    "parent_local_id": local_parent or None,
                    "parent_wordpress_id": parent_wp_id or None,
                    "wordpress_category_id": wp_id,
                })

        # Persist mappings back to project_categories.
        mapping_update_errors = []
        for row in update_rows:
            update_payload = {
                "wordpress_category_id": row["wordpress_category_id"],
                "wordpress_parent_category_id": row["wordpress_parent_category_id"],
                "wordpress_site_domain": row["wordpress_site_domain"],
                "updated_at": datetime.utcnow().isoformat(),
            }
            # wordpress_last_synced_at may not exist on older schemas; degrade gracefully.
            try:
                update_payload["wordpress_last_synced_at"] = row["wordpress_last_synced_at"]
                supabase.table("project_categories").update(update_payload).eq("id", row["id"]).eq("user_id", user_id).execute()
            except Exception:
                # First fallback: retry without wordpress_last_synced_at for older schemas.
                try:
                    update_payload.pop("wordpress_last_synced_at", None)
                    supabase.table("project_categories").update(update_payload).eq("id", row["id"]).eq("user_id", user_id).execute()
                except Exception as persist_err:
                    # Do not fail the whole sync if mapping persistence fails.
                    mapping_update_errors.append({
                        "local_category_id": row["id"],
                        "error": str(persist_err),
                    })

        return jsonify({
            "success": True,
            "project_id": project_id,
            "domain": project_domain,
            "synced": synced_count,
            "created": created_count,
            "updated": updated_count,
            "errors_count": len(sync_errors),
            "errors": sync_errors,
            "mapping_update_errors_count": len(mapping_update_errors),
            "mapping_update_errors": mapping_update_errors,
            "category_results": sync_details,
            "details": (
                f"Synced {synced_count} categories to WordPress "
                f"({created_count} created, {updated_count} updated, "
                f"{len(sync_errors)} sync errors, {len(mapping_update_errors)} mapping update errors)."
            ),
        }), 200
    except Exception as e:
        logger.error(f"Project category sync error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
