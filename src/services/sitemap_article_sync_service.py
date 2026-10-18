import os
import re
import html
import logging
import hashlib
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set, Tuple
from urllib.parse import urlparse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

from supabase_client import get_supabase_client
from src.api.wordpress import _insert_with_schema_fallback, _build_titles_payload_from_imported_post

logger = logging.getLogger(__name__)

# Default domains requested by user for reading available articles via front-end sitemaps
DEFAULT_SITEMAP_DOMAINS = [
    "giniloh.com",
    "wellroost.com",
    "aichieve.net",
]

# Domain aliases/fallbacks (e.g., welroost.com vs live wellroost.com)
DOMAIN_FALLBACKS: Dict[str, List[str]] = {
    "welroost.com": ["welroost.com", "wellroost.com"],
    "wellroost.com": ["wellroost.com", "welroost.com"],
}

# Common static and utility paths that are not articles
STATIC_EXCLUDE_PATHS: Set[str] = {
    "",
    "/",
    "/about",
    "/about/",
    "/articles",
    "/articles/",
    "/apps",
    "/apps/",
    "/calculators",
    "/calculators/",
    "/categories",
    "/categories/",
    "/author",
    "/author/",
    "/contact",
    "/contact/",
    "/privacy",
    "/privacy/",
    "/privacy-policy",
    "/privacy-policy/",
    "/terms",
    "/terms/",
    "/terms-of-service",
    "/terms-of-service/",
    "/legal",
    "/legal/",
    "/disclaimer",
    "/disclaimer/",
    "/showcase",
    "/showcase/",
    "/search",
    "/search/",
    "/tag",
    "/tag/",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _slug_to_title(slug: str) -> str:
    """Derive a readable title from an article slug."""
    if not slug:
        return "Untitled Article"
    # Remove leading date patterns e.g. 2026-03-01-
    clean = re.sub(r"^\d{4}[-/]\d{2}[-/]\d{2}[-/]", "", slug).strip("-")
    words = clean.replace("-", " ").replace("_", " ").split()
    return " ".join(w.capitalize() for w in words)


def _extract_primary_and_secondary_keywords(slug: str, title: str) -> Tuple[str, List[str]]:
    """Derive primary and secondary keywords from slug and title."""
    stop_words = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
        "by", "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "from", "up", "down", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "not", "this", "that", "these", "those", "how", "what", "why",
        "your", "our", "their", "its", "of", "can", "will", "just", "got", "cheaper"
    }
    
    clean_slug = re.sub(r"^\d{4}[-/]\d{2}[-/]\d{2}[-/]", "", slug or "").strip("-")
    slug_tokens = [t.lower() for t in re.split(r"[-_]+", clean_slug) if t and not t.isdigit()]
    meaningful = [t for t in slug_tokens if t not in stop_words and len(t) > 2]

    primary_keyword = ""
    if len(meaningful) >= 2:
        primary_keyword = " ".join(meaningful[:3])
    elif meaningful:
        primary_keyword = meaningful[0]
    elif slug_tokens:
        primary_keyword = " ".join(slug_tokens[:2])
    else:
        primary_keyword = title.lower()

    secondary_keywords = []
    if len(meaningful) > 3:
        for i in range(1, len(meaningful) - 1):
            phrase = " ".join(meaningful[i:i + 2])
            if phrase != primary_keyword and phrase not in secondary_keywords:
                secondary_keywords.append(phrase)

    return primary_keyword, secondary_keywords[:5]


def _clean_page_title(raw_title: str) -> str:
    """Clean raw <title> tag, stripping site name branding suffixes."""
    if not raw_title:
        return ""
    text = html.unescape(raw_title.strip())
    # Strip site brand suffixes like | WellRoost, | Giniloh, - Aichieve
    text = re.sub(
        r"\s*([|–—-])\s*(WellRoost|Well\s*Roost|Welroost|Gini\s*Loh|Giniloh|Aichieve|Astro).*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text or raw_title.strip()


def _fetch_page_head_meta(url: str, timeout: int = 4) -> Tuple[str, str]:
    """
    Fetch the first 8KB of an HTML page to extract <title> and meta description.
    Falls back gracefully if the network request fails or times out.
    """
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read(8192).decode("utf-8", errors="replace")
            
            # Extract title
            title_m = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
            raw_title = title_m.group(1).strip() if title_m else ""
            clean_title = _clean_page_title(raw_title)

            # Extract meta description
            desc_m = re.search(
                r'<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']',
                content,
                re.IGNORECASE | re.DOTALL,
            )
            if not desc_m:
                desc_m = re.search(
                    r'<meta\s+property=["\']og:description["\']\s+content=["\'](.*?)["\']',
                    content,
                    re.IGNORECASE | re.DOTALL,
                )
            desc = html.unescape(desc_m.group(1).strip()) if desc_m else ""

            return clean_title, desc
    except Exception as e:
        logger.debug("Failed to fetch head metadata for %s: %s", url, e)
        return "", ""


class SitemapArticleSyncService:
    """Service to discover, extract, and sync articles from frontend sitemaps."""

    def __init__(self, supabase=None):
        self.supabase = supabase or get_supabase_client()

    def discover_sitemap_urls(self, domain: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Attempt to fetch and parse sitemaps for the domain.
        Supports /sitemap.xml, /sitemap-index.xml, and /sitemap-0.xml.
        Returns a tuple of (list of url info dicts, resolved_domain).
        """
        domains_to_try = DOMAIN_FALLBACKS.get(domain.lower(), [domain])
        
        for candidate_domain in domains_to_try:
            url_entries, ok = self._parse_domain_sitemaps(candidate_domain)
            if ok and url_entries:
                return url_entries, candidate_domain

        logger.warning("Could not discover valid sitemap for %s (tried %s)", domain, domains_to_try)
        return [], domain

    def _parse_domain_sitemaps(self, domain: str) -> Tuple[List[Dict[str, Any]], bool]:
        """Check standard sitemap paths for a domain."""
        candidate_paths = ["/sitemap.xml", "/sitemap-index.xml", "/sitemap-0.xml"]
        collected_urls: List[Dict[str, Any]] = []

        for path in candidate_paths:
            sitemap_url = f"https://{domain}{path}"
            try:
                req = urllib.request.Request(sitemap_url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=6) as resp:
                    if resp.status == 200:
                        xml_bytes = resp.read()
                        entries = self._parse_sitemap_xml(xml_bytes, domain)
                        if entries:
                            collected_urls.extend(entries)
                            return collected_urls, True
            except Exception as e:
                logger.debug("Sitemap path %s failed: %s", sitemap_url, e)

        return [], False

    def _parse_sitemap_xml(self, xml_bytes: bytes, domain: str, depth: int = 0) -> List[Dict[str, Any]]:
        """Parse XML bytes as either a sitemapindex or urlset."""
        results: List[Dict[str, Any]] = []
        if depth > 2:
            return results

        try:
            root = ET.fromstring(xml_bytes)
            tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag

            if tag == "sitemapindex":
                for sm in root.findall("{*}sitemap"):
                    loc_elem = sm.find("{*}loc")
                    if loc_elem is not None and loc_elem.text:
                        sub_url = loc_elem.text.strip()
                        try:
                            req = urllib.request.Request(sub_url, headers=HEADERS)
                            with urllib.request.urlopen(req, timeout=6) as sub_resp:
                                sub_xml = sub_resp.read()
                                results.extend(self._parse_sitemap_xml(sub_xml, domain, depth=depth + 1))
                        except Exception as sub_err:
                            logger.warning("Failed sub-sitemap %s: %s", sub_url, sub_err)

            elif tag == "urlset":
                for u in root.findall("{*}url"):
                    loc_elem = u.find("{*}loc")
                    lastmod_elem = u.find("{*}lastmod")
                    if loc_elem is not None and loc_elem.text:
                        loc = loc_elem.text.strip()
                        lastmod = lastmod_elem.text.strip() if (lastmod_elem is not None and lastmod_elem.text) else None
                        results.append({"loc": loc, "lastmod": lastmod, "domain": domain})

        except Exception as parse_err:
            logger.error("XML parse error on sitemap for %s: %s", domain, parse_err)

        return results

    def filter_and_extract_articles(self, url_entries: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
        """Filter raw URLs down to article candidates and extract article details."""
        candidates: List[Dict[str, Any]] = []
        seen_urls: Set[str] = set()

        for entry in url_entries:
            loc = entry.get("loc") or ""
            if not loc or loc in seen_urls:
                continue

            parsed = urlparse(loc)
            # Only include URLs for this site domain (ignore foreign domains)
            if parsed.netloc and parsed.netloc.lower() != domain.lower() and not parsed.netloc.lower().endswith(domain.lower()):
                continue

            path = parsed.path
            # Check static/utility excludes
            if path in STATIC_EXCLUDE_PATHS:
                continue
            if any(path.startswith(prefix) for prefix in ["/author/", "/tag/", "/categories/"]):
                continue

            # Strip leading/trailing slashes for clean slug
            slug = path.strip("/").split("/")[-1]
            if not slug:
                continue

            seen_urls.add(loc)
            candidates.append({
                "loc": loc,
                "slug": slug,
                "path": path,
                "lastmod": entry.get("lastmod"),
                "domain": domain,
            })

        return candidates

    def build_article_records(
        self,
        candidates: List[Dict[str, Any]],
        user_id: str,
        domain: str,
        max_head_fetch: int = 150,
    ) -> List[Dict[str, Any]]:
        """
        Fetch page heads concurrently to extract clean titles & descriptions,
        then assemble ready-to-insert records for wordpress_imported_posts.
        """
        url_to_meta: Dict[str, Tuple[str, str]] = {}

        # Limit concurrent head requests to avoid excessive overhead
        head_targets = candidates[:max_head_fetch]

        def _worker(c: Dict[str, Any]):
            u = c["loc"]
            title, desc = _fetch_page_head_meta(u, timeout=4)
            return u, title, desc

        with ThreadPoolExecutor(max_workers=12) as executor:
            future_to_url = {executor.submit(_worker, c): c["loc"] for c in head_targets}
            for future in as_completed(future_to_url):
                try:
                    u, title, desc = future.result()
                    url_to_meta[u] = (title, desc)
                except Exception:
                    pass

        now_iso = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        for c in candidates:
            loc = c["loc"]
            slug = c["slug"]
            lastmod = c.get("lastmod") or now_iso

            fetched_title, fetched_desc = url_to_meta.get(loc, ("", ""))
            title = fetched_title or _slug_to_title(slug)
            excerpt = fetched_desc or f"Article on {title} published on {domain}."

            primary_kw, secondary_kws = _extract_primary_and_secondary_keywords(slug, title)

            # Deterministic integer post_id from URL hash (compatible with int/bigint/text)
            numeric_post_id = int(hashlib.md5(loc.encode()).hexdigest()[:8], 16)

            record = {
                "user_id": user_id,
                "source_site": domain,
                "post_id": numeric_post_id,
                "title": title,
                "link": loc,
                "slug": slug,
                "excerpt": excerpt,
                "content_html": f"<p>{html.escape(excerpt)}</p>",
                "status": "publish",
                "published_at": lastmod,
                "modified_at": lastmod,
                "created_at": lastmod,
                "updated_at": now_iso,
                "seo_title": title,
                "seo_description": excerpt,
                "canonical_url": loc,
                "focus_keyword": primary_kw,
                "primary_keyword": primary_kw,
                "secondary_keywords": secondary_kws,
                "seo_metadata": {
                    "source": "sitemap",
                    "domain": domain,
                    "title": title,
                    "description": excerpt,
                    "primary_keyword": primary_kw,
                    "secondary_keywords": secondary_kws,
                },
                "raw_post_json": {
                    "source": "sitemap_import",
                    "loc": loc,
                    "slug": slug,
                    "domain": domain,
                    "lastmod": lastmod,
                },
            }
            records.append(record)

        return records

    def sync_all_sources(
        self,
        user_id: str,
        domains: Optional[List[str]] = None,
        import_to_titles: bool = False,
    ) -> Dict[str, Any]:
        """
        Synchronize available articles from the specified domains (or DEFAULT_SITEMAP_DOMAINS)
        into the database table `wordpress_imported_posts`.
        """
        if not user_id:
            raise ValueError("user_id is required to sync sitemap articles")

        if not self.supabase:
            return {
                "total_synced": 0,
                "error": "Database connection not available",
                "logs": ["Supabase client not initialized"],
            }

        target_domains = list(domains or DEFAULT_SITEMAP_DOMAINS)

        # Also pull any project domains from `projects` table for this user
        try:
            proj_resp = self.supabase.table("projects").select("domain").eq("user_id", user_id).execute()
            for p in (proj_resp.data or []):
                d = (p.get("domain") or "").strip().lower()
                if d and d not in target_domains:
                    target_domains.append(d)
        except Exception as p_err:
            logger.debug("Could not fetch project domains for user: %s", p_err)

        debug_logs = [f"Starting sitemap sync for {len(target_domains)} domains: {target_domains}"]
        total_synced = 0
        titles_created = 0
        by_site: Dict[str, int] = {}

        for domain in target_domains:
            debug_logs.append(f"Discovering sitemaps for {domain}...")
            raw_entries, resolved_domain = self.discover_sitemap_urls(domain)
            
            if not raw_entries:
                debug_logs.append(f"No sitemap URLs discovered for {domain}")
                continue

            candidates = self.filter_and_extract_articles(raw_entries, resolved_domain)
            debug_logs.append(f"Found {len(candidates)} article candidates on {resolved_domain}")

            if not candidates:
                continue

            records = self.build_article_records(candidates, user_id=user_id, domain=resolved_domain)

            # Clear existing records for this user and source_site to keep it fresh
            try:
                self.supabase.table("wordpress_imported_posts") \
                    .delete() \
                    .eq("user_id", user_id) \
                    .eq("source_site", resolved_domain) \
                    .execute()
            except Exception as del_err:
                logger.debug("Note: Could not delete prior posts for %s: %s", resolved_domain, del_err)

            # Insert with schema fallback
            ok = _insert_with_schema_fallback(
                self.supabase,
                "wordpress_imported_posts",
                records,
                log_label=f"Sitemap Articles ({resolved_domain})",
            )

            if ok:
                count = len(records)
                total_synced += count
                by_site[resolved_domain] = count
                debug_logs.append(f"Successfully synced {count} articles for {resolved_domain}")

                if import_to_titles:
                    titles_payloads = [
                        _build_titles_payload_from_imported_post(user_id, None, resolved_domain, r)
                        for r in records
                    ]
                    t_ok = _insert_with_schema_fallback(
                        self.supabase,
                        "Titles",
                        titles_payloads,
                        log_label=f"Sitemap Titles ({resolved_domain})",
                    )
                    if t_ok:
                        titles_created += len(titles_payloads)
                        debug_logs.append(f"Imported {len(titles_payloads)} articles into Titles for {resolved_domain}")
            else:
                debug_logs.append(f"Failed to insert records for {resolved_domain}")

        return {
            "total_synced": total_synced,
            "titles_created": titles_created,
            "by_site": by_site,
            "details": f"Synced {total_synced} available articles from sitemaps across {len(by_site)} site(s).",
            "logs": debug_logs,
        }


def sync_sitemap_articles(
    user_id: str,
    domains: Optional[List[str]] = None,
    import_to_titles: bool = False,
    supabase=None,
) -> Dict[str, Any]:
    """Helper entry point to perform sitemap article synchronization."""
    service = SitemapArticleSyncService(supabase=supabase)
    return service.sync_all_sources(user_id=user_id, domains=domains, import_to_titles=import_to_titles)


def ensure_sitemap_articles_synced(
    user_id: str,
    force: bool = False,
    background: bool = False,
    supabase=None,
) -> Optional[Dict[str, Any]]:
    """
    Ensure available articles are synced for user_id.
    If no articles exist or force=True, triggers synchronization.
    If background=True, runs asynchronously in a daemon thread.
    """
    if not user_id:
        return None

    sb = supabase or get_supabase_client()
    if not sb:
        return None

    if not force:
        try:
            # Check if user already has articles in wordpress_imported_posts
            resp = sb.table("wordpress_imported_posts").select("id", count="exact").eq("user_id", user_id).limit(1).execute()
            count = resp.count if resp.count is not None else len(resp.data or [])
            if count > 0:
                logger.debug("User %s already has %d imported articles, skipping sync", user_id, count)
                return {"already_synced": True, "count": count}
        except Exception as check_err:
            logger.debug("Error checking existing imported posts: %s", check_err)

    if background:
        t = threading.Thread(
            target=sync_sitemap_articles,
            kwargs={"user_id": user_id, "supabase": sb},
            daemon=True,
        )
        t.start()
        logger.info("Launched background sitemap article sync for user %s", user_id)
        return {"background": True}
    else:
        return sync_sitemap_articles(user_id=user_id, supabase=sb)
