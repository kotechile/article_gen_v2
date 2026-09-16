"""
Editorial Factory Service for Content Generator V2.

Handles integration with the secondary Supabase instance (project_ref: ixfdkninqeqmwuxncpvh),
fetching editorial articles, transforming their content (Markdown to HTML, GEO Key Takeaways,
Hook/Thesis/Deck synthesis, citations parsing), and importing them into the local Titles table.
"""

from __future__ import annotations

import os
import re
import html
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from supabase import create_client, Client
from supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

def _get_editorial_supabase_url() -> str:
    ref = os.getenv("EDITORIAL_SUPABASE_PROJECT_REF", "ixfdkninqeqmwuxncpvh").strip()
    return os.getenv("EDITORIAL_SUPABASE_URL", f"https://{ref}.supabase.co").strip()

def _get_editorial_supabase_key() -> str:
    return (
        os.getenv("EDITORIAL_SUPABASE_KEY") or
        os.getenv("EDITORIAL_SUPABASE_SERVICE_ROLE_KEY") or
        os.getenv("EDITORIAL_SUPABASE_ANON_KEY") or
        os.getenv("EDITORIAL_SUPABASE_API_KEY") or
        ""
    ).strip()


def clean_citation_numbers(text: str) -> str:
    """Strip bracketed citation numbers (e.g., [1], [^1], [1][3], [1, 2], [1-4]) and stray punctuation."""
    if not text:
        return ""
    # Remove bracketed citation markers like [1], [^1], [1][2], [1, 3], [1-3]
    cleaned = re.sub(r"\[\^?\d+(?:[-,\s]+\^?\d+)*\]", "", text)
    # Remove leading stray punctuation like ". ", ": ", "- "
    cleaned = re.sub(r"^[\s.,:;–—\-]+", "", cleaned)
    # Clean whitespace before punctuation
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    # Clean repeated whitespace
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _render_inline_markdown(text: str) -> str:
    """Strip leading bullets and convert markdown bold/italic into HTML tags."""
    cleaned = re.sub(r"^(?:[\s\u2022\u2023\u25E6\u2043\u2219\-\–\—]+|(?:\*\s+)|(?:\d+[.)]\s+))+", "", str(text or "")).strip()
    escaped = html.escape(cleaned)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"__(.+?)__", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", escaped)
    escaped = re.sub(r"(?<!_)_([^_]+?)_(?!_)", r"<em>\1</em>", escaped)
    return escaped


class EditorialFactoryService:
    """Service to interact with the Editorial Factory Supabase database."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None
    ):
        self._supabase_url = supabase_url
        self._supabase_key = supabase_key
        self._client: Optional[Client] = None

    @property
    def supabase_url(self) -> str:
        return self._supabase_url or _get_editorial_supabase_url()

    @property
    def supabase_key(self) -> str:
        return self._supabase_key or _get_editorial_supabase_key()

    def get_client(self) -> Optional[Client]:
        """Lazy-initialize and return the Supabase client for Editorial Factory."""
        if not self._client and self.supabase_url and self.supabase_key:
            try:
                self._client = create_client(self.supabase_url, self.supabase_key)
            except Exception as err:
                logger.warning(f"[EditorialFactoryService] Could not initialize Supabase client: {err}")
                self._client = None
        return self._client

    def _fetch_imported_status_map(
        self,
        user_id: Optional[str] = None,
        domain: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch existing Titles from the local database to determine which Editorial Factory
        articles have already been imported.
        Returns a map keyed by editorial_factory_id and normalized title.
        """
        imported_map: Dict[str, Dict[str, Any]] = {}
        local_supabase = get_supabase_client()
        if not local_supabase:
            return imported_map

        try:
            query = local_supabase.table("Titles").select("id, Title, idea_metadata, dateCreatedOn, domain, user_id")
            if user_id:
                query = query.eq("user_id", user_id)
            if domain:
                query = query.eq("domain", domain)

            res = query.execute()
            rows = res.data or []

            # If user_id or domain filter was applied but returned no records, fall back to checking across user's titles
            if not rows and domain:
                fallback_query = local_supabase.table("Titles").select("id, Title, idea_metadata, dateCreatedOn, domain, user_id")
                if user_id:
                    fallback_query = fallback_query.eq("user_id", user_id)
                fallback_res = fallback_query.execute()
                rows = fallback_res.data or []

            for row in rows:
                meta = row.get("idea_metadata") if isinstance(row.get("idea_metadata"), dict) else {}
                ed_id = str(meta.get("editorial_factory_id") or "").strip()
                row_title = str(row.get("Title") or "").strip().lower()
                clean_row_title = clean_citation_numbers(row_title).strip().lower()

                record_info = {
                    "title_id": row.get("id"),
                    "imported_at": meta.get("imported_at") or row.get("dateCreatedOn"),
                    "domain": row.get("domain"),
                }

                if ed_id:
                    imported_map[ed_id] = record_info
                if row_title:
                    imported_map[f"title:{row_title}"] = record_info
                if clean_row_title and clean_row_title != row_title:
                    imported_map[f"title:{clean_row_title}"] = record_info
        except Exception as err:
            logger.warning(f"[EditorialFactoryService] Failed to fetch imported titles map: {err}")

        return imported_map

    def list_articles(
        self,
        search: str = "",
        limit: int = 50,
        offset: int = 0,
        user_id: Optional[str] = None,
        domain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List articles from the Editorial Factory 'articles' table.
        Enriches results with is_imported flag by checking the local Titles table.
        Falls back to REST API or empty list if client initialization fails.
        """
        client = self.get_client()
        articles: List[Dict[str, Any]] = []

        if client:
            try:
                query = client.table("articles").select("*")
                if search:
                    query = query.ilike("title", f"%{search}%")
                query = query.order("created_at", desc=True).range(offset, offset + limit - 1)
                res = query.execute()
                articles = res.data or []
            except Exception as err:
                logger.warning(f"[EditorialFactoryService] Supabase query failed: {err}")

        # Fallback via direct REST if client failed or key was anon
        if not articles and self.supabase_url and self.supabase_key:
            try:
                endpoint = f"{self.supabase_url}/rest/v1/articles"
                headers = {
                    "apikey": self.supabase_key,
                    "Authorization": f"Bearer {self.supabase_key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation",
                }
                params: Dict[str, Any] = {
                    "select": "*",
                    "order": "created_at.desc",
                    "limit": limit,
                    "offset": offset,
                }
                if search:
                    params["title"] = f"ilike.%{search}%"

                resp = requests.get(endpoint, headers=headers, params=params, timeout=15)
                if resp.status_code == 200:
                    articles = resp.json()
            except Exception as req_err:
                logger.warning(f"[EditorialFactoryService] Direct REST fetch failed: {req_err}")

        # Fetch local imported map
        imported_map = self._fetch_imported_status_map(user_id=user_id, domain=domain)

        # Normalize article structures and enrich with imported metadata
        normalized: List[Dict[str, Any]] = []
        for art in articles:
            norm = self._normalize_article(art)
            if norm:
                art_id = str(norm.get("id") or "").strip()
                norm_title = str(norm.get("title") or "").strip().lower()
                clean_norm_title = clean_citation_numbers(norm_title).strip().lower()

                match = (
                    (imported_map.get(art_id) if art_id else None)
                    or (imported_map.get(f"title:{norm_title}") if norm_title else None)
                    or (imported_map.get(f"title:{clean_norm_title}") if clean_norm_title else None)
                )

                if match:
                    norm["is_imported"] = True
                    norm["imported_title_id"] = match.get("title_id")
                    norm["imported_at"] = match.get("imported_at")
                    norm["imported_domain"] = match.get("domain")
                else:
                    norm["is_imported"] = False
                    norm["imported_title_id"] = None
                    norm["imported_at"] = None
                    norm["imported_domain"] = None

                normalized.append(norm)

        return normalized

    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single article by ID from Editorial Factory."""
        client = self.get_client()
        if client:
            try:
                res = client.table("articles").select("*").eq("id", article_id).maybe_single().execute()
                if res.data:
                    return self._normalize_article(res.data)
            except Exception as err:
                logger.warning(f"[EditorialFactoryService] Supabase get single failed: {err}")

        # Direct REST fallback
        if self.supabase_url and self.supabase_key:
            try:
                endpoint = f"{self.supabase_url}/rest/v1/articles"
                headers = {
                    "apikey": self.supabase_key,
                    "Authorization": f"Bearer {self.supabase_key}",
                    "Content-Type": "application/json",
                }
                params = {
                    "select": "*",
                    "id": f"eq.{article_id}",
                    "limit": 1,
                }
                resp = requests.get(
                    endpoint,
                    headers=headers,
                    params=params,
                    timeout=15
                )
                if resp.status_code == 200:
                    rows = resp.json()
                    if rows:
                        return self._normalize_article(rows[0])
            except Exception as req_err:
                logger.warning(f"[EditorialFactoryService] Direct REST fetch single failed: {req_err}")

        return None

    def _normalize_article(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize varying column names from Editorial Factory."""
        raw_id = str(row.get("id") or row.get("article_id") or "")
        title = (
            row.get("title")
            or row.get("Title")
            or row.get("headline")
            or "Untitled Editorial Article"
        )
        content = (
            row.get("content")
            or row.get("htmlArticle")
            or row.get("body")
            or row.get("markdown")
            or row.get("text")
            or ""
        )
        summary = (
            row.get("summary")
            or row.get("deck")
            or row.get("excerpt")
            or row.get("description")
            or row.get("userDescription")
            or ""
        )
        hook = row.get("hook") or row.get("Hook") or ""
        thesis = row.get("thesis") or row.get("Thesis") or ""
        tags = row.get("tags") or row.get("keywords") or row.get("Keywords") or []
        created_at = row.get("created_at") or row.get("dateCreatedOn") or datetime.utcnow().isoformat()
        author = row.get("author") or row.get("writer") or "Editorial Factory"
        takeaways = (
            row.get("takeaways")
            or row.get("key_takeaways")
            or row.get("at_a_glance")
            or row.get("tldr")
            or ((row.get("metadata") or {}).get("takeaways") if isinstance(row.get("metadata"), dict) else None)
            or []
        )

        # Calculate word count
        words = len(re.findall(r"\w+", content)) if content else 0

        return {
            "id": raw_id,
            "title": clean_citation_numbers(title),
            "content": content,
            "summary": clean_citation_numbers(summary),
            "hook": clean_citation_numbers(hook),
            "thesis": clean_citation_numbers(thesis),
            "tags": tags if isinstance(tags, list) else [str(tags)],
            "takeaways": takeaways if isinstance(takeaways, list) else [str(takeaways)],
            "created_at": created_at,
            "author": author,
            "word_count": words,
            "raw_data": row,
        }

    def markdown_to_html(self, text: str) -> str:
        """Convert Markdown content to clean HTML structure for the editor."""
        if not text:
            return ""

        # Remove trailing markdown references block from body HTML (the editor manages the structured references)
        text = re.sub(
            r"(?:^|\n)(?:#{1,4}\s*(?:References|Sources|Citations|Bibliography))[\s\S]*$",
            "",
            text,
            flags=re.IGNORECASE
        ).strip()

        # Remove explicit "At a glance" / "Key Takeaways" markdown headers if present in body
        text = re.sub(
            r"(?:^|\n)(?:#{1,4}\s*(?:At\s+a\s+glance|Key\s+Takeaways|TL;?DR|Executive\s+Summary))\s*$",
            "",
            text,
            flags=re.IGNORECASE | re.MULTILINE
        ).strip()

        # If it's already HTML (contains <p> or <h[1-6]>), return cleaned version
        if bool(re.search(r"<(p|h[1-6]|div|section|table|ul|ol)\b", text, re.IGNORECASE)):
            return text

        lines = text.split("\n")
        html_blocks: List[str] = []
        in_list = False
        list_type = "ul"
        in_table = False
        table_rows: List[str] = []

        def close_list():
            nonlocal in_list, list_type
            if in_list:
                html_blocks.append(f"</{list_type}>")
                in_list = False

        def close_table():
            nonlocal in_table, table_rows
            if in_table:
                if table_rows:
                    html_blocks.append('<table class="border-collapse w-full my-4">')
                    html_blocks.extend(table_rows)
                    html_blocks.append("</table>")
                table_rows = []
                in_table = False

        for line in lines:
            trimmed = line.strip()

            if not trimmed:
                close_list()
                close_table()
                continue

            # Table row
            if trimmed.startswith("|") and trimmed.endswith("|"):
                close_list()
                # Skip divider rows like |---|---|
                if re.match(r"^\|[\s\-:|]+\|$", trimmed):
                    continue
                cells = [c.strip() for c in trimmed.strip("|").split("|")]
                tag = "th" if not in_table and not table_rows else "td"
                row_html = "<tr>" + "".join(f"<{tag} class='border p-2'>{html.escape(c)}</{tag}>" for c in cells) + "</tr>"
                table_rows.append(row_html)
                in_table = True
                continue
            else:
                close_table()

            # Headers
            if trimmed.startswith("### "):
                close_list()
                html_blocks.append(f"<h3>{html.escape(trimmed[4:].strip())}</h3>")
            elif trimmed.startswith("## "):
                close_list()
                html_blocks.append(f"<h2>{html.escape(trimmed[3:].strip())}</h2>")
            elif trimmed.startswith("# "):
                close_list()
                html_blocks.append(f"<h1>{html.escape(trimmed[2:].strip())}</h1>")
            # Bullet list
            elif re.match(r"^[-*•]\s+", trimmed):
                if not in_list or list_type != "ul":
                    close_list()
                    html_blocks.append('<ul class="list-disc ml-4 space-y-1">')
                    in_list = True
                    list_type = "ul"
                content = re.sub(r"^[-*•]\s+", "", trimmed)
                html_blocks.append(f"<li>{self._format_inline_markdown(content)}</li>")
            # Ordered list
            elif re.match(r"^\d+\.\s+", trimmed):
                if not in_list or list_type != "ol":
                    close_list()
                    html_blocks.append('<ol class="list-decimal ml-4 space-y-1">')
                    in_list = True
                    list_type = "ol"
                content = re.sub(r"^\d+\.\s+", "", trimmed)
                html_blocks.append(f"<li>{self._format_inline_markdown(content)}</li>")
            # Blockquote
            elif trimmed.startswith(">"):
                close_list()
                quote_text = trimmed.lstrip(">").strip()
                html_blocks.append(f'<blockquote class="border-l-4 border-primary pl-4 italic my-4">{self._format_inline_markdown(quote_text)}</blockquote>')
            # Paragraph
            else:
                close_list()
                html_blocks.append(f"<p>{self._format_inline_markdown(trimmed)}</p>")

        close_list()
        close_table()

        return "\n".join(html_blocks)

    def _format_inline_markdown(self, text: str) -> str:
        """Format bold, italics, code, and links in inline markdown."""
        # Bold
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)
        # Italic
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
        text = re.sub(r"_(.+?)_", r"<em>\1</em>", text)
        # Inline code
        text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
        # Links
        text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>', text)
        return text

    def extract_citations(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract citations and references from the article row (JSON/metadata) or raw text.
        Returns a list of standardized citation dicts:
        [{"title": ..., "url": ..., "source_type": "web", "author": "", "publication_date": ""}]
        """
        citations: List[Dict[str, Any]] = []
        seen_urls = set()

        def add_citation(title: str, url: str, author: str = "", pub_date: str = "", source_type: str = "web"):
            url_clean = (url or "").strip()
            if not url_clean or url_clean == "#":
                return
            if url_clean in seen_urls:
                return
            seen_urls.add(url_clean)

            title_clean = clean_citation_numbers(title or "")
            if not title_clean or title_clean.lower() in ("#", "source", "link", "url", "reference", "references"):
                # Derive title from URL domain or path
                domain_m = re.search(r"https?://(?:www\.)?([^/]+)", url_clean)
                title_clean = domain_m.group(1).title() if domain_m else "Source"

            citations.append({
                "title": title_clean,
                "url": url_clean,
                "source_type": source_type or "web",
                "author": (author or "").strip(),
                "publication_date": (pub_date or "").strip(),
            })

        raw_data = article.get("raw_data") or article
        meta = raw_data.get("metadata") if isinstance(raw_data.get("metadata"), dict) else {}
        art_meta = raw_data.get("article_metadata") if isinstance(raw_data.get("article_metadata"), dict) else {}

        # 1. Check if citations/references already exist in the database row
        raw_citations = (
            raw_data.get("citations")
            or raw_data.get("references")
            or raw_data.get("sources")
            or raw_data.get("bibliography")
            or meta.get("citations")
            or meta.get("references")
            or meta.get("sources")
            or art_meta.get("citations")
            or art_meta.get("references")
        )

        if isinstance(raw_citations, str):
            try:
                import json
                raw_citations = json.loads(raw_citations)
            except Exception:
                raw_citations = None

        if isinstance(raw_citations, list) and len(raw_citations) > 0:
            for item in raw_citations:
                if isinstance(item, dict):
                    title = item.get("title") or item.get("source_title") or item.get("name") or item.get("text") or ""
                    url = item.get("url") or item.get("link") or item.get("href") or ""
                    author = item.get("author") or ""
                    pub_date = item.get("publication_date") or item.get("date") or ""
                    add_citation(title, url, author, pub_date)
                elif isinstance(item, str):
                    url_m = re.search(r"https?://[^\s)]+", item)
                    if url_m:
                        url = url_m.group(0)
                        title = item.replace(url, "").strip(" -:()[]\"'")
                        add_citation(title, url)

        # 2. Extract from raw content text / markdown / HTML
        content = article.get("content") or raw_data.get("content") or ""
        if content:
            # Check for dedicated References / Sources section with flexible heading matching
            ref_section_match = re.search(
                r"(?:^|\n)(?:#{1,4}\s*(?:References|Sources|Citations|Bibliography|Works\s+Cited)[^\n]*|<h[1-6]>[^<]*(?:References|Sources|Citations|Bibliography|Works\s+Cited)[^<]*</h[1-6]>|<strong>\s*(?:References|Sources|Citations|Bibliography)[^<]*</strong>)[\s\S]*$",
                content,
                re.IGNORECASE
            )

            scope = ref_section_match.group(0) if ref_section_match else content

            # Pattern A: Footnotes: [^1]: [Title](URL) or [1]: Title URL or [1] Title: URL
            footnote_pat = r"\[\^?(\d+)\]:?\s*(?:\[([^\]]+)\]\((https?://[^)]+)\)|(?:<strong>)?([^<\n]+?)(?:</strong>)?\s*[-:—–]?\s*(https?://[^\s<)\"']+)|([^\n]+))"
            for match in re.finditer(footnote_pat, scope):
                g = match.groups()
                # Case A1: [Title](URL)
                if g[1] and g[2]:
                    add_citation(g[1], g[2])
                # Case A2: Title - URL
                elif g[3] and g[4]:
                    add_citation(g[3], g[4])
                # Case A3: plain line
                elif g[5]:
                    plain = g[5].strip()
                    url_m = re.search(r"https?://[^\s<)\"']+", plain)
                    if url_m:
                        url = url_m.group(0)
                        title = plain.replace(url, "").strip(" -:—–()[]\"'<>")
                        add_citation(title, url)

            # Pattern B: Numbered / bulleted markdown links: 1. [Title](URL) or - [Title](URL)
            link_pat = r"(?:^|\n)[*\-•\d.]*\s*\[([^\]]+)\]\((https?://[^)]+)\)"
            for match in re.finditer(link_pat, scope):
                add_citation(match.group(1), match.group(2))

            # Pattern C: Numbered / bulleted plain URLs: 1. Title - https://... or 1. https://...
            plain_pat = r"(?:^|\n)[*\-•\d.]+\s*(?:<strong>)?([^<\n]+?)(?:</strong>)?\s*[-:—–]\s*(https?://[^\s<)\"']+)"
            for match in re.finditer(plain_pat, scope):
                add_citation(match.group(1), match.group(2))

            # Pattern D: HTML links in scope: <a href="URL">Title</a>
            html_link_pat = r'<a\s+[^>]*href=["\'](https?://[^"\']+)["\'][^>]*>(.*?)</a>'
            for match in re.finditer(html_link_pat, scope, re.IGNORECASE):
                url = match.group(1)
                raw_title = re.sub(r"<[^>]+>", " ", match.group(2)).strip()
                add_citation(raw_title, url)

            # Pattern E: If still no citations, extract ALL markdown links from the whole content
            if not citations:
                for match in re.finditer(r"\[([^\]]+)\]\((https?://[^)]+)\)", content):
                    add_citation(match.group(1), match.group(2))

        return citations

    def extract_citations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract citations and references from text."""
        return self.extract_citations({"content": text})

    def synthesize_metadata(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Extract or synthesize Hook, Thesis, Deck, and TL;DR from article content, without citation markers."""
        content = article.get("content", "")
        summary = article.get("summary", "")
        hook = article.get("hook", "")
        thesis = article.get("thesis", "")

        # Remove explicit References section before synthesizing hook/thesis/deck
        content_without_refs = re.sub(
            r"(?:^|\n)(?:#{1,4}\s*(?:References|Sources|Citations|Bibliography)|<strong>\s*(?:References|Sources)\s*</strong>)[\s\S]*$",
            "",
            content,
            flags=re.IGNORECASE
        )

        # Plain text extraction with citation numbers removed
        plain = clean_citation_numbers(re.sub(r"<[^>]+>", " ", content_without_refs))
        sentences = [
            clean_citation_numbers(s.strip())
            for s in re.split(r"(?<=[.!?])\s+", plain)
            if len(clean_citation_numbers(s.strip())) > 25
        ]

        # 1. Hook
        hook = clean_citation_numbers(hook)
        if not hook and sentences:
            hook = sentences[0]

        # 2. Thesis
        thesis = clean_citation_numbers(thesis)
        if not thesis and len(sentences) > 1:
            thesis = sentences[1]
        elif not thesis and sentences:
            thesis = sentences[0]

        # 3. Deck / TL;DR
        deck = clean_citation_numbers(summary)
        if not deck and sentences:
            deck = " ".join(sentences[:2])

        # 4. TL;DR Takeaways
        # 4. TL;DR Takeaways
        takeaways: List[str] = []

        # 4a. Check explicit takeaways from normalized article / raw data
        raw_takeaways = article.get("takeaways") or (article.get("raw_data") or {}).get("takeaways")
        if isinstance(raw_takeaways, list) and raw_takeaways:
            takeaways = [clean_citation_numbers(str(t)) for t in raw_takeaways if len(clean_citation_numbers(str(t))) > 15]

        # 4b. Check explicit "At a glance" / "Key Takeaways" / "TL;DR" section in content
        if not takeaways:
            section_m = re.search(
                r"(?:^|\n)(?:#{1,4}\s*(?:At\s+a\s+glance|Key\s+Takeaways|TL;?DR|Takeaways|Executive\s+Summary)[^\n]*|<h[1-6]>[^<]*(?:At\s+a\s+glance|Key\s+Takeaways|TL;?DR)[^<]*</h[1-6]>)\s*\n([\s\S]*?)(?=(?:^|\n)#{1,4}\s|\Z)",
                content_without_refs,
                re.IGNORECASE
            )
            if section_m:
                sec_text = section_m.group(1).strip()
                sec_bullets = re.findall(r"^[*\-•\d.]*\s*(.+)$", sec_text, re.MULTILINE)
                if sec_bullets:
                    takeaways = [clean_citation_numbers(b) for b in sec_bullets if len(clean_citation_numbers(b)) > 20]

        # 4c. Check bullet points across content
        if not takeaways:
            bullet_matches = re.findall(r"^[*\-•]\s+(.+)$", content_without_refs, re.MULTILINE)
            if bullet_matches:
                for b in bullet_matches[:4]:
                    cleaned_b = clean_citation_numbers(b)
                    if len(cleaned_b) > 25 and not re.search(r"^(references|sources|bibliography)", cleaned_b, re.IGNORECASE):
                        takeaways.append(cleaned_b)

        # 4d. Check trailing distinct takeaway block (common in Editorial Factory where 2-4 takeaways sit at the end)
        if not takeaways:
            raw_blocks = [b.strip() for b in re.split(r"\n\s*\n", content_without_refs.strip()) if b.strip()]
            if len(raw_blocks) >= 4:
                last_3 = raw_blocks[-3:]
                if all(25 < len(clean_citation_numbers(b)) < 350 and not b.startswith("#") for b in last_3):
                    takeaways = [clean_citation_numbers(b) for b in last_3]

        # 4e. Fallback to top sentences
        if not takeaways and len(sentences) >= 3:
            takeaways = sentences[1:4]

        # 5. Keywords
        tags = article.get("tags", [])
        primary_kw = tags[0] if tags else ""
        secondary_kws = tags[1:] if len(tags) > 1 else []

        return {
            "hook": clean_citation_numbers(hook),
            "thesis": clean_citation_numbers(thesis),
            "deck": clean_citation_numbers(deck),
            "takeaways": [clean_citation_numbers(t) for t in takeaways],
            "primary_keyword": primary_kw,
            "secondary_keywords": secondary_kws,
        }

    def remove_duplicate_takeaways_from_body(self, html_content: str, takeaways: List[str]) -> str:
        """
        Remove any trailing or standalone paragraphs/blocks from the body HTML that duplicate
        the key takeaways / At a glance items.
        """
        if not html_content or not takeaways:
            return html_content

        def _simplify(s: str) -> str:
            cleaned = clean_citation_numbers(re.sub(r"<[^>]+>", " ", s or ""))
            return re.sub(r"[^a-zA-Z0-9]+", "", cleaned).lower()

        simplified_takeaways = [_simplify(t) for t in takeaways if len(_simplify(t)) > 20]
        if not simplified_takeaways:
            return html_content

        # Remove any existing trailing or heading blocks for "At a glance" / "Key Takeaways" / "TL;DR"
        html_content = re.sub(
            r"<(?:h[1-6]|p|div|section)[^>]*>\s*(?:<strong>)?\s*(?:At\s+a\s+glance|Key\s+Takeaways|TL;?DR|Takeaways|Executive\s+Summary)\s*(?:</strong>)?\s*</(?:h[1-6]|p|div|section)>",
            "",
            html_content,
            flags=re.IGNORECASE
        )

        # Split into HTML blocks (<p>...</p>, <li>...</li>, <blockquote>...</blockquote>, etc.)
        block_pattern = r"(<(?:p|li|blockquote|div)\b[^>]*>[\s\S]*?</(?:p|li|blockquote|div)>)"
        parts = re.split(block_pattern, html_content, flags=re.IGNORECASE)

        filtered_parts = []
        for part in parts:
            if not part:
                continue
            if re.match(r"^<(?:p|li|blockquote|div)\b", part.strip(), re.IGNORECASE):
                simplified_part = _simplify(part)
                is_duplicate = False
                if len(simplified_part) > 20:
                    for simp_t in simplified_takeaways:
                        if simp_t == simplified_part or simp_t in simplified_part or simplified_part in simp_t:
                            is_duplicate = True
                            break
                        if len(simp_t) > 30 and len(simplified_part) > 30:
                            if simp_t[:40] in simplified_part or simplified_part[:40] in simp_t:
                                is_duplicate = True
                                break
                if not is_duplicate:
                    filtered_parts.append(part)
            else:
                filtered_parts.append(part)

        cleaned_html = "".join(filtered_parts)
        cleaned_html = re.sub(r"(?:<p>\s*</p>\s*)+", "", cleaned_html)
        return cleaned_html.strip()

    def inject_key_takeaways_html(self, html_content: str, takeaways: List[str]) -> str:
        """Inject structured At a glance section after the first header/paragraph."""
        if not takeaways or "geo-key-takeaways" in html_content or "<h2>At a glance</h2>" in html_content or "<h2>Key Takeaways</h2>" in html_content or "<h2>TL;DR</h2>" in html_content:
            return html_content

        takeaways_items = "".join(f"<li>{_render_inline_markdown(t)}</li>" for t in takeaways)
        takeaways_section = f"""
<section class="geo-key-takeaways" data-geo-injected="key-takeaways">
  <h2>At a glance</h2>
  <ul>
    {takeaways_items}
  </ul>
</section>
""".strip()

        # Insert after <h1> or after first <p>
        if "</h1>" in html_content:
            parts = html_content.split("</h1>", 1)
            return f"{parts[0]}</h1>\n\n{takeaways_section}\n\n{parts[1]}"
        elif "</h2>" in html_content:
            parts = html_content.split("</h2>", 1)
            return f"{parts[0]}</h2>\n\n{takeaways_section}\n\n{parts[1]}"

        return f"{takeaways_section}\n\n{html_content}"

    def import_article_to_titles(
        self,
        article_id: str,
        user_id: str,
        target_domain: Optional[str] = None,
        target_category_id: Optional[int] = None,
        target_parent_category_id: Optional[int] = None,
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Fetch article from Editorial Factory, transform it, and insert into the local Titles table.
        Returns (success, title_id, title_data).
        """
        article = self.get_article(article_id)
        if not article:
            logger.error(f"[EditorialFactoryService] Article {article_id} not found in Editorial Factory.")
            return False, None, {"error": f"Article {article_id} not found."}

        # Transform content
        raw_content = article.get("content", "")
        citations = self.extract_citations(article)
        metadata = self.synthesize_metadata(article)
        takeaways = metadata.get("takeaways", [])

        # Convert markdown to html
        html_body = self.markdown_to_html(raw_content)

        # Remove duplicate takeaway paragraphs from body HTML
        if takeaways:
            html_body = self.remove_duplicate_takeaways_from_body(html_body, takeaways)

        # Inject Key Takeaways if available
        if takeaways:
            html_body = self.inject_key_takeaways_html(html_body, takeaways)

        # If citations exist and html_body doesn't have an HTML References section, append standard References HTML
        if citations and not re.search(r"<h[1-6]>[^<]*References</h[1-6]>", html_body, re.IGNORECASE):
            ref_rows = []
            for i, c in enumerate(citations, 1):
                t_str = html.escape(c.get("title") or "Source")
                u_str = html.escape(c.get("url") or "#")
                author_str = f"{html.escape(c['author'])}. " if c.get("author") else ""
                date_str = f"({html.escape(c['publication_date'])}) " if c.get("publication_date") else ""
                ref_rows.append(f'<p><strong>[{i}]</strong> {author_str}{date_str}<a href="{u_str}" target="_blank" rel="noopener noreferrer" style="color: hsl(var(--primary)); text-decoration: underline;">{t_str}</a>.</p>')
            references_html = "\n\n<hr>\n\n<h2>References</h2>\n\n" + "\n".join(ref_rows)
            html_body += references_html

        plain_text = re.sub(r"<[^>]+>", " ", html_body).strip()
        now_iso = datetime.utcnow().isoformat()

        # Build payload for local Titles table
        title_payload: Dict[str, Any] = {
            "user_id": user_id,
            "Title": article.get("title", "Untitled Editorial Article"),
            "htmlArticle": html_body,
            "articleText": plain_text,
            "hook": metadata.get("hook") or article.get("hook", ""),
            "thesis": metadata.get("thesis") or article.get("thesis", ""),
            "deck": metadata.get("deck") or article.get("summary", ""),
            "userDescription": metadata.get("deck") or article.get("summary", ""),
            "Keywords": ", ".join(article.get("tags", [])) if article.get("tags") else "",
            "primary_keyword": metadata.get("primary_keyword") or None,
            "secondary_keywords_json": metadata.get("secondary_keywords") or [],
            "status": "Editing",
            "dateCreatedOn": now_iso,
            "domain": target_domain or None,
            "wordpress_category_id": target_category_id or None,
            "wordpress_parent_category_id": target_parent_category_id or None,
            "citations": citations if citations else [],
            "idea_metadata": {
                "source": "editorial-factory",
                "editorial_factory_id": article_id,
                "imported_at": now_iso,
                "author": article.get("author", "Editorial Factory"),
                "key_takeaways": metadata.get("takeaways", []),
            }
        }

        local_supabase = get_supabase_client()
        if not local_supabase:
            logger.error("[EditorialFactoryService] Local Supabase client not available.")
            return False, None, {"error": "Local database client unavailable."}

        try:
            res = local_supabase.table("Titles").insert(title_payload).execute()
            inserted_rows = res.data or []
            if inserted_rows:
                new_title_id = inserted_rows[0].get("id")
                logger.info(f"[EditorialFactoryService] Successfully imported article {article_id} as Title {new_title_id}")
                return True, new_title_id, inserted_rows[0]
            return False, None, {"error": "Insert succeeded but no row returned."}
        except Exception as insert_err:
            logger.error(f"[EditorialFactoryService] Failed to insert Titles row: {insert_err}", exc_info=True)
            return False, None, {"error": str(insert_err)}


editorial_factory_service = EditorialFactoryService()
