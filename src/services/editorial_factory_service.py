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

EDITORIAL_SUPABASE_PROJECT_REF = os.getenv("EDITORIAL_SUPABASE_PROJECT_REF", "ixfdkninqeqmwuxncpvh")
EDITORIAL_SUPABASE_URL = os.getenv(
    "EDITORIAL_SUPABASE_URL",
    f"https://{EDITORIAL_SUPABASE_PROJECT_REF}.supabase.co"
)
EDITORIAL_SUPABASE_KEY = os.getenv(
    "EDITORIAL_SUPABASE_KEY",
    os.getenv("EDITORIAL_SUPABASE_ANON_KEY", os.getenv("SUPABASE_KEY", ""))
)


class EditorialFactoryService:
    """Service to interact with the Editorial Factory Supabase database."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None
    ):
        self.supabase_url = supabase_url or EDITORIAL_SUPABASE_URL
        self.supabase_key = supabase_key or EDITORIAL_SUPABASE_KEY
        self._client: Optional[Client] = None

    def get_client(self) -> Optional[Client]:
        """Lazy-initialize and return the Supabase client for Editorial Factory."""
        if not self._client and self.supabase_url and self.supabase_key:
            try:
                self._client = create_client(self.supabase_url, self.supabase_key)
            except Exception as err:
                logger.warning(f"[EditorialFactoryService] Could not initialize Supabase client: {err}")
                self._client = None
        return self._client

    def list_articles(
        self,
        search: str = "",
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List articles from the Editorial Factory 'articles' table.
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

        # Normalize article structures
        normalized: List[Dict[str, Any]] = []
        for art in articles:
            norm = self._normalize_article(art)
            if norm:
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
                logger.warning(f"[EditorialFactoryService] Fetch article by id failed: {err}")

        if self.supabase_url and self.supabase_key:
            try:
                endpoint = f"{self.supabase_url}/rest/v1/articles"
                headers = {
                    "apikey": self.supabase_key,
                    "Authorization": f"Bearer {self.supabase_key}",
                }
                resp = requests.get(
                    endpoint,
                    headers=headers,
                    params={"id": f"eq.{article_id}", "select": "*"},
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

        # Calculate word count
        words = len(re.findall(r"\w+", content)) if content else 0

        return {
            "id": raw_id,
            "title": title.strip(),
            "content": content,
            "summary": summary.strip(),
            "hook": hook.strip(),
            "thesis": thesis.strip(),
            "tags": tags if isinstance(tags, list) else [str(tags)],
            "created_at": created_at,
            "author": author,
            "word_count": words,
            "raw_data": row,
        }

    def markdown_to_html(self, text: str) -> str:
        """Convert Markdown content to clean HTML structure for the editor."""
        if not text:
            return ""

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

    def extract_citations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract citations and references from text or bibliography sections."""
        citations: List[Dict[str, Any]] = []
        if not text:
            return citations

        # Match markdown footnotes or reference items like [1] URL or [^1]: Title URL
        ref_patterns = [
            r"\[\^?(\d+)\]:?\s*(?:\[([^\]]+)\]\(([^)]+)\)|([^\n]+))",
            r"(?:^|\n)\[(\d+)\]\s*(.+)",
        ]

        seen_urls = set()
        for pat in ref_patterns:
            for match in re.finditer(pat, text):
                groups = match.groups()
                title = groups[1] or groups[3] or groups[0] or "Reference Source"
                url = groups[2] if len(groups) > 2 and groups[2] else "#"

                # If URL found inside title
                url_match = re.search(r"https?://[^\s)]+", title)
                if url_match and url == "#":
                    url = url_match.group(0)
                    title = title.replace(url, "").strip(" -:()")

                if url not in seen_urls:
                    seen_urls.add(url)
                    citations.append({
                        "title": title.strip() or "Source",
                        "url": url,
                        "source_type": "web",
                        "author": "",
                        "publication_date": "",
                    })

        return citations

    def synthesize_metadata(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Extract or synthesize Hook, Thesis, Deck, and Key Takeaways from article content."""
        content = article.get("content", "")
        summary = article.get("summary", "")
        hook = article.get("hook", "")
        thesis = article.get("thesis", "")

        # Plain text extraction
        plain = re.sub(r"<[^>]+>", " ", content)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", plain) if len(s.strip()) > 20]

        # 1. Hook
        if not hook and sentences:
            hook = sentences[0]

        # 2. Thesis
        if not thesis and len(sentences) > 1:
            thesis = sentences[1]
        elif not thesis and sentences:
            thesis = sentences[0]

        # 3. Deck / TL;DR
        deck = summary
        if not deck and sentences:
            deck = " ".join(sentences[:2])

        # 4. Key Takeaways
        takeaways: List[str] = []
        # Check if content has bullet points
        bullet_matches = re.findall(r"^[*\-•]\s+(.+)$", content, re.MULTILINE)
        if bullet_matches:
            takeaways = [b.strip() for b in bullet_matches[:4] if len(b.strip()) > 25]
        if not takeaways and len(sentences) >= 3:
            takeaways = sentences[1:4]

        # 5. Keywords
        tags = article.get("tags", [])
        primary_kw = tags[0] if tags else ""
        secondary_kws = tags[1:] if len(tags) > 1 else []

        return {
            "hook": hook,
            "thesis": thesis,
            "deck": deck,
            "takeaways": takeaways,
            "primary_keyword": primary_kw,
            "secondary_keywords": secondary_kws,
        }

    def inject_key_takeaways_html(self, html_content: str, takeaways: List[str]) -> str:
        """Inject structured Key Takeaways GEO section after the first header/paragraph."""
        if not takeaways or "geo-key-takeaways" in html_content:
            return html_content

        takeaways_items = "".join(f"<li>{html.escape(t)}</li>" for t in takeaways)
        takeaways_section = f"""
<section class="geo-key-takeaways" data-geo-injected="key-takeaways">
  <h2>Key Takeaways</h2>
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
        html_body = self.markdown_to_html(raw_content)
        citations = self.extract_citations_from_text(raw_content)
        metadata = self.synthesize_metadata(article)

        # Inject Key Takeaways if available
        if metadata.get("takeaways"):
            html_body = self.inject_key_takeaways_html(html_body, metadata["takeaways"])

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
