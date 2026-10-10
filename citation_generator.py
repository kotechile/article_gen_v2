"""
Citation Generator for Content Generator V2.

This module generates and formats citations for articles based on
evidence sources and content references.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime
from urllib.parse import urlparse

# Configure logging
logger = logging.getLogger(__name__)

class CitationStyle(Enum):
    """Citation styles supported."""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    HARVARD = "harvard"
    IEEE = "ieee"

class SourceType(Enum):
    """Types of sources."""
    WEB = "web"
    RAG = "rag"
    JOURNAL = "journal"
    BOOK = "book"
    REPORT = "report"
    INTERVIEW = "interview"
    VIDEO = "video"
    PODCAST = "podcast"

# Known spam, content farm, scraper, and low-credibility domains to exclude from references
DISALLOWED_SPAM_DOMAINS = {
    # Social networks & user-generated forums (unvetted for professional editorial references)
    "pinterest.com",
    "quora.com",
    "reddit.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    # Content farms, low-quality aggregators, scraper spam
    "ezinearticles.com",
    "ehow.com",
    "hubpages.com",
    "buzzfeed.com",
    "answers.com",
    "ask.com",
    "about.com",
    "softonic.com",
    "examiner.com",
    "slideshare.net",
    "scribd.com",
    "blogspot.com",
    "wordpress.com",
    "tumblr.com",
    "wixsite.com",
    # Search engine redirect or internal URLs
    "google.com",
    "bing.com",
    "yahoo.com",
    "duckduckgo.com",
    # URL shorteners / trackers
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "doubleclick.net",
}

# Path patterns indicative of dead, non-content, or broken pages
BROKEN_OR_JUNK_PATH_PATTERNS = [
    r"/(?:404|403|500|error|not-found|page-not-found)\b",
    r"/(?:login|signin|signup|register|auth|account)\b",
    r"/(?:cart|checkout|basket|order|pricing|subscribe)\b",
    r"/(?:privacy|privacy-policy|terms|terms-of-service|cookie-policy|disclaimer)\b",
    r"/(?:wp-admin|wp-content|wp-includes|admin)\b",
    r"/(?:search|query|tag|category|author)/?$",
]

def clean_citation_url(url: Optional[str]) -> str:
    """
    Strip marketing tracking parameters (utm, fbclid, etc.) and fragments from citation URLs.
    """
    if not url or not isinstance(url, str):
        return ""
    cleaned = url.strip()
    if cleaned in {"#", "unknown", "about:blank", "none"} or cleaned.startswith("javascript:"):
        return ""
    try:
        from urllib.parse import parse_qsl, urlencode, urlunparse
        parsed = urlparse(cleaned if "://" in cleaned else f"https://{cleaned}")
        if parsed.scheme.lower() not in {"http", "https"}:
            return ""
        if not parsed.netloc:
            return ""
        cleaned_query = [
            (k, v) for k, v in parse_qsl(parsed.query)
            if not k.lower().startswith(("utm_", "fbclid", "gclid", "ref_", "mc_", "trk", "source", "session"))
        ]
        new_query = urlencode(cleaned_query)
        result = urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, parsed.params, new_query, ""))
        return result.rstrip("?")
    except Exception:
        return cleaned

def is_valid_citation_url(url: Optional[str]) -> bool:
    """
    Validate that a URL is a legitimate, non-spam, non-broken web reference.
    """
    if not url or not isinstance(url, str):
        return False
    cleaned = url.strip()
    if cleaned in {"#", "", "unknown", "about:blank", "none"} or cleaned.startswith("javascript:"):
        return False
    try:
        parsed = urlparse(cleaned if "://" in cleaned else f"https://{cleaned}")
        if parsed.scheme.lower() not in {"http", "https"}:
            return False
        host = (parsed.hostname or "").lower()
        if not host or "." not in host or host.endswith(".local") or host == "localhost":
            return False
        # Block known spam domains
        for spam_domain in DISALLOWED_SPAM_DOMAINS:
            if host == spam_domain or host.endswith(f".{spam_domain}"):
                return False
        # Block broken/junk paths
        path = (parsed.path or "").lower()
        for pattern in BROKEN_OR_JUNK_PATH_PATTERNS:
            if re.search(pattern, path):
                return False
        return True
    except Exception:
        return False

def is_serious_citation(citation: Any) -> bool:
    """
    Evaluate whether a citation represents a serious, credible, and informative reference.
    Filters out spam, placeholder titles, and dead links.
    """
    if not citation:
        return False

    if isinstance(citation, dict):
        title = (citation.get("title") or citation.get("source_title") or "").strip()
        url = citation.get("url") or citation.get("source") or ""
        source_type = (citation.get("source_type") or "").lower()
        content = (citation.get("content") or citation.get("content_excerpt") or "").strip()
    elif hasattr(citation, "title"):
        title = (getattr(citation, "title", "") or "").strip()
        url = getattr(citation, "url", "") or ""
        source_type = (getattr(citation, "source_type", "") or "").lower()
        content = (getattr(citation, "content_excerpt", "") or "").strip()
    else:
        return False

    # Filter out generic placeholder titles
    generic_titles = {
        "unknown source", "web source", "knowledge base source", "untitled",
        "home", "page not found", "sign in", "login", "register", "error",
        "document reference", "source", "404", "index",
    }
    if not title or title.lower() in generic_titles or title.lower().startswith("knowledge base source"):
        return False

    # Web sources must have a valid, non-spam URL
    if source_type in {"web", "linkup", "search"}:
        if not is_valid_citation_url(url):
            return False
    elif source_type in {"rag", "journal", "book", "report"}:
        if url and not is_valid_citation_url(url) and url not in {"#", ""}:
            return False
        if not content and not title:
            return False

    return True

def filter_and_curate_citations(citations: List[Any], max_citations: int = 8) -> List[Any]:
    """
    Filter citations to only serious, credible, non-spam sources, ranked and capped to max_citations.
    """
    if not citations:
        return []
    
    serious = [c for c in citations if is_serious_citation(c)]
    
    def _score(c):
        if isinstance(c, dict):
            auth = float(c.get("authority_score") or c.get("credibility_score") or 0.0)
            rel = float(c.get("relevance_score") or 0.0)
        else:
            auth = float(getattr(c, "authority_score", 0.0) or getattr(c, "credibility_score", 0.0))
            rel = float(getattr(c, "relevance_score", 0.0))
        return auth + (rel * 0.5)

    serious.sort(key=_score, reverse=True)
    return serious[:max_citations]

@dataclass
class Citation:
    """A single citation."""
    id: str
    source_type: str
    title: str
    author: Optional[str] = None
    url: Optional[str] = None
    domain: Optional[str] = None
    domain_rank: Optional[int] = None
    domain_frequency: int = 0
    authority_score: float = 0.0
    publication_date: Optional[str] = None
    publisher: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None
    accessed_date: Optional[str] = None
    credibility_score: float = 0.0
    relevance_score: float = 0.0
    content_excerpt: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class FormattedCitation:
    """A formatted citation for a specific style."""
    citation_id: str
    style: str
    in_text: str
    reference: str
    footnote: Optional[str] = None

class CitationGenerator:
    """
    Generates and formats citations for articles.
    """
    
    def __init__(self, llm_client, default_style: CitationStyle = CitationStyle.APA):
        """
        Initialize the citation generator.
        
        Args:
            llm_client: Configured LLM client
            default_style: Default citation style
        """
        self.llm_client = llm_client
        self.default_style = default_style
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.citation_counter = 0
    
    def generate_citations(self, evidence: List[Dict], content_sections: List[Dict], 
                          style: CitationStyle = None, include_in_text_citations: bool = True) -> Dict[str, Any]:
        """
        Generate citations from evidence and content.
        
        Args:
            evidence: List of evidence sources
            content_sections: List of content sections
            style: Citation style to use
            include_in_text_citations: Whether to add in-text citation references to content (default: True)
            
        Returns:
            Dictionary with citations and formatted references
        """
        try:
            citation_style = style or self.default_style
            
            # Extract citations from evidence
            citations = self._extract_citations_from_evidence(evidence)
            
            # Process content sections for citation placement (only if in-text citations are enabled)
            if include_in_text_citations:
                processed_sections = self._process_content_for_citations(content_sections, citations)
            else:
                # If in-text citations are disabled, return sections as-is without adding citation markers
                self.logger.info("In-text citations disabled - skipping citation placement in content")
                processed_sections = content_sections
            
            # Format citations according to style
            formatted_citations = self._format_citations(citations, citation_style)
            
            # Generate reference list
            reference_list = self._generate_reference_list(formatted_citations, citation_style)
            
            self.logger.info(f"Generated {len(citations)} citations in {citation_style.value} style")
            
            return {
                'citations': citations,
                'formatted_citations': formatted_citations,
                'reference_list': reference_list,
                'style': citation_style.value,
                'total_citations': len(citations),
                'processed_sections': processed_sections
            }
            
        except Exception as e:
            self.logger.error(f"Error generating citations: {str(e)}")
            return {
                'citations': [],
                'formatted_citations': [],
                'reference_list': [],
                'style': citation_style.value if citation_style else self.default_style.value,
                'total_citations': 0,
                'error': str(e)
            }
    
    def _extract_citations_from_evidence(self, evidence: List[Dict]) -> List[Citation]:
        """
        Extract citations from evidence sources.
        Strictly filters out empty content, spam domains, broken URLs, and duplicate sources.
        """
        citations = []
        seen_urls = set()
        seen_titles = set()
        
        for ev in evidence:
            try:
                # 1. Skip evidence without substantive content
                content = ev.get('content', '')
                if not content or not content.strip():
                    continue
                
                # 2. Determine source type and metadata
                source_type = self._determine_source_type(ev)
                metadata = ev.get('metadata', {})
                
                # 3. Clean and validate URL
                raw_url = ev.get('source') or ev.get('url', '') or metadata.get('url', '')
                url = clean_citation_url(raw_url)
                
                # If web source and URL is invalid or spam, skip entirely
                if source_type == SourceType.WEB:
                    if not is_valid_citation_url(url):
                        continue
                
                # 4. Extract clean, non-generic title
                title = ev.get('title') or ev.get('source_title', '') or metadata.get('title', '')
                title = str(title).strip()
                if not title or title.lower() in {'unknown source', 'web source', 'untitled', 'home', 'page not found'} or title.lower().startswith('knowledge base source'):
                    if content and len(content) > 20:
                        first_sentence = content.split('.')[0].strip()
                        if len(first_sentence) > 10:
                            title = first_sentence[:100] + ('...' if len(first_sentence) > 100 else '')
                        else:
                            continue
                    else:
                        continue
                
                # 5. Deduplicate by URL and title
                norm_url = url.lower().rstrip('/') if url else ''
                norm_title = re.sub(r'[^a-z0-9]', '', title.lower()[:50])
                if norm_url and norm_url in seen_urls:
                    continue
                if norm_title and norm_title in seen_titles:
                    continue
                if norm_url:
                    seen_urls.add(norm_url)
                if norm_title:
                    seen_titles.add(norm_title)
                
                # Generate unique citation ID
                self.citation_counter += 1
                citation_id = f"cite_{self.citation_counter:03d}"
                
                # Create citation
                citation = Citation(
                    id=citation_id,
                    source_type=source_type.value,
                    title=title,
                    author=metadata.get('author', ''),
                    url=url,
                    publication_date=metadata.get('published_date', metadata.get('datePublished', '')),
                    publisher=metadata.get('publisher', ''),
                    journal=metadata.get('journal', ''),
                    volume=metadata.get('volume', ''),
                    issue=metadata.get('issue', ''),
                    pages=metadata.get('pages', ''),
                    doi=metadata.get('doi', ''),
                    isbn=metadata.get('isbn', ''),
                    accessed_date=datetime.now().strftime('%Y-%m-%d'),
                    credibility_score=float(ev.get('credibility_score') or 0.0),
                    relevance_score=float(ev.get('relevance_score') or 0.0),
                    content_excerpt=content[:200] + '...' if len(content) > 200 else content,
                    metadata=metadata
                )
                
                if is_serious_citation(citation):
                    citations.append(citation)
                
            except Exception as e:
                self.logger.error(f"Error extracting citation from evidence: {str(e)}")
                continue
        
        self._apply_domain_authority_ranking(citations)
        self.logger.info(f"Extracted {len(citations)} serious citations from {len(evidence)} evidence items (filtered out {len(evidence) - len(citations)} spam/empty/broken items)")
        return citations

    def filter_and_curate_citations(self, citations: List[Any], max_citations: int = 8) -> List[Any]:
        """
        Filter citations to only serious, credible, non-spam sources, ranked and capped to max_citations.
        """
        return filter_and_curate_citations(citations, max_citations=max_citations)
    

    def _extract_root_domain(self, url: str) -> str:
        """Extract a normalized root domain from a URL."""
        if not url:
            return "unknown"

        multipart_suffixes = {
            "co.uk", "org.uk", "gov.uk", "ac.uk", "co.jp", "com.au", "net.au", "org.au", "co.nz", "com.br"
        }

        try:
            parsed = urlparse(url if re.match(r"^https?://", url, re.IGNORECASE) else f"https://{url}")
            host = (parsed.hostname or "").lower()
        except Exception:
            return "unknown"

        if not host:
            return "unknown"
        if host.startswith("www."):
            host = host[4:]

        parts = [part for part in host.split(".") if part]
        if len(parts) <= 2:
            return host

        last_two = ".".join(parts[-2:])
        if last_two in multipart_suffixes and len(parts) >= 3:
            return f"{parts[-3]}.{last_two}"

        return last_two

    def _apply_domain_authority_ranking(self, citations: List[Citation]) -> None:
        """Annotate citations with domain-based authority metrics."""
        if not citations:
            return

        domain_counts: Dict[str, int] = {}
        for citation in citations:
            domain = self._extract_root_domain(citation.url or "")
            citation.domain = domain
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        ranked_domains = sorted(domain_counts.items(), key=lambda item: (-item[1], item[0]))
        rank_lookup = {domain: idx + 1 for idx, (domain, _count) in enumerate(ranked_domains)}
        max_frequency = max(domain_counts.values()) if domain_counts else 1

        for citation in citations:
            frequency = domain_counts.get(citation.domain or "unknown", 0)
            citation.domain_frequency = frequency
            citation.domain_rank = rank_lookup.get(citation.domain or "unknown", len(ranked_domains))
            citation.authority_score = round(frequency / max_frequency, 4) if max_frequency else 0.0
            citation.metadata = citation.metadata or {}
            citation.metadata.update({
                "domain": citation.domain,
                "domain_rank": citation.domain_rank,
                "domain_frequency": citation.domain_frequency,
                "authority_score": citation.authority_score,
                "total_unique_domains": len(domain_counts),
            })

    def _determine_source_type(self, evidence: Dict[str, Any]) -> SourceType:
        """Determine the type of source based on evidence metadata."""
        source = evidence.get('source', '').lower()
        metadata = evidence.get('metadata', {})
        title = metadata.get('title', '').lower()
        
        # Check for specific indicators
        if any(indicator in source for indicator in ['.edu', '.gov', '.org']):
            return SourceType.JOURNAL
        elif any(indicator in source for indicator in ['youtube.com', 'vimeo.com']):
            return SourceType.VIDEO
        elif any(indicator in source for indicator in ['spotify.com', 'soundcloud.com']):
            return SourceType.PODCAST
        elif evidence.get('source_type') == 'rag':
            # For RAG sources, check if it's a book based on title patterns
            if any(indicator in title for indicator in ['book', 'guide', 'manual', 'handbook', 'for dummies', 'essential guide']):
                return SourceType.BOOK
            elif any(indicator in source for indicator in ['amazon.com', 'books.google']):
                return SourceType.BOOK
            else:
                return SourceType.RAG
        elif any(indicator in metadata.get('type', '').lower() for indicator in ['book', 'monograph']):
            return SourceType.BOOK
        elif any(indicator in metadata.get('type', '').lower() for indicator in ['report', 'white paper']):
            return SourceType.REPORT
        else:
            return SourceType.WEB
    
    def _process_content_for_citations(self, content_sections: List[Dict], 
                                     citations: List[Citation]) -> List[Dict]:
        """Process content sections to add citation references."""
        processed_sections = []
        
        for section in content_sections:
            try:
                # Process each content block in the section
                processed_blocks = []
                for block in section.get('content_blocks', []):
                    content = block.get('content', '')
                    
                    # Find relevant citations for this content
                    relevant_citations = self._find_relevant_citations(content, citations)
                    
                    # Add citation references to content
                    cited_content = self._add_citation_references(content, relevant_citations)
                    
                    # Update block with cited content
                    processed_block = block.copy()
                    processed_block['content'] = cited_content
                    processed_block['citations'] = relevant_citations
                    
                    processed_blocks.append(processed_block)
                
                # Update section with processed blocks
                processed_section = section.copy()
                processed_section['content_blocks'] = processed_blocks
                processed_sections.append(processed_section)
                
            except Exception as e:
                self.logger.error(f"Error processing section for citations: {str(e)}")
                processed_sections.append(section)
        
        return processed_sections
    
    def _find_relevant_citations(self, content: str, citations: List[Citation]) -> List[Citation]:
        """Find citations relevant to the content."""
        relevant_citations = []
        content_lower = content.lower()
        
        for citation in citations:
            # Check if citation content appears in the text
            if citation.content_excerpt and citation.content_excerpt.lower() in content_lower:
                relevant_citations.append(citation)
            # Check for keyword overlap
            elif self._has_keyword_overlap(content, citation):
                relevant_citations.append(citation)
        
        return relevant_citations[:3]  # Limit to top 3 relevant citations
    
    def _has_keyword_overlap(self, content: str, citation: Citation) -> bool:
        """Check if citation has keyword overlap with content."""
        content_words = set(content.lower().split())
        citation_words = set()
        
        # Add words from title
        if citation.title:
            citation_words.update(citation.title.lower().split())
        
        # Add words from content excerpt
        if citation.content_excerpt:
            citation_words.update(citation.content_excerpt.lower().split())
        
        # Check for overlap
        overlap = content_words.intersection(citation_words)
        return len(overlap) >= 2  # At least 2 words overlap
    
    def _add_citation_references(self, content: str, citations: List[Citation]) -> str:
        """Add citation references to content."""
        if not citations:
            return content
        
        # Add citation references at the end of relevant sentences
        cited_content = content
        
        for i, citation in enumerate(citations, 1):
            # Find a good place to insert the citation
            sentences = cited_content.split('. ')
            if len(sentences) > 1:
                # Insert citation after the first sentence
                sentences[0] += f" [{citation.id}]"
                cited_content = '. '.join(sentences)
            else:
                # Add citation at the end
                cited_content += f" [{citation.id}]"
        
        return cited_content
    
    def _format_citations(self, citations: List[Citation], style: CitationStyle) -> List[FormattedCitation]:
        """Format citations according to the specified style."""
        formatted_citations = []
        
        for citation in citations:
            try:
                if style == CitationStyle.APA:
                    formatted = self._format_apa_citation(citation)
                elif style == CitationStyle.MLA:
                    formatted = self._format_mla_citation(citation)
                elif style == CitationStyle.CHICAGO:
                    formatted = self._format_chicago_citation(citation)
                elif style == CitationStyle.HARVARD:
                    formatted = self._format_harvard_citation(citation)
                elif style == CitationStyle.IEEE:
                    formatted = self._format_ieee_citation(citation)
                else:
                    formatted = self._format_apa_citation(citation)  # Default to APA
                
                formatted_citations.append(formatted)
                
            except Exception as e:
                self.logger.error(f"Error formatting citation {citation.id}: {str(e)}")
                continue
        
        return formatted_citations
    
    def _format_apa_citation(self, citation: Citation) -> FormattedCitation:
        """Format citation in APA style."""
        # In-text citation
        if citation.author:
            in_text = f"({citation.author}, {citation.publication_date or 'n.d.'})"
        else:
            in_text = f"({citation.title[:50]}..., {citation.publication_date or 'n.d.'})"
        
        # Reference entry
        if citation.source_type == SourceType.WEB.value:
            reference = f"{citation.author or 'Unknown Author'}. ({citation.publication_date or 'n.d.'}). {citation.title}. Retrieved from {citation.url}"
        elif citation.source_type == SourceType.JOURNAL.value:
            reference = f"{citation.author or 'Unknown Author'}. ({citation.publication_date or 'n.d.'}). {citation.title}. {citation.journal or 'Journal'}, {citation.volume or '1'}({citation.issue or '1'}), {citation.pages or '1-10'}. doi: {citation.doi or 'N/A'}"
        else:
            reference = f"{citation.author or 'Unknown Author'}. ({citation.publication_date or 'n.d.'}). {citation.title}. {citation.publisher or 'Publisher'}"
        
        return FormattedCitation(
            citation_id=citation.id,
            style="apa",
            in_text=in_text,
            reference=reference
        )
    
    def _format_mla_citation(self, citation: Citation) -> FormattedCitation:
        """Format citation in MLA style."""
        # In-text citation
        if citation.author:
            in_text = f"({citation.author} {citation.publication_date or 'n.d.'})"
        else:
            in_text = f"({citation.title[:50]}...)"
        
        # Works Cited entry
        if citation.source_type == SourceType.WEB.value:
            reference = f'"{citation.title}." {citation.publisher or "Web"}. {citation.publication_date or "n.d."}. Web. {citation.accessed_date}.'
        else:
            reference = f"{citation.author or 'Unknown Author'}. \"{citation.title}.\" {citation.journal or citation.publisher or 'Publication'}. {citation.publication_date or 'n.d.'}. Print."
        
        return FormattedCitation(
            citation_id=citation.id,
            style="mla",
            in_text=in_text,
            reference=reference
        )
    
    def _format_chicago_citation(self, citation: Citation) -> FormattedCitation:
        """Format citation in Chicago style."""
        # In-text citation
        in_text = f"({citation.author or 'Unknown Author'}, {citation.publication_date or 'n.d.'})"
        
        # Bibliography entry
        if citation.source_type == SourceType.WEB.value:
            reference = f"{citation.author or 'Unknown Author'}. \"{citation.title}.\" {citation.publisher or 'Website'}. {citation.publication_date or 'n.d.'}. Accessed {citation.accessed_date}. {citation.url}."
        else:
            reference = f"{citation.author or 'Unknown Author'}. \"{citation.title}.\" {citation.journal or citation.publisher or 'Publication'}. {citation.publication_date or 'n.d.'}."
        
        return FormattedCitation(
            citation_id=citation.id,
            style="chicago",
            in_text=in_text,
            reference=reference
        )
    
    def _format_harvard_citation(self, citation: Citation) -> FormattedCitation:
        """Format citation in Harvard style."""
        # In-text citation
        if citation.author:
            in_text = f"({citation.author} {citation.publication_date or 'n.d.'})"
        else:
            in_text = f"({citation.title[:50]}..., {citation.publication_date or 'n.d.'})"
        
        # Reference list entry
        if citation.source_type == SourceType.WEB.value:
            reference = f"{citation.author or 'Unknown Author'} {citation.publication_date or 'n.d.'}, {citation.title}, {citation.publisher or 'Website'}, viewed {citation.accessed_date}, <{citation.url}>."
        else:
            reference = f"{citation.author or 'Unknown Author'} {citation.publication_date or 'n.d.'}, '{citation.title}', {citation.journal or citation.publisher or 'Publication'}, vol. {citation.volume or '1'}, no. {citation.issue or '1'}, pp. {citation.pages or '1-10'}."
        
        return FormattedCitation(
            citation_id=citation.id,
            style="harvard",
            in_text=in_text,
            reference=reference
        )
    
    def _format_ieee_citation(self, citation: Citation) -> FormattedCitation:
        """Format citation in IEEE style."""
        # In-text citation
        in_text = f"[{citation.id}]"
        
        # Reference list entry
        if citation.source_type == SourceType.WEB.value:
            reference = f"[{citation.id}] {citation.author or 'Unknown Author'}, \"{citation.title},\" {citation.publisher or 'Website'}, {citation.publication_date or 'n.d.'}. [Online]. Available: {citation.url}"
        else:
            reference = f"[{citation.id}] {citation.author or 'Unknown Author'}, \"{citation.title},\" {citation.journal or citation.publisher or 'Publication'}, vol. {citation.volume or '1'}, no. {citation.issue or '1'}, pp. {citation.pages or '1-10'}, {citation.publication_date or 'n.d.'}."
        
        return FormattedCitation(
            citation_id=citation.id,
            style="ieee",
            in_text=in_text,
            reference=reference
        )
    
    def _generate_reference_list(self, formatted_citations: List[FormattedCitation], 
                               style: CitationStyle) -> List[str]:
        """Generate a formatted reference list."""
        reference_list = []
        
        # Add header based on style
        if style == CitationStyle.APA:
            reference_list.append("References")
        elif style == CitationStyle.MLA:
            reference_list.append("Works Cited")
        elif style == CitationStyle.CHICAGO:
            reference_list.append("Bibliography")
        elif style == CitationStyle.HARVARD:
            reference_list.append("Reference List")
        elif style == CitationStyle.IEEE:
            reference_list.append("References")
        
        reference_list.append("")  # Empty line
        
        # Add formatted references
        for i, citation in enumerate(formatted_citations, 1):
            reference_list.append(f"{i}. {citation.reference}")
        
        return reference_list

# Factory function
def create_citation_generator(llm_client, default_style: CitationStyle = CitationStyle.APA) -> CitationGenerator:
    """
    Create a citation generator.
    
    Args:
        llm_client: Configured LLM client
        default_style: Default citation style
        
    Returns:
        CitationGenerator instance
    """
    return CitationGenerator(llm_client, default_style)

# Example usage
if __name__ == "__main__":
    # This would be used with a real LLM client
    print("Citation Generator - Ready for integration")

