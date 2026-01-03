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

@dataclass
class Citation:
    """A single citation."""
    id: str
    source_type: str
    title: str
    author: Optional[str] = None
    url: Optional[str] = None
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
        """Extract citations from evidence sources. Only creates citations from evidence with actual content."""
        citations = []
        
        for ev in evidence:
            try:
                # CRITICAL: Only create citations from evidence with actual content
                # Skip evidence without content to avoid creating mock citations
                content = ev.get('content', '')
                if not content or not content.strip():
                    self.logger.warning(f"Skipping evidence item - no content available. Title: {ev.get('title', 'N/A')[:50]}")
                    continue
                
                # Generate unique citation ID
                self.citation_counter += 1
                citation_id = f"cite_{self.citation_counter:03d}"
                
                # Determine source type
                source_type = self._determine_source_type(ev)
                
                # Extract metadata
                metadata = ev.get('metadata', {})
                
                # Extract title - prefer actual title, fall back to content snippet, but don't use generic placeholders
                title = ev.get('title') or ev.get('source_title', '')
                if not title or title == 'Unknown Source' or title.startswith('Knowledge Base Source') or title.startswith('Web Source'):
                    # Only extract from content if we have real content
                    if content and len(content) > 20:
                        # Use first sentence or meaningful phrase from content
                        first_sentence = content.split('.')[0].strip()
                        if len(first_sentence) > 10:
                            title = first_sentence[:100] + ('...' if len(first_sentence) > 100 else '')
                        else:
                            # Content too short, skip this citation to avoid generic titles
                            self.logger.warning(f"Skipping evidence item - content too short for meaningful title")
                            continue
                    else:
                        # No meaningful content, skip to avoid creating mock citation
                        self.logger.warning(f"Skipping evidence item - cannot create meaningful title from content")
                        continue
                
                # Extract URL - prefer actual URL
                url = ev.get('source') or ev.get('url', '')
                if not url or url == '#' or url == 'Unknown URL':
                    url = ''  # Leave empty rather than using placeholder
                
                # Create citation only if we have valid data
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
                    credibility_score=ev.get('credibility_score', 0.0),
                    relevance_score=ev.get('relevance_score', 0.0),
                    content_excerpt=content[:200] + '...' if len(content) > 200 else content,
                    metadata=metadata
                )
                
                citations.append(citation)
                
            except Exception as e:
                self.logger.error(f"Error extracting citation from evidence: {str(e)}")
                continue
        
        self.logger.info(f"Extracted {len(citations)} citations from {len(evidence)} evidence items (skipped {len(evidence) - len(citations)} items without valid content)")
        return citations
    
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

