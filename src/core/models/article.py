"""
Article-related data models and schemas.

This module defines the data structures for article generation,
formatting, and metadata.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class ArticleFormat(str, Enum):
    """Article output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    PLAINTEXT = "plaintext"


class ArticleTone(str, Enum):
    """Article tone options."""
    ACADEMIC = "academic"
    JOURNALISTIC = "journalistic"
    CASUAL = "casual"
    TECHNICAL = "technical"
    PERSUASIVE = "persuasive"


class CitationType(str, Enum):
    """Citation types."""
    WEB = "web"
    ACADEMIC = "academic"
    NEWS = "news"
    BLOG = "blog"
    REPORT = "report"
    OTHER = "other"


class Citation(BaseModel):
    """Citation model."""
    
    id: str = Field(..., description="Unique citation ID")
    title: str = Field(..., description="Source title")
    url: Optional[str] = Field(None, description="Source URL")
    author: Optional[str] = Field(None, description="Author name")
    publication: Optional[str] = Field(None, description="Publication name")
    date: Optional[datetime] = Field(None, description="Publication date")
    citation_type: CitationType = Field(..., description="Citation type")
    credibility_score: float = Field(0.0, ge=0.0, le=1.0, description="Credibility score")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score")
    content: Optional[str] = Field(None, description="Source content excerpt")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ArticleSection(BaseModel):
    """Article section model."""
    
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    word_count: int = Field(..., description="Word count")
    citations: List[str] = Field(default_factory=list, description="Citation IDs used in this section")
    order: int = Field(..., description="Section order")
    
    @validator('word_count')
    def validate_word_count(cls, v):
        """Validate word count is positive."""
        if v < 0:
            raise ValueError('Word count must be non-negative')
        return v


class ArticleMetadata(BaseModel):
    """Article metadata model."""
    
    title: str = Field(..., min_length=1, max_length=200, description="Article title")
    hook: str = Field(..., min_length=1, max_length=500, description="Article hook")
    excerpt: str = Field(..., min_length=1, max_length=500, description="Article excerpt")
    thesis: str = Field(..., min_length=1, max_length=1000, description="Central thesis")
    meta_description: str = Field(..., min_length=1, max_length=160, description="SEO meta description")
    keywords: List[str] = Field(..., description="SEO keywords")
    tags: List[str] = Field(default_factory=list, description="Article tags")
    reading_time_minutes: int = Field(..., description="Estimated reading time in minutes")
    word_count: int = Field(..., description="Total word count")
    character_count: int = Field(..., description="Total character count")
    
    @validator('meta_description')
    def validate_meta_description(cls, v):
        """Validate meta description length for SEO."""
        if len(v) > 160:
            raise ValueError('Meta description should be 160 characters or less for optimal SEO')
        return v
    
    @validator('title')
    def validate_title(cls, v):
        """Validate title length for SEO."""
        if len(v) > 70:
            raise ValueError('Title should be 70 characters or less for optimal SEO')
        return v


class Article(BaseModel):
    """Complete article model."""
    
    # Core Content
    title: str = Field(..., description="Article title")
    hook: str = Field(..., description="Article hook")
    excerpt: str = Field(..., description="Article excerpt")
    thesis: str = Field(..., description="Central thesis")
    sections: List[ArticleSection] = Field(..., description="Article sections")
    conclusion: str = Field(..., description="Article conclusion")
    
    # Metadata
    metadata: ArticleMetadata = Field(..., description="Article metadata")
    
    # Formatting
    format: ArticleFormat = Field(default=ArticleFormat.MARKDOWN, description="Output format")
    tone: ArticleTone = Field(..., description="Article tone")
    
    # Citations
    citations: List[Citation] = Field(default_factory=list, description="All citations")
    
    # Generation Info
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")
    model_used: str = Field(..., description="LLM model used for generation")
    generation_time_seconds: float = Field(..., description="Time taken to generate")
    
    # Quality Metrics
    readability_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Readability score")
    seo_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="SEO score")
    fact_check_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Fact-check score")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def get_total_word_count(self) -> int:
        """Calculate total word count."""
        return sum(section.word_count for section in self.sections)
    
    def get_citation_count(self) -> int:
        """Get total number of citations."""
        return len(self.citations)
    
    def get_section_count(self) -> int:
        """Get number of sections."""
        return len(self.sections)
    
    def to_markdown(self) -> str:
        """Convert article to Markdown format."""
        if self.format != ArticleFormat.MARKDOWN:
            raise ValueError("Article is not in Markdown format")
        
        markdown = f"# {self.title}\n\n"
        markdown += f"*{self.hook}*\n\n"
        markdown += f"## Excerpt\n\n{self.excerpt}\n\n"
        markdown += f"## Thesis\n\n{self.thesis}\n\n"
        
        for section in self.sections:
            markdown += f"## {section.title}\n\n{section.content}\n\n"
        
        markdown += f"## Conclusion\n\n{self.conclusion}\n\n"
        
        if self.citations:
            markdown += "## References\n\n"
            for i, citation in enumerate(self.citations, 1):
                markdown += f"{i}. {citation.title}"
                if citation.url:
                    markdown += f" - {citation.url}"
                markdown += "\n"
        
        return markdown
    
    def to_html(self) -> str:
        """Convert article to HTML format."""
        if self.format != ArticleFormat.HTML:
            raise ValueError("Article is not in HTML format")
        
        html_parts = [
            f"<article>",
            f"<h1>{self.title}</h1>"
        ]
        
        # Add hook if available
        if self.hook:
            html_parts.append(f'<div class="hook"><p><em>{self.hook}</em></p></div>')
        
        # Add excerpt if available
        if self.excerpt:
            html_parts.append(f'<div class="excerpt"><p>{self.excerpt}</p></div>')
        
        # Add thesis if available
        if self.thesis:
            html_parts.append(f'<div class="thesis"><p><strong>Thesis:</strong> {self.thesis}</p></div>')
        
        # Add sections with proper HTML structure
        for section in self.sections:
            html_parts.append(f"<section>")
            html_parts.append(f"<h2>{section.title}</h2>")
            
            # Clean and format section content
            content = self._clean_html_content(section.content)
            html_parts.append(content)
            
            html_parts.append(f"</section>")
        
        # Add conclusion if available
        if self.conclusion:
            html_parts.append(f"<section><h2>Conclusion</h2><p>{self.conclusion}</p></section>")
        
        # Add citations if available
        if self.citations:
            html_parts.append("<section><h2>References</h2><ol>")
            for citation in self.citations:
                html_parts.append(f'<li id="{citation.id}">{citation.title}')
                if citation.url:
                    html_parts.append(f' - <a href="{citation.url}">{citation.url}</a>')
                html_parts.append("</li>")
            html_parts.append("</ol></section>")
        
        html_parts.append("</article>")
        
        return "\n".join(html_parts)
    
    def _clean_html_content(self, content: str) -> str:
        """Clean and properly format HTML content."""
        if not content:
            return ""
        
        import re
        
        # Remove any existing HTML wrapper tags that might be in the content
        content = content.strip()
        
        # Remove markdown code blocks and HTML artifacts
        content = re.sub(r'```html\s*', '', content)
        content = re.sub(r'```\s*$', '', content)
        content = re.sub(r'<br>\s*', '\n', content)
        
        # Split into paragraphs and clean each one
        paragraphs = content.split('\n\n')
        cleaned_paragraphs = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Clean up nested paragraph tags
            paragraph = re.sub(r'<p>\s*<p>', '<p>', paragraph)
            paragraph = re.sub(r'</p>\s*</p>', '</p>', paragraph)
            
            # Handle headings properly - ensure proper hierarchy
            if paragraph.startswith('<h3>'):
                cleaned_paragraphs.append(paragraph)
            elif paragraph.startswith('<ul>') or paragraph.startswith('<ol>'):
                cleaned_paragraphs.append(paragraph)
            elif paragraph.startswith('<table>') or '<table>' in paragraph:
                # Clean and validate table structure
                cleaned_table = self._clean_table_content(paragraph)
                if cleaned_table:
                    cleaned_paragraphs.append(cleaned_table)
            else:
                # Wrap in paragraph tags if not already wrapped
                if not paragraph.startswith('<p>'):
                    paragraph = f"<p>{paragraph}</p>"
                cleaned_paragraphs.append(paragraph)
        
        return '\n\n'.join(cleaned_paragraphs)
    
    def _clean_table_content(self, table_content: str) -> str:
        """Clean and validate HTML table content."""
        if not table_content or not table_content.strip():
            return ""
        
        import re
        
        # Basic table structure validation
        if '<table>' not in table_content or '</table>' not in table_content:
            return ""
        
        # Ensure proper table structure
        table_content = table_content.strip()
        
        # Add basic table styling if not present
        if 'style=' not in table_content:
            table_content = table_content.replace('<table>', '<table style="border-collapse: collapse; width: 100%; margin: 1em 0;">')
        
        # Ensure proper table headers
        if '<th>' not in table_content and '<thead>' not in table_content:
            # Try to convert first row to header if no headers exist
            table_content = re.sub(
                r'<tr>(.*?)</tr>',
                r'<thead><tr>\1</tr></thead><tbody>',
                table_content,
                count=1
            )
            table_content = table_content.replace('</table>', '</tbody></table>')
        
        # Add basic cell styling
        table_content = table_content.replace('<th>', '<th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2; text-align: left;">')
        table_content = table_content.replace('<td>', '<td style="border: 1px solid #ddd; padding: 8px;">')
        
        return table_content
