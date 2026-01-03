"""
Article Structure Generator for Content Generator V2.

This module generates comprehensive article structures including titles, hooks,
excerpts, thesis statements, and detailed section outlines.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

# Import verbalized sampling client
from verbalized_sampling_client import VerbalizedSamplingClient, create_verbalized_sampling_client

# Configure logging
logger = logging.getLogger(__name__)

class ArticleType(Enum):
    """Types of articles that can be generated."""
    LISTICLE = "listicle"
    HOW_TO = "how_to"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"
    NEWS = "news"
    TUTORIAL = "tutorial"
    REVIEW = "review"
    OPINION = "opinion"

class Tone(Enum):
    """Article tones."""
    JOURNALISTIC = "journalistic"
    CONVERSATIONAL = "conversational"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    AUTHORITATIVE = "authoritative"
    FRIENDLY = "friendly"

@dataclass
class SectionOutline:
    """Outline for a single article section."""
    title: str
    subtitle: Optional[str] = None
    key_points: List[str] = None
    word_count_target: int = 300
    content_type: str = "paragraph"
    order: int = 1
    importance: str = "high"  # high, medium, low

@dataclass
class ArticleStructure:
    """Complete article structure."""
    title: str
    hook: str
    excerpt: str
    thesis: str
    meta_description: str
    target_word_count: int
    article_type: str
    tone: str
    sections: List[SectionOutline]
    keywords: List[str]
    target_audience: str
    call_to_action: Optional[str] = None

class ArticleStructureGenerator:
    """
    Generates comprehensive article structures using LLM.
    """
    
    def __init__(self, llm_client, use_verbalized_sampling: bool = True):
        """
        Initialize the article structure generator.
        
        Args:
            llm_client: Configured LLM client
            use_verbalized_sampling: Whether to use verbalized sampling for improved quality
        """
        self.llm_client = llm_client
        self.use_verbalized_sampling = use_verbalized_sampling
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize verbalized sampling client if enabled
        if self.use_verbalized_sampling:
            self.verbalized_client = create_verbalized_sampling_client(
                k=5,
                tau=0.10,
                temperature=0.9,
                seed=42,
                enabled=True
            )
            self.logger.info("Verbalized sampling enabled for article structure generation")
        else:
            self.verbalized_client = None
            self.logger.info("Verbalized sampling disabled, using standard generation")
    
    def generate_structure(self, research_data: Dict[str, Any], claims: List[Dict], evidence: List[Dict]) -> ArticleStructure:
        """
        Generate complete article structure.
        
        Args:
            research_data: Research parameters and brief
            claims: Extracted claims from research
            evidence: Collected evidence
            
        Returns:
            Complete ArticleStructure object
        """
        try:
            brief = research_data.get('brief', '')
            keywords = research_data.get('keywords', '')
            tone = research_data.get('tone', 'journalistic')
            target_word_count = research_data.get('target_word_count', 2000)
            
            # Determine article type based on brief content
            article_type = self._determine_article_type(brief)
            
            # Generate core elements
            draft_title = research_data.get('draft_title', '')
            title = self._generate_title(brief, keywords, article_type, tone, draft_title)
            hook = self._generate_hook(brief, claims, tone)
            excerpt = self._generate_excerpt(brief, claims, target_word_count, tone)
            thesis = self._generate_thesis(brief, claims, evidence, tone)
            meta_description = self._generate_meta_description(title, excerpt, keywords)
            
            # Generate section outlines
            sections = self._generate_sections(brief, claims, evidence, target_word_count, tone, article_type)
            
            # Log section titles for debugging
            section_titles = [s.title for s in sections]
            self.logger.info(f"Generated section titles: {section_titles}")
            
            # Log section titles for debugging
            section_titles = [s.title for s in sections]
            self.logger.info(f"Generated section titles: {section_titles}")
            
            # Determine target audience
            target_audience = self._determine_target_audience(brief, tone)
            
            # Generate call to action
            call_to_action = self._generate_call_to_action(article_type, tone)
            
            structure = ArticleStructure(
                title=title,
                hook=hook,
                excerpt=excerpt,
                thesis=thesis,
                meta_description=meta_description,
                target_word_count=target_word_count,
                article_type=article_type,
                tone=tone,
                sections=sections,
                keywords=keywords.split(',') if keywords else [],
                target_audience=target_audience,
                call_to_action=call_to_action
            )
            
            self.logger.info(f"Generated article structure with {len(sections)} sections")
            return structure
            
        except Exception as e:
            self.logger.error(f"Error generating article structure: {str(e)}")
            # Return fallback structure
            return self._create_fallback_structure(research_data)
    
    def _determine_article_type(self, brief: str) -> str:
        """Determine article type based on brief content."""
        brief_lower = brief.lower()
        
        if any(word in brief_lower for word in ['list', 'top', 'best', 'worst', 'ranking', 'countdown']):
            return ArticleType.LISTICLE.value
        elif any(word in brief_lower for word in ['how to', 'how-to', 'tutorial', 'guide', 'steps']):
            return ArticleType.HOW_TO.value
        elif any(word in brief_lower for word in ['vs', 'versus', 'compare', 'comparison', 'difference']):
            return ArticleType.COMPARISON.value
        elif any(word in brief_lower for word in ['analysis', 'analyze', 'breakdown', 'examine']):
            return ArticleType.ANALYSIS.value
        elif any(word in brief_lower for word in ['news', 'breaking', 'update', 'recent']):
            return ArticleType.NEWS.value
        elif any(word in brief_lower for word in ['review', 'opinion', 'thoughts', 'experience']):
            return ArticleType.REVIEW.value
        else:
            return ArticleType.ANALYSIS.value  # Default
    
    def _generate_title(self, brief: str, keywords: str, article_type: str, tone: str, draft_title: str = '') -> str:
        """Generate compelling article title."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert content strategist. Generate a compelling, SEO-optimized title for a {article_type} article.
                    
                    Requirements:
                    - 60-100 characters for optimal SEO and readability (prefer 70-90 for comprehensive topics)
                    - Include primary keywords naturally
                    - Match the {tone} tone
                    - Be specific and actionable
                    - Create curiosity without clickbait
                    - Use power words when appropriate
                    - If a draft title is provided, use it as inspiration and ensure the final title relates to it
                    - Ensure the title includes relevant keywords from the provided list
                    - Do NOT truncate with ellipsis - write a complete, natural title
                    
                    Return only the title, no quotes or formatting."""
                },
                {
                    "role": "user",
                    "content": f"Article Brief: {brief}\nKeywords: {keywords}\nArticle Type: {article_type}\nTone: {tone}{f'\nDraft Title: {draft_title}' if draft_title else ''}"
                }
            ]
            
            response = self.llm_client.generate(messages)
            title = response.content.strip().strip('"').strip("'")
            
            # Ensure title length is reasonable (safety limit to prevent abuse)
            # Allow up to 120 characters for comprehensive topics, but warn if over 100
            if len(title) > 120:
                self.logger.warning(f"Title very long ({len(title)} chars), truncating to 120")
                title = title[:117] + "..."
            elif len(title) > 100:
                self.logger.info(f"Title slightly over recommended length ({len(title)} chars), keeping as-is")
            
            return title
            
        except Exception as e:
            self.logger.error(f"Error generating title: {str(e)}")
            # Use longer fallback (80 chars) to avoid truncation of meaningful content
            brief_truncated = brief[:80] if len(brief) > 80 else brief
            return f"Comprehensive Guide: {brief_truncated}"
    
    def _generate_hook(self, brief: str, claims: List[Dict], tone: str) -> str:
        """Generate compelling opening hook."""
        try:
            # Extract key claims for context
            claim_text = "\n".join([claim.get('claim', '') for claim in claims[:3]])
            
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a master storyteller. Write a compelling opening hook for an article.
                    
                    Requirements:
                    - 1-2 sentences maximum
                    - Create immediate engagement and curiosity
                    - Match the {tone} tone
                    - Use a surprising fact, statistic, question, or bold statement
                    - Set up the article's value proposition
                    - Be specific and concrete
                    - DO NOT repeat or quote the article brief directly
                    - DO NOT use ellipsis (...) unless for dramatic effect
                    - Write a complete, engaging sentence that draws readers in
                    - Focus on the benefit or insight, not the topic description
                    {'- DO NOT start with greetings like "Hi friends", "Hey there", "Hello everyone" - start directly with engaging content' if tone.lower() == 'friendly' else ''}
                    
                    Examples of good hooks:
                    - "While 73% of professionals struggle with career transitions, only 12% have a strategic mentor relationship—here's how to join that elite group."
                    - "The average professional changes careers 5-7 times, but most never master the art of strategic skill development on job applications."
                    - "What if the secret to promotion readiness isn't about working harder, but about strategically showcasing your skills?"
                    
                    Return only the hook, no quotes, no ellipsis, no formatting."""
                },
                {
                    "role": "user",
                    "content": f"Article Topic: {brief}\nKey Claims: {claim_text}\nTone: {tone}\n\nWrite a compelling hook that engages readers without repeating the topic description."
                }
            ]
            
            response = self.llm_client.generate(messages)
            hook = response.content.strip().strip('"').strip("'")
            
            # Clean up any unwanted ellipsis that might have been added
            # Remove trailing ellipsis unless it's part of a question
            if hook.endswith('...') and not hook.endswith('...?'):
                hook = hook[:-3].rstrip()
            
            # Ensure hook is complete and engaging
            if len(hook) < 20:
                self.logger.warning(f"Generated hook is too short: {hook}")
                # Fall through to fallback
            
            return hook
            
        except Exception as e:
            self.logger.error(f"Error generating hook: {str(e)}")
            # Improved fallback without ellipsis
            # Extract key topic from brief (first few words)
            topic_words = brief.split()[:5]
            topic = ' '.join(topic_words)
            return f"Discover the essential strategies for {topic} that top professionals use to advance their careers."
    
    def _generate_excerpt(self, brief: str, claims: List[Dict], target_word_count: int, tone: str) -> str:
        """Generate article excerpt/summary."""
        try:
            # Calculate excerpt length based on target word count
            excerpt_length = min(150, max(50, target_word_count // 15))
            
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert content writer. Write a compelling article excerpt.
                    
                    Requirements:
                    - Approximately {excerpt_length} words
                    - Match the {tone} tone
                    - Summarize the article's main value proposition
                    - Include key benefits or outcomes
                    - Create urgency or interest
                    - Be specific and actionable
                    
                    Return only the excerpt, no quotes or formatting."""
                },
                {
                    "role": "user",
                    "content": f"Article Brief: {brief}\nTarget Word Count: {target_word_count}\nTone: {tone}"
                }
            ]
            
            response = self.llm_client.generate(messages)
            excerpt = response.content.strip().strip('"').strip("'")
            
            return excerpt
            
        except Exception as e:
            self.logger.error(f"Error generating excerpt: {str(e)}")
            return f"This comprehensive guide explores {brief[:100]}... providing actionable insights and practical solutions."
    
    def _generate_thesis(self, brief: str, claims: List[Dict], evidence: List[Dict], tone: str) -> str:
        """Generate clear thesis statement."""
        try:
            # Prepare evidence summary
            evidence_summary = f"Based on {len(evidence)} sources of evidence" if evidence else "Based on comprehensive research"
            
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert academic writer. Write a clear, compelling thesis statement.
                    
                    Requirements:
                    - 1-2 sentences maximum
                    - Match the {tone} tone
                    - State the main argument or position clearly
                    - Be specific and debatable
                    - Set up the article's structure
                    - Include the main benefit or outcome
                    
                    Return only the thesis statement, no quotes or formatting."""
                },
                {
                    "role": "user",
                    "content": f"Article Brief: {brief}\nEvidence: {evidence_summary}\nTone: {tone}"
                }
            ]
            
            response = self.llm_client.generate(messages)
            thesis = response.content.strip().strip('"').strip("'")
            
            return thesis
            
        except Exception as e:
            self.logger.error(f"Error generating thesis: {str(e)}")
            return f"This article demonstrates that {brief[:100]}... through evidence-based analysis and practical insights."
    
    def _generate_meta_description(self, title: str, excerpt: str, keywords: str) -> str:
        """Generate SEO-optimized meta description."""
        try:
            # Combine title and excerpt for context
            context = f"Title: {title}\nExcerpt: {excerpt}\nKeywords: {keywords}"
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an SEO expert. Write a compelling meta description.
                    
                    Requirements:
                    - 150-160 characters for optimal SEO
                    - Include primary keywords naturally
                    - Summarize the article's value proposition
                    - Include a call to action
                    - Be specific and engaging
                    - Avoid keyword stuffing
                    
                    Return only the meta description, no quotes or formatting."""
                },
                {
                    "role": "user",
                    "content": context
                }
            ]
            
            response = self.llm_client.generate(messages)
            meta_description = response.content.strip().strip('"').strip("'")
            
            # Ensure proper length
            if len(meta_description) > 160:
                meta_description = meta_description[:157] + "..."
            
            return meta_description
            
        except Exception as e:
            self.logger.error(f"Error generating meta description: {str(e)}")
            return f"Discover {title[:50]}... Learn everything you need to know with this comprehensive guide."
    
    def _analyze_evidence_distribution(self, evidence: List[Dict]) -> str:
        """Analyze evidence distribution to help with section planning."""
        if not evidence:
            return "No evidence available for analysis."
        
        # Count evidence by type
        evidence_types = {}
        source_types = {}
        
        for ev in evidence:
            ev_type = ev.get('source_type', 'unknown')
            evidence_types[ev_type] = evidence_types.get(ev_type, 0) + 1
            
            # Extract source information
            source = ev.get('source', '')
            if 'academic' in source.lower() or 'journal' in source.lower():
                source_types['academic'] = source_types.get('academic', 0) + 1
            elif 'news' in source.lower() or 'article' in source.lower():
                source_types['news'] = source_types.get('news', 0) + 1
            elif 'gov' in source.lower() or 'government' in source.lower():
                source_types['government'] = source_types.get('government', 0) + 1
            else:
                source_types['web'] = source_types.get('web', 0) + 1
        
        analysis = f"Evidence Analysis: {len(evidence)} total sources\n"
        analysis += f"By type: {', '.join([f'{k}: {v}' for k, v in evidence_types.items()])}\n"
        analysis += f"By source: {', '.join([f'{k}: {v}' for k, v in source_types.items()])}\n"
        
        return analysis
    
    def _generate_sections(self, brief: str, claims: List[Dict], evidence: List[Dict], 
                          target_word_count: int, tone: str, article_type: str) -> List[SectionOutline]:
        """Generate detailed section outlines with balanced word distribution."""
        try:
            # Calculate section count based on target word count with better distribution
            section_count = max(4, min(8, target_word_count // 400))  # Increased base count and word target
            
            # Calculate balanced word count per section
            words_per_section = target_word_count // section_count
            min_words = max(200, int(words_per_section * 0.7))  # 70% of target
            max_words = int(words_per_section * 1.3)  # 130% of target
            
            # Prepare context
            claims_text = "\n".join([f"- {claim.get('claim', '')}" for claim in claims[:5]])
            evidence_text = f"Evidence from {len(evidence)} sources" if evidence else "Research-based insights"
            
            # Analyze evidence distribution for better section planning
            evidence_types = self._analyze_evidence_distribution(evidence)
            
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert content strategist. Create a detailed, balanced outline for a {article_type} article.
                    
                    CRITICAL REQUIREMENTS:
                    - Create exactly {section_count} main sections
                    - Each section should be {min_words}-{max_words} words (target: {words_per_section} words)
                    - Ensure BALANCED content distribution - no single section should dominate
                    - Match the {tone} tone
                    - Include introduction and conclusion sections
                    - Order sections logically with smooth transitions
                    - Include practical, actionable content
                    - Distribute evidence and claims evenly across sections
                    
                    ⚠️ CRITICAL: AVOID GENERIC SECTION TITLES ⚠️
                    - DO NOT use generic titles like: "Getting Started", "Step-by-Step Process", "Key Concepts", "Practical Applications", "Understanding the Fundamentals", "Real-World Implementation"
                    - DO NOT use the same structure for every article
                    - CREATE UNIQUE, TOPIC-SPECIFIC section titles that directly relate to the article brief and claims
                    - Each section title should be specific to THIS article's topic, not a generic template
                    - Analyze the brief and claims to create sections that make sense for THIS specific topic
                    - Example: For an article about "skills for 2026", create sections like "Top In-Demand Technical Skills", "Essential Soft Skills for Hybrid Work", "How to Develop These Skills", NOT "Getting Started" or "Step-by-Step Process"
                    
                    EVIDENCE DISTRIBUTION ANALYSIS:
                    {evidence_types}
                    
                    SECTION BALANCING RULES:
                    - Introduction: 150-250 words (keep it concise, single paragraph style)
                    - Main content sections: {min_words}-{max_words} words each
                    - Conclusion: 150-250 words
                    - NO section should exceed {max_words} words
                    - NO section should be under {min_words} words
                    
                    INTRODUCTION REQUIREMENTS:
                    - Keep introduction SIMPLE and CONCISE
                    - Use a single paragraph or very brief structure
                    - Avoid multiple subsections in introduction
                    - Focus on hook, overview, and what reader will learn
                    
                    Content Types Available:
                    - "paragraph": Standard text content
                    - "list": Bulleted or numbered lists
                    - "step_by_step": Instructional content (only use if the article is actually a step-by-step guide)
                    - "comparison": Side-by-side comparisons
                    - "table": Data-rich content with tables
                    
                    IMPORTANT: Return ONLY valid JSON. Do not include any text before or after the JSON. The JSON must be parseable.
                    
                    Format as JSON:
                    {{
                        "sections": [
                            {{
                                "title": "Topic-Specific Section Title (NOT generic)",
                                "subtitle": "Optional subtitle",
                                "key_points": ["Point 1", "Point 2", "Point 3"],
                                "word_count_target": {words_per_section},
                                "content_type": "paragraph",
                                "order": 1,
                                "importance": "high"
                            }}
                        ]
                    }}"""
                },
                {
                    "role": "user",
                    "content": f"""Article Brief: {brief}

Key Claims to Address:
{claims_text}

Evidence Available: {evidence_text}
Target Word Count: {target_word_count}
Tone: {tone}

Create {section_count} topic-specific sections that directly relate to this article's content. Each section title should be unique to this topic, not a generic template. Analyze the brief and claims to determine what sections make sense for THIS specific article."""
                }
            ]
            
            response = self.llm_client.generate(messages)
            
            # Parse JSON response
            import json
            import re
            
            # Try to extract JSON from response (in case LLM adds extra text)
            response_text = response.content.strip()
            
            # Try to find JSON object in the response
            json_match = re.search(r'\{[^{}]*"sections"[^{}]*\[.*?\]\s*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            try:
                data = json.loads(response_text)
                sections_data = data.get('sections', [])
                
                if not sections_data:
                    raise ValueError("No sections found in response")
                
                sections = []
                for i, section_data in enumerate(sections_data):
                    section_title = section_data.get('title', f'Section {i+1}')
                    
                    # Warn if generic titles are detected
                    generic_titles = ['getting started', 'step-by-step process', 'step by step process', 
                                     'key concepts', 'practical applications', 'understanding the fundamentals',
                                     'real-world implementation', 'conclusion']
                    if any(generic in section_title.lower() for generic in generic_titles) and i > 0 and i < len(sections_data) - 1:
                        self.logger.warning(f"Generic section title detected: '{section_title}' - consider making it more topic-specific")
                    
                    section = SectionOutline(
                        title=section_title,
                        subtitle=section_data.get('subtitle'),
                        key_points=section_data.get('key_points', []),
                        word_count_target=section_data.get('word_count_target', 300),
                        content_type=section_data.get('content_type', 'paragraph'),
                        order=section_data.get('order', i+1),
                        importance=section_data.get('importance', 'high')
                    )
                    sections.append(section)
                
                self.logger.info(f"Successfully parsed {len(sections)} sections from LLM response")
                return sections
                
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"Failed to parse JSON response: {str(e)}")
                self.logger.warning(f"Response content (first 500 chars): {response_text[:500]}")
                # Use dynamic fallback that's more topic-specific
                return self._create_fallback_sections(brief, target_word_count, claims)
            
        except Exception as e:
            self.logger.error(f"Error generating sections: {str(e)}")
            return self._create_fallback_sections(brief, target_word_count)
    
    def _determine_target_audience(self, brief: str, tone: str) -> str:
        """Determine target audience based on brief and tone."""
        brief_lower = brief.lower()
        
        if any(word in brief_lower for word in ['professional', 'business', 'corporate', 'executive']):
            return "Business professionals and executives"
        elif any(word in brief_lower for word in ['beginner', 'newbie', 'start', 'introduction']):
            return "Beginners and newcomers"
        elif any(word in brief_lower for word in ['expert', 'advanced', 'technical', 'developer']):
            return "Experts and technical professionals"
        elif any(word in brief_lower for word in ['homeowner', 'consumer', 'personal', 'individual']):
            return "General consumers and homeowners"
        else:
            return "General audience interested in the topic"
    
    def _generate_call_to_action(self, article_type: str, tone: str) -> Optional[str]:
        """Generate appropriate call to action."""
        if article_type == ArticleType.HOW_TO.value:
            return "Ready to get started? Follow these steps and share your results with us!"
        elif article_type == ArticleType.COMPARISON.value:
            return "Which option works best for you? Let us know in the comments below!"
        elif article_type == ArticleType.REVIEW.value:
            return "Have you tried this? Share your experience and help others decide!"
        else:
            return "Found this helpful? Share it with others who might benefit!"
    
    def _create_fallback_structure(self, research_data: Dict[str, Any]) -> ArticleStructure:
        """Create fallback structure when generation fails."""
        brief = research_data.get('brief', 'Article about important topic')
        tone = research_data.get('tone', 'journalistic')
        target_word_count = research_data.get('target_word_count', 2000)
        
        return ArticleStructure(
            title=f"Complete Guide: {brief[:50]}...",
            hook=f"Discover everything you need to know about {brief[:50]}...",
            excerpt=f"This comprehensive guide explores {brief[:100]}... providing actionable insights and practical solutions.",
            thesis=f"This article demonstrates the key aspects of {brief[:50]}... through evidence-based analysis.",
            meta_description=f"Learn about {brief[:50]}... with this comprehensive guide. Get actionable insights and practical solutions.",
            target_word_count=target_word_count,
            article_type="analysis",
            tone=tone,
            sections=self._create_fallback_sections(brief, target_word_count),
            keywords=research_data.get('keywords', '').split(',') if research_data.get('keywords') else [],
            target_audience="General audience",
            call_to_action="Found this helpful? Share it with others who might benefit!"
        )
    
    def _create_fallback_sections(self, brief: str, target_word_count: int, claims: List[Dict] = None) -> List[SectionOutline]:
        """Create fallback section outlines with topic-specific structure based on brief and claims."""
        # Analyze brief to create more relevant sections
        brief_lower = brief.lower()
        
        # Extract key topics from brief (first few meaningful words)
        brief_words = [w for w in brief.split() if len(w) > 3][:3]
        topic_phrase = ' '.join(brief_words) if brief_words else "the topic"
        
        # Extract key themes from claims if available
        claim_themes = []
        if claims:
            for claim in claims[:3]:
                claim_text = claim.get('claim', '')
                # Extract key nouns/phrases (simple heuristic)
                words = [w for w in claim_text.split() if w.lower() not in ['the', 'a', 'an', 'is', 'are', 'and', 'or', 'but']]
                if words:
                    claim_themes.append(' '.join(words[:2]))
        
        # Determine article focus and create topic-specific sections
        if any(word in brief_lower for word in ['how to', 'guide', 'steps', 'process', 'tutorial']):
            # How-to article structure - but make it topic-specific
            main_topic = brief_words[0] if brief_words else "the process"
            sections = [
                SectionOutline(
                    title="Introduction",
                    key_points=["Overview of the topic", "Why this matters", "What you'll learn"],
                    word_count_target=200,
                    order=1,
                    importance="high"
                ),
                SectionOutline(
                    title=f"Essential {main_topic.title()} Basics" if main_topic else "Essential Basics",
                    key_points=["Core concepts", "Important principles", "What you need to know"],
                    word_count_target=400,
                    order=2,
                    importance="high"
                ),
                SectionOutline(
                    title=f"Mastering {main_topic.title()}" if main_topic else "Mastering the Process",
                    key_points=["Detailed approach", "Best practices", "Pro tips"],
                    word_count_target=600,
                    order=3,
                    importance="high"
                ),
                SectionOutline(
                    title="Conclusion",
                    key_points=["Key takeaways", "Next steps", "Final thoughts"],
                    word_count_target=200,
                    order=4,
                    importance="medium"
                )
            ]
        elif any(word in brief_lower for word in ['investment', 'financial', 'market', 'analysis']):
            # Financial/investment article structure
            sections = [
                SectionOutline(
                    title="Introduction",
                    key_points=["Market overview", "Current trends", "Why this matters"],
                    word_count_target=200,
                    order=1,
                    importance="high"
                ),
                SectionOutline(
                    title="Market Analysis",
                    key_points=["Current state", "Trends and patterns", "Data insights"],
                    word_count_target=500,
                    order=2,
                    importance="high"
                ),
                SectionOutline(
                    title="Investment Strategies",
                    key_points=["Approaches", "Risk assessment", "Opportunities"],
                    word_count_target=500,
                    order=3,
                    importance="high"
                ),
                SectionOutline(
                    title="Conclusion",
                    key_points=["Key takeaways", "Next steps", "Final thoughts"],
                    word_count_target=200,
                    order=4,
                    importance="medium"
                )
            ]
        elif any(word in brief_lower for word in ['skill', 'skills', 'career', 'development', 'learn']):
            # Skills/career article structure - make it topic-specific
            skill_focus = "Skills" if 'skill' in brief_lower else "Career Development"
            sections = [
                SectionOutline(
                    title="Introduction",
                    key_points=["Overview of the topic", "Why this matters", "What you'll learn"],
                    word_count_target=200,
                    order=1,
                    importance="high"
                ),
                SectionOutline(
                    title=f"Top In-Demand {skill_focus} for 2026" if '2026' in brief_lower or '2025' in brief_lower else f"Essential {skill_focus} to Master",
                    key_points=claim_themes[:3] if claim_themes else ["Key skills", "Why they matter", "Market demand"],
                    word_count_target=500,
                    order=2,
                    importance="high"
                ),
                SectionOutline(
                    title=f"How to Develop These {skill_focus}" if 'skill' in brief_lower else "Building Your Career Path",
                    key_points=["Actionable steps", "Learning resources", "Practical tips"],
                    word_count_target=500,
                    order=3,
                    importance="high"
                ),
                SectionOutline(
                    title="Conclusion",
                    key_points=["Key takeaways", "Next steps", "Final thoughts"],
                    word_count_target=200,
                    order=4,
                    importance="medium"
                )
            ]
        else:
            # General article structure - try to make it topic-specific based on brief
            # Extract main topic from brief
            main_topic = brief_words[0].title() if brief_words else "Key Concepts"
            second_topic = brief_words[1].title() if len(brief_words) > 1 else "Implementation"
            
            sections = [
                SectionOutline(
                    title="Introduction",
                    key_points=["Overview of the topic", "Why this matters", "What you'll learn"],
                    word_count_target=200,
                    order=1,
                    importance="high"
                ),
                SectionOutline(
                    title=f"Understanding {main_topic}" if main_topic else "Core Concepts",
                    key_points=claim_themes[:3] if claim_themes else ["Core concepts", "Important principles", "Key insights"],
                    word_count_target=400,
                    order=2,
                    importance="high"
                ),
                SectionOutline(
                    title=f"{second_topic} in Practice" if second_topic else "Practical Applications",
                    key_points=["Practical examples", "Case studies", "Best practices"],
                    word_count_target=500,
                    order=3,
                    importance="high"
                ),
                SectionOutline(
                    title="Conclusion",
                    key_points=["Key takeaways", "Next steps", "Final thoughts"],
                    word_count_target=200,
                    order=4,
                    importance="medium"
                )
            ]
        
        self.logger.info(f"Created fallback sections: {[s.title for s in sections]}")
        return sections

# Factory function
def create_article_structure_generator(llm_client, use_verbalized_sampling: bool = True) -> ArticleStructureGenerator:
    """
    Create an article structure generator.
    
    Args:
        llm_client: Configured LLM client
        use_verbalized_sampling: Whether to use verbalized sampling for improved quality
        
    Returns:
        ArticleStructureGenerator instance
    """
    return ArticleStructureGenerator(llm_client, use_verbalized_sampling)

# Example usage
if __name__ == "__main__":
    # This would be used with a real LLM client
    print("Article Structure Generator - Ready for integration")
