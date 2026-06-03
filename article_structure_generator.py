"""
Article Structure Generator for Content Generator V2.

This module generates comprehensive article structures including titles, hooks,
excerpts, thesis statements, and detailed section outlines.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

# Verbalized sampling client removed to avoid slowdown and errors
# from verbalized_sampling_client import VerbalizedSamplingClient, create_verbalized_sampling_client

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
    deck: str
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
    
    def __init__(self, llm_client, use_verbalized_sampling: bool = False):
        """
        Initialize the article structure generator.
        
        Args:
            llm_client: Configured LLM client
            use_verbalized_sampling: Whether to use verbalized sampling (DISABLED)
        """
        self.llm_client = llm_client
        self.use_verbalized_sampling = False # Explicitly disabled to avoid slowdown
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
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

            # Add competitor analysis insights to the brief
            competitor_context = self._build_competitor_context_text(research_data)
            if competitor_context:
                brief = f"{brief}\n\nCompetitor Analysis Insights:\n{competitor_context}"

            dossier_context = self._build_dossier_context_text(research_data)
            brief_with_dossier = f"{brief}\n\nDeep Research Context:\n{dossier_context}" if dossier_context else brief
            
            # Determine article type based on brief content
            article_type = self._determine_article_type(brief)
            
            # Generate core elements
            draft_title = research_data.get('draft_title', '')
            title = self._generate_title(brief_with_dossier, keywords, article_type, tone, draft_title)
            hook = self._generate_hook(brief_with_dossier, claims, tone)
            excerpt = self._generate_excerpt(brief_with_dossier, claims, target_word_count, tone)
            thesis = self._generate_thesis(brief_with_dossier, claims, evidence, tone)
            meta_description = self._generate_meta_description(title, excerpt, keywords)
            
            # Generate section outlines
            content_outline = research_data.get('content_outline')
            if content_outline and isinstance(content_outline, list) and len(content_outline) > 0:
                self.logger.info(f"Using provided content_outline from database ({len(content_outline)} items)")
                sections = self._generate_sections_from_outline(content_outline, target_word_count, tone)
            else:
                sections = self._generate_sections(brief_with_dossier, claims, evidence, target_word_count, tone, article_type)
            
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
                deck=self._generate_deck(brief, hook, tone),
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

    def _build_dossier_context_text(self, research_data: Dict[str, Any]) -> str:
        """Build compact deep-research context string for structure planning prompts."""
        dossier = research_data.get('research_dossier') or {}
        if not isinstance(dossier, dict):
            return ""

        summary = str(dossier.get('summary', '') or '').strip()
        claims = dossier.get('primary_claims') or []
        unresolved = dossier.get('unresolved_questions') or []
        stats = dossier.get('important_statistics') or []

        claim_lines = []
        if isinstance(claims, list):
            claim_lines = [
                c.get('claim', '').strip() for c in claims[:4]
                if isinstance(c, dict) and c.get('claim')
            ]
        unresolved_lines = [str(q).strip() for q in unresolved[:3]] if isinstance(unresolved, list) else []
        stat_lines = [str(s).strip() for s in stats[:3]] if isinstance(stats, list) else []

        parts = []
        if summary:
            parts.append(f"Summary: {summary[:500]}")
        if claim_lines:
            parts.append("Primary Claims: " + " | ".join(claim_lines))
        if unresolved_lines:
            parts.append("Unresolved Questions: " + " | ".join(unresolved_lines))
        if stat_lines:
            parts.append("Important Statistics: " + " | ".join(stat_lines))
        return "\n".join(parts)

    def _build_competitor_context_text(self, research_data: Dict[str, Any]) -> str:
        """Build a compact competitor analysis context string for structure planning prompts."""
        analysis = research_data.get('competitor_analysis') or {}
        if not isinstance(analysis, dict):
            return ""
        
        must_haves = analysis.get('must_haves') or []
        edge = analysis.get('competitive_edge') or []
        
        lines = []
        if must_haves:
            lines.append("Competitor Must-Haves (topics/details you MUST cover in this article):")
            for item in must_haves[:6]:
                lines.append(f"- {item}")
        if edge:
            lines.append("Our Competitive Edge (topics/gaps competitors missed or cover poorly that we should emphasize):")
            for item in edge[:6]:
                lines.append(f"- {item}")
        
        return "\n".join(lines) if lines else ""
    
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
        """Generate compelling article title with strict length enforcement."""
        try:
            # First attempt prompt
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert content strategist. Generate a compelling, SEO-optimized title for a {article_type} article.
                    
                    CRITICAL REQUIREMENTS:
                    - MUST be under 60 characters (strict SEO limit)
                    - Include one primary keyword naturally
                    - Match the {tone} tone
                    - Be specific and actionable
                    - Create curiosity without clickbait
                    - If a draft title is provided, use it as inspiration but make it shorter
                    
                    Return only the title, no quotes or formatting."""
                },
                {
                    "role": "user",
                    "content": f"Article Brief: {brief}\nKeywords: {keywords}\nArticle Type: {article_type}\nTone: {tone}" + (f"\nDraft Title: {draft_title}" if draft_title else "")
                }
            ]
            
            response = self.llm_client.generate(messages)
            title = response.content.strip().strip('"').strip("'")
            
            # Regeneration loop for length enforcement
            max_attempts = 3
            current_attempt = 1
            
            while len(title) > 60 and current_attempt < max_attempts:
                self.logger.info(f"Generated title '{title}' is {len(title)} chars (limit: 60). Regenerating (attempt {current_attempt}/{max_attempts})...")
                
                regeneration_messages = [
                    {
                        "role": "system",
                        "content": f"""You are an SEO expert. The previous title was too long.
                        
                        Rewrite this title to be strictly UNDER 60 CHARACTERS.
                        
                        Previous Title: "{title}"
                        
                        REQUIREMENTS:
                        - MAX 60 characters
                        - Must include one of these keywords: {keywords}
                        - Keep the core meaning but concise
                        - Match {tone} tone
                        
                        Return only the shortened title."""
                    },
                    {
                        "role": "user",
                        "content": "Shorten the title to under 60 characters while keeping a keyword."
                    }
                ]
                
                response = self.llm_client.generate(regeneration_messages)
                new_title = response.content.strip().strip('"').strip("'")
                
                if new_title and len(new_title) > 0:
                    title = new_title
                current_attempt += 1
            
            # Final verification
            if not title or len(title) < 5 or title.lower() == 'none':
                raise ValueError("Invalid title generated")
            
            if len(title) > 60:
                self.logger.warning(f"Title still over 60 chars ({len(title)}) after regeneration. Accepting longer title to avoid truncation as requested.")
            
            self.logger.info(f"Final generated title: '{title}' ({len(title)} chars)")
            return title
            
        except Exception as e:
            self.logger.error(f"Error generating title: {str(e)}")
            # Fallback
            brief_words = brief.split()[:5]
            short_brief = " ".join(brief_words)
            return f"{short_brief}..." if len(short_brief) < 55 else f"{short_brief[:55]}..."
    
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
            if not hook or len(hook) < 20 or hook.lower() == 'none':
                self.logger.warning(f"Generated hook is invalid or too short: '{hook}'. Using fallback.")
                raise ValueError("Invalid hook generated")
            
            return hook
            
        except Exception as e:
            self.logger.error(f"Error generating hook: {str(e)}")
            # Improved fallback without ellipsis
            # Extract key topic from brief (first few words)
            topic_words = brief.split()[:5]
            topic = ' '.join(topic_words)
            return f"Discover the essential strategies for {topic} that top professionals use to advance their careers."
    
    def _generate_deck(self, brief: str, hook: str, tone: str) -> str:
        """Generate a 'Deck' (teaser version of the hook)."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert editor. Write a 'Deck' for an article.
                    
                    Definition: A deck is a short summary or teaser that appears below the headline and image.
                    
                    Requirements:
                    - 15-25 words maximum
                    - Match the {tone} tone
                    - Act as a 'lite' version of the hook
                    - Summarize the main value in 5 seconds
                    - Make the reader want to click/read more
                    - Use italics formatting (return plain text, formatting handled by UI)
                    
                    Return only the deck text."""
                },
                {
                    "role": "user",
                    "content": f"Article Brief: {brief}\nHook: {hook}\nTone: {tone}\n\nWrite a Deck (teaser)."
                }
            ]
            
            response = self.llm_client.generate(messages)
            deck = response.content.strip().strip('"').strip("'")
            
            if not deck or len(deck) < 10 or deck.lower() == 'none':
                # Fallback to a shortened version of the hook
                words = hook.split()
                deck = ' '.join(words[:20]) + "..." if len(words) > 20 else hook
            
            return deck
            
        except Exception as e:
            self.logger.error(f"Error generating deck: {str(e)}")
            return "A quick guide to understanding the key concepts and practical applications."
    
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
            
            if not excerpt or len(excerpt) < 50 or excerpt.lower() == 'none':
                raise ValueError("Invalid excerpt generated")

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
            
            if not thesis or len(thesis) < 20 or thesis.lower() == 'none':
                raise ValueError("Invalid thesis generated")

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
            min_words = max(200, int(words_per_section * 0.8))  # 80% of target
            max_words = int(words_per_section * 1.2)  # 120% of target
            
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
                    - Integrate Competitor Insights: Incorporate all "Competitor Must-Haves" across the sections, and dedicate specific focus or sub-points to highlight our "Competitive Edge".
                    
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
                    - Main content sections: {min_words}-{max_words} words each (within +/- 20% of target)
                    - Conclusion: 150-250 words
                    - NO section should exceed {max_words} words (120% of target)
                    - NO section should be under {min_words} words (80% of target)
                    
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

                    TABLE PLANNING RULES:
                    - Across the entire article, plan between 0 and 4 sections with "table" or "comparison" content types.
                    - Plan tables/comparisons ONLY for sections where structured comparison or data presentation adds genuine value. If a section is purely expository and has no data/metrics or options to compare, use "paragraph" or "list". Do not force tables on sections that do not need them.
                    
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

    def _generate_sections_from_outline(self, content_outline: List[str], target_word_count: int, tone: str) -> List[SectionOutline]:
        """Generate detailed section outlines from existing content outline."""
        try:
            # Calculate rough word count distribution
            # Estimate number of main sections (excluding potential non-heading items)
            # This is an estimation, the LLM will refine it
            estimated_sections = max(3, len([item for item in content_outline if isinstance(item, str) and (item.lower().startswith('h2') or item.lower().startswith('section'))]))
            words_per_section = target_word_count // estimated_sections
            
            outline_text = "\n".join([str(item) for item in content_outline])
            
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert content strategist. specific task: Convert an existing article outline into a structured JSON format.
                    
                    REQUIREMENTS:
                    - STRICTLY follow the provided outline structure
                    - Map 'H2' or main items to section titles
                    - Map 'Intent' or descriptions to 'key_points' or 'subtitle'
                    - Assign appropriate word counts (aiming for total ~{target_word_count} words, within +/- 20%)
                    - Match the {tone} tone in any generated text
                    
                    Format as JSON:
                    {{
                        "sections": [
                            {{
                                "title": "Exact Title from Outline",
                                "subtitle": "Derived from Intent/Description",
                                "key_points": ["Key point 1", "Key point 2"],
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
                    "content": f"""Convert this outline into the detailed JSON structure:
                    
                    {outline_text}"""
                }
            ]
            
            response = self.llm_client.generate(messages)
            
            # Parse JSON response using same logic as _generate_sections
            import json
            import re
            
            response_text = response.content.strip()
            json_match = re.search(r'\{[^{}]*"sections"[^{}]*\[.*?\]\s*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            data = json.loads(response_text)
            sections_data = data.get('sections', [])
            
            if not sections_data:
                raise ValueError("No sections found in response")
            
            sections = []
            for i, section_data in enumerate(sections_data):
                section = SectionOutline(
                    title=section_data.get('title', f'Section {i+1}'),
                    subtitle=section_data.get('subtitle'),
                    key_points=section_data.get('key_points', []),
                    word_count_target=section_data.get('word_count_target', 300),
                    content_type=section_data.get('content_type', 'paragraph'),
                    order=section_data.get('order', i+1),
                    importance=section_data.get('importance', 'high')
                )
                sections.append(section)
            
            self.logger.info(f"Successfully parsed {len(sections)} sections from provided outline")
            return sections
            
        except Exception as e:
            self.logger.error(f"Error processing provided outline: {str(e)}")
            # Fallback to standard generation if outline processing fails
            self.logger.info("Falling back to standard section generation")
            return [] # This will be handled by the caller? No, caller expects list.
            # I should handle this path better.
            # Actually, if this fails, we should probably fall back to standard generation in the CALLER.
            # But simpler to re-raise and let caller handle, or return fallback here.
            # Let's return empty list and handle in caller?
            # Or just raise and let the try/catch in generate_structure handle it (it calls _create_fallback_structure)
            raise e
    
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
            deck=f"Essential insights on {brief[:30]}...",
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
def create_article_structure_generator(llm_client, use_verbalized_sampling: bool = False) -> ArticleStructureGenerator:
    """
    Create an article structure generator.
    
    Args:
        llm_client: Configured LLM client
        use_verbalized_sampling: Whether to use verbalized sampling (DISABLED)
        
    Returns:
        ArticleStructureGenerator instance
    """
    return ArticleStructureGenerator(llm_client, False)

# Example usage
if __name__ == "__main__":
    # This would be used with a real LLM client
    print("Article Structure Generator - Ready for integration")
