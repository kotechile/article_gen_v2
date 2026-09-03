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
    LINKEDIN_ARTICLE = "linkedin_article"
    LINKEDIN_POST = "linkedin_post"

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
    component_type: str = "tactical_insight"  # "lead", "tension", "tactical_insight", "nuanced_takeaway"

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
            
            # Determine article type based on brief content and research parameters
            article_type = self._determine_article_type(brief, research_data)
            
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
                sections = self._generate_sections(brief_with_dossier, claims, evidence, target_word_count, tone, article_type, research_data=research_data)
            
            # Log section titles for debugging
            section_titles = [f"[{s.component_type.upper()}] {s.title}" for s in sections]
            self.logger.info(f"Generated 4-component section titles: {section_titles}")
            
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
    
    def _determine_article_type(self, brief: str, research_data: Optional[Dict[str, Any]] = None) -> str:
        """Determine article type based on brief content and research metadata."""
        if research_data:
            explicit_type = research_data.get('article_type') or research_data.get('target_format')
            if explicit_type in [ArticleType.LINKEDIN_ARTICLE.value, 'linkedin_article', 'linkedin']:
                return ArticleType.LINKEDIN_ARTICLE.value
            if explicit_type in [ArticleType.LINKEDIN_POST.value, 'linkedin_post']:
                return ArticleType.LINKEDIN_POST.value
            if research_data.get('target_platform') == 'linkedin':
                target_words = int(research_data.get('target_word_count', 1000))
                return ArticleType.LINKEDIN_POST.value if target_words <= 500 else ArticleType.LINKEDIN_ARTICLE.value

        brief_lower = brief.lower()
        if 'linkedin post' in brief_lower or 'linkedin micro' in brief_lower:
            return ArticleType.LINKEDIN_POST.value
        elif 'linkedin article' in brief_lower or 'linkedin' in brief_lower:
            return ArticleType.LINKEDIN_ARTICLE.value
        elif any(word in brief_lower for word in ['list', 'top', 'best', 'worst', 'ranking', 'countdown']):
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
                          target_word_count: int, tone: str, article_type: str,
                          research_data: Optional[Dict[str, Any]] = None) -> List[SectionOutline]:
        """
        Generate detailed section outlines enforcing the 4-Component Narrative Sequence:
        1. Lead: Focused opening evidence, immediate stakes, and high-impact hook (~15% words)
        2. Tension: Systemic market evidence, structural friction, and root causes (~25% words)
        3. Tactical Insight: Deep practitioner step-by-step guidance, frameworks, and comparison tables (~45% words)
        4. Nuanced Takeaway: Honest limitations, edge cases, trade-offs, and strategic synthesis (~15% words)
        """
        try:
            research_dict = research_data or {}
            
            # Word count budget per component
            lead_words = max(180, int(target_word_count * 0.15))
            tension_words = max(250, int(target_word_count * 0.25))
            nuanced_words = max(180, int(target_word_count * 0.15))
            tactical_total_words = max(350, target_word_count - (lead_words + tension_words + nuanced_words))
            
            # Determine if tactical insight should be split into multiple subsections
            tactical_subsections_count = 2 if target_word_count >= 1800 else 1
            tactical_words_per_sec = tactical_total_words // tactical_subsections_count
            
            claims_text = "\n".join([f"- {claim.get('claim', '')}" for claim in claims[:6]])
            evidence_text = f"Evidence from {len(evidence)} sources" if evidence else "Research-based insights"
            evidence_types = self._analyze_evidence_distribution(evidence)
            
            # Controversies context if available
            selected_controversies = research_dict.get('selected_controversies', [])
            controversies_prompt_text = ""
            if selected_controversies:
                controversies_prompt_text = "\n\nCRITICAL DIRECTIVE - INTEGRATE SELECTED CONTROVERSIAL TOPICS:\n"
                for index, c in enumerate(selected_controversies):
                    controversies_prompt_text += f"- Controversy {index+1}: \"{c.get('title')}\"\n"
                    controversies_prompt_text += f"  * Summary of debate: {c.get('summary')}\n"
                    controversies_prompt_text += f"  * Chosen Take to defend: \"{c.get('selected_take_text')}\"\n"

            messages = [
                {
                    "role": "system",
                    "content": f"""You are an elite editorial strategist and content architect. Create a high-authority outline for a {article_type} article.

MANDATORY 4-COMPONENT NARRATIVE ARCHITECTURE:
Every article MUST be structured into exactly these 4 sequential components:

1. COMPONENT 1: LEAD (component_type="lead")
   - Goal: High-impact opening hook anchored directly in concrete opening evidence (empirical data, key benchmark, or case signal).
   - Target Word Count: ~{lead_words} words.
   - Requirement: Create an engaging, topic-specific title (e.g. "The $4.2B Blind Spot Behind Modern DevOps").

2. COMPONENT 2: TENSION (component_type="tension")
   - Goal: Broaden the scope by uncovering systemic market evidence, structural friction, conflicting incentives, and why conventional solutions break.
   - Target Word Count: ~{tension_words} words.
   - Requirement: Create a topic-specific title that captures the systemic conflict (e.g. "Why Standard Automation Creates Hidden Bottlenecks").

3. COMPONENT 3: TACTICAL INSIGHT (component_type="tactical_insight")
   - Goal: The execution core of the article. Granular practitioner guidance, step-by-step playbooks, actionable frameworks, and comparative data tables.
   - Subsections: Exactly {tactical_subsections_count} section(s) totaling ~{tactical_total_words} words (approx ~{tactical_words_per_sec} words each).
   - Requirement: At least one section MUST have content_type="table" or "comparison" with rich comparative metrics.

4. COMPONENT 4: NUANCED TAKEAWAY & COUNTER-ARGUMENTS (component_type="nuanced_takeaway")
   - Goal: Intellectual honesty and rigor. Explore edge cases, where this approach fails, counter-perspectives, trade-offs, and strategic synthesis.
   - Target Word Count: ~{nuanced_words} words.
   - Requirement: Create a topic-specific title emphasizing honest limits and strategic foresight.

CRITICAL RULES:
- Match the requested tone: {tone}
- NO GENERIC TITLES (Do NOT use "Lead", "Tension", "Tactical Insight", "Conclusion", "Getting Started"). Use specific, compelling titles tailored to the topic.
- Integrate competitor insights and unique angles.{controversies_prompt_text}

Format output ONLY as valid JSON:
{{
    "sections": [
        {{
            "title": "Topic-Specific Title for Lead",
            "subtitle": "Optional subtitle",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "word_count_target": {lead_words},
            "content_type": "paragraph",
            "component_type": "lead",
            "order": 1,
            "importance": "high"
        }},
        {{
            "title": "Topic-Specific Title for Tension",
            "subtitle": "Optional subtitle",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "word_count_target": {tension_words},
            "content_type": "paragraph",
            "component_type": "tension",
            "order": 2,
            "importance": "high"
        }},
        {{
            "title": "Topic-Specific Title for Tactical Insight",
            "subtitle": "Optional subtitle",
            "key_points": ["Step 1", "Step 2", "Step 3"],
            "word_count_target": {tactical_words_per_sec},
            "content_type": "table",
            "component_type": "tactical_insight",
            "order": 3,
            "importance": "high"
        }},
        {{
            "title": "Topic-Specific Title for Nuanced Takeaways & Limitations",
            "subtitle": "Optional subtitle",
            "key_points": ["Limitation 1", "Edge Case 2", "Strategic Takeaway 3"],
            "word_count_target": {nuanced_words},
            "content_type": "paragraph",
            "component_type": "nuanced_takeaway",
            "order": 4,
            "importance": "high"
        }}
    ]
}}"""
                },
                {
                    "role": "user",
                    "content": f"""Article Brief: {brief}

Key Claims:
{claims_text}

Evidence Distribution:
{evidence_types}

Target Word Count: {target_word_count}
Tone: {tone}

Generate the 4-component sequential outline in JSON format."""
                }
            ]

            response = self.llm_client.generate(messages)
            response_text = response.content.strip()

            import json
            import re

            json_match = re.search(r'\{[^{}]*"sections"[^{}]*\[.*?\]\s*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)

            try:
                data = json.loads(response_text)
                sections_data = data.get('sections', [])

                if not sections_data or len(sections_data) < 3:
                    raise ValueError("Insufficient sections in response")

                # Map component types sequentially if missing
                default_component_types = ["lead", "tension", "tactical_insight", "nuanced_takeaway"]
                sections = []
                for i, section_data in enumerate(sections_data):
                    comp_type = section_data.get('component_type')
                    if not comp_type or comp_type not in default_component_types:
                        if i == 0:
                            comp_type = "lead"
                        elif i == 1:
                            comp_type = "tension"
                        elif i == len(sections_data) - 1:
                            comp_type = "nuanced_takeaway"
                        else:
                            comp_type = "tactical_insight"

                    section = SectionOutline(
                        title=section_data.get('title', f'Section {i+1}'),
                        subtitle=section_data.get('subtitle'),
                        key_points=section_data.get('key_points', []),
                        word_count_target=section_data.get('word_count_target', 300),
                        content_type=section_data.get('content_type', 'paragraph'),
                        order=section_data.get('order', i+1),
                        importance=section_data.get('importance', 'high'),
                        component_type=comp_type,
                    )
                    sections.append(section)

                self.logger.info(f"Successfully generated {len(sections)} sections following 4-component narrative")
                return sections

            except Exception as parse_err:
                self.logger.warning(f"Failed to parse 4-component JSON outline: {parse_err}. Using fallback.")
                return self._create_fallback_sections(brief, target_word_count, claims)

        except Exception as e:
            self.logger.error(f"Error in _generate_sections: {e}", exc_info=True)
            return self._create_fallback_sections(brief, target_word_count, claims)

    def _generate_sections_from_outline(self, content_outline: List[str], target_word_count: int, tone: str) -> List[SectionOutline]:
        """Generate detailed section outlines from existing content outline while mapping to 4 components."""
        try:
            outline_text = "\n".join([str(item) for item in content_outline])
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert editor. Transform the provided outline into our 4-Component Narrative Sequence:
1. Lead (component_type="lead"): Focused opening evidence & hook (~15% words)
2. Tension (component_type="tension"): Systemic market evidence & friction (~25% words)
3. Tactical Insight (component_type="tactical_insight"): Step-by-step guidance & comparative tables (~45% words)
4. Nuanced Takeaways (component_type="nuanced_takeaway"): Honest limitations & strategic synthesis (~15% words)

Total target words: {target_word_count}. Return ONLY valid JSON with 'sections' array."""
                },
                {
                    "role": "user",
                    "content": f"Provided Outline:\n{outline_text}"
                }
            ]
            response = self.llm_client.generate(messages)
            import json, re
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
                comp_type = section_data.get('component_type')
                if not comp_type:
                    comp_type = "lead" if i == 0 else ("tension" if i == 1 else ("nuanced_takeaway" if i == len(sections_data) - 1 else "tactical_insight"))
                sections.append(SectionOutline(
                    title=section_data.get('title', f'Section {i+1}'),
                    subtitle=section_data.get('subtitle'),
                    key_points=section_data.get('key_points', []),
                    word_count_target=section_data.get('word_count_target', 300),
                    content_type=section_data.get('content_type', 'paragraph'),
                    order=section_data.get('order', i+1),
                    importance=section_data.get('importance', 'high'),
                    component_type=comp_type,
                ))
            return sections
        except Exception as e:
            self.logger.warning(f"Failed to process custom outline: {e}. Falling back to 4-component standard.")
            return self._create_fallback_sections(outline_text[:120], target_word_count)

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
        if article_type in [ArticleType.LINKEDIN_ARTICLE.value, ArticleType.LINKEDIN_POST.value]:
            return "What's your take on this? How do you approach this in your team or workflow? Drop your perspective in the comments below 👇"
        elif article_type == ArticleType.HOW_TO.value:
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
        """Create fallback section outlines strictly structured into the 4 Narrative Components."""
        brief_words = [w for w in brief.split() if len(w) > 3][:3]
        main_topic = ' '.join(brief_words).title() if brief_words else "Core Strategy"

        lead_words = max(180, int(target_word_count * 0.15))
        tension_words = max(250, int(target_word_count * 0.25))
        nuanced_words = max(180, int(target_word_count * 0.15))
        tactical_words = max(350, target_word_count - (lead_words + tension_words + nuanced_words))

        sections = [
            SectionOutline(
                title=f"The Reality Behind {main_topic}",
                key_points=["Current landscape and evidence", "Immediate stakes", "The core premise"],
                word_count_target=lead_words,
                content_type="paragraph",
                component_type="lead",
                order=1,
                importance="high",
            ),
            SectionOutline(
                title=f"Why Conventional Approaches to {main_topic} Fail",
                key_points=["Structural market friction", "Hidden systemic bottlenecks", "Why legacy solutions break"],
                word_count_target=tension_words,
                content_type="paragraph",
                component_type="tension",
                order=2,
                importance="high",
            ),
            SectionOutline(
                title=f"The Practitioner Framework for {main_topic}",
                key_points=["Step-by-step implementation", "Comparative breakdown", "Proven methodology"],
                word_count_target=tactical_words,
                content_type="table",
                component_type="tactical_insight",
                order=3,
                importance="high",
            ),
            SectionOutline(
                title=f"Trade-offs, Edge Cases, and Long-Term Outlook",
                key_points=["When not to use this", "Critical failure modes", "Strategic synthesis"],
                word_count_target=nuanced_words,
                content_type="paragraph",
                component_type="nuanced_takeaway",
                order=4,
                importance="high",
            ),
        ]
        self.logger.info(f"Created 4-component fallback sections: {[s.title for s in sections]}")
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
