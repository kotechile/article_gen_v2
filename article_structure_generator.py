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
    core_promise: Optional[str] = None
    viral_metadata: Optional[Dict[str, Any]] = None

SECTION_WORDS_MIN = 450
SECTION_WORDS_TARGET = 550
SECTION_WORDS_MAX = 700


def plan_section_budget(target_words: int, *, min_sections: int = 2, max_sections: int = 12) -> Dict[str, int]:
    """Decide how many sections a requested article length should have.

    Section count has to follow the length, not the other way round: a 2,500-word
    article split ten ways gives 250-word sections, which read as a list of
    summaries no matter how good the writing is. The planner aims at a 450-700
    word section (roughly one argument with evidence), and only goes past
    ``max_sections`` for genuinely long pieces.

    Returns ``section_count``, ``words_per_section`` and the per-section band.

    A very short piece (under ~900 words) cannot reach the band at two sections;
    the band is a target, the section count is the contract.
    """
    target = max(300, int(target_words or 0))
    count = max(min_sections, min(max_sections, round(target / SECTION_WORDS_TARGET)))
    words_per_section = max(200, round(target / count))
    # Too thin per section: add sections while there is room to.
    while words_per_section > SECTION_WORDS_MAX and count < max_sections:
        count += 1
        words_per_section = round(target / count)
    # Too fat per section for a short piece: merge sections.
    while words_per_section < SECTION_WORDS_MIN and count > min_sections:
        count -= 1
        words_per_section = round(target / count)
    return {
        "section_count": count,
        "words_per_section": words_per_section,
        "min_words": max(150, int(words_per_section * 0.8)),
        "max_words": int(words_per_section * 1.2),
    }


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
            raw_wc = (
                research_data.get('target_word_count')
                or research_data.get('articleLength')
                or research_data.get('article_length')
                or 2000
            )
            try:
                target_word_count = int(str(raw_wc).strip() or 2000)
            except Exception:
                target_word_count = 2000

            # Add competitor analysis insights to the brief
            competitor_context = self._build_competitor_context_text(research_data)
            if competitor_context:
                brief = f"{brief}\n\nCompetitor Analysis Insights:\n{competitor_context}"

            dossier_context = self._build_dossier_context_text(research_data)
            brief_with_dossier = f"{brief}\n\nDeep Research Context:\n{dossier_context}" if dossier_context else brief
            
            # Determine target audience
            target_audience = self._determine_target_audience(brief, tone, research_data)
            if target_audience:
                brief_with_dossier = f"{brief_with_dossier}\n\nTarget Audience Profile: {target_audience}"

            # Determine article type based on brief content and research parameters
            article_type = self._determine_article_type(brief, research_data)
            
            # Generate core elements with viral title engine and in-generation review
            draft_title = research_data.get('draft_title', '')
            title_gen_res = self._generate_title(
                brief_with_dossier,
                keywords,
                article_type,
                tone,
                draft_title,
                target_audience=target_audience,
                return_metadata=True,
            )
            if isinstance(title_gen_res, tuple):
                title, core_promise, viral_metadata = title_gen_res
            else:
                title, core_promise, viral_metadata = title_gen_res, f"Key insights into {title_gen_res}", {}

            hook = self._generate_hook(brief_with_dossier, claims, tone, title=title, core_promise=core_promise)
            thesis = self._generate_thesis(brief_with_dossier, claims, evidence, tone)

            # Generate section outlines
            content_outline = research_data.get('content_outline')
            if content_outline and isinstance(content_outline, list) and len(content_outline) > 0:
                self.logger.info(f"Using provided content_outline from database ({len(content_outline)} items)")
                sections = self._generate_sections_from_outline(content_outline, target_word_count, tone)
            else:
                sections = self._generate_sections(brief_with_dossier, claims, evidence, target_word_count, tone, article_type, research_data=research_data)

            # The excerpt has to describe the article this outline plans, so it is
            # written after the sections and checked against them. Generated from the
            # brief alone it kept advertising the intention - sections and numbers the
            # finished piece never contained.
            excerpt = self._generate_excerpt(
                brief_with_dossier,
                claims,
                target_word_count,
                tone,
                sections=sections,
            )
            meta_description = self._generate_meta_description(title, excerpt, keywords)
            
            # Log section titles for debugging
            section_titles = [s.title for s in sections]
            self.logger.info(f"Generated section titles: {section_titles}")
            
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
                call_to_action=call_to_action,
                core_promise=core_promise,
                viral_metadata=viral_metadata,
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
    
    def _generate_title(
        self,
        brief: str,
        keywords: str,
        article_type: str,
        tone: str,
        draft_title: str = '',
        target_audience: str = '',
        return_metadata: bool = False,
    ):
        """
        Generate compelling, viral article title with strict 6-word rule and in-generation review.
        """
        try:
            from src.services.viral_title_engine import ViralTitleEngine
            engine = ViralTitleEngine(llm_client=self.llm_client)
            focus_keyword = str(keywords or "").split(",")[0].strip()
            result = engine.generate_and_review_title(
                brief=brief,
                keywords=keywords,
                article_type=article_type,
                tone=tone,
                draft_title=draft_title,
                target_audience=target_audience,
                focus_keyword=focus_keyword,
            )
            title = result.get("title") or draft_title
            core_promise = result.get("core_promise", f"Key insights into {title}")
            viral_metadata = result

            evaluation = result.get("evaluation") or {}
            self.logger.info(
                f"Generated and reviewed viral title: '{title}' "
                f"(Words: {ViralTitleEngine.count_words(title)}, Trigger: {result.get('trigger_type')}, "
                f"Keyword: {evaluation.get('keyword_status')} for '{focus_keyword}')"
            )
            if evaluation.get("keyword_status") == "missing":
                self.logger.warning(
                    "Title could not carry the focus keyword '%s' within %s chars: '%s'",
                    focus_keyword,
                    len(title),
                    title,
                )
            if return_metadata:
                return title, core_promise, viral_metadata
            return title

        except Exception as e:
            self.logger.error(f"Error in viral title generation and review: {str(e)}")
            # Fallback
            brief_words = brief.split()[:5]
            short_brief = " ".join(brief_words)
            fallback_title = f"{short_brief}..." if len(short_brief) < 55 else f"{short_brief[:55]}..."
            if return_metadata:
                return fallback_title, f"Insights on {short_brief}", {"error": str(e)}
            return fallback_title
    
    def _generate_hook(
        self,
        brief: str,
        claims: List[Dict],
        tone: str,
        title: str = '',
        core_promise: str = '',
    ) -> str:
        """Generate compelling opening hook fulfilling title promise (Zero Bait-and-Switch)."""
        try:
            # Extract key claims for context
            claim_text = "\n".join([claim.get('claim', '') for claim in claims[:3]])
            
            title_constraint = ""
            if title:
                title_constraint = f"""
                CRITICAL - ZERO CLICKBAIT BAIT-AND-SWITCH:
                - The article headline is: "{title}"
                - The core promise to satisfy: "{core_promise or title}"
                - The hook MUST immediately anchor the reader by addressing and beginning to resolve this specific headline promise.
                - Never bait the reader with a catchy headline only to write a generic opening that delays or ignores it.
                """

            messages = [
                {
                    "role": "system",
                    "content": f"""You are a master storyteller and viral editorial hook writer. Write a compelling opening hook for an article.
                    
                    Requirements:
                    - 1-2 sentences maximum
                    - Create immediate engagement and resolve curiosity
                    - Match the {tone} tone
                    {title_constraint}
                    - Use a surprising fact, statistic, question, or bold statement
                    - Set up the article's value proposition without fluff
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
                    "content": f"Article Headline: {title or 'N/A'}\nArticle Topic: {brief}\nKey Claims: {claim_text}\nTone: {tone}\n\nWrite a compelling hook that immediately delivers on the headline promise."
                }
            ]
            
            response = self.llm_client.generate(messages)
            hook = response.content.strip().strip('"').strip("'")
            
            # Clean up any unwanted ellipsis that might have been added
            if hook.endswith('...') and not hook.endswith('...?'):
                hook = hook[:-3].rstrip()
            
            # Ensure hook is complete and engaging
            if not hook or len(hook) < 20 or hook.lower() == 'none':
                self.logger.warning(f"Generated hook is invalid or too short: '{hook}'. Using fallback.")
                raise ValueError("Invalid hook generated")
            
            return hook
            
        except Exception as e:
            self.logger.error(f"Error generating hook: {str(e)}")
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
    
    @staticmethod
    def _outline_corpus(sections: Optional[List[SectionOutline]]) -> str:
        """The planned sections rendered as the excerpt's source of truth."""
        lines: List[str] = []
        for index, section in enumerate(sections or [], start=1):
            parts = [f"{index}. {section.title}"]
            if section.subtitle:
                parts.append(f"   Focus: {section.subtitle}")
            for point in section.key_points or []:
                parts.append(f"   - {point}")
            lines.append("\n".join(parts))
        return "\n".join(lines)

    def _request_excerpt(self, system_content: str, user_content: str) -> str:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        response = self.llm_client.generate(messages)
        excerpt = response.content.strip().strip('"').strip("'").strip()

        if not excerpt or len(excerpt) < 50 or excerpt.lower() == 'none':
            raise ValueError("Invalid excerpt generated")
        return excerpt

    def _enforce_outline_alignment(
        self,
        excerpt: str,
        system_content: str,
        user_content: str,
        sections: Optional[List[SectionOutline]],
        outline_text: str,
    ) -> str:
        """Reject an excerpt the planned sections do not support.

        The excerpt is printed under the title and reused for social cards, so a
        promise about a section that is not in the outline is a broken page. Two
        grounded repairs: ask again with the failing terms named, then rebuild the
        excerpt from the outline's own key points. Nothing here invents material.
        """
        from src.services.excerpt_alignment import (
            derive_excerpt_from_outline,
            evaluate_excerpt_alignment,
        )

        report = evaluate_excerpt_alignment(excerpt, outline_text)
        attempt = 0
        while not report["matches"] and attempt < 2:
            attempt += 1
            self.logger.warning(
                "Excerpt rejected against the outline (attempt %s/%s): %s",
                attempt,
                2,
                report["reason"],
            )
            retry_user_content = (
                f"{user_content}\n\n"
                f"Your previous excerpt was rejected: {report['reason']}.\n"
                "Rewrite it using only what the planned sections above establish. "
                "Drop every claim, term and figure the sections do not contain."
            )
            try:
                candidate = self._request_excerpt(system_content, retry_user_content)
            except Exception as retry_error:
                self.logger.warning("Excerpt retry failed: %s", retry_error)
                break
            candidate_report = evaluate_excerpt_alignment(candidate, outline_text)
            if candidate_report["matches"]:
                self.logger.info(
                    "Excerpt accepted on attempt %s (support %s, %s terms)",
                    attempt + 1,
                    candidate_report["support"],
                    candidate_report["terms"],
                )
                return candidate
            excerpt, report = candidate, candidate_report

        if report["matches"]:
            self.logger.info(
                "Excerpt check: supported by the outline (support %s, %s terms)",
                report["support"],
                report["terms"],
            )
            return excerpt

        derived = derive_excerpt_from_outline(sections)
        if derived:
            self.logger.warning(
                "Excerpt replaced with one built from the outline's key points: %s",
                report["reason"],
            )
            return derived

        self.logger.warning(
            "Excerpt kept although the outline does not support it: %s", report["reason"]
        )
        return excerpt

    def _generate_excerpt(
        self,
        brief: str,
        claims: List[Dict],
        target_word_count: int,
        tone: str,
        sections: Optional[List[SectionOutline]] = None,
    ) -> str:
        """Generate the article excerpt, grounded in the planned outline."""
        try:
            # Calculate excerpt length based on target word count
            excerpt_length = min(150, max(50, target_word_count // 15))

            outline_text = self._outline_corpus(sections)
            if outline_text:
                system_content = f"""You are an expert content writer. Write the excerpt (standfirst) for this article.

                    Requirements:
                    - Approximately {excerpt_length} words, 1 to 3 sentences
                    - Summarize what THIS OUTLINE promises the reader: the sections it plans and the outcome they deliver
                    - Every claim and every number must come from the planned sections below. Never introduce a statistic, comparison, or benefit they do not contain.
                    - Match the {tone} tone
                    - Be specific and concrete: name the problem and the outcome
                    - No generic urgency, no "in today's world", no "this article explores", no mention of sections, outlines, or the article itself

                    Return only the excerpt, no quotes or formatting."""
                user_content = (
                    f"Article Brief: {brief}\n\n"
                    f"Planned sections (the excerpt may only summarize these):\n{outline_text}\n\n"
                    f"Target Word Count: {target_word_count}\nTone: {tone}"
                )
            else:
                system_content = f"""You are an expert content writer. Write a compelling article excerpt.

                    Requirements:
                    - Approximately {excerpt_length} words
                    - Match the {tone} tone
                    - Summarize the article's main value proposition
                    - Include key benefits or outcomes
                    - Create urgency or interest
                    - Be specific and actionable

                    Return only the excerpt, no quotes or formatting."""
                user_content = f"Article Brief: {brief}\nTarget Word Count: {target_word_count}\nTone: {tone}"

            excerpt = self._request_excerpt(system_content, user_content)
            if outline_text:
                excerpt = self._enforce_outline_alignment(
                    excerpt,
                    system_content,
                    user_content,
                    sections,
                    outline_text,
                )

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
                    "content": f"""You are an expert executive editor and strategist. Write a bold axiom or contrarian statement for "The Big Picture" of this article.
                    
                    Requirements:
                    - Exactly 1-2 sentences maximum
                    - Match the {tone} tone
                    - Highlight the fundamental tension, market shift, or architectural trap immediately
                    - Be bold, specific, and debatable (state the hard truth or counter-intuitive reality)
                    - Cut all introductory throat-clearing (never start with "This article explores...", "In today's...", or passive filler)
                    
                    Return only the statement, no quotes or formatting."""
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
            return f"The fundamental tension in {brief[:80]} lies between short-term execution speed and uncalculated architectural debt."
    
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
        """Generate detailed section outlines with balanced word distribution."""
        try:
            research_data = research_data or {}
            # Section count follows the requested length (see plan_section_budget):
            # a 2,500-word request is four to five sections that can carry an
            # argument each, not ten that can only carry a summary.
            budget = plan_section_budget(target_word_count)
            section_count = budget["section_count"]
            min_sections = section_count
            max_sections = section_count
            words_per_section = budget["words_per_section"]
            min_words = budget["min_words"]
            max_words = budget["max_words"]
            self.logger.info(
                "Section plan: %s sections x ~%s words for a %s-word request",
                section_count,
                words_per_section,
                target_word_count,
            )
                       # Prepare context
            claims_text = "\n".join([f"- {claim.get('claim', '')}" for claim in claims[:5]])
            evidence_text = f"Evidence from {len(evidence)} sources" if evidence else "Research-based insights"
            
            # Analyze evidence distribution for better section planning
            evidence_types = self._analyze_evidence_distribution(evidence)
            
            # Prepare controversial topics context if available
            selected_controversies = research_data.get('selected_controversies', [])
            controversies_prompt_text = ""
            if selected_controversies:
                controversies_prompt_text = "\n\nCRITICAL DIRECTIVE - INTEGRATE SELECTED CONTROVERSIAL TOPICS:\n"
                controversies_prompt_text += "The user has selected specific controversial topics and chosen a 'take' (opinion) to defend for this article. You MUST plan the article sections to address these controversies. For each controversy:\n"
                controversies_prompt_text += "1. Dedicate a section (or integrate it cleanly as a major point in a section) that defends the chosen take.\n"
                controversies_prompt_text += "2. The section must also present the alternative viewpoints fairly, leaving the reader to think and make their own choice in the end.\n"
                for index, c in enumerate(selected_controversies):
                    controversies_prompt_text += f"- Controversy {index+1}: \"{c.get('title')}\"\n"
                    controversies_prompt_text += f"  * Summary of debate: {c.get('summary')}\n"
                    controversies_prompt_text += f"  * Chosen Take to defend: \"{c.get('selected_take_text')}\"\n"
                    takes_list = c.get('takes', []) or []
                    alt_takes = [t.get('text') for t in takes_list if t.get('text') != c.get('selected_take_text')]
                    controversies_prompt_text += f"  * Alternative viewpoints to mention and leave open: {alt_takes}\n"

            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert content strategist. Create a detailed, balanced outline for a {article_type} article.
                    
                    CRITICAL REQUIREMENTS:
                    - Create EXACTLY {section_count} logical main sections for a {target_word_count}-word article (the introduction and the conclusion are two of them). Do not add extra sections: each one must be substantial enough to carry an argument with evidence, not a summary.
                    - Each section should be {min_words}-{max_words} words (target: ~{words_per_section} words)
                    - Ensure BALANCED content distribution - no single section should dominate
                    - Match the {tone} tone
                    - Include introduction and conclusion sections
                    - Order sections logically with smooth transitions
                    - Include practical, actionable content
                    - Distribute evidence and claims evenly across sections
                    - COMPLETE TOPIC & CONCEPT COVERAGE: Ensure the outline plans sections and key points that cover and fully develop all core themes, claims, statistics, and strategic comparisons (e.g., build vs. buy, cost/ROI disconnects, execution roadmaps, governance) present in the brief and claims, so that every concept highlighted in summaries and takeaways is thoroughly developed in the body sections.
                    - Integrate Competitor Insights: Incorporate all "Competitor Must-Haves" across the sections, and dedicate specific focus or sub-points to highlight our "Competitive Edge".{controversies_prompt_text}
                    
                    ⚠️ CRITICAL: ACTION-ORIENTED OR METRIC-DRIVEN SECTION HEADERS ⚠️
                    - Every substantive section title MUST be action-oriented or metric-driven (e.g., "Slashing Database Latency by 40%", "Auditing Distributed Lock Contention", "The $250k Build-vs-Buy Pitfall").
                    - DO NOT use generic titles like: "Getting Started", "Step-by-Step Process", "Key Concepts", "Practical Applications", "Understanding the Fundamentals", "Real-World Implementation", "Conclusion", "Summary".
                    - For each substantive section, plan key points to cover:
                      1. Context: Core problem, tool version, or financial stake stated immediately.
                      2. Key Metrics/Terms: Why it matters, empirical mechanics/constraints, and cost trade-offs or hidden failure modes.
                    - Ensure the outline also accounts for the closing modules:
                      * "What I'd Watch" (planning 3 indicators: benchmark to monitor, operational friction point, breakdown point).
                      * "The Catch" (acknowledging limitations, constraints, and counterarguments).
                      * "At a Glance" (The Big Shift, Why It Matters, The Bottom Line).
                    
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
                    
                    CONTENT TYPE SELECTION:
                    - Choose the most appropriate content type for each section:
                    - "paragraph": Standard narrative text
                    - "list": Bulleted or numbered lists
                    - "step_by_step": Instructional content (only use if the article is actually a step-by-step guide)
                    - "comparison": Side-by-side comparisons
                    - "table": Data-rich content with tables
 
                    TABLE PLANNING RULES:
                    - Across the entire article, plan between 1 and 4 sections with "table" or "comparison" content types (so at least one table/comparison is always included to explain concepts and numbers).
                    - Plan tables/comparisons for sections where structured comparison, data presentation, concepts, or numbers add genuine value.
                    
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
{controversies_prompt_text}

Create EXACTLY {section_count} topic-specific sections for this article's target length ({target_word_count} words, ~{words_per_section} words each, introduction and conclusion included). Each section title should be unique to this topic, not a generic template. Do not split the subject into more sections than this: depth per section beats a longer table of contents."""
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
                if len(sections) > section_count:
                    # The planner's count is the contract; extra sections are what
                    # turns a 2,500-word article into ten summaries.
                    self.logger.warning(
                        "Planner asked for %s sections, the model returned %s; keeping the first %s and dropping %s.",
                        section_count,
                        len(sections),
                        section_count,
                        len(sections) - section_count,
                    )
                    sections = sections[:section_count]
                elif len(sections) < section_count:
                    self.logger.warning(
                        "Planner asked for %s sections, the model returned %s; keeping the shorter outline.",
                        section_count,
                        len(sections),
                    )
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
    
    def _determine_target_audience(self, brief: str, tone: str, research_data: Optional[Dict[str, Any]] = None) -> str:
        """Determine target audience based on research_data, brief, and tone."""
        if research_data and research_data.get('target_audience'):
            audience = str(research_data['target_audience']).strip()
            if audience:
                return audience

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
        """Create fallback section outlines with topic-specific structure based on brief and claims."""
        # Same planner as the main path: the fallback must not produce a different
        # shape of article than the outline path.
        budget = plan_section_budget(target_word_count)
        target_count = budget["section_count"]

        # Analyze brief to create more relevant sections
        brief_lower = brief.lower()
        brief_words = [w for w in brief.split() if len(w) > 3][:6]
        main_topic = brief_words[0].title() if brief_words else "Strategy"
        
        # Extract key themes from claims if available
        claim_themes = []
        if claims:
            for claim in claims[:6]:
                claim_text = claim.get('claim', '')
                words = [w for w in claim_text.split() if w.lower() not in ['the', 'a', 'an', 'is', 'are', 'and', 'or', 'but']]
                if words:
                    claim_themes.append(' '.join(words[:3]))

        words_per_section = budget["words_per_section"]

        # Build dynamic section list matching target_count
        sections: List[SectionOutline] = [
            SectionOutline(
                title="Introduction",
                key_points=["Overview of the topic", "Why this matters", "Key takeaways preview"],
                word_count_target=words_per_section,
                order=1,
                importance="high"
            )
        ]

        # Candidate middle topic titles
        middle_titles = []
        for i in range(1, len(brief_words)):
            middle_titles.append(f"{brief_words[i].title()} Analysis & Frameworks")
        if claim_themes:
            for ct in claim_themes:
                middle_titles.append(f"{ct.title()} Dynamics")
        default_titles = [
            f"Core Dynamics of {main_topic}",
            f"Key Financial & Operational Metrics",
            f"Comparative Frameworks & Realities",
            f"Strategic Implementation Steps",
            f"Risk Management & Optimization",
            f"Future Outlook & High-Impact Decisions",
        ]
        for dt in default_titles:
            if len(middle_titles) < target_count:
                middle_titles.append(dt)

        # Add middle sections
        middle_count = max(1, target_count - 2)
        for idx in range(middle_count):
            title = middle_titles[idx] if idx < len(middle_titles) else f"Key Consideration {idx+1}"
            c_type = "table" if idx % 2 == 1 else "paragraph"
            sections.append(
                SectionOutline(
                    title=title,
                    key_points=[f"Primary factor for {title}", f"Comparative assessment", "Actionable guidance"],
                    word_count_target=words_per_section,
                    content_type=c_type,
                    order=len(sections) + 1,
                    importance="high"
                )
            )

        # Conclusion
        sections.append(
            SectionOutline(
                title="Conclusion",
                key_points=["Summary of strategic takeaways", "Next action steps", "Final perspective"],
                word_count_target=words_per_section,
                order=len(sections) + 1,
                importance="medium"
            )
        )

        self.logger.info(f"Created {len(sections)} fallback sections (target_word_count={target_word_count}): {[s.title for s in sections]}")
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
