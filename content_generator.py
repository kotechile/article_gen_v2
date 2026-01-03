"""
Content Generator for Content Generator V2.

This module generates detailed content for each article section based on
the structure, claims, and evidence.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

# Import verbalized sampling client
from verbalized_sampling_client import VerbalizedSamplingClient, create_verbalized_sampling_client

# Configure logging
logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Types of content that can be generated."""
    PARAGRAPH = "paragraph"
    LIST = "list"
    STEP_BY_STEP = "step_by_step"
    COMPARISON = "comparison"
    CASE_STUDY = "case_study"
    QUOTE = "quote"
    STATISTIC = "statistic"
    TABLE = "table"

@dataclass
class ContentBlock:
    """A single content block within a section."""
    content: str
    content_type: str
    word_count: int
    citations: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

@dataclass
class SectionContent:
    """Complete content for a single section."""
    title: str
    subtitle: Optional[str]
    content_blocks: List[ContentBlock]
    total_word_count: int
    key_points_covered: List[str]
    citations: List[Dict[str, Any]]
    section_order: int

class ContentGenerator:
    """
    Generates detailed content for article sections.
    """
    
    def __init__(self, llm_client, use_verbalized_sampling: bool = True):
        """
        Initialize the content generator.
        
        Args:
            llm_client: Configured LLM client
            use_verbalized_sampling: Whether to use verbalized sampling for improved content quality
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
            self.logger.info("Verbalized sampling enabled for content generation")
        else:
            self.verbalized_client = None
            self.logger.info("Verbalized sampling disabled, using standard generation")
    
    def generate_section_content(self, section_outline: Dict[str, Any], 
                               research_data: Dict[str, Any], 
                               claims: List[Dict], 
                               evidence: List[Dict],
                               previous_sections: List[SectionContent] = None) -> SectionContent:
        """
        Generate detailed content for a single section.
        
        Args:
            section_outline: Section outline from structure generator
            research_data: Research parameters and brief
            claims: Extracted claims from research
            evidence: Collected evidence
            previous_sections: Previously generated sections for context
            
        Returns:
            Complete SectionContent object
        """
        try:
            section_title = section_outline.get('title', 'Untitled Section')
            section_subtitle = section_outline.get('subtitle')
            key_points = section_outline.get('key_points', [])
            word_count_target = section_outline.get('word_count_target', 300)
            content_type = section_outline.get('content_type', 'paragraph')
            section_order = section_outline.get('order', 1)
            
            # Generate content blocks for this section
            content_blocks = self._generate_content_blocks(
                section_title, section_subtitle, key_points, word_count_target,
                content_type, research_data, claims, evidence, previous_sections
            )
            
            # Extract all citations from content blocks
            all_citations = []
            for block in content_blocks:
                if block.citations:
                    all_citations.extend(block.citations)
            
            # Calculate total word count
            total_word_count = sum(block.word_count for block in content_blocks)
            
            # Determine which key points were covered
            covered_points = self._extract_covered_points(content_blocks, key_points)
            
            section_content = SectionContent(
                title=section_title,
                subtitle=section_subtitle,
                content_blocks=content_blocks,
                total_word_count=total_word_count,
                key_points_covered=covered_points,
                citations=all_citations,
                section_order=section_order
            )
            
            self.logger.info(f"Generated content for section '{section_title}' with {total_word_count} words")
            return section_content
            
        except Exception as e:
            self.logger.error(f"Error generating section content: {str(e)}")
            return self._create_fallback_section_content(section_outline)
    
    def _generate_content_blocks(self, title: str, subtitle: Optional[str], 
                               key_points: List[str], word_count_target: int,
                               content_type: str, research_data: Dict[str, Any],
                               claims: List[Dict], evidence: List[Dict],
                               previous_sections: List[SectionContent] = None) -> List[ContentBlock]:
        """Generate content blocks for a section."""
        content_blocks = []
        
        try:
            # Prepare context for content generation
            context = self._prepare_content_context(
                title, subtitle, key_points, research_data, claims, evidence, previous_sections
            )
            
            # Generate main content based on content type
            if content_type == "list":
                content_blocks.extend(self._generate_list_content(context, word_count_target))
            elif content_type == "step_by_step":
                content_blocks.extend(self._generate_step_by_step_content(context, word_count_target))
            elif content_type == "comparison":
                content_blocks.extend(self._generate_comparison_content(context, word_count_target))
            elif content_type == "table":
                # Only use table content if we have evidence/data to work with
                if context.get('relevant_evidence') and len(context['relevant_evidence']) > 0:
                    content_blocks.extend(self._generate_table_content(context, word_count_target))
                else:
                    # Fall back to paragraph content if no evidence available
                    content_blocks.extend(self._generate_paragraph_content(context, word_count_target))
            else:  # Default to paragraph
                content_blocks.extend(self._generate_paragraph_content(context, word_count_target))
            
            # Add supporting content if needed - be more aggressive about meeting word count
            current_word_count = sum(block.word_count for block in content_blocks)
            if current_word_count < word_count_target * 0.9:  # Raise threshold from 0.8 to 0.9
                remaining_words = word_count_target - current_word_count
                self.logger.info(f"Content for '{title}' is {current_word_count} words (target: {word_count_target}), generating {remaining_words} more words")
                # Pass evidence in context for supporting content
                supporting_context = context.copy()
                additional_blocks = self._generate_supporting_content(supporting_context, remaining_words)
                content_blocks.extend(additional_blocks)
            
            # Check again and add more content if still short
            final_word_count = sum(block.word_count for block in content_blocks)
            if final_word_count < word_count_target * 0.85:
                remaining_words = word_count_target - final_word_count
                self.logger.warning(f"Content for '{title}' is still only {final_word_count} words, generating {remaining_words} more words")
                # Pass evidence in context for supporting content
                supporting_context = context.copy()
                additional_blocks = self._generate_supporting_content(supporting_context, remaining_words)
                content_blocks.extend(additional_blocks)
            
            # Balance word count if section is extremely long
            final_word_count = sum(block.word_count for block in content_blocks)
            if final_word_count > word_count_target * 1.5:
                content_blocks = self._balance_content_blocks(content_blocks, word_count_target)
            
            return content_blocks
            
        except Exception as e:
            self.logger.error(f"Error generating content blocks: {str(e)}")
            return [self._create_fallback_content_block(title, word_count_target)]
    
    def _format_evidence_for_citations(self, evidence: List[Dict]) -> str:
        """Format evidence for citation instructions."""
        if not evidence:
            return "No evidence sources available - write based on general knowledge and best practices, but still create complete, clear content with comparative tables when appropriate"
        
        # Filter out evidence with no content
        valid_evidence = [ev for ev in evidence if ev.get('content') and ev.get('content').strip()]
        
        if not valid_evidence:
            self.logger.warning(f"All {len(evidence)} evidence items have empty content - proceeding without evidence")
            return "No valid evidence sources available - write based on general knowledge and best practices, but still create complete, clear content with comparative tables when appropriate"
        
        formatted_evidence = []
        for i, ev in enumerate(valid_evidence, start=1):
            # Try multiple field names for title
            title = ev.get('title') or ev.get('source_title') or ev.get('source', 'Unknown Source')
            # Get full content for RAG sources (they contain valuable detailed information)
            # For RAG sources, use full content; for web sources, use longer snippet (1000 chars)
            source_type = ev.get('source_type', 'unknown')
            if source_type == 'rag':
                # Use full content for RAG - it contains structured, valuable information
                content = ev.get('content', '').strip()
            else:
                # For web sources, use longer snippet (1000 chars instead of 300)
                content = ev.get('content', '')[:1000] if ev.get('content') else ''
            # Get source URL
            source = ev.get('source') or ev.get('url', 'Unknown URL')
            
            # Check if this is a researched example for a specific instruction
            instruction_marker = ""
            if ev.get('instruction_topic'):
                instruction_marker = f" [RESEARCHED EXAMPLE for: {ev.get('instruction_topic')}]"
            
            # Format with proper citation marker
            # For RAG sources, include full content; for others, truncate if needed
            if source_type == 'rag' and content:
                formatted_evidence.append(f"[^{i}] {title}\n{content}\n(Source: {source}){instruction_marker}")
            else:
                formatted_evidence.append(f"[^{i}] {title}{': ' + content + '...' if content else ''} (Source: {source}){instruction_marker}")
        
        self.logger.info(f"Formatting {len(formatted_evidence)} evidence items for content generation")
        return "\n".join(formatted_evidence)
    
    def _detect_specific_instructions(self, key_points: List[str], research_brief: str) -> List[Dict[str, Any]]:
        """
        Detect specific instructions like "include N examples of X" in key points and research brief.
        
        Args:
            key_points: List of key points to check
            research_brief: Research brief to check
            
        Returns:
            List of detected instructions with number and topic
        """
        instructions = []
        text_to_check = " ".join(key_points) + " " + research_brief
        
        # Pattern to match "include N examples of X" or similar variations
        # Use a simpler approach: match up to sentence boundary or end
        patterns = [
            r'include\s+(\d+)\s+examples?\s+of\s+([^\.]+?)(?:\.|$)',
            r'provide\s+(\d+)\s+examples?\s+of\s+([^\.]+?)(?:\.|$)',
            r'list\s+(\d+)\s+examples?\s+of\s+([^\.]+?)(?:\.|$)',
            r'(\d+)\s+examples?\s+of\s+([^\.]+?)(?:\.|$)',
            r'include\s+(\d+)\s+([^\.]+?)\s+examples?(?:\.|$)',
            r'provide\s+(\d+)\s+([^\.]+?)\s+examples?(?:\.|$)',
            r'need\s+(\d+)\s+examples?\s+of\s+([^\.]+?)(?:\.|$)',
            r'should\s+include\s+(\d+)\s+examples?\s+of\s+([^\.]+?)(?:\.|$)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text_to_check, re.IGNORECASE)
            for match in matches:
                number = int(match.group(1))
                topic = match.group(2).strip()
                
                # Clean up the topic - remove trailing punctuation and common connectors
                topic = topic.strip('.,;:')
                # Remove trailing common words that might be part of the next sentence
                topic = re.sub(r'\s+(and|or|but|with|for|in|on|at|the|a|an)\s*$', '', topic, flags=re.IGNORECASE)
                topic = topic.strip()
                
                if topic and number > 0 and len(topic) > 3:  # Ensure topic is meaningful
                    instructions.append({
                        'number': number,
                        'topic': topic,
                        'full_match': match.group(0)
                    })
        
        # Remove duplicates
        seen = set()
        unique_instructions = []
        for inst in instructions:
            key = (inst['number'], inst['topic'].lower())
            if key not in seen:
                seen.add(key)
                unique_instructions.append(inst)
        
        if unique_instructions:
            instruction_list = [f"{i['number']} examples of {i['topic']}" for i in unique_instructions]
            self.logger.info(f"Detected {len(unique_instructions)} specific instruction(s): {instruction_list}")
        
        return unique_instructions
    
    def _research_examples(self, instruction: Dict[str, Any], research_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Research examples using RAG first, then Linkup if needed.
        
        Args:
            instruction: Instruction dict with 'number' and 'topic'
            research_data: Research data containing RAG and Linkup configuration
            
        Returns:
            List of evidence items with examples
        """
        number = instruction['number']
        topic = instruction['topic']
        examples_evidence = []
        
        self.logger.info(f"Researching {number} examples of '{topic}'")
        
        # Try RAG first if enabled
        if research_data.get('rag_enabled', False) and research_data.get('rag_endpoint'):
            try:
                from rag_client import create_rag_client, RAGQuery
                
                rag_client = create_rag_client(
                    endpoint=research_data.get('rag_endpoint'),
                    api_key=research_data.get('rag_api_key'),
                    collection=research_data.get('rag_collection'),
                    llm_provider=research_data.get('rag_llm_provider', 'deepseek'),
                    max_results=number * 2  # Get more results to filter
                )
                
                # Create query for examples
                query_text = f"{number} examples of {topic}"
                rag_query = RAGQuery(
                    query=query_text,
                    max_results=number * 2,
                    similarity_threshold=0.6
                )
                
                self.logger.info(f"Searching RAG for: {query_text}")
                rag_response = rag_client.query(rag_query)
                
                if rag_response.success and rag_response.results:
                    self.logger.info(f"Found {len(rag_response.results)} results from RAG")
                    
                    # Convert RAG results to evidence format
                    for result in rag_response.results[:number]:
                        evidence_item = {
                            'content': result.content,
                            'source': result.source,
                            'source_type': 'rag',
                            'title': result.metadata.get('title', result.source),
                            'relevance_score': result.relevance_score,
                            'credibility_score': result.credibility_score,
                            'metadata': result.metadata,
                            'instruction_topic': topic,
                            'instruction_number': number
                        }
                        examples_evidence.append(evidence_item)
                    
                    # If we have enough examples from RAG, return them
                    if len(examples_evidence) >= number:
                        self.logger.info(f"Found sufficient examples ({len(examples_evidence)}) from RAG")
                        return examples_evidence
                    else:
                        self.logger.info(f"Found {len(examples_evidence)} examples from RAG, need {number - len(examples_evidence)} more")
                else:
                    self.logger.info(f"RAG search returned no results or failed: {rag_response.error if not rag_response.success else 'no results'}")
                    
            except Exception as e:
                self.logger.warning(f"Error searching RAG for examples: {str(e)}")
        
        # If RAG didn't provide enough examples, try Linkup
        if len(examples_evidence) < number and research_data.get('linkup_api_key'):
            try:
                from linkup_client import create_linkup_client, SearchQuery
                
                linkup_client = create_linkup_client(
                    api_key=research_data.get('linkup_api_key'),
                    endpoint=research_data.get('linkup_endpoint'),
                    max_results=number * 2
                )
                
                # Create search query for examples
                query_text = f"{number} examples of {topic}"
                search_query = SearchQuery(
                    query=query_text,
                    max_results=number * 2,
                    depth="standard"
                )
                
                self.logger.info(f"Searching Linkup for: {query_text}")
                search_response = linkup_client.search(search_query)
                
                if search_response.success and search_response.results:
                    self.logger.info(f"Found {len(search_response.results)} results from Linkup")
                    
                    # Convert Linkup results to evidence format
                    needed = number - len(examples_evidence)
                    for result in search_response.results[:needed]:
                        evidence_item = {
                            'content': result.snippet + (f" {result.content}" if result.content else ""),
                            'source': result.url,
                            'source_type': 'linkup',
                            'title': result.title,
                            'relevance_score': result.relevance_score,
                            'credibility_score': result.credibility_score,
                            'metadata': result.metadata or {},
                            'instruction_topic': topic,
                            'instruction_number': number
                        }
                        examples_evidence.append(evidence_item)
                    
                    self.logger.info(f"Added {needed} examples from Linkup, total: {len(examples_evidence)}")
                else:
                    self.logger.info(f"Linkup search returned no results or failed: {search_response.error if not search_response.success else 'no results'}")
                    
            except Exception as e:
                self.logger.warning(f"Error searching Linkup for examples: {str(e)}")
        
        if examples_evidence:
            self.logger.info(f"Successfully researched {len(examples_evidence)} examples of '{topic}'")
        else:
            self.logger.warning(f"Could not find examples of '{topic}' in RAG or Linkup")
        
        return examples_evidence
    
    def _get_tone_specific_instructions(self, tone: str) -> str:
        """
        Get tone-specific writing instructions, especially for friendly tone.
        
        Args:
            tone: The writing tone (e.g., 'friendly', 'professional', 'journalistic')
            
        Returns:
            String with tone-specific instructions
        """
        # Use the standalone function to avoid code duplication
        return get_tone_specific_instructions(tone)
    
    def _get_citation_instructions(self, context: Dict[str, Any]) -> str:
        """
        Get citation instructions based on whether in-text citations are enabled.
        
        Args:
            context: Content generation context dictionary
            
        Returns:
            String with citation instructions
        """
        include_citations = context.get('include_in_text_citations', True)
        
        if not include_citations:
            return """
                    CRITICAL - NO IN-TEXT CITATIONS:
                    - Do NOT add any citation markers like [^1], [^2], [^3], etc. in your content
                    - Do NOT include any citation references in the text
                    - Write the content naturally without any citation markers
                    - The references section will be added separately, so you don't need to cite sources in the text
                    - Use the evidence and information provided, but do not add citation markers"""
        
        return """
                    CITATION INSTRUCTIONS (CRITICAL):
                    - If evidence sources are provided above, you MUST use citations when referencing information from them
                    - Reference evidence by number: [^1], [^2], [^3], etc. matching the evidence list position
                    - Cite sources frequently - whenever using statistics, data, examples, or claims from evidence
                    - Each citation number should match the position in the evidence list (first evidence = [^1], second = [^2], etc.)
                    - If evidence sources are provided above, you MUST include at least 2-4 citations in your response when using that information
                    - Use citations when: referencing statistics, mentioning studies, quoting data, citing examples, referencing research
                    - If no evidence sources are available above, write complete content based on general knowledge without citations
                    - Do NOT create fictional or generic citations - only cite if evidence is actually provided above
                    - IMPORTANT: When evidence is provided, actively use it and cite it with [^1], [^2], [^3] etc. directly in your content
                    - Even without evidence, create clear, complete content with comparative tables when comparisons are relevant"""
    
    def _prepare_content_context(self, title: str, subtitle: Optional[str],
                               key_points: List[str], research_data: Dict[str, Any],
                               claims: List[Dict], evidence: List[Dict],
                               previous_sections: List[SectionContent] = None) -> Dict[str, Any]:
        """Prepare context for content generation."""
        # Log evidence availability
        self.logger.info(f"Preparing context for section '{title}' - {len(evidence)} total evidence items, {len(claims)} claims")
        
        # Detect and research specific instructions (e.g., "include 5 examples of X")
        research_brief = research_data.get('brief', '')
        specific_instructions = self._detect_specific_instructions(key_points, research_brief)
        
        # Research examples for each instruction
        additional_evidence = []
        if specific_instructions:
            for instruction in specific_instructions:
                examples = self._research_examples(instruction, research_data)
                if examples:
                    additional_evidence.extend(examples)
                    self.logger.info(f"Added {len(examples)} evidence items for instruction: {instruction['number']} examples of {instruction['topic']}")
        
        # Add researched examples to evidence
        if additional_evidence:
            evidence = evidence + additional_evidence
            self.logger.info(f"Total evidence after researching examples: {len(evidence)} items")
        
        # Extract relevant claims for this section
        relevant_claims = self._filter_relevant_claims(claims, title, key_points)
        
        # Extract relevant evidence for this section
        relevant_evidence = self._filter_relevant_evidence(evidence, title, key_points)
        
        # Log filtered results
        self.logger.info(f"Section '{title}' - Filtered to {len(relevant_claims)} relevant claims, {len(relevant_evidence)} relevant evidence items")
        
        # Log evidence content status
        if relevant_evidence:
            valid_evidence = [ev for ev in relevant_evidence if ev.get('content') and ev.get('content').strip()]
            if len(valid_evidence) < len(relevant_evidence):
                self.logger.warning(f"Section '{title}' - {len(relevant_evidence) - len(valid_evidence)} evidence items have empty content")
        
        # Prepare previous sections context
        previous_context = ""
        if previous_sections:
            previous_context = "\n".join([
                f"Section {s.section_order}: {s.title} - {s.total_word_count} words"
                for s in previous_sections[-2:]  # Last 2 sections for context
            ])
        
        tone = research_data.get('tone', 'journalistic')
        # Log tone for debugging
        self.logger.info(f"Preparing content context for section '{title}' with tone: {tone}")
        
        return {
            "title": title,
            "subtitle": subtitle,
            "key_points": key_points,
            "research_brief": research_data.get('brief', ''),
            "draft_title": research_data.get('draft_title', ''),
            "tone": tone,
            "target_audience": research_data.get('target_audience', 'general'),
            "relevant_claims": relevant_claims,
            "relevant_evidence": relevant_evidence,
            "previous_sections": previous_context,
            "include_in_text_citations": research_data.get('include_in_text_citations', True),
            "keywords": research_data.get('keywords', '')
        }
    
    def _build_user_message(self, context: Dict[str, Any]) -> str:
        """Build user message for content generation with proper tone handling."""
        tone = context.get('tone', 'journalistic')
        
        # Build tone reminder
        tone_reminder = f"This article MUST be written in {tone} tone."
        if tone.lower() == 'friendly':
            tone_reminder += "\n\nFOR FRIENDLY TONE: Write like you're sharing a personal story with a friend. Use first-person (\"I've found\", \"Last month I\"), specific examples with details, casual language, and make it warm and engaging. Avoid formal words like \"crucial\", \"paramount\", \"necessitates\", \"individuals\". Make it interesting, not boring or professional."
            tone_reminder += "\n\nIMPORTANT: Do NOT start with greetings like \"Hi friends\", \"Hey there\", or \"Hello everyone\". Start directly with engaging content - friendly means warm and personal, not chatty greetings."
            tone_reminder += "\n\nREMEMBER: Write with personality, use first-person storytelling, include specific relatable examples, and make it warm and engaging - like the example: \"Generative AI has quietly become my favorite coworker. It proofreads my emails while I'm still sipping coffee...\" (notice it starts directly, no greeting)"
        
        # Build subtitle line
        subtitle_line = f"Subtitle: {context['subtitle']}\n\n" if context.get('subtitle') else ""
        
        # Build draft title line
        draft_title_line = f"Draft Title: {context['draft_title']}\n" if context.get('draft_title') else ""
        
        return f"""Section: {context['title']}
{subtitle_line}========================================
TONE REMINDER - CRITICAL
========================================
{tone_reminder}

========================================
CONTENT REQUIREMENTS
========================================
Key Points to Cover:
{chr(10).join(f"- {point}" for point in context['key_points'])}

Research Brief: {context['research_brief']}
{draft_title_line}Keywords to integrate naturally: {context.get('keywords', '')}

Relevant Claims:
{chr(10).join(f"- {claim.get('claim', '')}" for claim in context['relevant_claims'][:3])}

Supporting Evidence:
{self._format_evidence_for_citations(context['relevant_evidence'][:10])}

CRITICAL: If the evidence above contains specific examples (especially those marked with instruction_topic), you MUST include those exact examples in your content. Do not create generic examples - use the researched examples provided in the evidence.
{self._get_citation_instructions(context)}

Previous Context:
{context['previous_sections'] if context['previous_sections'] else 'This is the first section.'}"""
    
    def _generate_paragraph_content(self, context: Dict[str, Any], word_count_target: int) -> List[ContentBlock]:
        """Generate paragraph-based content using verbalized sampling for improved quality."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert content writer. Write clear, useful content for a {context['tone']} article.
                    
                    ========================================
                    ⚠️ CRITICAL: THE TONE FOR THIS ARTICLE IS {context['tone'].upper()} ⚠️
                    ========================================
                    YOU MUST USE ONLY THE {context['tone'].upper()} TONE AS SPECIFIED BELOW
                    
                    {self._get_tone_specific_instructions(context['tone'])}
                    
                    The tone instructions above are MANDATORY and take precedence over all other instructions below.
                    If there's any conflict between tone requirements and other instructions, follow the tone requirements.
                    
                    REMEMBER: The tone is {context['tone']} - use ONLY this tone consistently throughout.
                    
                    ========================================
                    CONTENT REQUIREMENTS
                    ========================================
                    - Target EXACTLY {word_count_target} words (be complete and detailed)
                    - Cover ALL the key points: {', '.join(context['key_points'])}
                    - Use evidence and claims to support arguments - cite sources when using them
                    - Write for {context['target_audience']} audience with appropriate depth
                    - Include specific examples, case studies, statistics, and practical details
                    - Make it clear and useful - avoid generic or filler content
                    - Create well-structured sections with clear organization
                    - Write complete content - this is professional content, not a summary
                    
                    ========================================
                    WRITING STYLE REQUIREMENTS
                    ========================================
                    NOTE: These must align with the tone requirements above. For friendly tone, prioritize conversational "you" language.
                    - Write in a natural, conversational style - avoid formal or formulaic language
                    - Use simple, clear sentences that flow smoothly from one to the next
                    - Each sentence should connect logically to the previous one - avoid abrupt jumps
                    - Use descriptive subheadings (H3) that break content into logical sections
                    - Create smooth transitions between paragraphs - use connecting words naturally, not excessively
                    - Include bullet points or numbered lists where appropriate for clarity
                    - MANDATORY: Include COMPARATIVE TABLES when comparing options, strategies, or presenting structured data
                    - Include multiple specific examples, data points, statistics, and practical insights throughout
                    - Write in a natural style that reads like human-written content
                    - Integrate the provided keywords into the content where they fit naturally
                    - Ensure keywords appear naturally in the text, not forced or repetitive
                    - Expand on ideas with examples and details
                    - Vary sentence length - mix shorter and longer sentences for natural rhythm
                    - Read your sentences aloud mentally - if they sound awkward, simplify them
                    - Make sure each paragraph has one clear main idea that flows to the next paragraph
                    
                    ========================================
                    AVOID AI-GENERATED LANGUAGE AND COMPLEX VOCABULARY
                    ========================================
                    - Do NOT use overly complex words: crucial, embark, paramount, meticulous, navigating, complexities, realm, dive, shall, tailored, towards, underpins, everchanging, ever-evolving, robust, elevate, unleash, cutting-edge, rapidly expanding, mastering, excels, harness, imagine, delve, tapestry, bustling, vibrant, metropolis, labyrinth, gossamer, enigma, whispering, indelible, potent, signifying, positioning, cultivating, commanding, proactive, strategic, adept, interconnected, specialized, blend, niche, trajectory, implementing, ensuring, sought after
                    - PREFER SIMPLER ALTERNATIVES: "important" not "crucial", "start" not "embark", "key" not "paramount", "careful" not "meticulous", "strong" not "potent", "shows" not "signifies", "place" not "position", "develop" not "cultivate", "earn" not "command", "active" not "proactive", "plan" not "strategic", "skilled" not "adept", "connected" not "interconnected", "special" not "specialized", "mix" not "blend", "small area" not "niche", "path" not "trajectory", "use" not "implement", "make sure" not "ensure", "wanted" not "sought after"
                    - Do NOT use phrases like: "the world of", "not only", "in today's digital age", "game changer", "designed to enhance", "it is advisable", "when it comes to", "in the realm of", "unlock the secrets", "unveil the secrets", "take a dive into", "as a professional", "you may want to", "it's worth noting that", "to summarize", "ultimately", "to put it simply", "in conclusion", "in summary", "remember that"
                    - Do NOT use transition words excessively: however, therefore, additionally, specifically, generally, consequently, importantly, indeed, thus, alternatively, notably, as well as, despite, essentially, while, unless, also, even though, because, in contrast, although, in order to, due to, even if, given that, arguably, on the other hand, as previously mentioned, subsequently
                    - Do NOT use: firstly, moreover, furthermore, vital, keen, fancy
                    - Do NOT use analogies to music, conductors, or other overly creative metaphors
                    - Write directly and clearly - avoid flowery or overly descriptive language
                    - Use simple, direct language instead of complex phrases
                    - Write as if you're explaining to a colleague, not writing marketing copy
                    - Keep vocabulary accessible - if a simpler word works, use it
                    
                    SPECIAL INSTRUCTIONS FOR INTRODUCTION SECTIONS:
                    - Keep introduction SIMPLE and CONCISE (unless target word count is high)
                    - Use minimal subheadings (preferably none) for introductions
                    - Write as flowing paragraphs rather than structured sections
                    - Focus on hook, overview, and what reader will learn
                    - Avoid breaking introduction into multiple subsections
                    
                    MANDATORY TABLE REQUIREMENTS:
                    - ALWAYS include COMPARATIVE TABLES when comparing:
                      * Different strategies, approaches, or methodologies
                      * Feature comparisons between options or tools
                      * Pros/cons lists (presented in table format with columns)
                      * Performance metrics, statistics, or data comparisons
                      * Decision frameworks or evaluation criteria
                      * Timeline comparisons or historical data
                    - Use tables for: market data, price comparisons, regional statistics, trend analysis
                    - Use tables for: numerical data, percentages, rankings, survey results, comparative metrics
                    - When comparing ANYTHING (strategies, tools, approaches, options), ALWAYS present it in a comparative table format
                    - Tables MUST include proper headers and well-organized data
                    - Include explanatory text before and after each table to provide context
                    - Create at least 1-2 meaningful comparative tables per section when comparisons are relevant
                    - DO NOT skip tables just because evidence is limited - create useful comparisons based on topic
                    
                    TABLE FORMAT:
                    <table style="border-collapse: collapse; width: 100%; margin: 1em 0;">
                    <thead>
                    <tr>
                    <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Header 1</th>
                    <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Header 2</th>
                    </tr>
                    </thead>
                    <tbody>
                    <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Data 1</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">Data 2</td>
                    </tr>
                    </tbody>
                    </table>
                    
                    OUTPUT FORMAT:
                    - Return ONLY clean HTML content without any wrapper tags
                    - Do NOT include any meta-commentary, explanations, or prefixes like "Here's the content", "optimized for X tone"
                    - Start directly with the HTML content (e.g., <p> or <h3>)
                    - Use proper HTML structure: <p> for paragraphs, <h3> for subheadings, <ul>/<ol> for lists
                    - Use proper HTML tables: <table>, <thead>, <tbody>, <tr>, <th>, <td> with appropriate styling
                    - Do NOT include <section>, <article>, or other wrapper tags
                    - Do NOT include markdown code blocks or ```html
                    - Do NOT include H2 headings (only H3 subheadings)
                    - Include tables with proper headers and data organization when presenting structured information
                    - Write in a natural style that flows smoothly from paragraph to paragraph
                    - Include inline citations as [^1], [^2], etc. when referencing evidence or claims
                    - Make it clear, complete, and useful with factual accuracy
                    - Write ENOUGH content to meet the {word_count_target} word target - be complete and detailed
                    - Avoid generic filler - every sentence should add value, insights, or information
                    - CRITICAL: Use proper punctuation - single commas, single periods, no repeated punctuation marks
                    - CRITICAL: Ensure sentences flow naturally - read each sentence to make sure it connects smoothly to the next
                    - CRITICAL: Write in {context['tone']} tone consistently throughout - use ONLY this tone"""
                },
                {
                    "role": "user",
                    "content": self._build_user_message(context)
                }
            ]
            
            # Use verbalized sampling if enabled
            if self.use_verbalized_sampling and self.verbalized_client:
                self.logger.info("Using verbalized sampling for paragraph content generation")
                verbalized_response = self.verbalized_client.generate_content_with_sampling(
                    messages=messages,
                    content_type="paragraph",
                    word_count_target=word_count_target
                )
                content = verbalized_response.text
                
                # Handle fallback case: if verbalized sampling returns empty/invalid content,
                # fall back to standard generation
                if not content or content.strip().startswith("Content generated for:") or len(content.strip()) < 50:
                    self.logger.warning("Verbalized sampling returned invalid/fallback content, using standard generation")
                    response = self.llm_client.generate(messages)
                    content = response.content.strip()
                    
                    metadata = {
                        "llm_model": response.model,
                        "generation_time": response.response_time,
                        "cost": response.cost,
                        "verbalized_sampling": {
                            "enabled": True,
                            "sample_index": verbalized_response.sample_index,
                            "total_samples": len(verbalized_response.all_samples),
                            "fallback": True,
                            "reason": "verbalized-sampling returned invalid content"
                        }
                    }
                else:
                    # Log sampling information
                    self.logger.info(f"Verbalized sampling: selected sample {verbalized_response.sample_index + 1} "
                                   f"out of {len(verbalized_response.all_samples)} samples")
                    
                    # Create enhanced metadata
                    metadata = {
                        "llm_model": "verbalized-sampling",
                        "generation_time": 0.0,  # Will be updated if we track this
                        "cost": 0.0,  # Will be updated if we track this
                        "verbalized_sampling": {
                            "enabled": True,
                            "sample_index": verbalized_response.sample_index,
                            "total_samples": len(verbalized_response.all_samples),
                            "sampling_config": verbalized_response.metadata
                        }
                    }
            else:
                # Standard generation
                self.logger.info("Using standard generation for paragraph content")
                response = self.llm_client.generate(messages)
                content = response.content.strip()
                
                metadata = {
                    "llm_model": response.model,
                    "generation_time": response.response_time,
                    "cost": response.cost,
                    "verbalized_sampling": {
                        "enabled": False
                    }
                }
            
            # Clean HTML content - remove citations if flag is disabled
            remove_citations = not context.get('include_in_text_citations', True)
            cleaned_content = self._clean_html_content(content, remove_citations=remove_citations)
            
            # Remove any meta-commentary the LLM might have added
            import re
            # Remove common LLM prefixes like "Here's the content", "optimized for X tone", etc.
            patterns_to_remove = [
                r'^Here\'s the content[^\n]*\n*',
                r'^Here is the content[^\n]*\n*',
                r'^Content[^\n]*\n*',
                r'optimized for [^\n]*tone[^\n]*\n*',
                r'^[^\<]*?(?=<)',  # Remove any text before the first HTML tag
            ]
            
            for pattern in patterns_to_remove:
                cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.IGNORECASE | re.MULTILINE)
            
            # If content doesn't start with HTML, try to find where HTML starts
            if not cleaned_content.strip().startswith('<'):
                html_match = re.search(r'<[^>]+>', cleaned_content)
                if html_match:
                    cleaned_content = cleaned_content[html_match.start():]
            
            cleaned_content = cleaned_content.strip()
            
            # Additional pass to fix any remaining punctuation issues in the final content
            cleaned_content = self._fix_punctuation_errors(cleaned_content)
            
            # Create content block
            content_block = ContentBlock(
                content=cleaned_content,
                content_type="paragraph",
                word_count=len(cleaned_content.split()),
                citations=self._extract_citations_from_content(cleaned_content, context['relevant_evidence']),
                metadata=metadata
            )
            
            return [content_block]
            
        except Exception as e:
            self.logger.error(f"Error generating paragraph content: {str(e)}")
            return [self._create_fallback_content_block(context['title'], word_count_target)]
    
    def _generate_list_content(self, context: Dict[str, Any], word_count_target: int) -> List[ContentBlock]:
        """Generate list-based content."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert content writer. Create a clear, useful list for a {context['tone']} article.
                    
                    Requirements:
                    {self._get_tone_specific_instructions(context['tone'])}
                    - Create a well-structured list
                    - Target approximately {word_count_target} words
                    - Cover the key points: {', '.join(context['key_points'])}
                    - Use evidence and claims to support each list item
                    - Write for {context['target_audience']} audience
                    - Include specific examples and practical advice
                    - Use clear, direct language
                    
                    WRITING STYLE REQUIREMENTS:
                    - Write in a conversational, human style - avoid formal or formulaic language
                    - Use descriptive subheadings (H3) that organize content logically
                    - Create clear transitions between paragraphs
                    - Include bullet points or numbered lists where appropriate
                    - Use TABLES when presenting data, statistics, comparisons, or structured information
                    - Include specific examples, data, and practical insights
                    - Write in a natural style that reads like human-written content
                    - Integrate the provided keywords into the content where they fit naturally
                    - Ensure keywords appear naturally in the text, not forced or repetitive
                    
                    AVOID AI-GENERATED LANGUAGE:
                    - Do NOT use words like: crucial, embark, paramount, meticulous, navigating, complexities, realm, dive, shall, tailored, towards, underpins, everchanging, ever-evolving, robust, elevate, unleash, cutting-edge, rapidly expanding, mastering, excels, harness, imagine, delve, tapestry, bustling, vibrant, metropolis, labyrinth, gossamer, enigma, whispering, indelible
                    - Do NOT use phrases like: "the world of", "not only", "in today's digital age", "game changer", "designed to enhance", "it is advisable", "when it comes to", "in the realm of", "unlock the secrets", "unveil the secrets", "take a dive into", "as a professional", "you may want to", "it's worth noting that", "to summarize", "ultimately", "to put it simply", "in conclusion", "in summary", "remember that"
                    - Do NOT use transition words excessively: however, therefore, additionally, specifically, generally, consequently, importantly, indeed, thus, alternatively, notably, as well as, despite, essentially, while, unless, also, even though, because, in contrast, although, in order to, due to, even if, given that, arguably, on the other hand, as previously mentioned, subsequently
                    - Do NOT use: firstly, moreover, furthermore, vital, keen, fancy
                    - Write directly and clearly - avoid flowery or overly descriptive language
                    - Use simple, direct language instead of complex phrases
                    
                    TABLE USAGE GUIDELINES:
                    - Use tables for: feature lists, pros/cons comparisons, step-by-step processes
                    - Use tables for: rankings, checklists, decision matrices, comparison charts
                    - Use tables for: numerical data, percentages, timelines, survey results
                    - Always include table headers and proper HTML table structure
                    - Keep tables concise and relevant to the content
                    
                    TABLE FORMAT:
                    <table style="border-collapse: collapse; width: 100%; margin: 1em 0;">
                    <thead>
                    <tr>
                    <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Header 1</th>
                    <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Header 2</th>
                    </tr>
                    </thead>
                    <tbody>
                    <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Data 1</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">Data 2</td>
                    </tr>
                    </tbody>
                    </table>
                    
                    OUTPUT FORMAT:
                    - Return ONLY clean HTML content without any wrapper tags
                    - Use proper HTML structure: <p> for paragraphs, <h3> for subheadings, <ul>/<ol> for lists
                    - Use proper HTML tables: <table>, <thead>, <tbody>, <tr>, <th>, <td> with appropriate styling
                    - Do NOT include <section>, <article>, or other wrapper tags
                    - Do NOT include markdown code blocks or ```html
                    - Do NOT include H2 headings (only H3 subheadings)
                    - Include tables with proper headers and data organization when presenting structured information
                    - Write in a natural style that flows smoothly from paragraph to paragraph
                    - Include inline citations as [^1], [^2], etc. when referencing evidence or claims
                    - Make it clear and useful with factual accuracy
                    
                    
                    Relevant Claims:
                    {chr(10).join(f"- {claim.get('claim', '')}" for claim in context['relevant_claims'][:3])}
                    
                    Supporting Evidence:
                    {self._format_evidence_for_citations(context['relevant_evidence'][:10])}
                    {self._get_citation_instructions(context)}"""
                }
            ]
            
            response = self.llm_client.generate(messages)
            content = response.content.strip()
            
            # Clean HTML content - remove citations if flag is disabled
            remove_citations = not context.get('include_in_text_citations', True)
            cleaned_content = self._clean_html_content(content, remove_citations=remove_citations)
            
            # Additional pass to fix any remaining punctuation issues
            cleaned_content = self._fix_punctuation_errors(cleaned_content)
            
            # Create content block
            content_block = ContentBlock(
                content=cleaned_content,
                content_type="list",
                word_count=len(cleaned_content.split()),
                citations=self._extract_citations_from_content(cleaned_content, context['relevant_evidence']),
                metadata={
                    "llm_model": response.model,
                    "generation_time": response.response_time,
                    "cost": response.cost
                }
            )
            
            return [content_block]
            
        except Exception as e:
            self.logger.error(f"Error generating list content: {str(e)}")
            return [self._create_fallback_content_block(context['title'], word_count_target)]
    
    def _generate_step_by_step_content(self, context: Dict[str, Any], word_count_target: int) -> List[ContentBlock]:
        """Generate step-by-step content."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert instructional writer. Create a clear step-by-step guide for a {context['tone']} article.
                    
                    Requirements:
                    {self._get_tone_specific_instructions(context['tone'])}
                    - Create clear, actionable steps
                    - Target approximately {word_count_target} words
                    - Cover the key points: {', '.join(context['key_points'])}
                    - Use evidence and claims to support each step
                    - Write for {context['target_audience']} audience
                    - Include specific instructions and examples
                    - Make it easy to follow and implement
                    
                    WRITING STYLE REQUIREMENTS:
                    - Write in a conversational, human style - avoid formal or formulaic language
                    - Use direct, clear instructions
                    - Write in a natural style that reads like human-written content
                    
                    AVOID AI-GENERATED LANGUAGE:
                    - Do NOT use words like: crucial, embark, paramount, meticulous, navigating, complexities, realm, dive, shall, tailored, towards, underpins, everchanging, ever-evolving, robust, elevate, unleash, cutting-edge, rapidly expanding, mastering, excels, harness, imagine, delve, tapestry, bustling, vibrant, metropolis, labyrinth, gossamer, enigma, whispering, indelible
                    - Do NOT use phrases like: "the world of", "not only", "in today's digital age", "game changer", "designed to enhance", "it is advisable", "when it comes to", "in the realm of", "unlock the secrets", "unveil the secrets", "take a dive into", "as a professional", "you may want to", "it's worth noting that", "to summarize", "ultimately", "to put it simply", "in conclusion", "in summary", "remember that"
                    - Do NOT use transition words excessively: however, therefore, additionally, specifically, generally, consequently, importantly, indeed, thus, alternatively, notably, as well as, despite, essentially, while, unless, also, even though, because, in contrast, although, in order to, due to, even if, given that, arguably, on the other hand, as previously mentioned, subsequently
                    - Do NOT use: firstly, moreover, furthermore, vital, keen, fancy
                    - Write directly and clearly - avoid flowery or overly descriptive language
                    - Use simple, direct language instead of complex phrases
                    
                    Supporting Evidence:
                    {self._format_evidence_for_citations(context['relevant_evidence'][:10])}
                    {self._get_citation_instructions(context)}"""
                }
            ]
            
            response = self.llm_client.generate(messages)
            content = response.content.strip()
            
            # Clean HTML content - remove citations if flag is disabled
            remove_citations = not context.get('include_in_text_citations', True)
            cleaned_content = self._clean_html_content(content, remove_citations=remove_citations)
            
            # Additional pass to fix any remaining punctuation issues
            cleaned_content = self._fix_punctuation_errors(cleaned_content)
            
            # Create content block
            content_block = ContentBlock(
                content=cleaned_content,
                content_type="step_by_step",
                word_count=len(cleaned_content.split()),
                citations=self._extract_citations_from_content(cleaned_content, context['relevant_evidence']),
                metadata={
                    "llm_model": response.model,
                    "generation_time": response.response_time,
                    "cost": response.cost
                }
            )
            
            return [content_block]
            
        except Exception as e:
            self.logger.error(f"Error generating step-by-step content: {str(e)}")
            return [self._create_fallback_content_block(context['title'], word_count_target)]
    
    def _generate_comparison_content(self, context: Dict[str, Any], word_count_target: int) -> List[ContentBlock]:
        """Generate comparison-based content."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert analytical writer. Create a detailed comparison for a {context['tone']} article.
                    
                    Requirements:
                    {self._get_tone_specific_instructions(context['tone'])}
                    - Create a complete, detailed comparison - this is the core of this section
                    - Target EXACTLY {word_count_target} words - be complete and detailed
                    - Cover ALL the key points: {', '.join(context['key_points'])}
                    - Use evidence and claims to support comparisons - cite sources when using them
                    - Write for {context['target_audience']} audience with appropriate depth
                    - Include specific examples, case studies, statistics, and data throughout
                    - Make clear, well-reasoned recommendations based on the comparison
                    - Be clear and useful - avoid generic content
                    
                    WRITING STYLE REQUIREMENTS:
                    - Write in a conversational, human style - avoid formal or formulaic language
                    - Use descriptive subheadings (H3) that break the comparison into logical sections
                    - Create clear transitions between paragraphs and sections
                    - Include bullet points or numbered lists where appropriate for clarity
                    - MANDATORY: Include MULTIPLE COMPARATIVE TABLES - this is a comparison section, tables are essential
                    - Include extensive specific examples, data points, statistics, and practical insights
                    - Write in a natural style that reads like human-written content
                    - Integrate the provided keywords into the content where they fit naturally
                    - Ensure keywords appear naturally in the text, not forced or repetitive
                    - Expand on comparisons with examples and details
                    
                    AVOID AI-GENERATED LANGUAGE:
                    - Do NOT use words like: crucial, embark, paramount, meticulous, navigating, complexities, realm, dive, shall, tailored, towards, underpins, everchanging, ever-evolving, robust, elevate, unleash, cutting-edge, rapidly expanding, mastering, excels, harness, imagine, delve, tapestry, bustling, vibrant, metropolis, labyrinth, gossamer, enigma, whispering, indelible
                    - Do NOT use phrases like: "the world of", "not only", "in today's digital age", "game changer", "designed to enhance", "it is advisable", "when it comes to", "in the realm of", "unlock the secrets", "unveil the secrets", "take a dive into", "as a professional", "you may want to", "it's worth noting that", "to summarize", "ultimately", "to put it simply", "in conclusion", "in summary", "remember that"
                    - Do NOT use transition words excessively: however, therefore, additionally, specifically, generally, consequently, importantly, indeed, thus, alternatively, notably, as well as, despite, essentially, while, unless, also, even though, because, in contrast, although, in order to, due to, even if, given that, arguably, on the other hand, as previously mentioned, subsequently
                    - Do NOT use: firstly, moreover, furthermore, vital, keen, fancy
                    - Write directly and clearly - avoid flowery or overly descriptive language
                    - Use simple, direct language instead of complex phrases
                    
                    MANDATORY TABLE REQUIREMENTS FOR COMPARISON SECTIONS:
                    - You MUST include at least 2-3 complete comparative tables in this section
                    - Tables are REQUIRED for: feature comparisons, pros/cons lists (in table format), price comparisons, performance metrics
                    - Tables are REQUIRED for: side-by-side analysis, decision matrices, timeline comparisons
                    - Tables are REQUIRED for: numerical data, percentages, rankings, survey results
                    - When comparing options, strategies, or approaches, ALWAYS use a comparative table format
                    - Always include table headers and proper HTML table structure with clear column organization
                    - Keep tables complete and useful - include enough data to be meaningful
                    - Include explanatory text before and after each table to provide context and insights
                    - Tables should be a central feature of this comparison section, not an afterthought
                    
                    TABLE FORMAT:
                    <table style="border-collapse: collapse; width: 100%; margin: 1em 0;">
                    <thead>
                    <tr>
                    <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Header 1</th>
                    <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Header 2</th>
                    </tr>
                    </thead>
                    <tbody>
                    <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Data 1</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">Data 2</td>
                    </tr>
                    </tbody>
                    </table>
                    
                    OUTPUT FORMAT:
                    - Return ONLY clean HTML content without any wrapper tags
                    - Use proper HTML structure: <p> for paragraphs, <h3> for subheadings, <ul>/<ol> for lists
                    - Use proper HTML tables: <table>, <thead>, <tbody>, <tr>, <th>, <td> with appropriate styling
                    - Do NOT include <section>, <article>, or other wrapper tags
                    - Do NOT include markdown code blocks or ```html
                    - Do NOT include H2 headings (only H3 subheadings)
                    - Include tables with proper headers and data organization when presenting structured information
                    - Write in a natural style that flows smoothly from paragraph to paragraph
                    - Include inline citations as [^1], [^2], etc. when referencing evidence or claims
                    - Make it clear and useful with factual accuracy
                    
                    Format as a structured comparison with clear sections."""
                },
                {
                    "role": "user",
                    "content": f"""Section: {context['title']}
                    {f"Subtitle: {context['subtitle']}" if context['subtitle'] else ""}
                    
                    Key Points to Cover:
                    {chr(10).join(f"- {point}" for point in context['key_points'])}
                    
                    Research Brief: {context['research_brief']}
                    {f"Draft Title: {context['draft_title']}" if context.get('draft_title') else ""}
                    Keywords to integrate naturally: {context.get('keywords', '')}
                    
                    Relevant Claims:
                    {chr(10).join(f"- {claim.get('claim', '')}" for claim in context['relevant_claims'][:3])}
                    
                    Supporting Evidence:
                    {self._format_evidence_for_citations(context['relevant_evidence'][:10])}
                    {self._get_citation_instructions(context)}"""
                }
            ]
            
            response = self.llm_client.generate(messages)
            content = response.content.strip()
            
            # Clean HTML content - remove citations if flag is disabled
            remove_citations = not context.get('include_in_text_citations', True)
            cleaned_content = self._clean_html_content(content, remove_citations=remove_citations)
            
            # Additional pass to fix any remaining punctuation issues
            cleaned_content = self._fix_punctuation_errors(cleaned_content)
            
            # Create content block
            content_block = ContentBlock(
                content=cleaned_content,
                content_type="comparison",
                word_count=len(cleaned_content.split()),
                citations=self._extract_citations_from_content(cleaned_content, context['relevant_evidence']),
                metadata={
                    "llm_model": response.model,
                    "generation_time": response.response_time,
                    "cost": response.cost
                }
            )
            
            return [content_block]
            
        except Exception as e:
            self.logger.error(f"Error generating comparison content: {str(e)}")
            return [self._create_fallback_content_block(context['title'], word_count_target)]
    
    def _generate_table_content(self, context: Dict[str, Any], word_count_target: int) -> List[ContentBlock]:
        """Generate table-based content."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert data analyst and content writer. Create clear table-based content for a {context['tone']} article.
                    
                    Requirements:
                    {self._get_tone_specific_instructions(context['tone'])}
                    - Create detailed, data-rich content with tables when appropriate
                    - Target approximately {word_count_target} words
                    - Cover the key points: {', '.join(context['key_points'])}
                    - Use evidence and claims to support data presentation when available
                    - Write for {context['target_audience']} audience
                    - Include specific data, statistics, and comparisons when you have real data
                    - Make data useful and clear
                    
                    WRITING STYLE REQUIREMENTS:
                    - Write in a conversational, human style - avoid formal or formulaic language
                    - Use descriptive subheadings (H3) that organize content logically
                    - Create clear transitions between paragraphs and tables
                    - Include explanatory text before and after each table
                    - Use TABLES as the primary method for presenting structured information
                    - Include specific examples, data, and practical insights
                    - Write in a natural style that reads like human-written content
                    
                    AVOID AI-GENERATED LANGUAGE:
                    - Do NOT use words like: crucial, embark, paramount, meticulous, navigating, complexities, realm, dive, shall, tailored, towards, underpins, everchanging, ever-evolving, robust, elevate, unleash, cutting-edge, rapidly expanding, mastering, excels, harness, imagine, delve, tapestry, bustling, vibrant, metropolis, labyrinth, gossamer, enigma, whispering, indelible
                    - Do NOT use phrases like: "the world of", "not only", "in today's digital age", "game changer", "designed to enhance", "it is advisable", "when it comes to", "in the realm of", "unlock the secrets", "unveil the secrets", "take a dive into", "as a professional", "you may want to", "it's worth noting that", "to summarize", "ultimately", "to put it simply", "in conclusion", "in summary", "remember that"
                    - Do NOT use transition words excessively: however, therefore, additionally, specifically, generally, consequently, importantly, indeed, thus, alternatively, notably, as well as, despite, essentially, while, unless, also, even though, because, in contrast, although, in order to, due to, even if, given that, arguably, on the other hand, as previously mentioned, subsequently
                    - Do NOT use: firstly, moreover, furthermore, vital, keen, fancy
                    - Write directly and clearly - avoid flowery or overly descriptive language
                    - Use simple, direct language instead of complex phrases
                    
                    TABLE USAGE GUIDELINES:
                    - ONLY create tables when you have specific, meaningful data to present
                    - Use tables for: market data, price comparisons, regional statistics, trend analysis
                    - Use tables for: pros/cons lists, feature comparisons, timeline data
                    - Use tables for: numerical data, percentages, rankings, survey results
                    - Use tables for: performance metrics, cost analysis, decision matrices
                    - DO NOT create tables with placeholder or generic data
                    - If you don't have real data, use paragraphs and lists instead
                    - Always include table headers and proper HTML table structure
                    - Keep tables concise and relevant to the content
                    - Include 1-2 tables per section only when you have actual data
                    
                    TABLE FORMAT:
                    <table style="border-collapse: collapse; width: 100%; margin: 1em 0;">
                    <thead>
                    <tr>
                    <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Header 1</th>
                    <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Header 2</th>
                    </tr>
                    </thead>
                    <tbody>
                    <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Data 1</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">Data 2</td>
                    </tr>
                    </tbody>
                    </table>
                    
                    OUTPUT FORMAT:
                    - Return ONLY clean HTML content without any wrapper tags
                    - Use proper HTML structure: <p> for paragraphs, <h3> for subheadings, <ul>/<ol> for lists
                    - Use proper HTML tables: <table>, <thead>, <tbody>, <tr>, <th>, <td> with appropriate styling
                    - Do NOT include <section>, <article>, or other wrapper tags
                    - Do NOT include markdown code blocks or ```html
                    - Do NOT include H2 headings (only H3 subheadings)
                    - Include tables with proper headers and data organization when presenting structured information
                    - Write in a natural style that flows smoothly from paragraph to paragraph
                    - Include inline citations as [^1], [^2], etc. when referencing evidence or claims
                    - Make it clear and useful with factual accuracy
                    
                    IMPORTANT: If you don't have specific data or evidence to present in tables, focus on creating well-structured paragraphs and lists instead. Only create tables when you have real, meaningful data to display."""
                },
                {
                    "role": "user",
                    "content": f"""Section: {context['title']}
                    {f"Subtitle: {context['subtitle']}" if context['subtitle'] else ""}
                    
                    Key Points to Cover:
                    {chr(10).join(f"- {point}" for point in context['key_points'])}
                    
                    Research Brief: {context['research_brief']}
                    {f"Draft Title: {context['draft_title']}" if context.get('draft_title') else ""}
                    Keywords to integrate naturally: {context.get('keywords', '')}
                    
                    Relevant Claims:
                    {chr(10).join(f"- {claim.get('claim', '')}" for claim in context['relevant_claims'][:3])}
                    
                    Supporting Evidence:
                    {self._format_evidence_for_citations(context['relevant_evidence'][:10])}
                    {self._get_citation_instructions(context)}"""
                }
            ]
            
            response = self.llm_client.generate(messages)
            content = response.content.strip()
            
            # Clean HTML content - remove citations if flag is disabled
            remove_citations = not context.get('include_in_text_citations', True)
            cleaned_content = self._clean_html_content(content, remove_citations=remove_citations)
            
            # Additional pass to fix any remaining punctuation issues
            cleaned_content = self._fix_punctuation_errors(cleaned_content)
            
            # Create content block
            content_block = ContentBlock(
                content=cleaned_content,
                content_type="table",
                word_count=len(cleaned_content.split()),
                citations=self._extract_citations_from_content(cleaned_content, context['relevant_evidence']),
                metadata={
                    "llm_model": response.model,
                    "generation_time": response.response_time,
                    "cost": response.cost
                }
            )
            
            return [content_block]
            
        except Exception as e:
            self.logger.error(f"Error generating table content: {str(e)}")
            return [self._create_fallback_content_block(context['title'], word_count_target)]
    
    def _generate_supporting_content(self, context: Dict[str, Any], remaining_words: int) -> List[ContentBlock]:
        """Generate additional supporting content if needed."""
        try:
            # Ensure we have a meaningful amount to add
            if remaining_words < 100:
                # For small gaps, be more lenient
                remaining_words = max(remaining_words, 150)
            
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert content writer. Add supporting content to complete the section and meet word count requirements.
                    
                    Requirements:
                    {self._get_tone_specific_instructions(context['tone'])}
                    - Add EXACTLY {remaining_words} words - be complete and detailed
                    - Provide additional context, examples, case studies, statistics, or practical insights
                    - Use evidence from the provided sources to support points - cite sources with [^1], [^2], etc.
                    - Make it clear and useful - avoid generic filler
                    - Expand on key points with specific details, examples, and insights
                    - Create well-structured content with clear paragraphs and transitions
                    - Include specific data, statistics, or examples when possible
                    
                    WRITING STYLE REQUIREMENTS:
                    - Write in a conversational, human style - avoid formal or formulaic language
                    - Write in a natural style that reads like human-written content
                    
                    AVOID AI-GENERATED LANGUAGE:
                    - Do NOT use words like: crucial, embark, paramount, meticulous, navigating, complexities, realm, dive, shall, tailored, towards, underpins, everchanging, ever-evolving, robust, elevate, unleash, cutting-edge, rapidly expanding, mastering, excels, harness, imagine, delve, tapestry, bustling, vibrant, metropolis, labyrinth, gossamer, enigma, whispering, indelible
                    - Do NOT use phrases like: "the world of", "not only", "in today's digital age", "game changer", "designed to enhance", "it is advisable", "when it comes to", "in the realm of", "unlock the secrets", "unveil the secrets", "take a dive into", "as a professional", "you may want to", "it's worth noting that", "to summarize", "ultimately", "to put it simply", "in conclusion", "in summary", "remember that"
                    - Do NOT use transition words excessively: however, therefore, additionally, specifically, generally, consequently, importantly, indeed, thus, alternatively, notably, as well as, despite, essentially, while, unless, also, even though, because, in contrast, although, in order to, due to, even if, given that, arguably, on the other hand, as previously mentioned, subsequently
                    - Do NOT use: firstly, moreover, furthermore, vital, keen, fancy
                    - Write directly and clearly - avoid flowery or overly descriptive language
                    - Use simple, direct language instead of complex phrases
                    
                    Supporting Evidence:
                    {self._format_evidence_for_citations(context.get('relevant_evidence', [])[:5])}
                    {self._get_citation_instructions(context)}
                    
                    Return only the additional content that adds real value and meets the word count target."""
                },
                {
                    "role": "user",
                    "content": f"""Section: {context['title']}
                    Research Brief: {context['research_brief']}
                    {f"Draft Title: {context['draft_title']}" if context.get('draft_title') else ""}
                    Keywords to integrate naturally: {context.get('keywords', '')}
                    
                    Add supporting content that enhances the main content and provides additional value to readers."""
                }
            ]
            
            response = self.llm_client.generate(messages)
            content = response.content.strip()
            
            # Clean and fix punctuation issues
            cleaned_content = self._clean_html_content(content, remove_citations=False)
            cleaned_content = self._fix_punctuation_errors(cleaned_content)
            
            # Create content block
            content_block = ContentBlock(
                content=cleaned_content,
                content_type="paragraph",
                word_count=len(cleaned_content.split()),
                citations=self._extract_citations_from_content(cleaned_content, context['relevant_evidence']),
                metadata={
                    "llm_model": response.model,
                    "generation_time": response.response_time,
                    "cost": response.cost,
                    "is_supporting": True
                }
            )
            
            return [content_block]
            
        except Exception as e:
            self.logger.error(f"Error generating supporting content: {str(e)}")
            return []
    
    def _filter_relevant_claims(self, claims: List[Dict], title: str, key_points: List[str]) -> List[Dict]:
        """Filter claims relevant to this section."""
        # Simple keyword-based filtering
        title_keywords = set(title.lower().split())
        key_point_keywords = set()
        for point in key_points:
            key_point_keywords.update(point.lower().split())
        
        relevant_claims = []
        for claim in claims:
            claim_text = claim.get('claim', '').lower()
            claim_keywords = set(claim_text.split())
            
            # Check for keyword overlap
            if title_keywords.intersection(claim_keywords) or key_point_keywords.intersection(claim_keywords):
                relevant_claims.append(claim)
        
        return relevant_claims[:5]  # Limit to top 5 relevant claims
    
    def _filter_relevant_evidence(self, evidence: List[Dict], title: str, key_points: List[str]) -> List[Dict]:
        """Filter evidence relevant to this section."""
        if not evidence:
            return []
        
        # Simple keyword-based filtering with fallback
        title_keywords = set(title.lower().split())
        key_point_keywords = set()
        for point in key_points:
            key_point_keywords.update(point.lower().split())
        
        # Remove stop words from keywords for better matching
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        title_keywords = title_keywords - stop_words
        key_point_keywords = key_point_keywords - stop_words
        
        relevant_evidence = []
        rag_evidence_count = 0
        for ev in evidence:
            content = ev.get('content', '').lower()
            if not content or not content.strip():
                continue
            
            source_type = ev.get('source_type', 'unknown')
            is_rag = (source_type == 'rag')
            
            content_keywords = set(content.split()) - stop_words
            
            # Check for keyword overlap
            title_match = len(title_keywords.intersection(content_keywords)) if title_keywords else 0
            key_point_match = len(key_point_keywords.intersection(content_keywords)) if key_point_keywords else 0
            
            # More lenient matching for RAG sources - they contain valuable structured information
            # Include RAG sources even with weak keyword matches, as they often contain relevant context
            if is_rag:
                # For RAG sources, be very lenient - include if there's any match or if content is substantial
                if title_match > 0 or key_point_match > 0 or len(content) > 500:
                    relevant_evidence.append(ev)
                    rag_evidence_count += 1
            else:
                # For other sources, use standard matching
                if title_match > 0 or key_point_match > 0:
                    relevant_evidence.append(ev)
        
        # Log RAG evidence inclusion
        if rag_evidence_count > 0:
            self.logger.info(f"Included {rag_evidence_count} RAG evidence items (using lenient filtering for RAG sources)")
        
        # If no evidence matched but we have evidence, return first few items as fallback
        # This ensures content generation has something to work with
        if not relevant_evidence and evidence:
            self.logger.info(f"No keyword-matched evidence for section '{title}', using top {min(3, len(evidence))} global evidence items")
            relevant_evidence = evidence[:min(10, len(evidence))]
        
        # Prioritize RAG evidence - include all RAG sources first, then others
        rag_evidence = [ev for ev in relevant_evidence if ev.get('source_type') == 'rag']
        other_evidence = [ev for ev in relevant_evidence if ev.get('source_type') != 'rag']
        # Return up to 10 items, prioritizing RAG sources
        prioritized_evidence = rag_evidence + other_evidence
        return prioritized_evidence[:10]  # Limit to top 10 relevant evidence (increased from 5)
    
    def _extract_citations_from_content(self, content: str, evidence: List[Dict]) -> List[Dict[str, Any]]:
        """Extract citations from generated content using proper citation format."""
        import re
        citations = []
        
        # Extract citation references like [^1], [^2], etc.
        citation_pattern = r'\[\^(\d+)\]'
        citation_matches = re.findall(citation_pattern, content)
        
        # Map citation numbers to evidence
        for citation_num in citation_matches:
            try:
                evidence_index = int(citation_num) - 1  # Convert to 0-based index
                if 0 <= evidence_index < len(evidence):
                    ev = evidence[evidence_index]
                    
                    # Determine source type and create proper citation
                    source_type = ev.get('source_type', 'unknown')
                    source_url = ev.get('source', ev.get('url', ''))
                    
                    # Create meaningful citation based on source type
                    if source_type == 'rag':
                        citation_title = f"Knowledge Base Source {citation_num}"
                        citation_source = "Internal Knowledge Base"
                    elif source_type == 'web':
                        citation_title = ev.get('title', ev.get('source_title', 'Web Source'))
                        citation_source = source_url if source_url else "Web Source"
                    else:
                        citation_title = ev.get('title', ev.get('source_title', 'Unknown Source'))
                        citation_source = source_url if source_url else "Unknown Source"
                    
                    citation = {
                        "citation_id": citation_num,
                        "source": citation_source,
                        "title": citation_title,
                        "content": ev.get('content', '')[:200] + "...",
                        "source_type": source_type,
                        "relevance_score": ev.get('relevance_score', 0.0),
                        "credibility_score": ev.get('credibility_score', 0.0),
                        "metadata": ev.get('metadata', {})
                    }
                    citations.append(citation)
                    
            except (ValueError, IndexError):
                # Invalid citation number, skip
                continue
        
        return citations
    
    def _extract_covered_points(self, content_blocks: List[ContentBlock], key_points: List[str]) -> List[str]:
        """Extract which key points were covered in the content."""
        covered_points = []
        all_content = " ".join([block.content for block in content_blocks]).lower()
        
        for point in key_points:
            point_keywords = set(point.lower().split())
            content_keywords = set(all_content.split())
            
            # Check if key point keywords appear in content
            if point_keywords.intersection(content_keywords):
                covered_points.append(point)
        
        return covered_points
    
    def _balance_content_blocks(self, content_blocks: List[ContentBlock], target_word_count: int) -> List[ContentBlock]:
        """Balance content blocks to meet target word count."""
        if not content_blocks:
            return content_blocks
        
        current_word_count = sum(block.word_count for block in content_blocks)
        
        # Only truncate if content is extremely long (5x target) to prevent abuse
        # This is much more conservative than the previous 2x threshold
        if current_word_count > target_word_count * 5.0:
            self.logger.warning(f"Content extremely long ({current_word_count} words vs {target_word_count} target), applying conservative truncation")
            
            # Keep the first few blocks and truncate the last one conservatively
            target_blocks = []
            accumulated_words = 0
            
            for block in content_blocks:
                if accumulated_words + block.word_count <= target_word_count * 3.0:  # Allow up to 3x target
                    target_blocks.append(block)
                    accumulated_words += block.word_count
                else:
                    # Only truncate if we have very little content left
                    remaining_words = (target_word_count * 3.0) - accumulated_words
                    if remaining_words > 100:  # Only if we have substantial content left
                        # Create a truncated version with proper sentence ending
                        words = block.content.split()
                        truncated_words = words[:remaining_words]
                        
                        # Try to end at a sentence boundary
                        truncated_content = " ".join(truncated_words)
                        if not truncated_content.endswith(('.', '!', '?')):
                            # Find the last sentence ending
                            last_sentence_end = max(
                                truncated_content.rfind('.'),
                                truncated_content.rfind('!'),
                                truncated_content.rfind('?')
                            )
                            if last_sentence_end > len(truncated_content) * 0.7:  # If we can keep 70% of content
                                truncated_content = truncated_content[:last_sentence_end + 1]
                            else:
                                truncated_content += "..."
                        
                        truncated_block = ContentBlock(
                            content=truncated_content,
                            content_type=block.content_type,
                            word_count=len(truncated_content.split()),
                            citations=block.citations,
                            metadata=block.metadata
                        )
                        target_blocks.append(truncated_block)
                    break
            
            return target_blocks
        
        # For normal content length, return as-is without truncation
        return content_blocks
    
    def _create_fallback_content_block(self, title: str, word_count_target: int) -> ContentBlock:
        """Create fallback content block when generation fails."""
        content = f"This section covers {title}. The content provides detailed information and insights on this topic, offering practical guidance and actionable advice for readers."
        
        return ContentBlock(
            content=content,
            content_type="paragraph",
            word_count=len(content.split()),
            citations=[],
            metadata={"is_fallback": True}
        )
    
    def _create_fallback_section_content(self, section_outline: Dict[str, Any]) -> SectionContent:
        """Create fallback section content when generation fails."""
        title = section_outline.get('title', 'Untitled Section')
        word_count_target = section_outline.get('word_count_target', 300)
        
        content_block = self._create_fallback_content_block(title, word_count_target)
        
        return SectionContent(
            title=title,
            subtitle=section_outline.get('subtitle'),
            content_blocks=[content_block],
            total_word_count=content_block.word_count,
            key_points_covered=section_outline.get('key_points', [])[:2],
            citations=[],
            section_order=section_outline.get('order', 1)
        )
    
    def _remove_citation_references(self, content: str) -> str:
        """Remove citation references like [^1], [^2] from content."""
        if not content:
            return content
        
        import re
        # Remove citation references like [^1], [^2], [^3], etc.
        citation_pattern = r'\[\^\d+\]'
        content = re.sub(citation_pattern, '', content)
        # Clean up any extra spaces left behind
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\s+([.,;:!?])', r'\1', content)  # Remove space before punctuation
        return content.strip()
    
    def _fix_punctuation_errors(self, content: str) -> str:
        """Fix common punctuation errors like multiple commas or periods."""
        if not content:
            return content
        
        import re
        
        # Split content into HTML tags and text content to process separately
        # This prevents breaking HTML structure
        parts = re.split(r'(<[^>]+>)', content)
        fixed_parts = []
        
        for part in parts:
            if part.startswith('<') and part.endswith('>'):
                # This is an HTML tag, keep it as-is
                fixed_parts.append(part)
            else:
                # This is text content, fix punctuation
                text = part
                
                # Fix multiple commas: "text,,,,." -> "text."
                text = re.sub(r',{2,}', ',', text)
                
                # Fix multiple periods: "text...." -> "text."
                text = re.sub(r'\.{2,}', '.', text)
                
                # Fix comma followed by period: "text,." -> "text."
                text = re.sub(r',\.', '.', text)
                
                # Fix multiple commas before period: "text,,,." -> "text."
                text = re.sub(r',+\.', '.', text)
                
                # Fix space before punctuation (should be no space, but be careful with HTML entities)
                text = re.sub(r'\s+([.,;:!?])', r'\1', text)
                
                # Fix missing space after punctuation (should have space, but not inside HTML)
                # Only add space if not followed by closing tag or another punctuation
                text = re.sub(r'([.,;:!?])([^\s<])', r'\1 \2', text)
                
                # Fix multiple spaces (but preserve single spaces)
                text = re.sub(r' {2,}', ' ', text)
                
                fixed_parts.append(text)
        
        result = ''.join(fixed_parts)
        
        # Final cleanup: remove any spaces that might have been added incorrectly
        # Remove space before closing tags
        result = re.sub(r'\s+(</)', r'\1', result)
        # Remove space after opening tags (but keep space after self-closing tags like <br />)
        result = re.sub(r'(>)\s+([.,;:!?])', r'\1\2', result)
        
        return result.strip()
    
    def _clean_html_content(self, content: str, remove_citations: bool = False) -> str:
        """Clean and properly format HTML content."""
        if not content:
            return ""
        
        import re
        
        # Remove citation references if requested
        if remove_citations:
            content = self._remove_citation_references(content)
        
        # Fix punctuation errors first
        content = self._fix_punctuation_errors(content)
        
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

def get_tone_specific_instructions(tone: str) -> str:
    """
    Get tone-specific writing instructions for any tone.
    
    This is a standalone function that can be used outside of ContentGenerator
    to get tone-specific instructions for refinement, editing, or other purposes.
    
    Args:
        tone: The writing tone (e.g., 'friendly', 'professional', 'journalistic')
        
    Returns:
        String with tone-specific instructions
    """
    if tone.lower() == 'friendly':
        return """FRIENDLY TONE - PERSONAL, STORY-DRIVEN, AND WARM (CRITICAL):
                
                ⚠️ IMPORTANT: Friendly does NOT mean overly casual greetings
                - DO NOT start with: "Hi friends", "Hey there", "Hello everyone", "What's up", or similar casual greetings
                - Start naturally with the content, not a greeting - jump into the topic conversationally
                - Friendly means warm and personal, not chatty or overly familiar in the opening
                
                MANDATORY: Write like you're sharing a personal story with a close friend
                - Use first-person storytelling: "I've found...", "Last month I...", "My favorite thing about...", "I remember when..."
                - Use "you" and "your" throughout to make it personal and direct
                - Share specific, relatable examples from real life (even if you need to create realistic scenarios)
                - Make it feel like you're having coffee with the reader, not giving a presentation
                - Start with an engaging hook or story, NOT a greeting
                
                MANDATORY: Use very casual, everyday language - NO formal words
                - NEVER use: crucial, paramount, necessitates, cultivate, strategic, trajectory, implement, ensure, facilitate, utilize, leverage, optimize, enhance, robust, comprehensive, meticulous, navigate, realm, embark, underpins, specialized, interconnected, positioning, signifying, potent, commanding, proactive, adept, blend, niche, sought after, individuals, professionals, one must, it is important to note
                - ALWAYS use: "important" not "crucial", "need" not "necessitates", "develop" not "cultivate", "plan" not "strategic", "path" not "trajectory", "use" not "implement", "make sure" not "ensure", "help" not "facilitate", "use" not "utilize", "use" not "leverage", "improve" not "optimize", "make better" not "enhance", "strong" not "robust", "complete" not "comprehensive", "careful" not "meticulous", "work with" not "navigate", "area" not "realm", "start" not "embark", "supports" not "underpins", "special" not "specialized", "connected" not "interconnected", "place" not "positioning", "shows" not "signifying", "strong" not "potent", "earn" not "commanding", "active" not "proactive", "skilled" not "adept", "mix" not "blend", "small area" not "niche", "wanted" not "sought after", "people" not "individuals", "you" not "one must"
                
                MANDATORY: Write with personality and voice - be interesting, not boring
                - Use contractions everywhere: "you'll", "you're", "I've", "it's", "that's", "here's", "don't", "won't", "can't"
                - Use casual phrases: "Here's the thing...", "You know what?", "Let me tell you...", "The cool part is...", "What's wild is..."
                - Ask engaging questions: "Ever wondered why...?", "Want to know something cool?", "Here's what blew my mind..."
                - Use personal touches: "I've been there", "Trust me on this", "I learned this the hard way", "My go-to is..."
                
                MANDATORY: Tell stories and use specific examples
                - Include concrete, relatable scenarios: "Like when I was learning Spanish and AI helped me practice..."
                - Use specific details: "Last month", "on Saturdays", "while I'm sipping coffee", "before lunch"
                - Make examples feel real and personal, not generic
                - Connect ideas through narrative, not just facts
                
                MANDATORY: Keep it warm, encouraging, and human
                - Write like you genuinely care and want to help, not like you're delivering information
                - Be encouraging: "You've got this", "Don't worry", "Here's the fun part...", "You'll love this..."
                - Show enthusiasm and personality - make it interesting to read
                - Avoid sounding like a manual or instruction book
                
                MANDATORY: Explain things simply, like you're talking to a friend
                - When you mention something complex, immediately explain it in plain language
                - Use everyday analogies and comparisons
                - Break things down into simple, digestible pieces
                - Don't assume they know technical terms - explain everything naturally
                
                MANDATORY: Write in a natural, flowing style
                - Each sentence should flow into the next like you're telling a story
                - Mix short punchy sentences with slightly longer ones for rhythm
                - Use casual transitions: "And", "Plus", "Also", "But", "So", "Now", "Here's the thing"
                - Read it aloud mentally - if it sounds formal or boring, make it more casual and personal
                
                EXAMPLE OF GOOD FRIENDLY TONE (notice it starts directly, no greeting):
                "Generative AI has quietly become my favorite coworker. It proofreads my emails while I'm still sipping coffee, turns bullet-point brainstorms into polished slides before lunch, and even codes little snippets so I can skip the Stack-Overflow rabbit hole. Last month it wrote the first draft of my mom's birthday speech—she cried happy tears, not knowing a robot helped me sound poetic. It's my 24-hour tutor, too: when I wanted to learn Spanish, it whipped up mini-conversations, corrected my accent, and cheered me on like a patient amigo. Sure, I double-check facts (old habits die hard), but the time I save lets me volunteer-teach kids on Saturdays—something I always 'meant to do' but never had space for. Basically, AI handles the busywork; I get the fun parts of being human."
                
                Notice: Starts directly with content (no "hi friends" or greeting), first-person, specific examples, casual language, personal stories, warm and engaging - NOT formal or professional, but also NOT overly casual with greetings"""
    elif tone.lower() == 'professional':
        return """PROFESSIONAL TONE - CLEAR AND ACCESSIBLE:
                - Write clearly and directly - professional doesn't mean complex
                - Use simple, precise language that anyone can understand
                - Avoid overly complex vocabulary - choose the simplest word that conveys your meaning
                - Prefer "important" over "crucial", "help" over "facilitate", "use" over "utilize"
                - Write sentences that flow naturally from one idea to the next
                - Connect ideas logically - each sentence should build on the previous one
                - Use active voice when possible - "professionals use" not "it is used by professionals"
                - Keep sentences at a reasonable length (15-25 words) - break up long sentences
                - Make sure each paragraph has a clear point and flows to the next
                - Write as if explaining to a smart colleague, not an academic audience
                - Be authoritative without being pretentious
                - Focus on clarity and usefulness over impressive vocabulary"""
    else:
        return f"""Write in {tone} tone - be clear, natural, and easy to follow"""

# Factory function
def create_content_generator(llm_client, use_verbalized_sampling: bool = True) -> ContentGenerator:
    """
    Create a content generator.
    
    Args:
        llm_client: Configured LLM client
        use_verbalized_sampling: Whether to use verbalized sampling for improved content quality
        
    Returns:
        ContentGenerator instance
    """
    return ContentGenerator(llm_client, use_verbalized_sampling)

# Example usage
if __name__ == "__main__":
    # This would be used with a real LLM client
    print("Content Generator - Ready for integration")
