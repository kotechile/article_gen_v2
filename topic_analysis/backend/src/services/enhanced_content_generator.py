"""
Enhanced Content Generator Service
Generates SEO-optimized content with keyword integration and affiliate opportunities
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
import structlog
from datetime import datetime
import uuid

from ..core.supabase_database import get_supabase_db
from .llm.llm_service import llm_service

logger = structlog.get_logger()

class EnhancedContentGenerator:
    def __init__(self):
        self.db = get_supabase_db()

    async def generate_enhanced_content(
        self,
        content_idea_id: str,
        user_id: str,
        content_type: str = "blog_post",
        target_word_count: int = 2000,
        include_affiliate_opportunities: bool = True
    ) -> Dict[str, Any]:
        """
        Generate enhanced content with SEO optimization and affiliate integration
        """
        try:
            logger.info("Starting enhanced content generation", 
                       content_idea_id=content_idea_id,
                       content_type=content_type)
            
            # Get content idea and keywords
            content_idea = await self._get_content_idea(content_idea_id, user_id)
            if not content_idea:
                raise ValueError("Content idea not found")
            
            keywords = await self._get_optimized_keywords(content_idea_id, user_id)
            if not keywords:
                raise ValueError("No optimized keywords found")
            
            # Generate content structure
            content_structure = await self._generate_content_structure(
                content_idea, keywords, content_type, target_word_count
            )
            
            # Generate SEO-optimized title and meta description
            seo_elements = await self._generate_seo_elements(content_idea, keywords)
            
            # Generate content sections
            content_sections = await self._generate_content_sections(
                content_structure, keywords, content_idea
            )
            
            # Generate affiliate integration points
            affiliate_integration = []
            if include_affiliate_opportunities:
                affiliate_integration = await self._generate_affiliate_integration(
                    content_idea, keywords
                )
            
            # Generate internal linking suggestions
            internal_links = await self._generate_internal_links(keywords, content_idea)
            
            # Generate content outline
            content_outline = await self._generate_content_outline(
                content_structure, content_sections, keywords
            )
            
            # Calculate content metrics
            content_metrics = await self._calculate_content_metrics(
                content_sections, keywords, seo_elements
            )
            
            # Generate content generation prompt
            generation_prompt = await self._generate_content_prompt(
                content_idea, keywords, content_structure, seo_elements
            )
            
            # Save generated content
            generated_content_id = await self._save_generated_content(
                content_idea_id, user_id, {
                    'content_structure': content_structure,
                    'seo_elements': seo_elements,
                    'content_sections': content_sections,
                    'affiliate_integration': affiliate_integration,
                    'internal_links': internal_links,
                    'content_outline': content_outline,
                    'content_metrics': content_metrics,
                    'generation_prompt': generation_prompt,
                    'target_word_count': target_word_count,
                    'content_type': content_type
                }
            )
            
            logger.info("Enhanced content generation completed", 
                       content_idea_id=content_idea_id,
                       generated_content_id=generated_content_id)
            
            return {
                'success': True,
                'generated_content_id': generated_content_id,
                'content_structure': content_structure,
                'seo_elements': seo_elements,
                'content_sections': content_sections,
                'affiliate_integration': affiliate_integration,
                'internal_links': internal_links,
                'content_outline': content_outline,
                'content_metrics': content_metrics,
                'generation_prompt': generation_prompt
            }
            
        except Exception as e:
            logger.error("Enhanced content generation failed", error=str(e))
            raise

    async def _get_content_idea(self, content_idea_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get content idea with enhancement data"""
        try:
            result = self.db.table('content_ideas').select('*').eq('id', content_idea_id).eq('user_id', user_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Failed to get content idea", error=str(e))
            return None

    async def _get_optimized_keywords(self, content_idea_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Get optimized keywords for content idea"""
        try:
            result = self.db.table('individual_keywords').select('*').eq('content_idea_id', content_idea_id).eq('user_id', user_id).eq('is_optimized', True).execute()
            return result.data or []
        except Exception as e:
            logger.error("Failed to get optimized keywords", error=str(e))
            return []

    async def _generate_content_structure(
        self, 
        content_idea: Dict[str, Any], 
        keywords: List[Dict[str, Any]], 
        content_type: str,
        target_word_count: int
    ) -> Dict[str, Any]:
        """Generate content structure based on keywords and content type"""
        
        # Group keywords by type and priority
        primary_keywords = [k for k in keywords if k.get('keyword_type') == 'primary']
        secondary_keywords = [k for k in keywords if k.get('keyword_type') == 'secondary']
        long_tail_keywords = [k for k in keywords if k.get('keyword_type') == 'long_tail']
        question_keywords = [k for k in keywords if 'question' in k.get('keyword_type', '')]
        
        # Calculate section distribution based on word count
        sections = []
        if content_type == "blog_post":
            sections = [
                {'type': 'introduction', 'word_count': int(target_word_count * 0.15), 'keywords': primary_keywords[:2]},
                {'type': 'main_content', 'word_count': int(target_word_count * 0.60), 'keywords': primary_keywords + secondary_keywords},
                {'type': 'detailed_sections', 'word_count': int(target_word_count * 0.15), 'keywords': long_tail_keywords},
                {'type': 'conclusion', 'word_count': int(target_word_count * 0.10), 'keywords': primary_keywords[:1]}
            ]
        elif content_type == "guide":
            sections = [
                {'type': 'introduction', 'word_count': int(target_word_count * 0.10), 'keywords': primary_keywords[:1]},
                {'type': 'step_by_step', 'word_count': int(target_word_count * 0.70), 'keywords': primary_keywords + secondary_keywords},
                {'type': 'tips_and_tricks', 'word_count': int(target_word_count * 0.15), 'keywords': long_tail_keywords},
                {'type': 'conclusion', 'word_count': int(target_word_count * 0.05), 'keywords': primary_keywords[:1]}
            ]
        elif content_type == "review":
            sections = [
                {'type': 'introduction', 'word_count': int(target_word_count * 0.10), 'keywords': primary_keywords[:1]},
                {'type': 'overview', 'word_count': int(target_word_count * 0.20), 'keywords': primary_keywords},
                {'type': 'detailed_review', 'word_count': int(target_word_count * 0.50), 'keywords': secondary_keywords + long_tail_keywords},
                {'type': 'pros_cons', 'word_count': int(target_word_count * 0.15), 'keywords': question_keywords},
                {'type': 'conclusion', 'word_count': int(target_word_count * 0.05), 'keywords': primary_keywords[:1]}
            ]
        
        return {
            'content_type': content_type,
            'target_word_count': target_word_count,
            'sections': sections,
            'keyword_distribution': {
                'primary': len(primary_keywords),
                'secondary': len(secondary_keywords),
                'long_tail': len(long_tail_keywords),
                'questions': len(question_keywords)
            }
        }

    async def _generate_seo_elements(
        self, 
        content_idea: Dict[str, Any], 
        keywords: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate SEO-optimized title and meta description"""
        
        # Get highest priority primary keyword
        primary_keywords = [k for k in keywords if k.get('keyword_type') == 'primary']
        if not primary_keywords:
            primary_keywords = sorted(keywords, key=lambda k: k.get('priority_score', 0), reverse=True)[:1]
        
        best_keyword = primary_keywords[0] if primary_keywords else None
        
        if not best_keyword:
            return {
                'title': content_idea.get('title', ''),
                'meta_description': content_idea.get('description', ''),
                'primary_keyword': '',
                'secondary_keywords': []
            }
        
        # Generate SEO-optimized title
        title_prompt = f"""
        Create an SEO-optimized title for this content idea:
        
        Content Title: {content_idea.get('title', '')}
        Content Description: {content_idea.get('description', '')}
        Primary Keyword: {best_keyword.get('keyword', '')}
        Content Type: {content_idea.get('content_type', 'blog_post')}
        
        Requirements:
        - Include the primary keyword naturally
        - Keep it under 60 characters
        - Make it compelling and click-worthy
        - Use power words when appropriate
        
        Return only the title, no additional text.
        """
        
        response = await llm_service.generate_text(
            prompt=title_prompt,
            max_tokens=100,
            temperature=0.7
        )
        seo_title = response.content
        
        # Generate meta description
        description_prompt = f"""
        Create an SEO-optimized meta description for this content:
        
        Content Title: {seo_title}
        Content Description: {content_idea.get('description', '')}
        Primary Keyword: {best_keyword.get('keyword', '')}
        Secondary Keywords: {[k.get('keyword') for k in keywords[:5]]}
        
        Requirements:
        - Include the primary keyword naturally
        - Keep it between 150-160 characters
        - Include a call-to-action
        - Make it compelling and descriptive
        
        Return only the meta description, no additional text.
        """
        
        response = await llm_service.generate_text(
            prompt=description_prompt,
            max_tokens=150,
            temperature=0.7
        )
        meta_description = response.content
        
        # Get secondary keywords for H2/H3 tags
        secondary_keywords = [k.get('keyword') for k in keywords if k.get('keyword_type') == 'secondary'][:5]
        
        return {
            'title': seo_title.strip(),
            'meta_description': meta_description.strip(),
            'primary_keyword': best_keyword.get('keyword', ''),
            'secondary_keywords': secondary_keywords,
            'focus_keyword_density': 0,  # Will be calculated during content generation
            'readability_score': 0,  # Will be calculated during content generation
            'seo_score': 0  # Will be calculated during content generation
        }

    async def _generate_content_sections(
        self, 
        content_structure: Dict[str, Any], 
        keywords: List[Dict[str, Any]], 
        content_idea: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate detailed content sections"""
        
        sections = []
        
        for section in content_structure['sections']:
            section_keywords = section.get('keywords', [])
            if not section_keywords:
                continue
            
            # Generate section content
            section_prompt = f"""
            Generate content for the {section['type']} section of a {content_structure['content_type']} about {content_idea.get('title', '')}.
            
            Section Type: {section['type']}
            Target Word Count: {section['word_count']}
            Keywords to Include: {[k.get('keyword') for k in section_keywords]}
            Content Description: {content_idea.get('description', '')}
            
            Requirements:
            - Include the specified keywords naturally
            - Write in an engaging, informative tone
            - Provide value to the reader
            - Use proper heading structure (H2, H3)
            - Include specific examples or data when relevant
            - Make it actionable and practical
            
            Format the response as structured content with headings and paragraphs.
            """
            
            response = await llm_service.generate_text(
                prompt=section_prompt,
                max_tokens=section['word_count'] * 2,  # Rough estimate
                temperature=0.7
            )
            section_content = response.content
            
            # Extract headings and content
            headings = self._extract_headings(section_content)
            
            sections.append({
                'type': section['type'],
                'word_count': section['word_count'],
                'content': section_content,
                'headings': headings,
                'keywords_used': [k.get('keyword') for k in section_keywords],
                'keyword_density': self._calculate_keyword_density(section_content, section_keywords)
            })
        
        return sections

    async def _generate_affiliate_integration(
        self, 
        content_idea: Dict[str, Any], 
        keywords: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate affiliate integration opportunities"""
        
        # Find keywords with high affiliate potential
        affiliate_keywords = [
            k for k in keywords 
            if k.get('affiliate_potential_score', 0) >= 60
        ]
        
        if not affiliate_keywords:
            return []
        
        integration_points = []
        
        for keyword in affiliate_keywords[:5]:  # Limit to top 5
            # Generate affiliate integration content
            integration_prompt = f"""
            Create affiliate integration content for this keyword: {keyword.get('keyword')}
            
            Content Context: {content_idea.get('title', '')}
            Keyword Type: {keyword.get('keyword_type', '')}
            Affiliate Potential Score: {keyword.get('affiliate_potential_score', 0)}
            Suggested Networks: {keyword.get('suggested_affiliate_networks', [])}
            Monetization Opportunities: {keyword.get('monetization_opportunities', [])}
            
            Create:
            1. A natural product recommendation paragraph
            2. A comparison section if applicable
            3. A call-to-action for the affiliate link
            4. Related product suggestions
            
            Make it valuable and not overly promotional.
            """
            
            response = await llm_service.generate_text(
                prompt=integration_prompt,
                max_tokens=300,
                temperature=0.6
            )
            integration_content = response.content
            
            integration_points.append({
                'keyword': keyword.get('keyword'),
                'affiliate_potential_score': keyword.get('affiliate_potential_score', 0),
                'suggested_networks': keyword.get('suggested_affiliate_networks', []),
                'integration_content': integration_content,
                'placement_suggestion': self._suggest_placement(keyword, content_idea),
                'monetization_opportunities': keyword.get('monetization_opportunities', [])
            })
        
        return integration_points

    async def _generate_internal_links(
        self, 
        keywords: List[Dict[str, Any]], 
        content_idea: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate internal linking suggestions"""
        
        internal_links = []
        
        for keyword in keywords[:10]:  # Limit to top 10
            if not keyword.get('internal_link_suggestions'):
                continue
            
            for suggestion in keyword.get('internal_link_suggestions', []):
                internal_links.append({
                    'keyword': keyword.get('keyword'),
                    'anchor_text': suggestion,
                    'target_topic': self._extract_topic_from_suggestion(suggestion),
                    'placement_context': f"Use when discussing {keyword.get('keyword')}",
                    'link_value': self._calculate_link_value(keyword)
                })
        
        return internal_links

    async def _generate_content_outline(
        self, 
        content_structure: Dict[str, Any], 
        content_sections: List[Dict[str, Any]], 
        keywords: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive content outline"""
        
        outline = {
            'title': '',
            'meta_description': '',
            'introduction': '',
            'main_sections': [],
            'conclusion': '',
            'key_points': [],
            'faq_section': [],
            'resources': []
        }
        
        # Generate FAQ section from question-based keywords
        question_keywords = [k for k in keywords if 'question' in k.get('keyword_type', '')]
        if question_keywords:
            faq_prompt = f"""
            Create a FAQ section based on these question keywords:
            {[k.get('keyword') for k in question_keywords]}
            
            Content Context: {content_structure.get('content_type', 'blog_post')}
            
            Generate 3-5 frequently asked questions with detailed answers.
            """
            
            response = await llm_service.generate_text(
                prompt=faq_prompt,
                max_tokens=500,
                temperature=0.7
            )
            faq_content = response.content
            
            outline['faq_section'] = self._parse_faq_content(faq_content)
        
        # Generate key points from high-priority keywords
        high_priority_keywords = [k for k in keywords if k.get('priority_score', 0) >= 80]
        outline['key_points'] = [k.get('keyword') for k in high_priority_keywords[:10]]
        
        return outline

    async def _calculate_content_metrics(
        self, 
        content_sections: List[Dict[str, Any]], 
        keywords: List[Dict[str, Any]], 
        seo_elements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate content quality metrics"""
        
        total_word_count = sum(section.get('word_count', 0) for section in content_sections)
        total_keywords_used = len(set([
            keyword for section in content_sections 
            for keyword in section.get('keywords_used', [])
        ]))
        
        # Calculate keyword density
        primary_keyword = seo_elements.get('primary_keyword', '')
        if primary_keyword:
            all_content = ' '.join([section.get('content', '') for section in content_sections])
            keyword_count = all_content.lower().count(primary_keyword.lower())
            keyword_density = (keyword_count / len(all_content.split())) * 100
        else:
            keyword_density = 0
        
        # Calculate readability score (simplified)
        avg_sentence_length = self._calculate_avg_sentence_length(content_sections)
        readability_score = max(0, min(100, 100 - (avg_sentence_length - 15) * 2))
        
        # Calculate SEO score
        seo_score = self._calculate_seo_score(seo_elements, keyword_density, total_keywords_used)
        
        return {
            'total_word_count': total_word_count,
            'total_keywords_used': total_keywords_used,
            'keyword_density': round(keyword_density, 2),
            'readability_score': round(readability_score, 1),
            'seo_score': round(seo_score, 1),
            'content_quality_score': round((readability_score + seo_score) / 2, 1),
            'keyword_coverage': round((total_keywords_used / len(keywords)) * 100, 1) if keywords else 0
        }

    async def _generate_content_prompt(
        self, 
        content_idea: Dict[str, Any], 
        keywords: List[Dict[str, Any]], 
        content_structure: Dict[str, Any], 
        seo_elements: Dict[str, Any]
    ) -> str:
        """Generate comprehensive content generation prompt"""
        
        primary_keywords = [k.get('keyword') for k in keywords if k.get('keyword_type') == 'primary']
        secondary_keywords = [k.get('keyword') for k in keywords if k.get('keyword_type') == 'secondary']
        
        prompt = f"""
        Generate a comprehensive {content_structure['content_type']} about: {content_idea.get('title', '')}
        
        CONTENT REQUIREMENTS:
        - Target word count: {content_structure['target_word_count']}
        - Content type: {content_structure['content_type']}
        - Tone: Professional yet engaging
        
        SEO OPTIMIZATION:
        - Primary keyword: {seo_elements.get('primary_keyword', '')}
        - Secondary keywords: {secondary_keywords[:5]}
        - Title: {seo_elements.get('title', '')}
        - Meta description: {seo_elements.get('meta_description', '')}
        
        KEYWORD INTEGRATION:
        - Include all primary keywords naturally
        - Use secondary keywords in subheadings and content
        - Maintain keyword density between 1-2%
        - Use long-tail keywords in detailed sections
        
        CONTENT STRUCTURE:
        {json.dumps(content_structure, indent=2)}
        
        AFFILIATE INTEGRATION:
        - Include product recommendations where relevant
        - Add comparison sections for commercial keywords
        - Use natural call-to-actions
        
        QUALITY REQUIREMENTS:
        - Write for humans, optimize for search engines
        - Include specific examples and data
        - Make it actionable and valuable
        - Use proper heading hierarchy (H1, H2, H3)
        - Include internal linking opportunities
        - Add FAQ section if applicable
        
        Generate the complete article following these guidelines.
        """
        
        return prompt

    async def _save_generated_content(
        self, 
        content_idea_id: str, 
        user_id: str, 
        content_data: Dict[str, Any]
    ) -> str:
        """Save generated content to database"""
        
        try:
            generated_content_id = str(uuid.uuid4())
            
            content_record = {
                'id': generated_content_id,
                'user_id': user_id,
                'content_idea_id': content_idea_id,
                'content_type': content_data.get('content_type', 'blog_post'),
                'title': content_data.get('seo_elements', {}).get('title', ''),
                'meta_description': content_data.get('seo_elements', {}).get('meta_description', ''),
                'content_structure': content_data.get('content_structure', {}),
                'content_sections': content_data.get('content_sections', []),
                'affiliate_integration': content_data.get('affiliate_integration', []),
                'internal_links': content_data.get('internal_links', []),
                'content_outline': content_data.get('content_outline', {}),
                'content_metrics': content_data.get('content_metrics', {}),
                'generation_prompt': content_data.get('generation_prompt', ''),
                'target_word_count': content_data.get('target_word_count', 0),
                'status': 'generated',
                'created_at': datetime.utcnow().isoformat()
            }
            
            # Save to generated_content table (you may need to create this table)
            # For now, we'll update the content_ideas table
            update_data = {
                'content_generation_prompt': content_data.get('generation_prompt', ''),
                'content_generation_parameters': {
                    'target_word_count': content_data.get('target_word_count', 0),
                    'content_type': content_data.get('content_type', 'blog_post'),
                    'generated_at': datetime.utcnow().isoformat()
                },
                'updated_at': datetime.utcnow().isoformat()
            }
            
            self.db.table('content_ideas').update(update_data).eq('id', content_idea_id).execute()
            
            logger.info("Generated content saved", generated_content_id=generated_content_id)
            return generated_content_id
            
        except Exception as e:
            logger.error("Failed to save generated content", error=str(e))
            raise

    def _extract_headings(self, content: str) -> List[str]:
        """Extract headings from content"""
        import re
        headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        return headings

    def _calculate_keyword_density(self, content: str, keywords: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate keyword density for content"""
        word_count = len(content.split())
        densities = {}
        
        for keyword in keywords:
            keyword_text = keyword.get('keyword', '')
            if keyword_text:
                keyword_count = content.lower().count(keyword_text.lower())
                density = (keyword_count / word_count) * 100 if word_count > 0 else 0
                densities[keyword_text] = round(density, 2)
        
        return densities

    def _suggest_placement(self, keyword: Dict[str, Any], content_idea: Dict[str, Any]) -> str:
        """Suggest placement for affiliate integration"""
        keyword_type = keyword.get('keyword_type', '')
        
        if keyword_type == 'primary':
            return "Introduction or main content section"
        elif keyword_type == 'comparison':
            return "Comparison or review section"
        elif keyword_type == 'question_based':
            return "FAQ or Q&A section"
        else:
            return "Relevant content section"

    def _extract_topic_from_suggestion(self, suggestion: str) -> str:
        """Extract topic from internal link suggestion"""
        # Simple topic extraction - you might want to improve this
        return suggestion.replace('link to', '').replace('connect to', '').strip()

    def _calculate_link_value(self, keyword: Dict[str, Any]) -> int:
        """Calculate internal link value score"""
        priority_score = keyword.get('priority_score', 0)
        opportunity_score = keyword.get('opportunity_score', 0)
        return int((priority_score + opportunity_score) / 2)

    def _parse_faq_content(self, faq_content: str) -> List[Dict[str, str]]:
        """Parse FAQ content into structured format"""
        # Simple FAQ parsing - you might want to improve this
        faqs = []
        lines = faq_content.split('\n')
        
        current_question = None
        current_answer = []
        
        for line in lines:
            if line.strip().endswith('?'):
                if current_question:
                    faqs.append({
                        'question': current_question,
                        'answer': ' '.join(current_answer).strip()
                    })
                current_question = line.strip()
                current_answer = []
            elif line.strip() and current_question:
                current_answer.append(line.strip())
        
        if current_question:
            faqs.append({
                'question': current_question,
                'answer': ' '.join(current_answer).strip()
            })
        
        return faqs

    def _calculate_avg_sentence_length(self, content_sections: List[Dict[str, Any]]) -> float:
        """Calculate average sentence length for readability"""
        all_content = ' '.join([section.get('content', '') for section in content_sections])
        sentences = all_content.split('.')
        words = all_content.split()
        
        if len(sentences) > 0:
            return len(words) / len(sentences)
        return 0

    def _calculate_seo_score(
        self, 
        seo_elements: Dict[str, Any], 
        keyword_density: float, 
        total_keywords_used: int
    ) -> float:
        """Calculate SEO score based on various factors"""
        score = 0
        
        # Title optimization (20 points)
        if seo_elements.get('title'):
            title = seo_elements['title']
            if len(title) <= 60:
                score += 20
            elif len(title) <= 70:
                score += 15
            else:
                score += 10
        
        # Meta description optimization (20 points)
        if seo_elements.get('meta_description'):
            meta_desc = seo_elements['meta_description']
            if 150 <= len(meta_desc) <= 160:
                score += 20
            elif 140 <= len(meta_desc) <= 170:
                score += 15
            else:
                score += 10
        
        # Keyword density (20 points)
        if 1 <= keyword_density <= 2:
            score += 20
        elif 0.5 <= keyword_density <= 3:
            score += 15
        else:
            score += 10
        
        # Keyword usage (20 points)
        if total_keywords_used >= 10:
            score += 20
        elif total_keywords_used >= 5:
            score += 15
        else:
            score += 10
        
        # Primary keyword in title (20 points)
        if seo_elements.get('primary_keyword') and seo_elements.get('title'):
            if seo_elements['primary_keyword'].lower() in seo_elements['title'].lower():
                score += 20
        
        return min(100, score)

