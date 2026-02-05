"""
Individual Keyword Optimizer Service
Handles LLM-based optimization of individual keywords with detailed analysis
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
import structlog
from datetime import datetime
import uuid

from ..core.supabase_database import get_supabase_db
from ..core.llm_config import LLMConfigManager

logger = structlog.get_logger()

class IndividualKeywordOptimizer:
    def __init__(self):
        self.db = get_supabase_db()
        self.llm_manager = LLMConfigManager()

    async def optimize_keywords(
        self,
        keywords: List[Dict[str, Any]],
        content_idea_id: str,
        user_id: str,
        optimization_session_id: str
    ) -> Dict[str, Any]:
        """
        Optimize individual keywords using LLM analysis
        """
        try:
            logger.info("Starting keyword optimization", 
                       keyword_count=len(keywords), 
                       content_idea_id=content_idea_id)
            
            optimized_keywords = []
            optimization_summary = {
                'total_keywords': len(keywords),
                'optimized_keywords': 0,
                'high_opportunity_keywords': 0,
                'affiliate_potential_keywords': 0,
                'content_suggestions': [],
                'seo_recommendations': [],
                'affiliate_opportunities': []
            }

            for keyword_data in keywords:
                try:
                    optimized_keyword = await self._optimize_single_keyword(
                        keyword_data, content_idea_id, user_id
                    )
                    optimized_keywords.append(optimized_keyword)
                    optimization_summary['optimized_keywords'] += 1
                    
                    # Track high-opportunity keywords
                    if optimized_keyword.get('opportunity_score', 0) >= 70:
                        optimization_summary['high_opportunity_keywords'] += 1
                    
                    # Track affiliate potential
                    if optimized_keyword.get('affiliate_potential_score', 0) >= 60:
                        optimization_summary['affiliate_potential_keywords'] += 1
                        
                except Exception as e:
                    logger.error("Failed to optimize keyword", 
                               keyword=keyword_data.get('keyword', 'unknown'), 
                               error=str(e))
                    continue

            # Generate content suggestions based on all keywords
            content_suggestions = await self._generate_content_suggestions(
                optimized_keywords, content_idea_id
            )
            optimization_summary['content_suggestions'] = content_suggestions

            # Generate SEO recommendations
            seo_recommendations = await self._generate_seo_recommendations(
                optimized_keywords, content_idea_id
            )
            optimization_summary['seo_recommendations'] = seo_recommendations

            # Generate affiliate opportunities
            affiliate_opportunities = await self._generate_affiliate_opportunities(
                optimized_keywords, content_idea_id
            )
            optimization_summary['affiliate_opportunities'] = affiliate_opportunities

            # Save optimization session
            await self._save_optimization_session(
                optimization_session_id, content_idea_id, user_id, 
                optimization_summary, optimized_keywords
            )

            # Save individual keywords to database
            await self._save_optimized_keywords(
                optimized_keywords, content_idea_id, user_id, optimization_session_id
            )

            # Update content idea with enhanced data
            await self._update_content_idea_enhancement(
                content_idea_id, optimization_summary, optimized_keywords
            )

            logger.info("Keyword optimization completed", 
                       optimized_count=len(optimized_keywords))
            
            return {
                'success': True,
                'optimized_keywords': optimized_keywords,
                'optimization_summary': optimization_summary,
                'session_id': optimization_session_id
            }

        except Exception as e:
            logger.error("Keyword optimization failed", error=str(e))
            raise

    async def _optimize_single_keyword(
        self, 
        keyword_data: Dict[str, Any], 
        content_idea_id: str, 
        user_id: str
    ) -> Dict[str, Any]:
        """
        Optimize a single keyword using LLM analysis
        """
        keyword = keyword_data.get('keyword', '')
        search_volume = keyword_data.get('search_volume', 0)
        keyword_difficulty = keyword_data.get('keyword_difficulty', 0)
        cpc = keyword_data.get('cpc', 0)
        opportunity_score = keyword_data.get('opportunity_score', 0)

        # Get content idea details for context
        content_idea = await self._get_content_idea(content_idea_id)
        if not content_idea:
            raise ValueError("Content idea not found")

        # Generate LLM optimization prompt
        optimization_prompt = self._create_optimization_prompt(
            keyword, keyword_data, content_idea
        )

        # Get LLM response
        llm_response = await self.llm_manager.generate_content(
            prompt=optimization_prompt,
            max_tokens=1000,
            temperature=0.7
        )

        # Parse LLM response
        optimization_data = self._parse_llm_response(llm_response)

        # Calculate additional metrics
        affiliate_potential_score = self._calculate_affiliate_potential(
            keyword_data, optimization_data
        )

        # Generate content suggestions
        content_suggestions = await self._generate_keyword_content_suggestions(
            keyword, keyword_data, content_idea
        )

        # Generate heading suggestions
        heading_suggestions = await self._generate_heading_suggestions(
            keyword, keyword_data, content_idea
        )

        # Generate internal link suggestions
        internal_link_suggestions = await self._generate_internal_link_suggestions(
            keyword, keyword_data, content_idea
        )

        # Generate related questions
        related_questions = await self._generate_related_questions(
            keyword, keyword_data, content_idea
        )

        # Generate affiliate suggestions
        affiliate_suggestions = await self._generate_affiliate_suggestions(
            keyword, keyword_data, content_idea
        )

        return {
            **keyword_data,
            'id': str(uuid.uuid4()),
            'content_idea_id': content_idea_id,
            'user_id': user_id,
            'is_optimized': True,
            'optimized_at': datetime.utcnow().isoformat(),
            
            # LLM optimizations
            'llm_optimized_title': optimization_data.get('optimized_title', ''),
            'llm_optimized_description': optimization_data.get('optimized_description', ''),
            'llm_keyword_variations': optimization_data.get('keyword_variations', []),
            'llm_content_angle': optimization_data.get('content_angle', ''),
            'llm_target_audience': optimization_data.get('target_audience', ''),
            
            # Content optimization
            'content_suggestions': content_suggestions,
            'heading_suggestions': heading_suggestions,
            'internal_link_suggestions': internal_link_suggestions,
            'related_questions': related_questions,
            
            # Affiliate integration
            'affiliate_potential_score': affiliate_potential_score,
            'suggested_affiliate_networks': affiliate_suggestions.get('networks', []),
            'monetization_opportunities': affiliate_suggestions.get('opportunities', []),
            
            # Quality metrics
            'relevance_score': optimization_data.get('relevance_score', 0),
            'optimization_score': optimization_data.get('optimization_score', 0),
            'priority_score': self._calculate_priority_score(keyword_data, optimization_data),
        }

    def _create_optimization_prompt(
        self, 
        keyword: str, 
        keyword_data: Dict[str, Any], 
        content_idea: Dict[str, Any]
    ) -> str:
        """
        Create LLM prompt for keyword optimization
        """
        return f"""
You are an expert SEO content strategist. Analyze this keyword and provide optimization recommendations for a content idea.

CONTENT IDEA:
Title: {content_idea.get('title', '')}
Description: {content_idea.get('description', '')}
Content Type: {content_idea.get('content_type', '')}

KEYWORD TO OPTIMIZE:
Keyword: {keyword}
Search Volume: {keyword_data.get('search_volume', 0):,}
Keyword Difficulty: {keyword_data.get('keyword_difficulty', 0)}/100
CPC: ${keyword_data.get('cpc', 0):.2f}
Opportunity Score: {keyword_data.get('opportunity_score', 0):.1f}%

Please provide a JSON response with the following structure:
{{
    "optimized_title": "SEO-optimized title incorporating the keyword",
    "optimized_description": "Meta description optimized for the keyword",
    "keyword_variations": ["list", "of", "related", "keyword", "variations"],
    "content_angle": "Specific angle or approach for this keyword",
    "target_audience": "Primary target audience for this keyword",
    "relevance_score": 85,
    "optimization_score": 90,
    "content_suggestions": [
        "Specific content suggestions for this keyword",
        "Additional content ideas"
    ],
    "heading_suggestions": [
        "H2 heading suggestions",
        "H3 heading suggestions"
    ],
    "internal_link_suggestions": [
        "Related topics to link to",
        "Anchor text suggestions"
    ],
    "related_questions": [
        "Questions people ask about this keyword",
        "FAQ suggestions"
    ]
}}

Focus on:
1. High search intent alignment
2. Content depth and value
3. SEO optimization opportunities
4. User engagement potential
5. Monetization possibilities
"""

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response and extract optimization data
        """
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback parsing if JSON not found
                return self._fallback_parse(response)
        except json.JSONDecodeError:
            return self._fallback_parse(response)

    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """
        Fallback parsing when JSON extraction fails
        """
        return {
            'optimized_title': f"Best {response[:50]}...",
            'optimized_description': f"Complete guide to {response[:100]}...",
            'keyword_variations': [],
            'content_angle': 'Informational',
            'target_audience': 'General audience',
            'relevance_score': 75,
            'optimization_score': 80,
            'content_suggestions': ['Focus on user value', 'Include practical examples'],
            'heading_suggestions': ['Introduction', 'Key Points', 'Conclusion'],
            'internal_link_suggestions': ['Related topics'],
            'related_questions': ['What is this about?', 'How does it work?']
        }

    def _calculate_affiliate_potential(
        self, 
        keyword_data: Dict[str, Any], 
        optimization_data: Dict[str, Any]
    ) -> float:
        """
        Calculate affiliate monetization potential score
        """
        base_score = keyword_data.get('opportunity_score', 0)
        cpc_bonus = min(keyword_data.get('cpc', 0) * 10, 20)
        search_volume_bonus = min(keyword_data.get('search_volume', 0) / 1000, 15)
        difficulty_penalty = keyword_data.get('keyword_difficulty', 0) * 0.2
        
        return max(0, min(100, base_score + cpc_bonus + search_volume_bonus - difficulty_penalty))

    def _calculate_priority_score(
        self, 
        keyword_data: Dict[str, Any], 
        optimization_data: Dict[str, Any]
    ) -> float:
        """
        Calculate priority score for content creation
        """
        opportunity = keyword_data.get('opportunity_score', 0)
        relevance = optimization_data.get('relevance_score', 0)
        search_volume = min(keyword_data.get('search_volume', 0) / 1000, 20)
        difficulty_penalty = keyword_data.get('keyword_difficulty', 0) * 0.3
        
        return max(0, min(100, opportunity * 0.4 + relevance * 0.3 + search_volume - difficulty_penalty))

    async def _generate_content_suggestions(
        self, 
        optimized_keywords: List[Dict[str, Any]], 
        content_idea_id: str
    ) -> List[str]:
        """
        Generate content suggestions based on all optimized keywords
        """
        # Group keywords by type and generate suggestions
        suggestions = []
        
        primary_keywords = [k for k in optimized_keywords if k.get('keyword_type') == 'primary']
        if primary_keywords:
            suggestions.append(f"Focus on {len(primary_keywords)} primary keywords for main content")
        
        long_tail_keywords = [k for k in optimized_keywords if k.get('keyword_type') == 'long_tail']
        if long_tail_keywords:
            suggestions.append(f"Include {len(long_tail_keywords)} long-tail keywords for detailed sections")
        
        question_keywords = [k for k in optimized_keywords if 'question' in k.get('keyword_type', '')]
        if question_keywords:
            suggestions.append(f"Create FAQ section with {len(question_keywords)} question-based keywords")
        
        return suggestions

    async def _generate_seo_recommendations(
        self, 
        optimized_keywords: List[Dict[str, Any]], 
        content_idea_id: str
    ) -> List[str]:
        """
        Generate SEO recommendations based on optimized keywords
        """
        recommendations = []
        
        high_volume_keywords = [k for k in optimized_keywords if k.get('search_volume', 0) > 1000]
        if high_volume_keywords:
            recommendations.append(f"Prioritize {len(high_volume_keywords)} high-volume keywords")
        
        low_difficulty_keywords = [k for k in optimized_keywords if k.get('keyword_difficulty', 100) < 40]
        if low_difficulty_keywords:
            recommendations.append(f"Target {len(low_difficulty_keywords)} low-difficulty keywords for quick wins")
        
        high_cpc_keywords = [k for k in optimized_keywords if k.get('cpc', 0) > 1.0]
        if high_cpc_keywords:
            recommendations.append(f"Focus on {len(high_cpc_keywords)} high-CPC commercial keywords")
        
        return recommendations

    async def _generate_affiliate_opportunities(
        self, 
        optimized_keywords: List[Dict[str, Any]], 
        content_idea_id: str
    ) -> List[str]:
        """
        Generate affiliate monetization opportunities
        """
        opportunities = []
        
        affiliate_keywords = [k for k in optimized_keywords if k.get('affiliate_potential_score', 0) > 60]
        if affiliate_keywords:
            opportunities.append(f"Monetize {len(affiliate_keywords)} high-affiliate-potential keywords")
        
        commercial_keywords = [k for k in optimized_keywords if k.get('search_intent') == 'commercial']
        if commercial_keywords:
            opportunities.append(f"Target {len(commercial_keywords)} commercial-intent keywords for product recommendations")
        
        return opportunities

    async def _get_content_idea(self, content_idea_id: str) -> Optional[Dict[str, Any]]:
        """
        Get content idea details from database
        """
        try:
            result = self.db.table('content_ideas').select('*').eq('id', content_idea_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error("Failed to get content idea", error=str(e))
            return None

    async def _save_optimization_session(
        self,
        session_id: str,
        content_idea_id: str,
        user_id: str,
        optimization_summary: Dict[str, Any],
        optimized_keywords: List[Dict[str, Any]]
    ):
        """
        Save optimization session to database
        """
        try:
            session_data = {
                'id': session_id,
                'user_id': user_id,
                'content_idea_id': content_idea_id,
                'session_name': f'Keyword Optimization - {datetime.utcnow().strftime("%Y-%m-%d %H:%M")}',
                'optimization_type': 'ahrefs_upload',
                'keywords_processed': optimization_summary['total_keywords'],
                'keywords_optimized': optimization_summary['optimized_keywords'],
                'optimization_summary': optimization_summary,
                'top_performing_keywords': [
                    k for k in optimized_keywords 
                    if k.get('priority_score', 0) >= 80
                ][:10],
                'optimization_recommendations': optimization_summary.get('content_suggestions', []),
                'overall_optimization_score': self._calculate_overall_score(optimized_keywords),
                'seo_improvement_score': self._calculate_seo_improvement(optimized_keywords),
                'content_potential_score': self._calculate_content_potential(optimized_keywords),
                'status': 'completed',
                'is_active': True,
                'completed_at': datetime.utcnow().isoformat()
            }
            
            self.db.table('keyword_optimization_sessions').insert(session_data).execute()
            logger.info("Optimization session saved", session_id=session_id)
            
        except Exception as e:
            logger.error("Failed to save optimization session", error=str(e))
            raise

    async def _save_optimized_keywords(
        self,
        optimized_keywords: List[Dict[str, Any]],
        content_idea_id: str,
        user_id: str,
        optimization_session_id: str
    ):
        """
        Save optimized keywords to database
        """
        try:
            keyword_records = []
            for keyword in optimized_keywords:
                record = {
                    'id': keyword.get('id'),
                    'user_id': user_id,
                    'content_idea_id': content_idea_id,
                    'keyword': keyword.get('keyword'),
                    'keyword_type': keyword.get('keyword_type', 'primary'),
                    'search_volume': keyword.get('search_volume', 0),
                    'keyword_difficulty': keyword.get('keyword_difficulty', 0),
                    'cpc': keyword.get('cpc', 0),
                    'traffic_potential': keyword.get('traffic_potential', 0),
                    'clicks': keyword.get('clicks', 0),
                    'impressions': keyword.get('impressions', 0),
                    'ctr': keyword.get('ctr', 0),
                    'position': keyword.get('position', 0),
                    'search_intent': keyword.get('search_intent', 'informational'),
                    'competition_level': keyword.get('competition_level', 'medium'),
                    'trend_score': keyword.get('trend_score', 0),
                    'opportunity_score': keyword.get('opportunity_score', 0),
                    'content_suggestions': keyword.get('content_suggestions', []),
                    'heading_suggestions': keyword.get('heading_suggestions', []),
                    'internal_link_suggestions': keyword.get('internal_link_suggestions', []),
                    'related_questions': keyword.get('related_questions', []),
                    'llm_optimized_title': keyword.get('llm_optimized_title', ''),
                    'llm_optimized_description': keyword.get('llm_optimized_description', ''),
                    'llm_keyword_variations': keyword.get('llm_keyword_variations', []),
                    'llm_content_angle': keyword.get('llm_content_angle', ''),
                    'llm_target_audience': keyword.get('llm_target_audience', ''),
                    'affiliate_potential_score': keyword.get('affiliate_potential_score', 0),
                    'suggested_affiliate_networks': keyword.get('suggested_affiliate_networks', []),
                    'monetization_opportunities': keyword.get('monetization_opportunities', []),
                    'source_tool': 'ahrefs',
                    'relevance_score': keyword.get('relevance_score', 0),
                    'optimization_score': keyword.get('optimization_score', 0),
                    'priority_score': keyword.get('priority_score', 0),
                    'is_optimized': True,
                    'is_used_in_content': False,
                    'is_archived': False,
                    'optimized_at': keyword.get('optimized_at')
                }
                keyword_records.append(record)
            
            self.db.table('individual_keywords').insert(keyword_records).execute()
            logger.info("Optimized keywords saved", count=len(keyword_records))
            
        except Exception as e:
            logger.error("Failed to save optimized keywords", error=str(e))
            raise

    async def _update_content_idea_enhancement(
        self,
        content_idea_id: str,
        optimization_summary: Dict[str, Any],
        optimized_keywords: List[Dict[str, Any]]
    ):
        """
        Update content idea with enhancement data
        """
        try:
            # Calculate enhanced metrics
            enhanced_keywords_data = [
                {
                    'keyword': k.get('keyword'),
                    'type': k.get('keyword_type'),
                    'search_volume': k.get('search_volume'),
                    'difficulty': k.get('keyword_difficulty'),
                    'opportunity_score': k.get('opportunity_score'),
                    'priority_score': k.get('priority_score')
                }
                for k in optimized_keywords
            ]
            
            # Generate SEO-optimized title and description
            seo_title = await self._generate_seo_optimized_title(optimized_keywords)
            seo_description = await self._generate_seo_optimized_description(optimized_keywords)
            
            # Get primary keywords
            primary_keywords = [k.get('keyword') for k in optimized_keywords if k.get('keyword_type') == 'primary']
            
            # Calculate keyword metrics summary
            keyword_metrics = {
                'total_keywords': len(optimized_keywords),
                'avg_search_volume': sum(k.get('search_volume', 0) for k in optimized_keywords) / len(optimized_keywords),
                'avg_difficulty': sum(k.get('keyword_difficulty', 0) for k in optimized_keywords) / len(optimized_keywords),
                'avg_opportunity_score': sum(k.get('opportunity_score', 0) for k in optimized_keywords) / len(optimized_keywords),
                'high_priority_keywords': len([k for k in optimized_keywords if k.get('priority_score', 0) >= 80])
            }
            
            # Get affiliate networks
            affiliate_networks = list(set([
                network for k in optimized_keywords 
                for network in k.get('suggested_affiliate_networks', [])
            ]))
            
            update_data = {
                'enhanced_keywords_data': enhanced_keywords_data,
                'seo_optimized_title': seo_title,
                'seo_optimized_description': seo_description,
                'primary_keywords_optimized': primary_keywords,
                'keyword_metrics_summary': keyword_metrics,
                'affiliate_networks_suggested': affiliate_networks,
                'is_enhanced': True,
                'enhancement_timestamp': datetime.utcnow().isoformat()
            }
            
            self.db.table('content_ideas').update(update_data).eq('id', content_idea_id).execute()
            logger.info("Content idea enhanced", content_idea_id=content_idea_id)
            
        except Exception as e:
            logger.error("Failed to update content idea enhancement", error=str(e))
            raise

    def _calculate_overall_score(self, optimized_keywords: List[Dict[str, Any]]) -> float:
        """Calculate overall optimization score"""
        if not optimized_keywords:
            return 0
        
        scores = [k.get('optimization_score', 0) for k in optimized_keywords]
        return sum(scores) / len(scores)

    def _calculate_seo_improvement(self, optimized_keywords: List[Dict[str, Any]]) -> float:
        """Calculate SEO improvement score"""
        if not optimized_keywords:
            return 0
        
        # Based on opportunity scores and search volumes
        scores = []
        for k in optimized_keywords:
            opportunity = k.get('opportunity_score', 0)
            volume_bonus = min(k.get('search_volume', 0) / 1000, 20)
            scores.append(opportunity + volume_bonus)
        
        return min(100, sum(scores) / len(scores))

    def _calculate_content_potential(self, optimized_keywords: List[Dict[str, Any]]) -> float:
        """Calculate content potential score"""
        if not optimized_keywords:
            return 0
        
        # Based on content suggestions and keyword diversity
        scores = []
        for k in optimized_keywords:
            base_score = k.get('priority_score', 0)
            content_bonus = len(k.get('content_suggestions', [])) * 5
            scores.append(min(100, base_score + content_bonus))
        
        return sum(scores) / len(scores)

    async def _generate_seo_optimized_title(self, optimized_keywords: List[Dict[str, Any]]) -> str:
        """Generate SEO-optimized title"""
        primary_keywords = [k for k in optimized_keywords if k.get('keyword_type') == 'primary']
        if not primary_keywords:
            return "Optimized Content Title"
        
        # Use the highest priority primary keyword
        best_keyword = max(primary_keywords, key=lambda k: k.get('priority_score', 0))
        return best_keyword.get('llm_optimized_title', f"Complete Guide to {best_keyword.get('keyword', '')}")

    async def _generate_seo_optimized_description(self, optimized_keywords: List[Dict[str, Any]]) -> str:
        """Generate SEO-optimized description"""
        primary_keywords = [k for k in optimized_keywords if k.get('keyword_type') == 'primary']
        if not primary_keywords:
            return "Optimized content description"
        
        # Use the highest priority primary keyword
        best_keyword = max(primary_keywords, key=lambda k: k.get('priority_score', 0))
        return best_keyword.get('llm_optimized_description', f"Learn everything about {best_keyword.get('keyword', '')} with expert tips and insights.")

    async def _generate_keyword_content_suggestions(
        self, 
        keyword: str, 
        keyword_data: Dict[str, Any], 
        content_idea: Dict[str, Any]
    ) -> List[str]:
        """Generate content suggestions for a specific keyword"""
        return [
            f"Create detailed section about {keyword}",
            f"Include practical examples of {keyword}",
            f"Add step-by-step guide for {keyword}",
            f"Provide expert tips on {keyword}"
        ]

    async def _generate_heading_suggestions(
        self, 
        keyword: str, 
        keyword_data: Dict[str, Any], 
        content_idea: Dict[str, Any]
    ) -> List[str]:
        """Generate heading suggestions for a specific keyword"""
        return [
            f"What is {keyword}?",
            f"How to Use {keyword}",
            f"Best Practices for {keyword}",
            f"{keyword} Tips and Tricks"
        ]

    async def _generate_internal_link_suggestions(
        self, 
        keyword: str, 
        keyword_data: Dict[str, Any], 
        content_idea: Dict[str, Any]
    ) -> List[str]:
        """Generate internal link suggestions for a specific keyword"""
        return [
            f"Link to related {keyword} topics",
            f"Connect to {keyword} resources",
            f"Reference {keyword} guides"
        ]

    async def _generate_related_questions(
        self, 
        keyword: str, 
        keyword_data: Dict[str, Any], 
        content_idea: Dict[str, Any]
    ) -> List[str]:
        """Generate related questions for a specific keyword"""
        return [
            f"What is {keyword}?",
            f"How does {keyword} work?",
            f"Why is {keyword} important?",
            f"When to use {keyword}?"
        ]

    async def _generate_affiliate_suggestions(
        self, 
        keyword: str, 
        keyword_data: Dict[str, Any], 
        content_idea: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Generate affiliate suggestions for a specific keyword"""
        return {
            'networks': ['Amazon', 'ShareASale', 'CJ Affiliate'],
            'opportunities': [
                f"{keyword} products",
                f"{keyword} tools",
                f"{keyword} services"
            ]
        }

