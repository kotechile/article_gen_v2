"""
AHREFS-based content idea generator with rich analytics and LLM integration

⚠️  DEPRECATED: This service has been replaced by the enhanced AHREFS processing 
    system in minimal_main.py which provides:
    - Better CSV parsing with keyword metrics (volume, KD, CPC, competition, trend)
    - More sophisticated content generation (30+ ideas vs basic templates)
    - Real SEO optimization scores based on AHREFS data
    - Better integration with the existing backend architecture
    
    The new system generates ~10 blog ideas per subtopic + 8 software ideas
    with real AHREFS data integration and superior performance.
    
    Status: ❌ DEPRECATED - Use minimal_main.py instead
"""

import structlog
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import random
import asyncio

# Import LLM functionality
from ..integrations.llm_providers import generate_content, llm_providers_manager
from ..core.llm_provider_config import llm_provider_config

logger = structlog.get_logger()

class AhrefsContentGenerator:
    """
    Generate content ideas using AHREFS keyword data with rich analytics
    """
    
    def __init__(self):
        self.logger = logger
        self.llm_enabled = True
        self.available_providers = []
    
    async def _check_llm_availability(self):
        """Check if LLM providers are available"""
        try:
            available = llm_providers_manager.get_available_providers()
            self.available_providers = available
            self.llm_enabled = len(available) > 0
            
            # Get the default provider from Supabase
            self.default_provider = llm_provider_config.get_default_provider()
            if self.default_provider:
                self.default_provider_type = self.default_provider.get('provider_type')
                self.logger.info("LLM availability check", 
                               available_providers=available,
                               default_provider=self.default_provider_type,
                               llm_enabled=self.llm_enabled)
            else:
                self.default_provider_type = available[0] if available else None
                self.logger.info("LLM availability check", 
                               available_providers=available,
                               using_first_available=self.default_provider_type,
                               llm_enabled=self.llm_enabled)
        except Exception as e:
            self.logger.warning("Failed to check LLM availability", error=str(e))
            self.llm_enabled = False
            self.available_providers = []
            self.default_provider = None
            self.default_provider_type = None
    
    async def _generate_llm_content(
        self, 
        content_type: str, 
        topic_title: str, 
        subtopic: str, 
        keywords: List[Dict[str, Any]], 
        idea_number: int
    ) -> Optional[Dict[str, Any]]:
        """Generate content using LLM"""
        if not self.llm_enabled or not self.available_providers:
            return None
        
        try:
            # Select primary keywords for the prompt
            primary_keywords = [kw.get('keyword', '') for kw in keywords[:3]]
            keyword_data = {
                'keywords': primary_keywords,
                'volumes': [kw.get('volume', 0) for kw in keywords[:3]],
                'difficulties': [kw.get('difficulty', 0) for kw in keywords[:3]],
                'cpcs': [kw.get('cpc', 0) for kw in keywords[:3]]
            }
            
            if content_type == 'blog':
                prompt = self._create_blog_llm_prompt(topic_title, subtopic, keyword_data, idea_number)
            else:
                prompt = self._create_software_llm_prompt(topic_title, subtopic, keyword_data, idea_number)
            
            # Use the default provider from Supabase, or fallback to first available
            provider = self.default_provider_type or (self.available_providers[0] if self.available_providers else 'openai')
            
            # Use provider-specific settings if available
            max_tokens = 500
            temperature = 0.8
            if self.default_provider:
                max_tokens = self.default_provider.get('max_tokens', 500)
                temperature = self.default_provider.get('temperature', 0.8)
            
            self.logger.info(f"Using LLM provider: {provider} (model: {self.default_provider.get('model_name', 'default') if self.default_provider else 'default'})")
            
            response = await generate_content(
                prompt=prompt,
                provider=provider,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if 'error' in response:
                self.logger.warning("LLM generation failed", error=response['error'])
                return None
            
            # Parse the LLM response
            return self._parse_llm_response(response, content_type, subtopic, keywords)
            
        except Exception as e:
            self.logger.warning("LLM generation error", error=str(e))
            return None
    
    def _create_blog_llm_prompt(self, topic_title: str, subtopic: str, keyword_data: Dict, idea_number: int) -> str:
        """Create a prompt for blog idea generation"""
        keywords_str = ', '.join(keyword_data['keywords'])
        volumes_str = ', '.join(map(str, keyword_data['volumes']))
        
        return f"""Generate a creative and SEO-optimized blog post idea for the topic "{topic_title}" focusing on the subtopic "{subtopic}".

Keywords to target: {keywords_str}
Search volumes: {volumes_str}
Idea number: {idea_number}

Requirements:
1. Create a compelling, click-worthy title (max 60 characters)
2. Write a detailed description (2-3 sentences, max 150 characters)
3. Suggest 3-5 target keywords from the provided list
4. Determine the content angle (tutorial, guide, review, news, etc.)
5. Assess the target audience (beginners, professionals, businesses, etc.)
6. Rate monetization potential (low, medium, high)
7. Estimate read time (5-15 minutes)

Format your response as JSON:
{{
    "title": "Your Blog Title Here",
    "description": "Your description here",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "content_angle": "tutorial",
    "target_audience": "beginners",
    "monetization_potential": "medium",
    "estimated_read_time": 8
}}"""
    
    def _create_software_llm_prompt(self, topic_title: str, subtopic: str, keyword_data: Dict, idea_number: int) -> str:
        """Create a prompt for software idea generation"""
        keywords_str = ', '.join(keyword_data['keywords'])
        volumes_str = ', '.join(map(str, keyword_data['volumes']))
        
        return f"""Generate a creative software/SaaS idea for the topic "{topic_title}" focusing on the subtopic "{subtopic}".

Keywords to target: {keywords_str}
Search volumes: {volumes_str}
Idea number: {idea_number}

Requirements:
1. Create a compelling software product name (max 50 characters)
2. Write a detailed description (2-3 sentences, max 150 characters)
3. Suggest 3-5 target keywords from the provided list
4. Determine the software category (SaaS, mobile app, desktop tool, etc.)
5. Assess technical complexity (low, medium, high)
6. Estimate development effort (low, medium, high)
7. Rate market demand (low, medium, high)
8. Suggest monetization model (freemium, subscription, one-time, etc.)

Format your response as JSON:
{{
    "title": "Your Software Name Here",
    "description": "Your description here",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "category": "saas",
    "technical_complexity": "medium",
    "development_effort": "medium",
    "market_demand": "high",
    "monetization_model": "subscription"
}}"""
    
    def _parse_llm_response(self, response: Dict[str, Any], content_type: str, subtopic: str, keywords: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse LLM response and create idea object"""
        try:
            content = response.get('content', '')
            if isinstance(content, list) and len(content) > 0:
                content = content[0].get('text', '')
            
            # Try to extract JSON from the response
            import json
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                idea_data = json.loads(json_match.group())
            else:
                # Fallback: create basic structure
                idea_data = {
                    "title": content[:100] + "..." if len(content) > 100 else content,
                    "description": content,
                    "keywords": [kw.get('keyword', '') for kw in keywords[:3]],
                    "content_angle": "tutorial" if content_type == 'blog' else "solution",
                    "target_audience": "general_public",
                    "monetization_potential": "medium"
                }
            
            # Create the idea object
            primary_keyword = keywords[0] if keywords else {}
            total_volume = sum(kw.get('volume', 0) for kw in keywords)
            avg_difficulty = int(sum(kw.get('difficulty', 0) for kw in keywords) / len(keywords)) if keywords else 0
            avg_cpc = sum(kw.get('cpc', 0) for kw in keywords) / len(keywords) if keywords else 0
            
            idea = {
                'id': str(uuid.uuid4()),
                'title': idea_data.get('title', f"{content_type.title()} Idea"),
                'description': idea_data.get('description', f"Generated {content_type} idea for {subtopic}"),
                'content_type': content_type,
                'category': 'seo_optimized' if content_type == 'blog' else 'software_tool',
                'subtopic': subtopic,
                'keywords': idea_data.get('keywords', [kw.get('keyword', '') for kw in keywords[:3]]),
                'primary_keyword': primary_keyword.get('keyword', ''),
                'secondary_keywords': [kw.get('keyword', '') for kw in keywords[1:3]],
                'seo_score': int(min(95, max(60, 100 - avg_difficulty))),
                'difficulty_level': 'easy' if avg_difficulty < 30 else 'medium' if avg_difficulty < 60 else 'hard',
                'estimated_read_time': idea_data.get('estimated_read_time', random.randint(8, 15)) if content_type == 'blog' else 0,
                'target_audience': idea_data.get('target_audience', 'general_public'),
                'content_angle': idea_data.get('content_angle', 'tutorial'),
                'monetization_potential': idea_data.get('monetization_potential', 'medium'),
                'technical_complexity': idea_data.get('technical_complexity', 'medium') if content_type == 'software' else 'low',
                'development_effort': idea_data.get('development_effort', 'medium') if content_type == 'software' else 'low',
                'market_demand': idea_data.get('market_demand', 'medium'),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'ahrefs_analytics': {
                    'total_volume': total_volume,
                    'avg_difficulty': int(avg_difficulty),
                    'avg_cpc': round(avg_cpc, 2),
                    'keyword_count': len(keywords),
                    'high_volume_keywords': len([kw for kw in keywords if kw.get('volume', 0) > 1000]),
                    'low_difficulty_keywords': len([kw for kw in keywords if kw.get('difficulty', 0) < 30]),
                    'commercial_keywords': len([kw for kw in keywords if any('commercial' in intent.lower() for intent in kw.get('intents', []))]),
                    'content_potential': 'high' if total_volume > 5000 else 'medium' if total_volume > 1000 else 'low',
                    'traffic_estimate': 'high' if avg_difficulty < 40 else 'medium' if avg_difficulty < 60 else 'low',
                    'competition_level': 'low' if avg_difficulty < 40 else 'medium' if avg_difficulty < 60 else 'high'
                }
            }
            
            return idea
            
        except Exception as e:
            self.logger.warning("Failed to parse LLM response", error=str(e))
            return None
    
    async def generate_content_ideas(
        self,
        topic_id: str,
        topic_title: str,
        subtopics: List[str],
        ahrefs_keywords: List[Dict[str, Any]],
        user_id: str,
        db
    ) -> Dict[str, Any]:
        """
        Generate content ideas using AHREFS keyword data with LLM + template generation
        """
        try:
            self.logger.info("Starting AHREFS content generation", 
                           topic_id=topic_id,
                           subtopics_count=len(subtopics),
                           keywords_count=len(ahrefs_keywords))
            
            # Check LLM availability
            await self._check_llm_availability()
            
            # Analyze keywords and group by subtopics
            keyword_analysis = self._analyze_keywords(ahrefs_keywords, subtopics)
            
            # Generate blog ideas with LLM + templates
            blog_ideas = await self._generate_blog_ideas_with_llm_and_templates(
                topic_title, subtopics, keyword_analysis
            )
            
            # Generate software ideas with LLM + templates
            software_ideas = await self._generate_software_ideas_with_llm_and_templates(
                topic_title, subtopics, keyword_analysis
            )
            
            # Save ideas to database
            all_ideas = blog_ideas + software_ideas
            await self._save_ideas_to_db(db, all_ideas, topic_id, user_id)
            
            # Calculate analytics summary
            analytics_summary = self._calculate_analytics_summary(ahrefs_keywords)
            
            self.logger.info("AHREFS content generation completed", 
                           total_ideas=len(all_ideas),
                           blog_ideas=len(blog_ideas),
                           software_ideas=len(software_ideas))
            
            return {
                'success': True,
                'message': f'Successfully generated {len(all_ideas)} content ideas using AHREFS data',
                'total_ideas': len(all_ideas),
                'blog_ideas': len(blog_ideas),
                'software_ideas': len(software_ideas),
                'ideas': all_ideas,
                'analytics_summary': analytics_summary
            }
            
        except Exception as e:
            self.logger.error("AHREFS content generation failed", error=str(e))
            raise
    
    def _analyze_keywords(self, keywords: List[Dict[str, Any]], subtopics: List[str]) -> Dict[str, Any]:
        """
        Analyze keywords and group by subtopics with rich metrics
        """
        analysis = {
            'by_subtopic': {},
            'high_volume': [],
            'low_difficulty': [],
            'commercial': [],
            'informational': [],
            'total_volume': 0,
            'avg_difficulty': 0,
            'avg_cpc': 0
        }
        
        # Initialize subtopic groups
        for subtopic in subtopics:
            analysis['by_subtopic'][subtopic] = {
                'keywords': [],
                'total_volume': 0,
                'avg_difficulty': 0,
                'avg_cpc': 0,
                'high_volume_count': 0,
                'low_difficulty_count': 0
            }
        
        total_volume = 0
        total_difficulty = 0
        total_cpc = 0
        
        for keyword in keywords:
            # Basic metrics
            volume = keyword.get('volume', 0)
            difficulty = keyword.get('difficulty', 0)
            cpc = keyword.get('cpc', 0)
            intents = keyword.get('intents', [])
            
            total_volume += volume
            total_difficulty += difficulty
            total_cpc += cpc
            
            # Categorize by volume and difficulty
            if volume > 1000:
                analysis['high_volume'].append(keyword)
            
            if difficulty < 30:
                analysis['low_difficulty'].append(keyword)
            
            # Categorize by intent
            if any('commercial' in intent.lower() or 'transactional' in intent.lower() for intent in intents):
                analysis['commercial'].append(keyword)
            elif any('informational' in intent.lower() for intent in intents):
                analysis['informational'].append(keyword)
            
            # Match to subtopics
            best_subtopic = self._find_best_subtopic_match(keyword['keyword'], subtopics)
            if best_subtopic:
                analysis['by_subtopic'][best_subtopic]['keywords'].append(keyword)
                analysis['by_subtopic'][best_subtopic]['total_volume'] += volume
                analysis['by_subtopic'][best_subtopic]['high_volume_count'] += (1 if volume > 1000 else 0)
                analysis['by_subtopic'][best_subtopic]['low_difficulty_count'] += (1 if difficulty < 30 else 0)
        
        # Calculate averages
        keyword_count = len(keywords)
        analysis['total_volume'] = total_volume
        analysis['avg_difficulty'] = total_difficulty / keyword_count if keyword_count > 0 else 0
        analysis['avg_cpc'] = total_cpc / keyword_count if keyword_count > 0 else 0
        
        # Calculate subtopic averages
        for subtopic in subtopics:
            subtopic_data = analysis['by_subtopic'][subtopic]
            if subtopic_data['keywords']:
                subtopic_data['avg_difficulty'] = sum(kw.get('difficulty', 0) for kw in subtopic_data['keywords']) / len(subtopic_data['keywords'])
                subtopic_data['avg_cpc'] = sum(kw.get('cpc', 0) for kw in subtopic_data['keywords']) / len(subtopic_data['keywords'])
        
        return analysis
    
    def _find_best_subtopic_match(self, keyword: str, subtopics: List[str]) -> Optional[str]:
        """
        Find the best subtopic match for a keyword
        """
        keyword_lower = keyword.lower()
        
        # Direct matches first
        for subtopic in subtopics:
            if any(word in keyword_lower for word in subtopic.lower().split()):
                return subtopic
        
        # Partial matches
        for subtopic in subtopics:
            subtopic_words = subtopic.lower().split()
            if any(word in keyword_lower for word in subtopic_words if len(word) > 3):
                return subtopic
        
        return None
    
    async def _generate_blog_ideas_with_llm_and_templates(
        self, 
        topic_title: str, 
        subtopics: List[str], 
        keyword_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate blog ideas using LLM + template fallback, ~10 per subtopic
        """
        ideas = []
        
        for subtopic in subtopics:
            subtopic_data = keyword_analysis['by_subtopic'][subtopic]
            keywords = subtopic_data['keywords']
            
            if not keywords:
                continue
            
            # Generate ~10 ideas per subtopic
            num_ideas = min(12, max(8, 10))
            
            self.logger.info(f"Generating {num_ideas} blog ideas for subtopic: {subtopic}")
            
            for i in range(num_ideas):
                # Select relevant keywords for this idea
                selected_keywords = self._select_keywords_for_idea(keywords, i)
                
                if not selected_keywords:
                    continue
                
                # Try LLM generation first (for 60% of ideas)
                idea = None
                should_use_llm = i < int(num_ideas * 0.6) and self.llm_enabled
                
                if should_use_llm:
                    self.logger.info(f"Attempting LLM generation for idea {i+1}/{num_ideas} in subtopic {subtopic}")
                    idea = await self._generate_llm_content(
                        'blog', topic_title, subtopic, selected_keywords, i + 1
                    )
                    if idea:
                        idea['generation_method'] = 'llm'
                        self.logger.info(f"LLM generation successful for idea {i+1}")
                    else:
                        self.logger.warning(f"LLM generation failed for idea {i+1}, falling back to template")
                
                # Fallback to template if LLM failed or for remaining ideas
                if idea is None:
                    idea = self._create_blog_idea_with_analytics(
                        topic_title, subtopic, selected_keywords, i + 1
                    )
                    # Mark as template-generated
                    idea['generation_method'] = 'template'
                    if should_use_llm:
                        self.logger.info(f"Using template fallback for idea {i+1}")
                    else:
                        self.logger.info(f"Using template for idea {i+1} (not in LLM range)")
                
                ideas.append(idea)
        
        return ideas
    
    async def _generate_software_ideas_with_llm_and_templates(
        self, 
        topic_title: str, 
        subtopics: List[str], 
        keyword_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate software ideas using LLM + template fallback, ~10 per subtopic
        """
        ideas = []
        
        for subtopic in subtopics:
            subtopic_data = keyword_analysis['by_subtopic'][subtopic]
            keywords = subtopic_data['keywords']
            
            if not keywords:
                continue
            
            # Generate ~10 software ideas per subtopic
            num_ideas = min(12, max(8, 10))
            
            self.logger.info(f"Generating {num_ideas} software ideas for subtopic: {subtopic}")
            
            for i in range(num_ideas):
                # Select high-value keywords for software ideas
                selected_keywords = self._select_software_keywords(keywords, i)
                
                if not selected_keywords:
                    continue
                
                # Try LLM generation first (for 60% of ideas)
                idea = None
                should_use_llm = i < int(num_ideas * 0.6) and self.llm_enabled
                
                if should_use_llm:
                    self.logger.info(f"Attempting LLM generation for software idea {i+1}/{num_ideas} in subtopic {subtopic}")
                    idea = await self._generate_llm_content(
                        'software', topic_title, subtopic, selected_keywords, i + 1
                    )
                    if idea:
                        idea['generation_method'] = 'llm'
                        self.logger.info(f"LLM generation successful for software idea {i+1}")
                    else:
                        self.logger.warning(f"LLM generation failed for software idea {i+1}, falling back to template")
                
                # Fallback to template if LLM failed or for remaining ideas
                if idea is None:
                    idea = self._create_software_idea_with_analytics(
                        topic_title, subtopic, selected_keywords, i + 1
                    )
                    # Mark as template-generated
                    idea['generation_method'] = 'template'
                    if should_use_llm:
                        self.logger.info(f"Using template fallback for software idea {i+1}")
                    else:
                        self.logger.info(f"Using template for software idea {i+1} (not in LLM range)")
                
                ideas.append(idea)
        
        return ideas
    
    def _select_keywords_for_idea(self, keywords: List[Dict[str, Any]], idea_index: int) -> List[Dict[str, Any]]:
        """
        Select relevant keywords for a blog idea
        """
        if not keywords:
            return []
        
        # Sort by a combination of volume and low difficulty
        scored_keywords = []
        for kw in keywords:
            volume_score = min(kw.get('volume', 0) / 1000, 10)  # Cap at 10
            difficulty_score = max(0, 10 - kw.get('difficulty', 50) / 10)  # Lower difficulty = higher score
            cpc_score = min(kw.get('cpc', 0) / 5, 2)  # Higher CPC = higher score
            
            total_score = volume_score + difficulty_score + cpc_score
            scored_keywords.append((total_score, kw))
        
        # Sort by score and take top keywords
        scored_keywords.sort(key=lambda x: x[0], reverse=True)
        
        # Select 2-4 keywords per idea
        num_keywords = min(4, max(2, len(keywords) // 3))
        selected = [kw for score, kw in scored_keywords[:num_keywords]]
        
        # Add some variety by rotating through different keyword sets
        if idea_index > 0:
            start_idx = (idea_index * 2) % len(keywords)
            selected = keywords[start_idx:start_idx + num_keywords]
        
        return selected
    
    def _select_software_keywords(self, keywords: List[Dict[str, Any]], idea_index: int) -> List[Dict[str, Any]]:
        """
        Select keywords suitable for software ideas (high commercial value)
        """
        if not keywords:
            return []
        
        # Filter for commercial/transactional keywords
        commercial_keywords = [
            kw for kw in keywords 
            if any('commercial' in intent.lower() or 'transactional' in intent.lower() 
                  for intent in kw.get('intents', []))
        ]
        
        if not commercial_keywords:
            commercial_keywords = keywords
        
        # Sort by CPC and volume (higher is better for software)
        scored_keywords = []
        for kw in commercial_keywords:
            cpc_score = kw.get('cpc', 0) * 2
            volume_score = min(kw.get('volume', 0) / 500, 5)
            total_score = cpc_score + volume_score
            scored_keywords.append((total_score, kw))
        
        scored_keywords.sort(key=lambda x: x[0], reverse=True)
        
        # Select 1-3 keywords for software ideas
        num_keywords = min(3, max(1, len(commercial_keywords) // 2))
        selected = [kw for score, kw in scored_keywords[:num_keywords]]
        
        return selected
    
    def _create_blog_idea_with_analytics(
        self, 
        topic_title: str, 
        subtopic: str, 
        keywords: List[Dict[str, Any]], 
        idea_number: int
    ) -> Dict[str, Any]:
        """
        Create a blog idea with rich AHREFS analytics
        """
        primary_keyword = keywords[0] if keywords else {}
        secondary_keywords = keywords[1:] if len(keywords) > 1 else []
        
        # Calculate SEO metrics
        total_volume = sum(kw.get('volume', 0) for kw in keywords)
        avg_difficulty = int(sum(kw.get('difficulty', 0) for kw in keywords) / len(keywords)) if keywords else 0
        avg_cpc = sum(kw.get('cpc', 0) for kw in keywords) / len(keywords) if keywords else 0
        
        # Generate content based on keyword data
        content_templates = [
            f"The Complete Guide to {subtopic}: {primary_keyword.get('keyword', subtopic)}",
            f"How to {primary_keyword.get('keyword', subtopic)}: A Step-by-Step Tutorial",
            f"{primary_keyword.get('keyword', subtopic)}: Everything You Need to Know",
            f"Best Practices for {primary_keyword.get('keyword', subtopic)} in 2025",
            f"Ultimate {subtopic} Guide: {primary_keyword.get('keyword', subtopic)} Edition",
            f"Mastering {primary_keyword.get('keyword', subtopic)}: Expert Tips and Tricks"
        ]
        
        title = content_templates[idea_number % len(content_templates)]
        
        # Generate description based on keywords
        keyword_phrases = [kw.get('keyword', '') for kw in keywords[:3]]
        description = f"Learn everything about {subtopic} with this comprehensive guide. " \
                     f"Covering {', '.join(keyword_phrases)} and more. " \
                     f"Perfect for beginners and experts alike."
        
        # Calculate content potential
        content_potential = self._calculate_content_potential(keywords)
        
        return {
            'id': str(uuid.uuid4()),
            'title': title,
            'description': description,
            'content_type': 'blog',
            'category': 'seo_optimized',
            'subtopic': subtopic,
            'keywords': [kw.get('keyword', '') for kw in keywords],
            'primary_keyword': primary_keyword.get('keyword', ''),
            'secondary_keywords': [kw.get('keyword', '') for kw in secondary_keywords],
            'seo_score': int(min(95, max(60, 100 - avg_difficulty))),
            'difficulty_level': 'easy' if avg_difficulty < 30 else 'medium' if avg_difficulty < 60 else 'hard',
            'estimated_read_time': random.randint(8, 15),
            'target_audience': self._determine_target_audience(keywords),
            'content_angle': self._determine_content_angle(keywords),
            'monetization_potential': 'high' if avg_cpc > 2 else 'medium' if avg_cpc > 0.5 else 'low',
            'technical_complexity': 'low',
            'development_effort': 'low',
            'market_demand': 'high' if total_volume > 5000 else 'medium' if total_volume > 1000 else 'low',
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            # AHREFS-specific analytics
            'ahrefs_analytics': {
                'total_volume': total_volume,
                'avg_difficulty': int(avg_difficulty),
                'avg_cpc': round(avg_cpc, 2),
                'keyword_count': len(keywords),
                'high_volume_keywords': len([kw for kw in keywords if kw.get('volume', 0) > 1000]),
                'low_difficulty_keywords': len([kw for kw in keywords if kw.get('difficulty', 0) < 30]),
                'commercial_keywords': len([kw for kw in keywords if any('commercial' in intent.lower() for intent in kw.get('intents', []))]),
                'content_potential': content_potential,
                'traffic_estimate': self._estimate_traffic_potential(keywords),
                'competition_level': self._assess_competition(keywords)
            }
        }
    
    def _create_software_idea_with_analytics(
        self, 
        topic_title: str, 
        subtopic: str, 
        keywords: List[Dict[str, Any]], 
        idea_number: int
    ) -> Dict[str, Any]:
        """
        Create a software idea with AHREFS analytics
        """
        primary_keyword = keywords[0] if keywords else {}
        
        # Calculate business metrics
        total_volume = sum(kw.get('volume', 0) for kw in keywords)
        avg_cpc = sum(kw.get('cpc', 0) for kw in keywords) / len(keywords) if keywords else 0
        commercial_score = len([kw for kw in keywords if any('commercial' in intent.lower() for intent in kw.get('intents', []))])
        
        # Generate software idea
        software_templates = [
            f"{subtopic} Management Tool: {primary_keyword.get('keyword', 'Solution')}",
            f"AI-Powered {subtopic} Assistant",
            f"Automated {primary_keyword.get('keyword', subtopic)} Platform",
            f"Smart {subtopic} Dashboard",
            f"Mobile {subtopic} App with {primary_keyword.get('keyword', 'Features')}",
            f"Cloud-Based {subtopic} Solution"
        ]
        
        title = software_templates[idea_number % len(software_templates)]
        
        description = f"Revolutionary {subtopic} software solution. " \
                     f"Built for {primary_keyword.get('keyword', 'modern users')} with " \
                     f"advanced features and intuitive design."
        
        return {
            'id': str(uuid.uuid4()),
            'title': title,
            'description': description,
            'content_type': 'software',
            'category': 'software_tool',
            'subtopic': subtopic,
            'keywords': [kw.get('keyword', '') for kw in keywords],
            'primary_keyword': primary_keyword.get('keyword', ''),
            'secondary_keywords': [kw.get('keyword', '') for kw in keywords[1:]],
            'seo_score': 0,  # Not applicable for software
            'difficulty_level': 'medium',
            'estimated_read_time': 0,
            'target_audience': 'businesses' if avg_cpc > 1 else 'individuals',
            'content_angle': 'product_development',
            'monetization_potential': 'high' if avg_cpc > 2 else 'medium',
            'technical_complexity': 'high' if commercial_score > 2 else 'medium',
            'development_effort': 'high' if total_volume > 5000 else 'medium',
            'market_demand': 'high' if total_volume > 3000 else 'medium',
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            # AHREFS-specific analytics
            'ahrefs_analytics': {
                'total_volume': total_volume,
                'avg_cpc': round(avg_cpc, 2),
                'commercial_score': commercial_score,
                'market_size_estimate': self._estimate_market_size(keywords),
                'revenue_potential': self._estimate_revenue_potential(keywords),
                'competition_analysis': self._analyze_software_competition(keywords)
            }
        }
    
    def _calculate_content_potential(self, keywords: List[Dict[str, Any]]) -> str:
        """Calculate content potential based on keyword data"""
        if not keywords:
            return 'low'
        
        high_volume = len([kw for kw in keywords if kw.get('volume', 0) > 1000])
        low_difficulty = len([kw for kw in keywords if kw.get('difficulty', 0) < 30])
        
        if high_volume >= 3 and low_difficulty >= 2:
            return 'very_high'
        elif high_volume >= 2 or low_difficulty >= 2:
            return 'high'
        elif high_volume >= 1 or low_difficulty >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _determine_target_audience(self, keywords: List[Dict[str, Any]]) -> str:
        """Determine target audience based on keyword data"""
        if not keywords:
            return 'general'
        
        avg_cpc = sum(kw.get('cpc', 0) for kw in keywords) / len(keywords)
        commercial_keywords = len([kw for kw in keywords if any('commercial' in intent.lower() for intent in kw.get('intents', []))])
        
        if avg_cpc > 3 and commercial_keywords > 1:
            return 'business_professionals'
        elif avg_cpc > 1:
            return 'affluent_consumers'
        else:
            return 'general_public'
    
    def _determine_content_angle(self, keywords: List[Dict[str, Any]]) -> str:
        """Determine content angle based on keyword data"""
        if not keywords:
            return 'informational'
        
        intents = [intent for kw in keywords for intent in kw.get('intents', [])]
        
        if any('commercial' in intent.lower() for intent in intents):
            return 'commercial_review'
        elif any('transactional' in intent.lower() for intent in intents):
            return 'buying_guide'
        else:
            return 'educational_tutorial'
    
    def _estimate_traffic_potential(self, keywords: List[Dict[str, Any]]) -> str:
        """Estimate traffic potential based on keyword data"""
        if not keywords:
            return 'low'
        
        total_volume = sum(kw.get('volume', 0) for kw in keywords)
        avg_difficulty = sum(kw.get('difficulty', 0) for kw in keywords) / len(keywords)
        
        if total_volume > 10000 and avg_difficulty < 40:
            return 'very_high'
        elif total_volume > 5000 and avg_difficulty < 50:
            return 'high'
        elif total_volume > 1000:
            return 'medium'
        else:
            return 'low'
    
    def _assess_competition(self, keywords: List[Dict[str, Any]]) -> str:
        """Assess competition level based on keyword difficulty"""
        if not keywords:
            return 'unknown'
        
        avg_difficulty = sum(kw.get('difficulty', 0) for kw in keywords) / len(keywords)
        
        if avg_difficulty > 70:
            return 'very_high'
        elif avg_difficulty > 50:
            return 'high'
        elif avg_difficulty > 30:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_market_size(self, keywords: List[Dict[str, Any]]) -> str:
        """Estimate market size for software ideas"""
        if not keywords:
            return 'small'
        
        total_volume = sum(kw.get('volume', 0) for kw in keywords)
        
        if total_volume > 20000:
            return 'very_large'
        elif total_volume > 10000:
            return 'large'
        elif total_volume > 5000:
            return 'medium'
        else:
            return 'small'
    
    def _estimate_revenue_potential(self, keywords: List[Dict[str, Any]]) -> str:
        """Estimate revenue potential for software ideas"""
        if not keywords:
            return 'low'
        
        avg_cpc = sum(kw.get('cpc', 0) for kw in keywords) / len(keywords)
        commercial_keywords = len([kw for kw in keywords if any('commercial' in intent.lower() for intent in kw.get('intents', []))])
        
        if avg_cpc > 5 and commercial_keywords > 2:
            return 'very_high'
        elif avg_cpc > 2 and commercial_keywords > 1:
            return 'high'
        elif avg_cpc > 1:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_software_competition(self, keywords: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze competition for software ideas"""
        if not keywords:
            return {'level': 'unknown', 'opportunities': []}
        
        avg_difficulty = sum(kw.get('difficulty', 0) for kw in keywords) / len(keywords)
        high_difficulty_count = len([kw for kw in keywords if kw.get('difficulty', 0) > 60])
        
        opportunities = []
        if avg_difficulty < 40:
            opportunities.append('Low competition - easy to rank')
        if high_difficulty_count < len(keywords) / 2:
            opportunities.append('Mixed difficulty - some easy wins')
        
        return {
            'level': 'high' if avg_difficulty > 60 else 'medium' if avg_difficulty > 40 else 'low',
            'opportunities': opportunities
        }
    
    def _calculate_analytics_summary(self, keywords: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall analytics summary"""
        if not keywords:
            return {
                'total_volume': 0,
                'avg_difficulty': 0,
                'avg_cpc': 0,
                'high_volume_keywords': 0,
                'low_difficulty_keywords': 0,
                'commercial_keywords': 0
            }
        
        total_volume = sum(kw.get('volume', 0) for kw in keywords)
        avg_difficulty = sum(kw.get('difficulty', 0) for kw in keywords) / len(keywords)
        avg_cpc = sum(kw.get('cpc', 0) for kw in keywords) / len(keywords)
        high_volume_keywords = len([kw for kw in keywords if kw.get('volume', 0) > 1000])
        low_difficulty_keywords = len([kw for kw in keywords if kw.get('difficulty', 0) < 30])
        commercial_keywords = len([kw for kw in keywords if any('commercial' in intent.lower() for intent in kw.get('intents', []))])
        
        return {
            'total_volume': total_volume,
            'avg_difficulty': int(avg_difficulty),
            'avg_cpc': round(avg_cpc, 2),
            'high_volume_keywords': high_volume_keywords,
            'low_difficulty_keywords': low_difficulty_keywords,
            'commercial_keywords': commercial_keywords
        }
    
    async def _save_ideas_to_db(self, db, ideas: List[Dict[str, Any]], topic_id: str, user_id: str):
        """Save generated ideas to the database"""
        try:
            for idea in ideas:
                # Create a copy of the idea with only the fields that exist in the database schema
                db_idea = {
                    'id': idea['id'],
                    'title': idea['title'],
                    'description': idea['description'],
                    'content_type': idea['content_type'],
                    'category': idea['category'],
                    'subtopic': idea['subtopic'],
                    'keywords': idea['keywords'],
                    'seo_score': idea['seo_score'],
                    'difficulty_level': idea['difficulty_level'],
                    'estimated_read_time': idea['estimated_read_time'],
                    'target_audience': idea['target_audience'],
                    'content_angle': idea['content_angle'],
                    'monetization_potential': idea['monetization_potential'],
                    'technical_complexity': idea['technical_complexity'],
                    'development_effort': idea['development_effort'],
                    'market_demand': idea['market_demand'],
                    'created_at': idea['created_at'],
                    'updated_at': idea['updated_at']
                }
                
                # Ensure all integer fields are actually integers
                db_idea['seo_score'] = int(db_idea['seo_score'])
                db_idea['estimated_read_time'] = int(db_idea['estimated_read_time'])
                
                # Prepare the database record with explicit type conversion
                db_record = {
                    'title': str(db_idea['title']),
                    'description': str(db_idea['description']),
                    'content_type': str(db_idea['content_type']),
                    'category': str(db_idea['category']),
                    'subtopic': str(db_idea['subtopic']),
                    'topic_id': str(topic_id),
                    'user_id': str(user_id),
                    'keywords': db_idea['keywords'],
                    'seo_score': int(db_idea['seo_score']),
                    'difficulty_level': str(db_idea['difficulty_level']),
                    'estimated_read_time': int(db_idea['estimated_read_time']),
                    'target_audience': str(db_idea['target_audience']),
                    'content_angle': str(db_idea['content_angle']),
                    'monetization_potential': str(db_idea['monetization_potential']),
                    'technical_complexity': str(db_idea['technical_complexity']),
                    'development_effort': str(db_idea['development_effort']),
                    'market_demand': str(db_idea['market_demand'])
                }
                
                db.client.table('content_ideas').insert(db_record).execute()
            
            self.logger.info("Ideas saved to database successfully", ideas_count=len(ideas))
            
        except Exception as e:
            self.logger.error("Failed to save ideas to database", error=str(e))
            raise
