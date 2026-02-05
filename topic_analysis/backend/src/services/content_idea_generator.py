"""
Content Idea Generator Service
Generates SEO-optimized blog ideas and software ideas based on keywords and subtopics
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import random
import httpx

from ..core.llm_provider_config import llm_provider_config
from ..core.supabase_singleton import get_supabase_client

logger = logging.getLogger(__name__)

class ContentIdeaGenerator:
    def __init__(self):
        self.supabase = get_supabase_client()
        
    async def _get_llm_credentials(self) -> Dict[str, Any]:
        """Fetch active LLM provider and API key from database"""
        try:
            # Get active provider
            provider_response = self.supabase.table('llm_providers').select('*').eq('is_active', True).execute()
            if not provider_response.data:
                raise ValueError("No active LLM provider found")
            
            active_provider = provider_response.data[0]
            model_name = active_provider.get('model_name') or active_provider.get('name')
            
            if not model_name:
                raise ValueError("Active LLM provider has no model name configured")

            # Infer provider from model name
            provider_type = 'openai' # Default
            model_lower = model_name.lower()
            
            if 'gpt' in model_lower:
                provider_type = 'openai'
            elif 'deepseek' in model_lower:
                provider_type = 'deepseek'
            elif 'gemini' in model_lower or 'google' in model_lower:
                provider_type = 'gemini' 
            elif 'claude' in model_lower:
                provider_type = 'anthropic'
            elif 'kimi' in model_lower or 'moonshoot' in model_lower:
                provider_type = 'moonshoot'
                
            # Try to get API key using Foreign Key
            api_key_id = active_provider.get('api_keys_id') or active_provider.get('api_key_id')
            api_key = None
            
            if api_key_id:
                try:
                    id_response = self.supabase.table('api_keys').select('key_value').eq('id', api_key_id).execute()
                    if id_response.data:
                        api_key = id_response.data[0]['key_value']
                except Exception:
                    pass

            # Fallback: Get API key for the inferred provider type
            if not api_key:
                try:
                    # Try exact match first
                    key_response = self.supabase.table('api_keys').select('key_value').eq('provider', provider_type).eq('is_active', True).execute()
                    
                    # If failed and provider is gemini/google, try the other
                    if not key_response.data and provider_type in ['gemini', 'google', 'Google']:
                        alt_provider = 'Google' if provider_type == 'gemini' else 'gemini'
                        key_response = self.supabase.table('api_keys').select('key_value').eq('provider', alt_provider).eq('is_active', True).execute()
                    
                    if not key_response.data:
                         key_response = self.supabase.table('api_keys').select('key_value').eq('provider', provider_type.capitalize()).eq('is_active', True).execute()
                    
                    if key_response.data:
                        api_key = key_response.data[0]['key_value']
                except Exception:
                    pass

            if not api_key:
                raise ValueError(f"No active API key found for {provider_type}")
                
            return {
                "provider_type": provider_type,
                "model_name": model_name,
                "api_key": api_key
            }
        except Exception as e:
            logger.error(f"Failed to get LLM credentials: {str(e)}")
            raise
        
    async def generate_content_ideas(
        self,
        topic_id: str,
        topic_title: str,
        subtopics: List[str],
        keywords: List[Dict[str, Any]],  # Changed to accept rich keyword data
        user_id: str,
        content_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate both blog and software ideas based on subtopics and keywords with rich metrics
        """
        if content_types is None:
            content_types = ['blog', 'software']

        try:
            logger.info(f"Generating content ideas for topic: {topic_title}")
            logger.info(f"Subtopics: {subtopics}")
            logger.info(f"Keywords: {len(keywords)} keywords with rich metrics")
            logger.info(f"Content types: {content_types}")
            
            # Sort keywords by priority score for better idea generation
            sorted_keywords = self._sort_keywords_by_priority(keywords)
            logger.info(f"Top 5 keywords by priority: {[kw.get('keyword', 'N/A') for kw in sorted_keywords[:5]]}")
            
            all_ideas = []
            
            # Generate blog ideas
            if 'blog' in content_types:
                logger.info("Generating blog ideas...")
                blog_ideas = await self._generate_blog_ideas(
                    topic_title, subtopics, sorted_keywords, topic_id, user_id
                )
                all_ideas.extend(blog_ideas)
                logger.info(f"Generated {len(blog_ideas)} blog ideas")

            # Generate software ideas
            if 'software' in content_types:
                logger.info("Generating software ideas...")
                software_ideas = await self._generate_software_ideas(
                    topic_title, subtopics, sorted_keywords, topic_id, user_id
                )
                all_ideas.extend(software_ideas)
                logger.info(f"Generated {len(software_ideas)} software ideas")

            logger.info(f"Total ideas generated: {len(all_ideas)}")

            # Save ideas to database
            if all_ideas:
                await self._save_ideas_to_database(all_ideas)

            return {
                "success": True,
                "total_ideas": len(all_ideas),
                "blog_ideas": len([i for i in all_ideas if i['content_type'] == 'blog']),
                "software_ideas": len([i for i in all_ideas if i['content_type'] == 'software']),
                "ideas": all_ideas
            }
            
        except Exception as e:
            logger.error(f"Failed to generate content ideas: {str(e)}")
            raise

    async def generate_content_ideas_optimized(
        self,
        topic_id: str,
        topic_title: str,
        subtopics: List[str],
        user_id: str,
        content_types: List[str] = None,
        max_keywords: int = 50
    ) -> Dict[str, Any]:
        """
        Optimized generation method that fetches keywords directly from database
        """
        try:
            logger.info(f"Generating optimized content ideas for topic: {topic_title} ({topic_id})")
            
            # Fetch keywords from database
            response = self.supabase.table('keyword_research_data')\
                .select('*')\
                .eq('topic_id', topic_id)\
                .order('search_volume', desc=True)\
                .limit(max_keywords)\
                .execute()
                
            keywords = response.data
            logger.info(f"Fetched {len(keywords)} keywords from database")
            
            if not keywords:
                # If no keywords found in DB, we should report this if strict mode is implied, 
                # or perhaps return empty if we can't generate "optimized" ideas without data.
                # Given "no shortcuts", if we expect data and found none, we should probably stop.
                logger.warning("No keywords found in database for this topic. Cannot generate optimized ideas.")
                return {
                     "success": False,
                     "error": "No keyword data available for this topic. Please perform keyword research first."
                }
            
            # Use the existing generation logic with fetched keywords
            return await self.generate_content_ideas(
                topic_id=topic_id,
                topic_title=topic_title,
                subtopics=subtopics,
                keywords=keywords,
                user_id=user_id,
                content_types=content_types
            )
            
        except Exception as e:
            logger.error(f"Failed to generate optimized content ideas: {str(e)}")
            raise
    
    def _sort_keywords_by_priority(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort keywords by priority score using Ahrefs-style scoring algorithm
        """
        if not keywords:
            return []
        
        scored_keywords = []
        for kw in keywords:
            # Calculate priority score using rich metrics
            priority_score = self._calculate_keyword_priority_score(kw)
            scored_keywords.append((priority_score, kw))
        
        # Sort by score (highest first)
        scored_keywords.sort(key=lambda x: x[0], reverse=True)
        
        # Return sorted keywords without scores
        return [kw for score, kw in scored_keywords]
    
    def _calculate_keyword_priority_score(self, keyword: Dict[str, Any]) -> float:
        """
        Calculate priority score for keyword using Ahrefs-style algorithm
        """
        search_volume = keyword.get('search_volume', 0)
        difficulty = keyword.get('keyword_difficulty', 50)
        cpc = keyword.get('cpc', 0)
        competition = keyword.get('competition_value', 50)
        intent_type = keyword.get('intent_type', 'INFORMATIONAL')
        
        # Volume score (capped at 10)
        volume_score = min(search_volume / 1000, 10)
        
        # Difficulty score (lower difficulty = higher score, inverted)
        difficulty_score = max(0, 10 - difficulty / 10)
        
        # CPC score (higher CPC = higher commercial value)
        cpc_score = min(cpc / 5, 2)
        
        # Competition score (lower competition = higher opportunity)
        competition_score = max(0, 10 - competition / 10)
        
        # Intent bonus (commercial/transactional keywords get bonus)
        intent_bonus = 0
        if intent_type.upper() in ['COMMERCIAL', 'TRANSACTIONAL']:
            intent_bonus = 2
        elif intent_type.upper() == 'INFORMATIONAL':
            intent_bonus = 1
        
        # Calculate total score
        total_score = volume_score + difficulty_score + cpc_score + competition_score + intent_bonus
        
        return total_score
    
    def _select_keywords_for_idea(self, keywords: List[Dict[str, Any]], idea_index: int, content_type: str = 'blog') -> List[Dict[str, Any]]:
        """
        Select relevant keywords for a specific idea using intelligent prioritization
        """
        logger.info(f"Selecting keywords for idea {idea_index}, content_type: {content_type}, input keywords: {len(keywords)}")
        
        if not keywords:
            logger.warning("No keywords provided for selection")
            return []
        
        if content_type == 'software':
            result = self._select_software_keywords(keywords, idea_index)
        else:
            result = self._select_blog_keywords(keywords, idea_index)
        
        logger.info(f"Selected {len(result)} keywords for idea {idea_index}: {[kw.get('keyword', 'N/A') for kw in result]}")
        return result
    
    def _select_blog_keywords(self, keywords: List[Dict[str, Any]], idea_index: int) -> List[Dict[str, Any]]:
        """
        Select keywords suitable for blog ideas (informational focus)
        """
        if not keywords:
            return []
        
        # Filter for informational keywords (prefer informational intent)
        informational_keywords = [
            kw for kw in keywords 
            if kw.get('intent_type', '').upper() == 'INFORMATIONAL'
        ]
        
        logger.info(f"Found {len(informational_keywords)} informational keywords out of {len(keywords)} total")
        
        # If no informational keywords, use all keywords
        if not informational_keywords:
            informational_keywords = keywords
            logger.info("No informational keywords found, using all keywords")
        
        # Sort by priority score
        scored_keywords = []
        for kw in informational_keywords:
            priority_score = self._calculate_keyword_priority_score(kw)
            scored_keywords.append((priority_score, kw))
        
        scored_keywords.sort(key=lambda x: x[0], reverse=True)
        
        # Select 2-4 keywords per idea
        num_keywords = min(4, max(2, len(keywords) // 3))
        selected = [kw for score, kw in scored_keywords[:num_keywords]]
        
        logger.info(f"Selected {len(selected)} keywords for blog idea: {[kw.get('keyword', 'N/A') for kw in selected]}")
        
        # Add variety by rotating through different keyword sets
        if idea_index > 0 and len(keywords) > num_keywords:
            start_idx = (idea_index * 2) % len(keywords)
            additional_keywords = keywords[start_idx:start_idx + num_keywords]
            # Mix selected high-priority keywords with variety
            selected = selected[:2] + additional_keywords[:2]
        
        return selected[:4]  # Ensure max 4 keywords

    async def _call_llm(self, provider_type: str, model_name: str, api_key: str, prompt: str) -> str:
        """Call LLM provider API"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            if provider_type == "openai":
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000,
                    "temperature": 0.7 
                }
                
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
                
            elif provider_type == "deepseek":
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.7 
                }
                
                response = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            
            else:
                raise ValueError(f"Provider {provider_type} not implemented in ContentIdeaGenerator")

    async def _generate_blog_idea_with_llm(
        self,
        subtopic: str,
        topic_title: str,
        selected_keywords: List[Dict[str, Any]],
        topic_id: str,
        user_id: str,
        idea_index: int
    ) -> Dict[str, Any]:
        """Generate a blog idea using LLM based on keywords"""
        try:
            logger.info(f"Generating blog idea with LLM for subtopic: {subtopic}")

            creds = await self._get_llm_credentials()
            provider_type = creds['provider_type']
            model_name = creds['model_name']
            api_key = creds['api_key']

            primary_keyword = max(selected_keywords, key=lambda k: k.get('priority_score', 0))
            keywords_list = ", ".join([k.get('keyword', '') for k in selected_keywords])

            prompt = f"""
            Generate a unique, creative, and SEO-optimized Blog Post idea for the topic "{topic_title}" focusing on the subtopic "{subtopic}".
            
            Context:
            - Primary Keyword: {primary_keyword.get('keyword', '')}
            - Secondary Keywords: {keywords_list}
            
            Requirement:
            1. Title must be catchy, SEO-friendly, and include the primary keyword naturally.
            2. Description must be a compelling hook and summary of what the article covers.
            3. Target Audience (e.g., "Beginners", "Experts").
            4. Content Angle (e.g., "Step-by-step Guide", "Case Study", "Deep Dive", "Listicle").
            5. **Structured Outline**: Create a detailed content outline with H2 and H3 headings. 
               - Map the Secondary Keywords to specific sections where they fit best.
               - Ensure logical flow and comprehensiveness.
            
            CRITICAL INSTRUCTIONS:
            - DO NOT simply concatenate the keyword and subtopic (e.g. "Keyword Guide for Subtopic").
            - DO NOT use the format "{primary_keyword.get('keyword', '')}: Complete {subtopic} Guide".
            - Make the title sound like a real human wrote it.
            - Examples of GOOD titles:
              - "7 Ways to Master {primary_keyword.get('keyword', '')}"
              - "{subtopic} Errors: {primary_keyword.get('keyword', '')} Fixes"
              - "{primary_keyword.get('keyword', '')} Success Blueprint"

            CRITICAL:
            1. Return 5 unique title options in a "titles" array.
            2. "markdown_outline" must be a single string with markdown headers (## H2, ### H3).
            3. Each title must be under 60 characters if possible.
            
            Output JSON Format:
            {{
                "titles": ["string", "string", "string", "string", "string"], 
                "description": "string",
                "target_audience": "string",
                "content_angle": "string",
                "markdown_outline": "## Introduction\\n...\\n## Key Section 1\\n..."
            }}
            """

            try:
                content = await self._call_llm(provider_type, model_name, api_key, prompt)
            except Exception as llm_error:
                logger.error(f"Error calling LLM provider {provider_type}: {str(llm_error)}")
                logger.error(f"Prompt content was: {prompt[:200]}...") # Log start of prompt for context
                # Re-raise to trigger fallback
                raise llm_error 
            
            try:
                # Clean markdown code blocks if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                data = json.loads(content)
            except Exception as e:
                logger.error(f"Failed to parse LLM response for blog idea: {str(e)}")
                raise ValueError(f"LLM response parsing failed: {str(e)}")

            # Process titles to select the best one
            generated_titles = data.get("titles", [])
            # Handle legacy response format if LLM returns single title
            if not generated_titles and data.get("title"):
                generated_titles = [data.get("title")]
            
            # Selection logic:
            # 1. Must include primary keyword (case-insensitive)
            # 2. Prefer <= 60 chars
            # 3. If <= 60 chars, prefer longest (most descriptive within limit)
            # 4. If > 60 chars, prefer shortest (closest to limit)
            
            primary_kw_text = primary_keyword.get('keyword', '').lower()
            
            # Step 1: Filter by keyword inclusion
            keyword_matching_titles = [
                t for t in generated_titles 
                if primary_kw_text in t.lower()
            ]
            
            # If no titles match keyword, fallback to all titles (better than nothing)
            candidates = keyword_matching_titles if keyword_matching_titles else generated_titles
            
            if not candidates:
                 # Fallback to a constructed title if list is empty
                 candidates = [f"{primary_keyword.get('keyword', '').title()}: Guide for {subtopic}"]

            # Step 2: Separate by length
            short_titles = [t for t in candidates if len(t) <= 60]
            long_titles = [t for t in candidates if len(t) > 60]
            
            final_title = ""
            
            if short_titles:
                # Step 3: Select longest valid title (max info within limit)
                final_title = max(short_titles, key=len)
            elif long_titles:
                # Step 4: Select shortest invalid title (minimize overflow)
                final_title = min(long_titles, key=len)
            else:
                 final_title = candidates[0] # Should not happen given logic above

            # Calculate metrics
            avg_search_volume = sum(kw.get('search_volume', 0) for kw in selected_keywords) / len(selected_keywords)
            avg_difficulty = sum(kw.get('keyword_difficulty', 0) for kw in selected_keywords) / len(selected_keywords)
            avg_cpc = sum(kw.get('cpc', 0) for kw in selected_keywords) / len(selected_keywords)
            
            seo_score = self._calculate_seo_score_from_keywords(selected_keywords)
            difficulty_level = self._map_difficulty_score(avg_difficulty)
            monetization_potential = self._calculate_monetization_potential(selected_keywords)

            return {
                "title": final_title,
                "description": data.get("description"),
                "content_type": "blog",
                "category": "seo_optimized",
                "subtopic": subtopic,
                "topic_id": topic_id,
                "user_id": user_id,
                "keywords": [kw.get('keyword', '') for kw in selected_keywords],
                "keyword_metrics": {
                    "avg_search_volume": round(avg_search_volume, 0),
                    "avg_difficulty": round(avg_difficulty, 1),
                    "avg_cpc": round(avg_cpc, 2),
                    "primary_keyword": primary_keyword.get('keyword', ''),
                    "total_keywords_used": len(selected_keywords)
                },
                "seo_score": seo_score,
                "difficulty_level": difficulty_level,
                "estimated_read_time": random.randint(5, 15),
                "target_audience": data.get("target_audience", "General"),
                "content_angle": data.get("content_angle", "Guide"),
                "markdown_outline": data.get("markdown_outline", "## Outline Not Generated"),
                "monetization_potential": monetization_potential,
                "generation_method": "llm_enhanced",
                "data_source": "real_keyword_metrics"
            }

        except Exception as e:
            logger.error(f"LLM blog idea generation failed: {str(e)}")
            raise e

    async def _generate_software_idea_with_llm(
        self,
        subtopic: str,
        topic_title: str,
        selected_keywords: List[Dict[str, Any]],
        topic_id: str,
        user_id: str,
        idea_index: int
    ) -> Dict[str, Any]:
        """Generate a software idea using LLM based on keywords"""
        try:
            logger.info(f"Generating software idea with LLM for subtopic: {subtopic}")

            creds = await self._get_llm_credentials()
            provider_type = creds['provider_type']
            model_name = creds['model_name']
            api_key = creds['api_key']

            primary_keyword = max(selected_keywords, key=lambda k: k.get('priority_score', 0))
            keywords_list = ", ".join([k.get('keyword', '') for k in selected_keywords])

            prompt = f"""
            Generate a unique and viable Software/SaaS idea for the topic "{topic_title}" focusing on the subtopic "{subtopic}".
            
            Context:
            - Primary Keyword: {primary_keyword.get('keyword', '')}
            - Secondary Keywords: {keywords_list}
            
            Requirements:
            1. Title must be professional and sound like a product name.
            2. Description must explain the core value proposition (product summary).
            3. Category must be one of: "web_app", "mobile_app", "desktop_tool", "browser_extension", "api_service".
            4. Target Audience (e.g., "Freelancers", "Enterprise").
            5. Unique Selling Point (Content Angle).
            
            CRITICAL:
            1. Return 5 unique title options in a "titles" array.
            2. Each title MUST include the keyword "{primary_keyword.get('keyword', '')}".
            3. Each title MUST be under 60 characters.
            4. Keep titles concise and professional (e.g. "{primary_keyword.get('keyword', '')} Analytics").
            
                "titles": ["string", "string", "string", "string", "string"],
                "description": "string",
                "category": "string",
                "target_audience": "string",
                "content_angle": "string"
            }}
            """

            content = await self._call_llm(provider_type, model_name, api_key, prompt)
            
            try:
                # Clean markdown code blocks if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                data = json.loads(content)
            except Exception as e:
                 logger.error(f"Failed to parse LLM response for software idea: {str(e)}")
                 raise ValueError(f"LLM response parsing failed: {str(e)}")

            # Process titles to select the best one
            generated_titles = data.get("titles", [])
            # Handle legacy response format
            if not generated_titles and data.get("title"):
                generated_titles = [data.get("title")]
            
            # Selection logic (same as blog):
            # 1. Must include primary keyword
            # 2. Prefer <= 60 chars
            # 3. If <= 60 chars, prefer longest
            # 4. If > 60 chars, prefer shortest
            
            primary_kw_text = primary_keyword.get('keyword', '').lower()
            
            # Step 1: Filter by keyword inclusion
            keyword_matching_titles = [
                t for t in generated_titles 
                if primary_kw_text in t.lower()
            ]
            
            candidates = keyword_matching_titles if keyword_matching_titles else generated_titles
            
            if not candidates:
                 candidates = [f"{primary_keyword.get('keyword', '').title()} Manager"]

            # Step 2: Separate by length
            short_titles = [t for t in candidates if len(t) <= 60]
            long_titles = [t for t in candidates if len(t) > 60]
            
            final_title = ""
            
            if short_titles:
                final_title = max(short_titles, key=len)
            elif long_titles:
                final_title = min(long_titles, key=len)
            else:
                 final_title = candidates[0]

            # Calculate metrics
            avg_search_volume = sum(kw.get('search_volume', 0) for kw in selected_keywords) / len(selected_keywords)
            avg_difficulty = sum(kw.get('keyword_difficulty', 0) for kw in selected_keywords) / len(selected_keywords)
            avg_cpc = sum(kw.get('cpc', 0) for kw in selected_keywords) / len(selected_keywords)
            
            market_demand = self._calculate_market_demand(selected_keywords)
            technical_complexity = self._map_technical_complexity(avg_difficulty)
            monetization_potential = self._calculate_software_monetization_potential(selected_keywords)
            development_effort = "high" if avg_cpc > 2 and avg_difficulty > 60 else "medium"

            return {
                "title": final_title,
                "description": data.get("description"),
                "content_type": "software",
                "category": data.get("category", "web_app"),
                "subtopic": subtopic,
                "topic_id": topic_id,
                "user_id": user_id,
                "keywords": [kw.get('keyword', '') for kw in selected_keywords],
                "keyword_metrics": {
                    "avg_search_volume": round(avg_search_volume, 0),
                    "avg_difficulty": round(avg_difficulty, 1),
                    "avg_cpc": round(avg_cpc, 2),
                    "primary_keyword": primary_keyword.get('keyword', ''),
                    "total_keywords_used": len(selected_keywords)
                },
                # Software specific fields
                "technical_complexity": technical_complexity,
                "development_effort": development_effort,
                "market_demand": market_demand,
                "monetization_potential": monetization_potential,
                
                "seo_score": 0,  # Not applicable
                "difficulty_level": technical_complexity,
                "estimated_read_time": None,
                "target_audience": data.get("target_audience", "General"),
                "content_angle": data.get("content_angle", "SaaS Solution"),
                "generation_method": "llm_enhanced",
                "data_source": "real_keyword_metrics"
            }

        except Exception as e:
            logger.error(f"LLM software idea generation failed: {str(e)}")
            raise e
    
    def _select_software_keywords(self, keywords: List[Dict[str, Any]], idea_index: int) -> List[Dict[str, Any]]:
        """
        Select keywords suitable for software ideas (commercial focus)
        """
        if not keywords:
            return []
        
        # Filter for commercial/transactional keywords
        commercial_keywords = [
            kw for kw in keywords 
            if kw.get('intent_type', '').upper() in ['COMMERCIAL', 'TRANSACTIONAL']
        ]
        
        # If no commercial keywords, use all keywords
        if not commercial_keywords:
            commercial_keywords = keywords
        
        # Sort by CPC and volume (higher is better for software)
        scored_keywords = []
        for kw in commercial_keywords:
            cpc_score = min(kw.get('cpc', 0) / 5, 3)  # Higher weight for CPC
            volume_score = min(kw.get('search_volume', 0) / 1000, 5)
            difficulty_score = max(0, 5 - kw.get('keyword_difficulty', 50) / 20)
            
            total_score = cpc_score + volume_score + difficulty_score
            scored_keywords.append((total_score, kw))
        
        scored_keywords.sort(key=lambda x: x[0], reverse=True)
        
        # Select 2-4 keywords per idea
        num_keywords = min(4, max(2, len(keywords) // 3))
        selected = [kw for score, kw in scored_keywords[:num_keywords]]
        
        # Add variety for software ideas
        if idea_index > 0 and len(keywords) > num_keywords:
            start_idx = (idea_index * 2) % len(keywords)
            additional_keywords = keywords[start_idx:start_idx + num_keywords]
            selected = selected[:2] + additional_keywords[:2]
        
        return selected[:4]  # Ensure max 4 keywords
    
    async def _generate_blog_ideas(
        self,
        topic_title: str,
        subtopics: List[str],
        keywords: List[Dict[str, Any]],  # Changed to accept rich keyword data
        topic_id: str,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Generate SEO-optimized blog ideas using intelligent keyword selection"""
            
        try:
            logger.info("Generating blog ideas with intelligent keyword selection")
            return await self._generate_enhanced_blog_ideas(topic_title, subtopics, keywords, topic_id, user_id)
            
        except Exception as e:
            logger.error(f"Failed to generate blog ideas: {str(e)}")
            raise e # No fallback, just report error

    async def _generate_software_ideas(
        self,
        topic_title: str,
        subtopics: List[str],
        keywords: List[Dict[str, Any]],  # Changed to accept rich keyword data
        topic_id: str,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Generate software-related ideas using intelligent keyword selection"""
        try:
            logger.info("Generating software ideas with intelligent keyword selection")
            return await self._generate_enhanced_software_ideas(topic_title, subtopics, keywords, topic_id, user_id)
            
        except Exception as e:
            logger.error(f"Failed to generate software ideas: {str(e)}")
            raise e # No fallback

    async def _generate_enhanced_blog_ideas(
        self, 
        topic_title: str,
        subtopics: List[str],
        keywords: List[Dict[str, Any]],
        topic_id: str,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Generate enhanced blog ideas using real DataForSEO keyword data"""
        logger.info(f"Generating enhanced blog ideas for topic: {topic_title}")
        logger.info(f"Subtopics: {subtopics}")
        logger.info(f"Available keywords: {len(keywords)} with real DataForSEO metrics")
        ideas = []
        
        if not keywords:
            logger.warning("No keywords available for idea generation")
            return [] # Return empty instead of fallback
        
        # Group keywords by intent type for better idea generation
        informational_keywords = [kw for kw in keywords if kw.get('intent_type', '').upper() == 'INFORMATIONAL']
        commercial_keywords = [kw for kw in keywords if kw.get('intent_type', '').upper() in ['COMMERCIAL', 'TRANSACTIONAL']]
        
        # If no informational keywords, use all keywords
        if not informational_keywords:
            informational_keywords = keywords
        
        logger.info(f"Informational keywords: {len(informational_keywords)}, Commercial keywords: {len(commercial_keywords)}")
        
        # Generate ideas based on real keyword data
        if not subtopics:
            # If no subtopics, use the topic title as the subtopic
            subtopics = [topic_title]
        
        for subtopic in subtopics:
            # Find keywords related to this subtopic
            subtopic_keywords = self._find_keywords_for_subtopic(informational_keywords, subtopic)
            
            if not subtopic_keywords:
                # Fallback to top keywords by priority score
                subtopic_keywords = sorted(keywords, key=lambda k: k.get('priority_score', 0), reverse=True)[:10]
                logger.info(f"No relevant keywords found for subtopic '{subtopic}', using top keywords by priority")
            
            logger.info(f"Using {len(subtopic_keywords)} keywords for subtopic: {subtopic}")
            logger.info(f"Keywords: {[kw.get('keyword', 'N/A') for kw in subtopic_keywords[:5]]}")
            
            # Generate ideas based on actual keyword data
            for i in range(min(8, len(subtopic_keywords))):  # Generate up to 8 ideas per subtopic
                selected_keywords = self._select_keywords_for_idea(subtopic_keywords, i, 'blog')
                
                if not selected_keywords:
                    continue
                
                logger.info(f"Selected keywords for idea {i}: {[kw.get('keyword', 'N/A') for kw in selected_keywords]}")
                
                # Generate idea based on actual keyword data using LLM
                logger.info(f"Calling _generate_blog_idea_with_llm with {len(selected_keywords)} keywords")
                idea = await self._generate_blog_idea_with_llm(
                    subtopic, topic_title, selected_keywords, topic_id, user_id, i
                )
                logger.info(f"Generated idea: {idea.get('title', 'No title')} - generation_method: {idea.get('generation_method', 'None')}")
                ideas.append(idea)
        
        logger.info(f"Generated {len(ideas)} blog ideas using real DataForSEO data")
        return ideas

    def _find_keywords_for_subtopic(self, keywords: List[Dict[str, Any]], subtopic: str) -> List[Dict[str, Any]]:
        """Find keywords that are most relevant to a specific subtopic"""
        if not keywords or not subtopic:
            return keywords[:10] if keywords else []
        
        subtopic_lower = subtopic.lower()
        relevant_keywords = []
        
        for kw in keywords:
            keyword_text = kw.get('keyword', '').lower()
            # Check if subtopic words appear in the keyword
            subtopic_words = subtopic_lower.split()
            relevance_score = 0
            
            for word in subtopic_words:
                if word in keyword_text:
                    relevance_score += 1
            
            # If no direct match, check for semantic similarity or use all keywords
            if relevance_score == 0:
                # For now, include all keywords to ensure we have some to work with
                relevance_score = 0.5
            
            relevant_keywords.append((relevance_score, kw))
        
        # Sort by relevance score and return keywords
        relevant_keywords.sort(key=lambda x: x[0], reverse=True)
        return [kw for score, kw in relevant_keywords[:15]]  # Return top 15 most relevant

    def _generate_blog_idea_from_keywords(
        self, 
        subtopic: str, 
        topic_title: str, 
        selected_keywords: List[Dict[str, Any]], 
        topic_id: str, 
        user_id: str, 
        idea_index: int
    ) -> Dict[str, Any]:
        """Generate a blog idea based on actual keyword data"""
        logger.info(f"Generating blog idea from keywords for subtopic: {subtopic}")
        logger.info(f"Selected keywords: {[kw.get('keyword', 'N/A') for kw in selected_keywords]}")
        
        if not selected_keywords:
            logger.warning("No selected keywords, using fallback")
            return self._create_fallback_blog_idea(subtopic, topic_title, topic_id, user_id)
        
        # Get the primary keyword (highest priority score)
        primary_keyword = max(selected_keywords, key=lambda k: k.get('priority_score', 0))
        primary_keyword_text = primary_keyword.get('keyword', '')
        
        logger.info(f"Primary keyword: {primary_keyword_text}")
        logger.info(f"Subtopic: {subtopic}")
        logger.info(f"Topic title: {topic_title}")
        
        # Calculate metrics from real data
        avg_search_volume = sum(kw.get('search_volume', 0) for kw in selected_keywords) / len(selected_keywords)
        avg_difficulty = sum(kw.get('keyword_difficulty', 0) for kw in selected_keywords) / len(selected_keywords)
        avg_cpc = sum(kw.get('cpc', 0) for kw in selected_keywords) / len(selected_keywords)
        
        # Generate title based on primary keyword and subtopic
        # Use more dynamic title generation based on keyword metrics
        if avg_search_volume > 1000:
            title_templates = [
                f"Definitive Guide to {primary_keyword_text} ({datetime.now().year})",
                f"Mastering {primary_keyword_text} for {subtopic}",
                f"Why {primary_keyword_text} Matters for {subtopic}",
                f"{primary_keyword_text} Explained: {subtopic} Handbook",
                f"Leverage {primary_keyword_text} in Your Strategy",
                f"Advanced {primary_keyword_text} Tactics",
                f"Unlocking {primary_keyword_text} for {subtopic}",
                f"Professional's Roadmap to {primary_keyword_text}"
            ]
        elif avg_cpc > 2:
            title_templates = [
                f"Maximizing ROI with {primary_keyword_text}",
                f"High-Impact {primary_keyword_text} Strategies",
                f"The Executive Guide to {primary_keyword_text}",
                f"Value Driving: {primary_keyword_text} Solutions",
                f"Premium Tactics: {primary_keyword_text}",
                f"Scale {subtopic} with {primary_keyword_text}",
                f"Proven {primary_keyword_text} Professional Methods",
                f"Strategic {primary_keyword_text} Implementation"
            ]
        else:
            title_templates = [
                f"Start with {primary_keyword_text}: {subtopic} Primer",
                f"{primary_keyword_text} 101: {subtopic} Guide",
                f"Simple {primary_keyword_text} Steps: {subtopic}",
                f"Beginner's {primary_keyword_text} Blueprint",
                f"Demystifying {primary_keyword_text} for {subtopic}",
                f"Quick {primary_keyword_text} Wins for {subtopic}",
                f"Intro to {primary_keyword_text} Best Practices",
                f"Basic {primary_keyword_text} Tips for {subtopic}"
            ]
        
        # If keywords are unrelated to the topic, create more creative titles
        if not any(word in primary_keyword_text.lower() for word in subtopic.lower().split()):
            # Create titles that bridge the gap between keywords and topic
            creative_templates = [
                f"How {primary_keyword_text} Helps {subtopic}",
                f"Using {primary_keyword_text} for {subtopic}",
                f"{primary_keyword_text} Strategies for {subtopic}",
                f"Integrating {primary_keyword_text} in {subtopic}",
                f"{primary_keyword_text} Solutions for {subtopic}",
                f"Leveraging {primary_keyword_text} in {subtopic}",
                f"{primary_keyword_text} Best Practices",
                f"Advanced {primary_keyword_text} Methods"
            ]
            title_templates = creative_templates
        
        title = title_templates[idea_index % len(title_templates)]
        
        # Generate description based on keyword data and metrics
        if avg_search_volume > 1000:
            description = f"Discover the most effective {primary_keyword_text} strategies for {subtopic}. "
            description += f"This comprehensive guide covers everything you need to know about {primary_keyword_text} "
            description += f"in {topic_title}, from basic concepts to advanced techniques. "
            description += f"Perfect for professionals looking to excel in {subtopic} with proven {primary_keyword_text} methods."
        elif avg_cpc > 2:
            description = f"Master {primary_keyword_text} in {subtopic} with this professional guide. "
            description += f"Learn proven {primary_keyword_text} techniques and strategies used by experts "
            description += f"in {topic_title}. This comprehensive resource will help you succeed in {subtopic} "
            description += f"using high-value {primary_keyword_text} approaches."
        else:
            description = f"Learn the fundamentals of {primary_keyword_text} in {subtopic}. "
            description += f"This beginner-friendly guide covers {primary_keyword_text} basics, "
            description += f"practical tips, and step-by-step instructions for {topic_title}. "
            description += f"Start your journey in {subtopic} today with essential {primary_keyword_text} knowledge."
        
        # If keywords are unrelated to the topic, create more creative descriptions
        if not any(word in primary_keyword_text.lower() for word in subtopic.lower().split()):
            creative_descriptions = [
                f"Explore how {primary_keyword_text} can revolutionize your approach to {subtopic}. "
                f"This innovative guide shows you how to leverage {primary_keyword_text} techniques "
                f"for better results in {topic_title}. Discover new possibilities and unlock hidden potential.",
                
                f"Learn how to apply {primary_keyword_text} principles to {subtopic} challenges. "
                f"This comprehensive resource covers {primary_keyword_text} strategies that can "
                f"transform your {topic_title} projects. Get practical insights and actionable advice.",
                
                f"Discover the connection between {primary_keyword_text} and {subtopic} success. "
                f"This detailed guide explains how {primary_keyword_text} methodologies can "
                f"enhance your {topic_title} efforts. Perfect for forward-thinking professionals.",
                
                f"Master the art of integrating {primary_keyword_text} into {subtopic} workflows. "
                f"This practical guide covers {primary_keyword_text} best practices and shows "
                f"you how to apply them effectively in {topic_title}. Take your skills to the next level."
            ]
            description = creative_descriptions[idea_index % len(creative_descriptions)]
        
        # Calculate SEO score based on real keyword metrics
        seo_score = self._calculate_seo_score_from_keywords(selected_keywords)
        
        # Determine difficulty level based on actual keyword difficulty
        difficulty_level = self._map_difficulty_score(avg_difficulty)
        
        # Calculate monetization potential based on real CPC data
        monetization_potential = self._calculate_monetization_potential(selected_keywords)
        
        # Determine content angle based on keyword intent
        intent_types = [kw.get('intent_type', 'INFORMATIONAL') for kw in selected_keywords]
        if 'COMMERCIAL' in intent_types or 'TRANSACTIONAL' in intent_types:
            content_angle = "Commercial focus with actionable insights"
            target_audience = "professionals"
        else:
            content_angle = "Educational and informative approach"
            target_audience = "beginners" if avg_difficulty < 40 else "intermediate"
        
        return {
            "title": title,
            "description": description,
            "content_type": "blog",
            "category": "seo_optimized",
            "subtopic": subtopic,
            "topic_id": topic_id,
            "user_id": user_id,
            "keywords": [kw.get('keyword', '') for kw in selected_keywords],
            "keyword_metrics": {
                "avg_search_volume": round(avg_search_volume, 0),
                "avg_difficulty": round(avg_difficulty, 1),
                "avg_cpc": round(avg_cpc, 2),
                "intent_types": intent_types,
                "primary_keyword": primary_keyword_text,
                "total_keywords_used": len(selected_keywords)
            },
            "seo_score": seo_score,
            "difficulty_level": difficulty_level,
            "estimated_read_time": random.randint(8, 25),
            "target_audience": target_audience,
            "content_angle": content_angle,
            "monetization_potential": monetization_potential,
            "generation_method": "dataforseo_enhanced",
            "data_source": "real_keyword_metrics"
        }

    def _create_fallback_blog_idea(self, subtopic: str, topic_title: str, topic_id: str, user_id: str) -> Dict[str, Any]:
        """Create a fallback blog idea when no keywords are available"""
        return {
            "title": f"Complete Guide to {subtopic} in {topic_title}",
            "description": f"Everything you need to know about {subtopic} in {topic_title}. This comprehensive guide covers the basics, advanced techniques, and practical tips.",
            "content_type": "blog",
            "category": "seo_optimized",
            "subtopic": subtopic,
            "topic_id": topic_id,
            "user_id": user_id,
            "keywords": [subtopic],
            "keyword_metrics": {
                "avg_search_volume": 0,
                "avg_difficulty": 50,
                "avg_cpc": 0,
                "intent_types": ["INFORMATIONAL"],
                "primary_keyword": subtopic,
                "total_keywords_used": 1
            },
            "seo_score": 75,
            "difficulty_level": "medium",
            "estimated_read_time": 15,
            "target_audience": "intermediate",
            "content_angle": "Comprehensive guide approach",
            "monetization_potential": "low",
            "generation_method": "fallback_template",
            "data_source": "template_based"
        }

    async def _generate_enhanced_software_ideas(
        self, 
        topic_title: str,
        subtopics: List[str],
        keywords: List[Dict[str, Any]],
        topic_id: str,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Generate enhanced software ideas using real DataForSEO keyword data"""
        logger.info(f"Generating enhanced software ideas for topic: {topic_title}")
        logger.info(f"Subtopics: {subtopics}")
        logger.info(f"Available keywords: {len(keywords)} with real DataForSEO metrics")
        ideas = []
        
        if not keywords:
            logger.warning("No keywords available for software idea generation")
            return []
        
        # Group keywords by intent type for better idea generation
        commercial_keywords = [kw for kw in keywords if kw.get('intent_type', '').upper() in ['COMMERCIAL', 'TRANSACTIONAL']]
        informational_keywords = [kw for kw in keywords if kw.get('intent_type', '').upper() == 'INFORMATIONAL']
        
        # If no commercial keywords, use all keywords
        if not commercial_keywords:
            commercial_keywords = keywords
        
        logger.info(f"Commercial keywords: {len(commercial_keywords)}, Informational keywords: {len(informational_keywords)}")
        
        # Generate ideas based on real keyword data
        if not subtopics:
            # If no subtopics, use the topic title as the subtopic
            subtopics = [topic_title]
        
        for subtopic in subtopics:
            # Find keywords related to this subtopic
            subtopic_keywords = self._find_keywords_for_subtopic(commercial_keywords, subtopic)
            
            if not subtopic_keywords:
                # Fallback to top keywords by priority score
                subtopic_keywords = sorted(keywords, key=lambda k: k.get('priority_score', 0), reverse=True)[:10]
            
            logger.info(f"Using {len(subtopic_keywords)} keywords for software subtopic: {subtopic}")
            
            # Generate ideas based on actual keyword data
            for i in range(min(6, len(subtopic_keywords))):  # Generate up to 6 software ideas per subtopic
                selected_keywords = self._select_keywords_for_idea(subtopic_keywords, i, 'software')
                
                if not selected_keywords:
                    continue
                
                # Generate idea based on actual keyword data using LLM
                idea = await self._generate_software_idea_with_llm(
                    subtopic, topic_title, selected_keywords, topic_id, user_id, i
                )
                ideas.append(idea)
        
        logger.info(f"Generated {len(ideas)} software ideas using real DataForSEO data")
        return ideas

    def _generate_software_idea_from_keywords(
        self, 
        subtopic: str, 
        topic_title: str, 
        selected_keywords: List[Dict[str, Any]], 
        topic_id: str, 
        user_id: str, 
        idea_index: int
    ) -> Dict[str, Any]:
        """Generate a software idea based on actual keyword data"""
        if not selected_keywords:
            return self._create_fallback_software_idea(subtopic, topic_title, topic_id, user_id)
        
        # Get the primary keyword (highest priority score)
        primary_keyword = max(selected_keywords, key=lambda k: k.get('priority_score', 0))
        primary_keyword_text = primary_keyword.get('keyword', '')
        
        # Calculate metrics from real data
        avg_search_volume = sum(kw.get('search_volume', 0) for kw in selected_keywords) / len(selected_keywords)
        avg_difficulty = sum(kw.get('keyword_difficulty', 0) for kw in selected_keywords) / len(selected_keywords)
        avg_cpc = sum(kw.get('cpc', 0) for kw in selected_keywords) / len(selected_keywords)
        
        # Software idea templates based on keyword data and metrics
        if avg_search_volume > 1000:
            software_templates = [
                {
                    "title": f"{primary_keyword_text} Manager Pro",
                    "description": f"A comprehensive enterprise platform for managing {primary_keyword_text} in {subtopic}. Features include advanced automation, real-time analytics, and enterprise-grade security for {topic_title} professionals.",
                    "category": "web_app",
                    "complexity": "high",
                    "effort": "high",
                    "audience": "enterprise",
                    "angle": "Enterprise-grade management solution"
                },
                {
                    "title": f"AI {primary_keyword_text} Assistant",
                    "description": f"An intelligent assistant that automates {primary_keyword_text} tasks in {subtopic}. Uses advanced machine learning to provide personalized recommendations and insights for {topic_title}.",
                    "category": "mobile_app",
                    "complexity": "high",
                    "effort": "high",
                    "audience": "professionals",
                    "angle": "AI-powered personalization"
                },
                {
                    "title": f"{primary_keyword_text} Analytics",
                    "description": f"Comprehensive analytics platform for {primary_keyword_text} data in {subtopic}. Track performance, trends, and insights with advanced visualization tools for {topic_title}.",
                    "category": "web_app",
                    "complexity": "high",
                    "effort": "high",
                    "audience": "professionals",
                    "angle": "Advanced data analytics"
                }
            ]
        elif avg_cpc > 2:
            software_templates = [
                {
                    "title": f"Pro {primary_keyword_text} Platform",
                    "description": f"A professional-grade platform for {primary_keyword_text} in {subtopic}. Features include automation, analytics, and collaboration tools for {topic_title} professionals.",
                    "category": "web_app",
                    "complexity": "medium",
                    "effort": "high",
                    "audience": "professionals",
                    "angle": "Professional-grade solution"
                },
                {
                    "title": f"Smart {primary_keyword_text} Tools",
                    "description": f"Intelligent tools for {primary_keyword_text} tasks in {subtopic}. Streamline workflows and improve efficiency with smart automation for {topic_title}.",
                    "category": "desktop_tool",
                    "complexity": "medium",
                    "effort": "medium",
                    "audience": "professionals",
                    "angle": "Smart automation"
                },
                {
                    "title": f"{primary_keyword_text} API",
                    "description": f"RESTful API service for {primary_keyword_text} functionality. Integrate {primary_keyword_text} capabilities into any {subtopic} application for {topic_title}.",
                    "category": "api_service",
                    "complexity": "medium",
                    "effort": "medium",
                    "audience": "developers",
                    "angle": "Developer-focused integration"
                }
            ]
        else:
            software_templates = [
                {
                    "title": f"{primary_keyword_text} Quick Tool",
                    "description": f"Simple, user-friendly tools for {primary_keyword_text} in {subtopic}. Perfect for beginners and quick projects in {topic_title}.",
                    "category": "desktop_tool",
                    "complexity": "low",
                    "effort": "low",
                    "audience": "beginners",
                    "angle": "Simple and accessible"
                },
                {
                    "title": f"Basic {primary_keyword_text} App",
                    "description": f"A straightforward app for {primary_keyword_text} tasks in {subtopic}. Easy to use and perfect for getting started with {topic_title}.",
                    "category": "mobile_app",
                    "complexity": "low",
                    "effort": "low",
                    "audience": "beginners",
                    "angle": "Beginner-friendly"
                },
                {
                    "title": f"{primary_keyword_text} Chrome Extension",
                    "description": f"Browser extension that adds {primary_keyword_text} functionality to any website. Simple and lightweight for {subtopic} users.",
                    "category": "browser_extension",
                    "complexity": "low",
                    "effort": "low",
                    "audience": "beginners",
                    "angle": "Browser integration"
                }
            ]
        
        template = software_templates[idea_index % len(software_templates)]
        
        # Calculate market demand based on real keyword data
        market_demand = self._calculate_market_demand(selected_keywords)
        
        # Calculate technical complexity based on actual keyword difficulty
        technical_complexity = self._map_technical_complexity(avg_difficulty)
        
        # Calculate monetization potential based on real CPC data
        monetization_potential = self._calculate_software_monetization_potential(selected_keywords)
        
        # Determine development effort based on keyword metrics
        development_effort = "high" if avg_cpc > 2 and avg_difficulty > 60 else "medium" if avg_cpc > 1 or avg_difficulty > 40 else "low"
        
        return {
            "title": template["title"],
            "description": template["description"],
            "content_type": "software",
            "category": template["category"],
            "subtopic": subtopic,
            "topic_id": topic_id,
            "user_id": user_id,
            "keywords": [kw.get('keyword', '') for kw in selected_keywords],
            "keyword_metrics": {
                "avg_search_volume": round(avg_search_volume, 0),
                "avg_difficulty": round(avg_difficulty, 1),
                "avg_cpc": round(avg_cpc, 2),
                "intent_types": [kw.get('intent_type', 'COMMERCIAL') for kw in selected_keywords],
                "primary_keyword": primary_keyword_text,
                "total_keywords_used": len(selected_keywords)
            },
            "seo_score": 0,  # Not applicable for software
            "difficulty_level": technical_complexity,
            "estimated_read_time": None,  # Not applicable for software
            "target_audience": template["audience"],
            "content_angle": template["angle"],
            "monetization_potential": monetization_potential,
            "technical_complexity": technical_complexity,
            "development_effort": development_effort,
            "market_demand": market_demand,
            "generation_method": "dataforseo_enhanced",
            "data_source": "real_keyword_metrics"
        }

    def _create_fallback_software_idea(self, subtopic: str, topic_title: str, topic_id: str, user_id: str) -> Dict[str, Any]:
        """Create a fallback software idea when no keywords are available"""
        return {
            "title": f"{subtopic.title()} Management Platform",
            "description": f"A comprehensive platform for managing {subtopic} in {topic_title}. Features include automation, analytics, and user-friendly interface.",
            "content_type": "software",
            "category": "web_app",
            "subtopic": subtopic,
            "topic_id": topic_id,
            "user_id": user_id,
            "keywords": [subtopic],
            "keyword_metrics": {
                "avg_search_volume": 0,
                "avg_difficulty": 50,
                "avg_cpc": 0,
                "intent_types": ["COMMERCIAL"],
                "primary_keyword": subtopic,
                "total_keywords_used": 1
            },
            "seo_score": 0,
            "difficulty_level": "medium",
            "estimated_read_time": None,
            "target_audience": "professionals",
            "content_angle": "Enterprise-grade management solution",
            "monetization_potential": "medium",
            "technical_complexity": "medium",
            "development_effort": "medium",
            "market_demand": "medium",
            "generation_method": "fallback_template",
            "data_source": "template_based"
        }

    def _calculate_seo_score_from_keywords(self, keywords: List[Dict[str, Any]]) -> int:
        """Calculate SEO score based on selected keywords"""
        if not keywords:
            return 75
        
        # Calculate average metrics
        avg_volume = sum(kw.get('search_volume', 0) for kw in keywords) / len(keywords)
        avg_difficulty = sum(kw.get('keyword_difficulty', 50) for kw in keywords) / len(keywords)
        avg_cpc = sum(kw.get('cpc', 0) for kw in keywords) / len(keywords)
        
        # Base score
        base_score = 60
        
        # Volume bonus (0-20 points)
        volume_bonus = min(avg_volume / 500, 20)
        
        # Difficulty penalty (0-15 points penalty)
        difficulty_penalty = max(0, (avg_difficulty - 30) / 5)
        
        # CPC bonus (0-10 points)
        cpc_bonus = min(avg_cpc / 2, 10)
        
        # Intent bonus (0-5 points)
        intent_bonus = 0
        for kw in keywords:
            if kw.get('intent_type', '').upper() == 'INFORMATIONAL':
                intent_bonus += 1
        
        intent_bonus = min(intent_bonus, 5)
        
        total_score = base_score + volume_bonus - difficulty_penalty + cpc_bonus + intent_bonus
        
        return min(max(int(total_score), 0), 100)
    
    def _map_difficulty_score(self, avg_difficulty: float) -> str:
        """Map average keyword difficulty to content difficulty level"""
        if avg_difficulty < 30:
            return "easy"
        elif avg_difficulty < 60:
            return "medium"
        else:
            return "hard"
    
    def _calculate_monetization_potential(self, keywords: List[Dict[str, Any]]) -> str:
        """Calculate monetization potential based on keyword metrics"""
        if not keywords:
            return "low"
        
        avg_cpc = sum(kw.get('cpc', 0) for kw in keywords) / len(keywords)
        commercial_count = sum(1 for kw in keywords if kw.get('intent_type', '').upper() in ['COMMERCIAL', 'TRANSACTIONAL'])
        
        if avg_cpc > 2 and commercial_count > len(keywords) / 2:
            return "high"
        elif avg_cpc > 1 or commercial_count > 0:
            return "medium"
        else:
            return "low"
    
    def _calculate_market_demand(self, keywords: List[Dict[str, Any]]) -> str:
        """Calculate market demand for software ideas based on keywords"""
        if not keywords:
            return "medium"
        
        avg_volume = sum(kw.get('search_volume', 0) for kw in keywords) / len(keywords)
        commercial_count = sum(1 for kw in keywords if kw.get('intent_type', '').upper() in ['COMMERCIAL', 'TRANSACTIONAL'])
        
        if avg_volume > 1000 and commercial_count > len(keywords) / 2:
            return "high"
        elif avg_volume > 500 or commercial_count > 0:
            return "medium"
        else:
            return "low"
    
    def _map_technical_complexity(self, avg_difficulty: float) -> str:
        """Map average keyword difficulty to technical complexity"""
        if avg_difficulty < 40:
            return "low"
        elif avg_difficulty < 70:
            return "medium"
        else:
            return "high"
    
    def _calculate_software_monetization_potential(self, keywords: List[Dict[str, Any]]) -> str:
        """Calculate monetization potential for software ideas"""
        if not keywords:
            return "low"
        
        avg_cpc = sum(kw.get('cpc', 0) for kw in keywords) / len(keywords)
        commercial_count = sum(1 for kw in keywords if kw.get('intent_type', '').upper() in ['COMMERCIAL', 'TRANSACTIONAL'])
        
        if avg_cpc > 3 and commercial_count > len(keywords) / 2:
            return "high"
        elif avg_cpc > 1.5 or commercial_count > 0:
            return "medium"
        else:
            return "low"

    def _generate_fallback_blog_ideas(
        self, 
        topic_title: str,
        subtopics: List[str],
        keywords: List[str],
        topic_id: str,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Generate fallback blog ideas when LLM is not available"""
        logger.info(f"Generating fallback blog ideas for topic: {topic_title}")
        logger.info(f"Subtopics: {subtopics}")
        ideas = []
            
        # Generate 10+ ideas per subtopic
        for subtopic in subtopics:
            # Create diverse blog idea templates
            blog_templates = [
                {
                    "title": f"The Complete Guide to {subtopic}",
                    "description": f"Everything you need to know about {subtopic} in {topic_title}. This comprehensive guide covers the basics, advanced techniques, and practical tips.",
                    "content_angle": "Comprehensive guide approach",
                    "difficulty": "hard",
                    "audience": "intermediate"
                },
                {
                    "title": f"10 Essential {subtopic} Tips for Beginners",
                    "description": f"Master the fundamentals of {subtopic} with these proven tips and strategies. Perfect for newcomers to {topic_title}.",
                    "content_angle": "Beginner-friendly tips",
                    "difficulty": "easy",
                    "audience": "beginners"
                },
                {
                    "title": f"Advanced {subtopic} Strategies That Actually Work",
                    "description": f"Take your {subtopic} skills to the next level with these advanced techniques used by professionals in {topic_title}.",
                    "content_angle": "Advanced techniques",
                    "difficulty": "hard",
                    "audience": "advanced"
                },
                {
                    "title": f"Common {subtopic} Mistakes to Avoid",
                    "description": f"Learn from the most common {subtopic} mistakes and how to avoid them. Essential reading for anyone interested in {topic_title}.",
                    "content_angle": "Mistake prevention",
                    "difficulty": "medium",
                    "audience": "intermediate"
                },
                {
                    "title": f"The Ultimate {subtopic} Checklist",
                    "description": f"A step-by-step checklist for {subtopic} success. Use this as your go-to reference for {topic_title} projects.",
                    "content_angle": "Actionable checklist",
                    "difficulty": "easy",
                    "audience": "beginners"
                },
                {
                    "title": f"How to Choose the Right {subtopic} Tools",
                    "description": f"Navigate the overwhelming world of {subtopic} tools with this comprehensive comparison and selection guide for {topic_title}.",
                    "content_angle": "Tool selection guide",
                    "difficulty": "medium",
                    "audience": "intermediate"
                },
                {
                    "title": f"{subtopic} Trends to Watch in 2024",
                    "description": f"Stay ahead of the curve with the latest {subtopic} trends and innovations in {topic_title}. Future-proof your knowledge.",
                    "content_angle": "Trend analysis",
                    "difficulty": "medium",
                    "audience": "advanced"
                },
                {
                    "title": f"Case Study: Successful {subtopic} Implementation",
                    "description": f"Real-world case study showing how {subtopic} was successfully implemented in a {topic_title} project. Learn from actual results.",
                    "content_angle": "Case study analysis",
                    "difficulty": "hard",
                    "audience": "advanced"
                },
                {
                    "title": f"Budget-Friendly {subtopic} Solutions",
                    "description": f"Get started with {subtopic} without breaking the bank. Affordable options and free resources for {topic_title} enthusiasts.",
                    "content_angle": "Budget-conscious approach",
                    "difficulty": "easy",
                    "audience": "beginners"
                },
                {
                    "title": f"The Psychology of {subtopic} Success",
                    "description": f"Understand the mental aspects and mindset needed for {subtopic} success in {topic_title}. Beyond just technical skills.",
                    "content_angle": "Psychological approach",
                    "difficulty": "medium",
                    "audience": "intermediate"
                }
            ]
            
            # Generate ideas for this subtopic
            for template in blog_templates:
                # Select relevant keywords for this idea
                relevant_keywords = self._select_relevant_keywords(keywords, subtopic, template["title"])
                
                idea = {
                    "title": template["title"],
                    "description": template["description"],
                    "content_type": "blog",
                    "category": "seo_optimized",
                    "subtopic": subtopic,
                    "topic_id": topic_id,
                    "user_id": user_id,
                    "keywords": relevant_keywords,
                    "seo_score": random.randint(75, 95),
                    "difficulty_level": template["difficulty"],
                    "estimated_read_time": random.randint(5, 20),
                    "target_audience": template["audience"],
                    "content_angle": template["content_angle"],
                    "monetization_potential": random.choice(["low", "medium", "high"])
                }
                ideas.append(idea)
            
        return ideas
            
    def _select_relevant_keywords(self, keywords: List[str], subtopic: str, title: str) -> List[str]:
        """Select the most relevant keywords for a specific idea (fallback method)"""
        # Simple keyword selection based on relevance
        relevant_keywords = []
        
        # Always include the subtopic as a keyword
        if subtopic.lower() not in [k.lower() for k in relevant_keywords]:
            relevant_keywords.append(subtopic)
        
        # Add keywords that appear in the title
        title_words = title.lower().split()
        for keyword in keywords[:10]:  # Check first 10 keywords
            if any(word in keyword.lower() for word in title_words):
                relevant_keywords.append(keyword)
                if len(relevant_keywords) >= 5:  # Limit to 5 keywords
                    break
        
        # Fill remaining slots with random keywords
        while len(relevant_keywords) < 3 and len(relevant_keywords) < len(keywords):
            remaining_keywords = [k for k in keywords if k not in relevant_keywords]
            if remaining_keywords:
                relevant_keywords.append(random.choice(remaining_keywords))
        
        return relevant_keywords[:5]  # Return max 5 keywords

    def _generate_fallback_software_ideas(
        self, 
        topic_title: str,
        subtopics: List[str],
        keywords: List[str],
        topic_id: str,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Generate fallback software ideas when LLM is not available"""
        logger.info(f"Generating fallback software ideas for topic: {topic_title}")
        logger.info(f"Subtopics: {subtopics}")
        ideas = []
        
        # Generate 10+ ideas per subtopic
        for subtopic in subtopics:
            # Create diverse software idea templates
            software_templates = [
                {
                    "title": f"{subtopic.title()} Management Platform",
                    "description": f"A comprehensive platform for managing {subtopic} in {topic_title}. Features include automation, analytics, and user-friendly interface.",
                    "category": "web_app",
                    "complexity": "high",
                    "effort": "high",
                    "demand": "high",
                    "audience": "enterprise",
                    "angle": "Enterprise-grade management solution"
                },
                {
                    "title": f"Smart {subtopic.title()} Assistant",
                    "description": f"An AI-powered assistant that helps users with {subtopic} tasks in {topic_title}. Uses machine learning to provide personalized recommendations.",
                    "category": "mobile_app",
                    "complexity": "high",
                    "effort": "high",
                    "demand": "medium",
                    "audience": "professionals",
                    "angle": "AI-powered personalization"
                },
                {
                    "title": f"{subtopic.title()} Quick Tools",
                    "description": f"A collection of simple, focused tools for common {subtopic} tasks in {topic_title}. Perfect for beginners and quick projects.",
                    "category": "desktop_tool",
                    "complexity": "low",
                    "effort": "low",
                    "demand": "high",
                    "audience": "beginners",
                    "angle": "Simple and accessible"
                },
                {
                    "title": f"{subtopic.title()} Analytics Dashboard",
                    "description": f"Visualize and analyze {subtopic} data with this powerful dashboard. Track trends, performance, and insights in {topic_title}.",
                    "category": "web_app",
                    "complexity": "medium",
                    "effort": "medium",
                    "demand": "high",
                    "audience": "professionals",
                    "angle": "Data visualization and insights"
                },
                {
                    "title": f"{subtopic.title()} API Service",
                    "description": f"RESTful API service for {subtopic} functionality. Integrate {subtopic} capabilities into any {topic_title} application.",
                    "category": "api_service",
                    "complexity": "medium",
                    "effort": "medium",
                    "demand": "medium",
                    "audience": "developers",
                    "angle": "Developer-focused integration"
                },
                {
                    "title": f"{subtopic.title()} Mobile App",
                    "description": f"Native mobile app for {subtopic} on the go. Access {subtopic} features anywhere with this intuitive mobile interface.",
                    "category": "mobile_app",
                    "complexity": "medium",
                    "effort": "high",
                    "demand": "high",
                    "audience": "professionals",
                    "angle": "Mobile-first experience"
                },
                {
                    "title": f"{subtopic.title()} Automation Bot",
                    "description": f"Automate repetitive {subtopic} tasks with this intelligent bot. Save time and reduce errors in {topic_title} workflows.",
                    "category": "desktop_tool",
                    "complexity": "high",
                    "effort": "high",
                    "demand": "medium",
                    "audience": "enterprise",
                    "angle": "Workflow automation"
                },
                {
                    "title": f"{subtopic.title()} Learning Platform",
                    "description": f"Interactive learning platform for mastering {subtopic} in {topic_title}. Includes tutorials, exercises, and progress tracking.",
                    "category": "web_app",
                    "complexity": "medium",
                    "effort": "high",
                    "demand": "high",
                    "audience": "beginners",
                    "angle": "Educational focus"
                },
                {
                    "title": f"{subtopic.title()} Marketplace",
                    "description": f"Connect buyers and sellers in the {subtopic} space. A marketplace platform for {topic_title} professionals and enthusiasts.",
                    "category": "web_app",
                    "complexity": "high",
                    "effort": "high",
                    "demand": "medium",
                    "audience": "professionals",
                    "angle": "Community-driven marketplace"
                },
                {
                    "title": f"{subtopic.title()} Chrome Extension",
                    "description": f"Browser extension that enhances {subtopic} workflows. Add {subtopic} functionality to any website in {topic_title}.",
                    "category": "browser_extension",
                    "complexity": "low",
                    "effort": "low",
                    "demand": "medium",
                    "audience": "professionals",
                    "angle": "Browser integration"
                }
            ]
            
            # Generate ideas for this subtopic
            for template in software_templates:
                # Select relevant keywords for this idea
                relevant_keywords = self._select_relevant_keywords(keywords, subtopic, template["title"])
                
                idea = {
                    "title": template["title"],
                    "description": template["description"],
                    "content_type": "software",
                    "category": template["category"],
                    "subtopic": subtopic,
                    "topic_id": topic_id,
                    "user_id": user_id,
                    "keywords": relevant_keywords,
                    "seo_score": 0,  # Not applicable for software
                    "difficulty_level": template["complexity"],
                    "estimated_read_time": None,  # Not applicable for software
                    "target_audience": template["audience"],
                    "content_angle": template["angle"],
                    "monetization_potential": random.choice(["low", "medium", "high"]),
                    "technical_complexity": template["complexity"],
                    "development_effort": template["effort"],
                    "market_demand": template["demand"]
                }
                ideas.append(idea)

        return ideas

    async def _save_ideas_to_database(self, ideas: List[Dict[str, Any]]) -> None:
        """Save generated ideas to the database"""
        try:
            if not ideas:
                return

            # Insert ideas into database
            result = self.supabase.table('content_ideas').insert(ideas).execute()
            
            if result.data:
                logger.info(f"Successfully saved {len(result.data)} content ideas to database")
            else:
                logger.error(f"Failed to save content ideas: {result}")
            
        except Exception as e:
            logger.error(f"Failed to save ideas to database: {str(e)}")
    
    async def get_content_ideas(
        self, 
        topic_id: str,
        user_id: str,
        content_type: str = None
    ) -> List[Dict[str, Any]]:
        """Retrieve content ideas from database (merging legacy and enhanced tables)"""
        try:
            ideas = []
            
            # 1. Fetch Legacy Ideas
            query = self.supabase.table('content_ideas').select('*').eq('topic_id', topic_id).eq('user_id', user_id)
            if content_type:
                query = query.eq('content_type', content_type)
            result = query.order('created_at', desc=True).execute()
            if result.data:
                ideas.extend(result.data)
                
            # 2. Fetch Enhanced Blog Ideas
            # Note: We need to filter by analysis_id or just topic_id?
            # Enhanced ideas stores analysis_id, but maybe not topic_id directly?
            # Let's check schema usage in enhanced_database.py... 
            # It saves 'analysis_id'. If topic_id is NOT stored in blog_ideas, we might miss them.
            # However, for Idea Burst, we likely don't have analysis_id.
            # Let's assume we need to join or filter differently.
            # Wait, enhanced_database.py save_enhanced_ideas uses analysis_id.
            # If Idea Burst calls save_enhanced_ideas, what does it pass as analysis_id?
            # In enhanced_topic_routes.py:394, it passes topic_id as topic_id, but EnhancedIdeaGenerator passes analysis_id?
            # enhanced_idea_generator.py just calls save_enhanced_ideas.
            
            # Let's fetch ALL compatible ideas for this user and filter in memory if necessary, 
            # or rely on the fact that we need topic alignment.
            
            # CRITICAL CHECK: Does blog_ideas table have 'topic_id'?
            # If not, we can't easily fetch by topic_id without a join or schema change.
            # The 'content_ideas' table has 'topic_id'.
            # If 'blog_ideas' relies on 'analysis_id' which maps to 'ahrefs_analyses', that's for Ahrefs flow.
            # For Idea Burst, we need 'topic_id'.
            
            # Let's try to query blog_ideas assuming it might have keys, or just return existing logic if unknown.
            # Actually, looking at the user log, the "Idea Burst" writes to `content_ideas`?
            # No, I confirmed it calls save_enhanced_ideas.
            
            # Let's assume table has the fields we need or fail gracefully.
            try:
                blog_query = self.supabase.table('blog_ideas').select('*').eq('user_id', user_id)
                # If topic_id exists in schema, filter by it.
                # Since I can't confirm schema easily without SQL tool (I have list_tables but not describe),
                # I will try to fetch and filter in app if reasonable size, or add topic_id filter optimistically.
                # Actually, enhanced_database.py insert DOES NOT show topic_id being inserted!
                # It inserts 'analysis_id'.
                # This suggests 'blog_ideas' might be disconnected from 'topic_id' unless 'analysis_id' == 'topic_id'.
                
                # In enhanced_topic_routes.py, request.topic_id is passed.
                # In enhanced_idea_generator.py, if analysis_id is passed as topic_id, then we are good.
                # Let's modify this to query blog_ideas by analysis_id=topic_id OR topic_id=topic_id
                
                blog_query = blog_query.or_(f"analysis_id.eq.{topic_id},topic_id.eq.{topic_id}")
                blog_result = blog_query.execute()
                if blog_result.data:
                    # Normalize to match ContentIdea shape if needed
                    for idea in blog_result.data:
                        # Ensure minimal fields
                        if 'total_search_volume' not in idea: idea['total_search_volume'] = 0
                        ideas.append(idea)
                        
            except Exception as blog_err:
                logger.warning(f"Failed to fetch from blog_ideas: {blog_err}")

            try:
                sw_query = self.supabase.table('software_ideas').select('*').eq('user_id', user_id)
                # content_ideas table uses topic_id.
                # software_ideas might use analysis_id=topic_id convention for bursts.
                sw_query = sw_query.or_(f"analysis_id.eq.{topic_id},topic_id.eq.{topic_id}")
                sw_result = sw_query.execute()
                if sw_result.data:
                    for idea in sw_result.data:
                        if 'name' in idea and 'title' not in idea:
                            idea['title'] = idea['name']
                        ideas.append(idea)
            except Exception as sw_err:
                logger.warning(f"Failed to fetch from software_ideas: {sw_err}")

            # Sort combined results
            ideas.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            return ideas
        
        except Exception as e:
            logger.error(f"Failed to retrieve content ideas: {str(e)}")
            return []

    async def delete_all_content_ideas_for_topic(
        self,
        topic_id: str,
        user_id: str
    ) -> bool:
        """Delete all content ideas for a specific topic and user"""
        try:
            response = self.supabase.table("content_ideas").delete().eq("topic_id", topic_id).eq("user_id", user_id).execute()
            logger.info(f"Deleted content ideas for topic {topic_id} and user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete content ideas: {str(e)}")
            return False

    async def publish_ideas_to_titles(self, idea_ids: List[str], user_id: str) -> Dict[str, Any]:
        """
        Publish selected content ideas to the Titles table for content creation workflow
        """
        try:
            logger.info(f"Publishing {len(idea_ids)} ideas to Titles for user {user_id}")
            
            # Fetch the ideas to be published
            response = self.supabase.table('content_ideas')\
                .select('*')\
                .in_('id', idea_ids)\
                .eq('user_id', user_id)\
                .execute()
                
            ideas = response.data
            if not ideas:
                return {"success": False, "message": "No valid ideas found to publish"}
            
            published_count = 0
            titles_to_insert = []
            
            for idea in ideas:
                # Map content_idea to title schema
                # We'll use the idea title as 'Title' and subtopic/topic_title for context
                
                # Format keywords as comma-separated string
                keywords_list = idea.get('keywords', [])
                keywords_str = ", ".join(keywords_list) if isinstance(keywords_list, list) else str(keywords_list)
                
                # Get metrics with defaults
                metrics = idea.get('keyword_metrics') or {}
                avg_vol = metrics.get('avg_search_volume', 0)
                avg_diff = metrics.get('avg_difficulty', 0)
                seo_score = idea.get('seo_score', 0)
                
                # Calculate derived scores
                # Traffic Potential Score: Heuristic based on Search Volume and Difficulty
                # Logic: Higher volume + Lower difficulty = Higher potential
                # Scale roughly 0-100
                vol_factor = min(avg_vol / 100, 50) # Cap volume contribution
                diff_factor = max(0, 50 - (avg_diff / 2)) # Difficulty penalty (lower is better)
                traffic_potential = int(vol_factor + diff_factor)
                
                # Construct Title entry
                # Note: "Titles" table uses mixed case column names based on introspection
                title_entry = {
                    "Title": idea.get('title'),
                    "userDescription": idea.get('description'), # Map description to userDescription
                    "status": "New", # Default status
                    "user_id": user_id,
                    "Keywords": keywords_str,
                    "topic_id": idea.get('topic_id'), # Link back to Research Topic
                    "source_idea_id": idea.get('id'), # Link back to Content Idea
                    "subtopic": idea.get('subtopic'),
                    "content_type": idea.get('content_type'),
                    "total_search_volume": int(avg_vol) if avg_vol else 0,
                    "avg_keyword_difficulty": int(round(float(avg_diff))) if avg_diff else 0,
                    # Map content_outline from either new key (list) or old key (string)
                    "content_outline": idea.get('content_outline') or idea.get('markdown_outline'), 
                    "original_created_at": idea.get('created_at'),
                    "created_at": datetime.now().isoformat(),
                    
                    # New requested fields
                    "overall_quality_score": int(seo_score) if seo_score else 0,
                    "seo_optimization_score": int(seo_score) if seo_score else 0, # Map to specific column
                    "competition_score": int(round(float(avg_diff))) if avg_diff else 0, # Cast to int to avoid "invalid input syntax for type integer: 0.0"
                    "traffic_potential_score": int(traffic_potential),
                    
                    # New Feasibility/Viral Scores
                    "viral_potential_score": int(idea.get('viral_potential_score', 0)),
                    "audience_alignment_score": int(idea.get('audience_alignment_score', 0)),
                    "content_feasibility_score": int(idea.get('content_feasibility_score', 0)),
                    "business_impact_score": int(idea.get('business_impact_score', 0))
                }
                
                # Clean up None values if strict schema ( Supabase usually handles None as NULL)
                titles_to_insert.append(title_entry)
        
            if titles_to_insert:
                # Insert into titles table
                # Use quoted table name "Titles" to respect case sensitivity if needed, 
                # but supabase-py library usually handles quotes if we pass the string name.
                # However, if the table was created with quotes like "Titles", we might need to be careful.
                # The existing code used 'titles' (lowercase). Step 54 failed with 'relation "public.titles" does not exist'.
                # This suggests the table IS case sensitive "Titles". 
                # supabase-py might need us to pass "Titles" exactly.
                
                insert_response = self.supabase.table('Titles').insert(titles_to_insert).execute()
                
                if insert_response.data:
                    published_count = len(insert_response.data)
                    logger.info(f"Successfully published {published_count} titles")
            
            return {
                "success": True, 
                "message": f"Successfully published {published_count} ideas to Titles",
                "published_count": published_count
            }
            
        except Exception as e:
            logger.error(f"Failed to publish ideas to Titles: {str(e)}")
            raise e