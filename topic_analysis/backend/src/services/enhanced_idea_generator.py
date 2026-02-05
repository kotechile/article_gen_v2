"""
Enhanced idea generator with separate paths for blog and software ideas
"""

import logging
from typing import Dict, List, Any, Optional, Union
from ..services.enhanced_database import enhanced_database_service
from src.core.config import settings
import re
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class EnhancedIdeaGenerator:
    """Enhanced idea generator with separate paths for blog and software ideas"""
    
    def __init__(self):
        self.llm_service = None  # Initialize with your existing LLM service
        self.google_autocomplete_service = None  # Initialize with your existing service
    
    async def generate_blog_ideas_from_keywords(
        self, 
        keywords_data: List[Dict[str, Any]], 
        user_id: str,
        analysis_id: str
    ) -> List[Dict[str, Any]]:
        """
        Generate blog ideas from Ahrefs keyword analysis
        
        Args:
            keywords_data: Processed Ahrefs keywords
            user_id: User ID
            analysis_id: Analysis ID
            
        Returns:
            List of blog ideas
        """
        try:
            # Get highest ranked keywords (bigger volume, easier to rank)
            top_keywords = self._get_top_keywords(keywords_data)
            
            # Generate blog ideas using LLM with enhanced keywords
            blog_ideas = await self._generate_blog_ideas_with_llm(
                top_keywords, 
                user_id, 
                analysis_id,
                enhanced_with_ahrefs=True
            )
            
            # Save to database
            await enhanced_database_service.save_enhanced_ideas({
                'user_id': user_id,
                'analysis_id': analysis_id,
                'blog_ideas': blog_ideas
            })
            
            logger.info(f"Generated {len(blog_ideas)} blog ideas from Ahrefs keywords")
            return blog_ideas
            
        except Exception as e:
            logger.error(f"Error generating blog ideas from keywords: {str(e)}")
            raise ValueError(f"Failed to generate blog ideas: {str(e)}")
    
    async def generate_software_ideas_separately(
        self, 
        user_id: str,
        seed_keywords: Optional[List[str]] = None,
        enhanced_with_ahrefs: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate software ideas separately (not keyword-related)
        
        Args:
            user_id: User ID
            seed_keywords: Optional seed keywords for context
            enhanced_with_ahrefs: Whether to use Ahrefs data for context
            
        Returns:
            List of software ideas
        """
        try:
            # Generate software ideas using LLM
            software_ideas = await self._generate_software_ideas_with_llm(
                user_id, 
                seed_keywords,
                enhanced_with_ahrefs
            )
            
            # Save to database
            await enhanced_database_service.save_enhanced_ideas({
                'user_id': user_id,
                'analysis_id': None,  # No analysis ID for separate generation
                'software_ideas': software_ideas
            })
            
            logger.info(f"Generated {len(software_ideas)} software ideas separately")
            return software_ideas
            
        except Exception as e:
            logger.error(f"Error generating software ideas: {str(e)}")
            raise ValueError(f"Failed to generate software ideas: {str(e)}")
    
    async def generate_blog_ideas_from_seed_keywords(
        self, 
        seed_keywords: List[str], 
        user_id: str
    ) -> List[Dict[str, Any]]:
        """
        Generate blog ideas from seed keywords (existing functionality)
        
        Args:
            seed_keywords: Seed keywords from user input
            user_id: User ID
            
        Returns:
            List of blog ideas
        """
        try:
            # Use existing LLM + Google Autocomplete functionality
            blog_ideas = await self._generate_blog_ideas_with_llm(
                seed_keywords, 
                user_id, 
                None,  # No analysis ID for seed keywords
                enhanced_with_ahrefs=False
            )
            
            logger.info(f"Generated {len(blog_ideas)} blog ideas from seed keywords")
            return blog_ideas
            
        except Exception as e:
            logger.error(f"Error generating blog ideas from seed keywords: {str(e)}")
            raise ValueError(f"Failed to generate blog ideas: {str(e)}")
    
    def _get_top_keywords(self, keywords_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get top keywords based on volume and difficulty"""
        # Sort by opportunity score and volume
        sorted_keywords = sorted(
            keywords_data,
            key=lambda x: (x.get('opportunity_score', 0), x.get('search_volume', 0)),
            reverse=True
        )
        
        # Return top 20 keywords
        return sorted_keywords[:20]
    
    async def _generate_blog_ideas_with_llm(
        self, 
        keywords: Union[List[str], List[Dict[str, Any]]], 
        user_id: str,
        analysis_id: Optional[str],
        enhanced_with_ahrefs: bool
    ) -> List[Dict[str, Any]]:
        """Generate blog ideas using LLM"""
        try:
            # Prepare keywords for LLM
            if isinstance(keywords[0], dict):
                # Ahrefs keywords with metrics
                keyword_texts = [k['keyword'] for k in keywords]
                keyword_metrics = {k['keyword']: k for k in keywords}
            else:
                # Seed keywords
                keyword_texts = keywords
                keyword_metrics = {}
            
            # Call your existing LLM service
            # This would integrate with your existing LLM functionality
            blog_ideas = await self._call_llm_for_blog_ideas(
                keyword_texts,
                keyword_metrics,
                enhanced_with_ahrefs
            )
            
            # Add metadata
            for idea in blog_ideas:
                idea['user_id'] = user_id
                idea['analysis_id'] = analysis_id
                idea['enhanced_with_ahrefs'] = enhanced_with_ahrefs
                idea['type'] = 'blog'
            
            return blog_ideas
            
        except Exception as e:
            logger.error(f"Error generating blog ideas with LLM: {str(e)}")
            raise ValueError(f"Failed to generate blog ideas with LLM: {str(e)}")
    
    async def _generate_software_ideas_with_llm(
        self, 
        user_id: str,
        seed_keywords: Optional[List[str]],
        enhanced_with_ahrefs: bool
    ) -> List[Dict[str, Any]]:
        """Generate software ideas using LLM"""
        try:
            # Call your existing LLM service for software ideas
            # This would integrate with your existing LLM functionality
            software_ideas = await self._call_llm_for_software_ideas(
                seed_keywords,
                enhanced_with_ahrefs
            )
            
            # Add metadata
            for idea in software_ideas:
                idea['user_id'] = user_id
                idea['analysis_id'] = None
                idea['enhanced_with_ahrefs'] = enhanced_with_ahrefs
                idea['type'] = 'software'
            
            return software_ideas
            
        except Exception as e:
            logger.error(f"Error generating software ideas with LLM: {str(e)}")
            raise ValueError(f"Failed to generate software ideas with LLM: {str(e)}")
    
    async def _call_llm_for_blog_ideas(
        self, 
        keywords: List[str], 
        keyword_metrics: Dict[str, Any],
        enhanced_with_ahrefs: bool
    ) -> List[Dict[str, Any]]:
        """Call LLM service for blog ideas using real provider"""
        try:
            from ..integrations.llm_providers import llm_providers_manager
            import json
            import re
            
            # Construct a rich context string from keyword metrics if available
            metrics_context = ""
            if keyword_metrics:
                metrics_context = "Here are the keyword statistics (use these real values for Difficulty and Traffic Potential where applicable):\n"
                for kw in keywords[:10]: # Limit context to top 10 keywords to save tokens
                    if kw in keyword_metrics:
                        data = keyword_metrics[kw]
                        # Extract DataForSEO specific fields safely
                        vol = data.get('search_volume') or data.get('volume', 'N/A')
                        diff = data.get('average_difficulty') or data.get('difficulty', 'N/A')
                        cpc = data.get('average_cpc') or data.get('cpc', 'N/A')
                        metrics_context += f"- '{kw}': Volume={vol}, Difficulty={diff}, CPC={cpc}\n"
            
            prompt = f"""
            You are a Senior Content Strategist. Your goal is to create high-performing content blueprints that target specific keyword clusters.
            
            CLUSTER CONTEXT:
            Topic Keywords: {', '.join(keywords[:15])}
            
            {metrics_context}
            
            TASK:
            Generate 5 detailed "Content Blueprints" (Blog Ideas) based on these keywords.
            
            CRITICAL REQUIREMENT: 
            For the 'content_outline' field, do NOT just list headers. You must create a "Strategic Blueprint" where each item in the list corresponds to a section, formatted exactly like this:
            
            "H2: [Heading using specific keyword] | Intent: [User Intent/Question] | Keywords: [Specific keywords to include] | Affiliate Hook: [Where to insert product/review]"
            
            Example of a good outline item:
            "H2: White Vinegar & Dish Soap Recipes | Intent: How to make safe weed killer at home | Keywords: vinegar weed killer recipe, dawn dish soap | Affiliate Hook: Review heavy-duty spray bottles"
            
            Required JSON Fields per idea:
            - title: Catchy, clear title (H1).
            - content_type: (article, guide, comparison, list, review, etc.)
            - primary_keywords: [list of top 3 keywords from cluster]
            - secondary_keywords: [list of next 5 keywords]
            - description: Compelling meta-description style summary.
            
            METRICS (0-100 score):
            - seo_optimization_score: How well this matches search intent (High=80+).
            - traffic_potential_score: Potential traffic volume (High=80+ for >10k vol).
            - audience_alignment_score: How well this fits the target audience (High=85+).
            - content_feasibility_score: How easy/feasible it is to write this (High = Easy).
            - business_impact_score: Potential for conversions/revenue (High=80+).
            
            ADDITIONAL DATA:
            - average_difficulty: Numeric 0-100 (Use provided data or estimate).
            - difficulty_level: String ('beginner', 'intermediate', 'expert').
            - estimated_read_time: Numeric (minutes).
            - estimated_word_count: Numeric.
            - target_audience: String (e.g., 'Small Business Owners', 'SEO Beginners').
            - average_cpc: Numeric (use provided data or estimate).
            - total_search_volume: Numeric (use provided data or estimate).
            - optimization_tips: [list of 2 strings].
            - content_outline: [List of 4-6 "Strategic Blueprint" strings as defined above].
            - monetization_hook: Specific angle to monetize this post (e.g., "Review X product").
            
            Return ONLY a valid JSON array of objects. Do not include markdown code blocks.
            """
            
            # Call the LLM (using defaults, e.g., OpenAI or currently active provider)
            # You might want to categorize the provider in settings or just use default.
            response = await llm_providers_manager.providers['openai'].generate_content(
                prompt=prompt,
                max_tokens=2500,
                temperature=0.7
            )
            
            if "error" in response:
                logger.error(f"LLM generation failed: {response['error']}")
                raise ValueError(f"LLM generation failed: {response['error']}")
                
            content_text = response.get("content", "")
            
            # Clean up potential markdown formatting (```json ... ```)
            content_text = re.sub(r'^```json\s*', '', content_text)
            content_text = re.sub(r'\s*```$', '', content_text)
            
            try:
                blog_ideas = json.loads(content_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON: {e}. Content: {content_text[:100]}...")
                # Fallback: return empty list or retry
                return []

            # Post-processing to ensure all keys exist and calculate Viability Score
            final_ideas = []
            for i, idea in enumerate(blog_ideas):
                # Ensure core ID is present
                idea['id'] = f"generated_idea_{i}_{int(datetime.now().timestamp())}"
                
                # Ensure other numeric fields are numbers
                for field in ['seo_optimization_score', 'traffic_potential_score', 
                              'audience_alignment_score', 'content_feasibility_score', 
                              'business_impact_score', 'average_difficulty', 
                              'total_search_volume', 'average_cpc']:
                    if field in idea:
                        try:
                            idea[field] = float(idea[field])
                        except (ValueError, TypeError):
                            idea[field] = 0

                # --- Viability Score Calculation ---
                # A. Trend Score (Str) - 30%
                # Use Google Trends slope (if available) or interest level. 
                # For now, we simulate this or use a placeholder if not in data.
                # Ideally, this should come from real trend data.
                # Fallback: Use traffic_potential_score / 10 as a proxy for trend if no direct trend data.
                trend_val = idea.get('traffic_potential_score', 50) 
                # If we had real trend value 0-100:
                s_tr = trend_val / 10.0
                
                # B. Monetization Score (Smon) - 40%
                # Input: Number of Affiliate Offers + Average Commission %.
                # We don't have real affiliate data here yet, so we use 'business_impact_score' as a proxy 
                # or estimate based on 'monetization_potential' string if we parsed it.
                # Logic: * 0 offers = 0. 1–2 offers = 5. 3+ offers = 10.
                # Let's map business_impact_score (0-100) to this 0-10 scale for now.
                biz_impact = idea.get('business_impact_score', 50)
                s_mon = biz_impact / 10.0
                # refined logic based on description if available could go here

                # C. SEO Ease Score (Sseo) - 30%
                # Input: Keyword Difficulty (KD) from DataForSEO (0–100).
                # Logic: Inverse relationship. 10 - (KD / 10).
                kd = idea.get('average_difficulty', 50)
                s_seo = max(0, 10 - (kd / 10.0))

                # Final Calculation
                # VS = (Wtr * Str) + (Wmon * Smon) + (Wseo * Sseo)
                w_tr = 0.3
                w_mon = 0.4
                w_seo = 0.3
                
                viability_score = (w_tr * s_tr) + (w_mon * s_mon) + (w_seo * s_seo)
                
                # Scale to 0-100 and round
                idea['viability_score'] = round(viability_score * 10, 1)
                
                # Store component scores for UI tooltip/breakdown (0-10 scale)
                idea['trend_score'] = round(s_tr, 1)
                idea['monetization_score'] = round(s_mon, 1)
                idea['seo_ease_score'] = round(s_seo, 1)
                
                final_ideas.append(idea)
                
            return final_ideas
        except Exception as e:
            logger.error(f"Error in _call_llm_for_blog_ideas: {str(e)}")
            # In case of major failure, returning empty list is safer than crashing
            return []
    
    async def _call_llm_for_software_ideas(
        self, 
        seed_keywords: Optional[List[str]],
        enhanced_with_ahrefs: bool
    ) -> List[Dict[str, Any]]:
        """Call LLM service for software ideas"""
        # This would integrate with your existing LLM service
        # For now, return mock data
        return [
            {
                'id': f"software_idea_{i}",
                'title': f"Software Idea {i}",
                'description': f"Description for software idea {i}",
                'features': [
                    "Feature 1",
                    "Feature 2",
                    "Feature 3"
                ],
                'target_market': "Small businesses",
                'monetization_strategy': "Subscription model",
                'technical_requirements': [
                    "Web development",
                    "Database design",
                    "API integration"
                ],
                'market_opportunity_score': 75,
                'development_difficulty': 60,
                'estimated_development_time': "6 months"
            }
            for i in range(5)
        ]
    
    async def get_combined_ideas(
        self, 
        user_id: str,
        analysis_id: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get combined blog and software ideas"""
        try:
            # Get blog ideas from analysis
            blog_ideas = []
            if analysis_id:
                analysis_data = await enhanced_database_service.get_enhanced_ideas(analysis_id, 'blog')
                blog_ideas = analysis_data.get('blog_ideas', [])
            
            # Get software ideas (separate generation)
            software_data = await enhanced_database_service.get_enhanced_ideas(None, 'software')
            software_ideas = software_data.get('software_ideas', [])
            
            return {
                'blog_ideas': blog_ideas,
                'software_ideas': software_ideas
            }
            
        except Exception as e:
            logger.error(f"Error getting combined ideas: {str(e)}")
            return {'blog_ideas': [], 'software_ideas': []}
    
    async def generate_content_for_subtopic(
        self,
        topic_id: str,
        subtopic: str,
        keywords: List[Any],
        affiliate_offers: List[Any],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Generate content ideas (blog + software) for a specific subtopic using Prompt B
        """
        try:
            # Clean keywords - handle mixed types (str or dict)
            clean_keywords = []
            for k in keywords:
                if isinstance(k, str):
                    clean_keywords.append(k)
                elif isinstance(k, dict):
                    # Try common keys or stringify
                    val = k.get('keyword') or k.get('term') or k.get('name') or str(k)
                    clean_keywords.append(str(val))
                else:
                    clean_keywords.append(str(k))
            
            # Clean affiliate offers
            clean_offers = []
            if affiliate_offers:
                for o in affiliate_offers:
                    if isinstance(o, str):
                        clean_offers.append(o)
                    elif isinstance(o, dict):
                        val = o.get('name') or o.get('title') or o.get('offer') or str(o)
                        clean_offers.append(str(val))
                    else:
                        clean_offers.append(str(o))

            # Construct prompt
            prompt = f"""
            ### ROLE
            You are a Senior Content Strategist. Your goal is to create high-performing content blueprints that target specific keyword clusters.

            ### INPUT DATA
            1. SUBTOPIC: {subtopic}
            2. KEYWORDS: {', '.join(clean_keywords)}
            3. AFFILIATE OFFERS: {', '.join(clean_offers) if clean_offers else "None provided (Suggest generic types)"}

            ### TASK
            Generate 5 detailed "Content Blueprints" (Blog Ideas) and 2 Simple Software Application ideas.
            
            CRITICAL REQUIREMENT: 
            For the 'Suggested Outline' field, do NOT just list headers. You must create a "Strategic Blueprint" where each item in the list corresponds to a section, formatted exactly like this:
            
            "H2: [Heading using specific keyword] | Intent: [User Intent/Question] | Keywords: [Specific keywords to include] | Affiliate Hook: [Where to insert product/review]"
            
            Example of a good outline item:
            "H2: White Vinegar & Dish Soap Recipes | Intent: How to make safe weed killer at home | Keywords: vinegar weed killer recipe, dawn dish soap | Affiliate Hook: Review heavy-duty spray bottles"

            ### OUTPUT FORMAT
            You must verify the response is strictly in the following TEXT DELIMITED format. Do not use JSON.

            [BLOG_IDEA]
            Title: <The Title>
            Angle: <Brief explanation of why this ranks>
            Target Affiliate: <Affiliate name>
            Estimated Search Volume: <Numeric Estimate>
            Suggested Outline: <3-5 strategic blueprint bullets separated by | >
            [END]

            [SOFTWARE_IDEA]
            Name: <App Name>
            Description: <What it does>
            Monetization Hook: <How it links to affiliates>
            Estimated Search Volume: <Numeric Estimate>
            [END]
            """

            # Call LLM service
            if not self.llm_service:
                 # Try to get default service if not initialized
                from ..integrations.llm_providers import llm_providers_manager
                self.llm_service = llm_providers_manager.providers.get('openai') or llm_providers_manager.providers.get('deepseek')
            
            if not self.llm_service:
                 raise ValueError("No LLM service available")

            response = await self.llm_service.generate_content(
                prompt=prompt,
                max_tokens=2500,
                temperature=0.7
            )

            if "error" in response:
                raise ValueError(f"LLM error: {response['error']}")

            content = response["content"]
            
            # Parse response
            blog_ideas = []
            software_ideas = []
            
            # Import uuid for ID generation
            import uuid

            # Parse Blog Ideas - Relaxed pattern
            blog_pattern = re.compile(
                r'\[BLOG_IDEA\]\s*'
                r'Title:\s*(.+?)\s*\n' # Relaxed: allow any whitespace after value, then newline
                r'\s*Angle:\s*(.+?)\s*\n'
                r'\s*Target Affiliate:\s*(.+?)\s*\n'
                r'\s*Estimated Search Volume:\s*(.+?)\s*\n'
                r'\s*Suggested Outline:\s*(.+?)\s*'
                r'\[END\]', 
                re.DOTALL | re.IGNORECASE
            )
            blog_matches = blog_pattern.findall(content)
            
            for match in blog_matches:
                title, angle, affiliate, volume_str, outline = match
                
                # Parse volume safely
                try:
                    total_search_volume = int(re.sub(r'[^\d]', '', volume_str))
                except:
                    total_search_volume = 0
                
                # Check description logic
                description = f"Angle: {angle.strip()}. Target Affiliate: {affiliate.strip()}"
                
                # Parse outline bullets
                outline_list = [s.strip() for s in outline.split('|') if s.strip()]

                blog_ideas.append({
                    "id": str(uuid.uuid4()),  # Generate ID immediately
                    "title": title.strip(),
                    "angle": angle.strip(),
                    "target_affiliate": affiliate.strip(),
                    "description": description, # Explicitly setting description for use in DB
                    "content_type": "blog",
                    "subtopic": subtopic,
                    "user_id": user_id,
                    "total_search_volume": total_search_volume,
                    "content_outline": outline_list,
                    "keywords": [str(k) for k in clean_keywords[:5]], # Add keywords to memory object
                    "created_at": datetime.utcnow().isoformat()
                })

            # Parse Software Ideas - Relaxed pattern
            soft_pattern = re.compile(
                r'\[SOFTWARE_IDEA\]\s*'
                r'Name:\s*(.+?)\s*\n'
                r'\s*Description:\s*(.+?)\s*\n'
                r'\s*Monetization Hook:\s*(.+?)\s*\n'
                r'\s*Estimated Search Volume:\s*(.+?)\s*'
                r'\[END\]', 
                re.DOTALL | re.IGNORECASE
            )
            soft_matches = soft_pattern.findall(content)
            
            for match in soft_matches:
                name, desc, hook, volume_str = match
                
                # Parse volume safely
                try:
                    total_search_volume = int(re.sub(r'[^\d]', '', volume_str))
                except:
                    total_search_volume = 0
                
                description = f"{desc.strip()} Monetization Hook: {hook.strip()}"
                
                software_ideas.append({
                    "id": str(uuid.uuid4()), # Generate ID immediately
                    "name": name.strip(),
                    "title": name.strip(), # Ensure title is present for DB map
                    "description": description, # Explicitly setting description
                    "monetization_hook": hook.strip(),
                    "content_type": "software",
                    "subtopic": subtopic,
                    "user_id": user_id,
                    "total_search_volume": total_search_volume,
                    "keywords": [str(k) for k in clean_keywords[:5]], # Add keywords to memory object
                    "created_at": datetime.utcnow().isoformat()
                })

            # Save to content_ideas table
            try:
                # Prepare data for bulk insert
                ideas_to_insert = []
                
                # Calculate metrics from input keywords
                metrics_summary = {}
                avg_vol = 0
                avg_diff = 0
                
                if clean_keywords:
                    # Try to reconstruct metrics if original keywords input had them
                    # The input 'keywords' might be mixed list of dicts/strs
                    # We can iterate the original 'keywords' arg, not just 'clean_keywords'
                    valid_metrics = []
                    for k in keywords:
                        if isinstance(k, dict):
                            valid_metrics.append(k)
                    
                    if valid_metrics:
                        total_vol = sum(float(k.get('search_volume', 0) or k.get('volume', 0)) for k in valid_metrics)
                        total_diff = sum(float(k.get('keyword_difficulty', 0) or k.get('difficulty', 0)) for k in valid_metrics)
                        avg_vol = total_vol / len(valid_metrics)
                        avg_diff = total_diff / len(valid_metrics)
                        
                        metrics_summary = {
                            "avg_search_volume": round(avg_vol, 0),
                            "avg_difficulty": round(avg_diff, 1),
                            "total_keywords": len(valid_metrics)
                        }

                for idea in blog_ideas:
                    # Use parsed volume if available, else usage calculated avg
                    vol = idea.get("total_search_volume", 0) or avg_vol

                    ideas_to_insert.append({
                        "id": idea["id"], 
                        "user_id": user_id,
                        "topic_id": topic_id,
                        "research_id": topic_id,
                        "subtopic": subtopic,
                        "title": idea["title"],
                        "description": idea["description"],
                        "content_type": "blog",
                        "category": "generated",
                        "status": "draft",
                        "keywords": [str(k) for k in clean_keywords[:5]], 
                        "content_outline": idea.get("content_outline", []),
                        "total_search_volume": int(vol),
                        "created_at": datetime.utcnow().isoformat(),
                        # Add missing metrics fields
                        "keyword_metrics": metrics_summary, # Save the calculated summary
                        "seo_score": 75, # Default good score for LLM generated content
                        "difficulty_level": "Medium", # Default
                        "monetization_potential": "High", # Default given intent
                        
                        # New Scores
                        "viral_potential_score": 85 if int(vol) > 1000 else 60,
                        "audience_alignment_score": 85,
                        "content_feasibility_score": 90,
                        "business_impact_score": 80
                    })
                    
                for idea in software_ideas:
                     # For software, 'name' maps to 'title' in DB
                    vol = idea.get("total_search_volume", 0) or avg_vol
                    
                    ideas_to_insert.append({
                        "id": idea["id"], 
                        "user_id": user_id,
                        "topic_id": topic_id,
                        "research_id": topic_id,
                        "subtopic": subtopic,
                        "title": idea["name"], 
                        "description": idea["description"],
                        "content_type": "software",
                        "category": "generated",
                        "status": "draft",
                        "keywords": [str(k) for k in clean_keywords[:5]],
                        "content_outline": [], 
                        "total_search_volume": int(vol),
                        "created_at": datetime.utcnow().isoformat(),
                        # Add missing metrics fields
                        "keyword_metrics": metrics_summary,
                        "seo_score": 0, # Not applicable for software
                        "difficulty_level": "High",
                        "monetization_potential": "Very High",
                        
                        # New Scores
                        "viral_potential_score": 75,
                        "audience_alignment_score": 90,
                        "content_feasibility_score": 60, # Software is harder
                        "business_impact_score": 95
                    })
                
                if ideas_to_insert:
                    # Import supabase client here to avoid circular dependencies if any
                    from ..core.supabase_singleton import get_supabase_client
                    supabase = get_supabase_client()
                    
                    # Insert into content_ideas
                    db_result = supabase.table('content_ideas').insert(ideas_to_insert).execute()
                    
                    logger.info(f"Saved {len(ideas_to_insert)} generated ideas to content_ideas table")
                    
                    # We already generated IDs, so no need to fetch back. 
                    # If DB save succeeds, great. If not, we have IDs in memory so React won't complain.

            except Exception as e:
                logger.error(f"Failed to save generated ideas to database: {str(e)}")
                # Log detailed traceback context if possible, but basic logging is imperative here.
                # Proceed to return the memory-generated ideas so the user at least sees them once
                pass

            return {
                "blog_ideas": blog_ideas,
                "software_ideas": software_ideas
            }

        except Exception as e:
            logger.error(f"Error generating content for subtopic {subtopic}: {str(e)}")
            return {"blog_ideas": [], "software_ideas": []}

    async def generate_ideas_for_idea_burst(
        self, 
        user_id: str,
        analysis_id: Optional[str] = None,
        include_software: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate ideas for Idea Burst page"""
        try:
            result = {'blog_ideas': [], 'software_ideas': []}
            
            # Get blog ideas from analysis if available
            if analysis_id:
                analysis_data = await enhanced_database_service.get_enhanced_ideas(analysis_id, 'blog')
                result['blog_ideas'] = analysis_data.get('blog_ideas', [])
            
            # Generate software ideas separately if requested
            if include_software:
                software_ideas = await self.generate_software_ideas_separately(user_id)
                result['software_ideas'] = software_ideas
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating ideas for Idea Burst: {str(e)}")
            return {'blog_ideas': [], 'software_ideas': []}

# Global instance
enhanced_idea_generator = EnhancedIdeaGenerator()

