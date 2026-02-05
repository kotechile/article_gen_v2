"""
Affiliate Research Service
Handles affiliate program discovery, analysis, and content generation
"""

import asyncio
import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import structlog
from src.core.supabase_database import get_supabase_db
from src.core.redis import cache
from src.core.llm_config import LLMConfigManager
from .web_search_service import WebSearchService
# Import LinkUp API directly to avoid relative import issues
try:
    from src.integrations.linkup_api import linkup_api
except ImportError:
    # Fallback if LinkUp API is not available
    linkup_api = None

# Import Real Affiliate Search Service for web scraping
try:
    from services.real_affiliate_search import RealAffiliateSearchService
except ImportError:
    # Fallback if Real Affiliate Search is not available
    RealAffiliateSearchService = None

logger = structlog.get_logger()

class AffiliateResearchService:
    def __init__(self):
        self.db = get_supabase_db()
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.llm_manager = LLMConfigManager()
    
    def _generate_cache_key(self, search_term: str, niche: Optional[str], budget_range: Optional[str]) -> str:
        """Generate a cache key for the search parameters"""
        cache_data = {
            "search_term": search_term.lower().strip(),
            "niche": niche or "any",
            "budget_range": budget_range or "any"
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        cache_hash = hashlib.md5(cache_string.encode()).hexdigest()
        return f"affiliate_search:{cache_hash}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached search result"""
        try:
            cached_data = cache.get(cache_key)
            if cached_data:
                logger.info("Cache hit for affiliate search", cache_key=cache_key)
                return cached_data
            return None
        except Exception as e:
            logger.warning("Failed to get cached result", cache_key=cache_key, error=str(e))
            return None
    
    def _set_cached_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache search result"""
        try:
            cache.set(cache_key, result, expire=self.cache_ttl)
            logger.info("Cached affiliate search result", cache_key=cache_key, ttl=self.cache_ttl)
        except Exception as e:
            logger.warning("Failed to cache result", cache_key=cache_key, error=str(e))
    
    def clear_search_cache(self, search_term: str, niche: Optional[str] = None, budget_range: Optional[str] = None) -> bool:
        """Clear cache for specific search parameters"""
        try:
            cache_key = self._generate_cache_key(search_term, niche, budget_range)
            cache.delete(cache_key)
            logger.info("Cleared cache for search", cache_key=cache_key)
            return True
        except Exception as e:
            logger.warning("Failed to clear cache", cache_key=cache_key, error=str(e))
            return False
    
    def clear_all_search_cache(self) -> bool:
        """Clear all affiliate search cache"""
        try:
            # Get all cache keys with affiliate_search prefix
            pattern = "affiliate_search:*"
            keys = cache.client.keys(pattern)
            if keys:
                cache.delete(*keys)
                logger.info("Cleared all affiliate search cache", keys_count=len(keys))
            return True
        except Exception as e:
            logger.warning("Failed to clear all search cache", error=str(e))
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            pattern = "affiliate_search:*"
            keys = cache.client.keys(pattern)
            return {
                "total_cached_searches": len(keys),
                "cache_ttl_seconds": self.cache_ttl,
                "cache_ttl_hours": self.cache_ttl / 3600
            }
        except Exception as e:
            logger.warning("Failed to get cache stats", error=str(e))
            return {"error": str(e)}
    
    async def search_affiliate_programs(
        self, 
        search_term: str,
        niche: Optional[str] = None,
        budget_range: Optional[str] = None,
        user_id: Optional[str] = None,
        ignore_cache: bool = False
    ) -> Dict[str, Any]:
        """
        Search for affiliate programs based on search criteria
        """
        # Check cache first (unless ignored)
        cache_key = self._generate_cache_key(search_term, niche, budget_range)
        
        if not ignore_cache:
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.info("Returning cached result", cache_key=cache_key)
                return cached_result
        else:
            logger.info("Ignoring cache for affiliate search", cache_key=cache_key)
        
        try:
            logger.info("Starting affiliate program search", 
                       search_term=search_term, niche=niche, budget_range=budget_range)
            
            # Get LLM provider for analysis
            llm_config = self.llm_manager.get_config()
            if not llm_config:
                raise Exception("No LLM provider available")
            
            # Search for existing programs in Supabase
            try:
                existing_programs = self.db.search_programs(search_term, limit=10)
                programs = existing_programs
                print(f"DEBUG: Supabase search returned {len(programs)} programs")
                logger.info("Found programs in Supabase", 
                           search_term=search_term, 
                           db_programs_found=len(programs))
            except Exception as e:
                logger.warning("Failed to search Supabase programs", error=str(e))
                programs = []
                print(f"DEBUG: Supabase search failed: {e}")
            
            # Filter out fake programs from database
            programs = self._filter_fake_programs_from_db(programs)
            
            logger.info("Found programs in database", 
                       search_term=search_term, 
                       db_programs_found=len(programs))
            
            # Analyze search term with LLM
            print(f"DEBUG: Analyzing search term '{search_term}' with LLM...")
            analysis_result = await self._analyze_search_term(search_term, llm_config)
            print(f"DEBUG: Analysis result category: {analysis_result.get('category')}")
            
            # Extract programs from LLM analysis result and add them to our list
            analysed_programs = analysis_result.get('affiliate_programs', [])
            print(f"DEBUG: Found {len(analysed_programs)} programs in LLM analysis")
            if analysed_programs:
                # Format them correctly if needed
                formatted_analysed = []
                for p in analysed_programs:
                    formatted_analysed.append({
                        "id": f"analysed_{hash(p.get('name', ''))}",
                        "name": p.get('name'),
                        "description": p.get('description'),
                        "commission_rate": p.get('commission'),
                        "network": p.get('network', 'Various'),
                        "link": p.get('link', ''),
                        "source": "llm_analysis"
                    })
                programs.extend(formatted_analysed)
                logger.info("Added programs from LLM analysis", count=len(formatted_analysed))
            
            # If we don't have enough relevant programs, search LinkUp.so
            relevant_programs = [p for p in programs if self._is_relevant_program(p, search_term)]
            print(f"DEBUG: Relevant programs before external search: {len(relevant_programs)}")
            logger.info("Program relevance check", 
                       search_term=search_term, 
                       total_programs=len(programs), 
                       relevant_count=len(relevant_programs))
            
            if len(relevant_programs) < 3:
                print(f"DEBUG: Low relevant programs ({len(relevant_programs)}), starting external searches for '{search_term}'...")
                # Start searches in parallel
                tasks = []
                
                if linkup_api:
                    print("DEBUG: Creating LinkUp search task...")
                    linkup_task = asyncio.create_task(
                        self._safe_linkup_search(search_term, analysis_result.get('category'))
                    )
                    tasks.append(linkup_task)
                else:
                    print("DEBUG: LinkUp API not available, skipping.")
                
                print("DEBUG: Creating Real Affiliate Search task...")
                real_search_task = asyncio.create_task(
                    self._safe_real_affiliate_search(search_term, analysis_result.get('category'))
                )
                tasks.append(real_search_task)
                
                print("DEBUG: Creating LLM search task...")
                llm_task = asyncio.create_task(
                    self._safe_llm_search(search_term, analysis_result.get('category'))
                )
                tasks.append(llm_task)
                
                # Wait for all tasks to finish (or timeout)
                print("DEBUG: Waiting for external search tasks...")
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=60.0  # 60 second timeout for all searches
                )
                print(f"DEBUG: External search tasks completed with {len(results)} results")
                
                # Process results
                for r in results:
                    if isinstance(r, list):
                        print(f"DEBUG: Search task returned {len(r)} programs")
                        programs.extend(r)
                    elif isinstance(r, Exception):
                        print(f"DEBUG: Search task failed with error: {r}")
                        logger.error("Search task failed", error=str(r))
                
                # Deduplicate and consolidate affiliate programs
                print(f"DEBUG: Programs count before deduplication: {len(programs)}")
                programs = self._deduplicate_programs(programs)
                logger.info("Programs deduplicated", total_programs=len(programs))
                
                # If we still have very few or no programs, try the fallback mechanism
                if len(programs) < 2:
                    print(f"DEBUG: Low program count ({len(programs)}), triggering fallback programs for '{search_term}'...")
                    fallback_programs = self._get_fallback_programs(search_term, analysis_result.get('category'))
                    print(f"DEBUG: Fallback returned {len(fallback_programs)} programs")
                    programs.extend(fallback_programs)
                    programs = self._deduplicate_programs(programs)
                
                print(f"DEBUG: Final programs count: {len(programs)}")
                
                # Prioritize quality programs over generic ones
                programs = self._prioritize_quality_programs(programs)
                logger.info("Programs prioritized", total_programs=len(programs))
                
            else:
                # Update usage statistics for found programs
                for program in programs:
                    try:
                        self.db.update_program_usage(program.get('id'))
                    except Exception as e:
                        logger.warning("Failed to update program usage", program_id=program.get('id'), error=str(e))
            
            # Save research to Supabase (skip if no user_id)
            research_record = None
            if user_id:
                try:
                    research_record = self._save_research_record(
                        search_term, niche, budget_range, programs, user_id
                    )
                except Exception as e:
                    logger.warning("Failed to save research record to Supabase", error=str(e))
            
            # Prepare result
            result = {
                "search_term": search_term,
                "niche": niche,
                "budget_range": budget_range,
                "programs": programs,
                "analysis": analysis_result,
                "research_id": str(research_record.id) if research_record else None,
                "total_programs": len(programs),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Cache the result
            self._set_cached_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error("Affiliate research failed", error=str(e))
            raise
    
    async def _analyze_search_term(self, search_term: str, llm_config) -> Dict[str, Any]:
        """
        Analyze search term using LLM-based semantic analysis
        """
        try:
            # Use simplified LLM category detection
            category = await self._detect_category_with_llm_simple(search_term)
            logger.info("LLM category detection", topic=search_term, category=category)
            
            # Generate content using LLM based on detected category
            related_areas, affiliate_programs = await self._generate_content_with_llm_simple(
                search_term, category, 6, 5
            )
            
            # Convert to the expected format
            content_opportunities = [area["area"] for area in related_areas]
            
            return {
                "topic": search_term,
                "category": category,
                "target_audience": "Content creators and marketers",
                "content_opportunities": content_opportunities,
                "affiliate_types": [program["category"] for program in affiliate_programs],
                "competition_level": "Medium",
                "earnings_potential": "$100-1000/month",
                "related_areas": related_areas,
                "affiliate_programs": affiliate_programs
            }
        except Exception as e:
            logger.error("Failed to analyze search term with LLM", error=str(e))
            # Fallback to basic analysis
            return {
                "topic": search_term,
                "category": "General",
                "target_audience": "General consumers",
                "content_opportunities": [
                    f"Best {search_term} products",
                    f"{search_term} reviews and comparisons",
                    f"How to choose {search_term}",
                    f"{search_term} buying guide"
                ],
                "affiliate_types": ["Product reviews", "Comparison guides", "Buying guides"],
                "competition_level": "Medium",
                "earnings_potential": "$100-500/month"
            }
    
    async def _detect_category_with_llm_simple(self, topic: str) -> str:
        """
        Simple LLM category detection without database dependency
        """
        try:
            import openai
            import os
            
            from ..core.api_key_manager import api_key_manager
            
            # Get OpenAI API key from manager (prefers DB, falls back to ENV)
            api_key = api_key_manager.get_openai_key()
            if not api_key:
                logger.warning("No OpenAI API key found, using fallback category detection")
                return self._fallback_category_detection(topic)
            
            # Create OpenAI client
            client = openai.AsyncOpenAI(api_key=api_key)
            
            # Create a prompt for category detection
            prompt = f"""
            Analyze the following topic and determine its primary category from these options:
            
            Categories:
            - outdoor_recreation (camping, hiking, fishing, water sports, RV, outdoor activities)
            - food_cooking (cooking, recipes, food, cuisine, baking, kitchen)
            - technology (software, apps, gadgets, tech products, digital tools)
            - health_fitness (fitness, health, wellness, supplements, medical)
            - education_learning (courses, books, training, academic, educational content)
            - home_garden (home improvement, gardening, DIY, furniture, appliances)
            - travel_hospitality (travel, hotels, vacation, tourism, hospitality)
            - fashion_beauty (clothing, beauty, cosmetics, fashion, accessories)
            - automotive (cars, motorcycles, automotive parts, vehicles)
            - business_services (B2B services, office supplies, business tools)
            - entertainment_gaming (games, entertainment, media, streaming)
            - finance_investing (money, investing, banking, financial services)
            - pets_animals (pet care, animal products, veterinary services)
            - sports_fitness (sports equipment, fitness gear, athletic activities)
            - general (general products, mixed categories, unclear)
            
            Topic: "{topic}"
            
            Respond with ONLY the category name (e.g., "outdoor_recreation") and nothing else.
            """
            
            # Call OpenAI API
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            # Parse response
            content = response.choices[0].message.content.strip().lower()
            logger.info("LLM category detection response", topic=topic, response=content)
            
            # Check if response matches one of our categories
            categories = [
                'outdoor_recreation', 'food_cooking', 'technology', 'health_fitness',
                'education_learning', 'home_garden', 'travel_hospitality', 'fashion_beauty',
                'automotive', 'business_services', 'entertainment_gaming', 'finance_investing',
                'pets_animals', 'sports_fitness', 'general'
            ]
            for category in categories:
                if category in content:
                    return category
            return 'general'
            
        except Exception as e:
            logger.error("LLM detection failed", error=str(e))
            return self._fallback_category_detection(topic)
    
    def _fallback_category_detection(self, topic: str) -> str:
        """
        Fallback category detection using keyword matching
        """
        topic_lower = topic.lower()
        
        if any(word in topic_lower for word in ['outdoor', 'camping', 'hiking', 'fishing', 'lake', 'weekend', 'nature', 'park', 'trail', 'mountain', 'river', 'beach', 'forest']):
            return 'outdoor_recreation'
        elif any(word in topic_lower for word in ['cooking', 'recipe', 'food', 'kitchen', 'baking', 'cuisine', 'meal', 'restaurant', 'coffee', 'pizza', 'maker', 'brewing']):
            return 'food_cooking'
        elif any(word in topic_lower for word in ['tech', 'software', 'app', 'gadget', 'computer', 'digital', 'ai', 'programming']):
            return 'technology'
        elif any(word in topic_lower for word in ['fitness', 'health', 'wellness', 'gym', 'workout', 'exercise', 'medical']):
            return 'health_fitness'
        elif any(word in topic_lower for word in ['course', 'book', 'learning', 'education', 'training', 'academic', 'study']):
            return 'education_learning'
        elif any(word in topic_lower for word in ['home', 'garden', 'diy', 'furniture', 'appliance', 'renovation']):
            return 'home_garden'
        elif any(word in topic_lower for word in ['travel', 'hotel', 'vacation', 'tourism', 'trip', 'flight']):
            return 'travel_hospitality'
        elif any(word in topic_lower for word in ['fashion', 'beauty', 'clothing', 'cosmetics', 'style', 'makeup']):
            return 'fashion_beauty'
        elif any(word in topic_lower for word in ['car', 'automotive', 'vehicle', 'motorcycle', 'auto']):
            return 'automotive'
        elif any(word in topic_lower for word in ['business', 'office', 'service', 'b2b', 'corporate']):
            return 'business_services'
        elif any(word in topic_lower for word in ['game', 'entertainment', 'gaming', 'media', 'streaming']):
            return 'entertainment_gaming'
        elif any(word in topic_lower for word in ['finance', 'money', 'investment', 'banking', 'crypto']):
            return 'finance_investing'
        elif any(word in topic_lower for word in ['pet', 'animal', 'dog', 'cat', 'veterinary']):
            return 'pets_animals'
        elif any(word in topic_lower for word in ['sport', 'fitness', 'athletic', 'equipment', 'gear']):
            return 'sports_fitness'
        else:
            return 'general'
    
    async def _generate_content_with_llm_simple(self, topic: str, category: str, max_areas: int, max_programs: int):
        """
        Simple LLM content generation without database dependency
        """
        try:
            import openai
            import os
            import json
            
            from ..core.api_key_manager import api_key_manager
            
            # Get OpenAI API key from manager
            api_key = api_key_manager.get_openai_key()
            if not api_key:
                logger.warning("No OpenAI API key found, using fallback content generation")
                return self._generate_fallback_content(topic, category, max_areas, max_programs)
            
            # Create OpenAI client
            client = openai.AsyncOpenAI(api_key=api_key)
            
            # Create a comprehensive prompt for content generation
            prompt = f"""
                You are an expert affiliate marketing analyst. Generate relevant subtopics and affiliate programs for the topic: "{topic}"
                
                Category: {category}
                
                Generate {max_areas} related subtopics/areas that would be valuable for content creators and marketers.
                Generate {max_programs} relevant affiliate programs with realistic details.
                
                For each subtopic, provide:
                - A specific, actionable area name
                - A brief description
                - A relevance score (0.0-1.0)
                
                For each affiliate program, provide:
                - A realistic program name (use well-known brands like REI, Patagonia, Amazon, etc.)
                - Commission rate (realistic percentage)
                - Category
                - Difficulty level (Easy/Medium/Hard)
                - Description
                - Affiliate link (use ONLY real, well-known affiliate program URLs like amazon.com, rei.com, etc.)
                - Estimated traffic
                - Competition level (Low/Medium/High)
                
                IMPORTANT: Only suggest real, well-known affiliate programs with legitimate URLs. Do not create fake programs or URLs.
                
                Focus on the category: {category}
                
                Return your response as a JSON object with this structure:
                {{
                    "related_areas": [
                        {{"area": "Area Name", "description": "Description", "relevance_score": 0.9}}
                    ],
                    "affiliate_programs": [
                        {{"name": "Program Name", "commission": "5-15%", "category": "Category", "difficulty": "Easy", "description": "Description", "link": "https://real-affiliate-program.com", "estimated_traffic": 10000, "competition_level": "Medium"}}
                    ]
                }}
                """
            
            # Call OpenAI API
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            # Parse LLM response
            content = response.choices[0].message.content.strip()
            logger.info("LLM content generation response", topic=topic, category=category)
            
            try:
                # Try to extract JSON from the response
                if '{' in content and '}' in content:
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    json_str = content[json_start:json_end]
                    parsed = json.loads(json_str)
                    
                    related_areas = parsed.get('related_areas', [])[:max_areas]
                    affiliate_programs = parsed.get('affiliate_programs', [])[:max_programs]
                    
                    # Filter out fake programs
                    affiliate_programs = self._filter_fake_programs(affiliate_programs)
                    
                    return related_areas, affiliate_programs
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM JSON response, using fallback")
            
            # Fallback to category-specific content
            return self._generate_fallback_content(topic, category, max_areas, max_programs)
            
        except Exception as e:
            logger.error("LLM content generation failed", error=str(e))
            return self._generate_fallback_content(topic, category, max_areas, max_programs)
    
    def _generate_fallback_content(self, topic: str, category: str, max_areas: int, max_programs: int):
        """
        Fallback content generation using hardcoded rules
        """
        if category == 'outdoor_recreation':
            related_areas = [
                {"area": f"{topic} Equipment", "description": f"Essential gear and equipment for {topic}", "relevance_score": 0.95},
                {"area": f"{topic} Destinations", "description": f"Best locations and spots for {topic}", "relevance_score": 0.9},
                {"area": f"{topic} Safety", "description": f"Safety tips and guidelines for {topic}", "relevance_score": 0.9},
                {"area": f"{topic} Planning", "description": f"Planning guides and checklists for {topic}", "relevance_score": 0.85},
                {"area": f"{topic} Activities", "description": f"Activities and things to do for {topic}", "relevance_score": 0.8},
                {"area": f"{topic} Community", "description": f"Online communities and forums for {topic}", "relevance_score": 0.7}
            ]
            affiliate_programs = [
                {"name": "REI Co-op Affiliate", "commission": "5-8%", "category": "Outdoor Gear", "difficulty": "Easy", "description": "Outdoor gear and clothing cooperative", "link": "https://www.rei.com/affiliate-program", "estimated_traffic": 12000, "competition_level": "Medium"},
                {"name": "Patagonia Affiliate", "commission": "3-8%", "category": "Outdoor Gear", "difficulty": "Medium", "description": "Sustainable outdoor clothing and gear", "link": "https://www.patagonia.com/affiliate-program", "estimated_traffic": 8000, "competition_level": "Medium"},
                {"name": "Backcountry.com Affiliate", "commission": "4-6%", "category": "Outdoor Gear", "difficulty": "Easy", "description": "Outdoor gear and equipment", "link": "https://www.backcountry.com/affiliate-program", "estimated_traffic": 6000, "competition_level": "Medium"},
                {"name": "Camping World Affiliate", "commission": "3-7%", "category": "RV & Camping", "difficulty": "Easy", "description": "RV and camping supplies", "link": "https://www.campingworld.com/affiliate-program", "estimated_traffic": 5000, "competition_level": "Low"},
                {"name": "Bass Pro Shops Affiliate", "commission": "4-8%", "category": "Fishing & Hunting", "difficulty": "Easy", "description": "Fishing and hunting equipment", "link": "https://www.basspro.com/affiliate-program", "estimated_traffic": 8000, "competition_level": "Medium"}
            ]
        else:
            # Generic fallback
            related_areas = [
                {"area": f"{topic} Reviews", "description": f"Product and service reviews for {topic}", "relevance_score": 0.9},
                {"area": f"{topic} Guides", "description": f"Comprehensive guides and tutorials", "relevance_score": 0.85},
                {"area": f"{topic} Equipment", "description": f"Essential equipment and tools", "relevance_score": 0.8},
                {"area": f"{topic} Services", "description": f"Professional services and consultations", "relevance_score": 0.75},
                {"area": f"{topic} Community", "description": f"Online communities and forums", "relevance_score": 0.7},
                {"area": f"{topic} Resources", "description": f"Educational resources and materials", "relevance_score": 0.65}
            ]
            affiliate_programs = [
                {"name": "Amazon Associates", "commission": "1-10%", "category": "General", "difficulty": "Easy", "description": "Wide range of products", "link": "https://affiliate-program.amazon.com/", "estimated_traffic": 50000, "competition_level": "High"},
                {"name": "ShareASale", "commission": "5-15%", "category": "Various", "difficulty": "Medium", "description": "Diverse merchant network", "link": "https://www.shareasale.com/", "estimated_traffic": 10000, "competition_level": "Medium"},
                {"name": "ClickBank", "commission": "20-75%", "category": "Digital Products", "difficulty": "Easy", "description": "Digital products and courses", "link": "https://www.clickbank.com/affiliate-program/", "estimated_traffic": 8000, "competition_level": "Medium"}
            ]
        
        return related_areas[:max_areas], affiliate_programs[:max_programs]
    
    def _filter_fake_programs_from_db(self, programs: list) -> list:
        """
        Filter out fake affiliate programs from database results
        """
        if not programs:
            return programs
            
        # Patterns that indicate fake programs (more comprehensive)
        fake_patterns = [
            'pro program', 'premium program', 'elite program',
            'solutions', 'marketplace', 'network', 'specialists', 
            'experts', 'gear co', 'emporium', 'llc', 'adventures',
            'affiliate network', 'comprehensive marketplace',
            'advanced affiliate program', 'professional affiliate program'
        ]
        
        # Check for programs that are clearly generated (contain the exact search term)
        filtered_programs = []
        
        for program in programs:
            if not isinstance(program, dict):
                continue
                
            name = program.get('name', '').lower()
            description = program.get('description', '').lower()
            source = program.get('source', '')
            
            # Bypass filtering for LLM-analysed programs or verified programs
            if source in ['llm_analysis', 'llm_identified'] or program.get('is_verified'):
                filtered_programs.append(program)
                continue
            if search_term_lower and search_term_lower in name:
                logger.info("Filtered out generated program", name=program.get('name'), search_term=search_term_lower)
                continue
            
            # Check if it matches fake patterns
            is_fake = any(pattern in name for pattern in fake_patterns)
            
            # Check if description contains fake patterns
            is_fake_description = any(pattern in description for pattern in fake_patterns)
            
            # Skip if it's from web_search source and looks fake
            if source == 'web_search' and (is_fake or is_fake_description):
                logger.info("Filtered out fake web search program", name=program.get('name'), source=source)
                continue
            
            filtered_programs.append(program)
        
        return filtered_programs

    def _filter_fake_programs(self, programs: list) -> list:
        """
        Filter out fake affiliate programs based on common patterns
        """
        if not programs:
            return programs
            
        # List of known legitimate affiliate programs and their domains
        legitimate_domains = {
            'amazon.com', 'affiliate-program.amazon.com',
            'rei.com', 'patagonia.com', 'backcountry.com',
            'campingworld.com', 'basspro.com', 'cabelas.com',
            'shareasale.com', 'clickbank.com', 'cj.com',
            'impact.com', 'awin.com', 'partnerize.com',
            'tradedoubler.com', 'webgains.com', 'zanox.com',
            'tradedoubler.com', 'webgains.com', 'zanox.com',
            'linkshare.com', 'rakuten.com', 'viglink.com',
            'skimlinks.com', 'monetizemore.com', 'flexoffers.com'
        }
        
        # Patterns that indicate fake programs
        fake_patterns = [
            'gear co', 'emporium', 'llc', 'adventures', 'solutions',
            'marketplace', 'network', 'specialists', 'experts',
            'pro program', 'premium program', 'elite program'
        ]
        
        filtered_programs = []
        
        for program in programs:
            if not isinstance(program, dict):
                continue
                
            name = program.get('name', '').lower()
            link = program.get('link', '').lower()
            
            # Check if it's a legitimate domain
            is_legitimate = any(domain in link for domain in legitimate_domains)
            
            # Check if it matches fake patterns
            is_fake = any(pattern in name for pattern in fake_patterns)
            
            # Check for obviously fake URLs (made-up domains)
            has_fake_url = any(fake_domain in link for fake_domain in [
                'outdooradventuregearco.com', 'lakefrontpropertyrentals.com',
                'fishinggearemporium.com', 'weekendbythelake.com',
                'adventuregearco.com', 'propertyrentals.com'
            ])
            
            # Include if legitimate or doesn't match fake patterns
            if is_legitimate or (not is_fake and not has_fake_url):
                filtered_programs.append(program)
            else:
                logger.info("Filtered out fake program", name=program.get('name'), link=program.get('link'))
        
        return filtered_programs
    
    def _generate_intelligent_content(self, topic: str, topic_lower: str, max_areas: int, max_programs: int):
        """
        Generate intelligent subtopics and affiliate programs for any topic
        """
        try:
            # Generate specific, intelligent subtopics based on the topic
            if any(word in topic_lower for word in ['quantum', 'physics', 'science', 'research', 'theoretical', 'academic', 'study', 'university', 'college', 'scholar', 'professor', 'lecture', 'textbook', 'knowledge', 'intellectual', 'scholarly']):
                # Education/Science topics
                related_areas = [
                    {"area": f"{topic} Courses", "description": f"Online courses and educational content for {topic}", "relevance_score": 0.95},
                    {"area": f"{topic} Research", "description": f"Research papers, studies, and academic resources", "relevance_score": 0.9},
                    {"area": f"{topic} Books", "description": f"Textbooks, reference materials, and educational books", "relevance_score": 0.85},
                    {"area": f"{topic} Software", "description": f"Software tools and applications for {topic}", "relevance_score": 0.8},
                    {"area": f"{topic} Equipment", "description": f"Laboratory equipment and scientific instruments", "relevance_score": 0.75},
                    {"area": f"{topic} Conferences", "description": f"Academic conferences and professional events", "relevance_score": 0.7}
                ]
                affiliate_programs = [
                    {"name": "Coursera Affiliate", "commission": "15-45%", "category": "Online Education", "difficulty": "Easy", "description": "Online courses from top universities", "link": "https://www.coursera.org/affiliate-program", "estimated_traffic": 15000, "competition_level": "High"},
                    {"name": "Udemy Affiliate", "commission": "50%", "category": "Online Courses", "difficulty": "Easy", "description": "Online learning platform with diverse courses", "link": "https://www.udemy.com/affiliate-program/", "estimated_traffic": 12000, "competition_level": "Medium"},
                    {"name": "Amazon Associates", "commission": "1-10%", "category": "Books & Media", "difficulty": "Easy", "description": "Educational books and reference materials", "link": "https://affiliate-program.amazon.com/", "estimated_traffic": 50000, "competition_level": "High"}
                ]
            elif any(word in topic_lower for word in ['extreme', 'sport', 'adventure', 'athletic', 'fitness', 'outdoor', 'climbing', 'skydiving', 'bungee', 'paragliding', 'base jumping', 'mountain', 'rock']):
                # Extreme sports topics
                related_areas = [
                    {"area": f"{topic} Equipment", "description": f"Specialized gear and safety equipment for {topic}", "relevance_score": 0.95},
                    {"area": f"{topic} Training", "description": f"Training programs and skill development", "relevance_score": 0.9},
                    {"area": f"{topic} Safety", "description": f"Safety guidelines and risk management", "relevance_score": 0.9},
                    {"area": f"{topic} Insurance", "description": f"Specialized insurance for extreme sports", "relevance_score": 0.8},
                    {"area": f"{topic} Destinations", "description": f"Best locations and venues for {topic}", "relevance_score": 0.85},
                    {"area": f"{topic} Community", "description": f"Online communities and forums", "relevance_score": 0.7}
                ]
                affiliate_programs = [
                    {"name": "REI Co-op Affiliate", "commission": "5-8%", "category": "Outdoor Gear", "difficulty": "Easy", "description": "Outdoor gear and clothing cooperative", "link": "https://www.rei.com/affiliate-program", "estimated_traffic": 12000, "competition_level": "Medium"},
                    {"name": "Patagonia Affiliate", "commission": "3-8%", "category": "Outdoor Gear", "difficulty": "Medium", "description": "Sustainable outdoor clothing and gear", "link": "https://www.patagonia.com/affiliate-program", "estimated_traffic": 8000, "competition_level": "Medium"},
                    {"name": "Backcountry.com Affiliate", "commission": "4-6%", "category": "Outdoor Gear", "difficulty": "Easy", "description": "Outdoor gear and equipment", "link": "https://www.backcountry.com/affiliate-program", "estimated_traffic": 6000, "competition_level": "Medium"}
                ]
            elif any(word in topic_lower for word in ['cookie', 'cookies', 'baking', 'recipe', 'cooking', 'cuisine', 'food', 'meal', 'kitchen', 'chef', 'restaurant', 'dining', 'ingredient', 'spice', 'wine', 'coffee', 'diet', 'nutrition', 'homemade', 'home made']):
                # Food topics
                related_areas = [
                    {"area": f"{topic} Recipes", "description": f"Recipe collections and cooking guides for {topic}", "relevance_score": 0.95},
                    {"area": f"{topic} Ingredients", "description": f"Ingredient guides and shopping tips", "relevance_score": 0.9},
                    {"area": f"{topic} Equipment", "description": f"Kitchen equipment and cooking tools", "relevance_score": 0.85},
                    {"area": f"{topic} Techniques", "description": f"Cooking techniques and methods", "relevance_score": 0.8},
                    {"area": f"{topic} Reviews", "description": f"Product reviews and recommendations", "relevance_score": 0.75},
                    {"area": f"{topic} Classes", "description": f"Cooking classes and tutorials", "relevance_score": 0.7}
                ]
                affiliate_programs = [
                    {"name": "Blue Apron Affiliate", "commission": "15-25%", "category": "Meal Kits", "difficulty": "Medium", "description": "Meal kit delivery service with fresh ingredients", "link": "https://www.blueapron.com/affiliate-program", "estimated_traffic": 20000, "competition_level": "High"},
                    {"name": "HelloFresh Affiliate", "commission": "15-25%", "category": "Meal Kits", "difficulty": "Medium", "description": "Meal kit delivery with easy-to-follow recipes", "link": "https://www.hellofresh.com/affiliate-program", "estimated_traffic": 18000, "competition_level": "High"},
                    {"name": "Williams Sonoma Affiliate", "commission": "4-8%", "category": "Kitchen", "difficulty": "Medium", "description": "High-end kitchen equipment and cookware", "link": "https://www.williams-sonoma.com/affiliate-program", "estimated_traffic": 15000, "competition_level": "Medium"}
                ]
            else:
                # Generic intelligent fallback for any other topic
                related_areas = [
                    {"area": f"{topic} Reviews", "description": f"Product and service reviews for {topic}", "relevance_score": 0.9},
                    {"area": f"{topic} Guides", "description": f"Comprehensive guides and tutorials", "relevance_score": 0.85},
                    {"area": f"{topic} Equipment", "description": f"Essential equipment and tools", "relevance_score": 0.8},
                    {"area": f"{topic} Services", "description": f"Professional services and consultations", "relevance_score": 0.75},
                    {"area": f"{topic} Community", "description": f"Online communities and forums", "relevance_score": 0.7},
                    {"area": f"{topic} Resources", "description": f"Educational resources and materials", "relevance_score": 0.65}
                ]
                affiliate_programs = [
                    {"name": "Amazon Associates", "commission": "1-10%", "category": "General", "difficulty": "Easy", "description": "Wide range of products", "link": "https://affiliate-program.amazon.com/", "estimated_traffic": 50000, "competition_level": "High"},
                    {"name": "ShareASale", "commission": "5-15%", "category": "Various", "difficulty": "Medium", "description": "Diverse merchant network", "link": "https://www.shareasale.com/", "estimated_traffic": 10000, "competition_level": "Medium"},
                    {"name": "ClickBank", "commission": "20-75%", "category": "Digital Products", "difficulty": "Easy", "description": "Digital products and courses", "link": "https://www.clickbank.com/affiliate-program/", "estimated_traffic": 8000, "competition_level": "Medium"}
                ]
            
            return related_areas[:max_areas], affiliate_programs[:max_programs]
            
        except Exception as e:
            logger.error("Failed to generate intelligent content", error=str(e))
            # Fallback to basic content
            related_areas = [
                {"area": f"{topic} Reviews", "description": f"Product and service reviews for {topic}", "relevance_score": 0.9},
                {"area": f"{topic} Guides", "description": f"Comprehensive guides and tutorials", "relevance_score": 0.85},
                {"area": f"{topic} Equipment", "description": f"Essential equipment and tools", "relevance_score": 0.8},
                {"area": f"{topic} Services", "description": f"Professional services and consultations", "relevance_score": 0.75}
            ]
            affiliate_programs = [
                {"name": "Amazon Associates", "commission": "1-10%", "category": "General", "difficulty": "Easy", "description": "Wide range of products", "link": "https://affiliate-program.amazon.com/", "estimated_traffic": 50000, "competition_level": "High"},
                {"name": "ShareASale", "commission": "5-15%", "category": "Various", "difficulty": "Medium", "description": "Diverse merchant network", "link": "https://www.shareasale.com/", "estimated_traffic": 10000, "competition_level": "Medium"}
            ]
            return related_areas[:max_areas], affiliate_programs[:max_programs]
    
    # Hardcoded programs removed - using LinkUp.so and web search for better relevance
    
    async def _safe_linkup_search(self, search_term: str, category: str = None) -> List[Dict[str, Any]]:
        """
        Safely search LinkUp.so with timeout handling
        """
        try:
            return await asyncio.wait_for(
                self._search_linkup_offers(search_term, category),
                timeout=30.0  # 30 second timeout for LinkUp search
            )
        except asyncio.TimeoutError:
            logger.warning("LinkUp.so search timed out", search_term=search_term)
            return []
        except Exception as e:
            logger.warning("LinkUp.so search failed", search_term=search_term, error=str(e))
            return []
    
    async def _safe_real_affiliate_search(self, search_term: str, category: str = None) -> List[Dict[str, Any]]:
        """
        Safely search using Real Affiliate Search Service with timeout handling
        """
        if not RealAffiliateSearchService:
            logger.warning("Real Affiliate Search Service not available, skipping search")
            return []
            
        try:
            return await asyncio.wait_for(
                self._search_real_affiliate_programs(search_term, category),
                timeout=30.0  # 30 second timeout for real affiliate search
            )
        except asyncio.TimeoutError:
            logger.warning("Real affiliate search timed out", search_term=search_term)
            return []
        except Exception as e:
            logger.warning("Real affiliate search failed", search_term=search_term, error=str(e))
            return []
    
    async def _safe_llm_search(self, search_term: str, category: str = None) -> List[Dict[str, Any]]:
        """
        Safely search using LLM with timeout handling
        """
        try:
            return await asyncio.wait_for(
                self._get_llm_identified_programs(search_term, category, []),
                timeout=45.0  # 45 second timeout for LLM search
            )
        except asyncio.TimeoutError:
            logger.warning("LLM search timed out", search_term=search_term)
            return []
        except Exception as e:
            logger.warning("LLM search failed", search_term=search_term, error=str(e))
            return []

    async def _search_linkup_offers(self, search_term: str, category: str = None) -> List[Dict[str, Any]]:
        """
        Search LinkUp.so for real-time affiliate offers
        """
        if not linkup_api:
            logger.warning("LinkUp API not available, skipping search")
            return []
            
        try:
            # Map our category to LinkUp category if needed
            linkup_category = self._map_category_to_linkup(category) if category else None
            
            # Search LinkUp.so for offers
            offers = await linkup_api.search_offers(
                query=search_term,
                category=linkup_category,
                limit=10
            )
            
            logger.info("LinkUp.so search completed", 
                       search_term=search_term, 
                       category=linkup_category,
                       offers_found=len(offers))
            
            return offers
            
        except Exception as e:
            logger.error("LinkUp.so search failed", 
                        search_term=search_term, 
                        error=str(e))
            return []
    
    async def _search_real_affiliate_programs(self, search_term: str, category: str = None) -> List[Dict[str, Any]]:
        """
        Search for real affiliate programs using web scraping
        """
        try:
            async with RealAffiliateSearchService() as search_service:
                programs = await search_service.search_affiliate_programs(search_term, category or "general")
                
                # Convert to our format
                formatted_programs = []
                for program in programs:
                    formatted_programs.append({
                        "id": program.get("id", f"real_{hash(program.get('name', ''))}"),
                        "name": program.get("name", "Unknown Program"),
                        "description": program.get("description", "No description available"),
                        "commission_rate": program.get("commission_rate", "Unknown"),
                        "network": program.get("network", "Unknown"),
                        "epc": program.get("epc", "0.00"),
                        "link": program.get("link", "#"),
                        "source": "real_search"
                    })
                
                logger.info("Real affiliate search completed", 
                           search_term=search_term, 
                           programs_found=len(formatted_programs))
                
                return formatted_programs
                
        except Exception as e:
            logger.error("Real affiliate search failed", 
                        search_term=search_term, 
                        error=str(e))
            return []
    
    def _map_category_to_linkup(self, category: str) -> str:
        """
        Map our internal categories to LinkUp.so categories
        """
        category_mapping = {
            'outdoor_recreation': 'outdoor',
            'food_cooking': 'food',
            'technology': 'technology',
            'health_fitness': 'health',
            'education_learning': 'education',
            'home_garden': 'home',
            'travel_hospitality': 'travel',
            'fashion_beauty': 'fashion',
            'automotive': 'automotive',
            'business_services': 'business',
            'entertainment_gaming': 'entertainment',
            'finance_investing': 'finance',
            'pets_animals': 'pets',
            'sports_fitness': 'sports',
            'general': None
        }
        return category_mapping.get(category, None)
    
    async def _search_web_for_programs(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search the web for affiliate programs when database results are insufficient
        """
        try:
            async with WebSearchService() as web_search:
                web_programs = await web_search.search_affiliate_programs(search_term, max_results=5)
                logger.info("Web search found programs", 
                           search_term=search_term, 
                           programs_found=len(web_programs))
                return web_programs
        except Exception as e:
            logger.error("Web search failed", search_term=search_term, error=str(e))
            return []
    
    def _is_relevant_program(self, program: Dict[str, Any], search_term: str) -> bool:
        """Check if a program is relevant to the search term - STRICT matching"""
        search_lower = search_term.lower()
        
        # Check if search term appears in program details
        program_text = f"{program.get('name', '')} {program.get('description', '')} {program.get('category', '')}"
        program_text_lower = program_text.lower()
        
        # Relaxed matching: check for individual important keywords if search term doesn't match directly
        important_keywords = [w for w in search_lower.split() if len(w) > 3 and w not in ['the', 'and', 'for', 'with', 'about', 'from']]
        if important_keywords:
            match_count = sum(1 for kw in important_keywords if kw in program_text_lower)
            # If we match at least 1 keyword and it's from a high-quality source, allow it
            if match_count >= 1 and (program.get('source') in ['llm_analysis', 'llm_identified'] or program.get('is_verified')):
                return True
            # If we match at least 2 keywords, allow it regardless of source
            if match_count >= 2:
                return True

        # If it's an LLM-identified program, we trust it more
        if program.get('source') in ['llm_analysis', 'llm_identified']:
            # Allow LLM programs if they have AT LEAST some overlap or categories are related
            return True
            
        return False
    
    def _is_quality_program(self, program: Dict[str, Any]) -> bool:
        """
        Check if a LinkUp.so program is high quality (strict filtering)
        """
        name = program.get('name', '').lower()
        description = program.get('description', '')
        commission = program.get('commission_rate', '')
        link = program.get('link', '')
        
        # Filter out generic results
        generic_patterns = [
            'affiliate program from https://',
            'program from https://',
            'found at https://',
            'amazon associates',
            'amazon.com',
            'bestbuy.com',
            'amazon product',
            'amazon telescope',
            'amazon astronomy'
        ]
        
        if any(pattern in name.lower() or pattern in link.lower() for pattern in generic_patterns):
            return False
        
        # Must have meaningful description
        if len(description) < 30:
            return False
        
        # Must have commission information
        if commission in ['N/A', None, 'Unknown', '']:
            return False
        
        # Must be a real affiliate program, not just a product link
        if 'affiliate program' not in name and 'commission' not in description.lower():
            return False
        
        return True
    
    def _get_fallback_programs(self, search_term: str, category: str = None) -> List[Dict[str, Any]]:
        """
        Get fallback affiliate programs for common categories
        """
        fallback_programs = []
        
        # Category-specific affiliate programs
        if category == 'outdoor_recreation' or 'telescope' in search_term.lower():
            fallback_programs = [
                {
                    "id": "celestron-affiliate",
                    "name": "Celestron Telescope Affiliate Program",
                    "description": "Earn commissions promoting Celestron telescopes, accessories, and astronomy equipment. High-quality products for amateur and professional astronomers.",
                    "commission_rate": "8-12%",
                    "network": "ShareASale",
                    "epc": "4.50",
                    "link": "https://www.shareasale.com/join/celestron",
                    "source": "fallback"
                },
                {
                    "id": "orion-telescopes-affiliate",
                    "name": "Orion Telescopes Affiliate Program",
                    "description": "Promote Orion telescopes and astronomy accessories. Known for quality telescopes and excellent customer service.",
                    "commission_rate": "6-10%",
                    "network": "Impact",
                    "epc": "3.20",
                    "link": "https://impact.com/orion-telescopes",
                    "source": "fallback"
                },
                {
                    "id": "meade-instruments-affiliate",
                    "name": "Meade Instruments Affiliate Program",
                    "description": "Promote Meade telescopes and astronomy equipment. Premium telescopes for serious astronomers.",
                    "commission_rate": "7-11%",
                    "network": "ShareASale",
                    "epc": "3.80",
                    "link": "https://www.shareasale.com/join/meade",
                    "source": "fallback"
                },
                {
                    "id": "sky-watcher-affiliate",
                    "name": "Sky-Watcher Telescope Affiliate Program",
                    "description": "High-quality telescopes and mounts for astronomy enthusiasts. Excellent commission rates and product quality.",
                    "commission_rate": "9-13%",
                    "network": "Impact",
                    "epc": "4.20",
                    "link": "https://impact.com/sky-watcher",
                    "source": "fallback"
                },
                {
                    "id": "amazon-astronomy-affiliate",
                    "name": "Amazon Associates Program",
                    "description": "Wide selection of telescopes, astronomy books, and accessories. High conversion rates and reliable payouts.",
                    "commission_rate": "4-8%",
                    "network": "Amazon Associates",
                    "epc": "2.80",
                    "link": "https://affiliate-program.amazon.com/",
                    "source": "fallback"
                },
                {
                    "id": "sky-telescope-affiliate",
                    "name": "Sky & Telescope Magazine Affiliate",
                    "description": "Promote astronomy magazines, books, and educational materials. Great for astronomy enthusiasts and educators.",
                    "commission_rate": "5-8%",
                    "network": "ShareASale",
                    "epc": "2.50",
                    "link": "https://www.shareasale.com/join/sky-telescope",
                    "source": "fallback"
                }
            ]
        elif category == 'food_cooking' or 'coffee' in search_term.lower():
            fallback_programs = [
                {
                    "id": "blue-bottle-affiliate",
                    "name": "Blue Bottle Coffee Affiliate Program",
                    "description": "Premium artisan coffee beans and brewing equipment. High-quality products for coffee enthusiasts.",
                    "commission_rate": "5-8%",
                    "network": "ShareASale",
                    "epc": "3.50",
                    "link": "https://bluebottlecoffee.com/affiliate",
                    "source": "fallback"
                },
                {
                    "id": "chemex-affiliate",
                    "name": "Chemex Coffee Affiliate Program",
                    "description": "Classic pour-over coffee makers and filters. Iconic design and excellent brewing results.",
                    "commission_rate": "7-10%",
                    "network": "Impact",
                    "epc": "4.20",
                    "link": "https://chemexcoffeemaker.com/affiliate",
                    "source": "fallback"
                }
            ]
        else:
            # Generic fallback programs
            fallback_programs = [
                {
                    "id": "shareasale-generic",
                    "name": f"{search_term.title()} Affiliate Programs",
                    "description": f"Discover affiliate programs related to {search_term}. Multiple merchants and commission structures available.",
                    "commission_rate": "5-15%",
                    "network": "ShareASale",
                    "epc": "2.50",
                    "link": "https://www.shareasale.com",
                    "source": "fallback"
                },
                {
                    "id": "impact-generic",
                    "name": f"Premium {search_term.title()} Programs",
                    "description": f"High-paying affiliate programs for {search_term} products and services.",
                    "commission_rate": "8-20%",
                    "network": "Impact",
                    "epc": "3.80",
                    "link": "https://impact.com",
                    "source": "fallback"
                }
            ]
        
        return fallback_programs
    
    async def _get_llm_identified_programs(self, search_term: str, category: str = None, subtopics: List[str] = None) -> List[Dict[str, Any]]:
        """
        Use LLM to identify specific companies and their affiliate programs
        """
        try:
            from ..integrations.llm_providers import generate_content
            
            # Create a comprehensive prompt that includes main topic + subtopics
            subtopics_text = ""
            if subtopics:
                subtopics_text = f"\n\nRelated subtopics to consider: {', '.join(subtopics[:3])}"  # Limit to 3 most relevant
            
            # Create a focused prompt to identify specific companies
            prompt = f"""
            For the topic "{search_term}" in the {category or 'general'} category{subtopics_text}, identify 8-12 REAL companies that actually offer affiliate programs. 
            
            IMPORTANT: Only include companies that you KNOW have active affiliate programs. Do not make up affiliate programs.
            
            Focus on:
            1. Specialized retailers with known affiliate programs
            2. Direct manufacturers with affiliate programs
            3. Industry-specific companies with affiliate programs
            
            For telescopes/astronomy, include companies like:
            - Agena Astro, OPT Telescopes, High Point Scientific (specialized retailers)
            - Unistellar, Orion Telescopes (manufacturers)
            - Leupold, Bushnell (optics companies)
            - Celestron, Meade, Sky-Watcher (telescope manufacturers)
            
            Consider these major affiliate networks when identifying programs:
            - Shopify Collabs
            - AWIN
            - CJ Affiliate
            - ClickBank
            - FlexOffers
            - Avangate Affiliate Network
            - Rakuten Advertising
            - Impact
            - affiliaXe
            - GiddyUp
            - ShareASale
            - Amazon Associates
            
            Return ONLY a JSON array of company objects with this structure:
            [
                {{
                    "name": "Company Name",
                    "description": "Brief description of what they sell",
                    "affiliate_program": "Name of their affiliate program",
                    "commission_rate": "X-Y%",
                    "network": "Affiliate network name",
                    "website": "company website",
                    "category": "retailer/manufacturer/optics"
                }}
            ]
            
            Be conservative - only include companies you're confident have real affiliate programs.
            """
            
            llm_result = await generate_content(
                prompt=prompt,
                provider="openai",
                max_tokens=1500,
                temperature=0.3
            )
            
            if "error" in llm_result:
                logger.warning("LLM identification failed", error=llm_result.get("error"))
                return []
            
            content = llm_result.get("content", "")
            logger.info("LLM identified companies", content_length=len(content))
            
            # Parse the JSON response
            import json
            try:
                # Extract JSON from the response
                json_start = content.find('[')
                json_end = content.rfind(']') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    companies = json.loads(json_str)
                    
                    # Convert to affiliate program format
                    programs = []
                    for company in companies:
                        if isinstance(company, dict) and 'name' in company:
                            programs.append({
                                "id": f"llm-{company['name'].lower().replace(' ', '-')}",
                                "name": f"{company['name']} Affiliate Program",
                                "description": f"Promote {company['name']} - {company.get('description', 'Quality products and services')}. {company.get('affiliate_program', 'Affiliate program')} with competitive commission rates.",
                                "commission_rate": company.get('commission_rate', '5-10%'),
                                "network": company.get('network', 'Direct'),
                                "epc": self._estimate_epc_from_commission(company.get('commission_rate', '5-10%')),
                                "link": company.get('website', '#'),
                                "source": "llm_identified"
                            })
                    
                    logger.info("LLM identified programs", programs_found=len(programs))
                    return programs
                    
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse LLM JSON response", error=str(e))
                return []
                
        except Exception as e:
            logger.warning("LLM company identification failed", error=str(e))
            return []
    
    def _estimate_epc_from_commission(self, commission_str: str) -> str:
        """Estimate EPC based on commission rate"""
        try:
            # Extract numeric value from commission string like "5-10%"
            import re
            numbers = re.findall(r'\d+', commission_str)
            if numbers:
                avg_commission = sum(int(n) for n in numbers) / len(numbers)
                # Rough EPC estimation: higher commission = higher EPC
                epc = round(avg_commission * 0.3, 2)
                return f"${epc}"
        except:
            pass
        return "$2.50"
    
    def _prioritize_quality_programs(self, programs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioritize high-quality affiliate programs over generic ones
        """
        # Separate programs by quality
        quality_programs = []
        generic_programs = []
        
        for program in programs:
            name = program.get('name', '').lower()
            description = program.get('description', '').lower()
            source = program.get('source', '')
            
            # Check if it's a quality program (only real affiliate programs)
            is_quality = (
                source == 'llm_identified' or  # Only LLM-identified programs
                ('affiliate program' in name and 'commission' in description) or  # Real affiliate programs with commission info
                (any(brand in name for brand in ['celestron', 'orion', 'meade', 'sky-watcher', 'agena', 'opt', 'unistellar', 'leupold', 'bushnell']) and 
                 program.get('commission_rate') not in ['N/A', None, 'Unknown'])  # Known brands with real commission rates
            )
            
            # Check if it's generic (Amazon product links, etc.)
            is_generic = (
                'amazon associates' in name or
                'amazon.com' in program.get('link', '') or
                'bestbuy.com' in program.get('link', '') or
                'affiliate program found at' in description or
                program.get('commission_rate') == 'N/A' or
                program.get('commission_rate') is None
            )
            
            if is_quality and not is_generic:
                quality_programs.append(program)
            else:
                generic_programs.append(program)
        
        # Return quality programs first, then generic ones
        prioritized = quality_programs + generic_programs
        logger.info("Programs prioritized", 
                   quality_programs=len(quality_programs), 
                   generic_programs=len(generic_programs),
                   total_programs=len(prioritized))
        
        return prioritized
    
    def _deduplicate_programs(self, programs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate and consolidate affiliate programs
        """
        seen_programs = {}
        deduplicated = []
        
        for program in programs:
            name = program.get('name', '').lower()
            network = program.get('network', '').lower()
            link = program.get('link', '')
            
            # Create a unique key for deduplication
            if 'amazon' in name or 'amazon' in network or 'amazon.com' in link:
                key = 'amazon_associates'
            elif 'best buy' in name or 'bestbuy' in link:
                key = 'best_buy_affiliate'
            elif 'shareasale' in network.lower():
                key = f"shareasale_{name}"
            elif 'impact' in network.lower():
                key = f"impact_{name}"
            elif 'cj affiliate' in network.lower():
                key = f"cj_{name}"
            elif 'shopify' in network.lower():
                key = f"shopify_{name}"
            elif 'awin' in network.lower():
                key = f"awin_{name}"
            elif 'clickbank' in network.lower():
                key = f"clickbank_{name}"
            elif 'flexoffers' in network.lower():
                key = f"flexoffers_{name}"
            elif 'rakuten' in network.lower():
                key = f"rakuten_{name}"
            elif 'affiliaxe' in network.lower():
                key = f"affiliaxe_{name}"
            elif 'giddyup' in network.lower():
                key = f"giddyup_{name}"
            else:
                # For other programs, use name + network as key
                key = f"{name}_{network}"
            
            # If we haven't seen this program before, add it
            if key not in seen_programs:
                seen_programs[key] = program
                deduplicated.append(program)
            else:
                # If we have a better version (with commission info), replace it
                existing = seen_programs[key]
                if (program.get('commission_rate') not in ['N/A', None, 'Unknown'] and 
                    existing.get('commission_rate') in ['N/A', None, 'Unknown']):
                    # Replace with better version
                    seen_programs[key] = program
                    deduplicated = [p for p in deduplicated if p != existing] + [program]
        
        logger.info("Programs deduplicated", 
                   original_count=len(programs), 
                   deduplicated_count=len(deduplicated))
        
        return deduplicated
    
    def _save_research_record(
        self, 
        search_term: str, 
        niche: Optional[str], 
        budget_range: Optional[str],
        programs: List[Dict[str, Any]],
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Save research record to Supabase
        """
        research_data = {
            "user_id": user_id,
            "topic": search_term,
            "search_query": f"{search_term} | {niche or 'Any'} | {budget_range or 'Any'}",
            "results": programs,
            "total_programs_found": len(programs),
            "status": "completed",
            "created_at": datetime.utcnow().isoformat()
        }
        
        result = self.db.client.table("affiliate_research").insert(research_data).execute()
        
        if result.data:
            logger.info("Research record saved to Supabase", research_id=result.data[0].get('id'))
            return result.data[0]
        else:
            raise Exception("Failed to save research record to Supabase")
    
    async def generate_content_ideas(
        self, 
        selected_programs: List[str],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate content ideas based on selected affiliate programs
        """
        try:
            # Get program details
            programs = self._get_programs_by_ids(selected_programs)
            
            # Generate content ideas for each program
            content_ideas = []
            for program in programs:
                program_ideas = program.get("content_opportunities", [])
                for i, idea in enumerate(program_ideas):
                    content_ideas.append({
                        "id": f"{program['id']}-{i}",
                        "title": idea,
                        "affiliate_program": program["name"],
                        "affiliate_program_id": program["id"],
                        "estimated_traffic": self._estimate_traffic(idea),
                        "competition_level": self._assess_competition(idea),
                        "content_type": self._suggest_content_type(idea),
                        "potential_earnings": self._estimate_earnings(program, idea),
                        "difficulty": program.get("difficulty", "Medium"),
                        "commission_rate": program.get("commission", "5-10%"),
                        "created_at": datetime.utcnow().isoformat()
                    })
            
            return {
                "content_ideas": content_ideas,
                "total_ideas": len(content_ideas),
                "selected_programs": len(programs),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Content idea generation failed", error=str(e))
            raise
    
    def _get_programs_by_ids(self, program_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get program details by IDs (mock implementation)
        """
        # In real implementation, this would query the database
        programs_db = [
            {
                "id": "1",
                "name": "EcoHome Solutions",
                "content_opportunities": [
                    "Best eco-friendly home products 2024",
                    "Sustainable living guide for beginners",
                    "Green home renovation tips",
                    "Eco-friendly cleaning product reviews"
                ],
                "commission": "8-12%",
                "difficulty": "Medium"
            },
            {
                "id": "2", 
                "name": "TechGadget Pro",
                "content_opportunities": [
                    "Top tech gadgets for 2024",
                    "Smart home setup guide",
                    "Gaming accessories review",
                    "Productivity tools comparison"
                ],
                "commission": "5-8%",
                "difficulty": "Easy"
            },
            {
                "id": "3",
                "name": "FitnessFirst",
                "content_opportunities": [
                    "Home workout equipment guide",
                    "Fitness supplement reviews",
                    "Weight loss journey tips",
                    "Muscle building program reviews"
                ],
                "commission": "10-15%",
                "difficulty": "Hard"
            }
        ]
        
        return [p for p in programs_db if p["id"] in program_ids]
    
    def _estimate_traffic(self, content_idea: str) -> int:
        """Estimate traffic potential for content idea"""
        # Simple estimation based on keywords
        if any(word in content_idea.lower() for word in ["best", "top", "guide", "review"]):
            return 2000 + (hash(content_idea) % 3000)
        return 1000 + (hash(content_idea) % 2000)
    
    def _assess_competition(self, content_idea: str) -> str:
        """Assess competition level for content idea"""
        if any(word in content_idea.lower() for word in ["best", "top", "2024"]):
            return "High"
        elif any(word in content_idea.lower() for word in ["guide", "tips", "how to"]):
            return "Medium"
        return "Low"
    
    def _suggest_content_type(self, content_idea: str) -> str:
        """Suggest content type based on idea"""
        if "review" in content_idea.lower():
            return "Review"
        elif "guide" in content_idea.lower():
            return "Guide"
        elif "tips" in content_idea.lower():
            return "Tips"
        return "Article"
    
    def _estimate_earnings(self, program: Dict[str, Any], content_idea: str) -> int:
        """Estimate potential earnings for content idea"""
        base_earnings = 50
        if program.get("difficulty") == "Easy":
            base_earnings += 25
        elif program.get("difficulty") == "Hard":
            base_earnings += 100
        
        # Add random variation
        return base_earnings + (hash(content_idea) % 150)
    
    def get_research_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get user's research history
        """
        try:
            research_records = self.db.query(AffiliateResearch)\
                .filter(AffiliateResearch.user_id == user_id)\
                .order_by(AffiliateResearch.created_at.desc())\
                .limit(limit)\
                .all()
            
            return [
                {
                    "id": str(record.id),
                    "search_term": record.search_term,
                    "niche": record.niche,
                    "budget_range": record.budget_range,
                    "total_programs": record.total_programs,
                    "status": record.status,
                    "created_at": record.created_at.isoformat()
                }
                for record in research_records
            ]
            
        except Exception as e:
            logger.error("Failed to get research history", error=str(e))
            return []
    
    async def list_user_researches(
        self, 
        user_id: str, 
        skip: int = 0, 
        limit: int = 20, 
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List user's affiliate researches with pagination and filtering
        """
        try:
            # Query affiliate researches from Supabase
            query = self.db.client.table("affiliate_research").select("*").eq("user_id", user_id)
            
            if status:
                query = query.eq("status", status)
            
            # Apply pagination
            query = query.order("created_at", desc=True).range(skip, skip + limit - 1)
            
            result = query.execute()
            
            if result.data:
                return [
                    {
                        "id": str(record["id"]),
                        "niche": record.get("topic", ""),
                        "status": record.get("status", "pending"),
                        "programs_found": record.get("total_programs_found", 0),
                        "created_at": record.get("created_at", ""),
                        "estimated_completion_time": None  # Not implemented yet
                    }
                    for record in result.data
                ]
            else:
                return []
                
        except Exception as e:
            logger.error("Failed to list user researches", user_id=user_id, error=str(e))
            return []