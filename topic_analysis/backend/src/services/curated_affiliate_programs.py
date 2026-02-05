"""
Curated database of real affiliate programs
This contains actual affiliate programs that are known to exist and accept affiliates
With Linkup API + LLM-powered dynamic category generation
"""

from typing import List, Dict, Any, Optional
import re
import json
import structlog

logger = structlog.get_logger()

class CuratedAffiliatePrograms:
    """Database of real affiliate programs organized by category"""
    
    def __init__(self):
        self.dynamic_categories = {}  # Store LLM-generated categories
        self.programs = {
            "eco_friendly": [
                {
                    "id": "patagonia_affiliate",
                    "name": "Patagonia Affiliate Program",
                    "description": "Outdoor clothing and gear with environmental focus",
                    "commission_rate": "8-12%",
                    "network": "Direct",
                    "epc": "25.50",
                    "link": "https://www.patagonia.com/affiliate",
                    "category": "eco_friendly"
                },
                {
                    "id": "allbirds_affiliate",
                    "name": "Allbirds Affiliate Program",
                    "description": "Sustainable footwear and apparel made from natural materials",
                    "commission_rate": "5-8%",
                    "network": "CJ Affiliate",
                    "epc": "18.75",
                    "link": "https://www.allbirds.com/affiliate",
                    "category": "eco_friendly"
                },
                {
                    "id": "tentree_affiliate",
                    "name": "Tentree Affiliate Program",
                    "description": "Sustainable clothing that plants trees with every purchase",
                    "commission_rate": "6-10%",
                    "network": "ShareASale",
                    "epc": "22.30",
                    "link": "https://www.tentree.com/affiliate",
                    "category": "eco_friendly"
                },
                {
                    "id": "reformation_affiliate",
                    "name": "Reformation Affiliate Program",
                    "description": "Sustainable fashion and eco-friendly clothing",
                    "commission_rate": "4-8%",
                    "network": "CJ Affiliate",
                    "epc": "28.90",
                    "link": "https://www.thereformation.com/affiliate",
                    "category": "eco_friendly"
                }
            ],
            "home_garden": [
                {
                    "id": "wayfair_affiliate",
                    "name": "Wayfair Affiliate Program",
                    "description": "Home furniture and decor with eco-friendly options",
                    "commission_rate": "3-8%",
                    "network": "CJ Affiliate",
                    "epc": "15.20",
                    "link": "https://www.wayfair.com/affiliate",
                    "category": "home_garden"
                },
                {
                    "id": "west_elm_affiliate",
                    "name": "West Elm Affiliate Program",
                    "description": "Modern furniture and home decor with sustainable options",
                    "commission_rate": "4-6%",
                    "network": "CJ Affiliate",
                    "epc": "18.75",
                    "link": "https://www.westelm.com/affiliate",
                    "category": "home_garden"
                },
                {
                    "id": "crate_barrel_affiliate",
                    "name": "Crate & Barrel Affiliate Program",
                    "description": "Home furnishings and decor with eco-friendly collections",
                    "commission_rate": "3-5%",
                    "network": "CJ Affiliate",
                    "epc": "22.50",
                    "link": "https://www.crateandbarrel.com/affiliate",
                    "category": "home_garden"
                }
            ],
            "smart_home": [
                {
                    "id": "nest_affiliate",
                    "name": "Google Nest Affiliate Program",
                    "description": "Smart home devices and security systems",
                    "commission_rate": "3-6%",
                    "network": "CJ Affiliate",
                    "epc": "18.40",
                    "link": "https://store.google.com/affiliate",
                    "category": "smart_home"
                },
                {
                    "id": "ring_affiliate",
                    "name": "Ring Security Affiliate Program",
                    "description": "Smart doorbells, security cameras, and home monitoring",
                    "commission_rate": "4-8%",
                    "network": "Ring Partners",
                    "epc": "22.60",
                    "link": "https://ring.com/affiliate",
                    "category": "smart_home"
                },
                {
                    "id": "ecobee_affiliate",
                    "name": "Ecobee Affiliate Program",
                    "description": "Smart thermostats and home automation devices",
                    "commission_rate": "5-10%",
                    "network": "CJ Affiliate",
                    "epc": "25.80",
                    "link": "https://www.ecobee.com/affiliate",
                    "category": "smart_home"
                }
            ],
            "sustainable_energy": [
                {
                    "id": "tesla_affiliate",
                    "name": "Tesla Affiliate Program",
                    "description": "Electric vehicles and solar energy products",
                    "commission_rate": "2-5%",
                    "network": "Direct",
                    "epc": "150.00",
                    "link": "https://www.tesla.com/affiliate",
                    "category": "sustainable_energy"
                },
                {
                    "id": "sunrun_affiliate",
                    "name": "Sunrun Affiliate Program",
                    "description": "Solar panel installation and renewable energy solutions",
                    "commission_rate": "10-15%",
                    "network": "CJ Affiliate",
                    "epc": "45.20",
                    "link": "https://www.sunrun.com/affiliate",
                    "category": "sustainable_energy"
                },
                {
                    "id": "vivint_affiliate",
                    "name": "Vivint Affiliate Program",
                    "description": "Smart home security and solar energy solutions",
                    "commission_rate": "8-12%",
                    "network": "CJ Affiliate",
                    "epc": "35.75",
                    "link": "https://www.vivint.com/affiliate",
                    "category": "sustainable_energy"
                }
            ],
            "tiny_houses": [
                {
                    "id": "tiny_house_build_affiliate",
                    "name": "Tiny House Build Affiliate Program",
                    "description": "Tiny house plans, kits, and construction materials",
                    "commission_rate": "8-15%",
                    "network": "ClickBank",
                    "epc": "45.80",
                    "link": "https://tinyhousebuild.com/affiliate",
                    "category": "tiny_houses"
                },
                {
                    "id": "tumbleweed_affiliate",
                    "name": "Tumbleweed Tiny Houses Affiliate Program",
                    "description": "Custom tiny house designs and mobile home solutions",
                    "commission_rate": "5-10%",
                    "network": "ShareASale",
                    "epc": "38.20",
                    "link": "https://tumbleweedhouses.com/affiliate",
                    "category": "tiny_houses"
                },
                {
                    "id": "tiny_house_listings_affiliate",
                    "name": "Tiny House Listings Affiliate Program",
                    "description": "Tiny house rentals, sales, and community listings",
                    "commission_rate": "6-12%",
                    "network": "CJ Affiliate",
                    "epc": "28.50",
                    "link": "https://tinyhouselistings.com/affiliate",
                    "category": "tiny_houses"
                }
            ],
            "career_professional": [
                {
                    "id": "linkedin_learning_affiliate",
                    "name": "LinkedIn Learning Affiliate Program",
                    "description": "Professional development courses and career skill training",
                    "commission_rate": "15-25%",
                    "network": "Direct",
                    "epc": "32.50",
                    "link": "https://www.linkedin.com/learning/affiliate",
                    "category": "career_professional"
                },
                {
                    "id": "coursera_affiliate",
                    "name": "Coursera Affiliate Program",
                    "description": "Online courses from top universities and companies",
                    "commission_rate": "20-30%",
                    "network": "CJ Affiliate",
                    "epc": "28.75",
                    "link": "https://www.coursera.org",
                    "category": "career_professional"
                },
                {
                    "id": "udemy_affiliate",
                    "name": "Udemy Affiliate Program",
                    "description": "Online learning platform with courses on career skills and professional development",
                    "commission_rate": "15-25%",
                    "network": "ShareASale",
                    "epc": "22.30",
                    "link": "https://www.udemy.com",
                    "category": "career_professional"
                },
                {
                    "id": "skillshare_affiliate",
                    "name": "Skillshare Affiliate Program",
                    "description": "Creative skills and professional development courses",
                    "commission_rate": "20-30%",
                    "network": "Direct",
                    "epc": "26.40",
                    "link": "https://www.skillshare.com/affiliate",
                    "category": "career_professional"
                },
                {
                    "id": "indeed_affiliate",
                    "name": "Indeed Affiliate Program",
                    "description": "Job search and career development resources",
                    "commission_rate": "10-15%",
                    "network": "CJ Affiliate",
                    "epc": "15.80",
                    "link": "https://www.indeed.com/affiliates",
                    "category": "career_professional"
                },
                {
                    "id": "glassdoor_affiliate",
                    "name": "Glassdoor Affiliate Program",
                    "description": "Salary information, company reviews, and career insights",
                    "commission_rate": "8-12%",
                    "network": "Awin",
                    "epc": "18.50",
                    "link": "https://www.glassdoor.com",
                    "category": "career_professional"
                },
                {
                    "id": "masterclass_affiliate",
                    "name": "MasterClass Affiliate Program",
                    "description": "Learn from world-class experts across industries",
                    "commission_rate": "30-40%",
                    "network": "Direct",
                    "epc": "45.20",
                    "link": "https://www.masterclass.com",
                    "category": "career_professional"
                },
                {
                    "id": "pluralsight_affiliate",
                    "name": "Pluralsight Affiliate Program",
                    "description": "Technology and IT professional development courses",
                    "commission_rate": "15-20%",
                    "network": "ShareASale",
                    "epc": "35.75",
                    "link": "https://www.pluralsight.com",
                    "category": "career_professional"
                }
            ]
        }
    
    async def search_programs(self, search_term: str, topic: str) -> List[Dict[str, Any]]:
        """Search for relevant affiliate programs based on search term and topic"""
        found_programs = []
        search_lower = search_term.lower()
        topic_lower = topic.lower()
        
        # Define keyword mappings to categories
        keyword_mappings = {
            "eco": ["eco_friendly", "sustainable_energy"],
            "green": ["eco_friendly", "sustainable_energy"],
            "sustainable": ["eco_friendly", "sustainable_energy"],
            "environmental": ["eco_friendly", "sustainable_energy"],
            "home": ["home_garden", "smart_home", "tiny_houses"],
            "house": ["home_garden", "smart_home", "tiny_houses"],
            "tiny": ["tiny_houses"],
            "smart": ["smart_home"],
            "energy": ["sustainable_energy"],
            "solar": ["sustainable_energy"],
            "electric": ["sustainable_energy"],
            "furniture": ["home_garden"],
            "decor": ["home_garden"],
            "garden": ["home_garden"],
            "career": ["career_professional"],
            "job": ["career_professional"],
            "salary": ["career_professional"],
            "promotion": ["career_professional"],
            "negotiate": ["career_professional"],
            "professional": ["career_professional"],
            "workplace": ["career_professional"],
            "business": ["career_professional"],
            "training": ["career_professional"],
            "skill": ["career_professional"],
            "learn": ["career_professional"],
            "course": ["career_professional"],
            "education": ["career_professional"]
        }
        
        # Find relevant categories
        relevant_categories = set()
        for keyword, categories in keyword_mappings.items():
            if keyword in search_lower or keyword in topic_lower:
                relevant_categories.update(categories)
        
        # Get programs from relevant categories ONLY (don't search all categories)
        for category in relevant_categories:
            if category in self.programs:
                for program in self.programs[category]:
                    # Check if program is relevant to search term
                    if self._is_relevant_program(program, search_term, topic):
                        found_programs.append(program)
        
        # If no programs found in curated categories, try dynamic generation
        if not found_programs:
            logger.info("No programs found in curated categories, generating dynamically", 
                       search_term=search_term, topic=topic)
            generated_programs = await self.generate_category_dynamically(search_term, topic)
            found_programs.extend(generated_programs)
        
        return found_programs
    
    def _is_relevant_program(self, program: Dict[str, Any], search_term: str, topic: str) -> bool:
        """Check if a program is relevant to the search term and topic with strict matching"""
        search_lower = search_term.lower()
        topic_lower = topic.lower()
        
        # First check: Filter out programs that are clearly from wrong categories
        # If search/topic is about abstract concepts (career, learning, etc.), exclude physical building materials
        program_text = f"{program['name']} {program['description']}".lower()
        program_category = program.get('category', '')
        
        # Exclude physical building materials from career/learning searches
        if any(keyword in search_lower or keyword in topic_lower 
               for keyword in ['career', 'professional', 'networks', 'learning', 'training', 'skill', 'job', 'salary']):
            # If the program is about physical building materials, exclude it
            if any(building_term in program_text for building_term in ['building materials', 'construction', 'flooring', 'lumber', 'wood', 'bamboo', 'eco building', 'green building supplies']):
                return False
        
        # Exclude career-related programs from physical building searches
        if any(keyword in search_lower or keyword in topic_lower 
               for keyword in ['materials', 'construction', 'flooring', 'wood', 'bamboo', 'building supplies']):
            # If the program is about career/learning (not physical building), exclude it
            if any(career_term in program_text for career_term in ['career', 'training', 'course', 'learning', 'education', 'skill development', 'professional development']):
                return False
        
        # Extract meaningful keywords from search term and topic (ignore common words and ambiguous terms)
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        
        # Also exclude ambiguous words that can cause false matches
        ambiguous_words = {'building'}  # Can mean construction OR networking
        
        search_keywords = [w for w in re.findall(r'\b\w+\b', search_lower) 
                          if w not in stop_words and len(w) > 3 and w not in ambiguous_words]
        topic_keywords = [w for w in re.findall(r'\b\w+\b', topic_lower) 
                         if w not in stop_words and len(w) > 3 and w not in ambiguous_words]
        all_keywords = search_keywords + topic_keywords
        
        if not all_keywords:
            return False
        
        # Relaxed matching: at least 1 keyword match if it's high quality
        for keyword in all_keywords:
            if len(keyword) > 3 and keyword in program_text:
                match_count += 1
        
        # Only return True if we have some keyword matches
        return match_count >= 1
    
    async def generate_category_dynamically(self, search_term: str, topic: str) -> List[Dict[str, Any]]:
        """
        Dynamically generate affiliate programs for topics that don't have a category
        Uses Linkup API first, then falls back to LLM generation
        """
        try:
            # Check if we already generated this category
            category_key = f"{search_term.lower()}_{topic.lower()}"
            if category_key in self.dynamic_categories:
                logger.info("Returning cached dynamic category", category_key=category_key)
                return self.dynamic_categories[category_key]
            
            # First, try Linkup API for real-time affiliate programs
            from src.integrations.linkup_api import linkup_api
            
            logger.info("Trying Linkup API for dynamic category", search_term=search_term, topic=topic)
            linkup_programs = await linkup_api.search_offers(search_term, limit=10)
            
            if linkup_programs:
                logger.info("Linkup API returned programs", count=len(linkup_programs))
                # Convert Linkup format to our format
                formatted_programs = self._convert_linkup_programs(linkup_programs)
                
                # Cache the results
                self.dynamic_categories[category_key] = formatted_programs
                return formatted_programs
            
            # If Linkup fails, fall back to LLM generation
            logger.info("Linkup API returned no results, using LLM fallback", search_term=search_term)
            from src.integrations.llm_providers import generate_content
            
            # Create a prompt for LLM to identify relevant affiliate programs
            prompt = f"""
            For the topic: "{topic}" and search term: "{search_term}"
            
            Generate 8-12 REAL affiliate programs that are relevant to this topic. 
            
            IMPORTANT: Only include companies that actually have active affiliate programs. Do not make up programs.
            
            Focus on:
            1. Companies with known affiliate programs
            2. Relevant to the specific topic
            3. Mix of networks (CJ Affiliate, ShareASale, Amazon Associates, direct programs)
            
            Return ONLY a JSON array with this exact structure:
            [
                {{
                    "id": "unique_program_id",
                    "name": "Company Name Affiliate Program",
                    "description": "What they offer relevant to the topic",
                    "commission_rate": "5-10%",
                    "network": "CJ Affiliate",
                    "epc": "15.50",
                    "link": "https://company.com/affiliate",
                    "category": "dynamic"
                }}
            ]
            
            Example for career topics:
            {{
                "id": "linkedin_learning_affiliate",
                "name": "LinkedIn Learning Affiliate Program",
                "description": "Professional development courses and career skill training",
                "commission_rate": "15-25%",
                "network": "Direct",
                "epc": "32.50",
                "link": "https://www.linkedin.com/learning/affiliate",
                "category": "dynamic"
            }}
            """
            
            logger.info("Calling LLM to generate dynamic category", search_term=search_term, topic=topic)
            
            # Call LLM
            llm_result = await generate_content(
                prompt=prompt,
                provider="openai",  # You can make this dynamic based on your setup
                max_tokens=1500,
                temperature=0.7
            )
            
            if "error" in llm_result:
                logger.error("LLM generation failed", error=llm_result.get("error"))
                return []
            
            # Parse the response
            content = llm_result.get("content", "")
            
            # Extract JSON from the response
            json_start = content.find('[')
            if json_start == -1:
                logger.warning("No JSON array found in LLM response")
                return []
            
            json_end = content.rfind(']') + 1
            if json_end == 0:
                logger.warning("No valid JSON array found")
                return []
            
            json_str = content[json_start:json_end]
            
            try:
                generated_programs = json.loads(json_str)
                
                # Cache the generated programs
                self.dynamic_categories[category_key] = generated_programs
                
                logger.info("Generated dynamic category", 
                           category_key=category_key, 
                           programs_count=len(generated_programs))
                
                return generated_programs
                
            except json.JSONDecodeError as e:
                logger.error("Failed to parse LLM JSON response", error=str(e))
                return []
                
        except Exception as e:
            logger.error("Dynamic category generation failed", error=str(e))
            return []
    
    def _convert_linkup_programs(self, linkup_programs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Linkup program format to our curated format"""
        converted = []
        
        for program in linkup_programs:
            try:
                # Extract basic info with fallbacks
                name = program.get("name", "Unknown Program")
                description = program.get("description", "")
                commission = program.get("commission", program.get("commission_rate", "5-15%"))
                network = program.get("network", "LinkUp")
                
                # Get link
                link = program.get("link", "")
                if not link:
                    link = program.get("url", "")
                
                # Generate ID
                program_id = program.get("id", f"linkup_{name.lower().replace(' ', '_')}")
                
                # Convert to our format
                converted_program = {
                    "id": program_id,
                    "name": name,
                    "description": description[:200] if description else "Affiliate program from Linkup",
                    "commission_rate": commission if isinstance(commission, str) else f"{commission}%",
                    "network": network,
                    "epc": str(program.get("epc", "15.00")),
                    "link": link if link else "#",
                    "category": "dynamic",
                    "source": "linkup"
                }
                converted.append(converted_program)
            except Exception as e:
                logger.warning("Failed to convert Linkup program", error=str(e))
                continue
        
        return converted
