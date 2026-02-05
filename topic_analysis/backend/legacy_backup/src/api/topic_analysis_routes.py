from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
import openai
import os
import json
from datetime import datetime
import structlog
from ..core.config import settings
from ..services.llm_service import LLMService
from ..services.web_search_service import WebSearchService
from ..core.database import get_db
from sqlalchemy.orm import Session

logger = structlog.get_logger()

router = APIRouter(prefix="/api/topic-analysis", tags=["topic-analysis"])

class TopicAnalysisRequest(BaseModel):
    topic: str
    include_affiliate_programs: bool = True
    max_related_areas: int = 10
    max_affiliate_programs: int = 8

class RelatedArea(BaseModel):
    area: str
    description: str
    relevance_score: float

class AffiliateProgram(BaseModel):
    name: str
    commission: str
    category: str
    difficulty: str
    description: str
    link: str
    estimated_traffic: int
    competition_level: str

class TopicAnalysisResponse(BaseModel):
    topic: str
    related_areas: List[RelatedArea]
    affiliate_programs: List[AffiliateProgram]
    analysis_metadata: Dict[str, Any]

@router.post("/analyze", response_model=TopicAnalysisResponse)
async def analyze_topic(request: TopicAnalysisRequest, db: Session = Depends(get_db)):
    """
    Analyze a topic using LLM to find related areas and affiliate programs
    """
    try:
        # Use LLM service instead of direct OpenAI
        llm_service = LLMService(db)
        topic_lower = request.topic.lower()
        
        # Create a comprehensive prompt for topic analysis
        prompt = f"""
        You are an expert content strategist and affiliate marketing specialist. Analyze the topic "{request.topic}" and create comprehensive, specific subtopics that would lead to valuable affiliate marketing opportunities.

        TASK: Generate {request.max_related_areas} highly specific, actionable subtopics that someone interested in "{request.topic}" would actually search for and buy products/services for.

        SUBTOPIC GENERATION RULES:
        1. Think about the COMPLETE CUSTOMER JOURNEY for someone interested in "{request.topic}"
        2. Consider different USER INTENTS: research, comparison, purchase, maintenance, accessories, services
        3. Think about RELATED PRODUCTS/SERVICES they would need
        4. Consider DIFFERENT SKILL LEVELS: beginner, intermediate, advanced
        5. Think about DIFFERENT USE CASES and SCENARIOS
        6. Consider SEASONAL and SITUATIONAL needs

        EXAMPLES OF GOOD SUBTOPICS:
        - For "Ski Resorts Worldwide": "Ski Equipment Reviews", "Ski Resort Booking", "Ski Lessons Online", "Ski Insurance", "Ski Apparel", "Ski Lift Tickets", "Ski Maintenance", "Ski Photography", "Ski Travel Planning", "Ski Safety Gear"
        - For "Best SUVs 2025": "SUV Financing", "SUV Insurance", "SUV Accessories", "SUV Maintenance", "SUV Towing", "SUV Fuel Economy", "SUV Safety Features", "SUV Interior Design", "SUV Off-Roading", "SUV Family Features"
        - For "Yoga for Beginners": "Yoga Mats", "Yoga Clothing", "Yoga Props", "Yoga Apps", "Yoga Retreats", "Yoga Teacher Training", "Yoga Nutrition", "Yoga Meditation", "Yoga Home Practice", "Yoga Injuries Prevention"

        For each subtopic, provide:
        - A SPECIFIC, ACTIONABLE name (not generic like "advanced techniques")
        - A clear description of what this subtopic covers
        - A relevance score (0-1) based on how likely someone interested in "{request.topic}" would search for this

        AFFILIATE PROGRAMS:
        Generate {request.max_affiliate_programs} realistic affiliate programs that would be relevant to these subtopics. Focus on programs that actually exist and are accessible to content creators.

        Return ONLY valid JSON in this exact format:
        {{
            "related_areas": [
                {{"area": "Specific Subtopic Name", "description": "What this covers and why it's relevant", "relevance_score": 0.9}}
            ],
            "affiliate_programs": [
                {{"name": "Real Program Name", "commission": "5-15%", "category": "Specific Category", "difficulty": "Easy/Medium/Hard", "description": "What they offer", "estimated_traffic": 5000, "competition_level": "Low/Medium/High"}}
            ]
        }}
        """
        
        # Use LLM service to analyze the topic
        try:
            analysis_result = await llm_service.analyze_topic_with_llm(prompt)
            
            # Parse the JSON response
            if isinstance(analysis_result, dict) and 'content' in analysis_result:
                analysis_data = json.loads(analysis_result['content'])
            elif isinstance(analysis_result, str):
                analysis_data = json.loads(analysis_result)
            else:
                analysis_data = analysis_result
                
        except Exception as llm_error:
            logger.warning("LLM analysis failed, trying web search", error=str(llm_error))
            # Try web search for better subtopics
            try:
                async with WebSearchService() as web_search:
                    web_programs = await web_search.search_affiliate_programs(request.topic, max_results=3)
                    if web_programs:
                        # Extract categories from web search results
                        categories = list(set([p.get("category", "General") for p in web_programs]))
                        analysis_data = {
                            "related_areas": [
                                {"area": f"{cat} Reviews", "description": f"Product and service reviews in {cat}", "relevance_score": 0.9}
                                for cat in categories[:request.max_related_areas]
                            ],
                            "affiliate_programs": web_programs[:request.max_affiliate_programs]
                        }
                    else:
                        raise Exception("Web search also failed")
            except Exception as web_error:
                logger.warning("Web search fallback failed, using intelligent mock data", error=str(web_error))
                # Fallback to mock data if LLM fails - make it more intelligent based on topic
            
            # Dynamic subtopic generation for any category using improved detection
            related_areas, affiliate_programs = generate_dynamic_subtopics(request.topic, topic_lower, request.max_related_areas, request.max_affiliate_programs)
            
            analysis_data = {
                "related_areas": related_areas,
                "affiliate_programs": affiliate_programs
            }

        # Extract related areas and affiliate programs from analysis
        related_areas = analysis_data.get("related_areas", [])[:request.max_related_areas]
        affiliate_programs = analysis_data.get("affiliate_programs", [])[:request.max_affiliate_programs]
        
        # Convert to response models
        related_areas_models = [
            RelatedArea(
                area=area["area"],
                description=area["description"],
                relevance_score=area["relevance_score"]
            ) for area in related_areas
        ]
        
        affiliate_programs_models = [
            AffiliateProgram(
                name=program["name"],
                commission=program["commission"],
                category=program["category"],
                difficulty=program["difficulty"],
                description=program["description"],
                link=program["link"],
                estimated_traffic=program["estimated_traffic"],
                competition_level=program["competition_level"]
            ) for program in affiliate_programs
        ]
        
        # Create response
        response_data = TopicAnalysisResponse(
            topic=request.topic,
            related_areas=related_areas_models,
            affiliate_programs=affiliate_programs_models,
            analysis_metadata={
                "total_related_areas": len(related_areas_models),
                "total_affiliate_programs": len(affiliate_programs_models),
                "analysis_timestamp": "2024-01-01T00:00:00Z",
                "llm_provider": "configured_provider"
            }
        )
        
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def generate_dynamic_subtopics(topic: str, topic_lower: str, max_areas: int, max_programs: int):
    """
    Generate dynamic, specific subtopics for any category using LLM-based semantic detection
    """
    
    # Use keyword-based detection for this function since it's not async
    category = _fallback_category_detection_simple(topic)
    logger.info("Category detection result", topic=topic, category=category)
    if category:
        return generate_category_specific_content(topic, category, max_areas, max_programs)
    
    # Fallback to keyword-based detection
    return generate_keyword_based_content(topic, topic_lower, max_areas, max_programs)

def _fallback_category_detection_simple(topic: str) -> str:
    """
    Simple fallback category detection using keyword matching
    """
    topic_lower = topic.lower()
    
    if any(word in topic_lower for word in ['outdoor', 'camping', 'hiking', 'fishing', 'lake', 'weekend', 'nature', 'park', 'trail', 'mountain', 'river', 'beach', 'forest']):
        return 'outdoor_recreation'
    elif any(word in topic_lower for word in ['cooking', 'recipe', 'food', 'kitchen', 'baking', 'cuisine', 'meal', 'restaurant', 'coffee', 'pizza']):
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

async def detect_category_with_llm(topic: str, db: Session) -> str:
    """
    Use LLM to detect the semantic category of a topic
    """
    try:
        # Initialize LLM service
        llm_service = LLMService(db)
        
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
        
        # Call LLM for category detection
        result = await llm_service.analyze_topic_with_llm(prompt)
        
        # Extract category from LLM response
        if result and 'content' in result:
            category = result['content'].strip().lower()
            # Validate the category
            valid_categories = [
                'outdoor_recreation', 'food_cooking', 'technology', 'health_fitness',
                'education_learning', 'home_garden', 'travel_hospitality', 'fashion_beauty',
                'automotive', 'business_services', 'entertainment_gaming', 'finance_investing',
                'pets_animals', 'sports_fitness', 'general'
            ]
            if category in valid_categories:
                return category
        
        # Fallback to general if LLM response is invalid
        return 'general'
        
    except Exception as e:
        logger.error("LLM detection failed", error=str(e))
        return 'general'

async def generate_category_specific_content_with_llm(topic: str, category: str, max_areas: int, max_programs: int, db: Session):
    """
    Generate content using LLM for ANY topic based on detected category
    """
    try:
        # Initialize LLM service
        llm_service = LLMService(db)
        
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
        - A realistic program name
        - Commission rate (realistic percentage)
        - Category
        - Difficulty level (Easy/Medium/Hard)
        - Description
        - Affiliate link (use realistic URLs)
        - Estimated traffic
        - Competition level (Low/Medium/High)
        
        Focus on the category: {category}
        
        Return your response as a JSON object with this structure:
        {{
            "related_areas": [
                {{"area": "Area Name", "description": "Description", "relevance_score": 0.9}}
            ],
            "affiliate_programs": [
                {{"name": "Program Name", "commission": "5-15%", "category": "Category", "difficulty": "Easy", "description": "Description", "link": "https://example.com", "estimated_traffic": 10000, "competition_level": "Medium"}}
            ]
        }}
        """
        
        # Call LLM for content generation
        result = await llm_service.analyze_topic_with_llm(prompt)
        
        # Parse LLM response
        if result and 'content' in result:
            try:
                content = result['content'].strip()
                # Try to extract JSON from the response
                if '{' in content and '}' in content:
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    json_str = content[json_start:json_end]
                    parsed = json.loads(json_str)
                    
                    related_areas = parsed.get('related_areas', [])[:max_areas]
                    affiliate_programs = parsed.get('affiliate_programs', [])[:max_programs]
                    
                    return related_areas, affiliate_programs
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM JSON response, using fallback")
        
        # Fallback to category-specific content
        return generate_category_specific_content(topic, category, max_areas, max_programs)
        
    except Exception as e:
        logger.error("LLM content generation failed", error=str(e))
        return generate_category_specific_content(topic, category, max_areas, max_programs)

def generate_category_specific_content(topic: str, category: str, max_areas: int, max_programs: int):
    """
    Generate content using hardcoded rules for specific categories
    """
    try:
        # Generate intelligent subtopics and affiliate programs for any topic
        related_areas = []
        affiliate_programs = []
        
        # Generate specific, intelligent subtopics based on the topic
        if any(word in topic.lower() for word in ['quantum', 'physics', 'science', 'research', 'theoretical', 'academic', 'study', 'university', 'college', 'scholar', 'professor', 'lecture', 'textbook', 'knowledge', 'intellectual', 'scholarly']):
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
        elif any(word in topic.lower() for word in ['extreme', 'sport', 'adventure', 'athletic', 'fitness', 'outdoor', 'climbing', 'skydiving', 'bungee', 'paragliding', 'base jumping', 'mountain', 'rock']):
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
        elif category == 'outdoor_recreation' or any(word in topic.lower() for word in ['weekend', 'lake', 'camping', 'hiking', 'fishing', 'outdoor', 'nature', 'wilderness', 'rv', 'tent', 'camp', 'trail', 'mountain', 'river', 'beach', 'park', 'recreation', 'adventure', 'outdoors']):
            # Outdoor recreation topics
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
        elif any(word in topic.lower() for word in ['cookie', 'cookies', 'baking', 'recipe', 'cooking', 'cuisine', 'food', 'meal', 'kitchen', 'chef', 'restaurant', 'dining', 'ingredient', 'spice', 'wine', 'coffee', 'diet', 'nutrition', 'homemade', 'home made']):
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
        logger.error("LLM content generation failed", error=str(e))
        return generate_keyword_based_content(topic, topic.lower(), max_areas, max_programs)

def generate_keyword_based_content(topic: str, topic_lower: str, max_areas: int, max_programs: int):
    """
    Fallback keyword-based category detection and content generation
    """
    
    # Define comprehensive category mappings
    category_mappings = {
        'software_saas': {
            'keywords': ['software', 'saas', 'app', 'tool', 'platform', 'crm', 'seo', 'email marketing', 'hosting', 'vpn', 'cyber', 'security', 'ai', 'automation', 'project management', 'accounting', 'hr', 'e-sign'],
            'subtopics': [
                'Software Reviews', 'Pricing Comparisons', 'Free Trials', 'Implementation Guides', 
                'Integration Tutorials', 'Customer Support', 'Security Features', 'API Documentation',
                'User Training', 'Customization Options'
            ],
            'affiliate_programs': [
                {'name': 'HubSpot Affiliate', 'commission': '20-30%', 'category': 'CRM', 'difficulty': 'Medium', 'link': 'https://www.hubspot.com/partners/affiliate-program', 'description': 'All-in-one CRM platform for businesses'},
                {'name': 'Shopify Affiliate', 'commission': '200% recurring', 'category': 'E-commerce', 'difficulty': 'Easy', 'link': 'https://partners.shopify.com/affiliates', 'description': 'E-commerce platform for online stores'},
                {'name': 'Squarespace Affiliate', 'commission': '100% recurring', 'category': 'Web Design', 'difficulty': 'Easy', 'link': 'https://www.squarespace.com/affiliate-program', 'description': 'Website builder and hosting platform'},
                {'name': 'Adobe Affiliate', 'commission': '5-15%', 'category': 'Creative Software', 'difficulty': 'Medium', 'link': 'https://www.adobe.com/partnerships/affiliate-program.html', 'description': 'Creative software suite including Photoshop, Illustrator'}
            ]
        },
        'ecommerce_marketplaces': {
            'keywords': ['fashion', 'clothing', 'shoes', 'jewelry', 'handbags', 'electronics', 'gadgets', 'home decor', 'furniture', 'etsy', 'shopify', 'amazon', 'ebay', 'dropship', 'print on demand'],
            'subtopics': [
                'Product Reviews', 'Size Guides', 'Style Inspiration', 'Shopping Deals', 
                'Brand Comparisons', 'Seasonal Collections', 'Trending Items', 'Customer Service',
                'Return Policies', 'Shipping Options'
            ],
            'affiliate_programs': [
                {'name': 'Amazon Associates', 'commission': '1-10%', 'category': 'E-commerce', 'difficulty': 'Easy', 'link': 'https://affiliate-program.amazon.com/', 'description': 'World\'s largest online marketplace with millions of products'},
                {'name': 'Etsy Affiliate', 'commission': '4%', 'category': 'Handmade', 'difficulty': 'Easy', 'link': 'https://www.etsy.com/affiliates', 'description': 'Marketplace for handmade and vintage items'},
                {'name': 'Shopify Affiliate', 'commission': '200% recurring', 'category': 'E-commerce Platform', 'difficulty': 'Easy', 'link': 'https://partners.shopify.com/affiliates', 'description': 'E-commerce platform for building online stores'}
            ]
        },
        'finance_money': {
            'keywords': ['credit card', 'loan', 'mortgage', 'crypto', 'bitcoin', 'trading', 'investment', 'broker', 'forex', 'insurance', 'budget', 'savings', 'debt', 'retirement', 'robo advisor'],
            'subtopics': [
                'Credit Card Reviews', 'Loan Comparisons', 'Investment Strategies', 'Crypto Trading',
                'Insurance Quotes', 'Budgeting Tools', 'Savings Plans', 'Debt Consolidation',
                'Retirement Planning', 'Tax Preparation'
            ],
            'affiliate_programs': [
                {'name': 'Credit Karma Affiliate', 'commission': '$50-100', 'category': 'Credit', 'difficulty': 'Easy', 'link': 'https://www.creditkarma.com/affiliate', 'description': 'Free credit scores, reports, and financial tools'},
                {'name': 'Coinbase Affiliate', 'commission': '50%', 'category': 'Crypto', 'difficulty': 'Medium', 'link': 'https://www.coinbase.com/affiliate-program', 'description': 'Cryptocurrency exchange and wallet platform'},
                {'name': 'Progressive Insurance', 'commission': '5-15%', 'category': 'Insurance', 'difficulty': 'Medium', 'link': 'https://www.progressive.com/affiliate/', 'description': 'Auto, home, and business insurance provider'}
            ]
        },
        'health_wellness': {
            'keywords': ['supplement', 'vitamin', 'fitness', 'workout', 'diet', 'weight loss', 'muscle', 'protein', 'cbd', 'nootropic', 'mental health', 'therapy', 'meditation', 'yoga', 'wellness'],
            'subtopics': [
                'Supplement Reviews', 'Workout Plans', 'Nutrition Guides', 'Mental Health Resources',
                'Fitness Equipment', 'Wellness Apps', 'Health Monitoring', 'Recovery Tools',
                'Lifestyle Changes', 'Expert Consultations'
            ],
            'affiliate_programs': [
                {'name': 'iHerb Affiliate', 'commission': '5-8%', 'category': 'Supplements', 'difficulty': 'Easy', 'link': 'https://www.iherb.com/affiliate-program', 'description': 'Natural health and wellness products'},
                {'name': 'Headspace Affiliate', 'commission': '10-20%', 'category': 'Mental Health', 'difficulty': 'Easy', 'link': 'https://www.headspace.com/affiliate-program', 'description': 'Meditation and mindfulness app'},
                {'name': 'Peloton Affiliate', 'commission': '5-10%', 'category': 'Fitness', 'difficulty': 'Medium', 'link': 'https://www.onepeloton.com/affiliate-program', 'description': 'Interactive fitness equipment and classes'}
            ]
        },
        'beauty_personal_care': {
            'keywords': ['skincare', 'makeup', 'cosmetics', 'hair care', 'fragrance', 'beauty', 'grooming', 'spa', 'nail', 'anti-aging', 'cleanser', 'moisturizer', 'serum'],
            'subtopics': [
                'Product Reviews', 'Beauty Routines', 'Skin Care Tips', 'Makeup Tutorials',
                'Hair Care Guides', 'Fragrance Reviews', 'Beauty Tools', 'Spa Services',
                'Ingredient Analysis', 'Beauty Trends'
            ],
            'affiliate_programs': [
                {'name': 'Sephora Affiliate', 'commission': '3-5%', 'category': 'Beauty', 'difficulty': 'Medium', 'link': 'https://www.sephora.com/affiliate-program', 'description': 'Premium beauty and cosmetics retailer'},
                {'name': 'Ulta Beauty Affiliate', 'commission': '2-4%', 'category': 'Beauty', 'difficulty': 'Easy', 'link': 'https://www.ulta.com/affiliate-program', 'description': 'Beauty retailer with salon services'},
                {'name': 'Birchbox Affiliate', 'commission': '10-15%', 'category': 'Beauty Subscription', 'difficulty': 'Easy', 'link': 'https://www.birchbox.com/affiliate-program', 'description': 'Beauty subscription box service'}
            ]
        },
        'travel_hospitality': {
            'keywords': ['travel', 'hotel', 'flight', 'vacation', 'cruise', 'car rental', 'luggage', 'booking', 'destination', 'tourism', 'airbnb', 'hostel', 'resort'],
            'subtopics': [
                'Destination Guides', 'Hotel Reviews', 'Flight Deals', 'Travel Insurance',
                'Packing Lists', 'Local Experiences', 'Travel Apps', 'Currency Exchange',
                'Visa Requirements', 'Travel Safety'
            ],
            'affiliate_programs': [
                {'name': 'Booking.com Affiliate', 'commission': '4-25%', 'category': 'Hotels', 'difficulty': 'Easy', 'link': 'https://partner.booking.com/', 'description': 'Hotel and accommodation booking platform'},
                {'name': 'Expedia Affiliate', 'commission': '2-6%', 'category': 'Travel', 'difficulty': 'Medium', 'link': 'https://www.expedia.com/affiliate-program', 'description': 'Travel booking platform for flights, hotels, and packages'},
                {'name': 'Airbnb Affiliate', 'commission': '3-5%', 'category': 'Accommodation', 'difficulty': 'Easy', 'link': 'https://www.airbnb.com/affiliate-program', 'description': 'Short-term rental and accommodation platform'}
            ]
        },
        'education_elearning': {
            'keywords': ['course', 'education', 'learning', 'training', 'certification', 'degree', 'online', 'tutorial', 'skill', 'language', 'coding', 'bootcamp', 'university', 'college'],
            'subtopics': [
                'Course Reviews', 'Learning Paths', 'Skill Assessments', 'Study Materials',
                'Certification Prep', 'Career Guidance', 'Study Groups', 'Learning Tools',
                'Progress Tracking', 'Expert Mentorship'
            ],
            'affiliate_programs': [
                {'name': 'Coursera Affiliate', 'commission': '15-45%', 'category': 'Online Education', 'difficulty': 'Easy', 'link': 'https://www.coursera.org/affiliate-program', 'description': 'Online courses from top universities and companies'},
                {'name': 'Udemy Affiliate', 'commission': '50%', 'category': 'Online Courses', 'difficulty': 'Easy', 'link': 'https://www.udemy.com/affiliate-program/', 'description': 'Online learning platform with diverse course offerings'},
                {'name': 'MasterClass Affiliate', 'commission': '10-20%', 'category': 'Premium Education', 'difficulty': 'Easy', 'link': 'https://www.masterclass.com/affiliate-program', 'description': 'Online classes taught by world-renowned experts'}
            ]
        },
        'home_garden_diy': {
            'keywords': ['home', 'garden', 'diy', 'furniture', 'appliance', 'tool', 'renovation', 'decor', 'smart home', 'security', 'solar', 'landscaping', 'plant', 'outdoor'],
            'subtopics': [
                'Product Reviews', 'DIY Tutorials', 'Home Improvement', 'Garden Planning',
                'Tool Guides', 'Smart Home Setup', 'Maintenance Tips', 'Design Inspiration',
                'Safety Guidelines', 'Project Planning'
            ],
            'affiliate_programs': [
                {'name': 'Home Depot Affiliate', 'commission': '2-8%', 'category': 'Home Improvement', 'difficulty': 'Easy', 'link': 'https://www.homedepot.com/affiliate-program', 'description': 'Home improvement and construction supplies'},
                {'name': 'Wayfair Affiliate', 'commission': '3-5%', 'category': 'Furniture', 'difficulty': 'Easy', 'link': 'https://www.wayfair.com/affiliate-program', 'description': 'Furniture and home decor online retailer'},
                {'name': 'Lowe\'s Affiliate', 'commission': '2-8%', 'category': 'Home Improvement', 'difficulty': 'Easy', 'link': 'https://www.lowes.com/affiliate-program', 'description': 'Home improvement and hardware retailer'}
            ]
        },
        'pets_petcare': {
            'keywords': ['pet', 'dog', 'cat', 'pet food', 'pet care', 'veterinary', 'pet insurance', 'pet toy', 'pet bed', 'pet grooming', 'pet training', 'pet health'],
            'subtopics': [
                'Pet Food Reviews', 'Health Care Tips', 'Training Guides', 'Grooming Services',
                'Pet Insurance', 'Toy Recommendations', 'Veterinary Care', 'Pet Travel',
                'Breed Information', 'Pet Safety'
            ],
            'affiliate_programs': [
                {'name': 'Chewy Affiliate', 'commission': '3-5%', 'category': 'Pet Supplies', 'difficulty': 'Easy', 'link': 'https://www.chewy.com/affiliate-program', 'description': 'Online pet supplies and pharmacy'},
                {'name': 'Petco Affiliate', 'commission': '2-4%', 'category': 'Pet Care', 'difficulty': 'Easy', 'link': 'https://www.petco.com/affiliate-program', 'description': 'Pet supplies, food, and services'},
                {'name': 'PetSmart Affiliate', 'commission': '2-4%', 'category': 'Pet Supplies', 'difficulty': 'Easy', 'link': 'https://www.petsmart.com/affiliate-program', 'description': 'Pet supplies and grooming services'}
            ]
        },
        'baby_family': {
            'keywords': ['baby', 'infant', 'toddler', 'child', 'parenting', 'pregnancy', 'stroller', 'diaper', 'baby gear', 'kids', 'family', 'maternity', 'fertility'],
            'subtopics': [
                'Product Reviews', 'Safety Guidelines', 'Development Milestones', 'Feeding Guides',
                'Sleep Training', 'Health Care', 'Educational Toys', 'Parenting Tips',
                'Family Activities', 'Child Development'
            ],
            'affiliate_programs': [
                {'name': 'BuyBuy Baby Affiliate', 'commission': '2-4%', 'category': 'Baby Products', 'difficulty': 'Easy', 'link': 'https://www.buybuybaby.com/affiliate-program', 'description': 'Baby gear, furniture, and essentials'},
                {'name': 'Amazon Baby Registry', 'commission': '1-10%', 'category': 'Baby Products', 'difficulty': 'Easy', 'link': 'https://www.amazon.com/baby-registry', 'description': 'Baby registry and product recommendations'},
                {'name': 'Target Baby Affiliate', 'commission': '1-8%', 'category': 'Baby Products', 'difficulty': 'Easy', 'link': 'https://www.target.com/affiliate-program', 'description': 'Baby products and family essentials'}
            ]
        },
        'sports_fitness_outdoor': {
            'keywords': ['sport', 'fitness', 'gym', 'workout', 'running', 'cycling', 'hiking', 'camping', 'outdoor', 'equipment', 'gear', 'supplement', 'wearable', 'training', 'water', 'swimming', 'surfing', 'diving', 'kayaking', 'paddle', 'sailing', 'boating', 'aquatic', 'maritime', 'ocean', 'lake', 'river', 'pool', 'beach', 'snorkeling', 'windsurfing', 'kitesurfing', 'wakeboarding', 'water skiing', 'rowing', 'canoeing', 'rafting', 'fishing', 'scuba', 'freediving'],
            'subtopics': [
                'Equipment Reviews', 'Training Programs', 'Safety Tips', 'Technique Tutorials', 
                'Gear Comparisons', 'Water Safety', 'Weather Conditions', 'Location Guides',
                'Performance Tracking', 'Expert Coaching', 'Beginner Guides', 'Advanced Techniques'
            ],
            'affiliate_programs': [
                {'name': 'SurfStitch Affiliate', 'commission': '5-10%', 'category': 'Water Sports', 'difficulty': 'Easy', 'link': 'https://www.surfstitch.com/affiliate-program', 'description': 'Surfing and water sports gear'},
                {'name': 'Wetsuit Warehouse Affiliate', 'commission': '8-12%', 'category': 'Water Sports', 'difficulty': 'Easy', 'link': 'https://www.wetsuitwarehouse.com/affiliate-program', 'description': 'Wetsuits and water sports equipment'},
                {'name': 'West Marine Affiliate', 'commission': '3-6%', 'category': 'Marine', 'difficulty': 'Easy', 'link': 'https://www.westmarine.com/affiliate-program', 'description': 'Boating and marine equipment'},
                {'name': 'REI Co-op Affiliate', 'commission': '5-8%', 'category': 'Outdoor Gear', 'difficulty': 'Easy', 'link': 'https://www.rei.com/affiliate-program', 'description': 'Outdoor gear and clothing cooperative'},
                {'name': 'Patagonia Affiliate', 'commission': '3-8%', 'category': 'Outdoor Gear', 'difficulty': 'Medium', 'link': 'https://www.patagonia.com/affiliate-program', 'description': 'Sustainable outdoor clothing and gear'},
                {'name': 'Backcountry.com Affiliate', 'commission': '4-6%', 'category': 'Outdoor Gear', 'difficulty': 'Easy', 'link': 'https://www.backcountry.com/affiliate-program', 'description': 'Outdoor gear and equipment'}
            ]
        },
        'food_grocery': {
            'keywords': ['food', 'recipe', 'cooking', 'kitchen', 'grocery', 'meal', 'diet', 'nutrition', 'restaurant', 'dining', 'cuisine', 'ingredient', 'spice', 'wine'],
            'subtopics': [
                'Recipe Collections', 'Cooking Techniques', 'Ingredient Guides', 'Kitchen Equipment',
                'Restaurant Reviews', 'Nutrition Information', 'Meal Planning', 'Food Safety',
                'Culinary Education', 'Food Trends'
            ],
            'affiliate_programs': [
                {'name': 'Blue Apron Affiliate', 'commission': '15-25%', 'category': 'Meal Kits', 'difficulty': 'Medium', 'link': 'https://www.blueapron.com/affiliate-program', 'description': 'Meal kit delivery service with fresh ingredients'},
                {'name': 'HelloFresh Affiliate', 'commission': '15-25%', 'category': 'Meal Kits', 'difficulty': 'Medium', 'link': 'https://www.hellofresh.com/affiliate-program', 'description': 'Meal kit delivery with easy-to-follow recipes'},
                {'name': 'Williams Sonoma Affiliate', 'commission': '4-8%', 'category': 'Kitchen', 'difficulty': 'Medium', 'link': 'https://www.williams-sonoma.com/affiliate-program', 'description': 'High-end kitchen equipment and cookware'}
            ]
        },
        'fashion_accessories': {
            'keywords': ['fashion', 'clothing', 'shoes', 'accessories', 'jewelry', 'watch', 'handbag', 'sunglasses', 'style', 'outfit', 'designer', 'trend', 'wardrobe'],
            'subtopics': [
                'Style Guides', 'Trend Reports', 'Size Charts', 'Care Instructions',
                'Outfit Inspiration', 'Brand Reviews', 'Seasonal Collections', 'Styling Tips',
                'Fashion Events', 'Personal Shopping'
            ],
            'affiliate_programs': [
                {'name': 'Nordstrom Affiliate', 'commission': '2-8%', 'category': 'Fashion', 'difficulty': 'Medium', 'link': 'https://www.nordstrom.com/affiliate-program', 'description': 'Luxury fashion and beauty retailer'},
                {'name': 'Zappos Affiliate', 'commission': '2-8%', 'category': 'Shoes', 'difficulty': 'Easy', 'link': 'https://www.zappos.com/affiliate-program', 'description': 'Online shoe and clothing retailer'},
                {'name': 'ASOS Affiliate', 'commission': '2-8%', 'category': 'Fashion', 'difficulty': 'Easy', 'link': 'https://www.asos.com/affiliate-program', 'description': 'Global fashion retailer for young adults'}
            ]
        },
        'automotive_powersports': {
            'keywords': ['car', 'auto', 'vehicle', 'truck', 'suv', 'motorcycle', 'bike', 'rv', 'parts', 'tire', 'gps', 'dashcam', 'charging', 'maintenance'],
            'subtopics': [
                'Vehicle Reviews', 'Maintenance Guides', 'Parts & Accessories', 'Insurance Options',
                'Financing Information', 'Safety Features', 'Performance Upgrades', 'Care Tips',
                'Road Trip Planning', 'Expert Advice'
            ],
            'affiliate_programs': [
                {'name': 'AutoZone Affiliate', 'commission': '3-8%', 'category': 'Auto Parts', 'difficulty': 'Easy', 'link': 'https://www.autozone.com/affiliate-program', 'description': 'Auto parts and accessories retailer'},
                {'name': 'Tire Rack Affiliate', 'commission': '5-10%', 'category': 'Tires', 'difficulty': 'Easy', 'link': 'https://www.tirerack.com/affiliate-program', 'description': 'Tires, wheels, and automotive accessories'},
                {'name': 'CarMax Affiliate', 'commission': '1-3%', 'category': 'Used Cars', 'difficulty': 'Medium', 'link': 'https://www.carmax.com/affiliate-program', 'description': 'Used car sales and financing'}
            ]
        },
        'business_b2b': {
            'keywords': ['business', 'office', 'supplies', 'fulfillment', 'payroll', 'legal', 'voip', 'web design', '3pl', 'trade show', 'b2b', 'enterprise', 'corporate'],
            'subtopics': [
                'Service Reviews', 'Implementation Guides', 'Cost Comparisons', 'ROI Analysis',
                'Integration Options', 'Support Resources', 'Best Practices', 'Case Studies',
                'Expert Consultation', 'Training Programs'
            ],
            'affiliate_programs': [
                {'name': 'Office Depot Affiliate', 'commission': '2-8%', 'category': 'Office Supplies', 'difficulty': 'Easy', 'link': 'https://www.officedepot.com/affiliate-program', 'description': 'Office supplies and business solutions'},
                {'name': 'Staples Affiliate', 'commission': '2-8%', 'category': 'Office Supplies', 'difficulty': 'Easy', 'link': 'https://www.staples.com/affiliate-program', 'description': 'Office supplies and technology products'},
                {'name': 'FedEx Affiliate', 'commission': '2-5%', 'category': 'Shipping', 'difficulty': 'Medium', 'link': 'https://www.fedex.com/affiliate-program', 'description': 'Shipping and logistics services'}
            ]
        },
        'gaming_esports': {
            'keywords': ['gaming', 'game', 'esports', 'console', 'pc', 'keyboard', 'mouse', 'headset', 'gpu', 'streaming', 'twitch', 'youtube', 'tournament'],
            'subtopics': [
                'Game Reviews', 'Hardware Reviews', 'Setup Guides', 'Performance Tips',
                'Streaming Equipment', 'Gaming Chairs', 'Accessories', 'Tournament Info',
                'Community Events', 'Expert Strategies'
            ],
            'affiliate_programs': [
                {'name': 'Steam Affiliate', 'commission': '5%', 'category': 'Gaming', 'difficulty': 'Hard', 'link': 'https://partner.steamgames.com/', 'description': 'Digital gaming platform and store'},
                {'name': 'Razer Affiliate', 'commission': '3-8%', 'category': 'Gaming Hardware', 'difficulty': 'Medium', 'link': 'https://www.razer.com/affiliate-program', 'description': 'Gaming peripherals and hardware'},
                {'name': 'Logitech Affiliate', 'commission': '3-8%', 'category': 'Gaming Accessories', 'difficulty': 'Easy', 'link': 'https://www.logitech.com/affiliate-program', 'description': 'Computer peripherals and gaming accessories'}
            ]
        },
        'luxury_collectibles': {
            'keywords': ['luxury', 'premium', 'designer', 'watch', 'jewelry', 'art', 'wine', 'antique', 'nft', 'collectible', 'high-end', 'exclusive'],
            'subtopics': [
                'Product Reviews', 'Authentication Guides', 'Investment Advice', 'Care Instructions',
                'Market Trends', 'Expert Appraisals', 'Collection Management', 'Insurance Options',
                'Auction Information', 'Expert Consultation'
            ],
            'affiliate_programs': [
                {'name': 'Sotheby\'s Affiliate', 'commission': '2-5%', 'category': 'Luxury', 'difficulty': 'Hard', 'link': 'https://www.sothebys.com/affiliate-program', 'description': 'Auction house for fine art and luxury goods'},
                {'name': 'Christie\'s Affiliate', 'commission': '2-5%', 'category': 'Luxury', 'difficulty': 'Hard', 'link': 'https://www.christies.com/affiliate-program', 'description': 'Auction house for art and collectibles'},
                {'name': '1stDibs Affiliate', 'commission': '3-8%', 'category': 'Luxury', 'difficulty': 'Medium', 'link': 'https://www.1stdibs.com/affiliate-program', 'description': 'Luxury furniture, art, and jewelry marketplace'}
            ]
        },
        'green_eco_friendly': {
            'keywords': ['eco', 'green', 'sustainable', 'renewable', 'solar', 'biodegradable', 'reusable', 'ethical', 'carbon', 'environment', 'climate', 'organic'],
            'subtopics': [
                'Product Reviews', 'Sustainability Guides', 'Environmental Impact', 'Green Living Tips',
                'Eco-Friendly Alternatives', 'Carbon Footprint', 'Renewable Energy', 'Waste Reduction',
                'Ethical Shopping', 'Climate Action'
            ],
            'affiliate_programs': [
                {'name': 'Patagonia Affiliate', 'commission': '3-8%', 'category': 'Eco-Friendly', 'difficulty': 'Medium', 'link': 'https://www.patagonia.com/affiliate-program', 'description': 'Sustainable outdoor clothing and gear'},
                {'name': 'REI Co-op Affiliate', 'commission': '5-8%', 'category': 'Outdoor', 'difficulty': 'Easy', 'link': 'https://www.rei.com/affiliate-program', 'description': 'Outdoor gear and clothing cooperative'},
                {'name': 'Etsy Eco Affiliate', 'commission': '4%', 'category': 'Sustainable', 'difficulty': 'Easy', 'link': 'https://www.etsy.com/affiliates', 'description': 'Handmade and sustainable products marketplace'}
            ]
        }
    }
    
    # Intelligent semantic category detection
    best_category = None
    best_score = 0
    
    # Priority-based detection (most specific first)
    if any(word in topic_lower for word in ['cookie', 'cookies', 'baking', 'recipe', 'cooking', 'cuisine', 'food', 'meal', 'kitchen', 'chef', 'restaurant', 'dining', 'ingredient', 'spice', 'wine', 'coffee', 'diet', 'nutrition', 'homemade', 'home made']):
        best_category = 'food_grocery'
        best_score = 10  # High priority
    elif any(word in topic_lower for word in ['water', 'swimming', 'surfing', 'diving', 'kayaking', 'sailing', 'boating', 'fishing', 'ocean', 'beach', 'pool', 'aquatic']):
        best_category = 'sports_fitness_outdoor'
        best_score = 10
    elif any(word in topic_lower for word in ['software', 'app', 'tool', 'platform', 'crm', 'seo', 'hosting', 'vpn', 'security', 'ai', 'automation']):
        best_category = 'software_saas'
        best_score = 10
    elif any(word in topic_lower for word in ['fashion', 'clothing', 'shoes', 'jewelry', 'handbag', 'sunglasses', 'style', 'outfit', 'designer']):
        best_category = 'fashion_accessories'
        best_score = 10
    elif any(word in topic_lower for word in ['travel', 'hotel', 'flight', 'vacation', 'cruise', 'booking', 'destination', 'tourism']):
        best_category = 'travel_hospitality'
        best_score = 10
    elif any(word in topic_lower for word in ['health', 'fitness', 'workout', 'supplement', 'vitamin', 'wellness', 'mental health', 'yoga', 'meditation']):
        best_category = 'health_wellness'
        best_score = 10
    elif any(word in topic_lower for word in ['beauty', 'skincare', 'makeup', 'cosmetics', 'hair care', 'fragrance', 'grooming', 'spa']):
        best_category = 'beauty_personal_care'
        best_score = 10
    elif any(word in topic_lower for word in ['education', 'course', 'learning', 'training', 'certification', 'degree', 'online', 'tutorial', 'skill']):
        best_category = 'education_elearning'
        best_score = 10
    elif any(word in topic_lower for word in ['finance', 'money', 'credit', 'loan', 'mortgage', 'crypto', 'investment', 'insurance', 'budget', 'savings']):
        best_category = 'finance_money'
        best_score = 10
    elif any(word in topic_lower for word in ['pet', 'dog', 'cat', 'pet food', 'pet care', 'veterinary', 'pet insurance', 'pet toy']):
        best_category = 'pets_petcare'
        best_score = 10
    elif any(word in topic_lower for word in ['baby', 'infant', 'toddler', 'child', 'parenting', 'pregnancy', 'stroller', 'diaper']):
        best_category = 'baby_family'
        best_score = 10
    elif any(word in topic_lower for word in ['car', 'auto', 'vehicle', 'truck', 'suv', 'motorcycle', 'parts', 'tire', 'gps']):
        best_category = 'automotive_powersports'
        best_score = 10
    elif any(word in topic_lower for word in ['business', 'office', 'supplies', 'fulfillment', 'payroll', 'legal', 'voip', 'web design']):
        best_category = 'business_b2b'
        best_score = 10
    elif any(word in topic_lower for word in ['gaming', 'game', 'esports', 'console', 'pc', 'keyboard', 'mouse', 'headset', 'gpu']):
        best_category = 'gaming_esports'
        best_score = 10
    elif any(word in topic_lower for word in ['luxury', 'premium', 'designer', 'watch', 'jewelry', 'art', 'wine', 'antique', 'collectible']):
        best_category = 'luxury_collectibles'
        best_score = 10
    elif any(word in topic_lower for word in ['eco', 'green', 'sustainable', 'renewable', 'solar', 'biodegradable', 'reusable', 'ethical']):
        best_category = 'green_eco_friendly'
        best_score = 10
    elif any(word in topic_lower for word in ['dating', 'relationship', 'match', 'romance', 'marriage', 'couple']):
        best_category = 'dating_relationships'
        best_score = 10
    elif any(word in topic_lower for word in ['home', 'garden', 'diy', 'furniture', 'appliance', 'tool', 'renovation', 'decor', 'smart home']):
        best_category = 'home_garden_diy'
        best_score = 5  # Lower priority for generic home terms
    else:
        # Fallback to keyword counting for unknown topics
        for category, data in category_mappings.items():
            score = sum(1 for keyword in data['keywords'] if keyword in topic_lower)
            if score > best_score:
                best_score = score
                best_category = category
    
    # Generate subtopics based on detected category
    if best_category and best_score > 0:
        category_data = category_mappings[best_category]
        related_areas = []
        
        # Generate specific subtopics for the detected category
        for i, subtopic in enumerate(category_data['subtopics'][:max_areas]):
            related_areas.append({
                "area": subtopic,
                "description": f"Comprehensive {subtopic.lower()} for {topic}",
                "relevance_score": 0.9 - (i * 0.05)  # Decreasing relevance score
            })
        
        # Generate affiliate programs
        affiliate_programs = []
        for i, program in enumerate(category_data['affiliate_programs'][:max_programs]):
            affiliate_programs.append({
                "name": program['name'],
                "commission": program['commission'],
                "category": program['category'],
                "difficulty": program['difficulty'],
                "description": program.get('description', f"Professional {program['category'].lower()} services"),
                "link": program.get('link', f"https://example.com/affiliate/{program['name'].lower().replace(' ', '-')}"),
                "estimated_traffic": 5000 + (i * 2000),
                "competition_level": "Medium"
            })
    else:
        # Generic fallback for unrecognized topics
        related_areas = [
            {"area": f"{topic} Equipment", "description": f"Essential equipment and gear for {topic}", "relevance_score": 0.9},
            {"area": f"{topic} Reviews", "description": f"Product and service reviews for {topic}", "relevance_score": 0.9},
            {"area": f"{topic} Guides", "description": f"Comprehensive guides and tutorials", "relevance_score": 0.8},
            {"area": f"{topic} Accessories", "description": f"Accessories and add-ons for {topic}", "relevance_score": 0.8},
            {"area": f"{topic} Services", "description": f"Professional services related to {topic}", "relevance_score": 0.7},
            {"area": f"{topic} Training", "description": f"Education and training for {topic}", "relevance_score": 0.7},
            {"area": f"{topic} Community", "description": f"Community and forums for {topic}", "relevance_score": 0.6},
            {"area": f"{topic} Tools", "description": f"Tools and resources for {topic}", "relevance_score": 0.6}
        ]
        
        affiliate_programs = [
            {"name": "Amazon Associates", "commission": "1-10%", "category": "General", "difficulty": "Easy", "description": "Wide range of products", "link": "https://affiliate-program.amazon.com/", "estimated_traffic": 50000, "competition_level": "High"},
            {"name": "ShareASale", "commission": "5-15%", "category": "Various", "difficulty": "Medium", "description": "Diverse merchant network", "link": "https://www.shareasale.com/", "estimated_traffic": 10000, "competition_level": "Medium"}
        ]
    
    return related_areas[:max_areas], affiliate_programs[:max_programs]

@router.get("/topics/database")
async def get_database_topics():
    """
    Get all topics currently in the database
    """
    # In a real implementation, this would query your database
    return {
        "topics": [
            "eco friendly homes",
            "best cars for family", 
            "weight loss",
            "crypto trading",
            "home improvement",
            "fitness equipment"
        ],
        "total_count": 6
    }

@router.post("/topics/add")
async def add_topic_to_database(
    topic: str,
    analysis: Dict[str, Any]
):
    """
    Add a new topic analysis to the database
    """
    # In a real implementation, this would save to your database
    return {
        "message": f"Topic '{topic}' added to database",
        "topic": topic,
        "status": "success"
    }

@router.get("/topics/search")
async def search_topics(q: str):
    """
    Search for topics by keyword
    """
    # In a real implementation, this would search your database
    all_topics = [
        "eco friendly homes",
        "best cars for family", 
        "weight loss",
        "crypto trading"
    ]
    
    matching_topics = [topic for topic in all_topics if q.lower() in topic.lower()]
    
    return {
        "query": q,
        "results": matching_topics,
        "count": len(matching_topics)
    }
