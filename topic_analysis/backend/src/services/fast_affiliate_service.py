"""
Fast Affiliate Research Service
A simplified, fast version that returns mock data without external API calls
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog

logger = structlog.get_logger()

class FastAffiliateService:
    """Fast affiliate research service with mock data - no external API calls"""
    
    def __init__(self):
        self.cache_ttl = 3600  # 1 hour cache TTL
    
    async def search_affiliate_programs(
        self, 
        search_term: str,
        niche: Optional[str] = None,
        budget_range: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fast affiliate program search with mock data
        """
        logger.info("Fast affiliate search started", 
                   search_term=search_term, niche=niche, budget_range=budget_range)
        
        try:
            # Simulate a small delay (100ms instead of 4+ minutes)
            await asyncio.sleep(0.1)
            
            # Generate mock programs based on search term
            programs = self._generate_mock_programs(search_term, niche)
            
            # Generate analysis result
            analysis_result = self._generate_mock_analysis(search_term, niche)
            
            # Prepare result
            result = {
                "search_term": search_term,
                "niche": niche,
                "budget_range": budget_range,
                "programs": programs,
                "analysis": analysis_result,
                "research_id": f"fast_research_{hash(search_term)}",
                "total_programs": len(programs),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info("Fast affiliate search completed", 
                       search_term=search_term, 
                       programs_found=len(programs))
            
            return result
            
        except Exception as e:
            logger.error("Fast affiliate search failed", error=str(e))
            raise
    
    def _generate_mock_programs(self, search_term: str, niche: Optional[str]) -> List[Dict[str, Any]]:
        """Generate realistic mock affiliate programs based on search term"""
        
        # Base programs that work for any search term
        base_programs = [
            {
                "id": "amazon-associates",
                "name": "Amazon Associates Program",
                "description": "Earn commissions promoting millions of products on Amazon. One of the most popular affiliate programs with reliable payouts.",
                "commission_rate": "1-10%",
                "network": "Amazon Associates",
                "epc": "2.50",
                "link": "https://affiliate-program.amazon.com/",
                "source": "mock",
                "category": "E-commerce",
                "difficulty": "Easy",
                "estimated_traffic": 50000,
                "competition_level": "High"
            },
            {
                "id": "shareasale-generic",
                "name": "ShareASale Affiliate Network",
                "description": "Join thousands of merchants on ShareASale. Wide variety of products and services with competitive commission rates.",
                "commission_rate": "5-15%",
                "network": "ShareASale",
                "epc": "3.20",
                "link": "https://www.shareasale.com/",
                "source": "mock",
                "category": "Various",
                "difficulty": "Medium",
                "estimated_traffic": 15000,
                "competition_level": "Medium"
            },
            {
                "id": "clickbank-digital",
                "name": "ClickBank Digital Products",
                "description": "High-commission digital products and courses. Perfect for content creators in education and self-improvement niches.",
                "commission_rate": "20-75%",
                "network": "ClickBank",
                "epc": "8.50",
                "link": "https://www.clickbank.com/affiliate-program/",
                "source": "mock",
                "category": "Digital Products",
                "difficulty": "Easy",
                "estimated_traffic": 8000,
                "competition_level": "Medium"
            }
        ]
        
        # Category-specific programs
        category_programs = self._get_category_specific_programs(search_term, niche)
        
        # Combine and deduplicate
        all_programs = base_programs + category_programs
        unique_programs = self._deduplicate_programs(all_programs)
        
        # Limit to 8-12 programs
        return unique_programs[:12]
    
    def _get_category_specific_programs(self, search_term: str, niche: Optional[str]) -> List[Dict[str, Any]]:
        """Get programs specific to the search category"""
        search_lower = search_term.lower()
        niche_lower = (niche or "").lower()
        
        programs = []
        
        # Outdoor/Recreation programs
        if any(word in search_lower for word in ['outdoor', 'camping', 'hiking', 'fishing', 'hunting', 'telescope', 'astronomy', 'nature', 'adventure']):
            programs.extend([
                {
                    "id": "rei-coop",
                    "name": "REI Co-op Affiliate Program",
                    "description": "Outdoor gear and clothing cooperative. High-quality products for outdoor enthusiasts with competitive commission rates.",
                    "commission_rate": "5-8%",
                    "network": "ShareASale",
                    "epc": "4.50",
                    "link": "https://www.rei.com/affiliate-program",
                    "source": "mock",
                    "category": "Outdoor Gear",
                    "difficulty": "Easy",
                    "estimated_traffic": 12000,
                    "competition_level": "Medium"
                },
                {
                    "id": "patagonia-affiliate",
                    "name": "Patagonia Affiliate Program",
                    "description": "Sustainable outdoor clothing and gear. Premium brand with environmentally conscious products.",
                    "commission_rate": "3-8%",
                    "network": "Impact",
                    "epc": "3.80",
                    "link": "https://www.patagonia.com/affiliate-program",
                    "source": "mock",
                    "category": "Outdoor Clothing",
                    "difficulty": "Medium",
                    "estimated_traffic": 8000,
                    "competition_level": "Medium"
                },
                {
                    "id": "backcountry-affiliate",
                    "name": "Backcountry.com Affiliate Program",
                    "description": "Outdoor gear and equipment for serious adventurers. Wide selection of high-quality outdoor products.",
                    "commission_rate": "4-6%",
                    "network": "Impact",
                    "epc": "3.20",
                    "link": "https://www.backcountry.com/affiliate-program",
                    "source": "mock",
                    "category": "Outdoor Gear",
                    "difficulty": "Easy",
                    "estimated_traffic": 6000,
                    "competition_level": "Medium"
                }
            ])
        
        # Technology programs
        elif any(word in search_lower for word in ['tech', 'software', 'computer', 'gadget', 'digital', 'ai', 'programming', 'coding']):
            programs.extend([
                {
                    "id": "microsoft-affiliate",
                    "name": "Microsoft Affiliate Program",
                    "description": "Promote Microsoft products including Office, Windows, and Azure services. High-value products with good commission rates.",
                    "commission_rate": "3-8%",
                    "network": "Microsoft",
                    "epc": "5.20",
                    "link": "https://www.microsoft.com/affiliate-program",
                    "source": "mock",
                    "category": "Software",
                    "difficulty": "Medium",
                    "estimated_traffic": 20000,
                    "competition_level": "High"
                },
                {
                    "id": "adobe-affiliate",
                    "name": "Adobe Creative Cloud Affiliate",
                    "description": "Promote Adobe's creative software suite. High-value subscriptions with excellent commission potential.",
                    "commission_rate": "5-12%",
                    "network": "Adobe",
                    "epc": "8.50",
                    "link": "https://www.adobe.com/affiliate-program",
                    "source": "mock",
                    "category": "Creative Software",
                    "difficulty": "Medium",
                    "estimated_traffic": 15000,
                    "competition_level": "Medium"
                }
            ])
        
        # Health/Fitness programs
        elif any(word in search_lower for word in ['fitness', 'health', 'wellness', 'gym', 'workout', 'exercise', 'yoga', 'supplement']):
            programs.extend([
                {
                    "id": "nike-affiliate",
                    "name": "Nike Affiliate Program",
                    "description": "Promote Nike athletic wear and footwear. Global brand with high recognition and conversion rates.",
                    "commission_rate": "4-8%",
                    "network": "CJ Affiliate",
                    "epc": "3.50",
                    "link": "https://www.nike.com/affiliate-program",
                    "source": "mock",
                    "category": "Athletic Wear",
                    "difficulty": "Easy",
                    "estimated_traffic": 25000,
                    "competition_level": "High"
                },
                {
                    "id": "vitacost-affiliate",
                    "name": "Vitacost Health & Wellness",
                    "description": "Health supplements, vitamins, and wellness products. High commission rates for health-focused content.",
                    "commission_rate": "8-15%",
                    "network": "ShareASale",
                    "epc": "4.20",
                    "link": "https://www.vitacost.com/affiliate-program",
                    "source": "mock",
                    "category": "Health & Wellness",
                    "difficulty": "Easy",
                    "estimated_traffic": 8000,
                    "competition_level": "Medium"
                }
            ])
        
        # Food/Cooking programs
        elif any(word in search_lower for word in ['cooking', 'food', 'recipe', 'kitchen', 'chef', 'restaurant', 'coffee', 'baking']):
            programs.extend([
                {
                    "id": "williams-sonoma",
                    "name": "Williams Sonoma Affiliate Program",
                    "description": "High-end kitchen equipment and cookware. Premium products with excellent commission rates.",
                    "commission_rate": "4-8%",
                    "network": "ShareASale",
                    "epc": "6.50",
                    "link": "https://www.williams-sonoma.com/affiliate-program",
                    "source": "mock",
                    "category": "Kitchen & Cooking",
                    "difficulty": "Medium",
                    "estimated_traffic": 12000,
                    "competition_level": "Medium"
                },
                {
                    "id": "blue-apron-affiliate",
                    "name": "Blue Apron Meal Kits",
                    "description": "Meal kit delivery service with fresh ingredients and easy-to-follow recipes.",
                    "commission_rate": "15-25%",
                    "network": "Blue Apron",
                    "epc": "12.50",
                    "link": "https://www.blueapron.com/affiliate-program",
                    "source": "mock",
                    "category": "Food Delivery",
                    "difficulty": "Medium",
                    "estimated_traffic": 15000,
                    "competition_level": "High"
                }
            ])
        
        # Home/Garden programs
        elif any(word in search_lower for word in ['home', 'garden', 'diy', 'furniture', 'decor', 'renovation', 'interior']):
            programs.extend([
                {
                    "id": "wayfair-affiliate",
                    "name": "Wayfair Home & Garden",
                    "description": "Furniture and home decor with competitive prices. Wide selection for all home improvement needs.",
                    "commission_rate": "3-8%",
                    "network": "CJ Affiliate",
                    "epc": "4.80",
                    "link": "https://www.wayfair.com/affiliate-program",
                    "source": "mock",
                    "category": "Home & Garden",
                    "difficulty": "Easy",
                    "estimated_traffic": 18000,
                    "competition_level": "Medium"
                },
                {
                    "id": "homedepot-affiliate",
                    "name": "Home Depot Affiliate Program",
                    "description": "Home improvement and construction supplies. Trusted brand with high conversion rates.",
                    "commission_rate": "2-6%",
                    "network": "CJ Affiliate",
                    "epc": "3.20",
                    "link": "https://www.homedepot.com/affiliate-program",
                    "source": "mock",
                    "category": "Home Improvement",
                    "difficulty": "Easy",
                    "estimated_traffic": 30000,
                    "competition_level": "High"
                }
            ])
        
        return programs
    
    def _generate_mock_analysis(self, search_term: str, niche: Optional[str]) -> Dict[str, Any]:
        """Generate mock analysis result"""
        return {
            "topic": search_term,
            "category": self._detect_category(search_term, niche),
            "target_audience": "Content creators and marketers",
            "content_opportunities": [
                f"Best {search_term} products 2024",
                f"{search_term} reviews and comparisons",
                f"How to choose {search_term}",
                f"{search_term} buying guide",
                f"Top {search_term} brands",
                f"{search_term} for beginners"
            ],
            "affiliate_types": ["Product reviews", "Comparison guides", "Buying guides", "Tutorial content"],
            "competition_level": "Medium",
            "earnings_potential": "$200-2000/month",
            "related_areas": [
                {"area": f"{search_term} Equipment", "description": f"Essential gear for {search_term}", "relevance_score": 0.95},
                {"area": f"{search_term} Reviews", "description": f"Product reviews and comparisons", "relevance_score": 0.9},
                {"area": f"{search_term} Guides", "description": f"Comprehensive guides and tutorials", "relevance_score": 0.85},
                {"area": f"{search_term} Community", "description": f"Online communities and forums", "relevance_score": 0.7}
            ],
            "affiliate_programs": [
                {"name": f"{search_term} Affiliate Program", "commission": "5-15%", "category": "General", "difficulty": "Medium", "description": f"Affiliate program for {search_term} products", "link": f"https://www.{search_term.lower().replace(' ', '')}.com/affiliate", "estimated_traffic": 5000, "competition_level": "Medium"}
            ]
        }
    
    def _detect_category(self, search_term: str, niche: Optional[str]) -> str:
        """Detect category from search term"""
        search_lower = search_term.lower()
        
        if any(word in search_lower for word in ['outdoor', 'camping', 'hiking', 'fishing', 'telescope', 'astronomy']):
            return 'outdoor_recreation'
        elif any(word in search_lower for word in ['tech', 'software', 'computer', 'gadget', 'digital']):
            return 'technology'
        elif any(word in search_lower for word in ['fitness', 'health', 'wellness', 'gym', 'workout']):
            return 'health_fitness'
        elif any(word in search_lower for word in ['cooking', 'food', 'recipe', 'kitchen', 'coffee']):
            return 'food_cooking'
        elif any(word in search_lower for word in ['home', 'garden', 'diy', 'furniture', 'decor']):
            return 'home_garden'
        else:
            return 'general'
    
    def _deduplicate_programs(self, programs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate programs"""
        seen = set()
        unique_programs = []
        
        for program in programs:
            key = (program.get('name', ''), program.get('network', ''))
            if key not in seen:
                seen.add(key)
                unique_programs.append(program)
        
        return unique_programs
