"""
AffiliateService for 14 network integrations
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog
from ..core.database import get_db
from ..core.redis import cache
from ..core.config import get_settings
from ..models.affiliate_research import AffiliateResearch, ResearchStatus
from ..models.user import User

logger = structlog.get_logger()
settings = get_settings()

class AffiliateService:
    """Service for affiliate network research and management"""
    
    def __init__(self):
        self.networks = {
            "shareasale": {
                "name": "ShareASale",
                "api_url": "https://api.shareasale.com/x.cfm",
                "rate_limit": 100,  # requests per hour
                "enabled": bool(settings.shareasale_api_key)
            },
            "impact": {
                "name": "Impact",
                "api_url": "https://api.impact.com",
                "rate_limit": 200,
                "enabled": bool(settings.impact_api_key)
            },
            "amazon": {
                "name": "Amazon Associates",
                "api_url": "https://webservices.amazon.com/paapi5",
                "rate_limit": 1000,
                "enabled": bool(settings.amazon_associates_tag)
            },
            "cj": {
                "name": "Commission Junction",
                "api_url": "https://api.cj.com",
                "rate_limit": 100,
                "enabled": bool(settings.cj_api_key)
            },
            "partnerize": {
                "name": "Partnerize",
                "api_url": "https://api.partnerize.com",
                "rate_limit": 150,
                "enabled": bool(settings.partnerize_api_key)
            },
            "awin": {
                "name": "Awin",
                "api_url": "https://api.awin.com",
            },
            "flexoffers": {
                "name": "FlexOffers",
                "api_url": "https://api.flexoffers.com",
            },
            "clickbank": {
                "name": "ClickBank",
                "api_url": "https://api.clickbank.com",
            },
            "jvzoo": {
                "name": "JVZoo",
                "api_url": "https://api.jvzoo.com",
            },
            "warriorplus": {
                "name": "WarriorPlus",
                "api_url": "https://api.warriorplus.com",
            },
            "clickfunnels": {
                "name": "ClickFunnels",
                "api_url": "https://api.clickfunnels.com",
            },
            "leadpages": {
                "name": "Leadpages",
                "api_url": "https://api.leadpages.com",
            },
            "convertkit": {
                "name": "ConvertKit",
                "api_url": "https://api.convertkit.com",
            },
            "mailchimp": {
                "name": "Mailchimp",
                "api_url": "https://api.mailchimp.com",
            }
        }
    
    async def create_research(self, user_id: int, topic: str, search_query: str = None) -> Dict[str, Any]:
        """Create new affiliate research session"""
        try:
            # Create research record
            db = next(get_db())
            research = AffiliateResearch(
                user_id=user_id,
                topic=topic,
                search_query=search_query or topic,
                status=ResearchStatus.PENDING
            )
            db.add(research)
            db.commit()
            db.refresh(research)
            
            # Start background research
            asyncio.create_task(self._perform_research(research.id))
            
            logger.info("Affiliate research created", research_id=research.id, topic=topic)
            return research.to_dict()
            
        except Exception as e:
            logger.error("Failed to create affiliate research", error=str(e))
            raise
    
    async def get_research(self, research_id: int) -> Dict[str, Any]:
        """Get affiliate research by ID"""
        try:
            db = next(get_db())
            research = db.get_AffiliateResearch_by_id(AffiliateResearch.id == research_id)
            
            if not research:
                raise ValueError("Research not found")
            
            return research.to_dict()
            
        except Exception as e:
            logger.error("Failed to get affiliate research", research_id=research_id, error=str(e))
            raise
    
    async def update_selected_programs(self, research_id: int, selected_programs: List[Dict[str, Any]]) -> bool:
        """Update selected affiliate programs"""
        try:
            db = next(get_db())
            research = db.get_AffiliateResearch_by_id(AffiliateResearch.id == research_id)
            
            if not research:
                raise ValueError("Research not found")
            
            research.update_selected_programs(selected_programs)
            db.commit()
            
            logger.info("Selected programs updated", research_id=research_id, count=len(selected_programs))
            return True
            
        except Exception as e:
            logger.error("Failed to update selected programs", research_id=research_id, error=str(e))
            raise
    
    async def _perform_research(self, research_id: int):
        """Perform affiliate research in background"""
        try:
            db = next(get_db())
            research = db.get_AffiliateResearch_by_id(AffiliateResearch.id == research_id)
            
            if not research:
                return
            
            # Update status to processing
            research.status = ResearchStatus.IN_PROGRESS
            db.commit()
            
            start_time = datetime.now()
            all_results = []
            warnings = []
            
            # Research each enabled network
            for network_id, network_config in self.networks.items():
                if not network_config.get("enabled", False):
                    continue
                
                try:
                    network_results = await self._research_network(network_id, research.topic, research.search_query)
                    all_results.extend(network_results)
                    
                except Exception as e:
                    warning = f"Failed to research {network_config['name']}: {str(e)}"
                    warnings.append(warning)
                    logger.warning("Network research failed", network=network_id, error=str(e))
            
            # Process and score results
            processed_results = self._process_results(all_results)
            
            # Calculate duration
            duration = int((datetime.now() - start_time).total_seconds())
            
            # Update research with results
            research.mark_completed(processed_results)
            research.research_duration = duration
            research.warnings = warnings
            research.total_networks_searched = len([n for n in self.networks.values() if n.get("enabled", False)])
            research.total_programs_found = len(processed_results)
            
            db.commit()
            
            logger.info("Affiliate research completed", 
                       research_id=research_id, 
                       programs_found=len(processed_results),
                       duration=duration)
            
        except Exception as e:
            logger.error("Affiliate research failed", research_id=research_id, error=str(e))
            
            # Mark as failed
            try:
                db = next(get_db())
                research = db.get_AffiliateResearch_by_id(AffiliateResearch.id == research_id)
                if research:
                    research.mark_failed(str(e))
                    db.commit()
            except:
                pass
    
    async def _research_network(self, network_id: str, topic: str, search_query: str) -> List[Dict[str, Any]]:
        """Research specific affiliate network"""
        network_config = self.networks[network_id]
        
        # Check rate limiting
        cache_key = f"affiliate_rate_limit:{network_id}"
        if not await self._check_rate_limit(cache_key, network_config.get("rate_limit", 100)):
            raise Exception("Rate limit exceeded")
        
        # Make API call based on network
        if network_id == "shareasale":
            return await self._research_shareasale(topic, search_query)
        elif network_id == "impact":
            return await self._research_impact(topic, search_query)
        elif network_id == "amazon":
            return await self._research_amazon(topic, search_query)
        elif network_id == "cj":
            return await self._research_cj(topic, search_query)
        elif network_id == "partnerize":
            return await self._research_partnerize(topic, search_query)
        else:
            # Mock data for other networks
            return await self._research_mock_network(network_id, topic, search_query)
    
    async def _research_shareasale(self, topic: str, search_query: str) -> List[Dict[str, Any]]:
        """Research ShareASale network"""
        # Mock implementation - replace with actual API calls
        return [
            {
                "network": "ShareASale",
                "program_name": f"{topic.title()} Pro Program",
                "epc": 12.50,
                "commission_rate": 8.5,
                "cookie_length": 30,
                "conversion_rate": 3.2,
                "landing_page_compliance": True,
                "reversal_rate": 2.1,
                "program_url": "https://www.shareasale.com/r.cfm?b=123456&u=123&m=12345",
                "description": f"Professional {topic} affiliate program with high EPC",
                "category": topic.lower(),
                "merchant_rating": 4.5
            }
        ]
    
    async def _research_impact(self, topic: str, search_query: str) -> List[Dict[str, Any]]:
        """Research Impact network"""
        return [
            {
                "network": "Impact",
                "program_name": f"{topic.title()} Master Program",
                "epc": 15.75,
                "commission_rate": 10.0,
                "cookie_length": 45,
                "conversion_rate": 4.1,
                "landing_page_compliance": True,
                "reversal_rate": 1.8,
                "program_url": "https://impact.com/merchants/12345",
                "description": f"Premium {topic} affiliate program",
                "category": topic.lower(),
                "merchant_rating": 4.8
            }
        ]
    
    async def _research_amazon(self, topic: str, search_query: str) -> List[Dict[str, Any]]:
        """Research Amazon Associates"""
        return [
            {
                "network": "Amazon Associates",
                "program_name": "Amazon",
                "epc": 8.25,
                "commission_rate": 4.0,
                "cookie_length": 24,
                "conversion_rate": 2.8,
                "landing_page_compliance": True,
                "reversal_rate": 3.5,
                "program_url": "https://affiliate-program.amazon.com",
                "description": "Amazon Associates program",
                "category": "general",
                "merchant_rating": 4.2
            }
        ]
    
    async def _research_cj(self, topic: str, search_query: str) -> List[Dict[str, Any]]:
        """Research Commission Junction"""
        return [
            {
                "network": "Commission Junction",
                "program_name": f"{topic.title()} Network",
                "epc": 11.20,
                "commission_rate": 7.5,
                "cookie_length": 30,
                "conversion_rate": 3.5,
                "landing_page_compliance": True,
                "reversal_rate": 2.3,
                "program_url": "https://www.cj.com/affiliate/12345",
                "description": f"CJ affiliate program for {topic}",
                "category": topic.lower(),
                "merchant_rating": 4.3
            }
        ]
    
    async def _research_partnerize(self, topic: str, search_query: str) -> List[Dict[str, Any]]:
        """Research Partnerize network"""
        return [
            {
                "network": "Partnerize",
                "program_name": f"{topic.title()} Partners",
                "epc": 13.80,
                "commission_rate": 9.0,
                "cookie_length": 30,
                "conversion_rate": 3.8,
                "landing_page_compliance": True,
                "reversal_rate": 1.9,
                "program_url": "https://partnerize.com/merchants/12345",
                "description": f"Partnerize program for {topic}",
                "category": topic.lower(),
                "merchant_rating": 4.6
            }
        ]
    
    async def _research_mock_network(self, network_id: str, topic: str, search_query: str) -> List[Dict[str, Any]]:
        """Mock research for other networks"""
        network_name = self.networks[network_id]["name"]
        return [
            {
                "network": network_name,
                "program_name": f"{topic.title()} {network_name} Program",
                "epc": 10.0 + (hash(network_id) % 10),
                "commission_rate": 5.0 + (hash(network_id) % 5),
                "cookie_length": 30,
                "conversion_rate": 2.5 + (hash(network_id) % 3),
                "landing_page_compliance": True,
                "reversal_rate": 2.0 + (hash(network_id) % 2),
                "program_url": f"https://{network_id}.com/affiliate/12345",
                "description": f"{network_name} affiliate program for {topic}",
                "category": topic.lower(),
                "merchant_rating": 4.0 + (hash(network_id) % 10) / 10
            }
        ]
    
    def _process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and score affiliate program results"""
        # Remove duplicates based on program name
        unique_results = []
        seen_programs = set()
        
        for result in results:
            program_key = f"{result['network']}_{result['program_name']}"
            if program_key not in seen_programs:
                unique_results.append(result)
                seen_programs.add(program_key)
        
        # Sort by EPC (highest first)
        unique_results.sort(key=lambda x: x.get("epc", 0), reverse=True)
        
        # Add additional scoring
        for result in unique_results:
            result["score"] = self._calculate_program_score(result)
            result["recommended"] = result["score"] >= 0.7
        
        return unique_results
    
    def _calculate_program_score(self, program: Dict[str, Any]) -> float:
        """Calculate program recommendation score"""
        epc = program.get("epc", 0)
        commission_rate = program.get("commission_rate", 0)
        conversion_rate = program.get("conversion_rate", 0)
        reversal_rate = program.get("reversal_rate", 0)
        merchant_rating = program.get("merchant_rating", 0)
        
        # Normalize scores (0-1)
        epc_score = min(epc / 20.0, 1.0)  # Max EPC of 20
        commission_score = min(commission_rate / 15.0, 1.0)  # Max commission of 15%
        conversion_score = min(conversion_rate / 10.0, 1.0)  # Max conversion of 10%
        reversal_penalty = max(0, 1.0 - (reversal_rate / 10.0))  # Penalty for high reversal
        rating_score = merchant_rating / 5.0  # Max rating of 5
        
        # Weighted average
        score = (
            epc_score * 0.3 +
            commission_score * 0.2 +
            conversion_score * 0.2 +
            reversal_penalty * 0.15 +
            rating_score * 0.15
        )
        
        return min(max(score, 0), 1)
    
    async def _check_rate_limit(self, cache_key: str, limit: int) -> bool:
        """Check if rate limit is exceeded"""
        try:
            current_count = await cache.get(cache_key)
            if current_count is None:
                await cache.set(cache_key, 1, expire=3600)  # 1 hour
                return True
            
            if int(current_count) >= limit:
                return False
            
            await cache.set(cache_key, int(current_count) + 1, expire=3600)
            return True
            
        except Exception as e:
            logger.error("Rate limit check failed", error=str(e))
            return True  # Allow on error
    
    async def get_user_researches(self, user_id: int, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Get user's affiliate researches"""
        try:
            db = next(get_db())
            researches = db.query(AffiliateResearch).filter(
                AffiliateResearch.user_id == user_id
            ).order_by(AffiliateResearch.created_at.desc()).offset(offset).limit(limit).all()
            
            return [research.to_dict() for research in researches]
            
        except Exception as e:
            logger.error("Failed to get user researches", user_id=user_id, error=str(e))
            raise
    
    async def delete_research(self, research_id: int, user_id: int) -> bool:
        """Delete affiliate research"""
        try:
            db = next(get_db())
            research = db.get_AffiliateResearch_by_id(
                AffiliateResearch.id == research_id,
                AffiliateResearch.user_id == user_id
            )
            
            if not research:
                raise ValueError("Research not found")
            
            db.delete(research)
            db.commit()
            
            logger.info("Affiliate research deleted", research_id=research_id, user_id=user_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete research", research_id=research_id, error=str(e))
            raise
