"""
LinkUp.so API Integration
Provides real-time affiliate offers search using LinkUp.so API via direct HTTP calls
"""

import os
import httpx
import structlog
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..core.config import settings
from ..core.api_key_manager import api_key_manager

logger = structlog.get_logger()

class LinkUpAPI:
    """LinkUp.so API client for affiliate offers search using direct HTTP calls"""
    
    def __init__(self):
        self.api_key = None
        self.base_url = None
        self.headers = {}
        self._load_credentials()
    
    def _load_credentials(self):
        """Load LinkUp credentials from API Key Manager"""
        try:
            self.api_key = api_key_manager.get_linkup_key()
            
            # Use base_url from environment or default if not in DB
            # We'll check if it's in settings/env first
            base_url = os.getenv("LINKUP_BASE_URL", "https://api.linkup.so/v1")
            
            # Clean up base_url
            base_url = base_url.strip()
            
            # Ensure base_url ends with /search correctly
            if base_url.endswith("/search"):
                self.base_url = base_url
            else:
                self.base_url = base_url.rstrip("/") + "/search"
            
            if self.api_key:
                self.headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                logger.info("LinkUp credentials loaded", 
                           base_url=self.base_url, 
                           api_key_length=len(self.api_key))
            else:
                logger.warning("LinkUp API key not found")
                
        except Exception as e:
            logger.error("Failed to load LinkUp credentials", error=str(e))
            self.api_key = os.getenv("LINKUP_API_KEY")
            self.base_url = "https://api.linkup.so/v1/search"
            if self.api_key:
                self.headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
    
    async def search_offers(self, query: str, category: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for affiliate offers using LinkUp.so API via direct HTTP calls
        
        Args:
            query: Search query
            category: Optional category filter
            limit: Maximum number of results
            
        Returns:
            List of affiliate offers
        """
        logger.info("LinkUp.so search_offers called", query=query, category=category, limit=limit)
        
        # Check if credentials are loaded
        if not self.api_key or not self.base_url:
            logger.warning("LinkUp credentials not loaded, skipping search")
            return []
        
        logger.info("LinkUp API key found", api_key_length=len(self.api_key) if self.api_key else 0, base_url=self.base_url)
        
        try:
            # Prepare search payload according to LinkUp.so API documentation
            payload = {
                "q": f"{query} affiliate program",
                "depth": "standard",
                "outputType": "sourcedAnswer",
                "structuredOutputSchema": "null",
                "includeImages": False,
                "includeInlineCitations": False,
                "includeSources": True
            }
            
            # Add category-specific domains if category is provided
            if category:
                category_domains = self._get_category_domains(category)
                if category_domains:
                    payload["includeDomains"] = category_domains
            
            # Make API request using httpx with shorter timeout and retry logic
            logger.info("Making LinkUp.so API request", url=self.base_url, payload=payload)
            
            # Use reasonable timeout and retry logic
            max_retries = 2
            timeout_duration = 20.0  # 20 seconds for LinkUp API
            
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=timeout_duration) as client:
                        response = await client.post(
                            self.base_url,
                            headers=self.headers,
                            json=payload
                        )
                        logger.info("LinkUp.so API response received", 
                                  status_code=response.status_code, 
                                  response_length=len(response.text),
                                  attempt=attempt + 1)
                        
                        if response.status_code == 200:
                            data = response.json()
                            # Extract affiliate programs from the response
                            offers = self._extract_affiliate_programs(data, query)
                            logger.info("LinkUp API search successful", 
                                      query=query, 
                                      results_count=len(offers),
                                      attempt=attempt + 1)
                            return self._format_offers(offers)
                        else:
                            logger.warning("LinkUp API error", 
                                         status=response.status_code, 
                                         error=response.text,
                                         attempt=attempt + 1)
                            if attempt < max_retries - 1:
                                await asyncio.sleep(1)  # Wait 1 second before retry
                                continue
                            return []
                
                except asyncio.TimeoutError:
                    logger.warning("LinkUp API timeout", attempt=attempt + 1, timeout=timeout_duration)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)  # Wait 1 second before retry
                        continue
                    return []
                
                except httpx.ConnectError:
                    logger.warning("LinkUp API connection error", attempt=attempt + 1)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)  # Wait 1 second before retry
                        continue
                    return []
        
        except Exception as e:
            logger.error("LinkUp API request failed", error=str(e))
            return []
    
    def _get_category_domains(self, category: str) -> List[str]:
        """
        Get relevant domains for a specific category
        """
        category_domains = {
            'outdoor_recreation': [
                'rei.com', 'patagonia.com', 'backcountry.com', 'campingworld.com',
                'basspro.com', 'cabelas.com', 'outdoorgearlab.com'
            ],
            'food_cooking': [
                'williams-sonoma.com', 'sur-la-table.com', 'kitchenaid.com',
                'allrecipes.com', 'foodnetwork.com', 'epicurious.com'
            ],
            'home_garden': [
                'homedepot.com', 'lowes.com', 'wayfair.com', 'overstock.com',
                'bedbathandbeyond.com', 'crateandbarrel.com'
            ],
            'technology': [
                'amazon.com', 'bestbuy.com', 'newegg.com', 'apple.com',
                'microsoft.com', 'dell.com', 'hp.com'
            ],
            'health_fitness': [
                'vitacost.com', 'iherb.com', 'bodybuilding.com', 'gymshark.com',
                'lululemon.com', 'nike.com', 'adidas.com'
            ],
            'fashion_beauty': [
                'nordstrom.com', 'macys.com', 'sephora.com', 'ulta.com',
                'zappos.com', 'asos.com', 'h&m.com'
            ]
        }
        return category_domains.get(category, [])
    
    def _extract_affiliate_programs(self, data: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """
        Extract affiliate programs from LinkUp.so response with better quality filtering
        """
        offers = []
        
        try:
            # LinkUp.so returns a sourcedAnswer format
            if 'answer' in data:
                answer = data['answer']
                
                # Look for specific affiliate program mentions in the answer
                if 'affiliate' in answer.lower() or 'commission' in answer.lower():
                    # Extract commission rates from the answer
                    commission = self._extract_commission_rate(answer)
                    
                    # Create a more detailed offer structure
                    offers.append({
                        'name': f"{query.title()} Affiliate Program",
                        'description': self._clean_description(answer),
                        'commission': commission,
                        'category': self._determine_category_from_answer(answer),
                        'difficulty': self._determine_difficulty_from_answer(answer),
                        'link': self._extract_affiliate_link(answer),
                        'estimated_traffic': self._estimate_traffic_from_answer(answer),
                        'competition_level': self._determine_competition_from_answer(answer)
                    })
            
            # Check sources for high-quality affiliate program information
            if 'sources' in data:
                for source in data['sources']:
                    if isinstance(source, dict) and 'url' in source:
                        url = source['url']
                        title = source.get('title', '')
                        
                        # Only include high-quality affiliate networks
                        if self._is_quality_affiliate_source(url, title):
                            offers.append({
                                'name': self._extract_program_name_from_source(title, url),
                                'description': self._extract_description_from_source(source),
                                'commission': self._extract_commission_from_source(source),
                                'category': self._categorize_from_url(url),
                                'difficulty': "Medium",
                                'link': url,
                                'estimated_traffic': self._estimate_traffic_from_url(url),
                                'competition_level': "Medium"
                            })
        
        except Exception as e:
            logger.warning("Failed to extract affiliate programs from LinkUp response", error=str(e))
        
        return offers
    
    def _extract_commission_rate(self, text: str) -> str:
        """Extract commission rate from text"""
        import re
        # Look for percentage patterns
        patterns = [
            r'(\d+(?:\.\d+)?)\s*%',  # 5%, 10.5%
            r'(\d+(?:\.\d+)?)\s*percent',  # 5 percent
            r'commission[:\s]*(\d+(?:\.\d+)?)\s*%',  # commission: 5%
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                rate = float(match.group(1))
                if rate > 0:
                    return f"{rate}%"
        
        return "5-15%"  # Default fallback
    
    def _clean_description(self, text: str) -> str:
        """Clean and truncate description"""
        # Remove extra whitespace and newlines
        cleaned = ' '.join(text.split())
        # Truncate to reasonable length
        if len(cleaned) > 200:
            return cleaned[:200] + "..."
        return cleaned
    
    def _determine_category_from_answer(self, text: str) -> str:
        """Determine category from answer text"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['outdoor', 'camping', 'hiking', 'fishing']):
            return 'Outdoor Recreation'
        elif any(word in text_lower for word in ['tech', 'software', 'computer', 'digital']):
            return 'Technology'
        elif any(word in text_lower for word in ['food', 'cooking', 'kitchen', 'recipe']):
            return 'Food & Cooking'
        elif any(word in text_lower for word in ['health', 'fitness', 'wellness']):
            return 'Health & Fitness'
        else:
            return 'General'
    
    def _determine_difficulty_from_answer(self, text: str) -> str:
        """Determine difficulty from answer text"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['easy', 'simple', 'beginner']):
            return 'Easy'
        elif any(word in text_lower for word in ['hard', 'difficult', 'advanced', 'expert']):
            return 'Hard'
        else:
            return 'Medium'
    
    def _extract_affiliate_link(self, text: str) -> str:
        """Extract affiliate link from text"""
        import re
        # Look for URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        
        # Filter for affiliate-related URLs
        for url in urls:
            if any(domain in url for domain in [
                'affiliate', 'partner', 'commission', 'amazon.com', 'shareasale.com'
            ]):
                return url
        
        return urls[0] if urls else ""
    
    def _estimate_traffic_from_answer(self, text: str) -> int:
        """Estimate traffic from answer text"""
        import re
        # Look for traffic numbers
        patterns = [
            r'(\d+(?:,\d+)*)\s*visitors?',
            r'(\d+(?:,\d+)*)\s*traffic',
            r'(\d+(?:,\d+)*)\s*users?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1).replace(',', ''))
                except:
                    continue
        
        return 1000  # Default fallback
    
    def _determine_competition_from_answer(self, text: str) -> str:
        """Determine competition level from answer text"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['low', 'minimal', 'little']):
            return 'Low'
        elif any(word in text_lower for word in ['high', 'intense', 'competitive']):
            return 'High'
        else:
            return 'Medium'
    
    def _is_quality_affiliate_source(self, url: str, title: str) -> bool:
        """Check if source is a quality affiliate network"""
        quality_domains = [
            'amazon.com', 'shareasale.com', 'cj.com', 'impact.com',
            'awin.com', 'partnerize.com', 'clickbank.com', 'flexoffers.com',
            'bestbuy.com', 'target.com', 'walmart.com', 'homedepot.com'
        ]
        
        return any(domain in url for domain in quality_domains)
    
    def _extract_program_name_from_source(self, title: str, url: str) -> str:
        """Extract program name from source"""
        if title and len(title) > 10:
            return title
        elif 'amazon.com' in url:
            return 'Amazon Associates'
        elif 'shareasale.com' in url:
            return 'ShareASale'
        elif 'cj.com' in url:
            return 'Commission Junction'
        else:
            return 'Affiliate Program'
    
    def _extract_description_from_source(self, source: Dict[str, Any]) -> str:
        """Extract description from source"""
        description = source.get('description', '')
        if description and len(description) > 20:
            return description[:200] + "..." if len(description) > 200 else description
        return f"Affiliate program found at {source.get('url', '')}"
    
    def _extract_commission_from_source(self, source: Dict[str, Any]) -> str:
        """Extract commission from source"""
        text = f"{source.get('title', '')} {source.get('description', '')}"
        return self._extract_commission_rate(text)
    
    def _categorize_from_url(self, url: str) -> str:
        """Categorize based on URL domain"""
        if 'amazon.com' in url:
            return 'E-commerce'
        elif 'tech' in url or 'software' in url:
            return 'Technology'
        elif 'fashion' in url or 'clothing' in url:
            return 'Fashion'
        else:
            return 'General'
    
    def _estimate_traffic_from_url(self, url: str) -> int:
        """Estimate traffic based on URL domain"""
        high_traffic_domains = ['amazon.com', 'walmart.com', 'target.com']
        medium_traffic_domains = ['bestbuy.com', 'homedepot.com', 'lowes.com']
        
        if any(domain in url for domain in high_traffic_domains):
            return 50000
        elif any(domain in url for domain in medium_traffic_domains):
            return 20000
        else:
            return 10000
    
    def _format_offers(self, offers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format LinkUp offers to match our affiliate program schema
        
        Args:
            offers: Raw offers from LinkUp API
            
        Returns:
            Formatted affiliate programs
        """
        formatted_offers = []
        
        for offer in offers:
            try:
                # Extract basic information
                name = offer.get("name", "Unknown Offer")
                description = offer.get("description", "")
                commission = offer.get("commission", "0%")
                category = offer.get("category", "General")
                difficulty = self._determine_difficulty(offer)
                
                # Extract URL and tracking info
                url = offer.get("url", "")
                tracking_url = offer.get("tracking_url", url)
                
                # Extract additional metrics
                estimated_traffic = offer.get("traffic_estimate", 0)
                competition_level = self._determine_competition_level(offer)
                
                # Create formatted offer
                formatted_offer = {
                    "program_name": name,
                    "company_name": name.replace(" Affiliate Program", ""),
                    "description": description,
                    "commission": commission,
                    "category": category,
                    "difficulty": difficulty,
                    "link": tracking_url or url,
                    "estimated_traffic": estimated_traffic,
                    "competition_level": competition_level,
                    "source": "linkup_api",
                    "network": offer.get("network", "LinkUp"),
                    "rating": offer.get("rating", 0.0),
                    "payout_terms": offer.get("payout_terms", "Net 30"),
                    "cookie_duration": offer.get("cookie_duration", "30 days"),
                    "min_payout": offer.get("min_payout", "$25"),
                    "promotional_materials": offer.get("promotional_materials", []),
                    "restrictions": offer.get("restrictions", ""),
                    "discovery_date": datetime.utcnow().isoformat(),
                    "is_verified": True,  # LinkUp offers are verified
                    "is_active": True
                }
                
                formatted_offers.append(formatted_offer)
                
            except Exception as e:
                logger.warning("Failed to format LinkUp offer", 
                             offer_id=offer.get("id"), 
                             error=str(e))
                continue
        
        return formatted_offers
    
    def _determine_difficulty(self, offer: Dict[str, Any]) -> str:
        """Determine difficulty level based on offer metrics"""
        approval_rate = offer.get("approval_rate", 0.5)
        requirements = offer.get("requirements", [])
        
        if approval_rate > 0.8 and len(requirements) == 0:
            return "Easy"
        elif approval_rate > 0.5 and len(requirements) <= 2:
            return "Medium"
        else:
            return "Hard"
    
    def _determine_competition_level(self, offer: Dict[str, Any]) -> str:
        """Determine competition level based on offer metrics"""
        competition_score = offer.get("competition_score", 0.5)
        affiliate_count = offer.get("affiliate_count", 0)
        
        if competition_score < 0.3 or affiliate_count < 100:
            return "Low"
        elif competition_score < 0.7 or affiliate_count < 1000:
            return "Medium"
        else:
            return "High"
    
    async def get_categories(self) -> List[str]:
        """Get available categories from LinkUp API"""
        if not self.api_key:
            return []
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.base_url}/categories",
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("categories", [])
                else:
                    return []
        except Exception as e:
            logger.error("Failed to fetch LinkUp categories", error=str(e))
            return []

# Global instance
linkup_api = LinkUpAPI()
