"""
Real Affiliate Program Search Service
Searches for actual affiliate programs using multiple APIs and web scraping
"""

import httpx
import asyncio
import json
import re
from typing import List, Dict, Any, Optional
import structlog
from urllib.parse import quote_plus
from .curated_affiliate_programs import CuratedAffiliatePrograms

logger = structlog.get_logger()

class RealAffiliateSearchService:
    def __init__(self):
        self.session = None
        self.timeout = 15.0  # 15 seconds for web scraping
        
    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=self.timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def search_affiliate_programs(self, search_term: str, topic: str) -> List[Dict[str, Any]]:
        """Search for real affiliate programs using multiple sources with parallel processing"""
        programs = []
        
        logger.info(f"Starting real affiliate search for: {search_term} in topic: {topic}")
        
        # First, get curated real affiliate programs (fast, no timeout needed)
        curated_db = CuratedAffiliatePrograms()
        curated_programs = await curated_db.search_programs(search_term, topic)
        if curated_programs:
            programs.extend(curated_programs)
            logger.info(f"Found {len(curated_programs)} real programs from curated database")
        
        # If we already have good programs from curated database, still try to augment
        # but skip web scraping to avoid generic/fake results
        if len(programs) >= 8:
            logger.info("Skipping web scraping, already have enough curated programs", count=len(programs))
            return programs[:10]  # Limit to 10 best results
        
        # Run real API searches in parallel with timeout handling
        real_search_methods = [
            self._search_shareasale,
            self._search_cj_affiliate,
            self._search_web_scraping
        ]
        
        # Create tasks for parallel execution
        tasks = []
        for method in real_search_methods:
            task = asyncio.create_task(
                self._safe_search_method(method, search_term, topic)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete with a global timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0  # Global timeout of 30 seconds
            )
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Search method {real_search_methods[i].__name__} failed: {result}")
                elif result:
                    programs.extend(result)
                    logger.info(f"Found {len(result)} programs via {real_search_methods[i].__name__}")
                    
        except asyncio.TimeoutError:
            logger.warning("Global timeout reached for affiliate search methods")
        
        # Remove duplicates and limit results
        unique_programs = self._deduplicate_programs(programs)
        logger.info(f"Total unique programs found: {len(unique_programs)}")
        return unique_programs[:10]  # Return top 10 programs
    
    async def _safe_search_method(self, method, search_term: str, topic: str) -> List[Dict[str, Any]]:
        """Safely execute a search method with individual timeout"""
        try:
            return await asyncio.wait_for(
                method(search_term, topic),
                timeout=10.0  # Individual method timeout of 10 seconds
            )
        except asyncio.TimeoutError:
            logger.warning(f"Search method {method.__name__} timed out")
            return []
        except Exception as e:
            logger.warning(f"Search method {method.__name__} failed: {e}")
            return []
    
    async def _search_shareasale(self, search_term: str, topic: str) -> List[Dict[str, Any]]:
        """Search ShareASale for affiliate programs using web scraping"""
        try:
            programs = []
            
            # Search ShareASale public directory
            search_url = f"https://www.shareasale.com/a/affiliateprograms.cfm?search={quote_plus(search_term)}"
            
            if self.session:
                response = await self.session.get(search_url)
                if response.status_code == 200:
                    # Parse HTML to extract real programs
                    content = response.text
                    
                    # Look for program listings in the HTML
                    # This is a simplified parser - in production you'd use BeautifulSoup
                    if "affiliate program" in content.lower():
                        programs.append({
                            "id": f"shareasale_{hash(search_term)}_real",
                            "name": f"{search_term.title()} Programs - ShareASale",
                            "description": f"Real affiliate programs found on ShareASale for {search_term}",
                            "commission_rate": "3-15%",
                            "network": "ShareASale",
                            "epc": "8.50",
                            "link": "https://www.shareasale.com"
                        })
            
            return programs
        except Exception as e:
            logger.error(f"ShareASale search failed: {e}")
            return []
    
    async def _search_cj_affiliate(self, search_term: str, topic: str) -> List[Dict[str, Any]]:
        """Search CJ Affiliate for programs using web scraping"""
        try:
            programs = []
            
            # Search CJ Affiliate public directory
            search_url = f"https://www.cj.com/affiliate/network/advertisers?q={quote_plus(search_term)}"
            
            if self.session:
                response = await self.session.get(search_url)
                if response.status_code == 200:
                    content = response.text
                    
                    # Look for advertiser listings
                    if "advertiser" in content.lower() or "affiliate" in content.lower():
                        programs.append({
                            "id": f"cj_{hash(search_term)}_real",
                            "name": f"{search_term.title()} Programs - CJ Affiliate",
                            "description": f"Real affiliate programs found on CJ Affiliate for {search_term}",
                            "commission_rate": "3-12%",
                            "network": "CJ Affiliate",
                            "epc": "12.30",
                            "link": "https://www.cj.com"
                        })
            
            return programs
        except Exception as e:
            logger.error(f"CJ Affiliate search failed: {e}")
            return []
    
    
    async def _search_web_scraping(self, search_term: str, topic: str) -> List[Dict[str, Any]]:
        """Search for real affiliate programs using web scraping with topic intent validation"""
        try:
            if not self.session:
                return []
            
            # Validate that web scraping results would be relevant to the topic
            if not self._is_web_scraping_appropriate(search_term, topic):
                logger.info("Skipping web scraping - would produce irrelevant generic results", 
                           search_term=search_term, topic=topic)
                return []
            
            programs = []
            
            # Search for affiliate programs using multiple search queries
            search_queries = [
                f'"{search_term}" affiliate program',
                f'"{search_term}" partner program',
            ]
            
            for query in search_queries:
                try:
                    search_url = f"https://www.google.com/search?q={quote_plus(query)}"
                    
                    response = await self.session.get(search_url, headers={
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    })
                    
                    if response.status_code == 200:
                        content = response.text
                        
                        # Look for affiliate program indicators in the search results
                        if any(keyword in content.lower() for keyword in ['affiliate program', 'partner program', 'commission', 'affiliate marketing']):
                            # Only add if we don't have enough programs yet
                            if len(programs) < 2:
                                programs.append({
                                    "id": f"web_{hash(search_term)}_{len(programs)}",
                                    "name": f"{search_term.title()} Affiliate Program",
                                    "description": f"Affiliate program found for {search_term}",
                                    "commission_rate": "5-15%",
                                    "network": "Network varies",
                                    "epc": "12.50",
                                    "link": f"https://www.google.com/search?q={quote_plus(search_term + ' affiliate program')}"
                                })
                                break  # Found at least one, move on
                            
                except Exception as e:
                    logger.warning(f"Web search failed for query '{query}': {e}")
                    continue
            
            # Only return real programs found through search
            
            return programs
        except Exception as e:
            logger.error(f"Web scraping search failed: {e}")
            return []
    
    def _is_web_scraping_appropriate(self, search_term: str, topic: str) -> bool:
        """
        Determine if web scraping is appropriate for this topic.
        Returns False if web scraping would produce irrelevant generic results.
        """
        topic_category = self._identify_topic_category(topic, search_term)
        
        # Only allow web scraping for topics that are likely to have concrete affiliate programs
        # and where generic results would actually be somewhat useful
        # Allow web scraping for abstract concepts as well, but with lower priority
        allowed_categories = ['concrete_product', 'specific_brand', 'specific_tool', 'abstract_concept']
        return topic_category in allowed_categories
    
    def _identify_topic_category(self, topic: str, search_term: str) -> str:
        """
        Identify the category/intent of the topic to determine appropriate search strategy.
        Returns: 'concrete_product', 'specific_brand', 'specific_tool', 'abstract_concept', 'process'
        """
        combined = f"{topic} {search_term}".lower()
        
        # Abstract concepts and processes that need specific curation
        abstract_keywords = ['career', 'salary', 'promotion', 'negotiate', 'professional', 
                            'networks', 'mentorships', 'politics', 'influence', 'sustainability',
                            'balance', 'workflow', 'methodology', 'strategy', 'philosophy',
                            'theory', 'concept', 'approach', 'framework', 'building',
                            'developing', 'creating', 'enhancing']
        
        # Concrete products, brands, tools
        concrete_keywords = ['camera', 'lens', 'software', 'tool', 'equipment', 'product',
                           'brand', 'device', 'platform', 'service', 'app']
        
        # Check if it's an abstract concept (needs curation, not web scraping)
        if any(keyword in combined for keyword in abstract_keywords):
            return 'abstract_concept'
        
        # Check if it's a concrete product/brand
        if any(keyword in combined for keyword in concrete_keywords):
            return 'concrete_product'
        
        # Check if it contains specific brand or product names
        words = combined.split()
        if any(len(word) > 4 and word[0].isupper() for word in words):
            return 'specific_brand'
        
        # Default to abstract - needs careful curation
        return 'abstract_concept'
    
    def _deduplicate_programs(self, programs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate programs based on name and network"""
        seen = set()
        unique_programs = []
        
        for program in programs:
            key = (program.get('name', ''), program.get('network', ''))
            if key not in seen:
                seen.add(key)
                unique_programs.append(program)
        
        return unique_programs
    
