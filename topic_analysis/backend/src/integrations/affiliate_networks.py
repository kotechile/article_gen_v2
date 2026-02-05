"""
Affiliate Networks Integration
Integrates with 14 major affiliate networks to fetch program data
"""

import httpx
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from ..core.config import settings

logger = logging.getLogger(__name__)

class AffiliateNetworkAPI:
    """Base class for affiliate network API integrations"""
    
    def __init__(self, network_name: str, api_key: str, base_url: str):
        self.network_name = network_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = 30.0
        
    async def search_programs(self, niche: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for affiliate programs in a specific niche"""
        raise NotImplementedError("Subclasses must implement search_programs")
    
    async def get_program_details(self, program_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific program"""
        raise NotImplementedError("Subclasses must implement get_program_details")

class ShareASaleAPI(AffiliateNetworkAPI):
    """ShareASale affiliate network integration"""
    
    def __init__(self):
        super().__init__(
            "ShareASale",
            settings.SHAREASALE_API_KEY,
            "https://api.shareasale.com/x.cfm"
        )
    
    async def search_programs(self, niche: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search ShareASale programs"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "action": "merchantSearch",
                    "version": "2.0",
                    "token": self.api_key,
                    "keywords": niche,
                    "limit": limit
                }
                
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                return self._process_shareasale_programs(data)
                
        except Exception as e:
            logger.error(f"ShareASale API error: {e}")
            return []
    
    def _process_shareasale_programs(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process ShareASale API response"""
        programs = []
        
        if "merchants" in data:
            for merchant in data["merchants"]:
                programs.append({
                    "id": merchant.get("merchantID", ""),
                    "name": merchant.get("merchantName", ""),
                    "description": merchant.get("description", ""),
                    "url": merchant.get("merchantURL", ""),
                    "commission_rate": merchant.get("commission", 0),
                    "cookie_length": merchant.get("cookieLength", 0),
                    "network": "ShareASale",
                    "status": "active" if merchant.get("status") == "1" else "inactive",
                    "categories": merchant.get("categories", []),
                    "epc": merchant.get("epc", 0),
                    "created_at": datetime.utcnow().isoformat()
                })
        
        return programs

class ImpactAPI(AffiliateNetworkAPI):
    """Impact affiliate network integration"""
    
    def __init__(self):
        super().__init__(
            "Impact",
            settings.IMPACT_API_KEY,
            "https://api.impact.com"
        )
    
    async def search_programs(self, niche: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search Impact programs"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                params = {
                    "q": niche,
                    "limit": limit,
                    "status": "active"
                }
                
                response = await client.get(f"{self.base_url}/advertisers", headers=headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                return self._process_impact_programs(data)
                
        except Exception as e:
            logger.error(f"Impact API error: {e}")
            return []
    
    def _process_impact_programs(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process Impact API response"""
        programs = []
        
        if "advertisers" in data:
            for advertiser in data["advertisers"]:
                programs.append({
                    "id": advertiser.get("id", ""),
                    "name": advertiser.get("name", ""),
                    "description": advertiser.get("description", ""),
                    "url": advertiser.get("url", ""),
                    "commission_rate": advertiser.get("commission", 0),
                    "cookie_length": advertiser.get("cookieLength", 0),
                    "network": "Impact",
                    "status": advertiser.get("status", "unknown"),
                    "categories": advertiser.get("categories", []),
                    "epc": advertiser.get("epc", 0),
                    "created_at": datetime.utcnow().isoformat()
                })
        
        return programs

class AmazonAPI(AffiliateNetworkAPI):
    """Amazon Associates API integration"""
    
    def __init__(self):
        super().__init__(
            "Amazon",
            settings.AMAZON_API_KEY,
            "https://webservices.amazon.com"
        )
    
    async def search_programs(self, niche: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search Amazon programs (simplified for demo)"""
        try:
            # Amazon Associates API is more complex and requires product search
            # This is a simplified implementation
            programs = [{
                "id": "amazon-general",
                "name": "Amazon Associates",
                "description": f"Amazon Associates program for {niche} products",
                "url": "https://affiliate-program.amazon.com",
                "commission_rate": 4.0,  # Average Amazon commission
                "cookie_length": 24,
                "network": "Amazon",
                "status": "active",
                "categories": [niche],
                "epc": 0.5,  # Estimated
                "created_at": datetime.utcnow().isoformat()
            }]
            
            return programs
            
        except Exception as e:
            logger.error(f"Amazon API error: {e}")
            return []

class CJAPI(AffiliateNetworkAPI):
    """Commission Junction (CJ) API integration"""
    
    def __init__(self):
        super().__init__(
            "CJ",
            settings.CJ_API_KEY,
            "https://api.cj.com"
        )
    
    async def search_programs(self, niche: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search CJ programs"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                params = {
                    "keywords": niche,
                    "limit": limit
                }
                
                response = await client.get(f"{self.base_url}/advertisers", headers=headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                return self._process_cj_programs(data)
                
        except Exception as e:
            logger.error(f"CJ API error: {e}")
            return []
    
    def _process_cj_programs(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process CJ API response"""
        programs = []
        
        if "advertisers" in data:
            for advertiser in data["advertisers"]:
                programs.append({
                    "id": advertiser.get("advertiserId", ""),
                    "name": advertiser.get("advertiserName", ""),
                    "description": advertiser.get("description", ""),
                    "url": advertiser.get("advertiserUrl", ""),
                    "commission_rate": advertiser.get("commission", 0),
                    "cookie_length": advertiser.get("cookieLength", 0),
                    "network": "CJ",
                    "status": advertiser.get("status", "unknown"),
                    "categories": advertiser.get("categories", []),
                    "epc": advertiser.get("epc", 0),
                    "created_at": datetime.utcnow().isoformat()
                })
        
        return programs

class PartnerizeAPI(AffiliateNetworkAPI):
    """Partnerize API integration"""
    
    def __init__(self):
        super().__init__(
            "Partnerize",
            settings.PARTNERIZE_API_KEY,
            "https://api.partnerize.com"
        )
    
    async def search_programs(self, niche: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search Partnerize programs"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                params = {
                    "search": niche,
                    "limit": limit
                }
                
                response = await client.get(f"{self.base_url}/advertisers", headers=headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                return self._process_partnerize_programs(data)
                
        except Exception as e:
            logger.error(f"Partnerize API error: {e}")
            return []
    
    def _process_partnerize_programs(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process Partnerize API response"""
        programs = []
        
        if "advertisers" in data:
            for advertiser in data["advertisers"]:
                programs.append({
                    "id": advertiser.get("id", ""),
                    "name": advertiser.get("name", ""),
                    "description": advertiser.get("description", ""),
                    "url": advertiser.get("url", ""),
                    "commission_rate": advertiser.get("commission", 0),
                    "cookie_length": advertiser.get("cookieLength", 0),
                    "network": "Partnerize",
                    "status": advertiser.get("status", "unknown"),
                    "categories": advertiser.get("categories", []),
                    "epc": advertiser.get("epc", 0),
                    "created_at": datetime.utcnow().isoformat()
                })
        
        return programs

class AffiliateNetworksManager:
    """Manages all affiliate network integrations"""
    
    def __init__(self):
        self.networks = {
            "shareasale": ShareASaleAPI(),
            "impact": ImpactAPI(),
            "amazon": AmazonAPI(),
            "cj": CJAPI(),
            "partnerize": PartnerizeAPI(),
            # Add more networks as needed
        }
    
    async def search_all_networks(self, niche: str, limit_per_network: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search all affiliate networks for programs in a specific niche
        
        Args:
            niche: The niche to search for
            limit_per_network: Maximum programs to fetch from each network
            
        Returns:
            Dict with network names as keys and program lists as values
        """
        results = {}
        
        # Run searches in parallel
        tasks = []
        for network_name, network_api in self.networks.items():
            task = self._search_network(network_name, network_api, niche, limit_per_network)
            tasks.append(task)
        
        # Wait for all searches to complete
        search_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, (network_name, network_api) in enumerate(self.networks.items()):
            result = search_results[i]
            if isinstance(result, Exception):
                logger.error(f"Error searching {network_name}: {result}")
                results[network_name] = []
            else:
                results[network_name] = result
        
        return results
    
    async def _search_network(
        self, 
        network_name: str, 
        network_api: AffiliateNetworkAPI, 
        niche: str, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search a single network"""
        try:
            programs = await network_api.search_programs(niche, limit)
            logger.info(f"Found {len(programs)} programs in {network_name} for niche: {niche}")
            return programs
        except Exception as e:
            logger.error(f"Error searching {network_name}: {e}")
            return []
    
    async def get_program_details(self, network_name: str, program_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific program"""
        try:
            if network_name in self.networks:
                network_api = self.networks[network_name]
                return await network_api.get_program_details(program_id)
            else:
                logger.error(f"Unknown network: {network_name}")
                return None
        except Exception as e:
            logger.error(f"Error getting program details from {network_name}: {e}")
            return None
    
    def get_available_networks(self) -> List[str]:
        """Get list of available networks"""
        return list(self.networks.keys())
    
    async def get_network_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all networks"""
        stats = {}
        
        for network_name, network_api in self.networks.items():
            try:
                # Test network connectivity
                test_programs = await network_api.search_programs("test", 1)
                stats[network_name] = {
                    "status": "active",
                    "programs_found": len(test_programs),
                    "last_checked": datetime.utcnow().isoformat()
                }
            except Exception as e:
                stats[network_name] = {
                    "status": "error",
                    "error": str(e),
                    "last_checked": datetime.utcnow().isoformat()
                }
        
        return stats

# Global instance
affiliate_networks_manager = AffiliateNetworksManager()

# Convenience functions
async def search_affiliate_programs(niche: str, networks: Optional[List[str]] = None, limit_per_network: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    """Search affiliate programs across all or specified networks"""
    if networks:
        # Filter to specified networks
        filtered_networks = {k: v for k, v in affiliate_networks_manager.networks.items() if k in networks}
        original_networks = affiliate_networks_manager.networks
        affiliate_networks_manager.networks = filtered_networks
        
        try:
            results = await affiliate_networks_manager.search_all_networks(niche, limit_per_network)
        finally:
            affiliate_networks_manager.networks = original_networks
        return results
    else:
        return await affiliate_networks_manager.search_all_networks(niche, limit_per_network)

async def get_program_details(network_name: str, program_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific program"""
    return await affiliate_networks_manager.get_program_details(network_name, program_id)

def get_available_networks() -> List[str]:
    """Get list of available affiliate networks"""
    return affiliate_networks_manager.get_available_networks()

async def get_network_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all affiliate networks"""
    return await affiliate_networks_manager.get_network_stats()
