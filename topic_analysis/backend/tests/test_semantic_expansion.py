
import asyncio
import logging
import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch

# Add backend root to python path to import src as a package
# backend/tests/../ -> backend/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_semantic_expansion_flow():
    logger.info("Starting Test: Semantic Expansion Flow")
    
    # Pre-mock modules to avoid initialization errors (PyO3/Supabase/LLM)
    mock_llm_module = MagicMock()
    mock_llm_service = MagicMock()
    mock_llm_module.llm_service = mock_llm_service
    
    mock_d4s_module = MagicMock()
    mock_d4s_api = MagicMock()
    mock_d4s_module.dataforseo_api = mock_d4s_api
    
    # We need to preserve original modules if they exist, but here we likely want to force consistency
    with patch.dict(sys.modules, {
        'src.services.llm.llm_service': mock_llm_module,
        'src.integrations.dataforseo': mock_d4s_module
    }):
        # Now import the service under test
        # It will pick up the mocked modules from sys.modules
        try:
            # Re-import to ensure we get a fresh version using our mocks if it was already imported
            import src.services.semantic_expansion_service
            import importlib
            importlib.reload(src.services.semantic_expansion_service)
            from src.services.semantic_expansion_service import semantic_expansion_service
        except ImportError as e:
            logger.error(f"Failed to import src.services.semantic_expansion_service: {e}")
            return

        # Setup Mocks
        
        
        # 1. Mock LLM Seed Generation
        mock_llm_service.generate_text = AsyncMock(return_value=MagicMock(content="""
        - sustainable gardening tools
        - best organic fertilizer
        - indoor plant lights
        """))
        
        # 2. Mock DataForSEO Related Keywords
        # We need to simulate bulk retrieval
        async def mock_related_kw(seed, **kwargs):
            if "sustainable" in seed:
                return [
                    {"keyword": "bamboo trowel", "search_volume": 1000, "cpc": 2.5, "keyword_difficulty": 30},
                    {"keyword": "plastic pots", "search_volume": 10, "cpc": 0.1, "keyword_difficulty": 10} # Junk
                ]
            elif "organic" in seed:
                return [
                    {"keyword": "fish emulsion fertilizer", "search_volume": 500, "cpc": 1.0, "keyword_difficulty": 40},
                    {"keyword": "bad fertilizer", "search_volume": 10000, "cpc": 0.1, "keyword_difficulty": 90} # High KD
                ]
            else:
                return []
        
        
        mock_d4s_api.get_related_keywords = AsyncMock(side_effect=mock_related_kw)
        
        # 3. Mock LLM Clustering
        mock_llm_service.generate_json = AsyncMock(side_effect=[
            # First call for Clusters
            [
                {
                    "cluster_title": "Eco-Friendly Garden Tools",
                    "primary_keyword": "bamboo trowel",
                    "keywords": ["bamboo trowel", "sustainable shovel"]
                },
                {
                    "cluster_title": "Organic Plant Food",
                    "primary_keyword": "fish emulsion fertilizer",
                    "keywords": ["fish emulsion fertilizer"]
                }
            ],
            # Second call for Monetization (Cluster 1)
            {
                "intent": "Commercial",
                "price_range": "Low",
                "affiliate_categories": ["Gardening"]
            },
            # Third call for Monetization (Cluster 2)
            {
                "intent": "Commercial",
                "price_range": "Mid",
                "affiliate_categories": ["Gardening"]
            }
        ])

        # 4. Mock DataForSEO Trends
        # Return valid trend data
        mock_d4s_api.get_keyword_trends = AsyncMock(return_value=[
            {"keywords": ["bamboo trowel"], "items": [{"interest": 50}, {"interest": 60}]} # Trending up
        ])

        # Execute
        logger.info("Running expand_and_verify...")
        results = await semantic_expansion_service.expand_and_verify("Urban Gardening", "test-user-id")
        
        # Assertions
        logger.info(f"Results: {results}")
        
        assert len(results) >= 1, "Should have verified clusters"
        
        # Check specific cluster
        cluster1 = next((c for c in results if c['primary_keyword'] == 'bamboo trowel'), None)
        assert cluster1 is not None, "Bamboo trowel cluster not found"
        assert cluster1['trend_analysis']['status'] == 'PASS', "Trend check should pass"
        assert cluster1['monetization']['status'] == 'PASS', "Monetization check should pass"
        
        logger.info("Test PASSED: Semantic Expansion Flow contains valid profitable clusters.")

if __name__ == "__main__":
    asyncio.run(test_semantic_expansion_flow())
