
import asyncio
import logging
import sys
import os

# Add backend to sys.path to allow imports
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from src.integrations.dataforseo import DataForSEOAPI
from src.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_standard_api():
    logger.info("Initializing DataForSEO API...")
    api = DataForSEOAPI()
    
    # Test Seed
    seeds = ["organic gardening tips"]
    
    logger.info(f"Testing get_related_keywords_standard with seeds: {seeds}")
    
    try:
        results = await api.get_related_keywords_standard(seeds, limit_per_seed=5)
        
        logger.info(f"REL KWs Result Count: {len(results)}")
        if results:
            logger.info(f"Sample: {results[0]}")
        else:
            logger.warning("No related keywords returned.")
            
    except Exception as e:
        logger.error(f"Error testing related keywords: {e}")

    # Test Bulk Metrics (Volume)
    logger.info(f"Testing get_bulk_metrics_standard with seeds: {seeds}")
    try:
        metrics = await api.get_bulk_metrics_standard(seeds)
        logger.info(f"BULK METRICS Result Count: {len(metrics)}")
        if metrics:
            logger.info(f"Sample: {metrics[0]}")
        else:
            logger.warning("No bulk metrics returned.")
    except Exception as e:
        logger.error(f"Error testing bulk metrics: {e}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_standard_api())
