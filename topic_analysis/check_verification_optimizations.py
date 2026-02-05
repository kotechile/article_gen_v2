
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'backend'))
from src.services.semantic_expansion_service import semantic_expansion_service

async def test_verify_clusters_parallel():
    # Setup mock methods to simulate latency
    async def mock_analyze_trend(kw):
        await asyncio.sleep(1) # Simulate 1s latency
        return {"status": "PASS", "slope": 0.5, "label": "Trending Up"}

    async def mock_check_monetization(kw, topic):
        await asyncio.sleep(1) # Simulate 1s latency
        return {"status": "PASS", "details": {"intent": "Commercial"}}

    # Patch the service methods
    semantic_expansion_service.analyze_trend = mock_analyze_trend
    semantic_expansion_service.check_monetization = mock_check_monetization

    # Create dummy clusters (e.g., 5 clusters)
    clusters = [
        {"primary_keyword": f"kw{i}", "cluster_title": f"Cluster {i}"}
        for i in range(5)
    ]

    print(f"Starting verification of {len(clusters)} clusters...")
    start_time = time.time()
    
    # Run verification
    verified = await semantic_expansion_service.verify_clusters(clusters)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Verification completed in {duration:.2f} seconds")
    print(f"Verified count: {len(verified)}")
    
    # Assertions
    # If serial: 5 clusters * 1s (trend) + 5 clusters * 1s (monetization) = 10s minimum
    # If parallel: max(1s, 1s) = ~1s total
    if duration < 2.0:
        print("✅ SUCCESS: Execution was parallel!")
    else:
        print("❌ FAILURE: Execution seemed serial.")

if __name__ == "__main__":
    asyncio.run(test_verify_clusters_parallel())
