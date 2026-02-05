import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

# Mock logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock settings
DATAFORSEO_URL = "https://api.dataforseo.com/v3"
# NOTE: User did not provide keys in this flow context, and I can't read from DB without Pyo3 error.
# I will assume the user has valid keys in the DB.
# This script is to verify the LOGIC and MAPPING, forcing a mock response to ensure it parses correctly.
# If I can't make a real call, I will verify that the parsing logic in `_parse_trends_explore_result` works as expected.

async def test_trends_parse_logic():
    print("\n--- Testing Google Trends Parse Logic ---")
    
    # Mock Response from DataForSEO Google Trends Explore
    mock_response = {
        "version": "0.1.20221230",
        "status_code": 20000,
        "status_message": "Ok.",
        "time": "0.1154 sec.",
        "cost": 0.00075,
        "tasks_count": 1,
        "tasks_error": 0,
        "tasks": [
            {
                "id": "02281858-1535-0139-2000-dcbd6717f9b8",
                "status_code": 20000,
                "status_message": "Ok.",
                "time": "0.0632 sec.",
                "cost": 0.00075,
                "result_count": 1,
                "path": [
                    "v3",
                    "keywords_data",
                    "google_trends",
                    "explore",
                    "task_get",
                    "02281858-1535-0139-2000-dcbd6717f9b8"
                ],
                "data": {
                    "api": "keywords_data",
                    "function": "explore",
                    "se": "google_trends",
                    "location_name": "United States",
                    "date_from": "2023-01-01",
                    "date_to": "2024-01-01",
                    "type": "web",
                    "category_code": 0,
                    "item_types": ["google_trends_graph"]
                },
                "result": [
                    {
                        "keywords": ["coffee"],
                        "type": "google_trends_graph",
                        "items_count": 1,
                        "items": [
                            {
                                "type": "google_trends_graph",
                                "data": [
                                    {"timestamp": 1672531200, "values": [80]},
                                    {"timestamp": 1675209600, "values": [85]},
                                    {"timestamp": 1677628800, "values": [75]}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }

    # Import locally to avoid top-level issues if any
    try:
        # Mock the service class to just use the parse method method
        from src.services.dataforseo_service import DataForSEOService
        service = DataForSEOService()
        
        parsed = service._parse_trends_explore_result(mock_response)
        
        print(f"Parsed keys: {parsed.keys()}")
        if "coffee" in parsed:
             print(f"Points for 'coffee': {len(parsed['coffee'])}")
             print(f"First point: {parsed['coffee'][0]}")
             print("✅ Parsing logic success")
        else:
             print("❌ Parsing logic failed")
             
    except ImportError:
        # If we can't even import, copy the function here to test logic in isolation
        print("Could not import service (env issues), implementing loop locally to verify logic:")
        trends_data = {}
        result = mock_response
        for task in result.get("tasks", []):
            for task_result in task.get("result", []):
                for item in task_result.get("items", []):
                    if item.get("type") == "google_trends_graph" and "data" in item:
                        # Map back to keywords
                        keywords = task_result.get("keywords", [])
                        for i, keyword in enumerate(keywords):
                            points = []
                            for data_point in item["data"]:
                                if not data_point.get("missing_data", False):
                                    timestamp = data_point.get("timestamp")
                                    # values is a list corresponding to the keyword index
                                    val = data_point.get("values", [0])[i] if i < len(data_point.get("values", [])) else 0
                                    points.append({
                                        "timestamp": timestamp,
                                        "date": "MOCKED_DATE", # skipping datetime import for simplicity in raw logic check
                                        "value": val
                                    })
                            trends_data[keyword] = points
        print(f"Logic Result: {trends_data}")

if __name__ == "__main__":
    asyncio.run(test_trends_parse_logic())
