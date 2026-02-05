
import asyncio
import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any

# Ensure we can import backend src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Valid DataForSEO Credentials must be loaded from config
from src.core.config import settings
from unittest.mock import MagicMock, AsyncMock


# Valid DataForSEO Credentials must be loaded from config
from src.core.config import settings
from unittest.mock import MagicMock, AsyncMock


# Valid DataForSEO Credentials must be loaded from config
from src.core.config import settings
from unittest.mock import MagicMock, AsyncMock
import argparse
import sys
import json
import urllib.request
import urllib.parse
from unittest.mock import patch

# --- HYBRID MOCKING INFRASTRUCTURE ---

# 1. Simple Supabase Client (Bypass PyO3/Gotrue issues)
class SimpleSupabaseClient:
    def __init__(self, url, key):
        self.url = url
        self.key = key
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }

    def table(self, name):
        return SimpleQueryBuilder(self.url, self.headers, name)

class SimpleQueryBuilder:
    def __init__(self, base_url, headers, table):
        self.base_url = f"{base_url}/rest/v1/{table}"
        self.headers = headers
        self.params = {}

    def select(self, cols):
        self.params['select'] = cols
        return self

    def eq(self, col, val):
        self.params[col] = f"eq.{val}"
        return self
    
    def limit(self, val):
        self.params['limit'] = str(val)
        return self

    def execute(self):
        # Build Query
        query_string = urllib.parse.urlencode(self.params)
        full_url = f"{self.base_url}?{query_string}"
        
        req = urllib.request.Request(full_url, headers=self.headers)
        try:
            with urllib.request.urlopen(req) as response:
                if 200 <= response.status < 300:
                    data = json.loads(response.read().decode('utf-8'))
                    # Mock the response object expected by supabase-py
                    return MagicMock(data=data)
                else:
                    logger.error(f"Supabase Error: {response.status}")
                    return MagicMock(data=[])
        except Exception as e:
            logger.error(f"Supabase Request Failed: {e}")
            return MagicMock(data=[])

# 2. Setup Mocks based on Mode
# We always mock 'src.database.supabase_client' to use SimpleSupabaseClient
# We Conditionally mock 'src.services.llm.llm_service' and 'src.integrations.dataforseo'

def setup_environment(real_llm=False, real_d4s=False):
    # Load .env manually for our Simple Client
    from src.core.config import settings
    
    # Inject Simple Client
    real_url = settings.supabase_url
    real_key = settings.supabase_service_role_key
    
    if not real_url or not real_key:
        logger.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in env/config (checked settings.supabase_url).")
        sys.exit(1)
        
    simple_client = SimpleSupabaseClient(real_url, real_key)
    
    mock_supabase_module = MagicMock()
    # When get_supabase_client() is called, return our simple client
    mock_supabase_module.get_supabase_client.return_value = simple_client
    sys.modules['src.database.supabase_client'] = mock_supabase_module
    
    # DataForSEO Logic
    if real_d4s:
         logger.info(">>> RUNNING IN REAL DATAFORSEO MODE (Integration) <<<")
         # Do not mock, let it import the real module.
         # The real module will import our mocked 'src.database.supabase_client'
         pass
    else:
        logger.info(">>> RUNNING IN MOCK DATAFORSEO MODE <<<")
        mock_d4s_module = MagicMock()
        class MockDataForSEO:
            async def get_related_keywords(self, keyword, **kwargs):
                if "sustainable" in keyword:
                     return [
                        {"keyword": "bamboo garden tools", "search_volume": 1200, "cpc": 1.5, "keyword_difficulty": 30},
                        {"keyword": "recycled plastic pots", "search_volume": 800, "cpc": 0.5, "keyword_difficulty": 20}
                     ]
                return [{"keyword": f"related to {keyword}", "search_volume": 100, "cpc": 0.5, "keyword_difficulty": 10}]

            async def get_keyword_trends(self, keywords, **kwargs):
                return [{"keywords": keywords, "items": [{"interest": 50}, {"interest": 60}]}]

        mock_d4s_module.dataforseo_api = MockDataForSEO()
        sys.modules['src.integrations.dataforseo'] = mock_d4s_module

    if not real_llm:
        logger.info(">>> RUNNING IN MOCK LLM MODE <<<")
        # Mock LLM Service
        class MockLLMService:
            async def generate_text(self, prompt: str, **kwargs):
                return MagicMock(content="- sustainable gardening tools\n- organic plant food")
            async def generate_json(self, prompt: str, **kwargs):
                if "Article Concepts" in prompt:
                    return [{"cluster_title": "Eco-Friendly Tools", "primary_keyword": "bamboo garden tools", "keywords": ["bamboo garden tools"]}]
                return {"intent": "Commercial", "price_range": "Mid", "affiliate_categories": ["Gardening"]}
        
        mock_llm_module = MagicMock()
        mock_llm_module.llm_service = MockLLMService()
        sys.modules['src.services.llm.llm_service'] = mock_llm_module
    else:
        logger.info(">>> RUNNING IN REAL LLM MODE (Integration) <<<")
        # Do NOT mock 'src.services.llm.llm_service'. 
        # But we MUST ensure 'src.services.llm' packages import correctly.
        # Since we mocked supabase_client, llm_service.py imports 'get_supabase_client' from there, which is our mock.
        # It should work!
        pass

# ------------------------------

# Constants for File Paths
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
FILE_SEEDS = os.path.join(TEST_DATA_DIR, '1_seeds.json')
FILE_RAW_KW = os.path.join(TEST_DATA_DIR, '2_raw_keywords.json')
FILE_GOLDEN_KW = os.path.join(TEST_DATA_DIR, '3_golden_keywords.json')
FILE_CLUSTERS = os.path.join(TEST_DATA_DIR, '4_clusters.json')
FILE_VERIFIED = os.path.join(TEST_DATA_DIR, '5_verified_clusters.json')

def load_json(path):
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path, data):
    # Create dir if not exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved data to {path}")

async def run_step(step_num: int, topic: str = "Urban Gardening", user_id: str = "test-user"):
    # Import service AFTER mocking
    try:
        from src.services.semantic_expansion_service import semantic_expansion_service
    except ImportError as e:
        logger.error(f"Failed to import service: {e}")
        return

    if step_num == 1:
        logger.info(f"--- Step 1: Semantic Explosion (LLM) for topic '{topic}' ---")
        seeds = await semantic_expansion_service.generate_seeds(topic)
        if seeds:
            save_json(FILE_SEEDS, seeds)
            logger.info(f"Generated {len(seeds)} seeds.")
        else:
            logger.error("Step 1 Failed: No seeds generated.")

    elif step_num == 2:
        logger.info("--- Step 2: Bulk Data Retrieval (DataForSEO) ---")
        seeds = load_json(FILE_SEEDS)
        if not seeds:
            logger.error("Step 2 Failed: Missing input seeds (1_seeds.json)")
            return
        
        raw_keywords = await semantic_expansion_service.fetch_bulk_keyword_data(seeds)
        if raw_keywords:
            save_json(FILE_RAW_KW, raw_keywords)
            logger.info(f"Fetched {len(raw_keywords)} raw keywords.")
        else:
            logger.error("Step 2 Failed: No keywords fetched.")

    elif step_num == 3:
        logger.info("--- Step 3: Profit Filtering (Math) ---")
        raw_keywords = load_json(FILE_RAW_KW)
        if not raw_keywords:
            logger.error("Step 3 Failed: Missing input raw keywords (2_raw_keywords.json)")
            return

        filtered_keywords = semantic_expansion_service.filter_profitable_keywords(raw_keywords)
        if filtered_keywords:
            save_json(FILE_GOLDEN_KW, filtered_keywords)
            logger.info(f"Filtered down to {len(filtered_keywords)} Golden keywords.")
        else:
            logger.error("Step 3 Failed: No profitable keywords found.")

    elif step_num == 4:
        logger.info("--- Step 4: Semantic Clustering (LLM) ---")
        golden_keywords = load_json(FILE_GOLDEN_KW)
        if not golden_keywords:
            logger.error("Step 4 Failed: Missing input golden keywords (3_golden_keywords.json)")
            return

        clusters = await semantic_expansion_service.cluster_keywords(golden_keywords)
        if clusters:
            save_json(FILE_CLUSTERS, clusters)
            logger.info(f"Generated {len(clusters)} clusters.")
        else:
            logger.error("Step 4 Failed: No clusters generated.")

    elif step_num == 5:
        logger.info("--- Step 5: Profitability Verification (Trends + Monetization) ---")
        clusters = load_json(FILE_CLUSTERS)
        if not clusters:
            logger.error("Step 5 Failed: Missing input clusters (4_clusters.json)")
            return

        # Note: In the service refactor, verify_clusters does both Trend and Monetization
        verified_clusters = await semantic_expansion_service.verify_clusters(clusters)
        
        if verified_clusters:
            save_json(FILE_VERIFIED, verified_clusters)
            logger.info(f"Verified {len(verified_clusters)} profitable clusters.")
            
            # Print summary
            print("\nVerification Summary:")
            for c in verified_clusters:
                print(f"- {c.get('cluster_title')} (Trend: {c.get('trend_analysis', {}).get('status')}, Intent: {c.get('monetization', {}).get('details', {}).get('intent')})")
        else:
            logger.warning("Step 5 Completed: No clusters passed verification.")

    elif step_num == 6:
        # Reserved for separate granular testing if needed, but current verify_clusters does both
        logger.info("Step 6 is currently integrated into Step 5 (Verification Loop). Running Step 5...")
        await run_step(5, topic, user_id)
        
    else:
        logger.error(f"Unknown step: {step_num}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Semantic Expansion Pipeline Step-by-Step")
    parser.add_argument('step', type=int, help="Step number (1-5)")
    parser.add_argument('--topic', type=str, default="Digital Nomad Gear", help="Topic for Step 1")
    parser.add_argument('--real-llm', action='store_true', help="Use REAL LLM Service (requires .env keys)")
    parser.add_argument('--real-d4s', action='store_true', help="Use REAL DataForSEO Service (fetches keys from Supabase)")
    
    args = parser.parse_args()
    
    # Setup Mocks / Real Environment
    setup_environment(real_llm=args.real_llm, real_d4s=args.real_d4s)
    
    asyncio.run(run_step(args.step, args.topic))
