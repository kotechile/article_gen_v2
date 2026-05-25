import asyncio
import json
from src.services.research_dataforseo_search_service import ResearchDataforseoSearchService
from uuid import UUID

async def run():
    svc = ResearchDataforseoSearchService()
    # Let's get the most recent dataforseo search of type expansion_funnel
    res = await svc.supabase_service.client.table("research_dataforseo_searches").select("*").eq("search_type", "expansion_funnel").order("created_at", desc=True).limit(1).execute()
    if res.data:
        record = res.data[0]
        print(json.dumps(record["result_summary_json"], indent=2))
        
asyncio.run(run())
