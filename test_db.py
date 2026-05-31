import asyncio
from src.services.supabase_service import supabase_service
def run():
    client = supabase_service.get_client()
    res = client.table("subtopics").select("*").limit(1).execute()
    print(res.data)
run()
