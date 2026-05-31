import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

async def run():
    url = os.getenv("DATABASE_URL")
    if not url:
        print("No DATABASE_URL found")
        return
        
    conn = await asyncpg.connect(url)
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS public.research_pipeline_runs (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                user_id UUID,
                seed_keyword TEXT NOT NULL,
                clusters JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_research_pipeline_runs_user_id ON public.research_pipeline_runs(user_id);
        """)
        print("Table created successfully")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await conn.close()

asyncio.run(run())
