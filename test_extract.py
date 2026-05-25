import asyncio
from src.services.research_pipeline_service import research_pipeline_service

async def main():
    try:
        res = await research_pipeline_service.extract_and_persist("hidden costs of owning a home", "anonymous")
        print(res)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
