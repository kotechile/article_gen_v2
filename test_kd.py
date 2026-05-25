import asyncio
from src.integrations.dataforseo import dataforseo_api

async def test():
    kws = ["hidden costs of owning a home"]
    res = await dataforseo_api.get_keyword_difficulty(kws, return_raw=True)
    print("Result:", res['raw']['response'])

if __name__ == "__main__":
    asyncio.run(test())
