import asyncio
import sys
import os
import httpx
from uuid import uuid4
from dotenv import load_dotenv

# Add backend directory to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

# Load environment variables
load_dotenv(os.path.join(os.getcwd(), 'backend', '.env'))

async def verify():
    # Configuration
    BASE_URL = "http://localhost:8000"
    
    # 1. First, we need a valid user ID (assuming one exists or we can mock/bypass auth or use a known one)
    # Ideally we'd login, but let's assume we can get one or the system allows us to reuse one from logs/previous runs.
    # For this script to work against a real running server, we need a token.
    # If we can't easily get a token, we might need to test the service directly rather than the API route.
    # Let's try to test the SERVICE class directly to avoid auth issues in this standalone script.
    
    print("Verifying EnhancedTopicDecompositionService strict mode...")
    
    try:
        from src.services.enhanced_topic_decomposition_service import EnhancedTopicDecompositionService
        from src.integrations.google_autocomplete import GoogleAutocompleteService
        
        service = EnhancedTopicDecompositionService()
        
        # Test Case 1: Active LLM Service (Happy Path)
        # Note: This relies on the environment having valid API keys. 
        # If no keys are present, this might fail, which is actually correct for "Strict Quality".
        print("\nTest 1: Normal Generation (expecting success if keys present, else ERROR)")
        try:
             result = await service.decompose_topic_enhanced(
                query="AI Marketing Trends 2025",
                user_id=str(uuid4()),
                max_subtopics=3,
                use_llm=True
            )
             print("✅ Success: Got result")
             print(f"   Subtopics found: {len(result.get('subtopics', []))}")
             if not result.get('subtopics'):
                 print("   ⚠️ Warning: Success returned but subtopics empty? Should have raised error in strict mode.")
                 
        except Exception as e:
            print(f"ℹ️ Result: Error raised (Expected if no API keys): {e}")
            if "Strict Quality Policy" in str(e):
                print("✅ Verified: Strict Quality Policy error message received.")
            else:
                print("⚠️ Warning: Different error message than expected.")

        # Test Case 2: Force Failure (Simulate Broker/LLM failure)
        print("\nTest 2: Forced Service Failure (Simulating broken LLM)")
        # Temporarily sabotage the llm provider
        original_provider = service.llm_service
        service.llm_service = None 
        
        try:
            await service.decompose_topic_enhanced(
                query="Failure Test Topic",
                user_id=str(uuid4()),
                max_subtopics=3,
                use_llm=True
            )
            print("❌ Failed: Should have raised exception but succeeded.")
        except Exception as e:
            print(f"✅ Verified: Exception raised as expected: {e}")
            if "Strict Quality Policy" in str(e):
                print("   Error message confirms strict policy.")
            else:
                print(f"   Error message: {e}")

    except ImportError as e:
        print(f"❌ Setup Failed: Could not import backend modules. {e}")
    except Exception as e:
        print(f"❌ Verification Failed with unexpected error: {e}")

    # Also quickly verify TrendService fix
    print("\nVerifying TrendService fix...")
    try:
        from src.services.trend_service import TrendService
        ts = TrendService()
        # This caused NameError before
        mock_data = ts._get_mock_google_trends_data(["test topic"])
        print(f"✅ TrendService mock data generated successfully: {mock_data.get('trend_direction')}")
    except Exception as e:
        print(f"❌ TrendService fix failed: {e}")

if __name__ == "__main__":
    asyncio.run(verify())
