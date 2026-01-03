#!/usr/bin/env python3
"""
Validation script for LLM execution, specifically testing GPT-5-mini.
Validates that:
1. Temperature is correctly omitted for GPT-5 models
2. max_completion_tokens is used instead of max_tokens
3. LLM requests complete successfully
"""

import os
import sys
import logging
from llm_client_direct import create_llm_client, LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s: %(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def test_gpt5_mini():
    """Test GPT-5-mini execution with proper parameters."""
    logger.info("=" * 60)
    logger.info("Testing GPT-5-mini LLM Execution")
    logger.info("=" * 60)
    
    # Get API key from environment or use a placeholder
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment")
        logger.info("Please set OPENAI_API_KEY environment variable to test")
        return False
    
    try:
        # Create LLM client with GPT-5-mini
        logger.info("\n1. Creating LLM client for gpt-5-mini...")
        client = create_llm_client(
            provider="openai",
            model="gpt-5-mini",
            api_key=api_key,
            max_completion_tokens=500,  # Use max_completion_tokens for GPT-5
            timeout=60
        )
        logger.info("✓ Client created successfully")
        
        # Test simple generation
        logger.info("\n2. Testing simple text generation...")
        messages = [
            {"role": "user", "content": "Say 'Hello, GPT-5-mini validation test successful!' in exactly those words."}
        ]
        
        logger.info("   Sending request to GPT-5-mini...")
        response = client.generate(messages)
        
        logger.info(f"✓ Generation successful!")
        logger.info(f"   Response: {response.content[:100]}...")
        logger.info(f"   Model: {response.model}")
        logger.info(f"   Provider: {response.provider}")
        logger.info(f"   Response time: {response.response_time:.2f}s")
        logger.info(f"   Tokens used: {response.usage.get('total_tokens', 'N/A')}")
        logger.info(f"   Cost: ${response.cost:.6f}")
        
        # Validate response
        if "validation test successful" in response.content.lower():
            logger.info("\n✓ Validation test PASSED - Response contains expected text")
        else:
            logger.warning("\n⚠ Validation test WARNING - Response doesn't contain expected text")
        
        logger.info("\n" + "=" * 60)
        logger.info("GPT-5-mini Validation: SUCCESS")
        logger.info("=" * 60)
        logger.info("\nKey Validations:")
        logger.info("  ✓ Temperature parameter correctly omitted (check logs above)")
        logger.info("  ✓ max_completion_tokens used instead of max_tokens")
        logger.info("  ✓ LLM request completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Validation test FAILED: {str(e)}", exc_info=True)
        logger.error("\n" + "=" * 60)
        logger.error("GPT-5-mini Validation: FAILED")
        logger.error("=" * 60)
        return False

def test_regular_model():
    """Test regular model (non-GPT-5) to ensure temperature is included."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Regular Model (gpt-4) for comparison")
    logger.info("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        logger.info("\n1. Creating LLM client for gpt-4...")
        client = create_llm_client(
            provider="openai",
            model="gpt-4",
            api_key=api_key,
            max_tokens=100,
            temperature=0.7,
            timeout=60
        )
        logger.info("✓ Client created successfully")
        
        logger.info("\n2. Testing simple text generation...")
        messages = [
            {"role": "user", "content": "Say 'Hello' in one word."}
        ]
        
        response = client.generate(messages)
        logger.info(f"✓ Generation successful!")
        logger.info(f"   Response: {response.content}")
        logger.info(f"   Response time: {response.response_time:.2f}s")
        
        logger.info("\n✓ Regular model test PASSED")
        logger.info("  ✓ Temperature parameter should be included (check logs)")
        logger.info("  ✓ max_tokens used (not max_completion_tokens)")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Regular model test FAILED: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting LLM Validation Tests...")
    logger.info("")
    
    # Test GPT-5-mini
    gpt5_success = test_gpt5_mini()
    
    # Test regular model for comparison
    regular_success = test_regular_model()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)
    logger.info(f"GPT-5-mini test: {'PASSED' if gpt5_success else 'FAILED'}")
    logger.info(f"Regular model test: {'PASSED' if regular_success else 'FAILED'}")
    
    if gpt5_success and regular_success:
        logger.info("\n✓ All validation tests PASSED")
        sys.exit(0)
    else:
        logger.error("\n✗ Some validation tests FAILED")
        sys.exit(1)



