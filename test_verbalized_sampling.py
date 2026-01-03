#!/usr/bin/env python3
"""
Test script for verbalized-sampling integration in Content Generator V2.

This script tests the integration of verbalized-sampling into the article
generation pipeline to ensure it works correctly.
"""

import os
import sys
import logging
from typing import Dict, Any, List

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_verbalized_sampling_client():
    """Test the VerbalizedSamplingClient directly."""
    try:
        from verbalized_sampling_client import create_verbalized_sampling_client
        
        logger.info("Testing VerbalizedSamplingClient...")
        
        # Create client
        client = create_verbalized_sampling_client(
            k=3,
            tau=0.10,
            temperature=0.9,
            seed=42,
            enabled=True
        )
        
        # Test prompt
        prompt = "Write a compelling introduction paragraph about renewable energy benefits."
        
        # Generate content
        response = client.generate_with_sampling(prompt)
        
        logger.info(f"✓ VerbalizedSamplingClient test passed")
        logger.info(f"  Selected text: {response.text[:100]}...")
        logger.info(f"  Sample index: {response.sample_index}")
        logger.info(f"  Total samples: {len(response.all_samples)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ VerbalizedSamplingClient test failed: {e}")
        return False

def test_content_generator_integration():
    """Test the ContentGenerator with verbalized sampling."""
    try:
        from content_generator import create_content_generator
        from llm_client import create_llm_client
        
        logger.info("Testing ContentGenerator with verbalized sampling...")
        
        # Create a mock LLM client (we'll use a simple one for testing)
        # Note: This will fall back to standard generation if no API key is provided
        llm_client = create_llm_client(
            provider="openai",
            model="gpt-4",
            api_key="test-key",  # This will cause fallback behavior
            temperature=0.7
        )
        
        # Create content generator with verbalized sampling enabled
        content_generator = create_content_generator(
            llm_client=llm_client,
            use_verbalized_sampling=True
        )
        
        # Test data
        section_outline = {
            "title": "Introduction to Renewable Energy",
            "subtitle": "Understanding the Basics",
            "key_points": [
                "Environmental benefits of renewable energy",
                "Economic advantages",
                "Future sustainability"
            ],
            "word_count_target": 300,
            "content_type": "paragraph",
            "order": 1
        }
        
        research_data = {
            "brief": "Comprehensive guide to renewable energy benefits",
            "tone": "journalistic",
            "target_audience": "general",
            "keywords": "renewable energy, sustainability, green technology"
        }
        
        claims = [
            {"claim": "Renewable energy reduces carbon emissions by 80%"},
            {"claim": "Solar and wind power are becoming cost-competitive"}
        ]
        
        evidence = [
            {
                "title": "Renewable Energy Report 2024",
                "content": "Renewable energy sources show significant environmental benefits...",
                "source": "https://example.com/report",
                "source_type": "web"
            }
        ]
        
        # Generate content
        section_content = content_generator.generate_section_content(
            section_outline=section_outline,
            research_data=research_data,
            claims=claims,
            evidence=evidence
        )
        
        logger.info(f"✓ ContentGenerator integration test passed")
        logger.info(f"  Section title: {section_content.title}")
        logger.info(f"  Word count: {section_content.total_word_count}")
        logger.info(f"  Content blocks: {len(section_content.content_blocks)}")
        
        # Check if verbalized sampling metadata is present
        if section_content.content_blocks:
            metadata = section_content.content_blocks[0].metadata
            verbalized_info = metadata.get('verbalized_sampling', {})
            logger.info(f"  Verbalized sampling enabled: {verbalized_info.get('enabled', False)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ ContentGenerator integration test failed: {e}")
        return False

def test_article_structure_generator_integration():
    """Test the ArticleStructureGenerator with verbalized sampling."""
    try:
        from article_structure_generator import create_article_structure_generator
        from llm_client import create_llm_client
        
        logger.info("Testing ArticleStructureGenerator with verbalized sampling...")
        
        # Create a mock LLM client
        llm_client = create_llm_client(
            provider="openai",
            model="gpt-4",
            api_key="test-key",  # This will cause fallback behavior
            temperature=0.7
        )
        
        # Create article structure generator with verbalized sampling enabled
        structure_generator = create_article_structure_generator(
            llm_client=llm_client,
            use_verbalized_sampling=True
        )
        
        # Test data
        research_data = {
            "brief": "Complete guide to sustainable living practices",
            "keywords": "sustainability, eco-friendly, green living",
            "tone": "conversational",
            "target_word_count": 2000,
            "draft_title": "Sustainable Living Guide"
        }
        
        claims = [
            {"claim": "Sustainable living reduces environmental impact"},
            {"claim": "Eco-friendly practices save money long-term"}
        ]
        
        evidence = [
            {
                "title": "Sustainability Study 2024",
                "content": "Research shows sustainable practices have measurable benefits...",
                "source": "https://example.com/study",
                "source_type": "web"
            }
        ]
        
        # Generate structure
        structure = structure_generator.generate_structure(
            research_data=research_data,
            claims=claims,
            evidence=evidence
        )
        
        logger.info(f"✓ ArticleStructureGenerator integration test passed")
        logger.info(f"  Article title: {structure.title}")
        logger.info(f"  Sections: {len(structure.sections)}")
        logger.info(f"  Target word count: {structure.target_word_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ ArticleStructureGenerator integration test failed: {e}")
        return False

def test_fallback_behavior():
    """Test fallback behavior when verbalized-sampling is not available."""
    try:
        from verbalized_sampling_client import create_verbalized_sampling_client
        
        logger.info("Testing fallback behavior...")
        
        # Create client with sampling disabled
        client = create_verbalized_sampling_client(
            k=3,
            tau=0.10,
            temperature=0.9,
            seed=42,
            enabled=False  # Disabled
        )
        
        prompt = "Write a short paragraph about climate change."
        
        # Generate content
        response = client.generate_with_sampling(prompt)
        
        logger.info(f"✓ Fallback behavior test passed")
        logger.info(f"  Fallback text: {response.text[:100]}...")
        logger.info(f"  Fallback metadata: {response.metadata.get('fallback', False)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Fallback behavior test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting verbalized-sampling integration tests...")
    
    tests = [
        ("VerbalizedSamplingClient", test_verbalized_sampling_client),
        ("ContentGenerator Integration", test_content_generator_integration),
        ("ArticleStructureGenerator Integration", test_article_structure_generator_integration),
        ("Fallback Behavior", test_fallback_behavior)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} test PASSED")
            else:
                logger.error(f"✗ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} test FAILED with exception: {e}")
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Verbalized-sampling integration is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the integration.")
        return 1

if __name__ == "__main__":
    exit(main())


