#!/usr/bin/env python3
"""
Example usage of verbalized-sampling integration in Content Generator V2.

This script demonstrates how to use the enhanced content generation
with verbalized sampling for improved article quality.
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

def example_basic_usage():
    """Example of basic verbalized sampling usage."""
    logger.info("=== Basic Verbalized Sampling Usage ===")
    
    try:
        from verbalized_sampling_client import create_verbalized_sampling_client
        
        # Create client with custom parameters
        client = create_verbalized_sampling_client(
            k=5,                    # Generate 5 samples
            tau=0.10,              # Temperature parameter
            temperature=0.9,       # Base temperature
            seed=42,               # For reproducibility
            enabled=True           # Enable verbalized sampling
        )
        
        # Example prompt
        prompt = "Write a compelling introduction paragraph about the benefits of solar energy for homeowners."
        
        # Generate content with sampling
        response = client.generate_with_sampling(prompt)
        
        print(f"Selected content: {response.text}")
        print(f"Selected from sample {response.sample_index + 1} out of {len(response.all_samples)}")
        print(f"Sampling configuration: {response.metadata}")
        
        # Show all generated samples
        print("\nAll generated samples:")
        for i, sample in enumerate(response.all_samples):
            print(f"Sample {i + 1}: {sample[:100]}...")
        
    except Exception as e:
        logger.error(f"Error in basic usage example: {e}")

def example_content_generation():
    """Example of content generation with verbalized sampling."""
    logger.info("=== Content Generation with Verbalized Sampling ===")
    
    try:
        from content_generator import create_content_generator
        from llm_client import create_llm_client
        
        # Create LLM client (replace with your actual API key)
        llm_client = create_llm_client(
            provider="openai",
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
            temperature=0.7,
            use_verbalized_sampling=True  # Enable verbalized sampling in LLM config
        )
        
        # Create content generator with verbalized sampling
        content_generator = create_content_generator(
            llm_client=llm_client,
            use_verbalized_sampling=True
        )
        
        # Example section outline
        section_outline = {
            "title": "Solar Energy Benefits for Homeowners",
            "subtitle": "Financial and Environmental Advantages",
            "key_points": [
                "Reduced electricity bills through solar panels",
                "Increased property value with renewable energy",
                "Environmental impact reduction",
                "Government incentives and tax credits"
            ],
            "word_count_target": 400,
            "content_type": "paragraph",
            "order": 1
        }
        
        # Research data
        research_data = {
            "brief": "Comprehensive guide to solar energy benefits for residential properties",
            "tone": "conversational",
            "target_audience": "homeowners",
            "keywords": "solar energy, renewable energy, home improvement, sustainability"
        }
        
        # Claims and evidence
        claims = [
            {"claim": "Solar panels can reduce electricity bills by 50-90%"},
            {"claim": "Homes with solar panels sell for 4% more on average"}
        ]
        
        evidence = [
            {
                "title": "Solar Energy Savings Report 2024",
                "content": "Recent studies show significant cost savings from residential solar installations...",
                "source": "https://example.com/solar-report",
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
        
        print(f"Generated section: {section_content.title}")
        print(f"Word count: {section_content.total_word_count}")
        print(f"Content blocks: {len(section_content.content_blocks)}")
        
        # Show the generated content
        if section_content.content_blocks:
            print(f"\nGenerated content:")
            print(section_content.content_blocks[0].content)
            
            # Show verbalized sampling metadata
            metadata = section_content.content_blocks[0].metadata
            verbalized_info = metadata.get('verbalized_sampling', {})
            if verbalized_info.get('enabled'):
                print(f"\nVerbalized sampling info:")
                print(f"  Sample selected: {verbalized_info.get('sample_index', 'N/A')}")
                print(f"  Total samples: {verbalized_info.get('total_samples', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Error in content generation example: {e}")

def example_custom_sampling_parameters():
    """Example of using custom sampling parameters for different content types."""
    logger.info("=== Custom Sampling Parameters ===")
    
    try:
        from verbalized_sampling_client import VerbalizedSamplingConfig, VerbalizedSamplingClient
        
        # Different configurations for different content types
        configs = {
            "paragraph": VerbalizedSamplingConfig(k=7, tau=0.08, temperature=0.9),
            "list": VerbalizedSamplingConfig(k=5, tau=0.12, temperature=0.9),
            "comparison": VerbalizedSamplingConfig(k=8, tau=0.15, temperature=0.9),
            "step_by_step": VerbalizedSamplingConfig(k=6, tau=0.10, temperature=0.9)
        }
        
        client = VerbalizedSamplingClient()
        
        prompt = "Create a comparison between solar and wind energy for residential use."
        
        # Use comparison-specific configuration
        response = client.generate_with_sampling(
            prompt=prompt,
            custom_config=configs["comparison"]
        )
        
        print(f"Comparison content (k={configs['comparison'].k}, tau={configs['comparison'].tau}):")
        print(response.text)
        print(f"Selected from {len(response.all_samples)} samples")
        
    except Exception as e:
        logger.error(f"Error in custom parameters example: {e}")

def example_integration_with_research_pipeline():
    """Example of integrating verbalized sampling with the full research pipeline."""
    logger.info("=== Full Pipeline Integration ===")
    
    try:
        # This would be used with the actual research pipeline
        research_data = {
            "brief": "Complete guide to sustainable home improvements",
            "keywords": "sustainability, home improvement, eco-friendly, green living",
            "tone": "conversational",
            "target_word_count": 2500,
            "draft_title": "Sustainable Home Improvement Guide",
            "use_verbalized_sampling": True,  # Enable verbalized sampling
            "provider": "openai",
            "model": "gpt-4",
            "llm_key": os.getenv("OPENAI_API_KEY", "your-api-key-here")
        }
        
        # The research pipeline would automatically use verbalized sampling
        # when use_verbalized_sampling is set to True in research_data
        
        print("Research data configured for verbalized sampling:")
        print(f"  Brief: {research_data['brief']}")
        print(f"  Target word count: {research_data['target_word_count']}")
        print(f"  Verbalized sampling enabled: {research_data['use_verbalized_sampling']}")
        
        print("\nThe research pipeline will automatically:")
        print("  1. Use verbalized sampling for article structure generation")
        print("  2. Use verbalized sampling for content generation")
        print("  3. Select the best samples based on quality metrics")
        print("  4. Include sampling metadata in the final article")
        
    except Exception as e:
        logger.error(f"Error in pipeline integration example: {e}")

def main():
    """Run all examples."""
    logger.info("Verbalized Sampling Integration Examples")
    logger.info("=" * 50)
    
    examples = [
        example_basic_usage,
        example_content_generation,
        example_custom_sampling_parameters,
        example_integration_with_research_pipeline
    ]
    
    for example_func in examples:
        try:
            example_func()
            print("\n" + "=" * 50 + "\n")
        except Exception as e:
            logger.error(f"Example failed: {e}")
            print("\n" + "=" * 50 + "\n")
    
    logger.info("Examples completed!")

if __name__ == "__main__":
    main()


