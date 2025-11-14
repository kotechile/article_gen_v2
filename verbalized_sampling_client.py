"""
Verbalized Sampling Client for Content Generator V2.

This module provides a wrapper around the verbalized-sampling package to improve
LLM API writing quality through distribution sampling and selection.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import random

try:
    from verbalized_sampling import verbalize
except ImportError:
    # Fallback if verbalized-sampling is not installed
    verbalize = None

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class VerbalizedSamplingConfig:
    """Configuration for verbalized sampling."""
    k: int = 5  # Number of samples to generate
    tau: float = 0.10  # Temperature parameter for sampling
    temperature: float = 0.9  # Base temperature for generation
    seed: Optional[int] = None  # Random seed for reproducibility
    enabled: bool = True  # Whether to use verbalized sampling

@dataclass
class VerbalizedResponse:
    """Response from verbalized sampling."""
    text: str
    distribution: Any  # The distribution object from verbalized-sampling
    sample_index: int  # Which sample was selected
    all_samples: List[str]  # All generated samples
    metadata: Dict[str, Any]

class VerbalizedSamplingClient:
    """
    Client for verbalized sampling to improve LLM content generation.
    
    This client wraps the verbalized-sampling package to generate multiple
    samples and select the best one for article writing.
    """
    
    def __init__(self, config: VerbalizedSamplingConfig = None):
        """
        Initialize the verbalized sampling client.
        
        Args:
            config: Configuration for verbalized sampling
        """
        self.config = config or VerbalizedSamplingConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Check if verbalized-sampling is available
        if verbalize is None:
            self.logger.warning("verbalized-sampling package not installed. Falling back to standard generation.")
            self.config.enabled = False
    
    def generate_with_sampling(
        self, 
        prompt: str, 
        messages: Optional[List[Dict[str, str]]] = None,
        custom_config: Optional[VerbalizedSamplingConfig] = None
    ) -> VerbalizedResponse:
        """
        Generate content using verbalized sampling.
        
        Args:
            prompt: The prompt to generate content for
            messages: Optional messages list (for compatibility with LLM client)
            custom_config: Optional custom configuration
            
        Returns:
            VerbalizedResponse with the selected sample
        """
        config = custom_config or self.config
        
        if not config.enabled or verbalize is None:
            # Fallback to standard generation
            return self._fallback_generation(prompt)
        
        try:
            # Set seed for reproducibility
            if config.seed is not None:
                random.seed(config.seed)
            
            # Generate distribution of responses
            self.logger.info(f"Generating {config.k} samples with tau={config.tau}, temperature={config.temperature}")
            
            distribution = verbalize(
                prompt, 
                k=config.k, 
                tau=config.tau, 
                temperature=config.temperature
            )
            
            # Get all samples
            all_samples = []
            for i in range(config.k):
                try:
                    sample = distribution.sample(seed=config.seed + i if config.seed else None)
                    all_samples.append(sample.text)
                except Exception as e:
                    self.logger.warning(f"Failed to get sample {i}: {e}")
                    continue
            
            if not all_samples:
                self.logger.error("No samples generated, falling back to standard generation")
                return self._fallback_generation(prompt)
            
            # Select the best sample (for now, we'll use the first one)
            # In the future, this could be enhanced with quality scoring
            selected_sample = self._select_best_sample(all_samples, prompt)
            selected_index = all_samples.index(selected_sample)
            
            self.logger.info(f"Selected sample {selected_index + 1} out of {len(all_samples)} samples")
            
            return VerbalizedResponse(
                text=selected_sample,
                distribution=distribution,
                sample_index=selected_index,
                all_samples=all_samples,
                metadata={
                    "k": config.k,
                    "tau": config.tau,
                    "temperature": config.temperature,
                    "seed": config.seed,
                    "total_samples": len(all_samples)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in verbalized sampling: {e}")
            return self._fallback_generation(prompt)
    
    def _select_best_sample(self, samples: List[str], prompt: str) -> str:
        """
        Select the best sample from the generated samples.
        
        Args:
            samples: List of generated samples
            prompt: Original prompt for context
            
        Returns:
            Selected sample text
        """
        if not samples:
            return ""
        
        # For now, use simple heuristics to select the best sample
        # This could be enhanced with more sophisticated scoring
        
        scored_samples = []
        for sample in samples:
            score = self._score_sample(sample, prompt)
            scored_samples.append((score, sample))
        
        # Sort by score (higher is better) and return the best
        scored_samples.sort(key=lambda x: x[0], reverse=True)
        return scored_samples[0][1]
    
    def _score_sample(self, sample: str, prompt: str) -> float:
        """
        Score a sample based on various quality metrics.
        
        Args:
            sample: The sample text to score
            prompt: Original prompt for context
            
        Returns:
            Quality score (higher is better)
        """
        score = 0.0
        
        # Length appropriateness (not too short, not too long)
        word_count = len(sample.split())
        if 50 <= word_count <= 2000:
            score += 1.0
        elif 20 <= word_count < 50:
            score += 0.5
        elif word_count > 2000:
            score += 0.3
        
        # Completeness (ends with proper punctuation)
        if sample.strip().endswith(('.', '!', '?')):
            score += 0.5
        
        # Avoidance of repetitive patterns
        words = sample.lower().split()
        if len(set(words)) / len(words) > 0.7:  # Good vocabulary diversity
            score += 0.5
        
        # Avoidance of common issues
        if not any(phrase in sample.lower() for phrase in [
            'i cannot', 'i can\'t', 'i am not able to',
            'as an ai', 'i don\'t have', 'i\'m sorry'
        ]):
            score += 0.5
        
        # HTML structure quality (for content generation)
        if '<p>' in sample and '</p>' in sample:
            score += 0.3
        
        return score
    
    def _fallback_generation(self, prompt: str) -> VerbalizedResponse:
        """
        Fallback generation when verbalized sampling is not available.
        
        Args:
            prompt: The prompt to generate content for
            
        Returns:
            VerbalizedResponse with fallback content
        """
        self.logger.info("Using fallback generation (verbalized sampling not available)")
        
        # Fallback: Use standard LLM generation instead of placeholder text
        # The content generator should handle this fallback gracefully
        self.logger.warning("Verbalized sampling unavailable - should use standard generation as fallback")
        
        # Return empty/placeholder that will trigger standard generation in content_generator
        # Don't include "Content generated for:" prefix as it appears in final content
        fallback_text = ""  # Empty text will signal to use standard generation
        
        return VerbalizedResponse(
            text=fallback_text,
            distribution=None,
            sample_index=0,
            all_samples=[fallback_text],
            metadata={
                "fallback": True,
                "reason": "verbalized-sampling not available"
            }
        )
    
    def generate_content_with_sampling(
        self, 
        messages: List[Dict[str, str]], 
        content_type: str = "paragraph",
        word_count_target: int = 300
    ) -> VerbalizedResponse:
        """
        Generate content using verbalized sampling with article-specific optimization.
        
        Args:
            messages: List of message dictionaries
            content_type: Type of content to generate
            word_count_target: Target word count
            
        Returns:
            VerbalizedResponse with optimized content
        """
        # Extract the user message content
        user_content = ""
        for message in messages:
            if message.get("role") == "user":
                user_content = message.get("content", "")
                break
        
        if not user_content:
            return self._fallback_generation("No user content found")
        
        # Adjust sampling parameters based on content type and word count
        config = self._get_optimized_config(content_type, word_count_target)
        
        return self.generate_with_sampling(user_content, messages, config)
    
    def _get_optimized_config(
        self, 
        content_type: str, 
        word_count_target: int
    ) -> VerbalizedSamplingConfig:
        """
        Get optimized configuration based on content type and word count.
        
        Args:
            content_type: Type of content to generate
            word_count_target: Target word count
            
        Returns:
            Optimized VerbalizedSamplingConfig
        """
        # Base configuration
        config = VerbalizedSamplingConfig(
            k=self.config.k,
            tau=self.config.tau,
            temperature=self.config.temperature,
            seed=self.config.seed,
            enabled=self.config.enabled
        )
        
        # Adjust parameters based on content type
        if content_type == "paragraph":
            config.k = min(7, max(3, word_count_target // 100))  # More samples for longer content
            config.tau = 0.08  # Slightly lower tau for more focused content
        elif content_type == "list":
            config.k = 5  # Standard sampling for lists
            config.tau = 0.12  # Higher tau for more variety in list items
        elif content_type == "step_by_step":
            config.k = 6  # More samples for instructional content
            config.tau = 0.10  # Balanced tau for clear instructions
        elif content_type == "comparison":
            config.k = 8  # More samples for analytical content
            config.tau = 0.15  # Higher tau for diverse perspectives
        elif content_type == "table":
            config.k = 4  # Fewer samples for structured content
            config.tau = 0.06  # Lower tau for more consistent data presentation
        
        # Adjust based on word count
        if word_count_target > 1000:
            config.k = min(config.k + 2, 10)  # More samples for longer content
        elif word_count_target < 200:
            config.k = max(config.k - 1, 3)  # Fewer samples for shorter content
        
        return config

# Factory function
def create_verbalized_sampling_client(
    k: int = 5,
    tau: float = 0.10,
    temperature: float = 0.9,
    seed: Optional[int] = None,
    enabled: bool = True
) -> VerbalizedSamplingClient:
    """
    Create a verbalized sampling client.
    
    Args:
        k: Number of samples to generate
        tau: Temperature parameter for sampling
        temperature: Base temperature for generation
        seed: Random seed for reproducibility
        enabled: Whether to use verbalized sampling
        
    Returns:
        VerbalizedSamplingClient instance
    """
    config = VerbalizedSamplingConfig(
        k=k,
        tau=tau,
        temperature=temperature,
        seed=seed,
        enabled=enabled
    )
    
    return VerbalizedSamplingClient(config)

# Example usage
if __name__ == "__main__":
    # Example usage
    client = create_verbalized_sampling_client(
        k=5,
        tau=0.10,
        temperature=0.9,
        seed=42
    )
    
    prompt = "Write a compelling introduction paragraph about the benefits of renewable energy."
    
    try:
        response = client.generate_with_sampling(prompt)
        print(f"Selected text: {response.text}")
        print(f"Sample index: {response.sample_index}")
        print(f"Total samples: {len(response.all_samples)}")
        print(f"Metadata: {response.metadata}")
    except Exception as e:
        print(f"Error: {e}")

