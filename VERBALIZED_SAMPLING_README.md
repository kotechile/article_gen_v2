# Verbalized Sampling Integration for Content Generator V2

This document describes the integration of verbalized-sampling into the Content Generator V2 system to improve LLM API writing quality through distribution sampling and selection.

## Overview

Verbalized sampling is a technique that generates multiple samples from an LLM and selects the best one based on quality metrics. This integration enhances the content generation pipeline by:

- Generating multiple content variations for each section
- Selecting the highest-quality sample based on scoring criteria
- Providing fallback behavior when verbalized-sampling is not available
- Maintaining compatibility with existing content generation workflows

## Installation

The verbalized-sampling package has been added to `requirements.txt`:

```bash
pip install verbalized-sampling
```

## Components

### 1. VerbalizedSamplingClient

A wrapper around the verbalized-sampling package that provides:

- **Distribution Generation**: Creates multiple samples using configurable parameters
- **Sample Selection**: Selects the best sample based on quality metrics
- **Fallback Support**: Graceful degradation when verbalized-sampling is unavailable
- **Content-Type Optimization**: Different sampling parameters for different content types

**Key Features:**
- Configurable sampling parameters (k, tau, temperature, seed)
- Quality scoring based on length, completeness, vocabulary diversity
- Support for different content types (paragraph, list, comparison, etc.)
- Comprehensive metadata tracking

### 2. Enhanced ContentGenerator

The `ContentGenerator` class now supports verbalized sampling:

```python
# Enable verbalized sampling
content_generator = create_content_generator(
    llm_client=llm_client,
    use_verbalized_sampling=True
)
```

**Features:**
- Automatic sampling for paragraph content generation
- Enhanced metadata including sampling information
- Backward compatibility with standard generation
- Configurable sampling parameters per content type

### 3. Enhanced ArticleStructureGenerator

The `ArticleStructureGenerator` also supports verbalized sampling for:

- Title generation
- Hook creation
- Excerpt writing
- Thesis statement generation
- Meta description creation

### 4. Updated LLMClient

The `LLMClient` now includes verbalized sampling configuration:

```python
config = LLMConfig(
    provider="openai",
    model="gpt-4",
    api_key="your-key",
    use_verbalized_sampling=True,
    verbalized_k=5,
    verbalized_tau=0.10,
    verbalized_temperature=0.9,
    verbalized_seed=42
)
```

## Usage

### Basic Usage

```python
from verbalized_sampling_client import create_verbalized_sampling_client

# Create client
client = create_verbalized_sampling_client(
    k=5,                    # Number of samples
    tau=0.10,              # Temperature parameter
    temperature=0.9,       # Base temperature
    seed=42,               # Random seed
    enabled=True           # Enable sampling
)

# Generate content
response = client.generate_with_sampling("Write about renewable energy benefits.")
print(response.text)  # Selected best sample
```

### Content Generation with Sampling

```python
from content_generator import create_content_generator
from llm_client import create_llm_client

# Create LLM client with verbalized sampling enabled
llm_client = create_llm_client(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key",
    use_verbalized_sampling=True
)

# Create content generator
content_generator = create_content_generator(
    llm_client=llm_client,
    use_verbalized_sampling=True
)

# Generate section content (automatically uses sampling)
section_content = content_generator.generate_section_content(
    section_outline=outline,
    research_data=research_data,
    claims=claims,
    evidence=evidence
)
```

### Research Pipeline Integration

Enable verbalized sampling in the research pipeline by setting the flag in research data:

```python
research_data = {
    "brief": "Your article brief",
    "use_verbalized_sampling": True,  # Enable sampling
    "provider": "openai",
    "model": "gpt-4",
    "llm_key": "your-api-key"
}
```

The pipeline will automatically:
1. Use verbalized sampling for article structure generation
2. Use verbalized sampling for content generation
3. Select the best samples based on quality metrics
4. Include sampling metadata in the final article

## Configuration Parameters

### Sampling Parameters

- **k**: Number of samples to generate (default: 5)
- **tau**: Temperature parameter for sampling (default: 0.10)
- **temperature**: Base temperature for generation (default: 0.9)
- **seed**: Random seed for reproducibility (default: None)

### Content-Type Specific Parameters

Different content types use optimized sampling parameters:

- **Paragraph**: k=7, tau=0.08 (more samples, focused content)
- **List**: k=5, tau=0.12 (standard sampling, varied items)
- **Comparison**: k=8, tau=0.15 (more samples, diverse perspectives)
- **Step-by-step**: k=6, tau=0.10 (balanced sampling, clear instructions)
- **Table**: k=4, tau=0.06 (fewer samples, consistent data)

## Quality Scoring

The system uses multiple criteria to score and select the best sample:

1. **Length Appropriateness**: Not too short (< 50 words) or too long (> 2000 words)
2. **Completeness**: Proper sentence endings and structure
3. **Vocabulary Diversity**: Avoids repetitive patterns
4. **Content Quality**: Avoids common AI limitations
5. **HTML Structure**: Proper formatting for web content

## Fallback Behavior

When verbalized-sampling is not available or disabled:

1. The system automatically falls back to standard LLM generation
2. All existing functionality remains unchanged
3. No breaking changes to the API
4. Graceful error handling and logging

## Testing

Run the test suite to verify the integration:

```bash
python test_verbalized_sampling.py
```

The test suite covers:
- VerbalizedSamplingClient functionality
- ContentGenerator integration
- ArticleStructureGenerator integration
- Fallback behavior
- Error handling

## Examples

See `example_verbalized_sampling.py` for comprehensive usage examples including:

- Basic verbalized sampling usage
- Content generation with sampling
- Custom sampling parameters
- Full pipeline integration

## Benefits

1. **Improved Content Quality**: Multiple samples allow selection of the best content
2. **Consistency**: Reproducible results with seed-based sampling
3. **Flexibility**: Configurable parameters for different content types
4. **Reliability**: Fallback behavior ensures system stability
5. **Transparency**: Comprehensive metadata tracking for analysis

## API Reference

### VerbalizedSamplingClient

```python
class VerbalizedSamplingClient:
    def __init__(self, config: VerbalizedSamplingConfig = None)
    def generate_with_sampling(self, prompt: str, messages: Optional[List[Dict]] = None, custom_config: Optional[VerbalizedSamplingConfig] = None) -> VerbalizedResponse
    def generate_content_with_sampling(self, messages: List[Dict], content_type: str = "paragraph", word_count_target: int = 300) -> VerbalizedResponse
```

### VerbalizedResponse

```python
@dataclass
class VerbalizedResponse:
    text: str                    # Selected sample text
    distribution: Any           # Distribution object
    sample_index: int          # Index of selected sample
    all_samples: List[str]     # All generated samples
    metadata: Dict[str, Any]   # Sampling metadata
```

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure `verbalized-sampling` package is installed
2. **API Key Issues**: Verify your OpenAI/OpenRouter API key is set
3. **Sampling Failures**: Check fallback behavior is working correctly
4. **Performance**: Adjust sampling parameters (k, tau) for your use case

### Debugging

Enable verbose logging to debug sampling behavior:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Potential improvements for future versions:

1. **Advanced Quality Metrics**: More sophisticated scoring algorithms
2. **Content-Type Detection**: Automatic parameter optimization
3. **Batch Processing**: Efficient sampling for multiple sections
4. **Cost Optimization**: Smart sampling based on content importance
5. **A/B Testing**: Compare sampling vs. standard generation results

## Contributing

When contributing to the verbalized sampling integration:

1. Maintain backward compatibility
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Follow existing code style and patterns
5. Include fallback behavior for new features

