"""
Context-Aware Image Generation Service Package.

Provides automated entity extraction, web reference image retrieval
(via Linkup and Tavily), image preprocessing, and reference-conditioned
image generation using Nano Banana Pro and Flux 2.
"""

from .entity_extractor import EntityExtractor, EntityExtractionResult
from .reference_search import ReferenceSearchClient, ReferenceImageItem
from .image_preprocessor import ImagePreprocessor
from .context_pipeline import ContextImagePipeline

__all__ = [
    "EntityExtractor",
    "EntityExtractionResult",
    "ReferenceSearchClient",
    "ReferenceImageItem",
    "ImagePreprocessor",
    "ContextImagePipeline",
]
