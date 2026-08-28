"""
End-to-End Orchestration Pipeline for Context-Aware Image Generation.

Coordinates:
1. Entity extraction and prompt synthesis from text excerpts.
2. Web reference image discovery via Linkup & Tavily.
3. Preprocessing, background isolation, and storage upload.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .entity_extractor import EntityExtractor, EntityExtractionResult
from .reference_search import ReferenceSearchClient, ReferenceImageItem
from .image_preprocessor import ImagePreprocessor
from supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class ContextImagePipeline:
    def __init__(
        self,
        entity_extractor: Optional[EntityExtractor] = None,
        search_client: Optional[ReferenceSearchClient] = None,
        preprocessor: Optional[ImagePreprocessor] = None
    ):
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.search_client = search_client or ReferenceSearchClient()
        self.preprocessor = preprocessor or ImagePreprocessor()

    def analyze_context(
        self,
        text: str,
        user_instructions: Optional[str] = None,
        max_reference_images: int = 6
    ) -> Dict[str, Any]:
        """
        Extract the target entity, generate search queries & prompt, and fetch reference images.
        """
        logger.info("Starting context analysis for article excerpt")
        extraction: EntityExtractionResult = self.entity_extractor.extract(
            text=text,
            user_instructions=user_instructions
        )

        references: List[ReferenceImageItem] = []
        if extraction.search_query:
            try:
                references = self.search_client.search_reference_images(
                    query=extraction.search_query,
                    max_results=max_reference_images
                )
            except Exception as e:
                logger.error(f"Error fetching reference images: {e}", exc_info=True)

        return {
            "has_physical_entity": extraction.has_physical_entity,
            "entity_type": extraction.entity_type,
            "is_metaphorical": extraction.is_metaphorical,
            "main_object": extraction.main_object,
            "search_query": extraction.search_query,
            "generation_prompt": extraction.generation_prompt,
            "object_fidelity_weight": extraction.object_fidelity_weight,
            "candidate_references": [ref.to_dict() for ref in references]
        }

    def prepare_reference_asset(
        self,
        reference_url: Optional[str] = None,
        reference_base64: Optional[str] = None,
        isolate_bg: bool = False,
        user_id: Optional[str] = None
    ) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Download, optionally isolate background, and upload reference to Supabase Storage.
        Returns: (reference_bytes, reference_http_url)
        """
        source = reference_url or reference_base64
        if not source:
            return None, None

        ref_bytes, ref_b64 = self.preprocessor.prepare_reference(source, isolate_bg=isolate_bg)

        # Upload to Supabase Storage so URL-based providers (like KIE Flux) have a permanent HTTP URL
        http_url = reference_url
        if user_id and ref_bytes:
            try:
                client = get_supabase_client()
                if client:
                    filename = f"context_ref_{int(datetime.utcnow().timestamp())}.jpg"
                    storage_path = f"{user_id}/context_refs/{filename}"
                    client.storage.from_('User Files').upload(
                        path=storage_path,
                        file=ref_bytes,
                        file_options={"content-type": "image/jpeg"}
                    )
                    http_url = client.storage.from_('User Files').get_public_url(storage_path)
            except Exception as e:
                logger.warning(f"Could not store reference image in Supabase: {e}")

        return ref_bytes, http_url
