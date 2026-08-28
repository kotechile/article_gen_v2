"""
Image Preprocessor for Context-Aware Image Generation.

Handles:
1. Downloading web reference images with headers, timeouts, and validation.
2. Optional background isolation using rembg.
3. Preparing the image for storage and multi-modal LLM conditioning.
"""

import base64
import io
import logging
from typing import Optional, Tuple
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def download_image(self, url: str) -> bytes:
        """Download image from a public URL with browser-like headers."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
        }
        res = requests.get(url, headers=headers, timeout=self.timeout, stream=True)
        res.raise_for_status()

        # Check content type
        content_type = res.headers.get("Content-Type", "")
        if "image" not in content_type and not any(ext in url.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]):
            logger.warning(f"URL {url} returned non-image content-type: {content_type}")

        return res.content

    def isolate_background(self, image_bytes: bytes) -> bytes:
        """
        Attempt to remove background using rembg if available.
        Falls back smoothly to original bytes if rembg is not installed.
        """
        try:
            import rembg
            output = rembg.remove(image_bytes)
            logger.info("Successfully removed background from reference image using rembg")
            return output
        except ImportError:
            logger.info("rembg is not installed; continuing with original reference image")
            return image_bytes
        except Exception as e:
            logger.warning(f"Failed to remove background with rembg: {e}; using original image")
            return image_bytes

    def prepare_reference(
        self,
        image_url_or_bytes: str | bytes,
        isolate_bg: bool = False
    ) -> Tuple[bytes, str]:
        """
        Prepares reference image, returning (image_bytes, base64_str).
        """
        if isinstance(image_url_or_bytes, str):
            if image_url_or_bytes.startswith("data:image"):
                # Data URL
                _, b64data = image_url_or_bytes.split(",", 1)
                image_bytes = base64.b64decode(b64data)
            elif image_url_or_bytes.startswith("http"):
                image_bytes = self.download_image(image_url_or_bytes)
            else:
                image_bytes = base64.b64decode(image_url_or_bytes)
        else:
            image_bytes = image_url_or_bytes

        if isolate_bg:
            image_bytes = self.isolate_background(image_bytes)

        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        return image_bytes, base64_str
