"""
Unit tests for ReferenceSearchClient (Tavily and Linkup).
"""

import unittest
from unittest.mock import MagicMock, patch

from src.services.context_image.reference_search import ReferenceSearchClient, ReferenceImageItem


class TestReferenceSearchClient(unittest.TestCase):
    def setUp(self):
        self.client = ReferenceSearchClient(
            tavily_api_key="fake-tavily-key",
            linkup_api_key="fake-linkup-key"
        )

    def test_sanitize_query(self):
        query = 'Apple "Watch" Ultra; \';;'
        sanitized = self.client._sanitize_query(query)
        self.assertEqual(sanitized, "Apple Watch Ultra")

    @patch("requests.post")
    def test_search_tavily_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "images": [
                {
                    "url": "https://cdn.example.com/apple_watch_front.jpg",
                    "description": "Apple Watch Ultra 2 front view"
                },
                "https://cdn.example.com/apple_watch_side.png"
            ],
            "results": [
                {
                    "title": "Apple Watch Review",
                    "images": ["https://cdn.example.com/apple_watch_wrist.jpg"]
                }
            ]
        }
        mock_post.return_value = mock_response

        images = self.client.search_reference_images("Apple Watch Ultra 2 studio photo", max_results=3)

        self.assertGreaterEqual(len(images), 2)
        self.assertEqual(images[0].url, "https://cdn.example.com/apple_watch_front.jpg")
        self.assertEqual(images[0].provider, "tavily")
        self.assertEqual(images[0].source_domain, "cdn.example.com")

        # Verify Tavily was called with include_images=True
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        self.assertTrue(payload["include_images"])
        self.assertEqual(payload["query"], "Apple Watch Ultra 2 studio photo")

    @patch("requests.post")
    def test_search_linkup_fallback_when_tavily_fails(self, mock_post):
        # First call (Tavily) fails with HTTP 500
        tavily_fail = MagicMock()
        tavily_fail.raise_for_status.side_effect = Exception("Tavily service unavailable")

        # Second call (Linkup) succeeds
        linkup_success = MagicMock()
        linkup_success.status_code = 200
        linkup_success.json.return_value = {
            "results": [
                {
                    "name": "Vintage Macintosh Official Product Page",
                    "url": "https://apple.com/mac-history",
                    "images": ["https://apple.com/assets/macintosh_128k.png"]
                }
            ]
        }

        mock_post.side_effect = [tavily_fail, linkup_success]

        images = self.client.search_reference_images("1984 Macintosh 128k studio photo")

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].url, "https://apple.com/assets/macintosh_128k.png")
        self.assertEqual(images[0].provider, "linkup")
        self.assertEqual(images[0].source_domain, "apple.com")

    def test_invalid_urls_filtered(self):
        self.assertFalse(self.client._is_valid_image_url(""))
        self.assertFalse(self.client._is_valid_image_url("ftp://example.com/pic.jpg"))
        self.assertFalse(self.client._is_valid_image_url("https://example.com/tracking/1x1.gif"))
        self.assertFalse(self.client._is_valid_image_url("https://example.com/favicon.ico"))
        self.assertTrue(self.client._is_valid_image_url("https://images.example.com/watch.png"))


if __name__ == "__main__":
    unittest.main()
