"""
Unit tests for ReferenceSearchClient (Tavily and Linkup).
"""

import unittest
from unittest.mock import MagicMock, patch

from src.services.context_image.reference_search import ReferenceSearchClient, ReferenceImageItem


class TestReferenceSearchClient(unittest.TestCase):
    def setUp(self):
        self.client = ReferenceSearchClient(
            tavily_api_key="fake-tavily-key"
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

    @patch("requests.get")
    @patch("requests.post")
    def test_search_openverse_fallback_when_tavily_fails(self, mock_post, mock_get):
        # Tavily fails with HTTP 500
        tavily_fail = MagicMock()
        tavily_fail.raise_for_status.side_effect = Exception("Tavily service unavailable")
        mock_post.return_value = tavily_fail

        # Openverse fallback succeeds
        openverse_success = MagicMock()
        openverse_success.status_code = 200
        openverse_success.json.return_value = {
            "results": [
                {
                    "title": "Vintage Macintosh 128k",
                    "url": "https://images.openverse.org/macintosh_128k.png"
                }
            ]
        }
        mock_get.return_value = openverse_success

        images = self.client.search_reference_images("1984 Macintosh 128k studio photo")

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].url, "https://images.openverse.org/macintosh_128k.png")
        self.assertEqual(images[0].provider, "openverse")

    def test_invalid_urls_filtered(self):
        self.assertFalse(self.client._is_valid_image_url(""))
        self.assertFalse(self.client._is_valid_image_url("ftp://example.com/pic.jpg"))
        self.assertFalse(self.client._is_valid_image_url("https://example.com/tracking/1x1.gif"))
        self.assertFalse(self.client._is_valid_image_url("https://example.com/favicon.ico"))
        self.assertTrue(self.client._is_valid_image_url("https://images.example.com/watch.png"))

    @patch("requests.get")
    def test_search_openverse_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "url": "https://images.openverse.org/steering_wheel.jpg",
                    "thumbnail": "https://images.openverse.org/steering_wheel_thumb.jpg",
                    "title": "Car Steering Wheel on Highway"
                }
            ]
        }
        mock_get.return_value = mock_response

        # Test direct openverse call
        images = self.client._search_openverse("car steering wheel", max_results=2)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].provider, "openverse")
        self.assertEqual(images[0].url, "https://images.openverse.org/steering_wheel.jpg")

    @patch("requests.get")
    def test_search_wikimedia_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "12345": {
                        "title": "File:Vintage_Compass_Map.jpg",
                        "imageinfo": [
                            {
                                "url": "https://upload.wikimedia.org/compass.jpg",
                                "thumburl": "https://upload.wikimedia.org/compass_thumb.jpg"
                            }
                        ]
                    }
                }
            }
        }
        mock_get.return_value = mock_response

        images = self.client._search_wikimedia("compass map", max_results=2)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].provider, "wikimedia")
        self.assertIn("compass.jpg", images[0].url)


if __name__ == "__main__":
    unittest.main()
