"""
Unit tests for LinkedIn Service and API Endpoints.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from src.services.linkedin_service import LinkedInService


class TestLinkedInService(unittest.TestCase):
    def setUp(self):
        self.service = LinkedInService(
            client_id="test_client_id",
            client_secret="test_client_secret",
            redirect_uri="http://localhost:5001/api/linkedin/callback",
        )

    def test_get_authorization_url(self):
        url = self.service.get_authorization_url(state="test_state_123")
        self.assertIn("https://www.linkedin.com/oauth/v2/authorization", url)
        self.assertIn("client_id=test_client_id", url)
        self.assertIn("redirect_uri=http%3A%2F%2Flocalhost%3A5001%2Fapi%2Flinkedin%2Fcallback", url)
        self.assertIn("state=test_state_123", url)
        self.assertIn("w_member_social", url)

    def test_get_authorization_url_missing_client_id(self):
        service = LinkedInService(client_id="", client_secret="secret")
        with self.assertRaises(ValueError):
            service.get_authorization_url()

    @patch("requests.post")
    def test_exchange_code_for_token(self, mock_post):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "access_token": "mock_access_token_123",
            "expires_in": 5184000,
            "refresh_token": "mock_refresh_token_456",
        }
        mock_post.return_value = mock_response

        token_data = self.service.exchange_code_for_token("sample_auth_code")
        self.assertEqual(token_data["access_token"], "mock_access_token_123")
        self.assertEqual(token_data["refresh_token"], "mock_refresh_token_456")
        self.assertIn("expires_at", token_data)

    @patch("requests.get")
    def test_get_member_profile(self, mock_get):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "sub": "user_abc_789",
            "name": "Jane Doe",
            "email": "jane@example.com",
            "picture": "https://media.licdn.com/image.jpg",
        }
        mock_get.return_value = mock_response

        profile = self.service.get_member_profile("mock_token")
        self.assertEqual(profile["urn"], "urn:li:person:user_abc_789")
        self.assertEqual(profile["name"], "Jane Doe")
        self.assertEqual(profile["picture"], "https://media.licdn.com/image.jpg")

    @patch("requests.post")
    def test_publish_text_post(self, mock_post):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.headers = {"x-restli-id": "urn:li:share:123456789"}
        mock_response.json.return_value = {"id": "urn:li:share:123456789"}
        mock_post.return_value = mock_response

        res = self.service.publish_post(
            access_token="mock_token",
            author_urn="urn:li:person:user_abc_789",
            commentary="Excited to announce our new product update! #Innovation",
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["post_urn"], "urn:li:share:123456789")
        self.assertIn("urn:li:share:123456789", res["post_url"])

        # Check payload sent to LinkedIn
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        self.assertEqual(payload["author"], "urn:li:person:user_abc_789")
        self.assertEqual(payload["commentary"], "Excited to announce our new product update! #Innovation")
        self.assertEqual(payload["visibility"], "PUBLIC")

    @patch("requests.post")
    def test_publish_article_link_share(self, mock_post):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.headers = {"x-restli-id": "urn:li:share:987654321"}
        mock_post.return_value = mock_response

        res = self.service.publish_post(
            access_token="mock_token",
            author_urn="urn:li:person:user_abc_789",
            commentary="Check out our latest deep dive:",
            article_url="https://example.com/blog/future-of-ai",
            article_title="The Future of AI in 2026",
            article_description="A breakdown of major shifts in agentic coding.",
        )
        self.assertTrue(res["success"])

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        self.assertIn("article", payload["content"])
        self.assertEqual(payload["content"]["article"]["source"], "https://example.com/blog/future-of-ai")
        self.assertEqual(payload["content"]["article"]["title"], "The Future of AI in 2026")

    @patch("src.services.linkedin_service.LinkedInService.upload_image")
    @patch("requests.post")
    def test_publish_image_post(self, mock_post, mock_upload):
        mock_upload.return_value = "urn:li:image:img_111222"
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.headers = {"x-restli-id": "urn:li:share:333444"}
        mock_post.return_value = mock_response

        res = self.service.publish_post(
            access_token="mock_token",
            author_urn="urn:li:person:user_abc_789",
            commentary="Visual breakdown attached:",
            image_url="https://example.com/chart.png",
            image_alt_text="AI Performance Chart",
        )
        self.assertTrue(res["success"])

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        self.assertEqual(payload["content"]["media"]["id"], "urn:li:image:img_111222")
        self.assertEqual(payload["content"]["media"]["altText"], "AI Performance Chart")

    def test_commentary_truncation_safety(self):
        long_commentary = "a" * 3500
        with patch("requests.post") as mock_post:
            mock_res = MagicMock()
            mock_res.ok = True
            mock_res.headers = {"x-restli-id": "urn:li:share:trunc_test"}
            mock_post.return_value = mock_res

            res = self.service.publish_post(
                access_token="token",
                author_urn="urn:li:person:user1",
                commentary=long_commentary,
            )
            self.assertTrue(res["success"])

            _, kwargs = mock_post.call_args
            sent_commentary = kwargs["json"]["commentary"]
            self.assertLessEqual(len(sent_commentary), 3000)
            self.assertTrue(sent_commentary.endswith("..."))


if __name__ == "__main__":
    unittest.main()
