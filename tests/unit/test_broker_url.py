"""Unit tests for broker (Redis) URL handling and failure classification.

These cover the bug that broke article generation: a broker hostname that
resolves to more than one container, and a password configured separately from
the URL that was never applied.
"""

import socket
import unittest
from unittest.mock import patch

from src.utils.broker import (
    classify_broker_error,
    generic_host_warning,
    get_broker_host,
    mask_credentials,
    normalize_broker_url,
    resolve_broker_addresses,
)


class NormalizeBrokerUrlTests(unittest.TestCase):
    def test_defaults_when_url_missing(self):
        self.assertEqual(normalize_broker_url(None), "redis://localhost:6379/0")
        self.assertEqual(normalize_broker_url(""), "redis://localhost:6379/0")

    def test_password_is_folded_into_url(self):
        self.assertEqual(
            normalize_broker_url("redis://broker:6379/0", "s3cret"),
            "redis://:s3cret@broker:6379/0",
        )

    def test_password_special_characters_are_encoded(self):
        normalized = normalize_broker_url("redis://broker:6379/0", "p@ss:w/rd")
        self.assertIn("p%40ss%3Aw%2Frd@broker", normalized)

    def test_explicit_url_password_wins(self):
        self.assertEqual(
            normalize_broker_url("redis://:fromurl@broker:6379/0", "fromenv"),
            "redis://:fromurl@broker:6379/0",
        )

    def test_username_is_preserved(self):
        self.assertEqual(
            normalize_broker_url("redis://default@broker:6379/1", "pw"),
            "redis://default:pw@broker:6379/1",
        )

    def test_db_index_defaults_to_zero(self):
        self.assertTrue(normalize_broker_url("redis://broker:6379").endswith("/0"))

    def test_rejects_non_redis_scheme(self):
        with self.assertRaises(ValueError):
            normalize_broker_url("http://broker:6379/0")


class MaskCredentialsTests(unittest.TestCase):
    def test_password_is_masked(self):
        masked = mask_credentials("redis://:s3cret@broker:6379/0")
        self.assertNotIn("s3cret", masked)
        self.assertIn("***", masked)
        self.assertIn("broker:6379", masked)

    def test_url_without_password_is_unchanged(self):
        self.assertEqual(mask_credentials("redis://broker:6379/0"), "redis://broker:6379/0")


class GenericHostTests(unittest.TestCase):
    def test_flags_the_shared_network_hostname(self):
        warning = generic_host_warning("redis://redis:6379/0")
        self.assertIsNotNone(warning)
        self.assertIn("redis", warning or "")

    def test_accepts_a_unique_service_name(self):
        self.assertIsNone(generic_host_warning("redis://broker:6379/0"))

    def test_get_broker_host(self):
        self.assertEqual(get_broker_host("redis://:pw@broker:6379/0"), "broker")


class ResolveAddressesTests(unittest.TestCase):
    def test_deduplicates_every_answer(self):
        fake_infos = [
            (socket.AF_INET, None, None, "", ("10.0.1.4", 6379)),
            (socket.AF_INET, None, None, "", ("10.0.1.12", 6379)),
            (socket.AF_INET, None, None, "", ("10.0.1.4", 6379)),
        ]
        with patch("socket.getaddrinfo", return_value=fake_infos):
            self.assertEqual(
                resolve_broker_addresses("redis", 6379),
                ["10.0.1.4", "10.0.1.12"],
            )

    def test_unresolvable_host_raises_readable_error(self):
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name or service not known")):
            with self.assertRaises(RuntimeError) as ctx:
                resolve_broker_addresses("does-not-exist", 6379)
        self.assertIn("does-not-exist", str(ctx.exception))


class ClassifyBrokerErrorTests(unittest.TestCase):
    def test_hello_authentication_error_is_authentication(self):
        exc = RuntimeError(
            "HELLO must be called with the client already authenticated, otherwise the "
            "HELLO <proto> AUTH <user> <pass> option can be used to authenticate the client"
        )
        self.assertEqual(classify_broker_error(exc), "authentication")

    def test_noauth_is_authentication(self):
        self.assertEqual(classify_broker_error(RuntimeError("NOAUTH Authentication required.")), "authentication")

    def test_connection_refused_is_connection(self):
        self.assertEqual(classify_broker_error(RuntimeError("Error 111 connecting: Connection refused.")), "connection")

    def test_unrelated_errors_stay_other(self):
        self.assertEqual(classify_broker_error(ValueError("bad payload")), "other")


if __name__ == "__main__":
    unittest.main()
