"""Broker (Redis) URL helpers and diagnostics for Celery.

Celery publishes tasks through ``kombu`` into Redis. When the broker URL is
wrong in a *hostname* sense - rather than a credentials sense - the failure is
easy to misread. The classic case on Coolify: a compose service called
``redis`` lives beside Coolify's own shared cache, which also answers to the
hostname ``redis`` on the ``coolify`` network. The name then resolves to two
containers, and every connection that lands on the password-protected one dies
with::

    redis.exceptions.AuthenticationError: HELLO must be called with the client
    already authenticated, otherwise the HELLO <proto> AUTH <user> <pass> option
    can be used ...

which reads like "our Redis password is wrong" when nothing about the password
is wrong - the container we reached is not ours at all.

The helpers here keep that diagnosis in one place: normalise a broker URL
(honouring a separate ``REDIS_PASSWORD``), resolve the hostname to *every*
address it answers on, and report which of those addresses accept the
configured credentials.
"""

from __future__ import annotations

import logging
import socket
from typing import Any, Dict, List, Optional
from urllib.parse import quote, unquote, urlsplit, urlunsplit

logger = logging.getLogger(__name__)

DEFAULT_REDIS_PORT = 6379
DEFAULT_BROKER_URL = "redis://localhost:6379/0"

# Hostnames that are too generic to be safe inside a shared Docker network: the
# platform's own cache container very often answers to the same name. Deliberately
# narrow - only names a platform cache actually uses - so the warning stays
# actionable instead of firing on every local setup.
GENERIC_BROKER_HOSTS = frozenset({"redis", "redis-server", "cache"})

AUTH_ERROR_MARKERS = (
    "hello must be called with the client already authenticated",
    "authentication required",
    "invalid password",
    "wrongpass",
    "noauth",
    "authenticationerror",
)


# --------------------------------------------------------------------------- #
# URL handling
# --------------------------------------------------------------------------- #
def normalize_broker_url(url: Optional[str], password: Optional[str] = None) -> str:
    """Return a usable broker URL, folding a standalone password into it.

    ``REDIS_PASSWORD`` is commonly configured on its own (docker-compose
    secrets, Coolify envs). Redis URLs that omit it fail with an authentication
    error that looks like a wrong password, so apply it here instead of hoping
    every process remembers to.

    An explicit password already present in the URL always wins.
    """
    raw = (url or "").strip() or DEFAULT_BROKER_URL
    parts = urlsplit(raw)

    if parts.scheme not in ("redis", "rediss"):
        raise ValueError(f"Unsupported broker scheme {parts.scheme!r} in {mask_credentials(raw)}")

    username = parts.username
    url_password = parts.password
    host = parts.hostname or "localhost"
    port = parts.port or DEFAULT_REDIS_PORT
    path = parts.path or "/0"

    effective_password = url_password if url_password is not None else (password or None)

    netloc = host
    if effective_password is not None:
        credentials = quote(unquote(effective_password), safe="")
        netloc = f":{credentials}@{host}"
        if username:
            netloc = f"{quote(username, safe='')}:{credentials}@{host}"
    elif username:
        netloc = f"{quote(username, safe='')}@{host}"

    netloc = f"{netloc}:{port}"
    return urlunsplit((parts.scheme, netloc, path, parts.query, ""))


def mask_credentials(url: Optional[str]) -> str:
    """Return the URL with any password replaced by ``***`` (safe for logs)."""
    if not url:
        return ""
    parts = urlsplit(url.strip())
    if parts.password is None:
        return url.strip()
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    user = f"{parts.username}:" if parts.username else ""
    return urlunsplit((parts.scheme, f"{user}***@{host}{port}", parts.path, parts.query, ""))


def get_broker_host(url: Optional[str]) -> str:
    """Return just the hostname of a broker URL."""
    return urlsplit((url or DEFAULT_BROKER_URL).strip()).hostname or "localhost"


def generic_host_warning(url: Optional[str]) -> Optional[str]:
    """Return a warning string when the broker hostname is too generic to be safe."""
    host = get_broker_host(url)
    if host in GENERIC_BROKER_HOSTS:
        return (
            f"Broker hostname {host!r} is generic: inside a shared Docker network "
            "(Coolify attaches every service to its own `coolify` network) another "
            "container - typically the platform's password-protected cache - answers "
            "to the same name. Point the broker at a service name nobody else uses, "
            "e.g. redis://broker:6379/0."
        )
    return None


# --------------------------------------------------------------------------- #
# Resolution / connectivity
# --------------------------------------------------------------------------- #
def resolve_broker_addresses(host: str, port: int = DEFAULT_REDIS_PORT) -> List[str]:
    """Return every address a broker hostname resolves to (deduplicated)."""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise RuntimeError(f"Cannot resolve broker host {host!r}: {exc}") from exc

    addresses: List[str] = []
    for info in infos:
        address = str(info[4][0])
        if address not in addresses:
            addresses.append(address)
    return addresses


def classify_broker_error(exc: BaseException) -> str:
    """Classify a broker exception as ``authentication``, ``connection`` or ``other``."""
    message = f"{type(exc).__name__}: {exc}".lower()
    if "hello must be called with the client already authenticated" in message:
        # A real auth error, but also the signature of a wrong-container hit.
        return "authentication"
    if any(marker in message for marker in AUTH_ERROR_MARKERS):
        return "authentication"
    if any(
        marker in message
        for marker in ("connection refused", "name or service not known",
                       "nodename nor servname", "timeout", "timed out", "cannot connect")
    ):
        return "connection"
    return "other"


def broker_failure_hint(exc: BaseException, url: Optional[str] = None) -> str:
    """Build an actionable one-line hint for a broker failure."""
    host = get_broker_host(url)
    kind = classify_broker_error(exc)

    if kind == "authentication":
        return (
            f"The broker at '{host}' rejected us as unauthenticated. Either its password is "
            f"missing from REDIS_URL/CELERY_BROKER_URL, or '{host}' resolves to a container "
            f"that is not ours (a second container answering to the same hostname on a shared "
            f"Docker network). Check which address answered with "
            f"`python3 scripts/check_celery_broker.py`."
        )
    if kind == "connection":
        return (
            f"The broker at '{host}' is unreachable from this process. Start the broker "
            f"container/service and confirm the hostname in REDIS_URL matches a running "
            f"service (not a sibling stack's)."
        )
    return f"The broker publish failed: {type(exc).__name__}: {exc}"


def check_broker(url: Optional[str], password: Optional[str] = None, timeout: float = 1.5) -> Dict[str, Any]:
    """Probe every address behind the broker hostname and report who answers.

    Returns a dict shaped for logs and health endpoints::

        {"status": "healthy"|"unhealthy", "host": "...", "port": ..., "addresses": [
            {"address": "10.0.1.3", "status": "authentication_error", "detail": "..."},
            {"address": "10.0.1.9", "status": "healthy"},
        ], "hint": ..., "suspect_alias_collision": bool}
    """
    normalized = normalize_broker_url(url, password)
    host = get_broker_host(normalized)
    port = urlsplit(normalized).port or DEFAULT_REDIS_PORT

    report: Dict[str, Any] = {
        "status": "unhealthy",
        "host": host,
        "port": port,
        "url": mask_credentials(normalized),
        "addresses": [],
        "suspect_alias_collision": False,
        "hint": None,
    }

    try:
        addresses = resolve_broker_addresses(host, port)
    except RuntimeError as exc:
        report["error"] = str(exc)
        report["hint"] = broker_failure_hint(exc, normalized)
        return report
    report["resolved_addresses"] = addresses

    try:
        import redis  # imported lazily so diagnostics still run without the package
    except ImportError as exc:  # pragma: no cover - redis is a hard dependency in practice
        report["error"] = f"redis package unavailable: {exc}"
        return report

    credentials = urlsplit(normalized)
    for address in addresses:
        entry: Dict[str, Any] = {"address": address}
        client = None
        try:
            client = redis.Redis(
                host=address,
                port=port,
                password=unquote(credentials.password) if credentials.password else None,
                username=unquote(credentials.username) if credentials.username else None,
                socket_connect_timeout=timeout,
                socket_timeout=timeout,
            )
            client.ping()
            entry["status"] = "healthy"
            try:
                entry["version"] = client.info("server").get("redis_version")
            except Exception:  # info() can be denied by ACLs; not fatal
                pass
        except Exception as exc:  # noqa: BLE001 - every address is reported, none abort
            kind = classify_broker_error(exc)
            entry["status"] = f"{kind}_error"
            entry["detail"] = str(exc)[:300]
        finally:
            try:
                if client is not None:
                    client.close()
            except Exception:
                pass
        report["addresses"].append(entry)

    healthy = [a for a in report["addresses"] if a["status"] == "healthy"]
    rejected = [a for a in report["addresses"] if a["status"] == "authentication_error"]

    if healthy:
        report["status"] = "healthy"
    if rejected and healthy:
        report["suspect_alias_collision"] = True
        report["hint"] = (
            f"'{host}' resolves to {len(report['resolved_addresses'])} addresses and "
            f"{len(rejected)} of them demand a password we do not have "
            f"({', '.join(a['address'] for a in rejected)}). That is a second container "
            f"answering to the same hostname, not our broker. Give the broker service a "
            f"hostname nobody else uses."
        )
    elif rejected and not healthy:
        report["hint"] = broker_failure_hint(
            RuntimeError("authentication required"), normalized
        ) if not report["addresses"] else (
            f"Every address behind '{host}' refused our credentials. If our own broker has no "
            f"password, another container is answering to that hostname; if it has one, "
            f"REDIS_PASSWORD does not match it."
        )
    elif not healthy:
        report["hint"] = broker_failure_hint(RuntimeError("connection refused"), normalized)

    return report
