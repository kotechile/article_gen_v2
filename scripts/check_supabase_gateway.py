#!/usr/bin/env python3
"""Verify that a Supabase URL really points at the API gateway (Kong/GoTrue), not Studio.

A very common self-hosted-Coolify failure is that `SERVICE_FQDN_*` ends up on the
Studio container instead of Kong. Everything then breaks in a confusing way:
`/auth/v1/authorize` (Google login) 404s with Supabase's own HTML 404 page and
`supabase-js` retries quietly.

Usage:
    python3 scripts/check_supabase_gateway.py                       # reads env / frontend/.env
    python3 scripts/check_supabase_gateway.py https://sb.example.com
    python3 scripts/check_supabase_gateway.py --json https://sb.example.com

Exit codes:
    0  a real API gateway answered
    1  no gateway answered (routing/env is wrong)
    2  bad invocation
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

TIMEOUT = 15
USER_AGENT = "supabase-gateway-check/1.0"

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# config discovery
# --------------------------------------------------------------------------- #
def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw = line.partition("=")
        values[key.strip()] = raw.strip().strip("'\"")
    return values


def discover_target() -> tuple[str | None, str | None]:
    """Return (supabase_url, anon_key) from the environment or checked-in env files."""
    url = os.environ.get("VITE_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = os.environ.get("VITE_SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if url and key:
        return url, key
    for rel in ("frontend/.env", ".env"):
        values = _read_env_file(REPO_ROOT / rel)
        url = url or values.get("VITE_SUPABASE_URL") or values.get("SUPABASE_URL")
        key = key or values.get("VITE_SUPABASE_ANON_KEY") or values.get("SUPABASE_ANON_KEY")
    return url, key


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
def _request(url: str, key: str | None = None) -> dict:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if key:
        headers["apikey"] = key
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # self-hosted stacks routinely use self-signed certs
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as resp:
            body = resp.read(4096).decode("utf-8", "replace")
            return {
                "status": resp.status,
                "headers": {k.lower(): v for k, v in resp.headers.items()},
                "body": body,
                "error": None,
            }
    except urllib.error.HTTPError as exc:  # 4xx/5xx still tells us who answered
        body = exc.read(4096).decode("utf-8", "replace")
        return {
            "status": exc.code,
            "headers": {k.lower(): v for k, v in (exc.headers or {}).items()},
            "body": body,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - network failures are expected here
        return {"status": None, "headers": {}, "body": "", "error": f"{type(exc).__name__}: {exc}"}


def _looks_like_html(response: dict) -> bool:
    ctype = response["headers"].get("content-type", "")
    if "html" in ctype.lower():
        return True
    head = response["body"].lstrip()[:200].lower()
    return head.startswith("<!doctype html") or head.startswith("<html")


def classify(base_url: str, anon_key: str | None) -> dict:
    base = base_url.rstrip("/")
    health = _request(f"{base}/auth/v1/health", anon_key)
    root = _request(f"{base}/", anon_key)
    rest = _request(f"{base}/rest/v1/", anon_key)

    try:
        health_json = json.loads(health["body"]) if health["body"] else None
    except json.JSONDecodeError:
        health_json = None

    powered_by = (health["headers"].get("x-powered-by") or root["headers"].get("x-powered-by") or "").lower()
    location = root["headers"].get("location", "")

    if health["error"] and health["status"] is None:
        verdict = "UNREACHABLE"
        detail = health["error"]
    elif isinstance(health_json, dict) and str(health_json.get("name", "")).lower() in {"gotrue", "auth"}:
        verdict = "GATEWAY_OK"
        detail = f"GoTrue {health_json.get('version', 'unknown')} on {base}/auth/v1"
    elif isinstance(health_json, dict) and health_json.get("message") == "no Route matched with those values":
        verdict = "GATEWAY_NO_ROUTE"
        detail = "Kong answered but has no /auth route — the Supabase service mapping is incomplete"
    elif "next.js" in powered_by or "/project/" in location or "supabase studio" in health["body"].lower():
        verdict = "STUDIO"
        detail = (
            "Supabase Studio owns this hostname (Next.js, "
            f"root redirect '{location or 'n/a'}'); the API gateway is not reachable here"
        )
    elif "coolify" in health["body"].lower() or "/login" in location:
        verdict = "COOLIFY_UI"
        detail = "Coolify's own dashboard owns this hostname, not Supabase"
    elif _looks_like_html(health) or _looks_like_html(rest):
        verdict = "HTML_404"
        detail = "An HTML app answered instead of the Supabase API — wrong service is bound to this hostname"
    else:
        verdict = "NOT_SUPABASE"
        detail = f"unexpected response (health status={health['status']}, root redirect='{location or 'n/a'}')"

    return {
        "url": base,
        "verdict": verdict,
        "detail": detail,
        "auth_health": {"status": health["status"], "content_type": health["headers"].get("content-type")},
        "root": {"status": root["status"], "location": location or None, "x_powered_by": powered_by or None},
        "rest": {"status": rest["status"], "content_type": rest["headers"].get("content-type")},
    }


FIX_HINTS = {
    "STUDIO": [
        "In Coolify, open the Supabase resource and check the domain variables:",
        "  SERVICE_FQDN_SUPABASEKONG / SERVICE_URL_SUPABASEKONG must hold the API hostname (e.g. sbcontent.example.com)",
        "  SERVICE_FQDN_SUPABASESTUDIO must NOT use the same hostname (e.g. studio.sbcontent.example.com)",
        "Redeploy/restart the stack, then re-run this check.",
        "Frontend + backend VITE_SUPABASE_URL / SUPABASE_URL must point at the KONG hostname, never Studio.",
    ],
    "GATEWAY_NO_ROUTE": [
        "Kong is reachable but the route is missing: confirm supabase-auth is running and SUPABASE_URL is set in the Kong service.",
    ],
    "COOLIFY_UI": [
        "This hostname is Coolify itself. Give the Supabase Kong service its own FQDN.",
    ],
    "UNREACHABLE": [
        "DNS/TLS/route failure — confirm the record points at the Supabase host and the stack is running.",
    ],
    "HTML_404": [
        "An HTML front end answered: the FQDN is bound to the wrong container (Studio or an app).",
    ],
    "NOT_SUPABASE": [
        "The hostname answers, but not like Supabase. Re-check the Coolify domain mapping.",
    ],
}


def next_gates(anon_key: str | None, api_base: str) -> list[str]:
    """Remaining config that only becomes checkable once routing works."""
    gates = [
        f"Google Cloud Console redirect URI must be {api_base}/auth/v1/callback",
        "gotrue env: GOTRUE_EXTERNAL_GOOGLE_ENABLED=true + CLIENT_ID/CLIENT_SECRET",
        f"gotrue env: GOTRUE_EXTERNAL_GOOGLE_REDIRECT_URI={api_base}/auth/v1/callback",
        "gotrue env: SITE_URL=<frontend origin> and ADDITIONAL_REDIRECT_URLS must include <frontend origin>/auth/callback",
    ]
    if not anon_key:
        gates.append("no anon key found in env — set VITE_SUPABASE_ANON_KEY / SUPABASE_ANON_KEY")
    return gates


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", nargs="?", help="Supabase base URL (defaults to env / frontend/.env)")
    parser.add_argument("--anon-key", help="anon key (defaults to env / frontend/.env)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    url = args.url
    anon_key = args.anon_key
    if not url:
        url, anon_key = discover_target()
        anon_key = args.anon_key or anon_key
    if not url:
        print("error: no Supabase URL given and none found in env or frontend/.env", file=sys.stderr)
        return 2

    if not re.match(r"^https?://", url):
        print(f"error: URL must include a scheme: {url}", file=sys.stderr)
        return 2

    result = classify(url, anon_key)
    ok = result["verdict"] == "GATEWAY_OK"
    result["ok"] = ok
    result["next_gates"] = [] if ok else FIX_HINTS.get(result["verdict"], []) + next_gates(anon_key, result["url"])

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Supabase URL : {result['url']}")
        print(f"Verdict      : {result['verdict']}")
        print(f"Detail       : {result['detail']}")
        print(f"  /auth/v1/health -> {result['auth_health']['status']} ({result['auth_health']['content_type']})")
        print(f"  /              -> {result['root']['status']} location={result['root']['location']} x-powered-by={result['root']['x_powered_by']}")
        print(f"  /rest/v1/      -> {result['rest']['status']} ({result['rest']['content_type']})")
        if result["next_gates"]:
            print("\nTo fix / verify next:")
            for hint in result["next_gates"]:
                print(f"  - {hint}")
        else:
            print("\nAPI gateway is healthy. If OAuth still fails, work through the gates above in Coolify -> supabase-auth.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
