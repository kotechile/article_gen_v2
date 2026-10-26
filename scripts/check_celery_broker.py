#!/usr/bin/env python3
"""Verify that a Celery broker hostname really points at *our* Redis.

A broker URL like ``redis://redis:6379/0`` looks fine and still fails, because a
hostname is not an identity. On Coolify every compose service is attached to the
platform's shared ``coolify`` network, where Coolify's own cache container also
answers to the name ``redis``. DNS then returns two addresses, and every
connection that lands on the password-protected one dies with::

    redis.exceptions.AuthenticationError: HELLO must be called with the client
    already authenticated, otherwise the HELLO <proto> AUTH <user> <pass> ...

from the API (500 on POST /api/v1/research) and, for the worker, an endless::

    consumer: Cannot connect to redis://redis:6379/0: HELLO must be called ...
    Trying again in 32.00 seconds... (73/100)

This script answers the only question that matters: of the addresses behind the
broker hostname, which ones accept our credentials, and are any of them somebody
else's container?

Usage:
    python3 scripts/check_celery_broker.py                  # env: CELERY_BROKER_URL / REDIS_URL
    python3 scripts/check_celery_broker.py redis://broker:6379/0
    python3 scripts/check_celery_broker.py --json
    python3 scripts/check_celery_broker.py --workers        # also ping Celery workers

Exit codes:
    0  broker reachable (and workers answered, when --workers was asked)
    1  broker unreachable, rejected our credentials, or ambiguous hostname
    2  bad invocation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.broker import (  # noqa: E402  (path set up above)
    check_broker,
    generic_host_warning,
    mask_credentials,
)


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
    url = os.environ.get("CELERY_BROKER_URL") or os.environ.get("REDIS_URL")
    password = os.environ.get("REDIS_PASSWORD")
    if url:
        return url, password
    env = _read_env_file(REPO_ROOT / ".env")
    return (
        env.get("CELERY_BROKER_URL") or env.get("REDIS_URL"),
        env.get("REDIS_PASSWORD"),
    )


def probe_workers(url: str) -> dict:
    """Ask the Celery workers on this broker to ping back."""
    os.environ.setdefault("CELERY_BROKER_URL", url)
    try:
        from celery import Celery
    except ImportError as exc:  # pragma: no cover
        return {"status": "unknown", "error": f"celery unavailable: {exc}"}

    app = Celery("broker-check", broker=url)
    try:
        inspector = app.control.inspect(timeout=2.0)
        pings = (inspector.ping() if inspector else None) or {}
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return {"status": "unhealthy", "error": f"{type(exc).__name__}: {exc}"}
    if not pings:
        return {"status": "unhealthy", "error": "no worker answered within 2s"}
    return {"status": "healthy", "workers": list(pings.keys())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", nargs="?", help="broker URL (defaults to CELERY_BROKER_URL / REDIS_URL)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="machine-readable output")
    parser.add_argument("--workers", action="store_true", help="also ping Celery workers on this broker")
    parser.add_argument("--timeout", type=float, default=2.0, help="per-address connect timeout (default 2s)")
    args = parser.parse_args(argv)

    url = args.url or discover_target()[0]
    if not url:
        print("No broker URL: pass one or set CELERY_BROKER_URL / REDIS_URL", file=sys.stderr)
        return 2
    password = None if args.url else discover_target()[1]

    report = check_broker(url, password, timeout=args.timeout)
    report["generic_host_warning"] = generic_host_warning(report.get("url") or url)
    if args.workers and report.get("status") == "healthy":
        report["workers_check"] = probe_workers(report["url"])

    healthy = [a for a in report["addresses"] if a["status"] == "healthy"]
    rejected = [a for a in report["addresses"] if a["status"] == "authentication_error"]

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"broker      {mask_credentials(report.get('url') or url)}")
        print(f"hostname    {report.get('host')}:{report.get('port')}")
        print(f"addresses   {', '.join(report.get('resolved_addresses') or []) or '(none)'}")
        for entry in report["addresses"]:
            detail = f"  ({entry['detail'][:90]})" if entry.get("detail") else ""
            print(f"  - {entry['address']:<18} {entry['status']}{detail}")
        if report.get("workers_check"):
            print(f"workers     {report['workers_check']['status']}: "
                  f"{report['workers_check'].get('workers') or report['workers_check'].get('error')}")
        print(f"status      {report.get('status')}")
        if report.get("hint"):
            print(f"hint        {report['hint']}")
        if report.get("generic_host_warning"):
            print(f"warning     {report['generic_host_warning']}")

    ok = report.get("status") == "healthy" and not report.get("suspect_alias_collision")
    if rejected and healthy:
        ok = False  # two containers behind one name: only a matter of luck which answers
    if args.workers:
        ok = ok and report.get("workers_check", {}).get("status") == "healthy"
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
