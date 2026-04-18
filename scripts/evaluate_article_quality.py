#!/usr/bin/env python3
"""
Evaluate generated article quality for existing Titles rows.

Phase 0 rubric runner:
- Scores existing articles using the same quality evaluator used in generation
- Prints aggregate metrics for baseline comparisons
- Optionally backfills quality_report and overall_quality_score in Titles

Usage examples:
  python3 scripts/evaluate_article_quality.py --limit 50
  python3 scripts/evaluate_article_quality.py --status Created --limit 100
  python3 scripts/evaluate_article_quality.py --title-id <uuid>
  python3 scripts/evaluate_article_quality.py --status Created --limit 100 --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean
from typing import Any, Dict, List

# Ensure project root imports work when run from scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from supabase_client import get_supabase_client  # noqa: E402
from src.services.article_quality_evaluator import build_article_quality_report  # noqa: E402


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _build_query(args, supabase):
    query = (
        supabase.table("Titles")
        .select("id,user_id,Title,status,articleText,htmlArticle,citations,overall_quality_score")
        .limit(args.limit)
    )

    if args.status:
        query = query.eq("status", args.status)
    if args.title_id:
        query = query.eq("id", args.title_id)
    if args.user_id:
        query = query.eq("user_id", args.user_id)

    return query


def _parse_citations(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _print_summary(reports: List[Dict[str, Any]], skipped: int):
    if not reports:
        print("No evaluatable article records found.")
        if skipped:
            print(f"Skipped rows: {skipped}")
        return

    overall_scores = [_safe_float(r["overall_score"]) for r in reports]
    human_scores = [_safe_float(r["humanization_score"]) for r in reports]
    grounding_scores = [_safe_float(r["grounding_score"]) for r in reports]
    geo_scores = [_safe_float(r["geo_score"]) for r in reports]

    print("\n=== Article Quality Baseline ===")
    print(f"Evaluated rows: {len(reports)}")
    print(f"Skipped rows: {skipped}")
    print(f"Average overall score: {mean(overall_scores):.2f}")
    print(f"Average humanization score: {mean(human_scores):.2f}")
    print(f"Average grounding score: {mean(grounding_scores):.2f}")
    print(f"Average GEO score: {mean(geo_scores):.2f}")
    print(f"Min/Max overall score: {min(overall_scores):.2f} / {max(overall_scores):.2f}")

    low_quality = [r for r in reports if _safe_float(r["overall_score"]) < 60]
    print(f"Rows below 60 overall: {len(low_quality)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate quality for generated Titles articles")
    parser.add_argument("--limit", type=int, default=50, help="Maximum rows to evaluate")
    parser.add_argument("--status", type=str, default="Created", help="Filter by Titles.status")
    parser.add_argument("--title-id", type=str, default="", help="Evaluate only one title id")
    parser.add_argument("--user-id", type=str, default="", help="Filter by user id")
    parser.add_argument("--apply", action="store_true", help="Persist quality_report and overall_quality_score")
    args = parser.parse_args()

    supabase = get_supabase_client()
    if not supabase:
        raise SystemExit("Supabase client unavailable. Check SUPABASE_URL and key env vars.")

    response = _build_query(args, supabase).execute()
    rows = response.data or []

    if not rows:
        print("No matching Titles rows found.")
        return

    reports: List[Dict[str, Any]] = []
    skipped = 0
    updated = 0

    for row in rows:
        title_id = row.get("id")
        title = row.get("Title") or "Untitled Article"
        html_article = row.get("htmlArticle") or ""
        article_text = row.get("articleText") or ""
        citations = _parse_citations(row.get("citations"))

        if not html_article and not article_text:
            skipped += 1
            continue

        report = build_article_quality_report(
            title=title,
            html_content=html_article,
            plain_text=article_text,
            citations=citations,
            sections=[],
            evidence_count=len(citations),  # Backfill approximation
        )
        reports.append(report)

        if args.apply:
            update_payload = {
                "overall_quality_score": int(round(_safe_float(report.get("overall_score", 0.0)))),
                "quality_report": report,
            }
            try:
                supabase.table("Titles").update(update_payload).eq("id", title_id).execute()
                updated += 1
            except Exception as update_error:
                # Fallback for environments missing the new JSONB column.
                if "quality_report" in str(update_error):
                    supabase.table("Titles").update(
                        {"overall_quality_score": int(round(_safe_float(report.get("overall_score", 0.0))))}
                    ).eq("id", title_id).execute()
                    updated += 1
                else:
                    raise

    _print_summary(reports, skipped)

    if args.apply:
        print(f"Persisted updates: {updated}")
    else:
        print("Dry-run complete. Use --apply to persist scores and reports.")


if __name__ == "__main__":
    main()
