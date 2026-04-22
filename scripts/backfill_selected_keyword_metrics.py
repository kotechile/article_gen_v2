#!/usr/bin/env python3
"""
Backfill structured selected keyword metrics for existing Titles rows.

Usage:
  python3 scripts/backfill_selected_keyword_metrics.py --dry-run
  python3 scripts/backfill_selected_keyword_metrics.py --apply --limit 200
  python3 scripts/backfill_selected_keyword_metrics.py --apply --status Written

Behavior:
- Prefers exact per-keyword DataForSEO metrics from linked content_ideas.keyword_metrics
- Falls back to estimated aggregate carryover when exact metrics are unavailable
- Updates Titles.selected_keyword_metrics_json and aligns selected volume/difficulty
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from supabase_client import get_supabase_client  # noqa: E402


def _normalize_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return []


def _normalize_keyword_metric_map(keyword_metrics_map: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in (keyword_metrics_map or {}).items():
        keyword = str(key or "").strip().lower()
        if not keyword:
            continue
        normalized[keyword] = value or {}
    return normalized


def _build_exact_payload(
    primary_keyword: str,
    secondary_keywords: List[str],
    keyword_metrics_map: Dict[str, Any],
    target_intent: str,
    keyword_source: str,
) -> Dict[str, Any]:
    normalized_map = _normalize_keyword_metric_map(keyword_metrics_map)

    def _row(keyword: str) -> Dict[str, Any]:
        metric = normalized_map.get(str(keyword or "").strip().lower(), {})
        return {
            "keyword": str(keyword or "").strip(),
            "search_volume": int(metric.get("search_volume") or 0),
            "keyword_difficulty": float(metric.get("keyword_difficulty") or 0.0),
            "cpc": float(metric.get("cpc") or 0.0),
            "metric_source": "research_keyword_dossier",
            "is_estimated": False,
        }

    return {
        "primary": {
            **_row(primary_keyword),
            "intent": str(target_intent or "").strip().lower() or "informational",
        },
        "secondary": [_row(keyword) for keyword in secondary_keywords if str(keyword).strip()],
        "candidate_count": len(([primary_keyword] if primary_keyword else []) + [k for k in secondary_keywords if str(k).strip()]),
        "source": str(keyword_source or "").strip().lower() or "dataforseo",
    }


def _build_estimated_payload(
    primary_keyword: str,
    secondary_keywords: List[str],
    search_volume: Any,
    difficulty: Any,
    target_intent: str,
    keyword_source: str,
    candidate_keywords: List[str],
) -> Dict[str, Any]:
    try:
        search_volume_value = int(search_volume or 0)
    except Exception:
        search_volume_value = 0
    try:
        difficulty_value = float(difficulty or 0.0)
    except Exception:
        difficulty_value = 0.0

    return {
        "primary": {
            "keyword": str(primary_keyword or "").strip(),
            "search_volume": search_volume_value,
            "keyword_difficulty": difficulty_value,
            "intent": str(target_intent or "").strip().lower() or "informational",
            "metric_source": "aggregate_idea_metrics",
            "is_estimated": True,
        },
        "secondary": [
            {
                "keyword": keyword,
                "metric_source": "unscored_secondary_keyword",
                "is_estimated": True,
            }
            for keyword in secondary_keywords
            if str(keyword).strip()
        ],
        "candidate_count": len(candidate_keywords) or len(([primary_keyword] if primary_keyword else []) + [k for k in secondary_keywords if str(k).strip()]),
        "source": str(keyword_source or "").strip().lower() or "unknown",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Titles.selected_keyword_metrics_json")
    parser.add_argument("--apply", action="store_true", help="Persist updates to Supabase")
    parser.add_argument("--dry-run", action="store_true", help="Preview only (default if --apply not set)")
    parser.add_argument("--limit", type=int, default=200, help="Maximum Titles rows to scan")
    parser.add_argument("--status", type=str, default="", help="Optional Titles.status filter")
    parser.add_argument("--title-id", type=str, default="", help="Backfill a single Titles row by id")
    args = parser.parse_args()

    supabase = get_supabase_client()
    if not supabase:
        raise SystemExit("Supabase client unavailable. Check SUPABASE_URL and service key env vars.")

    query = (
        supabase.table("Titles")
        .select(
            "id,Title,status,source_idea_id,primary_keyword,secondary_keywords_json,"
            "keyword_candidates_json,selected_keyword_metrics_json,selected_keyword_search_volume,"
            "selected_keyword_difficulty,selected_keyword_intent,keyword_research_source,"
            "total_search_volume,avg_keyword_difficulty"
        )
        .limit(args.limit)
    )
    if args.status:
        query = query.eq("status", args.status)
    if args.title_id:
        query = query.eq("id", args.title_id)

    titles = query.execute().data or []
    if not titles:
        print("No matching Titles rows found.")
        return

    source_idea_ids = sorted({row.get("source_idea_id") for row in titles if row.get("source_idea_id")})
    ideas_by_id: Dict[str, Dict[str, Any]] = {}
    if source_idea_ids:
        idea_rows = []
        try:
            idea_rows = (
                supabase.table("content_ideas")
                .select("id,primary_keywords,secondary_keywords,keywords,keyword_metrics,target_intent,total_search_volume,average_difficulty,idea_metadata")
                .in_("id", source_idea_ids)
                .execute()
                .data
                or []
            )
        except Exception:
            # Backward-compatible fallback for deployments with older/different content_ideas schemas.
            idea_rows = (
                supabase.table("content_ideas")
                .select("*")
                .in_("id", source_idea_ids)
                .execute()
                .data
                or []
            )
        ideas_by_id = {row["id"]: row for row in idea_rows if row.get("id")}

    updated = 0
    skipped = 0
    for row in titles:
        title_id = row["id"]
        primary_keyword = str(row.get("primary_keyword") or "").strip()
        secondary_keywords = _normalize_list(row.get("secondary_keywords_json"))
        candidate_keywords = _normalize_list(row.get("keyword_candidates_json"))
        target_intent = str(row.get("selected_keyword_intent") or "").strip().lower() or "informational"
        keyword_source = str(row.get("keyword_research_source") or "").strip().lower() or "unknown"

        source_idea = ideas_by_id.get(row.get("source_idea_id")) or {}
        keyword_metrics_map = source_idea.get("keyword_metrics") or {}
        if not keyword_metrics_map:
            idea_enrichment = (source_idea.get("idea_metadata") or {}).get("seo_offer_enrichment") or {}
            keyword_metrics_map = idea_enrichment.get("keyword_metrics") or {}

        if not primary_keyword:
            primary_candidates = _normalize_list(source_idea.get("primary_keywords") or source_idea.get("keywords"))
            if primary_candidates:
                primary_keyword = primary_candidates[0]
        if not secondary_keywords:
            secondary_keywords = _normalize_list(source_idea.get("secondary_keywords"))
        if not secondary_keywords and len(_normalize_list(source_idea.get("keywords"))) > 1:
            secondary_keywords = _normalize_list(source_idea.get("keywords"))[1:]

        if keyword_metrics_map and primary_keyword:
            payload = _build_exact_payload(
                primary_keyword=primary_keyword,
                secondary_keywords=secondary_keywords,
                keyword_metrics_map=keyword_metrics_map,
                target_intent=target_intent or source_idea.get("target_intent") or "informational",
                keyword_source=keyword_source or "dataforseo",
            )
        else:
            payload = _build_estimated_payload(
                primary_keyword=primary_keyword,
                secondary_keywords=secondary_keywords,
                search_volume=row.get("selected_keyword_search_volume") or row.get("total_search_volume") or source_idea.get("total_search_volume"),
                difficulty=row.get("selected_keyword_difficulty") or row.get("avg_keyword_difficulty") or source_idea.get("average_difficulty"),
                target_intent=target_intent or source_idea.get("target_intent") or "informational",
                keyword_source=keyword_source,
                candidate_keywords=candidate_keywords,
            )

        existing_payload = row.get("selected_keyword_metrics_json") or {}
        existing_primary = (existing_payload.get("primary") or {}) if isinstance(existing_payload, dict) else {}
        existing_keyword = str(existing_primary.get("keyword") or "").strip()
        existing_metric_source = str(existing_primary.get("metric_source") or "").strip()
        new_primary = payload.get("primary") or {}

        # Skip if row already has a structured payload for the same primary keyword and same source quality.
        if existing_keyword and existing_keyword == str(new_primary.get("keyword") or "").strip():
            if existing_metric_source == str(new_primary.get("metric_source") or "").strip():
                skipped += 1
                continue

        update_payload = {
            "primary_keyword": primary_keyword,
            "secondary_keywords_json": secondary_keywords,
            "selected_keyword_metrics_json": payload,
            "selected_keyword_search_volume": int(new_primary.get("search_volume") or 0),
            "selected_keyword_difficulty": float(new_primary.get("keyword_difficulty") or 0.0),
            "selected_keyword_intent": str(new_primary.get("intent") or target_intent or "informational"),
        }

        if args.apply:
            supabase.table("Titles").update(update_payload).eq("id", title_id).execute()
        updated += 1
        mode = "APPLY" if args.apply else "DRY"
        print(
            f"{mode} title={title_id} primary={primary_keyword!r} "
            f"source={payload.get('source')} metric_source={new_primary.get('metric_source')} "
            f"volume={new_primary.get('search_volume')} kd={new_primary.get('keyword_difficulty')}"
        )

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{mode} complete. scanned={len(titles)} updated={updated} skipped={skipped}")


if __name__ == "__main__":
    main()
