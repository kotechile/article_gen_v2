#!/usr/bin/env python3
"""
Backfill missing angle metadata on existing research topics.

Usage:
  python3 scripts/backfill_angle_metadata.py --dry-run
  python3 scripts/backfill_angle_metadata.py --apply --limit 200

Requires:
  SUPABASE_URL
  SUPABASE_SERVICE_KEY
"""

import argparse
import os
import re
from typing import Dict, List, Optional

from supabase import create_client


def _safe_string(value):
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = " ".join(value.split()).strip()
        return cleaned or None
    return str(value)


def _derive_intent_bucket(title: str, category_path: str) -> str:
    text = f"{title} {category_path}".lower()
    if any(term in text for term in ["vs", "compare", "comparison", "best", "top", "alternative"]):
        return "commercial_evaluation"
    if any(term in text for term in ["calculator", "tool", "template", "checklist", "framework"]):
        return "solution_enablement"
    if any(term in text for term in ["cost", "roi", "value", "profit", "returns", "pricing"]):
        return "decision_financial"
    return "informational_decision"


def _derive_value_layer_tags(title: str, category_path: str) -> List[str]:
    text = f"{title} {category_path}".lower()
    tags: List[str] = []
    if any(term in text for term in ["roi", "return", "profit", "resale", "yield"]):
        tags.append("roi-focused")
    if any(term in text for term in ["cost", "price", "expense", "budget", "hidden cost"]):
        tags.append("cost-vs-value")
    if any(term in text for term in ["timing", "when to", "cycle", "market timing"]):
        tags.append("timing-decision")
    if any(term in text for term in ["location", "city", "state", "geo", "geographic"]):
        tags.append("location-decision")
    if any(term in text for term in ["audit", "scorecard", "framework", "evaluation"]):
        tags.append("hidden-cost-audit")
    if any(term in text for term in ["tool", "calculator", "dashboard", "app", "automation"]):
        tags.append("tool-builder")
    if not tags:
        tags.append("decision-support")
    return tags[:4]


def _derive_angle_metadata(
    title: str,
    description: Optional[str],
    primary_category_name: Optional[str],
    secondary_category_name: Optional[str],
    project_description: Optional[str],
) -> Dict:
    title_clean = _safe_string(title) or "Untitled topic"
    description_clean = _safe_string(description)
    category_parts = [p for p in [primary_category_name, secondary_category_name] if p]
    category_path = " / ".join(category_parts)

    intent_bucket = _derive_intent_bucket(title_clean, category_path)
    decision_focus = (
        description_clean
        or f"Help users evaluate options and make a better decision about {title_clean}."
    )
    angle_question = f"How should someone evaluate {title_clean} and decide the best next action?"

    target_audience = None
    if project_description:
        lowered = project_description.lower()
        if any(term in lowered for term in ["investor", "investing", "portfolio", "capital"]):
            target_audience = "investors and operators"
        elif any(term in lowered for term in ["homeowner", "home", "property owner"]):
            target_audience = "homeowners and property buyers"

    related_terms = []
    for token in re.split(r"[^a-zA-Z0-9]+", title_clean.lower()):
        if token and len(token) > 3 and token not in related_terms:
            related_terms.append(token)
    if secondary_category_name:
        related_terms.append(secondary_category_name.lower())

    metadata = {
        "intent_bucket": intent_bucket,
        "decision_focus": decision_focus,
        "angle_question": angle_question,
        "value_layer_tags": _derive_value_layer_tags(title_clean, category_path),
        "target_audience": target_audience,
        "related_terms": related_terms[:8],
    }
    if category_path:
        metadata["evidence_sources"] = [f"category:{category_path}"]

    return {key: value for key, value in metadata.items() if value not in [None, "", []]}


def _is_missing_angle_metadata(topic: Dict) -> bool:
    tags = topic.get("value_layer_tags") or []
    return not (
        topic.get("intent_bucket")
        and topic.get("decision_focus")
        and topic.get("angle_question")
        and isinstance(tags, list)
        and len(tags) > 0
    )


def main():
    parser = argparse.ArgumentParser(description="Backfill angle metadata for research_topics")
    parser.add_argument("--apply", action="store_true", help="Persist updates to Supabase")
    parser.add_argument("--dry-run", action="store_true", help="Preview only (default if --apply not set)")
    parser.add_argument("--limit", type=int, default=1000, help="Max number of topics to scan")
    args = parser.parse_args()

    sb_url = os.getenv("SUPABASE_URL")
    sb_key = os.getenv("SUPABASE_SERVICE_KEY")
    if not sb_url or not sb_key:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")

    supabase = create_client(sb_url, sb_key)
    response = (
        supabase.table("research_topics")
        .select(
            "id,title,description,project_id,primary_category_id,secondary_category_id,"
            "intent_bucket,decision_focus,angle_question,value_layer_tags,target_audience,evidence_sources,related_terms"
        )
        .limit(args.limit)
        .execute()
    )
    topics = response.data or []
    if not topics:
        print("No research topics found.")
        return

    project_ids = sorted({t.get("project_id") for t in topics if t.get("project_id")})
    category_ids = sorted(
        {
            cid
            for t in topics
            for cid in [t.get("primary_category_id"), t.get("secondary_category_id")]
            if cid
        }
    )

    projects_by_id = {}
    categories_by_id = {}
    if project_ids:
        projects = (
            supabase.table("projects")
            .select("id,domain,app_name,site_description,websitedescription,targetaudiencedescription")
            .in_("id", project_ids)
            .execute()
            .data
            or []
        )
        projects_by_id = {p["id"]: p for p in projects}
    if category_ids:
        categories = (
            supabase.table("project_categories")
            .select("id,name")
            .in_("id", category_ids)
            .execute()
            .data
            or []
        )
        categories_by_id = {c["id"]: c.get("name") for c in categories}

    updated = 0
    skipped = 0
    for topic in topics:
        if not _is_missing_angle_metadata(topic):
            skipped += 1
            continue

        primary_name = categories_by_id.get(topic.get("primary_category_id"))
        secondary_name = categories_by_id.get(topic.get("secondary_category_id"))
        project_record = projects_by_id.get(topic.get("project_id")) or {}
        project_description = (
            project_record.get("site_description")
            or project_record.get("websitedescription")
            or project_record.get("targetaudiencedescription")
        )
        derived = _derive_angle_metadata(
            title=topic.get("title") or "",
            description=topic.get("description"),
            primary_category_name=primary_name,
            secondary_category_name=secondary_name,
            project_description=project_description,
        )

        patch = {}
        for key in [
            "intent_bucket",
            "decision_focus",
            "angle_question",
            "value_layer_tags",
            "target_audience",
            "evidence_sources",
            "related_terms",
        ]:
            current = topic.get(key)
            if current in [None, "", []]:
                if key in derived:
                    patch[key] = derived[key]

        if not patch:
            skipped += 1
            continue

        if args.apply:
            supabase.table("research_topics").update(patch).eq("id", topic["id"]).execute()
        updated += 1
        print(f"{'APPLY' if args.apply else 'DRY'} topic={topic['id']} title={topic.get('title','')[:70]!r} patch_keys={list(patch.keys())}")

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{mode} complete. scanned={len(topics)} updated={updated} skipped={skipped}")


if __name__ == "__main__":
    main()
