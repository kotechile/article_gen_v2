#!/usr/bin/env python3
"""
Generate a before/after quality snapshot for Phase 12.

This report compares existing topic angle metadata vs derived fallback metadata
for a sample of topics. It does not mutate data.

Usage:
  python3 scripts/generate_phase12_before_after_report.py --sample 5
  python3 scripts/generate_phase12_before_after_report.py --sample 5 --output PHASE12_SAMPLE_REPORT.md
"""

import argparse
import os
from datetime import datetime, timezone

from supabase import create_client

from backfill_angle_metadata import _derive_angle_metadata


def _safe(v):
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v)


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 12 before/after sample report")
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--output", default="PHASE12_SAMPLE_REPORT.md")
    args = parser.parse_args()

    sb_url = os.getenv("SUPABASE_URL")
    sb_key = os.getenv("SUPABASE_SERVICE_KEY")
    if not sb_url or not sb_key:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")

    supabase = create_client(sb_url, sb_key)

    topics = (
        supabase.table("research_topics")
        .select(
            "id,title,description,project_id,primary_category_id,secondary_category_id,"
            "intent_bucket,decision_focus,angle_question,value_layer_tags,target_audience,evidence_sources,related_terms"
        )
        .order("created_at", desc=True)
        .limit(max(args.sample * 4, 20))
        .execute()
        .data
        or []
    )
    if not topics:
        raise SystemExit("No topics found")

    project_ids = sorted({t.get("project_id") for t in topics if t.get("project_id")})
    category_ids = sorted(
        {
            cid
            for t in topics
            for cid in [t.get("primary_category_id"), t.get("secondary_category_id")]
            if cid
        }
    )

    projects = (
        supabase.table("projects")
        .select("id,description,domain,app_name")
        .in_("id", project_ids)
        .execute()
        .data
        if project_ids
        else []
    ) or []
    categories = (
        supabase.table("project_categories")
        .select("id,name")
        .in_("id", category_ids)
        .execute()
        .data
        if category_ids
        else []
    ) or []

    projects_by_id = {p["id"]: p for p in projects}
    categories_by_id = {c["id"]: c.get("name") for c in categories}

    rows = []
    for topic in topics:
        primary_name = categories_by_id.get(topic.get("primary_category_id"))
        secondary_name = categories_by_id.get(topic.get("secondary_category_id"))
        project = projects_by_id.get(topic.get("project_id")) or {}
        derived = _derive_angle_metadata(
            title=topic.get("title") or "",
            description=topic.get("description"),
            primary_category_name=primary_name,
            secondary_category_name=secondary_name,
            project_description=project.get("description"),
        )
        rows.append((topic, derived, project, primary_name, secondary_name))

    sample_rows = rows[: args.sample]

    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Phase 12 Sample Report (Before vs Derived)",
        "",
        f"Generated at: `{now}`",
        f"Sample size: `{len(sample_rows)}`",
        "",
    ]

    for idx, (topic, derived, project, primary_name, secondary_name) in enumerate(sample_rows, start=1):
        category_path = " / ".join([p for p in [primary_name, secondary_name] if p]) or "N/A"
        project_name = project.get("domain") or project.get("app_name") or "N/A"
        lines.extend(
            [
                f"## {idx}. {topic.get('title','(untitled)')}",
                "",
                f"- Topic ID: `{topic.get('id')}`",
                f"- Project: `{project_name}`",
                f"- Category: `{category_path}`",
                "",
                "| Field | Current | Derived |",
                "|---|---|---|",
                f"| intent_bucket | {_safe(topic.get('intent_bucket'))} | {_safe(derived.get('intent_bucket'))} |",
                f"| decision_focus | {_safe(topic.get('decision_focus'))} | {_safe(derived.get('decision_focus'))} |",
                f"| angle_question | {_safe(topic.get('angle_question'))} | {_safe(derived.get('angle_question'))} |",
                f"| value_layer_tags | {_safe(topic.get('value_layer_tags'))} | {_safe(derived.get('value_layer_tags'))} |",
                f"| target_audience | {_safe(topic.get('target_audience'))} | {_safe(derived.get('target_audience'))} |",
                f"| evidence_sources | {_safe(topic.get('evidence_sources'))} | {_safe(derived.get('evidence_sources'))} |",
                f"| related_terms | {_safe(topic.get('related_terms'))} | {_safe(derived.get('related_terms'))} |",
                "",
            ]
        )

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
