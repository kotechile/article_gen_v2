#!/usr/bin/env python3
"""
Phase 8 benchmark runner for pipeline comparison.

Compares two datasets (baseline vs candidate) and emits:
- JSON snapshot with aggregate scores and deltas
- Markdown summary report for review workflows

Input modes:
1) Files:
   --baseline-file path/to/baseline.json --candidate-file path/to/candidate.json
2) Supabase Titles pull:
   --baseline-status Created --candidate-status "Needs Review" --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from supabase_client import get_supabase_client  # noqa: E402
from src.services.article_quality_evaluator import build_article_quality_report  # noqa: E402


@dataclass
class EvalRow:
    id: str
    title: str
    status: str
    html: str
    text: str
    citations: List[Dict[str, Any]]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


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


def _extract_row(raw: Dict[str, Any], idx: int) -> EvalRow:
    title = str(raw.get("Title") or raw.get("title") or f"Untitled {idx}")
    html = str(
        raw.get("htmlArticle")
        or raw.get("html_content")
        or raw.get("html")
        or raw.get("content")
        or ""
    )
    text = str(raw.get("articleText") or raw.get("plain_text") or raw.get("text") or "")
    citations = _parse_citations(raw.get("citations"))
    return EvalRow(
        id=str(raw.get("id") or raw.get("uuid") or f"row_{idx}"),
        title=title,
        status=str(raw.get("status") or ""),
        html=html,
        text=text,
        citations=citations,
    )


def _load_rows_from_file(path: str) -> List[EvalRow]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        data = payload.get("rows") or payload.get("articles") or payload.get("data") or []
    elif isinstance(payload, list):
        data = payload
    else:
        data = []
    return [_extract_row(row, idx + 1) for idx, row in enumerate(data)]


def _load_rows_from_supabase(
    status: str,
    limit: int,
    from_ts: str = "",
    to_ts: str = "",
) -> List[EvalRow]:
    supabase = get_supabase_client()
    if not supabase:
        raise RuntimeError("Supabase client unavailable")
    query = (
        supabase.table("Titles")
        .select("id,status,Title,articleText,htmlArticle,citations,created_at")
        .limit(limit)
    )
    if status:
        query = query.eq("status", status)
    if from_ts:
        query = query.gte("created_at", from_ts)
    if to_ts:
        query = query.lt("created_at", to_ts)
    response = query.execute()
    return [_extract_row(row, idx + 1) for idx, row in enumerate(response.data or [])]


def _evaluate_rows(rows: List[EvalRow]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    evaluated: List[Dict[str, Any]] = []
    skipped = 0
    for row in rows:
        if not row.html and not row.text:
            skipped += 1
            continue
        report = build_article_quality_report(
            title=row.title,
            html_content=row.html,
            plain_text=row.text,
            citations=row.citations,
            sections=[],
            evidence_count=len(row.citations),
        )
        evaluated.append(
            {
                "id": row.id,
                "title": row.title,
                "status": row.status,
                "overall_score": _safe_float(report.get("overall_score")),
                "humanization_score": _safe_float(report.get("humanization_score")),
                "grounding_score": _safe_float(report.get("grounding_score")),
                "geo_score": _safe_float(report.get("geo_score")),
                "warnings": report.get("warnings") or [],
            }
        )

    def _avg(key: str) -> float:
        return round(mean([_safe_float(r.get(key)) for r in evaluated]), 2) if evaluated else 0.0

    aggregates = {
        "evaluated_count": len(evaluated),
        "skipped_count": skipped,
        "avg_overall_score": _avg("overall_score"),
        "avg_humanization_score": _avg("humanization_score"),
        "avg_grounding_score": _avg("grounding_score"),
        "avg_geo_score": _avg("geo_score"),
        "below_60_overall_count": len([r for r in evaluated if _safe_float(r.get("overall_score")) < 60]),
    }
    return evaluated, aggregates


def _build_snapshot(
    baseline_name: str,
    candidate_name: str,
    baseline_agg: Dict[str, Any],
    candidate_agg: Dict[str, Any],
) -> Dict[str, Any]:
    delta = {
        "overall_score_delta": round(
            _safe_float(candidate_agg.get("avg_overall_score")) - _safe_float(baseline_agg.get("avg_overall_score")),
            2,
        ),
        "humanization_score_delta": round(
            _safe_float(candidate_agg.get("avg_humanization_score")) - _safe_float(baseline_agg.get("avg_humanization_score")),
            2,
        ),
        "grounding_score_delta": round(
            _safe_float(candidate_agg.get("avg_grounding_score")) - _safe_float(baseline_agg.get("avg_grounding_score")),
            2,
        ),
        "geo_score_delta": round(
            _safe_float(candidate_agg.get("avg_geo_score")) - _safe_float(baseline_agg.get("avg_geo_score")),
            2,
        ),
        "below_60_overall_delta": int(candidate_agg.get("below_60_overall_count", 0)) - int(
            baseline_agg.get("below_60_overall_count", 0)
        ),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": {"name": baseline_name, "aggregates": baseline_agg},
        "candidate": {"name": candidate_name, "aggregates": candidate_agg},
        "delta": delta,
    }


def _snapshot_markdown(snapshot: Dict[str, Any]) -> str:
    b = snapshot["baseline"]["aggregates"]
    c = snapshot["candidate"]["aggregates"]
    d = snapshot["delta"]
    lines = [
        "# Phase 8 Benchmark Snapshot",
        "",
        f"Generated at: `{snapshot['generated_at']}`",
        "",
        "## Aggregate Comparison",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|",
        f"| Avg overall score | {b.get('avg_overall_score',0)} | {c.get('avg_overall_score',0)} | {d.get('overall_score_delta',0)} |",
        f"| Avg humanization score | {b.get('avg_humanization_score',0)} | {c.get('avg_humanization_score',0)} | {d.get('humanization_score_delta',0)} |",
        f"| Avg grounding score | {b.get('avg_grounding_score',0)} | {c.get('avg_grounding_score',0)} | {d.get('grounding_score_delta',0)} |",
        f"| Avg GEO score | {b.get('avg_geo_score',0)} | {c.get('avg_geo_score',0)} | {d.get('geo_score_delta',0)} |",
        f"| Rows below 60 overall | {b.get('below_60_overall_count',0)} | {c.get('below_60_overall_count',0)} | {d.get('below_60_overall_delta',0)} |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 8 benchmark comparison")
    parser.add_argument("--baseline-file", type=str, default="", help="JSON dataset for baseline")
    parser.add_argument("--candidate-file", type=str, default="", help="JSON dataset for candidate")
    parser.add_argument("--baseline-status", type=str, default="", help="Pull baseline from Titles status")
    parser.add_argument("--candidate-status", type=str, default="", help="Pull candidate from Titles status")
    parser.add_argument("--baseline-from", type=str, default="", help="Baseline created_at lower bound (inclusive)")
    parser.add_argument("--baseline-to", type=str, default="", help="Baseline created_at upper bound (exclusive)")
    parser.add_argument("--candidate-from", type=str, default="", help="Candidate created_at lower bound (inclusive)")
    parser.add_argument("--candidate-to", type=str, default="", help="Candidate created_at upper bound (exclusive)")
    parser.add_argument("--limit", type=int, default=50, help="Rows per dataset when using status mode")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, ".benchmarks", "phase8", "snapshots"),
        help="Snapshot output directory",
    )
    args = parser.parse_args()

    use_file_mode = bool(args.baseline_file and args.candidate_file)
    use_status_mode = bool(args.baseline_status and args.candidate_status)
    if not use_file_mode and not use_status_mode:
        raise SystemExit("Provide either --baseline-file/--candidate-file or --baseline-status/--candidate-status")

    if use_file_mode:
        baseline_rows = _load_rows_from_file(args.baseline_file)
        candidate_rows = _load_rows_from_file(args.candidate_file)
        baseline_name = os.path.basename(args.baseline_file)
        candidate_name = os.path.basename(args.candidate_file)
    else:
        baseline_rows = _load_rows_from_supabase(
            status=args.baseline_status,
            limit=args.limit,
            from_ts=args.baseline_from,
            to_ts=args.baseline_to,
        )
        candidate_rows = _load_rows_from_supabase(
            status=args.candidate_status,
            limit=args.limit,
            from_ts=args.candidate_from,
            to_ts=args.candidate_to,
        )
        baseline_name = (
            f"Titles.status={args.baseline_status or '*'}"
            f"|{args.baseline_from or '-inf'}..{args.baseline_to or '+inf'}"
        )
        candidate_name = (
            f"Titles.status={args.candidate_status or '*'}"
            f"|{args.candidate_from or '-inf'}..{args.candidate_to or '+inf'}"
        )

    baseline_eval, baseline_agg = _evaluate_rows(baseline_rows)
    candidate_eval, candidate_agg = _evaluate_rows(candidate_rows)
    if baseline_agg.get("evaluated_count", 0) == 0 or candidate_agg.get("evaluated_count", 0) == 0:
        print("WARNING: One or both cohorts have zero evaluatable rows. Deltas may be meaningless.")
    if baseline_agg.get("evaluated_count", 0) < 10 or candidate_agg.get("evaluated_count", 0) < 10:
        print("WARNING: Small cohort size (<10) detected. Treat comparison as directional only.")
    snapshot = _build_snapshot(baseline_name, candidate_name, baseline_agg, candidate_agg)
    snapshot["baseline"]["samples"] = baseline_eval[:10]
    snapshot["candidate"]["samples"] = candidate_eval[:10]

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(args.output_dir, f"phase8_snapshot_{ts}.json")
    md_path = os.path.join(args.output_dir, f"phase8_snapshot_{ts}.md")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(snapshot, jf, indent=2)
    with open(md_path, "w", encoding="utf-8") as mf:
        mf.write(_snapshot_markdown(snapshot))

    print(f"Snapshot JSON: {json_path}")
    print(f"Snapshot MD:   {md_path}")
    print(json.dumps(snapshot["delta"], indent=2))


if __name__ == "__main__":
    main()
