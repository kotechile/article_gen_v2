import sys
from pathlib import Path

import pytest

# Ensure repo root is importable when tests run without an installed package.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("celery")
pytest.importorskip("kombu")

import tasks  # noqa: E402

pytestmark = pytest.mark.skip(reason="Dossier system deprecated and removed")


def _valid_dossier():
    return {
        "summary": (
            "This dossier summarizes mortgage prepayment trade-offs, liquidity constraints, opportunity cost, "
            "tax context, rate sensitivity, refinancing alternatives, and portfolio considerations in enough "
            "detail to support grounded article planning. It includes multiple sourced claims and enough context "
            "to satisfy the minimum summary threshold for dossier-backed generation in production."
        ),
        "primary_claims": [
            {"claim": "Liquidity should be preserved before aggressive principal paydown."},
            {"claim": "Expected after-tax returns should be compared against mortgage APR."},
        ],
        "source_quality_summary": {"source_count": 2},
        "dossier_quality_score": 45,
    }


def _valid_citations():
    return [
        {
            "title": "Federal Reserve data",
            "url": "https://example.gov/fed",
            "content": "Fed data on rates and mortgage conditions.",
        },
        {
            "title": "Market outlook",
            "url": "https://example.org/outlook",
            "content": "Institutional market outlook for 2026.",
        },
    ]


def test_dossier_preflight_reuses_valid_stored_dossier_and_citations(monkeypatch):
    refresh_calls = []
    monkeypatch.setattr(
        tasks,
        "_refresh_dossier_via_deep_research",
        lambda **kwargs: refresh_calls.append(kwargs) or {},
    )

    research_data = {"source_strategy": "dossier_plus_rag"}
    title_row = {
        "research_dossier": _valid_dossier(),
        "dossier_status": "ready",
        "dossier_quality_score": 45,
        "citations": _valid_citations(),
    }

    tasks._ensure_dossier_prerequisites(
        article_id="title-1",
        title_row=title_row,
        research_data=research_data,
        supabase=None,
        use_dossier_context=True,
    )

    assert refresh_calls == []
    assert research_data["dossier_validated"] is True
    assert len(research_data["prior_citations"]) == 2
    assert research_data["research_dossier"]["summary"].startswith("This dossier summarizes")


def test_dossier_preflight_refreshes_when_dossier_missing(monkeypatch):
    refresh_calls = []

    def _refresh(**kwargs):
        refresh_calls.append(kwargs["reason"])
        return {
            "research_dossier": _valid_dossier(),
            "dossier_status": "ready",
            "dossier_quality_score": 45,
            "prior_citations": _valid_citations(),
        }

    monkeypatch.setattr(tasks, "_refresh_dossier_via_deep_research", _refresh)

    research_data = {"source_strategy": "dossier_only"}
    title_row = {
        "research_dossier": None,
        "dossier_status": None,
        "dossier_quality_score": 0,
        "citations": [],
    }

    tasks._ensure_dossier_prerequisites(
        article_id="title-2",
        title_row=title_row,
        research_data=research_data,
        supabase=None,
        use_dossier_context=True,
    )

    assert refresh_calls == ["missing_or_invalid_dossier"]
    assert research_data["dossier_validated"] is True
    assert len(research_data["prior_citations"]) == 2


def test_dossier_preflight_refreshes_when_citations_missing(monkeypatch):
    refresh_calls = []

    def _refresh(**kwargs):
        refresh_calls.append(kwargs["reason"])
        return {
            "research_dossier": _valid_dossier(),
            "dossier_status": "ready",
            "dossier_quality_score": 45,
            "prior_citations": _valid_citations(),
        }

    monkeypatch.setattr(tasks, "_refresh_dossier_via_deep_research", _refresh)

    research_data = {"source_strategy": "dossier_plus_rag"}
    title_row = {
        "research_dossier": _valid_dossier(),
        "dossier_status": "ready",
        "dossier_quality_score": 45,
        "citations": [],
    }

    tasks._ensure_dossier_prerequisites(
        article_id="title-3",
        title_row=title_row,
        research_data=research_data,
        supabase=None,
        use_dossier_context=True,
    )

    assert refresh_calls == ["missing_prior_citations"]
    assert research_data["dossier_validated"] is True
    assert len(research_data["prior_citations"]) == 2
