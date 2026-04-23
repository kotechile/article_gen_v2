import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path when running tests without an installed package.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tasks  # noqa: E402


def test_keyword_intelligence_blocks_when_titles_candidates_missing(monkeypatch):
    monkeypatch.setattr(
        tasks,
        "_load_keyword_gate_settings",
        lambda: {
            "strict_keyword_mode": True,
            "strict_titles_authority": True,
            "min_keyword_candidates": 1,
            "allow_llm_fallback": False,
        },
    )

    result = {
        "research_data": {
            "keywords": "brief keyword, another brief keyword",
            "keyword_candidates": [],
        }
    }

    with pytest.raises(ValueError, match="Titles keyword gate blocked generation"):
        tasks._run_keyword_intelligence(result)


def test_keyword_intelligence_uses_titles_candidates_only_in_strict_mode(monkeypatch):
    monkeypatch.setattr(
        tasks,
        "_load_keyword_gate_settings",
        lambda: {
            "strict_keyword_mode": True,
            "strict_titles_authority": True,
            "min_keyword_candidates": 1,
            "allow_llm_fallback": False,
        },
    )

    result = {
        "research_data": {
            "keywords": "brief-only keyword, another brief-only keyword",
            "keyword_candidates": ["titles-owned keyword"],
            "keyword_research_source": "hybrid",
            "keyword_research_status": "ready",
            "target_intent": "informational",
        }
    }

    updated = tasks._run_keyword_intelligence(result)
    research_data = updated["research_data"]

    assert research_data["primary_keyword"] == "titles-owned keyword"
    assert "brief-only keyword" not in research_data["keyword_candidates"]
    assert research_data["keyword_candidates"] == ["titles-owned keyword"]
