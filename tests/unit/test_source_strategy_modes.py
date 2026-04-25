import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure repo root is importable when tests run without an installed package.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("celery")
pytest.importorskip("kombu")

import tasks  # noqa: E402


class _Counters:
    def __init__(self):
        self.rag_calls = 0
        self.linkup_calls = 0
        self.rag_results = [
            SimpleNamespace(
                source="rag://doc1",
                content="RAG evidence content",
                metadata={"title": "Doc 1"},
                relevance_score=0.9,
                credibility_score=0.8,
                similarity_score=0.85,
            )
        ]


def _patch_dependencies(monkeypatch, counters: _Counters):
    monkeypatch.setenv("SOURCE_STRATEGY_REFACTOR_ENABLED", "true")

    # Keep provider routing deterministic for tests.
    monkeypatch.setattr(tasks, "get_linkup_api_key", lambda: "test-linkup-key")
    monkeypatch.setattr(tasks, "_assess_rag_coverage", lambda *args, **kwargs: {
        "sufficient": False,
        "source_count": 0,
        "avg_relevance": 0.0,
        "keyword_coverage": 0.0,
        "assessment": "test",
    })

    optimization = SimpleNamespace(
        rag_coverage_min_sources=3,
        rag_coverage_min_relevance=0.6,
        cache_enabled=False,
        deep_trigger_min_sources=2,
        deep_trigger_min_avg_relevance=0.45,
        deep_trigger_min_keyword_coverage=0.3,
        deep_min_standard_results_threshold=3,
    )
    config = SimpleNamespace(
        RAG_API_URL=None,
        RAG_API_KEY=None,
        linkup_optimization=optimization,
    )
    monkeypatch.setattr(tasks, "get_config", lambda: config)

    class DummyRagClient:
        def query(self, _query):
            counters.rag_calls += 1
            return SimpleNamespace(
                success=True,
                error=None,
                results=counters.rag_results,
            )

    class DummyLinkupClient:
        def search(self, _query):
            counters.linkup_calls += 1
            return SimpleNamespace(
                success=True,
                error=None,
                results=[
                    SimpleNamespace(
                        url="https://example.com/news",
                        content="Web evidence content",
                        snippet="Web evidence snippet",
                        relevance_score=0.8,
                        credibility_score=0.7,
                        metadata={},
                    )
                ],
            )

    monkeypatch.setattr(tasks, "create_rag_client", lambda **kwargs: DummyRagClient())
    monkeypatch.setattr(tasks, "create_linkup_client", lambda **kwargs: DummyLinkupClient())


def _base_result(strategy: str):
    return {
        "research_data": {
            "source_strategy": strategy,
            "brief": "Mortgage prepayment strategies in 2026.",
            "keywords": "mortgage prepayment, interest rates, refinance",
            "depth": "standard",
            "research_provider_strategy": "linkup_only",
            "rag_endpoint": "http://rag.local/query_simple",
            "rag_collection": "mortgage_notes",
            "rag_balance_emphasis": "balanced",
            "research_dossier": {
                "version": "test",
                "summary": "Deep Research says 2026 mortgage prepayment decisions depend on rates, liquidity, taxes, and opportunity cost.",
                "primary_claims": [
                    {"claim": "Prepayment decisions should compare mortgage APR against expected after-tax investment returns."},
                    {"claim": "Emergency liquidity changes whether extra principal payments are prudent."},
                ],
                "source_map": [
                    {
                        "title": "Federal mortgage rate data",
                        "url": "https://example.gov/mortgage-rates",
                        "source": "Government mortgage rate dataset",
                    },
                    {
                        "title": "Market return assumptions",
                        "url": "https://example.org/market-outlook",
                        "source": "Institutional market outlook",
                    },
                ],
                "source_quality_summary": {"source_count": 2},
                "dossier_quality_score": 40,
            },
            "dossier_status": "ready",
            "dossier_validated": True,
        },
        "claims": [{"claim": "Prepaying principal can reduce total interest paid."}],
    }


def test_mode_dossier_only_skips_rag_and_web(monkeypatch):
    counters = _Counters()
    _patch_dependencies(monkeypatch, counters)

    result = tasks._collect_evidence(_base_result("dossier_only"))

    assert counters.rag_calls == 0
    assert counters.linkup_calls == 0
    assert result["stage_data"]["dossier_sources"] == 2
    assert result["stage_data"]["rag_sources"] == 0
    assert result["stage_data"]["web_sources"] == 0
    assert len(result["citation_seed_evidence"]) == 2


def test_mode_dossier_plus_rag_uses_rag_only(monkeypatch):
    counters = _Counters()
    _patch_dependencies(monkeypatch, counters)

    result = tasks._collect_evidence(_base_result("dossier_plus_rag"))

    assert counters.rag_calls == 1
    assert counters.linkup_calls == 0
    assert result["stage_data"]["dossier_sources"] == 2
    assert result["stage_data"]["rag_sources"] >= 1
    assert result["stage_data"]["web_sources"] == 0
    assert len(result["citation_seed_evidence"]) >= 3


def test_mode_dossier_plus_rag_plus_live_web_uses_both_sources(monkeypatch):
    counters = _Counters()
    _patch_dependencies(monkeypatch, counters)

    result = tasks._collect_evidence(_base_result("dossier_plus_rag_plus_live_web"))

    assert counters.rag_calls == 1
    assert counters.linkup_calls >= 1
    assert result["stage_data"]["dossier_sources"] == 2
    assert result["stage_data"]["rag_sources"] >= 1
    assert result["stage_data"]["web_sources"] >= 1


def test_mode_rag_only_uses_rag_without_web(monkeypatch):
    counters = _Counters()
    _patch_dependencies(monkeypatch, counters)

    result = tasks._collect_evidence(_base_result("rag_only"))

    assert counters.rag_calls == 1
    assert counters.linkup_calls == 0
    assert result["stage_data"]["dossier_sources"] == 0
    assert result["stage_data"]["rag_sources"] >= 1
    assert result["stage_data"]["web_sources"] == 0


def test_legacy_inference_defaults_to_dual_source_when_rag_and_claims_enabled(monkeypatch):
    counters = _Counters()
    _patch_dependencies(monkeypatch, counters)

    result_data = _base_result("")
    result_data["research_data"].pop("source_strategy", None)
    result_data["research_data"]["rag_enabled"] = True
    result_data["research_data"]["claims_research_enabled"] = True

    result = tasks._collect_evidence(result_data)

    assert counters.rag_calls == 1
    assert counters.linkup_calls >= 1
    assert result["stage_data"]["rag_sources"] >= 1
    assert result["stage_data"]["web_sources"] >= 1


def test_dossier_plus_rag_empty_rag_preserves_dossier_evidence(monkeypatch):
    counters = _Counters()
    counters.rag_results = []
    _patch_dependencies(monkeypatch, counters)

    result = tasks._collect_evidence(_base_result("dossier_plus_rag"))

    assert counters.rag_calls == 1
    assert counters.linkup_calls == 0
    assert result["stage_data"]["dossier_sources"] == 2
    assert result["stage_data"]["rag_sources"] == 0
    assert len(result["citation_seed_evidence"]) == 2


def test_source_strategy_raises_when_no_allowed_source_has_evidence(monkeypatch):
    counters = _Counters()
    _patch_dependencies(monkeypatch, counters)
    result_data = _base_result("dossier_only")
    result_data["research_data"].pop("research_dossier")
    result_data["research_data"]["prior_citations"] = []

    with pytest.raises(RuntimeError, match="No citation-grade evidence collected"):
        tasks._collect_evidence(result_data)
