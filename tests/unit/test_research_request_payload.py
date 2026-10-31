"""Resuming a generation round-trips a task payload through request validation.

A RAG-enabled request carries `rag_collection` with no `rag_endpoint`, because the
endpoint comes from config on the worker side. The create-task endpoint accepted it
when the key was omitted and rejected the identical resume when the worker's payload
spelled it as an explicit null — which is what "Validation failed" at the
controversy gate was.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("pydantic")

from src.core.models.research import (  # noqa: E402
    ResearchRequest,
    build_research_request_payload,
    sanitize_research_data_for_storage,
)


def _resume_payload(**overrides):
    payload = {
        "brief": "Fixing order fill rate in manufacturing - research brief",
        "keywords": "Order Fill Rate, Case Fill Rate, CFR",
        "provider": "deepseek",
        "model": "deepseek-flash",
        "depth": "standard",
        "tone": "professional",
        "target_word_count": 5000,
        "target_audience": "Supply chain executives",
        "source_strategy": "rag_plus_live_web",
        "claims_research_enabled": True,
        "rag_enabled": True,
        "rag_collection": "giniloh",
        "rag_endpoint": None,
        "rag_llm_provider": None,
        "identify_controversies": False,
        "article_id": "7ea9e922-a55f-42d1-abd9-cd1fb73f2e56",
        "selected_controversies": [
            {"id": "controversy_1", "selected_take_id": "take_1"},
        ],
    }
    payload.update(overrides)
    return payload


def test_explicit_nulls_are_rejected_without_normalisation():
    """Documents the defect: the same request accepted when the keys are omitted."""
    with pytest.raises(Exception) as excinfo:
        ResearchRequest(**_resume_payload())

    message = str(excinfo.value)
    assert "rag_endpoint" in message
    assert "rag_llm_provider" in message


def test_resume_payload_validates_once_nulls_are_dropped():
    payload = build_research_request_payload(_resume_payload())

    assert "rag_endpoint" not in payload
    assert "rag_llm_provider" not in payload

    model = ResearchRequest(**payload)
    assert model.rag_collection == "giniloh"
    assert model.rag_enabled is True
    # Not a model field: it reaches the worker through the endpoint's extra_fields.
    assert payload["identify_controversies"] is False


def test_required_fields_still_fail_when_null():
    payload = build_research_request_payload(_resume_payload(brief=None))

    with pytest.raises(Exception) as excinfo:
        ResearchRequest(**payload)

    assert "brief" in str(excinfo.value)


def test_false_and_empty_values_survive_normalisation():
    """Only nulls are stripped: False/0/"" carry meaning (e.g. identify_controversies)."""
    payload = build_research_request_payload(
        _resume_payload(identify_controversies=False, target_word_count=0, writer_notes="")
    )

    assert payload["identify_controversies"] is False
    assert payload["target_word_count"] == 0
    assert payload["writer_notes"] == ""


def test_storage_copy_keeps_the_request_but_drops_provider_keys():
    cleaned = sanitize_research_data_for_storage(
        _resume_payload(api_key="sk-live-secret", rag_api_key="rag-secret")
    )

    assert "api_key" not in cleaned
    assert "rag_api_key" not in cleaned
    assert cleaned["brief"].startswith("Fixing order fill rate")
    assert cleaned["keywords"] == "Order Fill Rate, Case Fill Rate, CFR"
    # The stored copy is itself a valid resume payload.
    assert ResearchRequest(**build_research_request_payload(cleaned)).rag_collection == "giniloh"


def test_storage_copy_accepts_a_missing_payload():
    assert sanitize_research_data_for_storage({}) == {}
    assert sanitize_research_data_for_storage(None) == {}
