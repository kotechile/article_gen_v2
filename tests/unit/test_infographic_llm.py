import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.services.infographic_llm import (
    append_infographic_llm_log,
    normalize_infographic_icon,
    normalize_infographic_payload_icons,
)


def test_normalize_infographic_icon_accepts_known_free_icon():
    icon, audit = normalize_infographic_icon("fa-solid fa-fire")

    assert icon == "fa-solid fa-fire"
    assert audit["status"] == "accepted"


def test_normalize_infographic_icon_upgrades_legacy_prefix():
    icon, audit = normalize_infographic_icon("fas fa-chart-line")

    assert icon == "fa-solid fa-chart-line"
    assert audit["status"] == "accepted"


def test_normalize_infographic_icon_falls_back_for_unknown_icon():
    icon, audit = normalize_infographic_icon(
        "fa-solid fa-totally-made-up-icon",
        item_text="Automate",
        description_text="Set up automatic bank transfers to grow savings.",
    )

    assert icon == "fa-solid fa-gears"
    assert audit["status"] == "fallback"


def test_normalize_infographic_payload_icons_updates_items_in_place_copy():
    payload = {
        "title": "Emergency Fund",
        "items": [
            {
                "item1": "Burn Rate",
                "description1": "Track monthly spending.",
                "icon1": "fas fa-chart-line",
            },
            {
                "item2": "Volatility",
                "description2": "Measure income risk.",
                "icon2": "fa-solid fa-does-not-exist",
            },
        ],
    }

    normalized, audit = normalize_infographic_payload_icons(payload)

    assert normalized["items"][0]["icon1"] == "fa-solid fa-chart-line"
    assert normalized["items"][1]["icon2"] == "fa-solid fa-bolt"
    assert payload["items"][0]["icon1"] == "fas fa-chart-line"
    assert len(audit) == 2


def test_append_infographic_llm_log_writes_jsonl(tmp_path):
    log_path = tmp_path / "infographic_llm_responses.jsonl"

    append_infographic_llm_log(
        str(log_path),
        {
            "status": "success",
            "raw_response": '{"title":"Sample"}',
        },
    )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["status"] == "success"
    assert payload["raw_response"] == '{"title":"Sample"}'
    assert "timestamp" in payload
