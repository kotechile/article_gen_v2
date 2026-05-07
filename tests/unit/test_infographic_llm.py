import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.services.infographic_llm import (
    apply_icon_markup_to_html,
    append_infographic_llm_log,
    build_fontawesome_icon_markup,
    inject_fontawesome_icon_styles,
    normalize_infographic_icon,
    normalize_infographic_payload_icons,
    write_infographic_render_debug_artifacts,
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


def test_apply_icon_markup_to_html_replaces_class_and_text_placeholders():
    html = """
    <div class="material-icons +icon1+"></div>
    <span>+icon2+</span>
    <p data-icon="+icon3+">+icon3+</p>
    """

    updated_html, audit = apply_icon_markup_to_html(
        html,
        {
            "icon1": "fa-solid fa-chart-column",
            "icon2": "fa-solid fa-robot",
            "icon3": "fa-solid fa-bolt",
        },
    )

    assert "cg-fa-icon cg-fa-icon--solid fa-solid fa-chart-column" in updated_html
    assert '<span><i class="cg-fa-icon cg-fa-icon--solid fa-solid fa-robot" aria-hidden="true"></i></span>' in updated_html
    assert 'data-icon="fa-solid fa-bolt"' in updated_html
    assert any(entry["field"] == "icon1" and entry["attribute_replacements"] == 1 for entry in audit)
    assert any(entry["field"] == "icon2" and entry["text_replacements"] == 1 for entry in audit)


def test_inject_fontawesome_icon_styles_only_adds_helper_once():
    css = ".infographic-template { width: 100px; }"

    first = inject_fontawesome_icon_styles(css)
    second = inject_fontawesome_icon_styles(first)

    assert "Codex infographic icon hardening" in first
    assert second.count("Codex infographic icon hardening") == 1


def test_build_fontawesome_icon_markup_uses_helper_classes():
    markup = build_fontawesome_icon_markup("fa-brands fa-github")

    assert 'class="cg-fa-icon cg-fa-icon--brands fa-brands fa-github"' in markup
    assert 'aria-hidden="true"' in markup


def test_write_infographic_render_debug_artifacts_writes_html_and_css(tmp_path):
    paths = write_infographic_render_debug_artifacts(
        str(tmp_path),
        template_id=39,
        request_timestamp="20260507T174000_123456",
        html_content="<div>hello</div>",
        css_content=".foo { color: red; }",
    )

    assert Path(paths["html_path"]).read_text(encoding="utf-8") == "<div>hello</div>"
    assert Path(paths["css_path"]).read_text(encoding="utf-8") == ".foo { color: red; }"
