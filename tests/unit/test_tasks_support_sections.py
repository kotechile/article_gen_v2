import sys
from pathlib import Path
from types import SimpleNamespace
import types


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_task_import_stubs():
    class DummyCeleryApp:
        def task(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    sys.modules.setdefault("celery", SimpleNamespace(current_task=None))
    sys.modules.setdefault("celery_config", SimpleNamespace(celery=DummyCeleryApp()))
    sys.modules.setdefault(
        "supabase_client",
        SimpleNamespace(
            LLM_ROLE_ARTICLE_GENERATION="article_generation",
            LLM_ROLE_FINAL_REVIEW="final_review",
            get_supabase_client=lambda: None,
            get_llm_api_key=lambda *args, **kwargs: "",
            get_linkup_api_key=lambda: "",
            get_default_llm_provider=lambda: ("openai", "gpt-4", ""),
            get_llm_provider_for_role=lambda role: ("openai", "gpt-4", ""),
        ),
    )
    sys.modules.setdefault("llm_client", SimpleNamespace(create_llm_client=lambda **kwargs: None))
    sys.modules.setdefault("rag_client", SimpleNamespace(create_rag_client=lambda **kwargs: None, RAGQuery=dict))
    sys.modules.setdefault("linkup_client", SimpleNamespace(create_linkup_client=lambda **kwargs: None, SearchQuery=dict))
    sys.modules.setdefault("article_structure_generator", SimpleNamespace(create_article_structure_generator=lambda *args, **kwargs: None))
    sys.modules.setdefault(
        "content_generator",
        SimpleNamespace(
            create_content_generator=lambda *args, **kwargs: None,
            get_tone_specific_instructions=lambda tone: str(tone or ""),
        ),
    )
    sys.modules.setdefault(
        "citation_generator",
        SimpleNamespace(create_citation_generator=lambda *args, **kwargs: None, CitationStyle=types.SimpleNamespace()),
    )
    sys.modules.setdefault("src.utils.config", SimpleNamespace(get_config=lambda: {}))


_install_task_import_stubs()

import tasks  # noqa: E402


class FakeLLMClient:
    def __init__(self, content: str):
        self.content = content

    def generate(self, _messages):
        return SimpleNamespace(content=self.content)


def test_extract_json_payload_from_response_handles_fenced_json():
    payload = tasks._extract_json_payload_from_response(
        'Here you go:\n```json\n{"takeaways":[{"text":"First."}]}\n```'
    )

    assert payload == {"takeaways": [{"text": "First."}]}


def test_normalize_takeaway_items_filters_prompt_leakage():
    items = tasks._normalize_takeaway_items(
        {
            "takeaways": [
                {"text": "Primary keyword: lease break costs"},
                {"text": "Reasoning: the article compares penalty fees with subletting timelines."},
                {"text": "Breaking a lease often costs more than the penalty alone because deposits, ongoing rent, and re-rental fees can all stack up quickly."},
                {"text": "State law and the lease contract both shape what a landlord can legally charge, so readers need to check both before making a move."},
                {"text": "Subletting can reduce losses, but it usually works only when timing, landlord approval, and local demand line up in your favor."},
            ]
        }
    )

    assert len(items) == 3
    assert all("keyword" not in item.lower() for item in items)
    assert all("reasoning" not in item.lower() for item in items)


def test_generate_key_takeaways_from_article_renders_html():
    client = FakeLLMClient(
        '{"takeaways":['
        '{"text":"Breaking a lease can trigger several overlapping costs, so the headline penalty is rarely the full financial picture."},'
        '{"text":"Readers need to compare the lease language with local law because legal caps and mitigation rules vary by state and city."},'
        '{"text":"Subletting usually works best when renters act early enough to secure approval and line up a qualified replacement tenant."}'
        ']}'
    )

    html = tasks._generate_key_takeaways_from_article(
        llm_client=client,
        article_title="Lease Exit Costs",
        article_text="A full article body about breaking a lease.",
    )

    assert "<h2>Key Takeaways</h2>" in html
    assert html.count("<li>") == 3


def test_generate_faq_from_article_renders_html():
    client = FakeLLMClient(
        '{"faq":['
        '{"question":"Can a landlord charge any lease-break fee?","answer":"Not always. Some states cap fees or require landlords to reduce damages by trying to re-rent the unit."},'
        '{"question":"Is subletting always cheaper than paying the penalty?","answer":"No. It can save money, but only if the lease allows it and you can find a replacement tenant in time."},'
        '{"question":"What should renters review before making a decision?","answer":"They should compare the lease terms, local law, timing, and the realistic cost of each exit option."}'
        ']}'
    )

    html = tasks._generate_faq_from_article(
        llm_client=client,
        article_title="Lease Exit Costs",
        article_text="A full article body about breaking a lease.",
    )

    assert "<h2>FAQ</h2>" in html
    assert html.count("<h3>") == 3
    assert "Question:" not in html


def test_pop_named_h2_section_removes_existing_support_block():
    html = (
        "<p>Intro.</p>\n"
        "<h2>Key Takeaways</h2>\n<ul><li>One.</li></ul>\n"
        "<h2>Body</h2>\n<p>Rest.</p>"
    )

    cleaned, extracted = tasks._pop_named_h2_section(html, r"key\s+takeaways")

    assert "<h2>Key Takeaways</h2>" not in cleaned
    assert "<h2>Body</h2>" in cleaned
    assert extracted.startswith("<h2>Key Takeaways</h2>")
