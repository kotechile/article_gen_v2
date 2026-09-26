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


def test_smart_brevity_polishing_agent_formats_full_structure(monkeypatch):
    sample_smart_brevity_output = """
<p>Moving to a new city for a tech role often delivers a massive salary bump, but hidden expenses can instantly turn that career milestone into a financial deficit [1].</p>

<p><strong>The big picture:</strong> Employers gladly pay a 10% to 20% "mobility premium" to pull specialized talent away from their current zip codes [2]. However, scoring a new job title doesn't guarantee actual wealth.</p>

<p><strong>By the numbers:</strong></p>
<ul>
  <li><strong>10% to 20%:</strong> The median salary increase professionals command when relocating to a new metropolitan area.</li>
  <li><strong>3% to 5%:</strong> The typical annual adjustment for a lateral job move within your own local market.</li>
  <li><strong>49%:</strong> The portion of relocated workers who face a higher cost of living that swallows their raise.</li>
  <li><strong>14%:</strong> The percentage of relocators who actually take a pay cut just to make the move happen for long-term positioning.</li>
  <li><strong>$14,289:</strong> The average federal tax liability attached to a relocation package if the employer doesn't provide a tax "gross-up".</li>
</ul>

<p><strong>Why it matters:</strong> Jumping to a Tier 1 tech hub resets your compensation to a higher global standard, raising the floor for all future salary negotiations. Yet, you must calculate your "real income" by subtracting localized housing, grocery, and tax expenses from your new base pay. Without doing this math, you risk building nominal wealth while your actual standard of living declines.</p>

<p><strong>The catch:</strong> While executives enjoy "white-glove" full-service moves, mid-level employees usually receive taxable lump-sum checks. This gives you flexibility but dumps the administrative burden, vendor management, and heavy tax liabilities entirely on your shoulders.</p>

<p><strong>Go deeper:</strong> Review the full market data on the economic impact of mobility to see if that offer letter is actually a strategic career advancement or a costly mistake.</p>

<h2>Go Deeper</h2>

<h3>Cost of Living Breakdown</h3>
<p>Relocation packages often look generous on paper until cost of living adjustments are computed [1].</p>

<table>
  <thead>
    <tr><th>Expense Category</th><th>Origin Metro</th><th>Target Tech Hub</th><th>Variance</th></tr>
  </thead>
  <tbody>
    <tr><td>Median 2BR Rent</td><td>$1,800/mo</td><td>$3,400/mo</td><td>+88%</td></tr>
    <tr><td>State Income Tax</td><td>0%</td><td>9.3%</td><td>+9.3%</td></tr>
  </tbody>
</table>
"""
    client = FakeLLMClient(sample_smart_brevity_output)
    monkeypatch.setattr(tasks, "create_llm_client", lambda **kwargs: client)

    html_in = "<h2>Relocation Overview</h2><p>Here is raw unformatted draft about tech relocations with citations [1] and [2].</p>"
    research_data = {
        "primary_keyword": "tech relocation packages",
        "tone": "authoritative",
        "writer_notes": "Focus on hidden tax burdens",
    }
    structure = {"title": "The True Economics of Relocating for Tech Jobs"}

    out = tasks._polish_and_format_article(html_in, research_data, structure)

    # Verify all Smart Brevity components are present
    assert "<strong>The big picture:</strong>" in out
    assert "<strong>By the numbers:</strong>" in out
    assert "<strong>10% to 20%:</strong>" in out
    assert "<strong>$14,289:</strong>" in out
    assert "<strong>Why it matters:</strong>" in out
    assert "<strong>The catch:</strong>" in out
    assert "<strong>Go deeper:</strong>" in out
    assert "<h2>Go Deeper</h2>" in out
    # Verify comparative table is present
    assert "<table>" in out
    # Verify citations are preserved
    assert "[1]" in out
    assert "[2]" in out


def test_validate_and_ensure_smart_brevity_structure_synthesizes_missing_blocks():
    # Raw HTML with no Smart Brevity lead
    raw_html = """
<h2>Tech Relocation Realities</h2>
<p>Relocating for a senior engineering role brings average salary jumps of 10% to 20%, but $14,289 in taxes can surprise workers. Around 49% of relocators face unexpected living costs.</p>
<p>Detailed analysis of moving costs, housing deposits, and tax gross-up clauses [1].</p>
"""
    research_data = {
        "primary_keyword": "tech relocation costs",
        "brief": "Analysis of moving costs and hidden relocation expenses.",
    }
    structure = {
        "title": "Tech Relocation Guide",
        "hook": "Relocation packages can deliver a salary boost or a hidden tax trap.",
        "thesis": "Evaluating real income after local taxes and housing costs is critical for career moves.",
    }

    out = tasks._validate_and_ensure_smart_brevity_structure(raw_html, research_data, structure)

    assert "<strong>The big picture:</strong>" in out
    assert "<strong>By the numbers:</strong>" in out
    assert "<strong>Why it matters:</strong>" in out
    assert ("<strong>The bottom line:</strong>" in out or "<strong>The catch:</strong>" in out)
    assert "<strong>Go deeper:</strong>" in out
    assert "<h2>Go Deeper</h2>" in out
    # Check that numbers were extracted into By the numbers
    assert "10% to 20%" in out or "$14,289" in out or "49%" in out
    assert "[1]" in out


def test_extract_numbers_for_smart_brevity():
    text = "<p>Data shows that 49% of workers struggle with rent, while $14,289 is the average tax bill. Also 10% to 20% salary bumps are common.</p>"
    items = tasks._extract_numbers_for_smart_brevity(text, {}, {})
    
    assert len(items) >= 2
    assert any("49%" in item for item in items)
    assert any("$14,289" in item for item in items)
    assert any("<strong>" in item for item in items)


def test_finalize_article_with_smart_brevity_integration():
    body_text = (
        "<p>Relocating across state lines introduces $15,000 in unexpected moving expenses and a 49% cost increase [1].</p>"
        "<h3>Housing Analysis</h3><p>Local rent accounts for the largest budget deviation [2].</p>"
    )
    result = {
        "structure": {
            "title": "Navigating Tech Relocations",
            "hook": "Relocating for a tech role offers a salary bump with hidden tax trade-offs.",
            "thesis": "Calculating localized expenses is essential to ensure a net-positive move.",
            "keywords": ["tech relocation", "cost of living"],
            "tone": "professional",
            "target_audience": "tech professionals",
        },
        "content": {
            "sections": [
                {
                    "title": "Introduction",
                    "content": body_text,
                }
            ],
            "word_count": 600,
        },
        "citations": [
            {
                "title": "Tech Labor Mobility Index",
                "url": "https://example.com/mobility-index",
                "author": "Labor Institute",
                "publication_date": "2026",
            },
            {
                "title": "Metropolitan Living Cost Survey",
                "url": "https://example.com/cost-survey",
                "author": "Metro Analytics",
                "publication_date": "2026",
            },
        ],
        "claim_bundles": [],
        "research_data": {
            "articleLength": 1000,
            "include_in_text_citations": True,
            "primary_keyword": "tech relocation",
        },
    }

    finalized = tasks._finalize_article(result)
    final_article = finalized["final_article"]
    html = final_article["html_content"]

    # Verify Smart Brevity structure in final output
    assert "<strong>The big picture:</strong>" in html
    assert "<strong>By the numbers:</strong>" in html
    assert "<strong>Why it matters:</strong>" in html
    assert ("<strong>The bottom line:</strong>" in html or "<strong>The catch:</strong>" in html)
    assert "<strong>Go deeper:</strong>" in html
    assert "<h2>Go Deeper</h2>" in html
    assert "References" in html
    assert 'href="https://example.com/mobility-index"' in html


def test_polish_and_format_article_prompt_mandates_go_deeper_concept_coverage(monkeypatch):
    captured_prompts = []

    class CapturingLLMClient:
        def generate(self, messages):
            captured_prompts.extend([m.get("content", "") for m in messages])
            return SimpleNamespace(content="<p><strong>The big picture:</strong> AI is evolving.</p><h2>Go Deeper</h2><p>Full breakdown.</p>")

    monkeypatch.setattr(tasks, "create_llm_client", lambda **kwargs: CapturingLLMClient())

    tasks._polish_and_format_article(
        "<p>Initial draft.</p>",
        {"primary_keyword": "agentic AI"},
        {"title": "Agentic AI Adoption"}
    )

    combined = "\n".join(captured_prompts)
    assert "CRITICAL ALIGNMENT - COMPLETE CONCEPT DEVELOPMENT UNDER 'GO DEEPER'" in combined
    assert "Every single topic, theme, concept, statistic, data point, and comparison mentioned in the Executive Smart Brevity Lead" in combined
    assert "MUST be thoroughly and substantively developed, explained, and substantiated inside the substantive body sections under `<h2>Go Deeper</h2>`" in combined
