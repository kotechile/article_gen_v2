import sys
from pathlib import Path
from types import SimpleNamespace
import types

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _force(name, module):
    """Install `module` as the import target, even if it was already imported.

    `setdefault` deferred to a real module another test file had imported, which
    made these tests pass alone and fail in a full run.
    """
    sys.modules[name] = module


def _install_task_import_stubs():
    class DummyCeleryApp:
        def task(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    _force("celery", SimpleNamespace(current_task=None))
    _force("celery_config", SimpleNamespace(celery=DummyCeleryApp()))
    _force(
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
    _force("llm_client", SimpleNamespace(create_llm_client=lambda **kwargs: None))
    _force("rag_client", SimpleNamespace(create_rag_client=lambda **kwargs: None, RAGQuery=dict))
    _force("linkup_client", SimpleNamespace(create_linkup_client=lambda **kwargs: None, SearchQuery=dict))
    _force("article_structure_generator", SimpleNamespace(create_article_structure_generator=lambda *args, **kwargs: None))
    _force(
        "content_generator",
        SimpleNamespace(
            create_content_generator=lambda *args, **kwargs: None,
            get_tone_specific_instructions=lambda tone: str(tone or ""),
        ),
    )
    _force("src.utils.config", SimpleNamespace(get_config=lambda: {}))


_install_task_import_stubs()

# Another test file may have imported the real `tasks` first. Its module-level
# stubs only take effect on a fresh import, so force one; tests already holding
# the real module keep their own reference.
sys.modules.pop("tasks", None)

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
        # Evidence the numbers block is allowed to draw from: without a retained
        # source a number is dropped instead of published.
        "research_dossier": {
            "source_map": [
                {
                    "title": "Tech Labor Mobility Index",
                    "url": "https://example.com/mobility-index",
                    "content": "Employers pay a 10% to 20% mobility premium to relocate specialized talent between metropolitan areas.",
                },
                {
                    "title": "Relocation tax gross-up analysis",
                    "url": "https://example.com/tax-grossup",
                    "content": "The average federal tax liability attached to a relocation package is $14,289 when the employer does not provide a gross-up.",
                },
            ]
        },
    }
    structure = {"title": "The True Economics of Relocating for Tech Jobs"}

    out = tasks._polish_and_format_article(html_in, research_data, structure)

    # Verify all Smart Brevity components are present
    assert "<strong>The big picture:</strong>" in out
    assert "<strong>By the numbers:</strong>" in out
    assert "10% to 20%" in out
    assert "$14,289" in out
    assert "<strong>Why it matters:</strong>" in out
    assert "<strong>The catch:</strong>" in out
    # The "Go deeper: Review the detailed breakdown..." announcement is gone; the
    # divider is enough.
    assert "Review the detailed breakdown" not in out
    assert "<h2>Go Deeper</h2>" in out
    # Verify comparative table is present
    assert "<table>" in out
    # Verify citations are preserved
    assert "[1]" in out or "[2]" in out


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
        "research_dossier": {
            "source_map": [
                {
                    "title": "Tech Labor Mobility Index",
                    "url": "https://example.com/mobility-index",
                    "content": "Relocating for a senior engineering role brings average salary jumps of 10% to 20% in the target market.",
                },
                {
                    "title": "Relocation tax gross-up analysis",
                    "url": "https://example.com/tax-grossup",
                    "content": "$14,289 in taxes can surprise workers when the employer does not gross up the relocation package.",
                },
                {
                    "title": "Metropolitan Living Cost Survey",
                    "url": "https://example.com/cost-survey",
                    "content": "Around 49% of relocators face unexpected living costs in the first year after moving.",
                },
            ]
        },
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
    # Closing modules are not invented: whatever the draft does not contain, the
    # article does not get. What must not happen is a generic filler sentence.
    assert "Evaluate the full lifecycle costs" not in out
    assert "Every tactical gain introduces" not in out
    assert "Review the detailed breakdown" not in out
    assert "<h2>Go Deeper</h2>" in out
    # Numbers are extracted into By the numbers only because they are backed by
    # evidence, and each keeps its subject plus a link to the source.
    assert "10% to 20%" in out or "$14,289" in out or "49%" in out
    assert 'href="https://example.com/' in out
    assert "[1]" in out


def test_validate_structure_omits_numbers_block_without_evidence():
    raw_html = (
        "<h2>Tech Relocation Realities</h2>"
        "<p>Relocating brings average salary jumps of 10% to 20%, but $14,289 in taxes can surprise workers.</p>"
    )
    research_data = {"primary_keyword": "tech relocation costs"}
    structure = {"title": "Tech Relocation Guide"}

    out = tasks._validate_and_ensure_smart_brevity_structure(raw_html, research_data, structure)

    # No retained evidence -> no invented numbers block, and no invented lead
    # sentence either: with no thesis to draw on the block is simply absent.
    assert "<strong>By the numbers:</strong>" not in out
    assert "Evaluate the full lifecycle costs" not in out
    assert "Mastering the fundamentals" not in out


def test_extract_numbers_for_smart_brevity():
    text = (
        "<p>Data shows that 49% of workers struggle with rent, while $14,289 is the average tax bill. "
        "Relocation packages lift salaries by 10% to 20% in the target market.</p>"
    )
    research_data = {
        "research_dossier": {
            "source_map": [
                {
                    "title": "Metropolitan Living Cost Survey",
                    "url": "https://example.com/cost-survey",
                    "content": "Data shows that 49% of workers struggle with rent in their first year after relocating.",
                },
                {
                    "title": "Relocation tax gross-up analysis",
                    "url": "https://example.com/tax-grossup",
                    "content": "The average tax bill for a relocation package is $14,289 without a gross-up clause.",
                },
                {
                    "title": "Tech Labor Mobility Index",
                    "url": "https://example.com/mobility-index",
                    "content": "Relocation packages lift salaries by 10% to 20% in the target market.",
                },
            ]
        }
    }
    items = tasks._extract_numbers_for_smart_brevity(text, research_data, {})

    assert len(items) >= 3
    assert any("49%" in item for item in items)
    assert any("$14,289" in item for item in items)
    # Every published number carries a subject label and a verifiable link.
    for item in items:
        assert "<strong>" in item
        assert 'href="https://example.com/' in item
        # A unit from a neighbouring number must never be grafted on.
        assert "$14,289 %" not in item


def test_extract_numbers_for_smart_brevity_returns_nothing_without_evidence():
    text = "<p>Data shows that 49% of workers struggle with rent, while $14,289 is the average tax bill.</p>"
    assert tasks._extract_numbers_for_smart_brevity(text, {}, {}) == []


def test_finalize_article_with_smart_brevity_integration():
    body_text = (
        "<p>Relocating across state lines introduces $15,000 in unexpected moving expenses and a 49% cost increase [1].</p>"
        "<p>Housing deposits absorb a further $3,000 before the first paycheck [2].</p>"
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
            # Matches the fixture's own 600-word body: the publish gate compares the
            # finished HTML to the requested length, so the request has to be the
            # size this synthetic article actually claims.
            "articleLength": 120,  # matches what this fixture actually produces
            "include_in_text_citations": True,
            "primary_keyword": "tech relocation",
            # Without retained evidence the numbers block is skipped by design.
            "research_dossier": {
                "source_map": [
                    {
                        "title": "Moving cost survey",
                        "url": "https://example.com/moving-costs",
                        "content": "Relocating across state lines introduces $15,000 in unexpected moving expenses on average.",
                    },
                    {
                        "title": "Cost of living index",
                        "url": "https://example.com/cost-increase",
                        "content": "Relocated employees report a 49% cost increase in their first year.",
                    },
                    {
                        "title": "Housing deposit analysis",
                        "url": "https://example.com/housing-deposits",
                        "content": "Housing deposits absorb a further $3,000 before the first paycheck arrives.",
                    },
                ]
            },
        },
    }

    finalized = tasks._finalize_article(result)
    final_article = finalized["final_article"]
    html = final_article["html_content"]

    # Verify Smart Brevity structure in final output
    assert "<strong>The big picture:</strong>" in html
    assert "<strong>By the numbers:</strong>" in html
    assert "<strong>Why it matters:</strong>" in html
    # No invented conclusion: the fixture's draft carries none, so none appears.
    assert "Evaluate the full lifecycle costs" not in html
    assert "Every tactical gain introduces" not in html
    assert "Review the detailed breakdown" not in html
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


def test_editorial_architecture_polishing_and_validation(monkeypatch):
    editorial_sample_html = """
<p><strong>The Big Picture:</strong> Premature microservice decomposition creates unmanageable distributed transaction debt before achieving product-market fit.</p>
<hr>

<h2>Go Deeper</h2>

<h3>Slashing Distributed Lock Contention by 40%</h3>
<p>Scaling concurrent database writes on Postgres 16 quickly exhausts connection pools and triggers high latency cascades under heavy checkout loads.</p>
<ul>
  <li><strong>Lock Hold Duration:</strong> Spikes from 4ms to 180ms during contention spikes, directly bottlenecking API response times.</li>
  <li><strong>Connection Pool Ceiling:</strong> Strict max_connections limits force thread starvation in application containers.</li>
  <li><strong>Rollback Overhead:</strong> Aborted transactions consume 25% of CPU cycles without completing business logic.</li>
</ul>
<hr>

<h3>What I'd Watch:</h3>
<ul>
  <li><strong>Deadlock Frequency:</strong> Track automated deadlock detection events in pg_stat_activity logs.</li>
  <li><strong>Serialization Failures:</strong> Monitor repeatable read retry rates across checkout services.</li>
  <li><strong>Replication Lag Drift:</strong> Where read replicas diverge by more than 500ms during failovers.</li>
</ul>
<hr>

<h3>The Catch</h3>
<p>Optimizing lock contention with optimistic concurrency control shifts operational complexity into application retry mechanisms. Benchmarks under synthetic loads underestimate distributed edge cases in multi-region environments.</p>
<hr>

<h3>At a Glance</h3>
<ul>
  <li><strong>The Big Shift:</strong> High-concurrency systems are replacing heavy locking protocols with event-driven optimistic workflows.</li>
  <li><strong>Why It Matters:</strong> Unmitigated lock contention degrades user checkout conversion and inflates cloud compute costs by 3x.</li>
  <li><strong>The Bottom Line:</strong> Audit write transaction boundaries before attempting costly database re-platforming.</li>
</ul>
"""
    client = FakeLLMClient(editorial_sample_html)
    monkeypatch.setattr(tasks, "create_llm_client", lambda **kwargs: client)

    html_in = "<h2>Database Overview</h2><p>Draft database text.</p>"
    research_data = {
        "primary_keyword": "database lock contention",
        "brief": "Analysis of distributed locking and connection starvation.",
    }
    structure = {
        "title": "Why Fast Growth Kills Databases",
        "thesis": "Premature microservice decomposition creates unmanageable distributed transaction debt before achieving product-market fit.",
    }

    polished = tasks._polish_and_format_article(html_in, research_data, structure)
    validated = tasks._validate_and_ensure_smart_brevity_structure(polished, research_data, structure)

    # Verify The Big Picture
    assert "<strong>The Big Picture:</strong>" in validated or "<strong>The big picture:</strong>" in validated
    # Verify What I'd Watch
    assert "What I'd Watch" in validated
    # Verify The Catch
    assert "The Catch" in validated
    # Verify At a Glance and its 3 bullets
    assert "At a Glance" in validated
    assert "<strong>The Big Shift:</strong>" in validated
    assert "<strong>Why It Matters:</strong>" in validated
    assert "<strong>The Bottom Line:</strong>" in validated


def test_quality_evaluator_detects_editorial_architecture():
    from src.services.article_quality_evaluator import build_article_quality_report

    html_content = """
<p><strong>The Big Picture:</strong> Distributed lock contention destroys database throughput.</p>
<hr>
<h3>Slashing Distributed Lock Contention by 40%</h3>
<p>Scaling concurrent database writes on Postgres 16 quickly exhausts connection pools [1].</p>
<ul>
  <li><strong>Lock Hold Duration:</strong> Spikes from 4ms to 180ms during contention spikes.</li>
  <li><strong>Connection Pool Ceiling:</strong> Limits force thread starvation.</li>
  <li><strong>Rollback Overhead:</strong> Consumes 25% CPU cycles.</li>
</ul>
<hr>
<h3>What I'd Watch:</h3>
<ul>
  <li><strong>Deadlock Frequency:</strong> Track automated deadlock detection events.</li>
  <li><strong>Serialization Failures:</strong> Monitor repeatable read retry rates.</li>
  <li><strong>Replication Lag Drift:</strong> Where read replicas diverge by more than 500ms.</li>
</ul>
<hr>
<h3>The Catch</h3>
<p>Optimizing lock contention with optimistic concurrency shifts complexity into application retry mechanisms.</p>
<hr>
<h3>At a Glance</h3>
<ul>
  <li><strong>The Big Shift:</strong> High-concurrency systems are replacing heavy locking protocols.</li>
  <li><strong>Why It Matters:</strong> Unmitigated lock contention degrades user checkout conversion.</li>
  <li><strong>The Bottom Line:</strong> Audit write transaction boundaries before re-platforming.</li>
</ul>
<h2>Frequently Asked Questions</h2>
<h3>What is lock contention?</h3><p>Lock contention is a concurrency conflict.</p>
<h2>References</h2>
<p id="ref-1"><strong>[1]</strong> Database Survey 2026.</p>
"""
    report = build_article_quality_report(
        title="Why Fast Growth Kills Databases",
        html_content=html_content,
        citations=[{"id": 1, "url": "https://example.com"}],
        sections=[{"title": "Slashing Distributed Lock Contention by 40%"}],
        evidence_count=3,
    )

    geo = report["diagnostics"]["geo_signals"]
    assert geo["has_big_picture"] is True
    assert geo["has_what_id_watch"] is True
    assert geo["has_the_catch"] is True
    assert geo["has_at_a_glance"] is True
    assert geo["has_big_shift"] is True
    assert geo["has_why_matters"] is True
    assert geo["has_bottom_line"] is True
    assert not any("Missing 'The Big Picture'" in w for w in report["warnings"])
    assert not any("Missing 'What I\'d Watch'" in w for w in report["warnings"])
    assert not any("Missing 'The Catch'" in w for w in report["warnings"])
    assert not any("Missing 'At a Glance'" in w for w in report["warnings"])


def test_citation_reconciliation_keeps_only_used_references():
    content = (
        "Quantum computing enables 10x cryptanalysis speedup [3]. "
        "However, error rates remain a bottleneck [1]. "
        "Classical algorithms are still preferred for now [3]."
    )
    candidate_citations = [
        {"title": "IBM Quantum Hardware Report", "url": "https://ibm.com/quantum/report", "source_type": "web"},
        {"title": "Unused Source 2", "url": "https://example.org/article2", "source_type": "web"},
        {"title": "MIT Cryptography Journal", "url": "https://mit.edu/crypto/paper", "source_type": "journal"},
        {"title": "Unused Source 4", "url": "https://example.org/article4", "source_type": "web"},
        {"title": "Unused Source 5", "url": "https://example.org/article5", "source_type": "web"},
    ]

    updated_content, curated = tasks._reconcile_and_curate_citations(
        full_content=content,
        citations=candidate_citations,
        evidence=[],
        include_in_text_citations=True,
    )

    # Only 2 sources actually cited in text
    assert len(curated) == 2
    assert curated[0]["title"] == "MIT Cryptography Journal"
    assert curated[1]["title"] == "IBM Quantum Hardware Report"
    # Text markers renumbered sequentially: [3] -> [1], [1] -> [2]
    assert "[1]" in updated_content
    assert "[2]" in updated_content
    assert "[3]" not in updated_content

    # Now verify anchor linking
    html_linked = tasks._create_citation_links(updated_content, curated)
    assert '<a href="#ref-1" class="citation-link">[1]</a>' in html_linked
    assert '<a href="#ref-2" class="citation-link">[2]</a>' in html_linked


def test_citation_filters_spam_broken_urls_and_cleans_utm():
    from citation_generator import clean_citation_url, is_serious_citation, is_valid_citation_url

    # Spam domains
    assert is_valid_citation_url("https://www.pinterest.com/pin/12345") is False
    assert is_valid_citation_url("https://quora.com/what-is-quantum") is False
    assert is_valid_citation_url("https://buzzfeed.com/tech-news") is False

    # Broken/junk paths
    assert is_valid_citation_url("https://nature.com/404") is False
    assert is_valid_citation_url("https://nature.com/login") is False

    # Tracking parameter cleaning
    dirty_url = "https://nature.com/articles/s41586-026?utm_source=twitter&utm_medium=social&fbclid=XYZ123"
    cleaned = clean_citation_url(dirty_url)
    assert cleaned == "https://nature.com/articles/s41586-026"
    assert "utm_" not in cleaned
    assert "fbclid" not in cleaned

    # Placeholder title filtering
    placeholder_cite = {"title": "Knowledge Base Source 1", "url": "https://mit.edu/doc", "source_type": "rag"}
    assert is_serious_citation(placeholder_cite) is False

    serious_cite = {"title": "Quantum Error Mitigation", "url": "https://nature.com/articles/123", "source_type": "web"}
    assert is_serious_citation(serious_cite) is True


def test_finalize_article_reconciles_and_caps_hundreds_of_citations():
    # Simulate an article where 50 candidates were collected, but only 2 cited in text
    large_citations = [
        {
            "title": f"Scientific Benchmark Study #{i}",
            "url": f"https://arxiv.org/abs/260{i:03d}",
            "author": f"Dr. Scientist {i}",
            "publication_date": "2026",
            "source_type": "journal",
            "content": f"Substantive research findings on topic #{i}.",
            "authority_score": 0.8,
            "relevance_score": 0.9,
        }
        for i in range(1, 51)
    ]

    article_html = """
<p>Distributed key-value stores face latency spikes under heavy replication [10].</p>
<p><strong>The Big Picture:</strong> Consensus bottlenecks determine transaction throughput [25].</p>
<p><strong>At a Glance:</strong></p>
<ul>
  <li><strong>The Big Shift:</strong> Migration from Raft to Paxos variants.</li>
  <li><strong>Why It Matters:</strong> Lowers tail latency by 45%.</li>
  <li><strong>The Bottom Line:</strong> Verify network topology before re-architecting.</li>
</ul>
<h2>Frequently Asked Questions</h2>
<h3>What causes replication lag?</h3><p>Network latency and serialization overhead.</p>
"""
    result = {
        "structure": {"title": "Why Fast Distributed Storage Fails", "keywords": ["distributed storage", "consensus"]},
        "content": {"content": article_html, "word_count": 550, "sections": []},
        "citations": large_citations,
        # the mocked model writes ~100 words, so that is the length this fixture
        # actually claims; the publish gate compares the finished HTML to it
        "research_data": {"include_in_text_citations": True, "articleLength": 100},
    }

    finalized = tasks._finalize_article(result)
    article = finalized["article"]

    # Out of 50 candidates, only the 2 actually cited ([10] and [25]) should be retained!
    assert article["citations_count"] == 2
    assert len(article["citations"]) == 2
    # Both cited sources survive and nothing else does. The order follows the
    # document, and the document opens with the lead paragraph (which carries the
    # [25] marker), so #25 is numbered first.
    assert {c["title"] for c in article["citations"]} == {
        "Scientific Benchmark Study #10",
        "Scientific Benchmark Study #25",
    }
    assert article["citations"][0]["title"] == "Scientific Benchmark Study #25"

    html = article["htmlArticle"]
    assert '<a href="#ref-1" class="citation-link">[1]</a>' in html
    assert '<a href="#ref-2" class="citation-link">[2]</a>' in html
    assert '<p id="ref-1"><strong>[1]</strong>' in html
    assert '<p id="ref-2"><strong>[2]</strong>' in html
    # Ensure there are NO phantom references past [2]
    assert '<p id="ref-3">' not in html
    assert '<p id="ref-10">' not in html
    assert '<p id="ref-50">' not in html

