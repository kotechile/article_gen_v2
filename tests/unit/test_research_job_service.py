from src.services.research_job_service import ResearchJobService


def test_normalize_job_text_removes_framework_prefixes():
    assert (
        ResearchJobService._normalize_job_text(
            "Decision tree: Choosing between an AI-summarized newsletter service and a deep-read app."
        )
        == "I need to choose between an AI-summarized newsletter service and a deep-read app"
    )


def test_normalize_job_text_rewrites_workflow_style_copy():
    assert (
        ResearchJobService._normalize_job_text(
            "Workflow: Using AI to extract and compare 2026 warranty terms for three appliance brands."
        )
        == "I need to use AI to extract and compare 2026 warranty terms for three appliance brands"
    )


def test_normalize_job_text_rewrites_comparison_prefix():
    assert (
        ResearchJobService._normalize_job_text(
            "Comparison: Perplexity Pro vs. SearchGPT for 2026 consumer electronics research."
        )
        == "I need to compare Perplexity Pro vs. SearchGPT for 2026 consumer electronics research"
    )
