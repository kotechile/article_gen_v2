from src.services.research_job_service import ResearchJobService
from src.services import research_job_service as research_job_service_module


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


class _FakeLLMService:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    async def generate_json(self, prompt, task_role=None, max_tokens=None):
        self.prompts.append(prompt)
        return self._responses.pop(0)


def test_generate_jobs_retries_with_relaxed_overlap_when_first_pass_is_empty():
    fake_llm = _FakeLLMService(
        [
            {"jobs": []},
            {
                "jobs": [
                    {
                        "job_text": "Choosing the right AI budget app for shared vacation planning",
                        "job_type_hint": "hybrid",
                        "generation_metadata": {"why": "focused variant"},
                    }
                ]
            },
        ]
    )
    service = ResearchJobService()

    original_llm = research_job_service_module.llm_service
    research_job_service_module.llm_service = fake_llm
    try:
        result = __import__("asyncio").run(
            service.generate_jobs(
                context={"focus_area": "vacation budget apps"},
                count=5,
                negative_context={"recent_existing_jobs": [{"job_text": "I need to plan a vacation budget"}]},
            )
        )
    finally:
        research_job_service_module.llm_service = original_llm

    assert len(fake_llm.prompts) == 2
    assert "Use previous jobs only to avoid near-exact duplicates." in fake_llm.prompts[1]
    assert result[0]["job_text"] == "I need to choose the right AI budget app for shared vacation planning"
    assert result[0]["job_source"] == "llm_generation_retry"


def test_normalize_generated_jobs_keeps_search_seeds_and_intent_type():
    service = ResearchJobService()

    result = service._normalize_generated_jobs(
        [
            {
                "job_text": "When I am planning a trip, I want to compare vacation budget apps, so I can stay on budget.",
                "generation_metadata": {
                    "intent_type": "Transactional",
                    "search_seeds": ["vacation budget app", "trip budget planner", "travel cost calculator"],
                },
            }
        ]
    )

    assert result[0]["generation_metadata"]["intent_type"] == "transactional"
    assert result[0]["generation_metadata"]["search_seeds"] == [
        "vacation budget app",
        "trip budget planner",
        "travel cost calculator",
    ]
