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


def test_normalize_job_text_converts_full_jtbd_to_readable_summary():
    assert (
        ResearchJobService._normalize_job_text(
            "I need to when I am drowning in back-to-back video calls, I want to compare AI meeting assistants, so I can choose the one that best syncs action items to my 2026 task manager."
        )
        == "I need to compare AI meeting assistants"
    )


def test_normalize_job_text_simplifies_prompt_engineering_style_phrasing():
    assert (
        ResearchJobService._normalize_job_text(
            "I need to when I am researching a new industry trend, I want to cross-reference multiple research papers using an AI chain, so I can identify the consensus and conflicting viewpoints quickly."
        )
        == "I need to compare research papers with AI"
    )


def test_normalize_job_text_simplifies_feed_the_ai_phrasing():
    assert (
        ResearchJobService._normalize_job_text(
            "I need to when I am drafting a professional report in 2026, I want to feed the AI my previous work samples, so I can generate a first draft that matches my unique writing style."
        )
        == "I need to use AI with my previous work samples"
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


def test_generate_jobs_retries_when_first_pass_returns_too_few_jobs():
    fake_llm = _FakeLLMService(
        [
            {
                "jobs": [
                    {
                        "job_text": "When I am in back-to-back meetings, I want to compare AI meeting assistants, so I can keep my follow-ups organized.",
                        "job_type_hint": "software",
                        "generation_metadata": {"why": "first pass"},
                    }
                ]
            },
            {
                "jobs": [
                    {
                        "job_text": "I need to compare AI meeting notes apps",
                        "job_type_hint": "software",
                        "generation_metadata": {"why": "retry"},
                    },
                    {
                        "job_text": "I need to choose an AI meeting assistant for task sync",
                        "job_type_hint": "software",
                        "generation_metadata": {"why": "retry"},
                    },
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
                context={"focus_area": "AI meeting assistants"},
                count=4,
                negative_context={},
            )
        )
    finally:
        research_job_service_module.llm_service = original_llm

    assert len(fake_llm.prompts) == 2
    assert len(result) == 3
    assert result[0]["job_text"] == "I need to compare AI meeting assistants"
    assert result[1]["job_source"] == "llm_generation_retry"


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


def test_normalize_generated_jobs_preserves_jtbd_statement_in_metadata():
    service = ResearchJobService()

    result = service._normalize_generated_jobs(
        [
            {
                "job_text": "When I am drowning in back-to-back video calls, I want to compare AI meeting assistants, so I can choose the one that best syncs action items to my 2026 task manager.",
                "generation_metadata": {},
            }
        ]
    )

    assert result[0]["job_text"] == "I need to compare AI meeting assistants"
    assert result[0]["generation_metadata"]["jtbd_statement"] == (
        "When I am drowning in back-to-back video calls, I want to compare AI meeting assistants, "
        "so I can choose the one that best syncs action items to my 2026 task manager."
    )


def test_build_generate_jobs_prompt_adds_daily_life_guardrails():
    service = ResearchJobService()

    prompt = service._build_generate_jobs_prompt(
        context={
            "website_description": "Use AI, automation, and better personal workflows to save time, reduce admin, and get more done.",
            "secondary_category_name": "AI for Daily Work",
            "secondary_category_description": "This category is about using AI and tools in real daily life so people augments their capabilities at home and at work.",
            "focus_area": "Prompt Engineering and Context Management for daily work",
        },
        count=8,
        negative_notes={},
        current_date="May 13, 2026",
        current_year=2026,
        relaxed_overlap=False,
    )

    assert "Prioritize real-life automation and everyday workflow needs" in prompt
    assert "Avoid drifting into enterprise, analyst, consultant, or academic-language tasks" in prompt
    assert "translate that into the human outcome the person wants in daily life" in prompt
