# Research Rebuild Service Interfaces

## Purpose

This document sketches the first backend service module interfaces for the Research rebuild.

These are not final implementations. They are intended to give engineering a stable starting point for:

- service boundaries
- method responsibilities
- input/output shapes
- orchestration flow

The examples below assume Python service modules under `src/services/`.

## Design Principles

- keep services narrow and composable
- separate generation, validation, routing, and compatibility concerns
- prefer plain dict payloads or typed dataclasses at boundaries
- keep persistence orchestration outside endpoint modules
- keep endpoint modules thin

## Shared Types

These are suggested logical shapes, not strict code requirements.

### `WebsiteContext`

```python
WebsiteContext = {
    "user_id": str,
    "project_id": str,
    "project_name": str | None,
    "website_description": str | None,
    "primary_category_id": str | None,
    "secondary_category_id": str | None,
    "primary_category_name": str | None,
    "secondary_category_name": str | None,
    "primary_category_description": str | None,
    "secondary_category_description": str | None,
    "target_audience": str | None,
    "trend_titles": list[str],
}
```

### `UserJobRecord`

```python
UserJobRecord = {
    "id": str,
    "user_id": str,
    "project_id": str,
    "job_text": str,
    "job_type_hint": str | None,
    "status": str,
    "generation_metadata": dict,
}
```

### `OpportunityCandidateRecord`

```python
OpportunityCandidateRecord = {
    "id": str,
    "user_job_id": str,
    "candidate_type": str,
    "candidate_text": str,
    "status": str,
    "candidate_metadata": dict,
}
```

### `ValidationResult`

```python
ValidationResult = {
    "candidate_id": str,
    "validated_at": str,
    "expires_at": str | None,
    "freshness_state": str,
    "eligibility_passed": bool,
    "scores": {
        "intent_match_score": float | None,
        "serp_weakness_score": float | None,
        "serp_gap_score": float | None,
        "software_pattern_score": float | None,
        "feasibility_score": float | None,
        "monetization_fit_score": float | None,
        "volume_score": float | None,
        "kd_ease_score": float | None,
        "niche_drift_score": float | None,
        "achievability_score": float | None,
    },
    "validation_reason_codes": list[str],
    "validation_metadata": dict,
}
```

### `RoutingDecision`

```python
RoutingDecision = {
    "candidate_id": str,
    "route": str,
    "route_reason_codes": list[str],
    "route_metadata": dict,
}
```

## Service 1: `research_job_service.py`

Responsibilities:

- generate jobs from website/category context
- persist jobs
- approve/reject/archive jobs
- capture rejection tags and free-text feedback

Suggested interface:

```python
class ResearchJobService:
    async def generate_jobs(
        self,
        context: dict,
        count: int = 30,
        negative_context: dict | None = None,
    ) -> list[dict]:
        ...

    def save_jobs(
        self,
        *,
        user_id: str,
        project_id: str,
        primary_category_id: str | None,
        secondary_category_id: str | None,
        website_context_snapshot: dict,
        jobs: list[dict],
    ) -> list[dict]:
        ...

    def list_jobs(
        self,
        *,
        user_id: str,
        project_id: str,
        primary_category_id: str | None = None,
        secondary_category_id: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        ...

    def approve_job(self, *, job_id: str, user_id: str) -> dict:
        ...

    def reject_job(
        self,
        *,
        job_id: str,
        user_id: str,
        rejection_reason_tags: list[str],
        rejection_reason_free_text: str | None = None,
    ) -> dict:
        ...

    def build_negative_context(
        self,
        *,
        user_id: str,
        project_id: str,
    ) -> dict:
        ...
```

## Service 2: `research_candidate_service.py`

Responsibilities:

- derive candidates from jobs
- classify them into article/software/editorial
- persist candidates

Suggested interface:

```python
class ResearchCandidateService:
    async def derive_candidates_from_job(
        self,
        *,
        job: dict,
        website_context: dict,
    ) -> list[dict]:
        ...

    def save_candidates(
        self,
        *,
        user_id: str,
        project_id: str,
        user_job_id: str,
        candidates: list[dict],
    ) -> list[dict]:
        ...

    def list_candidates(
        self,
        *,
        user_id: str,
        project_id: str,
        user_job_id: str | None = None,
        candidate_type: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        ...

    def reject_candidate(
        self,
        *,
        candidate_id: str,
        user_id: str,
        rejection_reason_tags: list[str],
        rejection_reason_free_text: str | None = None,
    ) -> dict:
        ...
```

## Service 3: `research_validation_service.py`

Responsibilities:

- validate candidates
- fetch keyword metrics and SERP evidence
- compute scores
- write validation runs
- manage freshness

Suggested interface:

```python
class ResearchValidationService:
    async def validate_candidate(
        self,
        *,
        candidate: dict,
        website_context: dict,
        force_refresh: bool = False,
    ) -> dict:
        ...

    async def validate_candidates(
        self,
        *,
        candidates: list[dict],
        website_context: dict,
        max_candidates: int | None = None,
    ) -> list[dict]:
        ...

    def get_latest_validation(
        self,
        *,
        candidate_id: str,
        user_id: str,
    ) -> dict | None:
        ...

    def is_validation_stale(
        self,
        *,
        validation_run: dict,
        ttl_days: int,
    ) -> bool:
        ...

    def persist_validation(
        self,
        *,
        user_id: str,
        project_id: str,
        candidate_id: str,
        validation_result: dict,
    ) -> dict:
        ...
```

Important internal helpers:

- score calculator
- SERP snapshot collector
- TTL policy resolver
- cost-control limiter
- rate-limit batch planner

## Service 4: `research_feasibility_service.py`

Responsibilities:

- evaluate software feasibility
- return explainable feasibility score and category

Suggested interface:

```python
class ResearchFeasibilityService:
    def score_software_feasibility(
        self,
        *,
        candidate: dict,
        website_context: dict,
    ) -> dict:
        ...
```

Suggested output:

```python
{
    "feasibility_score": 0.72,
    "feasibility_bucket": "client_only_feasible",
    "reason_codes": ["no_backend_required", "no_external_api_required"],
    "metadata": {},
}
```

## Service 5: `research_routing_service.py`

Responsibilities:

- interpret validation and choose a route

Suggested interface:

```python
class ResearchRoutingService:
    def route_candidate(
        self,
        *,
        candidate: dict,
        validation_run: dict,
    ) -> dict:
        ...

    def persist_route(
        self,
        *,
        user_id: str,
        project_id: str,
        candidate_id: str,
        validation_run_id: str,
        routing_decision: dict,
    ) -> dict:
        ...
```

## Service 6: `research_generation_service.py`

Responsibilities:

- generate routed outcomes from validated candidates
- bridge them into `content_ideas`

Suggested interface:

```python
class ResearchGenerationService:
    async def generate_from_candidate(
        self,
        *,
        candidate: dict,
        validation_run: dict,
        routing_decision: dict,
        website_context: dict,
    ) -> list[dict]:
        ...

    def persist_generated_outcomes(
        self,
        *,
        user_id: str,
        project_id: str,
        candidate_id: str,
        validation_run_id: str | None,
        routing_decision_id: str | None,
        outcomes: list[dict],
    ) -> list[dict]:
        ...

    def bridge_outcome_to_content_idea(
        self,
        *,
        outcome: dict,
        compatibility_context: dict,
    ) -> dict:
        ...
```

## Service 7: `research_keyword_pack_service.py`

Responsibilities:

- attach primary and secondary keywords
- enforce keyword-pack readiness

Suggested interface:

```python
class ResearchKeywordPackService:
    def build_keyword_pack(
        self,
        *,
        candidate: dict,
        validation_run: dict,
    ) -> dict:
        ...

    def persist_keyword_pack(
        self,
        *,
        user_id: str,
        project_id: str,
        candidate_id: str,
        validation_run_id: str,
        keyword_pack: dict,
    ) -> dict:
        ...
```

## Service 8: `research_internal_link_fit_service.py`

Responsibilities:

- discover likely parent/child/hub content candidates

Suggested interface:

```python
class ResearchInternalLinkFitService:
    def find_link_candidates(
        self,
        *,
        user_id: str,
        project_id: str,
        candidate: dict,
        validation_run: dict | None = None,
    ) -> list[dict]:
        ...

    def persist_link_candidates(
        self,
        *,
        user_id: str,
        project_id: str,
        candidate_id: str,
        validation_run_id: str | None,
        link_candidates: list[dict],
    ) -> list[dict]:
        ...
```

## Service 9: `research_compatibility_adapter_service.py`

Responsibilities:

- adapt the new research model to existing publishing flows

Suggested interface:

```python
class ResearchCompatibilityAdapterService:
    def build_content_idea_payload(
        self,
        *,
        candidate: dict,
        validation_run: dict,
        routing_decision: dict,
        keyword_pack: dict | None,
        website_context: dict,
    ) -> dict:
        ...

    def ensure_legacy_metadata_contracts(
        self,
        *,
        content_idea_payload: dict,
        website_context: dict,
    ) -> dict:
        ...
```

## Suggested Endpoint Module Shape

Suggested module:

```python
src/api/endpoints/research_rebuild.py
```

Suggested endpoint families:

- jobs
- candidates
- validations
- routes
- outcomes
- keyword packs
- internal links

The endpoint module should:

- authenticate the user
- validate inputs
- call services
- return normalized payloads

The endpoint module should not:

- contain scoring logic
- contain prompt-building logic
- contain persistence orchestration beyond service calls

## Suggested Orchestration Flow

### Happy Path

```text
website context
  -> generate jobs
  -> approve jobs
  -> derive candidates
  -> validate candidates
  -> route candidates
  -> generate outcomes
  -> build keyword packs
  -> discover internal-link fit
  -> bridge to content_ideas
```

## Suggested First Repository Layer

If you want to keep service code cleaner, add lightweight repository helpers such as:

- `research_job_repo.py`
- `research_candidate_repo.py`
- `research_validation_repo.py`
- `research_serp_snapshot_repo.py`
- `research_routing_repo.py`
- `research_keyword_pack_repo.py`
- `research_internal_link_repo.py`
- `research_generated_outcome_repo.py`

This is optional but recommended if the service files start growing too quickly.

## First Implementation Rule

Do not try to solve every edge case in the first pass.

The first implementation should focus on:

- clean upstream job model
- explicit validation persistence
- explicit routing
- compatibility with existing downstream objects

That gives the team a stable base to iterate on later. 
