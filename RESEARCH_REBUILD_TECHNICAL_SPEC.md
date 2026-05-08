# Research Rebuild Technical Spec

## Purpose

This document is the concise technical specification for the Research rebuild.

It translates the roadmap and decision log into implementation-oriented guidance for engineering. It is intentionally narrower than `RESEARCH_REBUILD_EXECUTION_ROADMAP.md` and should be used as the baseline technical reference during design and build work.

## Scope

This spec covers:

- source-of-truth model
- core entities
- validation logic
- achievability scoring
- software feasibility
- validation freshness
- rejection feedback
- internal-link fit discovery
- downstream compatibility requirements

This spec does not attempt to define every UI detail or every API shape exhaustively.

## System Goal

Replace the current topic-first, heuristic-heavy research model with a validation-first workflow:

1. website and category context
2. user jobs
3. opportunity candidates
4. validation and scoring
5. routing
6. idea generation
7. final keyword pack gating
8. downstream handoff

## Source-of-Truth Strategy

Use a hybrid model.

### New Source-of-Truth Objects

The new Research system should persist dedicated records for:

- user jobs
- opportunity candidates
- validation runs
- SERP snapshots
- routing decisions
- keyword packs

### Compatibility Surfaces

The following existing objects remain important during migration:

- `research_topics`
- `content_ideas`
- `Titles`

These are not the preferred source of truth for discovery and validation logic in the rebuilt system, but they must continue to support:

- Content Studio
- article publishing flows
- WordPress export flows
- historical compatibility

## Core Entity Model

The following logical entities should exist, whether as separate tables or equivalent persisted models.

### `research_user_jobs`

Purpose:

- first-class representation of user jobs derived from website and category context

Required fields:

- `id`
- `project_id`
- `primary_category_id`
- `secondary_category_id`
- `website_context_snapshot`
- `job_text`
- `job_type_hint`
- `job_source`
- `generation_metadata`
- `status`
- `created_at`
- `updated_at`

Suggested statuses:

- `draft`
- `approved`
- `rejected`
- `archived`

### `research_opportunity_candidates`

Purpose:

- concrete opportunities derived from user jobs

Required fields:

- `id`
- `user_job_id`
- `candidate_type`
- `candidate_text`
- `candidate_metadata`
- `status`
- `created_at`
- `updated_at`

Suggested candidate types:

- `seo_article`
- `software`
- `editorial`

### `research_validation_runs`

Purpose:

- persisted validation result for a candidate

Required fields:

- `id`
- `candidate_id`
- `validated_at`
- `expires_at`
- `freshness_state`
- `intent_match_score`
- `serp_weakness_score`
- `serp_gap_score`
- `software_pattern_score`
- `feasibility_score`
- `monetization_fit_score`
- `volume_score`
- `kd_ease_score`
- `niche_drift_score`
- `eligibility_passed`
- `achievability_score`
- `validation_reason_codes`
- `validation_metadata`
- `created_at`
- `updated_at`

Suggested freshness states:

- `fresh`
- `stale`
- `expired`

### `research_serp_snapshots`

Purpose:

- persisted Top 10 evidence used to score and explain candidates

Required fields:

- `id`
- `candidate_id`
- `validation_run_id`
- `query_text`
- `snapshot_source`
- `validated_at`
- `top_results_json`
- `serp_summary_json`
- `created_at`

Each top result should ideally include:

- title
- snippet
- domain
- url
- page_type
- authority hint if available
- weak/strong classification hint

### `research_routing_decisions`

Purpose:

- final interpretation of what a candidate should become

Required fields:

- `id`
- `candidate_id`
- `validation_run_id`
- `route`
- `route_reason_codes`
- `route_metadata`
- `created_at`
- `updated_at`

Suggested routes:

- `article_ready`
- `software_ready`
- `article_plus_software`
- `editorial_only`
- `software_backlog_low_feasibility`
- `needs_more_keyword_validation`
- `rejected_low_achievability`

### `research_keyword_packs`

Purpose:

- final keyword handoff object for content generation and Content Studio readiness

Required fields:

- `id`
- `candidate_id`
- `validation_run_id`
- `primary_keyword`
- `secondary_keywords_json`
- `keyword_metrics_json`
- `keyword_pack_status`
- `keyword_pack_reason_codes`
- `created_at`
- `updated_at`

Suggested keyword pack statuses:

- `ready`
- `cluster_too_thin`
- `needs_more_keyword_validation`

## Discovery Flow

### Inputs

Job discovery must use:

- website description
- project name or domain
- primary category
- secondary category
- category descriptions
- target audience context
- trend context when available

### Output

Generate 20 to 40 category-specific user jobs.

Jobs should:

- be concrete
- be action-oriented
- reflect actual user intent
- be specific enough to later support SEO, editorial, or software routing

Jobs should not:

- be broad topic lanes
- read like abstract content themes
- be framed primarily as titles

## Candidate Generation

Candidates should be derived from jobs, not from broad topics alone.

### Expected Candidate Families

- SEO-shaped search opportunities
- software/workflow/tool opportunities
- editorial-only opportunities

### Negative Feedback Reuse

Rejected jobs and candidates must persist structured rejection signals.

These signals should be reusable as negative context in later generation cycles.

## Validation Logic

Validation should happen before idea generation.

### Required Subscores

The engine should compute:

- `intent_match_score`
- `serp_weakness_score`
- `serp_gap_score`
- `software_pattern_score`
- `feasibility_score`
- `monetization_fit_score`
- `volume_score`
- `kd_ease_score`
- `niche_drift_score`

### Article Eligibility Gates

Article candidates should usually require:

- `intent_match_score >= 0.65`
- `serp_weakness_score >= 0.35`
- at least one measurable exact or near-exact keyword
- at least three supporting secondary keywords by final handoff
- no severe domain mismatch

### Software Eligibility Gates

Software candidates should usually require:

- `intent_match_score >= 0.70`
- `software_pattern_score >= 0.60`
- `serp_gap_score >= 0.30`
- `feasibility_score >= 0.60`
- one primary workflow keyword
- at least two supporting workflow keywords

### Editorial Eligibility Gates

Editorial candidates bypass SEO achievability gates and should instead be validated through strategic fit.

## Achievability Scoring

### Article Formula

```text
achievability_article =
  0.40 * serp_weakness
  + 0.25 * intent_match
  + 0.15 * kd_ease
  + 0.10 * monetization_fit
  + 0.10 * volume_score
  - 0.20 * niche_drift
```

### Software Formula

```text
achievability_software =
  0.25 * intent_match
  + 0.20 * serp_gap
  + 0.20 * software_pattern_score
  + 0.20 * feasibility_score
  + 0.10 * monetization_fit
  + 0.05 * volume_score
```

### Scoring Priorities

For this product, the system should prefer:

- low competition over raw volume
- strong route fit over generic opportunity
- evidence-backed opportunities over heuristic-only “high potential”

## Software Feasibility

Software ideas require a real feasibility gate.

### v1 Feasibility Model

Use a rule-based system with the following buckets:

- `client_only_feasible`
- `light_persistence_feasible`
- `backend_required`
- `external_api_required`
- `infra_heavy`

### Default Interpretation

- `client_only_feasible` -> usually eligible for `software_ready`
- `light_persistence_feasible` -> usually eligible for `software_ready`
- `backend_required` -> valid but score should decrease
- `external_api_required` -> likely backlog unless explicitly prioritized
- `infra_heavy` -> likely backlog unless explicitly prioritized

### Feasibility UI Requirement

Users should be able to inspect why `feasibility_score` was assigned.

## Validation Freshness

SERP evidence is time-sensitive.

### Required Fields

- `validated_at`
- `expires_at`
- `freshness_state`

### TTL Rule

Use category-sensitive TTL.

Default:

- 14 days for volatile categories
- 30 days for evergreen categories

### Behavior

If validation is stale or expired:

- block new idea generation by default
- prompt the user to refresh validation
- allow policy-based override only if explicitly designed

## Cost Control and Rate Limiting

Validation can become expensive.

### Required Operational Safeguards

- batch limits
- staged validation
- concurrency caps
- retry/backoff
- early stopping for low-potential candidates

### Default Cost-Control Strategy

1. generate and classify jobs cheaply
2. rank jobs coarsely
3. validate highest-potential candidates first
4. defer or skip weak candidates

The system should not assume every job must receive full SERP validation immediately.

## Routing Rules

Routing should happen after validation.

### Default Routing Behavior

- strong article fit -> `article_ready`
- strong software fit -> `software_ready`
- strong fit for both -> `article_plus_software`
- strong strategic fit but weak SEO -> `editorial_only`
- search-attractive but low-feasibility software -> `software_backlog_low_feasibility`
- incomplete evidence -> `needs_more_keyword_validation`
- weak candidate -> `rejected_low_achievability`

## Generation Rules

Idea generation must consume:

- validated opportunity context
- SERP evidence
- route
- keyword evidence

The system should not force one idea from every cluster or every candidate.

Allowed outcomes:

- zero ideas from weak candidates
- one idea from a normal strong candidate
- multiple ideas from an unusually strong validated opportunity when justified

## Final Keyword Pack Gating

Before handoff to Content Studio, an idea should have:

- one primary keyword
- three to eight secondary keywords
- measurable support for at least three secondaries
- secondaries aligned to the same user job/intention

If this fails, the opportunity should not be treated as ready.

## Rejection Feedback

Use a controlled vocabulary plus optional free text.

### Recommended Controlled Tags

- `too_broad`
- `off_brand`
- `wrong_audience`
- `weak_serp`
- `low_monetization`
- `technically_impossible`
- `duplicate_of_existing_content`
- `poor_internal_link_fit`
- `low_confidence_keyword_support`
- `not_a_priority_right_now`

### Reuse Rule

Rejection tags should feed later prompt cycles as negative context.

## Internal-Link Fit Discovery

Internal-link fit should be discovered before Content Studio handoff.

### v1 Strategy

Use a progressive hybrid approach:

- start with current imported WordPress content
- infer likely parent/child relationships from titles, links, and shallow metadata
- persist likely internal-link neighborhoods
- upgrade later when richer imported article snapshots exist

### Output

Each opportunity should ideally include:

- parent candidates
- child candidates
- site-architecture fit hints

## Downstream Compatibility Requirements

The rebuild must preserve these downstream expectations during migration:

- `topic_id` continuity where required
- `source_idea_id` continuity where required
- `idea_metadata.category_context` continuity
- Content Studio category path continuity
- WordPress export category behavior

The new Research system may produce richer upstream objects, but it must not silently break existing publishing and handoff behavior.

## UI Traceability Requirement

Scores must be explainable.

At minimum, users should be able to inspect:

- why `serp_weakness_score` is high or low
- why `feasibility_score` is high or low
- why `niche_drift_score` is high or low
- why a candidate was routed or rejected

If a score is not inspectable, it is too opaque for this system.

## Non-Goals for v1

The following are not required to complete the first successful migration:

- perfect SERP page-type classification
- full historical backfill of all old research data
- removal of all legacy tables immediately
- advanced win-rate feedback calibration

## Definition of Technical Readiness

The system is technically ready for cutover when:

- jobs are the upstream object
- candidates are validated before generation
- validation includes freshness, feasibility, and traceability
- routing is explicit
- final keyword packs gate readiness
- downstream compatibility remains intact
