# Research Rebuild Schema Proposal

## Purpose

This document proposes a database/schema design for the rebuilt Research workflow.

It is designed to support:

- user-job-first discovery
- validation-first opportunity scoring
- explicit routing into article, software, or editorial outcomes
- freshness-aware SERP evidence
- keyword-pack gating before Content Studio
- compatibility with existing downstream objects

This is a proposal, not a final migration script. It should be used to guide PR 3 and related implementation work.

## Design Strategy

Use a hybrid model:

- create new source-of-truth tables for jobs, candidates, validations, SERP evidence, routing, and keyword packs
- preserve current downstream compatibility with:
  - `research_topics`
  - `content_ideas`
  - `Titles`

The rebuilt Research system should not overload legacy tables with too many new responsibilities.

## Current Tables to Preserve

These existing tables are still important and should remain available during migration:

- `research_topics`
- `content_ideas`
- `Titles`
- `wordpress_imported_posts`
- `project_categories`
- `projects`

### Why Preserve Them

- `research_topics` still anchors category-aware research history and existing UI
- `content_ideas` still feeds Content Library and publishing workflows
- `Titles` still feeds Content Studio and WordPress export
- `wordpress_imported_posts` is needed for internal-link fit discovery until a richer import model exists

## Proposed New Tables

### 1. `research_user_jobs`

Purpose:

- first-class representation of user jobs derived from website and category context

Suggested columns:

- `id uuid primary key`
- `user_id uuid not null`
- `project_id uuid not null`
- `primary_category_id uuid null`
- `secondary_category_id uuid null`
- `job_text text not null`
- `job_type_hint text null`
- `job_source text not null default 'llm_generation'`
- `status text not null default 'draft'`
- `website_context_snapshot jsonb not null default '{}'::jsonb`
- `generation_metadata jsonb not null default '{}'::jsonb`
- `rejection_reason_tags jsonb not null default '[]'::jsonb`
- `rejection_reason_free_text text null`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`

Suggested status values:

- `draft`
- `approved`
- `rejected`
- `archived`

Suggested indexes:

- `(user_id, project_id)`
- `(project_id, primary_category_id, secondary_category_id)`
- `(status)`

### 2. `research_opportunity_candidates`

Purpose:

- concrete opportunities derived from approved jobs

Suggested columns:

- `id uuid primary key`
- `user_id uuid not null`
- `project_id uuid not null`
- `user_job_id uuid not null references research_user_jobs(id) on delete cascade`
- `candidate_type text not null`
- `candidate_text text not null`
- `normalized_candidate_text text null`
- `status text not null default 'draft'`
- `candidate_metadata jsonb not null default '{}'::jsonb`
- `source_keywords_json jsonb not null default '[]'::jsonb`
- `rejection_reason_tags jsonb not null default '[]'::jsonb`
- `rejection_reason_free_text text null`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`

Suggested candidate type values:

- `seo_article`
- `software`
- `editorial`

Suggested status values:

- `draft`
- `validated`
- `rejected`
- `archived`

Suggested indexes:

- `(user_job_id)`
- `(user_id, project_id)`
- `(candidate_type, status)`
- `(normalized_candidate_text)`

### 3. `research_validation_runs`

Purpose:

- persisted validation result for each candidate at a point in time

Suggested columns:

- `id uuid primary key`
- `user_id uuid not null`
- `project_id uuid not null`
- `candidate_id uuid not null references research_opportunity_candidates(id) on delete cascade`
- `validation_version text not null`
- `validated_at timestamptz not null default now()`
- `expires_at timestamptz null`
- `freshness_state text not null default 'fresh'`
- `eligibility_passed boolean not null default false`
- `intent_match_score numeric(5,4) null`
- `serp_weakness_score numeric(5,4) null`
- `serp_gap_score numeric(5,4) null`
- `software_pattern_score numeric(5,4) null`
- `feasibility_score numeric(5,4) null`
- `monetization_fit_score numeric(5,4) null`
- `volume_score numeric(5,4) null`
- `kd_ease_score numeric(5,4) null`
- `niche_drift_score numeric(5,4) null`
- `achievability_score numeric(5,4) null`
- `validation_reason_codes jsonb not null default '[]'::jsonb`
- `validation_metadata jsonb not null default '{}'::jsonb`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`

Suggested freshness state values:

- `fresh`
- `stale`
- `expired`

Suggested indexes:

- `(candidate_id, validated_at desc)`
- `(user_id, project_id)`
- `(freshness_state)`
- `(eligibility_passed)`

### 4. `research_serp_snapshots`

Purpose:

- persisted Top 10 SERP evidence tied to a candidate and validation run

Suggested columns:

- `id uuid primary key`
- `user_id uuid not null`
- `project_id uuid not null`
- `candidate_id uuid not null references research_opportunity_candidates(id) on delete cascade`
- `validation_run_id uuid not null references research_validation_runs(id) on delete cascade`
- `query_text text not null`
- `snapshot_source text not null`
- `validated_at timestamptz not null`
- `top_results_json jsonb not null default '[]'::jsonb`
- `serp_summary_json jsonb not null default '{}'::jsonb`
- `created_at timestamptz not null default now()`

Suggested indexes:

- `(candidate_id, validated_at desc)`
- `(validation_run_id)`

### 5. `research_routing_decisions`

Purpose:

- final route chosen for a validated candidate

Suggested columns:

- `id uuid primary key`
- `user_id uuid not null`
- `project_id uuid not null`
- `candidate_id uuid not null references research_opportunity_candidates(id) on delete cascade`
- `validation_run_id uuid not null references research_validation_runs(id) on delete cascade`
- `route text not null`
- `route_reason_codes jsonb not null default '[]'::jsonb`
- `route_metadata jsonb not null default '{}'::jsonb`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`

Suggested route values:

- `article_ready`
- `software_ready`
- `article_plus_software`
- `editorial_only`
- `software_backlog_low_feasibility`
- `needs_more_keyword_validation`
- `rejected_low_achievability`

Suggested indexes:

- `(candidate_id)`
- `(validation_run_id)`
- `(route)`

### 6. `research_keyword_packs`

Purpose:

- final keyword handoff object used before content generation and Content Studio readiness

Suggested columns:

- `id uuid primary key`
- `user_id uuid not null`
- `project_id uuid not null`
- `candidate_id uuid not null references research_opportunity_candidates(id) on delete cascade`
- `validation_run_id uuid not null references research_validation_runs(id) on delete cascade`
- `primary_keyword text null`
- `secondary_keywords_json jsonb not null default '[]'::jsonb`
- `keyword_metrics_json jsonb not null default '{}'::jsonb`
- `keyword_pack_status text not null default 'draft'`
- `keyword_pack_reason_codes jsonb not null default '[]'::jsonb`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`

Suggested keyword pack status values:

- `draft`
- `ready`
- `cluster_too_thin`
- `needs_more_keyword_validation`

Suggested indexes:

- `(candidate_id)`
- `(validation_run_id)`
- `(keyword_pack_status)`

### 7. `research_internal_link_candidates`

Purpose:

- internal-link neighborhood suggestions discovered during validation or keyword-pack assembly

Suggested columns:

- `id uuid primary key`
- `user_id uuid not null`
- `project_id uuid not null`
- `candidate_id uuid not null references research_opportunity_candidates(id) on delete cascade`
- `validation_run_id uuid null references research_validation_runs(id) on delete set null`
- `wordpress_imported_post_id uuid null`
- `link_role text not null`
- `match_score numeric(5,4) null`
- `match_reason_codes jsonb not null default '[]'::jsonb`
- `match_metadata jsonb not null default '{}'::jsonb`
- `created_at timestamptz not null default now()`

Suggested link role values:

- `parent_candidate`
- `child_candidate`
- `sibling_candidate`
- `hub_candidate`

Suggested indexes:

- `(candidate_id)`
- `(link_role)`

### 8. `research_generated_outcomes`

Purpose:

- normalized mapping from validated opportunities into generated idea outputs

Suggested columns:

- `id uuid primary key`
- `user_id uuid not null`
- `project_id uuid not null`
- `candidate_id uuid not null references research_opportunity_candidates(id) on delete cascade`
- `validation_run_id uuid null references research_validation_runs(id) on delete set null`
- `routing_decision_id uuid null references research_routing_decisions(id) on delete set null`
- `content_idea_id uuid null`
- `outcome_type text not null`
- `status text not null default 'draft'`
- `outcome_metadata jsonb not null default '{}'::jsonb`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`

Suggested outcome type values:

- `article`
- `software`
- `editorial`

Suggested indexes:

- `(candidate_id)`
- `(content_idea_id)`
- `(outcome_type, status)`

## Suggested Relationships

### Primary Relationship Flow

```text
projects
  -> research_user_jobs
    -> research_opportunity_candidates
      -> research_validation_runs
        -> research_serp_snapshots
        -> research_routing_decisions
        -> research_keyword_packs
        -> research_internal_link_candidates
      -> research_generated_outcomes
        -> content_ideas
          -> Titles
```

### Compatibility Relationship Notes

- `research_topics` may remain as a category-aware container or history object during migration.
- `research_generated_outcomes.content_idea_id` should be used to connect the new Research model to existing `content_ideas`.
- `content_ideas` remains the staging object for publishing into `Titles`.

## Suggested Minimal JSON Structures

### `website_context_snapshot`

```json
{
  "project_name": "example.com",
  "website_description": "Site description",
  "primary_category_name": "Category A",
  "secondary_category_name": "Category B",
  "primary_category_description": "Description",
  "secondary_category_description": "Description",
  "target_audience": "Audience description",
  "trend_titles": ["Trend A", "Trend B"]
}
```

### `validation_metadata`

```json
{
  "validation_ttl_days": 14,
  "ttl_policy": "category_sensitive",
  "cost_control_tier": "priority_first",
  "rate_limit_bucket": "default",
  "validated_query_count": 6
}
```

### `serp_summary_json`

```json
{
  "weak_result_ratio": 0.4,
  "low_authority_ratio": 0.3,
  "poor_intent_match_ratio": 0.2,
  "stale_result_ratio": 0.1,
  "weak_content_depth_ratio": 0.3,
  "page_type_counts": {
    "ugc": 2,
    "tool": 1,
    "article": 6,
    "product": 1
  }
}
```

### `route_metadata`

```json
{
  "article_score": 0.74,
  "software_score": 0.61,
  "editorial_fit_score": 0.42,
  "selected_route": "article_ready"
}
```

### `keyword_metrics_json`

```json
{
  "primary": {
    "keyword": "example keyword",
    "search_volume": 700,
    "keyword_difficulty": 24,
    "cpc": 1.7
  },
  "secondary": [
    {
      "keyword": "example keyword tool",
      "search_volume": 150,
      "keyword_difficulty": 19,
      "cpc": 2.1
    }
  ]
}
```

## Suggested Enums or Controlled Values

If the database supports enum-like constraints or check constraints, consider enforcing them for:

- `research_user_jobs.status`
- `research_opportunity_candidates.candidate_type`
- `research_opportunity_candidates.status`
- `research_validation_runs.freshness_state`
- `research_routing_decisions.route`
- `research_keyword_packs.keyword_pack_status`
- `research_internal_link_candidates.link_role`
- `research_generated_outcomes.outcome_type`

## Suggested Compatibility Strategy

### Keep Existing Downstream Fields Stable

The new schema should not break the current downstream assumptions that rely on:

- `topic_id`
- `source_idea_id`
- `idea_metadata.category_context`
- current Content Studio article loading logic
- current WordPress export category mapping logic

### Recommended Adapter Pattern

Use adapter logic when creating `content_ideas` from `research_generated_outcomes`:

- map the selected job/opportunity into a category-aware `content_idea`
- keep `topic_id` if `research_topics` still exists as a compatibility anchor
- include new validation metadata inside `idea_metadata`
- include keyword pack metrics in `keyword_metrics`

## Suggested Migration Strategy

### Migration 1

- create new tables
- add indexes
- do not alter old flows yet

### Migration 2

- add adapter paths from new entities into `content_ideas`
- keep old UI working

### Migration 3

- add archival flags or deprecation metadata to old research artifacts if needed

### Migration 4

- remove or simplify obsolete legacy structures only after cutover confidence is high

## Open Questions for Implementation

These should be resolved during detailed design:

- Should `research_topics` remain mandatory, optional, or eventually become a compatibility-only wrapper?
- Should `research_validation_runs` persist one row per candidate refresh, or should there also be a materialized “latest validation” view/table?
- Should `research_internal_link_candidates.wordpress_imported_post_id` point to the current shallow import table now and be migrated later if a richer import table is introduced?
- Do we want separate article/software validation rows, or one shared validation row with candidate-type-specific fields?

## Recommended First Build Order

1. `research_user_jobs`
2. `research_opportunity_candidates`
3. `research_validation_runs`
4. `research_serp_snapshots`
5. `research_routing_decisions`
6. `research_keyword_packs`
7. `research_generated_outcomes`
8. `research_internal_link_candidates`

This order supports the core execution path first, with internal-link discovery added as soon as the candidate and validation layers exist.
