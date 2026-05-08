# Research Rebuild Backend Implementation Plan

## Purpose

This document is the first-pass backend implementation plan for the Research rebuild.

It maps the target-state workflow into:

- backend services
- endpoint groups
- persistence responsibilities
- rollout order
- compatibility requirements

It should be read together with:

- `RESEARCH_REBUILD_EXECUTION_ROADMAP.md`
- `RESEARCH_REBUILD_TECHNICAL_SPEC.md`
- `RESEARCH_REBUILD_SCHEMA_PROPOSAL.md`
- `migrations/add_research_rebuild_tables.sql`
- `migrations/add_research_rebuild_rls.sql`
- `RESEARCH_REBUILD_SERVICE_INTERFACES.md`

## Backend Objectives

The backend rebuild should:

1. make user jobs the upstream discovery object
2. validate opportunities before idea generation
3. persist explainable scores and SERP evidence
4. route candidates explicitly into article, software, editorial, or rejection states
5. preserve compatibility with `research_topics`, `content_ideas`, and `Titles`

## Implementation Artifacts

The first-pass backend foundation for this rebuild now has dedicated artifacts for:

- base schema creation in `migrations/add_research_rebuild_tables.sql`
- Supabase-ready row-level security in `migrations/add_research_rebuild_rls.sql`
- service/module boundaries in `RESEARCH_REBUILD_SERVICE_INTERFACES.md`

Recommended usage:

1. apply the base schema migration first
2. apply the RLS migration immediately after
3. use the service-interface document to shape the first backend PRs

## Recommended Service Boundaries

### 1. `research_job_service`

Responsibilities:

- build user jobs from website and category context
- persist generated jobs
- update job status
- store rejection tags and free-text feedback
- prepare negative-context inputs for future generation

Primary tables:

- `research_user_jobs`

Likely inputs:

- project context
- category context
- target audience
- trend context

### 2. `research_candidate_service`

Responsibilities:

- derive opportunity candidates from approved jobs
- classify candidates into article, software, or editorial families
- normalize candidate text
- persist candidate records

Primary tables:

- `research_opportunity_candidates`

### 3. `research_validation_service`

Responsibilities:

- run validation on candidates
- fetch keyword metrics
- fetch or persist SERP evidence
- calculate achievability subscores
- calculate freshness state
- apply eligibility gates
- persist validation runs

Primary tables:

- `research_validation_runs`
- `research_serp_snapshots`

Important notes:

- this service should own TTL and staleness logic
- this service should own rate limiting and cost-control orchestration

### 4. `research_feasibility_service`

Responsibilities:

- evaluate software feasibility using the agreed rule-based buckets
- provide an explainable feasibility score
- classify software candidates into:
  - ready
  - low-feasibility backlog
  - reject

Primary tables:

- `research_validation_runs`

Important notes:

- this can be implemented as a dedicated helper inside validation initially
- if complexity grows, it should become its own service

### 5. `research_routing_service`

Responsibilities:

- interpret validation results
- apply routing rules
- persist routing decision
- emit route reason codes

Primary tables:

- `research_routing_decisions`

### 6. `research_generation_service`

Responsibilities:

- generate article ideas from validated SEO opportunities
- generate software ideas from validated software opportunities
- generate editorial ideas from editorial-only candidates
- write generated outcome mappings
- persist or bridge into `content_ideas`

Primary tables:

- `research_generated_outcomes`
- `content_ideas`

Important notes:

- this service should consume validation and routing output, not raw broad topics
- this service should not force one idea per cluster

### 7. `research_keyword_pack_service`

Responsibilities:

- assign final primary and secondary keywords
- enforce readiness thresholds
- persist keyword packs
- block thin candidates from appearing ready

Primary tables:

- `research_keyword_packs`

### 8. `research_internal_link_fit_service`

Responsibilities:

- identify parent, child, sibling, and hub candidates from imported WordPress content
- persist internal-link fit suggestions

Primary tables:

- `research_internal_link_candidates`
- `wordpress_imported_posts`

Important notes:

- v1 can use shallow title/link/excerpt data
- this should be upgradeable later if imported content gets richer

### 9. `research_compatibility_adapter_service`

Responsibilities:

- bridge the new Research model into:
  - `research_topics`
  - `content_ideas`
  - `Titles`
- preserve category-context metadata
- preserve downstream expectations during migration

Primary tables:

- new research rebuild tables
- `research_topics`
- `content_ideas`
- `Titles`

## Recommended Endpoint Families

These are suggested backend endpoint groups, not final route contracts.

### Job Endpoints

Suggested operations:

- create/generate jobs
- list jobs
- approve job
- reject job
- archive job
- regenerate jobs

### Candidate Endpoints

Suggested operations:

- generate candidates from jobs
- list candidates
- reject candidate
- archive candidate

### Validation Endpoints

Suggested operations:

- validate selected candidates
- refresh stale validation
- get latest validation for candidate
- list validation history
- get SERP evidence breakdown

### Routing Endpoints

Suggested operations:

- compute route for candidate
- list routed candidates
- update route if policy allows

### Generation Endpoints

Suggested operations:

- generate article ideas from routed candidates
- generate software ideas from routed candidates
- generate editorial ideas from routed candidates
- persist outcome to `content_ideas`

### Keyword Pack Endpoints

Suggested operations:

- compute keyword pack
- refresh keyword pack
- list keyword pack status

### Internal-Link Endpoints

Suggested operations:

- compute internal-link fit
- fetch internal-link candidates for a selected opportunity

## Rollout Order for Backend Work

### Step 1: Schema and Repository Layer

Deliver:

- migration file
- repository/helpers for:
  - jobs
  - candidates
  - validations
  - SERP snapshots
  - routing
  - keyword packs
  - internal-link candidates
  - generated outcomes

### Step 2: Job Discovery and Persistence

Deliver:

- `research_job_service`
- job generation endpoint(s)
- job approval/rejection endpoint(s)

### Step 3: Candidate Derivation

Deliver:

- `research_candidate_service`
- candidate generation endpoint(s)
- candidate listing and rejection endpoint(s)

### Step 4: Validation Engine

Deliver:

- `research_validation_service`
- score computation
- SERP persistence
- freshness logic
- cost-control and rate-limit orchestration

### Step 5: Routing Layer

Deliver:

- `research_routing_service`
- routing persistence
- route explanation payloads

### Step 6: Routed Generation

Deliver:

- `research_generation_service`
- route-aware generation entry points
- bridge into `content_ideas`

### Step 7: Keyword Packs and Internal-Link Fit

Deliver:

- `research_keyword_pack_service`
- `research_internal_link_fit_service`
- final readiness output

### Step 8: Compatibility Adapters

Deliver:

- mapping from validated generated outcomes into legacy publishing flows
- stability for Content Studio and WordPress export

## Validation Implementation Notes

### Freshness

The validation service should write:

- `validated_at`
- `expires_at`
- `freshness_state`

Suggested logic:

- derive TTL from category volatility
- mark stale when `now() > expires_at`
- require refresh before generation if validation is stale

### Cost Control

The validation service should not fully validate every generated job by default.

Suggested staged approach:

1. job generation
2. coarse candidate scoring
3. validate top candidates first
4. stop early on obvious failures

### Rate Limiting

Validation orchestration should support:

- batch size limits
- concurrency caps
- retry/backoff
- partial completion reporting

If the current Flask request/response model makes this awkward, validation should move to an async task flow instead of blocking the web request for large workloads.

## Generation Implementation Notes

### Input Contract

The generation layer should receive:

- candidate
- latest fresh validation run
- routing decision
- latest keyword pack if already computed
- SERP evidence summary

### Output Contract

The generation layer should produce:

- generated idea text
- route-specific metadata
- source linkage to candidate and validation
- persistence mapping into `content_ideas`

## Compatibility Notes

### `research_topics`

Recommended role during migration:

- category-aware container or history wrapper
- not the main source of truth for validation logic

### `content_ideas`

Recommended role during migration:

- staging/publishing surface for generated ideas
- should continue to carry:
  - `topic_id`
  - `idea_metadata.category_context`
  - keyword handoff fields where needed

### `Titles`

Recommended role during migration:

- publishing-ready content object for Content Studio and WordPress export

The backend must not break current downstream expectations while the new Research system comes online.

## Suggested Backend File/Module Pattern

This is a suggested shape, not a strict rule.

Suggested modules:

- `src/services/research_job_service.py`
- `src/services/research_candidate_service.py`
- `src/services/research_validation_service.py`
- `src/services/research_routing_service.py`
- `src/services/research_generation_service.py`
- `src/services/research_keyword_pack_service.py`
- `src/services/research_internal_link_fit_service.py`
- `src/services/research_compatibility_adapter_service.py`

Suggested endpoint module:

- `src/api/endpoints/research_rebuild.py`

Important note:

- do not overload the existing `research_topics.py` endpoint file with all new logic if that would make maintenance worse
- a dedicated endpoint module is preferable once the new flow becomes substantial

## Suggested First Backend PR Slices

### Backend Slice 1

- schema migration
- repository helpers
- no UI dependency

### Backend Slice 2

- job generation and persistence
- job approval/rejection endpoints

### Backend Slice 3

- candidate derivation and persistence

### Backend Slice 4

- validation engine v1
- SERP snapshot persistence
- freshness fields

### Backend Slice 5

- routing persistence
- route explanation payloads

### Backend Slice 6

- route-aware generation into `content_ideas`

### Backend Slice 7

- final keyword pack builder
- internal-link fit discovery

### Backend Slice 8

- compatibility and cutover helpers
- legacy clean-up support

## Acceptance Conditions for Backend Readiness

Backend implementation is ready for integrated QA when:

- jobs persist correctly
- candidates persist correctly
- validation writes scores plus SERP evidence
- routing is deterministic and explainable
- generation consumes validated opportunities
- keyword pack gating works
- internal-link fit suggestions persist
- `content_ideas` compatibility is preserved

## Explicit Non-Goals for First Backend Pass

- perfect RLS redesign across all new tables
- full historical migration of all legacy research records
- deep imported WordPress article reconstruction
- complete removal of legacy research endpoints

These can happen later once the new path is stable.
