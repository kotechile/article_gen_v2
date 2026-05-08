# Research Rebuild Execution Roadmap

## Purpose

This file is the tracking document for rebuilding the Research workflow from the current topic-first, heuristic-heavy model into a validation-first system centered on user jobs, achievability scoring, and explicit routing into article, editorial, or software outcomes.

This roadmap assumes that significant parts of the current Research implementation will become obsolete and must be cleaned up as part of the migration.

## Target Outcome

The rebuilt Research system should:

1. start from website context and category/subcategory context
2. derive category-specific user jobs
3. split opportunities into SEO, editorial, and software-first tracks
4. validate candidates before idea generation
5. score achievability using SERP evidence and keyword evidence
6. generate ideas only from validated opportunities
7. attach final keyword packs before ideas move into Content Studio
8. retire obsolete cluster-first and heuristic-only flows
9. evaluate software feasibility before software ideas are treated as ready
10. refresh stale validation before acting on old SERP evidence
11. learn from rejected jobs and candidates
12. identify internal-link placement before handoff to Content Studio

## Migration Principles

- Prefer phased migration over big-bang replacement.
- Keep Content Studio and WordPress export contracts stable during the rollout.
- Treat cleanup as part of delivery, not post-delivery maintenance.
- Avoid adding new product logic to paths that are marked for deprecation.
- Preserve category/subcategory context end to end throughout the migration.

## Planning Conventions

This file uses the following planning conventions.

### Owner Labels

- `BE`: backend engineering
- `FE`: frontend engineering
- `DATA`: schema, migrations, backfills, cleanup scripts
- `PM`: product management / requirements / sequencing
- `QA`: manual QA and regression validation

### Effort Bands

- `XS`: 1 to 2 focused engineering days
- `S`: 3 to 5 engineering days
- `M`: 1 to 2 engineering weeks
- `L`: 2 to 4 engineering weeks
- `XL`: multi-week cross-functional effort

These are intentionally rough planning estimates, not commitments.

### Status Values

- `not_started`
- `in_progress`
- `blocked`
- `ready_for_qa`
- `done`

## Milestone Plan

### Milestone 1: Audit and Foundation

Objective:

- establish the migration baseline
- create feature-flag and schema scaffolding

Includes:

- PR 1
- PR 2
- PR 3

Exit criteria:

- all current Research paths are inventoried
- deprecation map exists
- feature flags exist
- schema foundation is merged

### Milestone 2: Job Discovery and Candidate Formation

Objective:

- shift the upstream model from broad topics to user jobs

Includes:

- PR 4
- PR 5
- PR 6
- PR 7

Exit criteria:

- user jobs can be generated, reviewed, classified, and converted into candidates

### Milestone 3: Validation and Scoring

Objective:

- validate opportunities before idea generation

Includes:

- PR 8
- PR 9
- PR 10

Exit criteria:

- candidates have SERP evidence
- achievability scoring is persisted
- routing evidence is visible in the UI

### Milestone 4: Routed Generation and Keyword Gating

Objective:

- generate only from validated opportunities and enforce final keyword support

Includes:

- PR 11
- PR 12
- PR 13
- PR 14
- PR 15

Exit criteria:

- article, software, and editorial outcomes are separated
- final keyword pack gating is active

### Milestone 5: Frontend Migration and Compatibility

Objective:

- move the UI to the new model without breaking in-flight work

Includes:

- PR 16
- PR 17
- PR 18

Exit criteria:

- jobs-first Research flow is usable
- opportunity detail UI is usable
- legacy data is still readable

### Milestone 6: Cutover and Cleanup

Objective:

- make the new path primary and remove obsolete system pieces

Includes:

- PR 19
- PR 20
- PR 21
- PR 22

Exit criteria:

- new Research flow is default
- legacy flow is deprecated or hidden
- cleanup work is complete or explicitly scheduled

## Milestone Tracker

Use this section as the live implementation tracker.

### Milestone 1: Audit and Foundation

- Status: `not_started`
- Target PRs: `1, 2, 3`
- Owners: `PM, BE, FE, DATA`
- Exit criteria:
  - [ ] Research dependency map completed
  - [ ] Deprecation map completed
  - [ ] Feature flags merged
  - [ ] Schema foundation merged
- Notes:
  - none yet

### Milestone 2: Job Discovery and Candidate Formation

- Status: `not_started`
- Target PRs: `4, 5, 6, 7`
- Owners: `BE, FE`
- Exit criteria:
  - [ ] User jobs generated and persisted
  - [ ] Job review UI available
  - [ ] Job classification persisted
  - [ ] Candidates derive from jobs, not only topic titles
- Notes:
  - none yet

### Milestone 3: Validation and Scoring

- Status: `not_started`
- Target PRs: `8, 9, 10`
- Owners: `BE, FE, DATA`
- Exit criteria:
  - [ ] SERP snapshots persisted
  - [ ] `validated_at` and freshness state persisted
  - [ ] Achievability subscores persisted
  - [ ] Validation review UI available
  - [ ] Cost-control and rate-limit strategy implemented or stubbed
- Notes:
  - none yet

### Milestone 4: Routed Generation and Keyword Gating

- Status: `not_started`
- Target PRs: `11, 12, 13, 14, 15`
- Owners: `BE, FE`
- Exit criteria:
  - [ ] SEO article generation uses validated opportunities
  - [ ] Software generation uses validated opportunities
  - [ ] Editorial generation is independent of SEO gates
  - [ ] Final keyword pack builder active
  - [ ] Content Studio readiness gate active
- Notes:
  - none yet

### Milestone 5: Frontend Migration and Compatibility

- Status: `not_started`
- Target PRs: `16, 17, 18`
- Owners: `FE, BE`
- Exit criteria:
  - [ ] Jobs-first landing experience shipped
  - [ ] Opportunity detail view shipped
  - [ ] Legacy compatibility layer shipped
  - [ ] Score traceability is visible in the UI
- Notes:
  - none yet

### Milestone 6: Cutover and Cleanup

- Status: `not_started`
- Target PRs: `19, 20, 21, 22`
- Owners: `PM, BE, FE, DATA, QA`
- Exit criteria:
  - [ ] New flow enabled in staging
  - [ ] New flow enabled in production
  - [ ] Legacy path hidden or deprecated
  - [ ] Cleanup PRs merged or scheduled
- Notes:
  - none yet

## Decision Log

Use this section to track open and resolved architecture or rollout decisions.

### Decision Status Values

- `open`
- `decided`
- `deferred`
- `superseded`

### Decision Template

Copy this template for new decisions:

```text
Decision ID:
Title:
Status:
Date:
Owners:
Context:
Options Considered:
Decision:
Why:
Impact:
Follow-up Work:
```

### Active Decisions

#### Decision ID: D-001

- Title: Whether to extend existing `research_topics` and `content_ideas` tables or introduce parallel source-of-truth tables for new Research objects
- Status: `decided`
- Date: `2026-05-07`
- Owners: `BE, DATA, PM`
- Context:
  - The new Research model needs user jobs, validation runs, routing decisions, SERP snapshots, and keyword packs.
  - Existing tables support the old topic-first workflow and downstream consumers.
- Options Considered:
  - extend current tables heavily
  - create parallel target-state tables with compatibility adapters
  - hybrid approach
- Decision:
  - Use a hybrid approach with new parallel source-of-truth tables for user jobs, validations, SERP snapshots, routing decisions, and keyword packs, while keeping `research_topics`, `content_ideas`, and `Titles` as downstream compatibility surfaces during migration.
- Why:
  - Extending current tables too aggressively would entangle the new model with old assumptions.
  - A fully separate system with no compatibility layer would create unnecessary downstream breakage.
  - The hybrid approach gives the new Research model clean primitives while preserving safe handoff into existing publishing and Content Studio flows.
- Impact:
  - High
- Follow-up Work:
  - Implement schema boundaries in PR 3.
  - Define compatibility adapters before frontend cutover.

#### Decision ID: D-002

- Title: Validation TTL policy for SERP freshness
- Status: `decided`
- Date: `2026-05-07`
- Owners: `BE, PM, QA`
- Context:
  - Persisted SERP evidence will become stale.
  - Categories may have different refresh needs.
- Options Considered:
  - fixed 14-day TTL
  - fixed 30-day TTL
  - category-sensitive TTL
- Decision:
  - Start with a category-sensitive TTL using 14 days as the default for volatile or trend-sensitive categories and 30 days for slower-moving evergreen categories.
- Why:
  - A fixed 14-day TTL is safer but can drive avoidable validation cost.
  - A fixed 30-day TTL is cheaper but risks stale decisions in volatile spaces.
  - Category-sensitive TTL balances freshness and cost without overcomplicating the first implementation.
- Impact:
  - Medium
- Follow-up Work:
  - Add `validated_at`, `expires_at`, and freshness state in PR 8.
  - Define the first category volatility mapping in PR 9.

#### Decision ID: D-003

- Title: Software feasibility scoring model
- Status: `decided`
- Date: `2026-05-07`
- Owners: `BE, FE, PM`
- Context:
  - Software opportunities need a technical feasibility gate aligned to the current stack and product scope.
- Options Considered:
  - lightweight heuristic scoring
  - rule-based scoring by dependency type
  - richer scoring with architecture-aware capability profiles
- Decision:
  - Use a rule-based scoring model first, based on dependency type and delivery complexity, then evolve toward architecture-aware capability profiles later if needed.
- Why:
  - A lightweight heuristic is too vague for real delivery decisions.
  - A full capability-model system is powerful but too heavy for the first pass.
  - A rule-based model is explicit, explainable in the UI, and good enough to block unrealistic software ideas early.
- Impact:
  - High
- Follow-up Work:
  - Define initial feasibility buckets in PR 9, such as:
    - client-only feasible
    - light persistence feasible
    - backend required
    - external API required
    - long-running or infrastructure-heavy
  - Use those buckets to derive `feasibility_score` and `software_backlog_low_feasibility`.

#### Decision ID: D-004

- Title: Internal-link discovery source of truth
- Status: `decided`
- Date: `2026-05-07`
- Owners: `BE, DATA, PM`
- Context:
  - Internal-link hook discovery should use imported WordPress content, but imported data is currently shallow.
- Options Considered:
  - title/link-only matching for first version
  - enriched imported article table
  - hybrid approach with progressive enhancement
- Decision:
  - Use a hybrid approach with progressive enhancement: ship a first version using title/link-based matching on current imported content, but design the system to upgrade cleanly to richer imported article snapshots when available.
- Why:
  - Waiting for a full import rebuild would delay the Research migration too much.
  - Title/link-only matching is weak but still useful as an initial architecture signal.
  - A hybrid approach lets Research start identifying likely parents and children now while making richer article import a future quality upgrade.
- Impact:
  - Medium
- Follow-up Work:
  - Implement a basic internal-link fit pass in PR 14.
  - Revisit enrichment once imported WordPress content becomes deeper than title/link/excerpt.

#### Decision ID: D-005

- Title: Rejection feedback taxonomy
- Status: `decided`
- Date: `2026-05-07`
- Owners: `PM, FE, BE`
- Context:
  - Rejected jobs and candidates should feed later prompt cycles, but the reason taxonomy needs to stay simple enough for consistent use.
- Options Considered:
  - fixed controlled vocabulary only
  - controlled vocabulary plus optional free text
  - free text only
- Decision:
  - Use a controlled vocabulary plus optional free text.
- Why:
  - Controlled vocabulary makes feedback usable for prompt loops, reporting, and filtering.
  - Optional free text lets users capture nuance without breaking the structure needed for system learning.
  - Free text only would be hard to normalize and too inconsistent to guide future generation.
- Impact:
  - Medium
- Follow-up Work:
  - Implement the first controlled vocabulary in PR 5.
  - Reuse those tags as negative context in PR 7 and later generation cycles.

## Recommended Defaults Summary

These are the recommended defaults that should be treated as the baseline implementation unless the team explicitly changes them.

- Source-of-truth model:
  - Hybrid model with new parallel Research entities plus compatibility adapters into `research_topics`, `content_ideas`, and `Titles`.
- Validation freshness:
  - Category-sensitive TTL with 14 days for volatile categories and 30 days for evergreen categories.
- Software feasibility:
  - Rule-based feasibility scoring in v1, using dependency and delivery complexity buckets.
- Internal-link fit:
  - Hybrid progressive approach, starting with current WordPress import data and upgrading later.
- Rejection learning:
  - Controlled rejection tags plus optional free text.

## Recommended Initial Feasibility Buckets

These buckets are suggested starting defaults for software feasibility scoring.

- `client_only_feasible`
  - calculator, converter, estimator, checker, lightweight comparer
- `light_persistence_feasible`
  - tracker, planner, saved preferences, simple history
- `backend_required`
  - authenticated workflows, team data, complex business logic
- `external_api_required`
  - tool depends on third-party enrichment or paid external data
- `infra_heavy`
  - long-running jobs, async processing, queues, large data movement, intensive integrations

Suggested interpretation:

- `client_only_feasible` and `light_persistence_feasible` are usually eligible for `software_ready`
- `backend_required` may still be viable but should lower `feasibility_score`
- `external_api_required` and `infra_heavy` should usually route toward backlog unless the team explicitly wants to invest

## Recommended Initial Rejection Tag Set

Use this as the starting controlled vocabulary.

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

## Current-State Problems Driving This Rewrite

- broad topic generation happens before real opportunity validation
- keyword viability is too heuristic-heavy
- cluster-first generation can force weak ideas into existence
- article and software generation are too tightly coupled to the same cluster flow
- iteration through manual seeds is brittle and destructive
- UI defaults bias the user toward early heuristic outputs
- final keyword attachment happens too late in the process

## Target-State Architecture

### Core Objects

The new Research system should revolve around these logical entities:

- `research_user_jobs`
- `research_opportunity_candidates`
- `research_validation_runs`
- `research_serp_snapshots`
- `research_keyword_packs`
- `research_routing_decisions`
- `research_generated_outcomes`

### Required Persisted Context

Each validated opportunity should ideally persist:

- website context
- category path
- user job
- candidate type
- validation reason codes
- top 10 SERP snapshot
- validated timestamp and freshness state
- keyword evidence
- achievability subscores
- final routing decision
- final keyword pack
- feasibility score for software-shaped opportunities
- rejection reason tags
- internal-link parent and child candidates

## Rollout Plan

### Phase 0: Audit and Freeze

Goal:

- establish a reliable inventory of the current Research system
- prevent more investment into soon-to-be-obsolete paths

Tasks:

- inventory all frontend Research entry points
- inventory all backend endpoints used by Research
- inventory all Research-related tables and persisted fields
- identify current primary flows and current legacy flows
- identify all downstream consumers of research artifacts
- classify each current path as:
  - keep and adapt
  - keep temporarily behind compatibility layer
  - deprecate
  - delete after migration

Deliverables:

- dependency map
- legacy path list
- deprecation map
- migration risk register

### Phase 1: Feature Flags and Data Foundation

Goal:

- create a safe foundation for parallel implementation

Tasks:

- add feature flags for:
  - new job-based discovery
  - new validation engine
  - new routing UI
  - new keyword-pack gating
- design and add schema for user jobs, validations, SERP snapshots, routing decisions, and keyword packs
- define compatibility strategy between:
  - new research objects
  - existing `research_topics`
  - existing `content_ideas`
  - existing `Titles`

Deliverables:

- schema design
- migrations
- feature flag plan
- compatibility notes

### Phase 2: User Job Discovery

Goal:

- move the starting point of Research from broad topics to user jobs

Tasks:

- build a job discovery service using:
  - website description
  - primary category
  - secondary category
  - category descriptions
  - target audience
  - trend context
- generate 20 to 40 category-specific jobs
- persist jobs with provenance metadata
- classify jobs into likely tracks:
  - SEO
  - editorial
  - software-first

Deliverables:

- job generation service
- job classification service
- persistence layer
- basic job review API

### Phase 3: Opportunity Candidate Generation

Goal:

- convert user jobs into concrete search and workflow opportunities

Tasks:

- derive keyword candidates from user jobs
- derive workflow/tool candidates from software-shaped jobs
- stop relying on broad topic titles as the primary seed source
- persist candidates with job linkage
- persist rejection metadata for jobs the user explicitly dismisses
- feed rejection tags back into later generation cycles as negative context

Deliverables:

- candidate generation service
- candidate persistence
- candidate-to-job mapping
- negative feedback loop input model

### Phase 4: Validation Engine

Goal:

- validate candidates before idea generation

Tasks:

- fetch keyword metrics
- fetch or persist SERP snapshots
- persist `validated_at` and freshness status for each validation
- compute:
  - intent match
  - SERP weakness
  - SERP gap
  - software pattern score
  - feasibility score
  - monetization fit
  - volume score
  - KD ease
  - niche drift
- enforce candidate-type eligibility gates
- compute final achievability score
- persist reason codes and evidence
- define TTL and refresh rules for stale validation
- design rate limiting and queueing for validation workloads
- design cost-control rules so not every job is fully validated immediately

Deliverables:

- validation service
- achievability scoring service
- eligibility gate rules
- validation persistence
- validation freshness rules
- rate-limit and cost-control strategy

### Phase 5: Routing and Decision Layer

Goal:

- make the system explicit about what each opportunity should become

Tasks:

- implement routing rules for:
  - article
  - software
  - article plus software
  - editorial-only
  - software backlog due to low feasibility
  - needs more validation
  - rejected
- reroute strong jobs with poor article fit into software-first
- reroute strategically valuable but weak SEO candidates into editorial-only
- reroute search-attractive but technically unrealistic software candidates into backlog or article-only outcomes

Deliverables:

- routing decision service
- routing reason codes
- routed candidate states

### Phase 6: Opportunity-Driven Idea Generation

Goal:

- generate ideas only from validated and routed opportunities

Tasks:

- generate article ideas from validated SEO opportunities
- generate software ideas from validated software-first opportunities
- generate editorial ideas from strategic editorial jobs
- replace one-idea-per-cluster assumptions in the new path
- allow:
  - zero ideas from weak candidates
  - multiple ideas from unusually strong validated opportunities when justified
- feed recent SERP evidence back into generation prompts
- refuse generation from stale validation unless refreshed or explicitly overridden by policy

Deliverables:

- new generation prompts
- new generation payloads
- routed generation endpoints

### Phase 7: Final Keyword Pack Attachment

Goal:

- make sure ideas are supported before reaching Content Studio

Tasks:

- assign:
  - 1 primary keyword
  - 3 to 8 secondary keywords
- require measurable support for secondaries
- block thin opportunities from being treated as ready
- discover parent/child internal-link neighborhoods from imported WordPress content
- persist internal-link hook candidates before Content Studio handoff

Suggested blocking states:

- `cluster_too_thin`
- `needs_more_keyword_validation`

Deliverables:

- keyword pack builder
- readiness gate
- persistence for keyword pack quality state
- internal-link fit discovery

### Phase 8: Frontend Migration

Goal:

- shift the UI to the new mental model

Tasks:

- redesign the New Research flow around jobs
- redesign the detail view around validated opportunities
- expose achievability score breakdowns
- expose SERP evidence
- expose routing decisions and rejection reasons
- support migration UI states for legacy research rows

Deliverables:

- new frontend flows
- updated services and types
- legacy compatibility states

### Phase 9: Compatibility and Downstream Protection

Goal:

- keep current downstream workflows safe while Research is rebuilt

Tasks:

- preserve `topic_id`, `source_idea_id`, and `idea_metadata.category_context`
- preserve category path continuity for Content Studio
- preserve WordPress export category behavior
- add compatibility adapters where needed

Deliverables:

- compatibility layer
- downstream contract checklist

### Phase 10: Cutover and Cleanup

Goal:

- make the new path primary and retire obsolete paths

Tasks:

- switch default Research entry points to the new flow
- hide or deprecate legacy flows in the UI
- retire obsolete services and endpoints
- archive or remove obsolete data structures as planned
- update documentation to reflect the new source of truth

Deliverables:

- production cutover
- cleanup PRs
- final documentation pass

## Proposed PR Sequence

### PR 1: Research Inventory and Deprecation Map

Scope:

- current flow inventory
- path classification
- migration risk notes

Acceptance criteria:

- all major Research entry points are listed
- all key Research tables are listed
- each current path is labeled keep, temporary, deprecate, or delete

### PR 2: Feature Flags and Migration Scaffolding

Scope:

- backend and frontend feature flags
- rollout toggles

Acceptance criteria:

- new flow can be enabled or disabled independently
- legacy flow remains functional when flag is off

### PR 3: Schema v1 for New Research Model

Scope:

- new tables and/or new columns
- migrations
- indexes

Acceptance criteria:

- migrations apply cleanly
- all new core objects have a persistence strategy

### PR 4: User Job Generation Service

Scope:

- user job generation from site and category context
- persistence

Acceptance criteria:

- the system can generate and save 20 to 40 jobs
- generated jobs are category-aware and action-oriented

### PR 5: Job Review API and Basic UI

Scope:

- list, approve, reject, regenerate jobs
- early frontend view

Acceptance criteria:

- users can inspect and curate jobs before validation

### PR 6: Job Classification

Scope:

- classify jobs into SEO, editorial, or software-first

Acceptance criteria:

- every job receives a route hint and reason codes

### PR 7: Candidate Generation From Jobs

Scope:

- keyword and workflow candidate generation

Acceptance criteria:

- candidates are derived from jobs, not only from broad topic titles
- rejected jobs can persist structured rejection reasons for future negative prompting

### PR 8: SERP Snapshot Persistence

Scope:

- persist Top 10 SERP evidence

Acceptance criteria:

- candidate-level SERP evidence is queryable and reusable
- validation rows store `validated_at`
- staleness can be computed from persisted timestamps

### PR 9: Achievability Scoring Engine

Scope:

- implement eligibility gates
- implement achievability subscores
- persist decisions

Acceptance criteria:

- candidates can be accepted, rerouted, downgraded, or rejected using explicit scores
- scoring includes niche drift for article opportunities
- scoring includes feasibility for software opportunities
- validation freshness rules exist
- rate limiting and cost-control rules are implemented or stubbed behind the execution layer

### PR 10: Validation Review UI

Scope:

- frontend for achievability and reason-code review

Acceptance criteria:

- users can see why a candidate passed or failed

### PR 11: SEO Article Generation From Validated Opportunities

Scope:

- new article generation path

Acceptance criteria:

- article ideas are created only from validated SEO opportunities

### PR 12: Software-First Generation

Scope:

- new software generation path

Acceptance criteria:

- software ideas are created from workflow/tool opportunities, not as a side effect of article clustering
- low-feasibility software opportunities do not silently reach ready states

### PR 13: Editorial-Only Generation Path

Scope:

- standalone editorial generation

Acceptance criteria:

- editorial ideas are not blocked by SEO achievability gates

### PR 14: Final Keyword Pack Builder

Scope:

- primary plus secondary keyword attachment

Acceptance criteria:

- validated ideas receive strong keyword packs or are marked too thin
- internal-link parent/child candidates can be attached before Content Studio handoff

### PR 15: Content Studio Readiness Gate

Scope:

- prevent weak ideas from looking fully validated

Acceptance criteria:

- thin or under-validated ideas cannot silently move forward as ready

### PR 16: New Research Landing Experience

Scope:

- jobs-first research flow in the frontend

Acceptance criteria:

- the default user path begins with job discovery rather than broad topic selection

### PR 17: Opportunity Detail View

Scope:

- validated opportunity detail page or equivalent replacement for the current Topic Detail mental model

Acceptance criteria:

- the detail view centers on jobs, validation, routing, and final outcomes

### PR 18: Compatibility Layer for Legacy Records

Scope:

- keep old rows readable during rollout

Acceptance criteria:

- old in-flight research remains viewable and does not break the UI

### PR 19: Default to the New Flow

Scope:

- staging and production cutover

Acceptance criteria:

- new flow becomes default without breaking downstream workflows

### PR 20: Retire Obsolete Backend Paths

Scope:

- deprecate or remove old backend logic

Acceptance criteria:

- obsolete heuristic-only and cluster-first logic is no longer primary

### PR 21: Retire Obsolete Frontend Paths

Scope:

- remove old components, services, and copy

Acceptance criteria:

- users are no longer steered into the retired workflow

### PR 22: Data Cleanup and Archival

Scope:

- archive, migrate, or remove obsolete research artifacts

Acceptance criteria:

- stale transitional research data is handled according to the chosen retention policy

## PR Planning Table

| PR | Title | Primary Owners | Effort | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | Research Inventory and Deprecation Map | PM, BE, FE | S | not_started | Establish baseline before design drift continues |
| 2 | Feature Flags and Migration Scaffolding | BE, FE | S | not_started | Needed to run old and new flows safely |
| 3 | Schema v1 for New Research Model | DATA, BE | M | not_started | Foundation for jobs, validations, SERP, routing |
| 4 | User Job Generation Service | BE | M | not_started | First major architecture shift |
| 5 | Job Review API and Basic UI | FE, BE | M | not_started | Gives humans a curation checkpoint |
| 6 | Job Classification | BE | S | not_started | SEO vs editorial vs software-first route hinting |
| 7 | Candidate Generation From Jobs | BE | M | not_started | Replaces broad-topic seed dependence |
| 8 | SERP Snapshot Persistence | BE, DATA | M | not_started | Must exist before robust achievability and freshness checks |
| 9 | Achievability Scoring Engine | BE | L | not_started | Core ranking, routing, drift control, and feasibility layer |
| 10 | Validation Review UI | FE | M | not_started | Makes the new decision model visible |
| 11 | SEO Article Generation From Validated Opportunities | BE | M | not_started | New article generation path |
| 12 | Software-First Generation | BE | M | not_started | New software generation path |
| 13 | Editorial-Only Generation Path | BE | S | not_started | Keeps non-SEO ideas alive without abusing SEO logic |
| 14 | Final Keyword Pack Builder | BE | M | not_started | Required before Content Studio handoff |
| 15 | Content Studio Readiness Gate | FE, BE | S | not_started | Prevents thin ideas from looking production-ready |
| 16 | New Research Landing Experience | FE | L | not_started | Major UX transition |
| 17 | Opportunity Detail View | FE, BE | L | not_started | Likely replaces current Topic Detail mental model |
| 18 | Compatibility Layer for Legacy Records | BE, FE | M | not_started | Protects in-flight work during rollout |
| 19 | Default to the New Flow | PM, BE, FE, QA | M | not_started | Staging then production cutover |
| 20 | Retire Obsolete Backend Paths | BE, DATA | M | not_started | Cleanup is part of delivery |
| 21 | Retire Obsolete Frontend Paths | FE | M | not_started | Remove stale UX and services |
| 22 | Data Cleanup and Archival | DATA, BE | M | not_started | Final cleanup after cutover confidence |

## Suggested Ownership by Function

### Backend Lead

Recommended scope:

- PR 2
- PR 4
- PR 6
- PR 7
- PR 8
- PR 9
- PR 11
- PR 12
- PR 13
- PR 14
- PR 18
- PR 20

### Frontend Lead

Recommended scope:

- PR 2
- PR 5
- PR 10
- PR 15
- PR 16
- PR 17
- PR 18
- PR 21

### Data / Migration Lead

Recommended scope:

- PR 3
- PR 8
- PR 20
- PR 22

### Product / Program Owner

Recommended scope:

- PR 1
- milestone sequencing
- acceptance criteria signoff
- cutover readiness for PR 19

### QA Owner

Recommended scope:

- validation rules regression checks
- routing behavior validation
- Content Studio compatibility validation
- WordPress export regression validation
- cutover signoff

## Dependencies

- PR 2 depends on PR 1
- PR 3 depends on PR 1
- PR 4 depends on PR 3
- PR 5 depends on PR 4
- PR 6 depends on PR 4
- PR 7 depends on PR 4 and PR 6
- PR 8 depends on PR 7
- PR 9 depends on PR 7 and PR 8
- PR 10 depends on PR 9
- PR 11 depends on PR 9
- PR 12 depends on PR 9
- PR 13 depends on PR 6
- PR 14 depends on PR 11 and PR 12
- PR 15 depends on PR 14
- PR 16 depends on PR 5 and PR 10
- PR 17 depends on PR 10, PR 11, PR 12, and PR 13
- PR 18 depends on PR 16 or PR 17
- PR 19 depends on PR 15, PR 16, PR 17, and PR 18
- PR 20 depends on PR 19
- PR 21 depends on PR 19
- PR 22 depends on PR 19 and the agreed retention plan

## Workstreams

### Workstream A: Data and Backend Foundation

- PR 1
- PR 2
- PR 3
- PR 4
- PR 6
- PR 7
- PR 8
- PR 9

### Workstream B: Generation and Routing

- PR 11
- PR 12
- PR 13
- PR 14
- PR 15

### Workstream C: Frontend Migration

- PR 5
- PR 10
- PR 16
- PR 17
- PR 18

### Workstream D: Cleanup and Cutover

- PR 19
- PR 20
- PR 21
- PR 22

## Suggested Sequencing by Sprint or Planning Window

This is a suggested order, not a strict calendar.

### Window 1

- PR 1
- PR 2
- PR 3

Outcome:

- migration baseline and technical foundation exist

### Window 2

- PR 4
- PR 5
- PR 6

Outcome:

- jobs can be generated, reviewed, and classified

### Window 3

- PR 7
- PR 8
- PR 9

Outcome:

- validated opportunities and achievability scoring exist

### Window 4

- PR 10
- PR 11
- PR 12
- PR 13

Outcome:

- validated opportunities can drive routed generation

### Window 5

- PR 14
- PR 15
- PR 16

Outcome:

- final keyword gating exists and the new landing flow is usable

### Window 6

- PR 17
- PR 18

Outcome:

- opportunity detail UX is stable and legacy compatibility is in place

### Window 7

- PR 19
- PR 20
- PR 21
- PR 22

Outcome:

- new flow is primary and cleanup is complete

## Risk Notes

### High-Risk Areas

- compatibility with existing `content_ideas` and `Titles`
- preserving category-context behavior in Content Studio
- not regressing WordPress export category selection
- maintaining visibility into legacy in-flight research during the cutover
- avoiding duplicated logic while both systems coexist
- validation cost explosion when evaluating too many jobs or candidates at once
- stale SERP evidence causing low-confidence decisions to look current
- software opportunities looking attractive in search but being unrealistic to deliver
- weak traceability in the UI if users cannot inspect why a score was assigned

### Mitigations

- keep feature flags until cutover confidence is high
- ship compatibility adapters before hiding legacy flows
- validate category metadata persistence on every migration phase
- schedule cleanup only after staging proves the new flow is safe
- validate only the best-scoring jobs first when API cost is a concern
- add `validated_at` and TTL-triggered refresh behavior
- add explicit feasibility scoring before software ideas are marked ready
- make score breakdowns clickable and evidence-backed in the UI

## Validation Operations Requirements

These operational concerns should be treated as first-class implementation requirements.

### Rate Limiting

Phase 4 validation may evaluate 20 to 40 jobs and many derived candidates. The implementation should include:

- batch limits
- queueing or staged execution
- concurrency caps for DataForSEO and SERP lookups
- retry and backoff behavior

The system should not assume all candidate validation can run immediately or cheaply in a single synchronous step.

### Cost Control

Validation is materially more expensive than heuristic topic scoring. The implementation should prioritize likely winners first.

Recommended strategy:

- score jobs cheaply first
- validate the highest-potential candidates first
- defer full validation for weak or low-priority jobs
- stop validation early for candidates that fail eligibility gates quickly

### UI Traceability

Score transparency is part of product quality.

Users should be able to inspect why a score exists. For example:

- click `serp_weakness` and see the weak competitors or page types that contributed
- click `feasibility_score` and see whether the issue is backend complexity, missing API dependency, or stateful workflow needs
- click `niche_drift` and see which terms or categories made the candidate feel off-brand

If a score cannot be explained in the UI, it is too opaque for this workflow.

## Optional Future Enhancements After Core Migration

These should not block the core rebuild:

- advanced SERP page-type classification
- domain authority scoring refinements
- richer software gap analysis
- opportunity portfolio analytics
- historical win-rate feedback loop for achievability calibration

## Cleanup Plan

Cleanup is required work.

### Code Cleanup

- remove obsolete heuristic-only viability logic after replacement
- remove or downgrade broad topic-first logic as the primary architecture
- remove forced one-idea-per-cluster assumptions in new-path code and UI copy
- remove auto-selection patterns that bias users toward the first cluster or first heuristic result
- remove dead frontend services, components, and types tied only to retired flows
- remove dead backend endpoints and helpers after cutover

### Data Cleanup

- identify legacy research runs that should be archived, migrated, or deleted
- decide retention rules for obsolete keyword clusters and candidate rows
- remove duplicate fields once the new source-of-truth model is stable
- clean up stale cached research artifacts and transitional rows

### UX Cleanup

- remove copy that presents the old cluster-first workflow as the recommended process
- remove UI affordances that imply manual seed reruns are the primary iteration model if no longer true
- replace vague viability labels with explicit scores and reason codes

### Documentation Cleanup

- update `AGENTS.md` once the new flow becomes primary
- update or deprecate existing Research implementation docs that describe the old architecture
- document the final source-of-truth entities and flow

## Tracking Checklist

### Planning

- [ ] Current Research dependency map completed
- [ ] Deprecation map completed
- [ ] Feature flag plan approved
- [ ] Data model approved

### Backend

- [ ] User job generation implemented
- [ ] Job classification implemented
- [ ] Candidate generation implemented
- [ ] SERP snapshot persistence implemented
- [ ] Achievability scoring implemented
- [ ] Routing rules implemented
- [ ] Opportunity-driven idea generation implemented
- [ ] Final keyword pack builder implemented

### Frontend

- [ ] Job review UI implemented
- [ ] Validation review UI implemented
- [ ] New Research landing flow implemented
- [ ] Opportunity detail view implemented
- [ ] Legacy compatibility states implemented

### Downstream Safety

- [ ] Content Studio metadata continuity preserved
- [ ] WordPress export category behavior preserved
- [ ] `topic_id`, `source_idea_id`, and `idea_metadata.category_context` continuity preserved

### Cleanup

- [ ] Obsolete backend paths retired
- [ ] Obsolete frontend paths retired
- [ ] Obsolete data handled according to retention plan
- [ ] Obsolete documentation updated or deprecated

### Cutover

- [ ] New flow enabled in staging
- [ ] QA completed for routing and validation behavior
- [ ] New flow enabled in production
- [ ] Legacy flow hidden or clearly deprecated

## Definition of Done

The rebuild is complete when:

- Research starts from user jobs
- opportunities are validated before idea generation
- article, software, and editorial paths are distinct
- achievability scoring is explicit and evidence-backed
- validated ideas receive final keyword packs before Content Studio
- legacy cluster-first flow is no longer the primary product path
- obsolete Research code and data are retired or formally deprecated
