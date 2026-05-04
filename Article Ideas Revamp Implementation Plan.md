# Article Ideas Revamp Implementation Plan

## Purpose

This document turns the proposal in [Redefine article ideas.md](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/Redefine%20article%20ideas.md) into an implementation plan for this repository.

The goal is to replace the current subtopic-first idea generation flow with a topic-first keyword intelligence pipeline that:

- starts from one selected research topic
- discovers and scores the best keyword opportunities around that topic
- clusters opportunities by intent
- generates one or more article ideas and software ideas from those clusters
- preserves downstream compatibility with `content_ideas`, Content Studio, and publishing to `Titles`

## What Must Change

Today the flow is centered on:

- `research_topics -> subtopics -> idea burst`
- UI actions in [frontend/src/pages/TopicDetail.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/pages/TopicDetail.tsx)
- modal generation in [frontend/src/components/IdeaBurstModal.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/components/IdeaBurstModal.tsx)
- backend generation in [src/api/endpoints/research_topics.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/research_topics.py)
- enrichment and publishing in [src/api/endpoints/content_ideas.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/content_ideas.py)

The proposed flow is centered on:

- `research_topics -> keyword research run -> keyword candidates -> intent clusters -> ideas`

This is not a prompt tweak. It is a workflow and persistence redesign.

## Non-Negotiable Compatibility Requirements

The revamp must preserve these existing contracts:

- `content_ideas.topic_id` must still be set.
- `content_ideas.idea_metadata.category_context` must still include primary and secondary category context.
- ideas published to `Titles` must still preserve `source_idea_id`, `topic_id`, and `idea_metadata`.
- records in `Titles` must not be deleted as part of research-flow migration, regeneration, cleanup, or overwrite operations.
- Content Studio must keep resolving category path through the current fallback chain documented in [AGENTS.md](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/AGENTS.md).
- WordPress export must continue to support multiple category IDs and preserve Level1 and Level2 mappings.

## Approved Data Reset Rule

For this revamp, existing research-side Supabase content may be overwritten or regenerated when necessary, including:

- topic-level keyword research artifacts
- subtopics created by the old workflow
- draft `content_ideas` tied to the replaced research flow
- related intermediate research tables that support ideation

Hard boundary:

- do not delete records from `Titles`
- if old research records are replaced, preserve or detach any `Titles.source_idea_id` links before cleanup, following the same safety principle already present in [src/api/endpoints/research_topics.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/research_topics.py)

## Target Workflow

### 1. Topic Selection

Keep the current "New Research" entry point and topic creation flow.

The user should:

- create or choose a research topic
- open that topic detail page
- run one research process for that topic

### 2. Seed Set Generation

From the topic title, topic description, project/site context, and category path:

- generate 10-30 seed phrases
- keep them directional, not only exact-match rewrites
- store the generated seed set for auditability

### 3. Directional Keyword Discovery

Use DataForSEO in this order:

1. `dataforseo_labs/google/keyword_ideas/live`
2. `dataforseo_labs/google/related_keywords/live`
3. `keywords_data/google_ads/keywords_for_keywords/live`
4. `dataforseo_labs/google/keyword_suggestions/live`

This stage should produce a large candidate pool plus raw endpoint provenance.

### 4. Cleaning And Normalization

Normalize the candidate pool by:

- removing duplicates
- removing zero-volume and clearly irrelevant terms
- normalizing punctuation and near-duplicate phrase variants
- tracking a canonical phrase plus retained variants
- labeling likely intent mismatches such as `jobs`, `near me`, `pdf`, or purely local queries

### 5. Metric Enrichment

Enrich the candidate set with:

- search volume
- CPC
- paid competition
- competition index
- trend data
- keyword difficulty

Primary endpoints:

- `keywords_data/google_ads/search_volume/live`
- `dataforseo_labs/google/bulk_keyword_difficulty/live`

### 6. Opportunity Scoring

Score each keyword using a weighted model that favors:

- lower keyword difficulty
- higher competition index
- enough search volume
- commercial value via CPC/bids
- non-declining trend
- topical fit to the research topic and site/category lens

The exact formula should be configurable, not hardcoded in prompts only.

### 7. Intent Clustering

Group candidates into clusters so that one article idea targets one intent cluster instead of one keyword.

Each cluster should produce:

- one primary keyword
- several secondary/supporting keywords
- a cluster label
- an intent label
- a rationale for why the cluster is article-worthy

### 8. SERP Validation

Validate the top clusters with SERP data before idea generation.

The validation should answer:

- is blog/article content actually ranking?
- is the result set dominated by product, local, or government pages?
- is the opportunity realistic for the site?
- are there format clues such as lists, comparisons, guides, calculators, or snippets?

### 9. Idea Generation

Generate from the best validated clusters:

- one or more article ideas
- one or more software ideas

Important distinction:

- article ideas should be keyword-cluster-backed
- software ideas can use the cluster and site context, but do not need to be strictly keyword-limited

### 10. Persistence And Publishing

Persist the resulting ideas into `content_ideas` so existing content library and publishing flows continue to work.

## Recommended Architecture

## A. Keep `research_topics` As The Parent Object

Do not replace `research_topics`.

It already carries:

- selected topic
- project context
- category IDs
- category strategy metadata

That makes it the correct parent for keyword research runs and generated ideas.

## B. Introduce Topic-Level Research Artifacts

Add dedicated persistence for the new workflow instead of overloading `subtopics`.

Recommended new tables:

### `topic_keyword_research_runs`

One record per execution of the topic research pipeline.

Suggested fields:

- `id`
- `topic_id`
- `user_id`
- `status`
- `seed_keywords_json`
- `filters_json`
- `score_config_json`
- `summary_json`
- `created_at`
- `updated_at`

### `topic_keyword_candidates`

Canonical keyword opportunities discovered for a run.

Suggested fields:

- `id`
- `research_run_id`
- `topic_id`
- `user_id`
- `keyword`
- `canonical_keyword`
- `variant_keywords_json`
- `source_endpoints_json`
- `search_volume`
- `cpc`
- `competition`
- `competition_index`
- `keyword_difficulty`
- `trend_json`
- `intent_label`
- `topical_fit_score`
- `opportunity_score`
- `is_filtered_out`
- `filter_reason`
- `created_at`

### `topic_keyword_clusters`

Intent clusters built from candidates.

Suggested fields:

- `id`
- `research_run_id`
- `topic_id`
- `user_id`
- `cluster_name`
- `primary_keyword`
- `secondary_keywords_json`
- `intent_label`
- `serp_validation_json`
- `opportunity_score`
- `software_opportunity_score`
- `article_angle`
- `rationale`
- `created_at`
- `updated_at`

### Optional: `topic_keyword_serp_snapshots`

Use this only if we want raw SERP snapshots separate from cluster JSON.

## C. Keep `content_ideas` As The Delivery Object

Do not replace `content_ideas`.

Instead, extend the payload written into it so it can represent the new origin:

- `topic_id` remains required
- `subtopic` can be deprecated as a driver, but should remain backward-compatible for now
- `primary_keywords` and `secondary_keywords` remain populated
- `idea_metadata` should include:
  - `category_context`
  - `research_run_id`
  - `keyword_cluster_id`
  - `keyword_cluster`
  - `seed_keywords`
  - `keyword_sources`
  - `serp_validation`
  - `scoring_breakdown`
  - `generation_origin: "topic_keyword_pipeline_v1"`

Recommended compatibility rule:

- for article ideas, set `subtopic` to the human-readable cluster name during the migration period
- for software ideas, set `subtopic` to the cluster name that inspired the idea when one exists

That keeps existing UI grouping working while we transition away from true subtopics.

## Backend Implementation Plan

### Phase 1. Add Persistence And Service Layer

Create a new backend service module, for example:

- `src/services/topic_keyword_research_service.py`

Responsibilities:

- generate seed phrases
- call DataForSEO endpoints
- normalize keyword candidates
- enrich metrics
- score opportunities
- cluster by intent
- validate top clusters
- persist runs, candidates, and clusters

Keep this logic out of `research_topics.py` as much as possible so the endpoint file does not grow further.

### Phase 2. Add Topic Research Endpoints

Recommended endpoints:

- `POST /api/research-topics/<topic_id>/keyword-research/run`
- `GET /api/research-topics/<topic_id>/keyword-research/latest`
- `GET /api/research-topics/<topic_id>/keyword-research/runs`
- `GET /api/research-topics/<topic_id>/keyword-research/runs/<run_id>`
- `GET /api/research-topics/<topic_id>/keyword-research/runs/<run_id>/keywords`
- `GET /api/research-topics/<topic_id>/keyword-research/runs/<run_id>/clusters`
- `POST /api/research-topics/<topic_id>/keyword-research/runs/<run_id>/generate-ideas`

These should eventually become the primary replacement for:

- `POST /api/research-topics/<topic_id>/subtopics/generate`
- `POST /api/research-topics/idea-burst`

### Phase 3. Implement Scoring And Cluster Selection Rules

Move selection criteria into code and config, not only LLM instructions.

Suggested configurable values:

- `min_search_volume`
- `max_keyword_difficulty`
- `min_competition_index`
- `min_cpc`
- `allowed_intents`
- `serp_validation_required`
- `max_clusters_for_generation`

This makes the pipeline testable and easier to tune without changing prompts.

### Phase 4. Generate Ideas From Validated Clusters

Refactor idea generation so it receives structured cluster input instead of raw subtopic input.

Input should include:

- topic context
- category context
- project/site context
- cluster summary
- primary keyword
- secondary keywords
- SERP observations
- monetization context

Output should still map cleanly into today’s `content_ideas` schema.

### Phase 5. Preserve Publish-To-Titles Contract

Update publishing logic in [src/api/endpoints/content_ideas.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/content_ideas.py) so that ideas produced by the new pipeline still:

- insert `source_idea_id`
- insert `topic_id`
- persist `idea_metadata`
- preserve keyword handoff fields
- preserve WordPress category mapping fields

### Phase 6. Deprecate Subtopic-Centric Backend Flows

Once the new topic-level pipeline is stable:

- stop generating new `subtopics` for this workflow
- keep old read paths functional for historical topics
- keep deletion logic backward-compatible until data migration is complete

## Frontend Implementation Plan

### Phase 1. Replace Topic Detail Action Model

Update [frontend/src/pages/TopicDetail.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/pages/TopicDetail.tsx) so the primary action is no longer `Generate Sub-Topics`.

Recommended new sequence:

1. `Run Keyword Research`
2. `Review Keyword Opportunities`
3. `Review Clusters`
4. `Generate Ideas`

### Phase 2. Replace `IdeaBurstModal`

Current `IdeaBurstModal` expects a `Subtopic`.

Replace it with a new modal or page-scoped panel that works from:

- topic
- research run
- selected clusters

Possible replacement name:

- `TopicIdeaPipelineModal`
- `KeywordClusterIdeasModal`

### Phase 3. Reuse Keyword Intelligence UI Where Useful

The proposal explicitly wants keyword research to still be stored and shown in Keyword Intelligence.

Recommended approach:

- reuse the current keyword display patterns when practical
- add clear labels that distinguish:
  - raw keyword candidates
  - filtered/scored keywords
  - selected cluster keywords

### Phase 4. Keep Content Ideas And Content Studio Stable

The revamp should not require a redesign of Content Studio.

The frontend compatibility goal is:

- newly generated ideas still appear in Content Ideas
- ideas can still be published into Content Studio
- Content Studio still recovers the category path from `topic_id`, `source_idea_id`, and `idea_metadata.category_context`

## Data Migration Strategy

Use a staged migration.

### Stage A. Add New Tables And Code Paths

- add new tables
- ship backend services
- leave `subtopics` untouched

### Stage B. Start Writing New Topic-Level Research Runs

- new research executions write to topic keyword research tables
- existing historical subtopics remain readable

### Stage C. Bridge Old UI Contracts

- populate `content_ideas.subtopic` with cluster names for compatibility
- continue exposing content ideas through existing list and publish endpoints
- allow replacement of obsolete research artifacts for a topic as long as linked `Titles` records are preserved

### Stage D. Remove Subtopic Dependence From UI

- stop requiring subtopics to access idea generation
- move progress indicators from subtopic counts to research-run and cluster counts

### Stage E. Optional Cleanup

After the new pipeline is proven:

- archive or hide subtopic generation UI
- consider deprecating subtopic-specific endpoints for new flows

## Testing Plan

### Backend

- seed generation creates 10-30 directional phrases
- each DataForSEO stage handles rate limits, empty results, and partial failures
- keyword normalization deduplicates close variants correctly
- scoring logic is deterministic for fixed input
- clustering groups similar intent keywords together
- SERP validation rejects poor article-intent clusters
- generated `content_ideas` rows preserve `topic_id` and category context
- publishing to `Titles` still carries `source_idea_id`, `topic_id`, and keyword metadata

### Frontend

- topic detail page can launch a research run without subtopics
- keyword results render for a completed run
- cluster review supports selecting one or many clusters
- generated ideas appear in Content Ideas
- published ideas still appear in Content Studio with correct category path

### Regression

- Content Studio category path fallback chain still works
- WordPress export still supports multiple category IDs
- existing historical ideas and titles remain readable

## Risks And Decisions To Confirm Early

### 1. Whether To Reuse `subtopics` As A Temporary Cluster Store

Recommendation:

- do not use `subtopics` as the long-term model
- allow only a short compatibility bridge through `content_ideas.subtopic`

### 2. Whether Keyword Intelligence Should Read From New Tables Or Duplicated Payloads

Recommendation:

- make the new topic keyword research tables the source of truth
- adapt Keyword Intelligence readers to them instead of duplicating the same data elsewhere

### 3. How Much SERP Validation To Automate Before Idea Generation

Recommendation:

- implement automated SERP validation for the top clusters only
- do not run expensive SERP checks on every keyword candidate

### 4. Whether Software Ideas Should Be Generated For Every Cluster

Recommendation:

- no
- generate software ideas only for clusters with strong utility, repeat workflow, comparison, estimation, planning, or calculator potential

## Suggested Delivery Phases

### Milestone 1. Foundation

- schema added
- service layer added
- topic-level research run endpoint added
- raw keyword candidate persistence working

### Milestone 2. Scoring And Clustering

- keyword cleaning and scoring implemented
- cluster persistence implemented
- keyword intelligence UI can view the run

### Milestone 3. Idea Generation

- cluster-based article and software idea generation implemented
- `content_ideas` persistence compatible with current publishing flow

### Milestone 4. UI Migration

- topic detail page updated
- subtopic-first UX removed from the happy path

### Milestone 5. Hardening

- regression tests added
- rollout metrics reviewed
- legacy subtopic flow deprecated for new runs

## Implementation Checklist

### Discovery And Design

- [ ] Confirm final source of truth for keyword research data.
- [ ] Confirm whether old subtopic endpoints stay read-only for historical topics.
- [ ] Define the scoring formula and tunable thresholds.
- [ ] Define the clustering approach and acceptance criteria.
- [ ] Define rules for when software ideas should be generated.

### Database

- [ ] Add migration for `topic_keyword_research_runs`.
- [ ] Add migration for `topic_keyword_candidates`.
- [ ] Add migration for `topic_keyword_clusters`.
- [ ] Add indexes for `topic_id`, `research_run_id`, and `user_id`.
- [ ] Decide whether SERP snapshots need a separate table.

### Backend Services

- [ ] Create `topic_keyword_research_service.py`.
- [ ] Implement seed phrase generation from topic and category context.
- [ ] Implement directional keyword discovery via `keyword_ideas/live`.
- [ ] Implement adjacent discovery via `related_keywords/live`.
- [ ] Implement commercial discovery via `keywords_for_keywords/live`.
- [ ] Implement exact-match expansion via `keyword_suggestions/live`.
- [ ] Implement keyword cleanup and canonicalization.
- [ ] Implement search volume and competition enrichment.
- [ ] Implement bulk keyword difficulty enrichment.
- [ ] Implement opportunity scoring.
- [ ] Implement intent clustering.
- [ ] Implement SERP validation for top clusters.
- [ ] Persist runs, candidates, and clusters.

### Backend APIs

- [ ] Add endpoint to start a topic keyword research run.
- [ ] Add endpoint to fetch the latest run for a topic.
- [ ] Add endpoint to list runs for a topic.
- [ ] Add endpoint to fetch keywords for a run.
- [ ] Add endpoint to fetch clusters for a run.
- [ ] Add endpoint to generate ideas from selected clusters.
- [ ] Keep authentication and topic ownership checks aligned with current endpoints.

### Idea Persistence

- [ ] Extend `content_ideas` payloads with `research_run_id`.
- [ ] Extend `content_ideas` payloads with `keyword_cluster_id`.
- [ ] Extend `idea_metadata` with seed, source, cluster, SERP, and scoring context.
- [ ] Keep `topic_id`, `source_idea_id`, and category context intact.
- [ ] Use cluster names as `subtopic` during the migration period.

### Frontend

- [ ] Replace `Generate Sub-Topics` action in Topic Detail.
- [ ] Add UI for running topic keyword research.
- [ ] Add UI for reviewing keyword candidates and scores.
- [ ] Add UI for reviewing intent clusters.
- [ ] Replace `IdeaBurstModal` with a cluster-based idea generation UI.
- [ ] Reuse or adapt Keyword Intelligence views for the new data model.
- [ ] Keep Content Ideas screens working with newly generated ideas.

### Publishing And Downstream Compatibility

- [ ] Verify publish-to-`Titles` still writes `topic_id`.
- [ ] Verify publish-to-`Titles` still writes `source_idea_id`.
- [ ] Verify `idea_metadata.category_context` survives into `Titles`.
- [ ] Verify Content Studio still resolves Level1 and Level2 category path.
- [ ] Verify WordPress export still keeps multiple category IDs selected.

### Testing

- [ ] Add backend tests for seed generation and keyword normalization.
- [ ] Add backend tests for scoring and filtering.
- [ ] Add backend tests for cluster generation.
- [ ] Add backend tests for `content_ideas` persistence from the new pipeline.
- [ ] Add frontend tests for topic research run UX.
- [ ] Add regression tests for Content Studio category path behavior.
- [ ] Add regression tests for WordPress export category selection behavior.

### Rollout

- [ ] Ship behind a feature flag if practical.
- [ ] Test with a small set of topics across multiple projects.
- [ ] Review output quality for duplicate reduction and keyword quality.
- [ ] Compare new pipeline ideas against old subtopic-first output.
- [ ] Verify overwrite/regeneration flows never delete `Titles` rows.
- [ ] Deprecate the old generation path after validation.

## Recommended First Slice

The safest first implementation slice is:

1. add new topic keyword research tables
2. implement backend run generation and persistence
3. expose read endpoints for keywords and clusters
4. build a read-only frontend review screen
5. add cluster-to-idea generation
6. wire ideas into existing `content_ideas` and publishing flow

This sequence lets us prove the keyword pipeline before replacing the whole UX.

## Execution Breakdown

## Recommended Implementation Order

### Step 1. Lock The Safety Rules

Goal:

- define the overwrite behavior before changing schemas or endpoints

Implementation:

- treat `Titles` as immutable from a deletion perspective
- allow overwrite or cleanup of topic research artifacts
- require detach-first behavior for any cleanup touching records that may still be referenced by `Titles.source_idea_id`

Primary files:

- [src/api/endpoints/research_topics.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/research_topics.py)
- [src/api/endpoints/content_ideas.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/content_ideas.py)

### Step 2. Add New Supabase Tables

Goal:

- introduce the new topic-level research model without breaking the old flow

Implementation:

- add migrations for `topic_keyword_research_runs`
- add migrations for `topic_keyword_candidates`
- add migrations for `topic_keyword_clusters`
- add indexes on `topic_id`, `research_run_id`, and `user_id`

Recommended output:

- one migration file per table group if the repo’s migration style allows it
- include comments or naming that makes rollback and support easier

Primary files:

- `migrations/...new topic keyword research tables...sql`

### Step 3. Build The Service Layer

Goal:

- move the new research pipeline into a dedicated service instead of expanding endpoint files further

Implementation:

- create `src/services/topic_keyword_research_service.py`
- add methods for:
  - topic context assembly
  - seed phrase generation
  - directional keyword discovery
  - cleanup and canonicalization
  - metric enrichment
  - scoring
  - clustering
  - SERP validation
  - persistence

Recommended helper modules if needed:

- `src/services/topic_keyword_scoring.py`
- `src/services/topic_keyword_clustering.py`
- `src/services/topic_keyword_serialization.py`

Primary files:

- `src/services/topic_keyword_research_service.py`
- optional helper modules under `src/services/`

### Step 4. Add New Research Endpoints

Goal:

- expose the new workflow alongside the current one so frontend migration can be incremental

Implementation:

- add `POST /api/research-topics/<topic_id>/keyword-research/run`
- add `GET /api/research-topics/<topic_id>/keyword-research/latest`
- add `GET /api/research-topics/<topic_id>/keyword-research/runs/<run_id>`
- add `GET /api/research-topics/<topic_id>/keyword-research/runs/<run_id>/keywords`
- add `GET /api/research-topics/<topic_id>/keyword-research/runs/<run_id>/clusters`
- add `POST /api/research-topics/<topic_id>/keyword-research/runs/<run_id>/generate-ideas`

Primary files:

- [src/api/endpoints/research_topics.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/research_topics.py)

### Step 5. Adapt Idea Persistence

Goal:

- keep `content_ideas` as the handoff object while changing the upstream generator

Implementation:

- create a new persistence path for cluster-based ideas
- keep `topic_id`
- keep WordPress mapping fields
- keep `primary_keywords` and `secondary_keywords`
- set `idea_metadata.generation_origin = "topic_keyword_pipeline_v1"`
- store `research_run_id` and `keyword_cluster_id`
- during migration, use cluster name in `subtopic`

Primary files:

- [src/api/endpoints/content_ideas.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/content_ideas.py)

### Step 6. Add Frontend Services And Types

Goal:

- let the frontend consume the new topic keyword research model without reusing `Subtopic` types incorrectly

Implementation:

- add a new frontend service for topic keyword research endpoints
- add new types for:
  - research run
  - keyword candidate
  - keyword cluster
  - cluster-based idea generation request

Primary files:

- `frontend/src/services/topic-keyword-research.service.ts`
- `frontend/src/types/research.ts`
- `frontend/src/types/idea-burst.ts` or a new dedicated type file

### Step 7. Replace The Topic Detail Happy Path

Goal:

- move the user from subtopic generation to topic keyword research without replacing the entire page at once

Implementation:

- replace `Generate Sub-Topics` with `Run Keyword Research`
- show latest run status
- show counts for keyword candidates and selected clusters
- keep legacy subtopic UI hidden behind fallback logic until the new path is stable

Primary files:

- [frontend/src/pages/TopicDetail.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/pages/TopicDetail.tsx)
- [frontend/src/services/subtopics.service.ts](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/services/subtopics.service.ts) only if temporary bridging is needed

### Step 8. Replace `IdeaBurstModal`

Goal:

- stop passing a `Subtopic` into idea generation

Implementation:

- build a replacement modal or panel driven by:
  - topic
  - selected research run
  - selected clusters
- show:
  - primary keyword
  - secondary keywords
  - score summary
  - SERP validation summary
  - article ideas
  - software ideas

Primary files:

- [frontend/src/components/IdeaBurstModal.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/components/IdeaBurstModal.tsx)
- likely a new component such as `frontend/src/components/KeywordClusterIdeasModal.tsx`

### Step 9. Integrate With Keyword Intelligence

Goal:

- reuse the keyword review experience instead of creating a second disconnected keyword surface

Implementation:

- update Keyword Intelligence views or adapters to read from topic keyword run data
- distinguish:
  - discovered keywords
  - filtered keywords
  - selected cluster keywords

Primary files:

- keyword intelligence components and service files in `frontend/src/components/` and `frontend/src/services/`

### Step 10. Regression-Proof Publishing And Content Studio

Goal:

- verify the revamp changes nothing downstream in article production

Implementation:

- publish a new-pipeline idea to `Titles`
- verify `topic_id`, `source_idea_id`, `idea_metadata.category_context`
- verify Content Studio category path still resolves
- verify WordPress export still receives multi-category candidates

Primary files:

- [frontend/src/pages/ContentStudio.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/pages/ContentStudio.tsx)
- [frontend/src/components/WordPressExportModal.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/components/WordPressExportModal.tsx)
- [frontend/src/services/wordpressService.ts](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/services/wordpressService.ts)
- [src/api/endpoints/content_ideas.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/content_ideas.py)

## File-Level Task Map

### Backend

`src/api/endpoints/research_topics.py`

- add topic keyword research endpoints
- keep ownership and auth checks aligned with current topic endpoints
- eventually demote subtopic generation from the happy path
- preserve topic delete behavior so research cleanup never deletes `Titles`

`src/api/endpoints/content_ideas.py`

- add cluster-based idea persistence path
- preserve publish-to-`Titles` behavior
- preserve keyword metadata handoff
- preserve category context handoff

`src/services/topic_keyword_research_service.py`

- new orchestration service for the revamp
- should own DataForSEO sequencing, scoring, clustering, and persistence

`src/services/subtopics_service.py`

- likely unchanged at first
- later may become legacy-only if subtopics are removed from the new workflow

### Frontend

`frontend/src/pages/TopicDetail.tsx`

- primary frontend migration point
- replace subtopic-first actions with topic keyword research actions

`frontend/src/components/IdeaBurstModal.tsx`

- current generation UX is subtopic-bound
- replace or retire once cluster-based modal is live

`frontend/src/services/content-ideas.service.ts`

- add client methods for generate-from-clusters
- keep existing content idea list, publish, and enrich methods working

`frontend/src/services/subtopics.service.ts`

- legacy compatibility only
- should not be the main entry point after the migration

`frontend/src/services/topic-keyword-research.service.ts`

- new client for run, latest, keywords, clusters, and idea generation endpoints

`frontend/src/types/research.ts`

- add topic keyword run summary types
- add keyword candidate and cluster summary types

`frontend/src/types/idea-burst.ts`

- update only if we intentionally reuse this file
- otherwise create a dedicated type file to avoid mixing subtopic and cluster concepts

### Downstream Verification Targets

`frontend/src/pages/ContentStudio.tsx`

- verify category path still resolves from `idea_metadata.category_context`, `topic_id`, and `source_idea_id`

`frontend/src/components/WordPressExportModal.tsx`

- verify linked category auto-selection remains multi-source and multi-category

`frontend/src/services/wordpressService.ts`

- verify `categories: number[]` behavior remains unchanged

## Suggested Work Packages

### Package 1. Schema And Service Foundation

- migrations
- topic keyword research service
- backend persistence helpers

### Package 2. Read APIs And Inspection UI

- run and cluster read endpoints
- frontend service
- read-only topic research review UI

### Package 3. Cluster-To-Idea Generation

- generate ideas from selected clusters
- persist to `content_ideas`
- keep `Titles` handoff intact

### Package 4. Topic Detail Migration

- replace subtopic-first actions
- retire or hide legacy subtopic UI

### Package 5. Regression And Cleanup

- test Content Studio and WordPress export
- validate overwrite flows
- deprecate old generation path for new runs

## Milestones

## Milestone 0. Planning And Safety Guardrails

Outcome:

- the team agrees on the target data model, overwrite rules, and non-regression boundaries before implementation begins

Tasks:

- confirm the new topic-level research tables and naming
- confirm that `Titles` is never deleted by migration or regeneration flows
- confirm temporary compatibility strategy for `content_ideas.subtopic`
- confirm scoring thresholds and whether they are global or configurable per project
- confirm whether historical subtopics remain visible or become legacy-only

Dependencies:

- none

Estimated complexity:

- low

Primary deliverables:

- finalized implementation plan
- migration safety notes
- agreed API shape

## Milestone 1. Schema And Backend Foundation

Outcome:

- the repo can persist topic-level keyword research runs, candidates, and clusters independently of subtopics

Tasks:

- add Supabase migrations for new topic keyword research tables
- add indexes and any required constraints
- create `src/services/topic_keyword_research_service.py`
- add persistence helpers for runs, candidates, and clusters
- add serialization helpers for research output

Dependencies:

- Milestone 0

Estimated complexity:

- medium

Primary files:

- `migrations/...`
- `src/services/topic_keyword_research_service.py`

Definition of done:

- a topic research run can be created and saved end-to-end with placeholder or fixture data

## Milestone 2. Keyword Discovery Pipeline

Outcome:

- the backend can take one topic and produce a cleaned, enriched keyword candidate set

Tasks:

- implement topic-to-seed generation
- implement DataForSEO `keyword_ideas/live` integration
- implement `related_keywords/live` integration
- implement `keywords_for_keywords/live` integration
- implement `keyword_suggestions/live` integration
- implement deduplication and canonicalization
- implement filter rules for irrelevant intent
- implement search volume, CPC, competition, and KD enrichment

Dependencies:

- Milestone 1

Estimated complexity:

- high

Primary files:

- `src/services/topic_keyword_research_service.py`
- possible helper modules in `src/services/`

Definition of done:

- given a topic, the backend can persist a keyword candidate set with normalized metrics and filter labels

## Milestone 3. Scoring, Clustering, And SERP Validation

Outcome:

- the candidate set turns into ranked clusters that are suitable inputs for idea generation

Tasks:

- implement opportunity scoring model
- implement cluster generation by intent
- implement cluster-level primary and secondary keyword selection
- implement SERP validation for top clusters
- persist cluster summaries and scoring breakdowns

Dependencies:

- Milestone 2

Estimated complexity:

- high

Primary files:

- `src/services/topic_keyword_research_service.py`
- optional scoring or clustering helpers

Definition of done:

- a completed research run yields ranked clusters with validation summaries and article-worthiness signals

## Milestone 4. Topic-Level Research APIs

Outcome:

- frontend and future tooling can access the new research workflow through stable endpoints

Tasks:

- add endpoint to start a keyword research run
- add endpoint to fetch latest run
- add endpoint to fetch one run by ID
- add endpoint to list keywords for a run
- add endpoint to list clusters for a run
- add endpoint to generate ideas from one or more clusters

Dependencies:

- Milestone 3

Estimated complexity:

- medium

Primary files:

- [src/api/endpoints/research_topics.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/research_topics.py)

Definition of done:

- the full new workflow is accessible without using subtopic endpoints

## Milestone 5. Cluster-To-Idea Persistence

Outcome:

- validated clusters can generate article and software ideas that land in `content_ideas` and can flow into `Titles`

Tasks:

- implement cluster-based idea generation request handling
- map output into `content_ideas`
- preserve `topic_id`, WordPress mapping fields, and keyword fields
- write `research_run_id`, `keyword_cluster_id`, `generation_origin`, and scoring context into `idea_metadata`
- use cluster names in `subtopic` during compatibility phase
- verify publish-to-`Titles` keeps `source_idea_id` and category context intact

Dependencies:

- Milestone 4

Estimated complexity:

- high

Primary files:

- [src/api/endpoints/content_ideas.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/content_ideas.py)

Definition of done:

- a new-pipeline idea can be generated, saved, and published into `Titles` without breaking downstream consumers

## Milestone 6. Frontend Read Path

Outcome:

- users can run topic keyword research and inspect results before we fully replace the current generation UX

Tasks:

- add frontend service for topic keyword research APIs
- add types for runs, candidates, and clusters
- add read-only UI for latest run status
- add UI for keyword candidate and cluster review

Dependencies:

- Milestone 4

Estimated complexity:

- medium

Primary files:

- `frontend/src/services/topic-keyword-research.service.ts`
- `frontend/src/types/research.ts`
- new UI components under `frontend/src/components/`

Definition of done:

- a user can run research for a topic and inspect keyword and cluster outputs in the frontend

## Milestone 7. Frontend Generation UX Migration

Outcome:

- the happy path shifts from subtopics to clusters

Tasks:

- update `TopicDetail.tsx` to replace `Generate Sub-Topics` with `Run Keyword Research`
- add cluster selection UI
- replace or retire `IdeaBurstModal`
- add generate-ideas-from-clusters UX
- keep legacy subtopic path available only as fallback if needed during rollout

Dependencies:

- Milestone 5
- Milestone 6

Estimated complexity:

- high

Primary files:

- [frontend/src/pages/TopicDetail.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/pages/TopicDetail.tsx)
- [frontend/src/components/IdeaBurstModal.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/components/IdeaBurstModal.tsx)
- new replacement modal/component files

Definition of done:

- a user can complete the new end-to-end idea generation flow without using subtopics

## Milestone 8. Regression Hardening

Outcome:

- the new workflow is safe to make default

Tasks:

- add backend tests for seed generation, scoring, clustering, and persistence
- add frontend tests for topic research run UX
- verify Content Studio category-path resolution still works
- verify WordPress export multi-category selection still works
- verify overwrite flows never delete `Titles`
- compare outputs between old and new flow on a controlled topic sample

Dependencies:

- Milestone 5
- Milestone 7

Estimated complexity:

- medium

Primary files:

- backend test files
- frontend test files
- regression coverage around Content Studio and WordPress export

Definition of done:

- the new flow passes regression checks and is ready to become the default path

## Milestone 9. Cutover And Legacy Cleanup

Outcome:

- the new topic keyword pipeline becomes the default idea generation path

Tasks:

- hide or remove subtopic-first actions from the main UX
- keep legacy read paths only where historical data requires them
- mark subtopic generation endpoints as legacy for new research flows
- clean up obsolete adapters once the new path is stable

Dependencies:

- Milestone 8

Estimated complexity:

- medium

Primary files:

- `frontend/src/pages/TopicDetail.tsx`
- legacy service and endpoint adapters

Definition of done:

- new topics use only the new workflow, while historical data remains readable

## Dependency Summary

- Milestone 0 unlocks Milestone 1.
- Milestone 1 unlocks Milestone 2.
- Milestone 2 unlocks Milestone 3.
- Milestone 3 unlocks Milestone 4.
- Milestone 4 unlocks Milestone 5 and Milestone 6.
- Milestone 5 plus Milestone 6 unlock Milestone 7.
- Milestone 5 plus Milestone 7 unlock Milestone 8.
- Milestone 8 unlocks Milestone 9.

## Suggested Team Split

If this work is split across people, the cleanest division is:

- Backend foundation: Milestones 1-4
- Idea persistence and publish compatibility: Milestone 5
- Frontend read and generation UX: Milestones 6-7
- Regression and rollout: Milestones 8-9

## Estimated Critical Path

The likely critical path is:

1. Milestone 1
2. Milestone 2
3. Milestone 3
4. Milestone 4
5. Milestone 5
6. Milestone 6 and Milestone 7
7. Milestone 8
8. Milestone 9

If we want the fastest safe delivery, the best checkpoint demo is after Milestone 6:

- backend pipeline exists
- results are persisted
- frontend can inspect them

That gives us a quality checkpoint before we replace the generation UX.

## Sprint Plan

## Phase 1. Foundation And Research Pipeline

Goal:

- create the new topic-level keyword research backbone without changing the user-facing happy path yet

Suggested sprint outcome:

- a topic can run through the new backend keyword pipeline and persist runs, candidates, and clusters

Suggested tickets:

### Ticket 1. Finalize topic keyword research schema

Scope:

- define tables, columns, indexes, and naming for:
  - `topic_keyword_research_runs`
  - `topic_keyword_candidates`
  - `topic_keyword_clusters`

Acceptance criteria:

- migration spec is approved
- overwrite behavior is documented
- no migration path touches `Titles` deletes

Complexity:

- small

Dependencies:

- none

### Ticket 2. Add Supabase migrations for topic keyword research

Scope:

- create migration files
- add indexes and constraints

Acceptance criteria:

- migrations apply cleanly
- tables are queryable
- rollback approach is understood

Complexity:

- medium

Dependencies:

- Ticket 1

### Ticket 3. Create `topic_keyword_research_service`

Scope:

- add service module and core persistence helpers
- wire run creation and storage lifecycle

Acceptance criteria:

- service can create a research run record
- service can persist candidate and cluster payloads

Complexity:

- medium

Dependencies:

- Ticket 2

### Ticket 4. Implement topic-to-seed generation

Scope:

- generate 10-30 directional seed phrases from topic, project, and category context

Acceptance criteria:

- seeds are persisted with the run
- seeds are not limited to exact phrase rewrites

Complexity:

- medium

Dependencies:

- Ticket 3

### Ticket 5. Implement multi-step DataForSEO discovery pipeline

Scope:

- integrate:
  - `keyword_ideas/live`
  - `related_keywords/live`
  - `keywords_for_keywords/live`
  - `keyword_suggestions/live`

Acceptance criteria:

- pipeline returns merged candidate keywords with endpoint provenance
- partial endpoint failure does not crash the run

Complexity:

- large

Dependencies:

- Ticket 4

### Ticket 6. Implement normalization and enrichment

Scope:

- deduplicate candidates
- normalize canonical keywords and variants
- enrich metrics for search volume, CPC, competition, and KD

Acceptance criteria:

- candidates are saved with normalized metrics
- irrelevant keywords can be marked filtered out with reasons

Complexity:

- large

Dependencies:

- Ticket 5

### Ticket 7. Implement scoring and clustering

Scope:

- add opportunity scoring
- cluster keywords by intent
- assign primary and secondary keywords

Acceptance criteria:

- a run yields ranked clusters
- cluster payload includes scoring breakdown

Complexity:

- large

Dependencies:

- Ticket 6

### Ticket 8. Implement SERP validation for top clusters

Scope:

- validate article-worthiness of top clusters before idea generation

Acceptance criteria:

- clusters store validation summaries
- poor-intent clusters can be excluded from idea generation

Complexity:

- medium

Dependencies:

- Ticket 7

## Phase 2. APIs, Read Path, And Review UI

Goal:

- expose the new pipeline through stable endpoints and let users inspect it in the frontend

Suggested sprint outcome:

- a user can run topic keyword research and review keyword/clustering output in the UI

Suggested tickets:

### Ticket 9. Add topic keyword research API endpoints

Scope:

- add endpoints for:
  - start run
  - latest run
  - run details
  - run keywords
  - run clusters

Acceptance criteria:

- endpoints enforce topic ownership and auth
- endpoint responses are stable enough for frontend integration

Complexity:

- medium

Dependencies:

- Phase 1 complete

### Ticket 10. Add frontend service and types for topic keyword research

Scope:

- create `topic-keyword-research.service.ts`
- add types for runs, candidates, and clusters

Acceptance criteria:

- frontend can fetch latest run, keywords, and clusters
- new types do not overload `Subtopic`

Complexity:

- medium

Dependencies:

- Ticket 9

### Ticket 11. Add read-only research review UI to Topic Detail

Scope:

- show latest run state
- show candidate and cluster counts
- render review panels or sections for keyword and cluster output

Acceptance criteria:

- user can inspect new pipeline output without using subtopic generation
- existing page still loads historical topic data safely

Complexity:

- medium

Dependencies:

- Ticket 10

### Ticket 12. Add keyword intelligence integration or adapter

Scope:

- reuse or adapt keyword review surfaces for the new run data model

Acceptance criteria:

- keyword research from the new pipeline is visible in the keyword review experience
- selected keywords and filtered keywords are distinguishable

Complexity:

- medium

Dependencies:

- Ticket 11

## Phase 3. Cluster-To-Idea Generation And Publishing Compatibility

Goal:

- generate article and software ideas from validated clusters and keep the current publishing handoff intact

Suggested sprint outcome:

- a selected cluster can produce `content_ideas` rows that publish into `Titles` safely

Suggested tickets:

### Ticket 13. Add generate-ideas-from-clusters API

Scope:

- add endpoint that accepts selected cluster IDs or a run-plus-selection payload

Acceptance criteria:

- endpoint generates article and software ideas from validated clusters
- request shape is documented and testable

Complexity:

- medium

Dependencies:

- Phase 2 complete

### Ticket 14. Persist cluster-based ideas into `content_ideas`

Scope:

- map generated ideas into the existing `content_ideas` schema
- add compatibility metadata

Acceptance criteria:

- persisted rows include:
  - `topic_id`
  - `primary_keywords`
  - `secondary_keywords`
  - `idea_metadata.category_context`
  - `idea_metadata.research_run_id`
  - `idea_metadata.keyword_cluster_id`
  - `idea_metadata.generation_origin`
- cluster name is written into `subtopic` for compatibility

Complexity:

- large

Dependencies:

- Ticket 13

### Ticket 15. Verify publish-to-`Titles` compatibility

Scope:

- test publishing new-pipeline ideas through the existing flow

Acceptance criteria:

- publish keeps `source_idea_id`
- publish keeps `topic_id`
- publish keeps category context
- no code path deletes `Titles`

Complexity:

- medium

Dependencies:

- Ticket 14

### Ticket 16. Preserve WordPress category handoff

Scope:

- validate WordPress export behavior with new-pipeline ideas

Acceptance criteria:

- article export still resolves linked categories
- multi-category selection remains intact

Complexity:

- small

Dependencies:

- Ticket 15

## Phase 4. UX Cutover

Goal:

- make the new cluster-based workflow the default user path

Suggested sprint outcome:

- users generate ideas from topic keyword clusters instead of subtopics

Suggested tickets:

### Ticket 17. Replace `Generate Sub-Topics` action with `Run Keyword Research`

Scope:

- update the primary CTA and status flow in `TopicDetail.tsx`

Acceptance criteria:

- the new path is visible and usable as the main flow
- legacy actions are hidden or secondary

Complexity:

- medium

Dependencies:

- Phase 3 complete

### Ticket 18. Replace or retire `IdeaBurstModal`

Scope:

- build a cluster-based generation modal or panel

Acceptance criteria:

- UI works from selected clusters, not `Subtopic`
- user can review primary/secondary keywords before generation

Complexity:

- large

Dependencies:

- Ticket 17

### Ticket 19. Add multi-cluster selection and generation UX

Scope:

- allow one or more clusters to be selected for idea generation

Acceptance criteria:

- user can generate multiple related ideas in one run
- duplicate-like clusters are easier to avoid in the UI

Complexity:

- medium

Dependencies:

- Ticket 18

## Phase 5. Hardening And Legacy Cleanup

Goal:

- make the new flow reliable enough to become default and reduce reliance on the old one

Suggested sprint outcome:

- the new workflow is regression-tested and the old subtopic path is no longer the default

Suggested tickets:

### Ticket 20. Add backend regression tests

Scope:

- add coverage for seed generation, normalization, scoring, clustering, and persistence

Acceptance criteria:

- deterministic test cases exist for ranking and cluster generation

Complexity:

- medium

Dependencies:

- Phase 4 complete

### Ticket 21. Add frontend regression tests

Scope:

- add coverage for topic detail research flow and cluster review

Acceptance criteria:

- new flow has basic interaction coverage
- critical render states are tested

Complexity:

- medium

Dependencies:

- Phase 4 complete

### Ticket 22. Add downstream regression checks

Scope:

- validate Content Studio category path and WordPress export behavior

Acceptance criteria:

- category path still resolves correctly for new-pipeline ideas
- export still supports multiple categories

Complexity:

- medium

Dependencies:

- Ticket 15

### Ticket 23. Deprecate old subtopic-first generation flow

Scope:

- move legacy subtopic generation off the happy path
- keep historical data readable

Acceptance criteria:

- new topics use the new pipeline by default
- historical records remain accessible

Complexity:

- medium

Dependencies:

- Tickets 20, 21, and 22

## Suggested Sprint Framing

If this is planned as 5 delivery sprints, a clean framing is:

- Sprint 1: Tickets 1-4
- Sprint 2: Tickets 5-8
- Sprint 3: Tickets 9-12
- Sprint 4: Tickets 13-16
- Sprint 5: Tickets 17-23

If the team wants lower risk, split Sprint 5 into two sprints:

- Sprint 5: Tickets 17-19
- Sprint 6: Tickets 20-23

## Suggested Tracking Fields

For Linear or Notion, each ticket should track:

- owner
- status
- dependency
- risk level
- affected files
- migration impact
- test coverage required
- rollout note

## Best Demo Points

The best review/demo checkpoints are:

- after Ticket 8: backend pipeline quality review
- after Ticket 12: frontend review of keyword and cluster outputs
- after Ticket 16: end-to-end idea generation and publish compatibility review
- after Ticket 23: final cutover review

## Linear Ticket Drafts

Use the following as import-friendly ticket drafts.

### T1. Finalize Topic Keyword Research Schema

Description:

- Define the final table and column design for `topic_keyword_research_runs`, `topic_keyword_candidates`, and `topic_keyword_clusters`.
- Confirm overwrite behavior for research-side data and the hard rule that `Titles` records must never be deleted.

Acceptance Criteria:

- Final schema is documented and approved.
- Overwrite and detach-first rules are documented.
- No migration or cleanup path requires deleting `Titles`.

Dependencies:

- None

### T2. Add Supabase Migrations For Topic Keyword Research

Description:

- Create the SQL migrations for the new topic keyword research tables, indexes, and basic constraints.

Acceptance Criteria:

- Migrations apply cleanly.
- New tables are queryable in Supabase.
- Indexes exist for `topic_id`, `research_run_id`, and `user_id`.

Dependencies:

- T1

### T3. Create `topic_keyword_research_service`

Description:

- Add a dedicated backend service to own topic keyword research orchestration and persistence.

Acceptance Criteria:

- Service can create a run record.
- Service can persist candidate and cluster payloads.
- Endpoint files do not need to own the full pipeline logic.

Dependencies:

- T2

### T4. Implement Topic-To-Seed Generation

Description:

- Generate 10-30 directional seed phrases from research topic, project context, and category context.

Acceptance Criteria:

- Seeds are persisted with each run.
- Seeds are broader than exact phrase rewrites.
- Category and niche context influence the generated seeds.

Dependencies:

- T3

### T5. Implement Multi-Step DataForSEO Discovery Pipeline

Description:

- Integrate `keyword_ideas/live`, `related_keywords/live`, `keywords_for_keywords/live`, and `keyword_suggestions/live` into the new research pipeline.

Acceptance Criteria:

- Pipeline returns a merged raw candidate pool.
- Source endpoint provenance is retained.
- Partial endpoint failures do not fail the entire run.

Dependencies:

- T4

### T6. Implement Keyword Normalization And Enrichment

Description:

- Deduplicate and canonicalize keywords, remove obvious junk, and enrich candidates with metrics such as search volume, CPC, competition, and KD.

Acceptance Criteria:

- Candidate rows store canonical keyword plus variants when relevant.
- Irrelevant or low-value keywords can be marked filtered out with a reason.
- Enriched metric fields are persisted for valid candidates.

Dependencies:

- T5

### T7. Implement Opportunity Scoring And Intent Clustering

Description:

- Add the ranking model and cluster candidates into article-worthy intent groups.

Acceptance Criteria:

- Each cluster has a primary keyword and secondary keyword set.
- Cluster payload includes score and rationale.
- Scoring rules are implemented in code, not only prompt text.

Dependencies:

- T6

### T8. Implement SERP Validation For Top Clusters

Description:

- Validate top clusters against SERP intent before idea generation.

Acceptance Criteria:

- Validation summaries are persisted.
- Clusters with poor article intent can be excluded or flagged.
- Validation captures practical notes such as article-vs-product intent mismatch.

Dependencies:

- T7

### T9. Add Topic Keyword Research API Endpoints

Description:

- Add backend endpoints to start a run and fetch latest run, run details, keywords, and clusters.

Acceptance Criteria:

- Endpoints enforce auth and topic ownership checks.
- Responses are stable enough for frontend use.
- New flow can be used without subtopic endpoints.

Dependencies:

- T8

### T10. Add Frontend Service And Types For Topic Keyword Research

Description:

- Create the frontend client and TypeScript types for research runs, keyword candidates, and clusters.

Acceptance Criteria:

- Frontend can fetch latest run, keywords, and clusters.
- New types do not overload `Subtopic`.
- Service methods align with new backend endpoints.

Dependencies:

- T9

### T11. Add Read-Only Research Review UI To Topic Detail

Description:

- Add a non-destructive frontend review path that shows topic keyword research output without changing the full generation UX yet.

Acceptance Criteria:

- User can see latest run status and counts.
- User can inspect keyword candidates and clusters.
- Existing topic detail page remains stable for historical data.

Dependencies:

- T10

### T12. Integrate Or Adapt Keyword Intelligence For New Run Data

Description:

- Reuse or adapt keyword review UI so new topic keyword runs can be explored without duplicating the whole keyword experience.

Acceptance Criteria:

- New pipeline keyword data is visible in the keyword review experience.
- Filtered, selected, and supporting keywords are distinguishable.

Dependencies:

- T11

### T13. Add Generate-Ideas-From-Clusters API

Description:

- Add a backend endpoint that takes selected clusters and produces article and software idea candidates.

Acceptance Criteria:

- Endpoint accepts selected cluster IDs or equivalent run selection payload.
- Response includes article and software ideas.
- Output is based on validated cluster context, not subtopics.

Dependencies:

- T12

### T14. Persist Cluster-Based Ideas Into `content_ideas`

Description:

- Save new-pipeline ideas into `content_ideas` while preserving compatibility with downstream flows.

Acceptance Criteria:

- Rows include `topic_id`, keyword fields, and category context.
- `idea_metadata` includes `research_run_id`, `keyword_cluster_id`, and `generation_origin`.
- Cluster name is written to `subtopic` during the compatibility phase.

Dependencies:

- T13

### T15. Verify Publish-To-`Titles` Compatibility

Description:

- Confirm that ideas generated from the new pipeline can publish into `Titles` without breaking lineage or metadata.

Acceptance Criteria:

- Publish preserves `source_idea_id`.
- Publish preserves `topic_id`.
- Publish preserves category context and keyword handoff fields.
- No publish or cleanup path deletes `Titles`.

Dependencies:

- T14

### T16. Preserve WordPress Category Handoff For New-Pipeline Ideas

Description:

- Validate that WordPress export behavior stays correct when ideas originate from the new cluster-based pipeline.

Acceptance Criteria:

- Export still resolves linked categories.
- Multiple category IDs remain supported.
- Level1 and Level2 category handoff still works.

Dependencies:

- T15

### T17. Replace `Generate Sub-Topics` With `Run Keyword Research`

Description:

- Update Topic Detail so the primary action reflects the new workflow.

Acceptance Criteria:

- Main CTA is `Run Keyword Research`.
- New run status is visible in the page.
- Legacy subtopic path is no longer the primary happy path.

Dependencies:

- T16

### T18. Replace Or Retire `IdeaBurstModal`

Description:

- Build a new cluster-based generation UI and stop relying on `Subtopic` as the generation input.

Acceptance Criteria:

- UI is driven by selected clusters.
- User can review primary and secondary keywords before generation.
- Article and software idea previews render from cluster context.

Dependencies:

- T17

### T19. Add Multi-Cluster Selection And Generation UX

Description:

- Allow the user to select one or more clusters for idea generation.

Acceptance Criteria:

- Multiple clusters can be selected and submitted.
- UX reduces duplicate-like idea generation across similar clusters.
- Selected cluster state is clear and recoverable.

Dependencies:

- T18

### T20. Add Backend Regression Tests

Description:

- Add automated backend coverage for seed generation, normalization, scoring, clustering, and persistence.

Acceptance Criteria:

- Deterministic tests exist for scoring and clustering.
- Core pipeline stages have regression coverage.

Dependencies:

- T19

### T21. Add Frontend Regression Tests

Description:

- Add automated frontend coverage for the new topic research and cluster review flow.

Acceptance Criteria:

- Key read and interaction states are covered.
- New generation flow has baseline regression coverage.

Dependencies:

- T19

### T22. Add Downstream Regression Checks For Content Studio And WordPress Export

Description:

- Verify that the new idea source does not break category-path resolution or WordPress export category selection.

Acceptance Criteria:

- Content Studio still resolves category path correctly.
- WordPress export still supports multiple category IDs.
- New-pipeline ideas behave the same as existing ideas downstream.

Dependencies:

- T15

### T23. Deprecate Old Subtopic-First Generation Flow

Description:

- Remove the old subtopic-first idea generation path from the happy path while keeping historical data readable.

Acceptance Criteria:

- New topics use the new pipeline by default.
- Historical subtopics remain readable where needed.
- Legacy generation paths are clearly marked or hidden.

Dependencies:

- T20
- T21
- T22
