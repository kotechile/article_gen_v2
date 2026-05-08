# AGENTS.md

## Purpose

This file is the working guide for agents editing this repository. It reflects the current frontend behavior, the WordPress handoff path, and the current limitations around rebuilding imported WordPress articles inside an Astro app.

## Repo Areas That Matter Most

- `frontend/`: main React/Vite app used for research, content ideas, Content Studio, and WordPress export.
- `src/api/endpoints/research_topics.py`: topic and idea-burst orchestration, including category-aware content idea persistence.
- `src/api/endpoints/content_ideas.py`: publishing content ideas into `Titles`.
- `src/api/wordpress.py`: WordPress post/category sync endpoints.
- `src/api/internal_links.py`: internal-link suggestions based on imported WordPress posts.

## Research Workflow Target State

This repository currently contains multiple research and idea-generation paths. Agents should treat the process below as the preferred end-to-end product direction for Research work, even when the current implementation is only partially aligned.

### Core Principle

The Research workflow should optimize for solving real user problems, not just generating content. The system should start from category-specific user jobs, validate real search opportunity, and only then generate article or software ideas.

### End-to-End Research Process

1. Start with website context.
2. Build category-specific user jobs.
3. Split candidates into SEO, editorial, and software-first paths.
4. Validate SEO and software candidates using keyword data and live SERP evidence.
5. Cluster validated opportunities.
6. Generate ideas only from validated opportunities.
7. Attach final keyword packs before ideas reach Content Studio.

### Step 1: Website Context

Every research flow should begin with:

- website description
- primary category
- secondary category
- category descriptions when available
- target audience context
- trend context when available

The purpose of this step is to narrow opportunity discovery to the actual site niche. Agents should avoid broad topic ideation that is only loosely related to the selected category path.

### Step 2: Build User Jobs First

Before generating article titles, the system should derive 20 to 40 category-specific user jobs or repeated tasks.

Examples:

- "I need to compare X before buying."
- "I need to calculate Y."
- "I need to track Z over time."
- "I need to choose between A and B."

Why this matters:

- topics are abstract
- user jobs are functional
- user jobs align much better to article intent, software intent, and keyword validation

Agents should prefer job statements over broad topic lanes when designing or revising the Research flow.

### Step 3: Split Into Three Tracks

The preferred workflow has three explicit tracks.

#### SEO Track

Use this track for opportunities that should rank in search and can be validated with measurable demand.

#### Editorial Track

Use this track for strategic, brand, authority, or thought-leadership ideas that may not have strong keyword demand but are still valuable.

Editorial ideas should not be forced through SEO achievability gates.

#### Software-First Track

Use this track when the user job implies a repeated action or workflow that could be solved with a tool.

Strong trigger patterns include:

- calculate
- estimate
- compare
- track
- convert
- check
- plan
- generate

If a user job matches one of these patterns, the system should explicitly evaluate software opportunity instead of treating software ideas as a side effect of article clustering.

### Step 4: Validate Opportunities Before Idea Generation

For SEO and software-first candidates, validation must happen before idea generation.

Validation should include:

- measurable keyword demand
- keyword difficulty
- CPC or monetization signals
- SERP structure
- intent match
- signs of weak competition

Agents should strongly prefer real SERP evidence over heuristic-only viability labels.

### Achievability Framework

The preferred scoring model is a gated system:

1. eligibility gates
2. weighted ranking score

Do not let a flat weighted average rescue a poor-fit candidate.

#### Article Eligibility Gates

A candidate should normally pass all of these before article generation:

- `intent_match >= 0.65`
- `serp_weakness >= 0.35`
- `secondary_keyword_count >= 3`
- at least one exact or near-exact measurable keyword
- no severe domain mismatch

#### Software Eligibility Gates

A candidate should normally pass all of these before software generation:

- `intent_match >= 0.70`
- `software_pattern_score >= 0.60`
- `serp_gap >= 0.30`
- `feasibility_score >= 0.60`
- one clear primary workflow keyword
- at least two supporting workflow keywords

#### Editorial Eligibility Gates

Editorial ideas bypass SEO achievability gates and should instead require strong strategic value.

### Achievability Scoring Weights

For this website, the default priority is low competition over raw traffic volume. The goal is quick-rank opportunities, not simply the biggest keywords.

#### Article Achievability

```text
achievability_article =
  0.40 * serp_weakness
  + 0.25 * intent_match
  + 0.15 * kd_ease
  + 0.10 * monetization_fit
  + 0.10 * volume_score
```

#### Software Achievability

```text
achievability_software =
  0.25 * intent_match
  + 0.20 * serp_gap
  + 0.20 * software_pattern_score
  + 0.20 * feasibility_score
  + 0.10 * monetization_fit
  + 0.05 * volume_score
```

Where:

- `kd_ease = 1 - normalized_keyword_difficulty`
- higher is always better

#### Article Achievability Refinement

To reduce weak-SERP but off-brand opportunities, article scoring should also penalize niche drift.

```text
achievability_article =
  0.40 * serp_weakness
  + 0.25 * intent_match
  + 0.15 * kd_ease
  + 0.10 * monetization_fit
  + 0.10 * volume_score
  - 0.20 * niche_drift
```

### Subscore Definitions

#### Intent Match

How well the user job matches what the SERP is already rewarding.

- `1.0`: perfect match
- `0.75`: mostly aligned
- `0.50`: mixed
- `0.25`: weak
- `0.0`: wrong intent

#### SERP Weakness

How winnable the Top 10 results are for an article.

Useful ingredients:

- weak-result ratio
- low-authority ratio
- poor-intent-match ratio
- stale-result ratio
- weak-content-depth ratio

#### SERP Gap

How much the SERP suggests a tool should exist but does not yet have strong tool coverage.

This matters most for software-first opportunities.

#### Software Pattern Score

How strongly the job/query language implies a repeated workflow or interactive utility.

#### Feasibility Score

How realistic the software idea is for the current product and technical stack.

This should distinguish between:

- tools that can be shipped as client-side React functionality
- tools that need lightweight persistence or database integration
- tools that require heavy backend processing, external APIs, long-running jobs, or infrastructure that does not yet exist

Software ideas that are attractive in search but unrealistic to deliver in the current architecture should be downgraded, backlogged, or routed into an article-only outcome.

#### Monetization Fit

Blend signals such as:

- CPC
- affiliate potential
- lead-gen potential
- software upsell potential

#### Volume Score

Use a log scale so large keywords do not overpower winnability.

### Routing Rules

Preferred routing behavior:

- if article intent is weak but software intent is strong, reroute to software
- if the query is strategically valuable but weak for SEO, reroute to editorial
- if the user job is strong but the SERP is unwinnable, reject or backlog it
- if the SERP shows strong article demand and weak tool competition, article-first is acceptable
- if the SERP shows strong workflow intent but poor tool coverage, software-first is preferred
- if the software opportunity is search-attractive but technically unrealistic for the current stack, downgrade it to backlog, article-only, or a lower-feasibility software state

### Step 5: Cluster Validated Opportunities

Clustering should happen after validation, not before.

Important rule:

- cluster validated keyword opportunities, not broad abstract topics
- use clustering time to discover site-architecture fit, not only keyword similarity

This avoids forcing one idea out of every cluster regardless of quality.

During or immediately after clustering, the system should also inspect imported WordPress content to identify:

- likely parent articles
- likely child/supporting articles
- related category/tag neighborhoods
- candidate internal-link hubs

The goal is that each new idea has a likely place in the existing site architecture before it reaches Content Studio.

### Step 6: Generate Ideas From Winning Opportunities

Article and software ideas should be generated only after:

- the candidate passed its eligibility gates
- achievability was scored
- routing was decided

Agents should prefer:

- one or more ideas from strong opportunities
- zero ideas from weak opportunities

Do not force an exact one-idea-per-cluster contract if the underlying cluster is thin.

### Step 7: Final Keyword Attachment Before Content Studio

Before an idea is considered ready for Content Studio, it should have:

- one primary keyword
- three to eight secondary keywords
- at least three secondaries with measurable demand
- at least two secondaries aligned to the same user job and intent

If the system cannot attach a strong keyword pack, the idea should be marked as too thin and blocked from being treated as fully validated.

Suggested status:

- `cluster_too_thin`

### Validation Freshness

Validation must be treated as time-sensitive.

Recommended behavior:

- store `validated_at` on validation results and SERP snapshots
- define a TTL for validation freshness, typically 14 to 30 days depending on category volatility
- if a user attempts to generate or promote ideas from stale validation data, trigger a refresh-validation step before proceeding

Agents should not assume that persisted SERP evidence remains reliable indefinitely.

### Negative Feedback Loop

Rejected jobs and rejected opportunities should be treated as training signals for later runs.

Recommended persisted rejection reasons include:

- too broad
- off-brand
- wrong audience
- weak SERP
- low monetization
- technically impossible
- duplicate of existing content
- poor site-architecture fit

This feedback should be reusable as negative context for later job generation, candidate generation, and ranking cycles so the system does not keep proposing the same low-quality patterns.

### Suggested Final Routing States

Every validated candidate should end in one of these states:

- `article_ready`
- `software_ready`
- `article_plus_software`
- `editorial_only`
- `software_backlog_low_feasibility`
- `needs_more_keyword_validation`
- `rejected_low_achievability`

### Data To Persist For Research Validation

Agents extending the Research pipeline should strongly consider persisting these fields for each validated opportunity:

- `user_job`
- `candidate_type`
- `primary_keyword`
- `secondary_keywords`
- `intent_match_score`
- `serp_weakness_score`
- `serp_gap_score`
- `software_pattern_score`
- `monetization_fit_score`
- `volume_score`
- `kd_ease_score`
- `final_achievability_score`
- `routing_decision`
- `top_10_serp_snapshot`
- `validated_at`
- `reason_codes`
- `rejection_reason_tags`
- `internal_link_parent_candidates`
- `internal_link_child_candidates`
- `feasibility_score`

### Implementation Guidance For Agents

When reviewing or extending the Research system:

- prefer validation-first over generation-first
- prefer user jobs over broad topics
- treat editorial, SEO, and software as different tracks
- prioritize quick-rank, weak-SERP opportunities over high-volume, unwinnable ones
- feed SERP snapshots back into later LLM prompts when possible so idea generation uses actual competitive context
- avoid hiding weak opportunities behind heuristic-only labels such as "high viability" without SERP evidence
- add explicit validation freshness rules and refresh behavior for stale SERP data
- add rejection tagging and reuse it as a negative feedback loop for future discovery cycles
- include internal-link neighborhood discovery before ideas are considered ready
- include software feasibility as a real gate, not just a post-hoc implementation concern

## Research Implementation Plan

This section is the implementation plan for migrating the current Research system toward the target-state workflow above.

### Guiding Assumption

Large parts of the current research flow are likely to become obsolete or materially downgraded in importance.

In practical terms:

- the current broad topic-first workflow should no longer be treated as the long-term source of truth
- heuristic-only keyword viability should no longer be treated as a sufficient decision layer
- cluster-first idea generation should no longer be treated as the default planning model
- the legacy subtopic and idea-burst flows may remain temporarily for compatibility, but they should not define the target architecture

### Migration Goals

The implementation should achieve these outcomes:

1. build research around user jobs instead of broad topics
2. separate editorial, SEO, and software-first routes
3. validate opportunities before idea generation
4. score opportunities with explicit achievability logic
5. persist validation context so the UI can explain why an idea exists
6. remove or retire obsolete research paths once the new flow is stable

### Recommended Rollout Strategy

Use a phased migration, not a big-bang rewrite.

#### Phase 0: Audit and Freeze

Goal:

- inventory the current research paths
- identify what must be preserved temporarily
- stop adding new logic to the weakest legacy paths

Work:

- map all entry points that create or mutate research topics, subtopics, keyword candidates, clusters, and content ideas
- identify all UI surfaces that depend on legacy research artifacts
- mark current endpoints, services, and tables as one of:
  - keep and adapt
  - keep temporarily behind compatibility layer
  - deprecate
  - delete after migration
- add clear internal documentation for which path is currently considered primary during the migration

Deliverables:

- dependency map of Research-related frontend pages, services, backend endpoints, and tables
- deprecation list
- migration risk list

#### Phase 1: Introduce New Core Data Model

Goal:

- create durable primitives for the new process without breaking the UI yet

Suggested new logical entities:

- `research_user_jobs`
- `research_opportunity_candidates`
- `research_validation_runs`
- `research_serp_snapshots`
- `research_keyword_packs`
- `research_routing_decisions`

At minimum, the system needs to persist:

- website/category context used for discovery
- user job
- candidate type
- validation results
- achievability subscores
- final routing decision
- keyword pack
- top SERP evidence

Work:

- design new tables or extend current tables carefully
- define IDs and relationships between jobs, candidates, validations, ideas, and titles
- decide which current tables can be evolved versus replaced
- add migration scripts

Deliverables:

- schema plan
- migrations
- ERD or relationship notes

#### Phase 2: Build Job Discovery Layer

Goal:

- derive category-specific user jobs from project and category context before topic generation

Work:

- add a job discovery service that takes:
  - website description
  - primary category
  - secondary category
  - category descriptions
  - trend context
  - target audience
- generate 20 to 40 user jobs
- classify jobs into likely candidate tracks:
  - SEO
  - editorial
  - software-first

Important rule:

- jobs must be concrete, user-centered, and action-oriented
- avoid broad topic lanes as the first-class object

Deliverables:

- job-generation service
- job persistence
- job classification metadata

#### Phase 3: Build Opportunity Validation Layer

Goal:

- validate candidates before idea generation

Work:

- convert jobs into keyword and workflow candidates
- fetch keyword metrics
- fetch or derive SERP snapshots
- calculate:
  - intent match
  - SERP weakness
  - SERP gap
  - software pattern score
  - monetization fit
  - volume score
  - KD ease
- implement eligibility gates by candidate type
- implement final achievability score

Important rule:

- no candidate should reach idea generation without passing the correct validation path

Deliverables:

- validation service
- achievability scoring service
- routing rules service
- persisted reason codes and evidence

#### Phase 4: Generate Ideas From Validated Opportunities

Goal:

- generate content only from validated winners

Work:

- replace "one idea per cluster" logic with opportunity-driven generation
- allow:
  - zero ideas from weak opportunities
  - multiple ideas from a strong opportunity when justified
- generate:
  - article ideas from validated SEO opportunities
  - software ideas from validated workflow opportunities
  - editorial ideas from strategic editorial jobs

Important rule:

- generation should consume validation context and SERP context
- the LLM should see why the opportunity is attractive, not only the topic title

Deliverables:

- new generation prompts and payloads
- idea persistence updates
- routing-aware output metadata

#### Phase 5: Attach Final Keyword Packs

Goal:

- block thin ideas before they reach Content Studio

Work:

- assign one primary keyword plus three to eight secondary keywords
- require measurable support for the keyword pack
- mark thin opportunities explicitly

Suggested blocking statuses:

- `cluster_too_thin`
- `needs_more_keyword_validation`

Deliverables:

- keyword pack builder
- readiness gate before Content Studio publish/promotion

#### Phase 6: UI Migration

Goal:

- update the frontend to reflect the new mental model

Research UI should evolve from:

- topic list
- keyword run
- cluster selection
- idea generation

Toward:

- user jobs
- validated opportunities
- achievability and routing views
- article/software/editorial outcomes

Work:

- redesign Landing / New Research flow around jobs and candidate tracks
- redesign Topic Detail or replace it with opportunity detail views
- expose validation reasons and SERP evidence
- show routing state clearly
- show why a candidate was rejected, rerouted, or downgraded

Deliverables:

- updated page architecture
- revised services and types
- migration UI states for legacy data

#### Phase 7: Cutover and Cleanup

Goal:

- make the new path primary and retire obsolete paths

Work:

- feature-flag the new flow during rollout
- backfill or migrate important existing research data when feasible
- switch the default UI path to the new research model
- retire or delete obsolete endpoints, services, and tables

Deliverables:

- cutover checklist
- deprecation completion list
- cleanup PRs

## Research Implementation Checklist

This checklist is meant to be used during implementation work.

### Discovery and Planning

- Inventory all current Research entry points in frontend and backend.
- Inventory all current tables used by topics, subtopics, keyword candidates, clusters, and content ideas.
- List all legacy endpoints and services that will be affected.
- Define which parts are temporary compatibility paths versus target-state paths.
- Document feature flags needed for rollout.

### Data Model

- Define new or revised entities for user jobs, opportunity candidates, validations, SERP snapshots, routing decisions, and keyword packs.
- Add migrations for required tables and columns.
- Add indexes for candidate lookup, validation history, and idea routing.
- Decide how current `research_topics`, `content_ideas`, and `Titles` relate to the new entities.
- Define archival strategy for legacy records that should not be deleted immediately.

### Backend Services

- Build user job generation service.
- Build job classification service for SEO, editorial, and software-first routes.
- Build keyword candidate generation from jobs.
- Build SERP snapshot collection or ingestion layer.
- Build achievability scoring layer.
- Build routing rules layer.
- Build final keyword pack builder.
- Update idea generation endpoints to consume validated opportunities instead of raw broad topics or clusters.

### Frontend

- Redesign the New Research flow around jobs instead of broad topic candidates.
- Add UI states for candidate type and routing.
- Add UI for achievability score breakdown.
- Add UI for SERP evidence and validation reasons.
- Add UI states for rejected, rerouted, editorial-only, and thin-keyword-pack candidates.
- Ensure Content Studio still receives `topic_id`, `idea_metadata`, and category context until any replacement is complete.

### Persistence and Compatibility

- Preserve `topic_id`, `source_idea_id`, and `idea_metadata.category_context` until all downstream consumers are migrated.
- Preserve category path continuity in Content Studio and WordPress export.
- Add compatibility adapters if the UI still expects old cluster or topic payloads during rollout.
- Backfill validation context for important in-flight records if required for UX continuity.

### QA and Evaluation

- Test consumer-friendly sites and confirm the pipeline does not drift into enterprise or B2B keywords.
- Test software-trigger jobs and confirm they route to software-first when appropriate.
- Test editorial jobs and confirm they are not blocked by keyword gates.
- Test low-volume but weak-SERP opportunities and confirm they rank higher than high-volume unwinnable ones.
- Test thin keyword packs and confirm they are blocked before Content Studio.
- Verify category/subcategory context is preserved end to end.

## Research Cleanup Checklist

Cleanup work is part of the implementation, not a nice-to-have.

### Code Cleanup

- Identify and remove obsolete heuristic-only viability logic once replaced by real validation.
- Remove or downgrade broad topic-first generation as the primary architecture once jobs are live.
- Remove forced one-idea-per-cluster assumptions from backend prompts and UI copy.
- Remove auto-selection behavior that biases users toward the first cluster or first heuristic result if that UI still exists.
- Remove dead frontend components, services, and types that only support retired flows.
- Remove dead backend endpoints and helper functions after cutover.

### Data Cleanup

- Mark legacy research runs that are not compatible with the new model.
- Decide whether to archive, migrate, or hard-delete obsolete cluster/candidate data.
- Remove duplicate or redundant persistence fields where the new model becomes the source of truth.
- Clean up stale cached research artifacts once the new flow is stable.

### UX Cleanup

- Remove outdated copy that describes the old cluster-first workflow as the recommended process.
- Remove UI affordances that imply "Use as Seed and rerun" is the primary iteration method if that is no longer true.
- Replace vague viability labels with transparent score explanations and reason codes.

### Documentation Cleanup

- Update `AGENTS.md` again once the new system is primary.
- Add deprecation notes for old endpoints and tables.
- Document what legacy research data still means after migration.
- Document the final source-of-truth entities for Research.

## Research Cutover Checklist

Use this when preparing to make the new system primary.

- Feature flag is enabled in staging.
- New job discovery flow is stable.
- Validation and achievability scores are persisted.
- Routing decisions are visible in the UI.
- Idea generation consumes validated opportunities.
- Final keyword pack gating is enforced.
- Content Studio still receives required metadata.
- WordPress export category behavior is unaffected.
- Legacy paths are either hidden or clearly marked as deprecated.
- Cleanup tasks for retired code and data are scheduled or complete.

## Current Frontend State

### Content Studio

File: [frontend/src/pages/ContentStudio.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/pages/ContentStudio.tsx)

Recent behavior that must be preserved:

- Content Studio now resolves and displays the full category path, not just a single WordPress category ID.
- Category path resolution should prefer `Titles.idea_metadata.category_context.category_path`.
- If that is missing, it should fall back to:
  1. `Titles.topic_id`
  2. `Titles.source_idea_id -> content_ideas.topic_id`
  3. `research_topics.primary_category_id + secondary_category_id -> project_categories.name`
- `Titles` records created from ideas should carry `topic_id` and `idea_metadata` so downstream UI does not lose category/subcategory context.

Important implication:

- If an agent touches article loading in Content Studio, do not regress the `topic_id` and `source_idea_id` fallback chain.
- If an agent changes idea publishing or article persistence, verify that Level1 and Level2 category context still appears in Content Studio.

### WordPress Export Modal

File: [frontend/src/components/WordPressExportModal.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/components/WordPressExportModal.tsx)

Recent behavior that must be preserved:

- Category auto-selection is no longer single-source.
- The modal now merges category candidates from:
  1. `articleData.wordpress_category_id`
  2. previously saved export settings
  3. linked topic/project category mappings via `resolveLinkedWordPressCategoryIds`
- This is specifically to preserve Level2/subcategory handoff to WordPress.
- When the linked resolver returns both subcategory and primary category, both should remain selected.

Important implication:

- Do not reintroduce early returns after the first matching category.
- When editing WordPress export UX, remember that multiple category IDs are intentional.

### WordPress Service Expectations

File: [frontend/src/services/wordpressService.ts](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/services/wordpressService.ts)

Expected behavior:

- `publishToWordPress(...)` sends `categories: number[]` to WordPress, not a single category.
- `resolveLinkedWordPressCategoryIds(...)` should prefer subcategory first, then primary category.
- Publishing loopback updates `Titles` with post status, post URL, post ID, and canonicalized SEO title/description.

## Backend Persistence Rules

### Idea Burst Persistence

File: [src/api/endpoints/research_topics.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/research_topics.py)

Current expectations:

- New content ideas should persist `idea_metadata.category_context`.
- `category_context` should include:
  - `project_id`
  - `primary_category_id`
  - `secondary_category_id`
  - `primary_category_name`
  - `secondary_category_name`
  - `primary_category_description`
  - `secondary_category_description`
  - `category_path`
- This context exists to keep category/subcategory information available after the user leaves the Research view.

If an agent edits topic generation or idea persistence:

- Keep category/subcategory traceability intact.
- Do not collapse the context down to a single generic `category` string.

### Publishing Ideas Into Titles

File: [src/api/endpoints/content_ideas.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/content_ideas.py)

Current expectations:

- When a blog idea is published into `Titles`, the inserted record should include:
  - `source_idea_id`
  - `topic_id`
  - `idea_metadata`
  - WordPress mapping fields
  - keyword handoff fields
- This is required so Content Studio and WordPress export can still derive category path and topic lineage later.

If an agent edits this path:

- Verify that Titles created from content ideas still preserve topic linkage.
- Verify that category/subcategory context still reaches Content Studio and WordPress export.

## Category/Subcategory Contract

The current product expectation is:

- topic generation starts from project category/subcategory
- content ideas retain both Level1 and Level2 context
- Content Studio can recover and display that context
- WordPress export can send both matching categories when available

In practical terms:

- `primary_category_id` is Level1
- `secondary_category_id` is Level2
- `category_path` should usually be `Level1 / Level2`

If an agent sees only one generic `category` field being used, that is not enough for the current workflow.

## WordPress Import Reality

File: [src/api/wordpress.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/wordpress.py)

Current import behavior:

- `POST /api/wordpress/sync-posts` fetches recent posts from configured WordPress sites.
- Imported rows saved to `wordpress_imported_posts` currently contain only:
  - `user_id`
  - `wordpress_detail_id`
  - `post_id`
  - `title`
  - `link`
  - `excerpt`

What this means:

- The current sync is good enough for internal-link suggestions.
- It is not sufficient to recreate full articles inside an Astro app.
- There is no Astro app in this repository today, so any Astro rebuild flow is either external or still to be created.

## Recreating WordPress Articles In Astro

If the goal is to read WordPress articles and recreate them in an Astro app, agents should treat the current system as incomplete and extend it deliberately.

### Minimum Data Needed

To reconstruct articles in Astro reliably, imported records should also store:

- `slug`
- `content.rendered` or cleaned article HTML
- `date`
- `modified`
- `status`
- `author`
- `featured_media`
- resolved media URL
- category IDs and category names
- tag IDs and tag names
- SEO metadata if available
  - Yoast/RankMath title
  - meta description
  - canonical URL
- source domain

### Recommended Import Upgrade

When extending `sync-posts`, prefer storing richer post snapshots in `wordpress_imported_posts` or a new dedicated table such as `wordpress_imported_articles`.

Recommended snapshot fields:

- `post_id`
- `slug`
- `title`
- `excerpt`
- `content_html`
- `link`
- `published_at`
- `modified_at`
- `featured_image_url`
- `category_ids`
- `category_names`
- `tag_ids`
- `tag_names`
- `seo_title`
- `seo_description`
- `canonical_url`
- `raw_post_json`

### Astro Reconstruction Guidance

For an Astro app, agents should prefer this pipeline:

1. Sync full WordPress article data into Supabase.
2. Normalize HTML enough for Astro rendering.
3. Preserve canonical metadata and taxonomy.
4. Convert or wrap content into Astro-compatible templates.
5. Rebuild internal links using imported post inventory.

Important caveat:

- Do not promise lossless Astro recreation from the current `wordpress_imported_posts` table. Right now it only stores title, link, and excerpt.

## Internal Linking

File: [src/api/internal_links.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/internal_links.py)

Current behavior:

- Internal link suggestions are generated from `wordpress_imported_posts`.
- Because imported post data is currently shallow, suggestions are title/link-driven rather than full-content-aware.

If the WordPress import is upgraded for Astro reconstruction:

- Consider upgrading internal-link candidate quality to use article body, categories, tags, and slug context.

## Agent Checklist Before Shipping Changes

- If you touch Content Studio, verify category path still appears for Titles created from ideas.
- If you touch WordPress export, verify multiple category IDs can still be sent.
- If you touch idea persistence, verify `topic_id`, `source_idea_id`, and `idea_metadata.category_context` survive.
- If you touch WordPress sync, be explicit whether you are optimizing for internal-link suggestions or full Astro article reconstruction.
- If you add Astro-related work, document whether the Astro app lives inside this repo or is an external consumer.

## Known Limitation To Keep In Mind

The system currently has two different WordPress-related use cases:

- publish newly generated content to WordPress
- read existing WordPress content back for reuse

These are not symmetric today. Publishing is much richer than importing. Agents should not assume the import side already has enough data to recreate a post in Astro without schema and endpoint upgrades.
