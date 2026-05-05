# Topic-To-Idea Mode Split Implementation Plan

## Goal

Implement an end-to-end workflow that separates:

- `keyword_first` topics
- `editorial_first` topics
- `hybrid` topics

The system should stay automation-first, improve topic generation upstream, reduce dead-end keyword runs, and only expose manual seed creation when automation fails or when the user explicitly requests it.


## Why This Change Is Needed

The current topic generation path is optimized for good editorial topics, not for topics that are likely to survive keyword research.

Current behavior:

- topic generation produces broad editorial topics
- topic keyword research then tries to force those topics into keyword demand
- informational or abstract topics often return:
  - weak seeds
  - no qualifying keywords
  - low-signal clusters
  - no idea generation path

This creates two user-facing problems:

- good editorial topics appear “broken” because they do not produce measurable keywords
- keyword-first workflows get polluted by topics that should never have entered that path


## Recommended Target Architecture

Target flow:

`project/category context -> topic generation -> topic classification -> route by mode -> keyword research or editorial idea generation -> idea review -> publish`

Three topic modes:

1. `keyword_first`
- concrete, search-shaped, likely to produce measurable keyword demand
- use strict keyword pipeline

2. `editorial_first`
- strong informational/editorial potential
- do not require strong keyword validation to generate ideas

3. `hybrid`
- topic may have both search and editorial value
- try keyword path first
- if weak, fall back to editorial idea generation


## Current System Review

### Topic Generation

Current files:

- [src/services/editorial_topic_generation_service.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/services/editorial_topic_generation_service.py)
- [src/services/topic_generation_brief_service.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/services/topic_generation_brief_service.py)
- [src/api/endpoints/ai.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/ai.py)
- [frontend/src/pages/Landing.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/pages/Landing.tsx)

Current behavior:

- generates editorial topic candidates
- explicitly asks for broad topics, not keyword phrases
- prefers decision spaces, frameworks, comparisons, timing, and broad user problems
- does not score keyword viability
- does not classify topics by downstream path

### Topic Keyword Research

Current files:

- [src/services/topic_keyword_research_service.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/services/topic_keyword_research_service.py)
- [src/api/endpoints/research_topics.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/research_topics.py)
- [frontend/src/components/TopicKeywordResearchPanel.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/components/TopicKeywordResearchPanel.tsx)
- [frontend/src/pages/TopicDetail.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/pages/TopicDetail.tsx)

Current behavior:

- LLM-led seed generation
- DataForSEO expansion and enrichment
- strict filtering for viable candidates
- cluster generation
- cluster-to-idea generation

Current weakness:

- all topics are treated as if they should be keyword-friendly


## Product Decision

Implement the following recommended option:

- improve topic generation upstream
- classify topics at creation time
- branch the workflow by topic mode
- keep the system automated by default
- expose manual seeds only after failure or by explicit user action


## End-To-End Implementation Plan

## Phase 0: Safety, Baseline, and Success Criteria

### Objectives

- preserve existing `Titles` safety boundary
- benchmark the current topic-to-keyword-to-idea flow
- avoid regressions in Content Studio and WordPress handoff

### Tasks

- Capture 10-15 example topics across:
  - strong keyword topics
  - weak informational topics
  - mixed/borderline topics
- Save baseline outcomes for each:
  - generated topic title
  - seed count
  - active candidate count
  - cluster count
  - ideas generated
- Define success metrics:
  - lower rate of `0 candidate` keyword runs
  - better topic-mode fit
  - more useful ideas for informational topics
  - fewer junk clusters

### Deliverables

- baseline dataset
- before/after quality comparison sheet


## Phase 1: Add Topic Classification and Viability Metadata

### Objectives

- persist topic mode and keyword viability early
- make downstream routing explicit

### Data Model Changes

Add fields to `research_topics`:

- `topic_mode`
  - `keyword_first`
  - `editorial_first`
  - `hybrid`
- `keyword_viability_score`
  - integer or float 0-100
- `keyword_viability_label`
  - `high`
  - `medium`
  - `low`
- `topic_generation_reasoning`
  - short structured explanation

Optional JSON field:

- `topic_generation_metadata`
  - source prompt mode
  - mode confidence
  - viability breakdown
  - related search-likeness notes

### Files Likely To Change

- [src/api/endpoints/research_topics.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/research_topics.py)
- migration file in [migrations](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/migrations)
- frontend types:
  - [frontend/src/types/research.ts](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/types/research.ts)

### Deliverables

- schema migration
- API exposure for new fields


## Phase 2: Refactor Topic Generation Into Dual Prompt Modes

### Objectives

- stop producing only editorial topics
- generate topics with clearer downstream intent

### Recommended Design

Add two generator prompt modes inside topic generation:

1. `keyword-first topic generator`
- produces broad but search-shaped topics
- favors:
  - user problems
  - comparisons
  - costs
  - alternatives
  - product/tool/platform searches
  - recurring measurable user questions

2. `editorial-first topic generator`
- keeps current strengths
- favors:
  - strategic/informational themes
  - interpretation
  - high-quality editorial framing
  - broad decision contexts

### Implementation Options

#### Option A: Two separate service methods

- `generate_keyword_first_topics(...)`
- `generate_editorial_first_topics(...)`

Pros:

- cleanest prompt separation
- easiest to tune independently

#### Option B: One service with `generation_mode`

- `generate(brief, generation_mode="keyword_first" | "editorial_first" | "mixed")`

Pros:

- lower code duplication
- easier shared parsing

Recommendation:

- use Option B first
- split later only if prompt divergence becomes too large

### Files Likely To Change

- [src/services/editorial_topic_generation_service.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/services/editorial_topic_generation_service.py)
- [src/services/topic_generation_brief_service.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/services/topic_generation_brief_service.py)
- [src/api/endpoints/ai.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/ai.py)
- [frontend/src/pages/Landing.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/pages/Landing.tsx)

### Deliverables

- new prompt mode support
- generated topics with mode and viability metadata


## Phase 3: Add Automated Topic Viability Scoring

### Objectives

- estimate whether a topic should flow into keyword research
- prevent weak topics from entering the wrong path

### Viability Factors

Score each topic on:

- search-likeness
- concreteness
- user-problem clarity
- commercial/tool/comparison potential
- expected measurable demand
- category fit
- plain-language phrasing

### Output

Each generated topic gets:

- `topic_mode`
- `keyword_viability_score`
- `keyword_viability_label`
- short explanation:
  - why this is keyword-first, editorial-first, or hybrid

### Routing Rules

Suggested initial thresholds:

- `keyword_first`
  - viability `>= 70`
- `hybrid`
  - viability `45-69`
- `editorial_first`
  - viability `< 45`

### Deliverables

- score calculator
- routing metadata persisted with topic


## Phase 4: Update Topic Creation UX

### Objectives

- make topic mode visible before keyword research starts
- reduce confusion on Topic Detail pages

### UX Changes

On topic generation screens:

- show topic mode badge
- show keyword viability badge
- show short reasoning text

Examples:

- `Keyword-First · High Viability`
- `Hybrid · Medium Viability`
- `Editorial-First · Low Keyword Demand`

### On Topic Detail

Show a small strategy summary:

- `This topic is keyword-first and will use the keyword pipeline`
- `This topic is editorial-first and will generate ideas without requiring strong keyword demand`
- `This topic is hybrid and will try keyword research first`

### Files Likely To Change

- [frontend/src/pages/Landing.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/pages/Landing.tsx)
- [frontend/src/pages/Research.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/pages/Research.tsx)
- [frontend/src/pages/TopicDetail.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/pages/TopicDetail.tsx)


## Phase 5: Route Topic Detail Workflow By Mode

### Objectives

- stop forcing all topics through the same pipeline

### Routing Behavior

#### `keyword_first`

- current topic keyword research remains primary path
- article/software ideas come from clusters

#### `editorial_first`

- skip strict keyword pipeline as the default
- generate article ideas directly from:
  - topic title
  - rationale
  - intent bucket
  - decision focus
  - angle question
  - category context
  - related terms
- optionally run light keyword enrichment later, but not as a gate

#### `hybrid`

- try keyword pipeline first
- if `candidate_count == 0` or below threshold:
  - offer automatic fallback:
    - `Generate Editorial Ideas Instead`
  - optionally auto-fallback after one failed run

### Deliverables

- mode-aware Topic Detail orchestration
- fewer dead-end zero-candidate experiences


## Phase 6: Add Editorial-First Idea Generation Path

### Objectives

- support good informational topics even when keyword demand is weak

### New Backend Behavior

Add a dedicated idea-generation flow that does not depend on clusters:

- `POST /api/research-topics/<topic_id>/editorial-ideas/generate`

Prompt inputs:

- topic title
- rationale
- intent bucket
- decision focus
- angle question
- related terms
- category path
- value layer tags

Output:

- article ideas
- software ideas only when tool potential is plausible

### Storage

Persist into `content_ideas` with metadata:

- `generation_origin = "editorial_topic_pipeline_v1"`
- preserve:
  - `topic_id`
  - `idea_metadata.category_context`
  - downstream WordPress compatibility fields

### Files Likely To Change

- [src/api/endpoints/research_topics.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/research_topics.py)
- [frontend/src/services/topic-keyword-research.service.ts](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/services/topic-keyword-research.service.ts) or new topic-ideas service
- [frontend/src/pages/TopicDetail.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/pages/TopicDetail.tsx)


## Phase 7: Add Hybrid Fallback Logic

### Objectives

- keep hybrid topics automated
- reduce need for immediate manual intervention

### Behavior

If a `hybrid` topic returns:

- no active candidates
- no usable clusters
- or weak candidate density

Then automatically surface:

- `Keyword demand is weak for this topic`
- `Generate editorial ideas instead`

Optional second-step automation:

- auto-fallback without waiting for user confirmation
- only if topic mode is `hybrid` and viability is below a configured threshold


## Phase 8: Add Manual Seed Creation As Rescue Path

### Objectives

- keep the system automated by default
- expose manual control only when needed

### UX Behavior

Manual seed creation should appear:

- after keyword research failure
- after `0 candidate` result
- when user clicks:
  - `Use Manual Seeds`

### Recommended Capabilities

- edit current automated seeds
- add new seeds
- rerun keyword research from manual seeds
- replace previous research artifacts for the topic

### Important Rule

- manual seeds are not the default entry point
- they are a recovery/override path


## Phase 9: Improve Topic Generation Brief Inputs

### Objectives

- produce stronger topics before the LLM even starts

### Current Issue

`topic_generation_brief_service` mostly passes:

- project description
- category descriptions
- trend titles

But it does not yet explicitly capture:

- user jobs to be done
- search behavior patterns
- commercial vs informational intent preference
- desired topic mix

### Recommended Additions

Add optional brief fields:

- `generation_goal`
  - `keyword_opportunity`
  - `editorial_authority`
  - `balanced`
- `intent_preference`
  - `commercial`
  - `informational`
  - `mixed`
- `topic_mix`
  - counts or ratios by mode

This makes topic generation more controllable without becoming manual.


## Phase 10: Update Idea Review UX

### Objectives

- make keyword-first and editorial-first outputs feel like intentional modes, not broken variants

### UX Changes

In generated idea panels, show origin badges:

- `Keyword Pipeline`
- `Editorial Pipeline`
- `Hybrid Fallback`

For editorial-first ideas:

- do not show missing keyword metrics as failure
- show:
  - editorial rationale
  - user decision helped
  - category fit
  - internal-link potential

For keyword-first ideas:

- keep current metrics-heavy review


## Phase 11: Rollout Strategy

### Stage 1

- add schema
- add topic mode and viability scoring
- no workflow branching yet

### Stage 2

- introduce dual-mode topic generation
- show mode badges in UI

### Stage 3

- add editorial-first idea generation endpoint
- branch Topic Detail by topic mode

### Stage 4

- add hybrid fallback
- add manual seed rescue

### Stage 5

- tune prompts and thresholds using real topics
- compare:
  - candidate counts
  - cluster quality
  - idea usefulness
  - fallback frequency


## Risks and Mitigations

### Risk 1: More system complexity

Mitigation:

- use one research topic table
- add mode metadata instead of parallel entity models

### Risk 2: Users get confused by multiple flows

Mitigation:

- make mode explicit in UI
- show simple workflow messaging per topic

### Risk 3: Editorial-first topics feel “second-class”

Mitigation:

- give them a strong dedicated idea-generation path
- do not present zero-keyword outcomes as failure when the topic is editorial-first

### Risk 4: Hybrid routing becomes unpredictable

Mitigation:

- keep routing thresholds visible and logged
- preserve explainability in metadata


## Recommendation Summary

Recommended implementation order:

1. add `topic_mode` and `keyword_viability_score`
2. refactor topic generation into keyword-first and editorial-first modes
3. show mode/viability in topic creation and topic detail UI
4. add dedicated editorial-first idea generation
5. add hybrid fallback
6. add manual seed rescue only after failure or by explicit request

This is the cleanest path because it improves:

- topic quality
- routing accuracy
- automation rate
- informational-topic handling
- user trust

without making manual intervention the default.


## Execution Roadmap

## Milestone 0: Baseline and Design Lock

### Outcome

- shared agreement on success metrics
- sample topics captured for before/after comparison
- routing model approved

### Scope

- collect baseline topic examples
- define keyword-first vs editorial-first vs hybrid acceptance criteria
- confirm rollout strategy and feature flag approach

### Dependencies

- none

### Complexity

- low

### Definition of Done

- baseline dataset documented
- thresholds and mode definitions agreed
- implementation sequence approved


## Milestone 1: Schema and Type Foundation

### Outcome

- `research_topics` can persist mode and viability metadata

### Scope

- add `topic_mode`
- add `keyword_viability_score`
- add `keyword_viability_label`
- add `topic_generation_reasoning`
- optionally add `topic_generation_metadata`
- update API serializers
- update frontend types

### Dependencies

- Milestone 0

### Complexity

- medium

### Definition of Done

- schema migration applied
- backend CRUD supports new fields
- frontend types compile with new topic fields


## Milestone 2: Topic Generation Refactor

### Outcome

- topic generation can produce keyword-first and editorial-first candidates intentionally

### Scope

- refactor topic-generation prompt/service to support generation modes
- add mode-aware prompt instructions
- preserve existing editorial strength while adding keyword-shaped generation

### Dependencies

- Milestone 1

### Complexity

- medium-high

### Definition of Done

- AI topic generation returns mode-tagged topic candidates
- generated topics remain category-aware and plain-language


## Milestone 3: Automated Viability Scoring

### Outcome

- each topic gets an automated estimate of keyword potential and routing mode

### Scope

- add viability scoring logic
- persist `topic_mode`, `keyword_viability_score`, and reasoning
- define initial thresholds

### Dependencies

- Milestone 2

### Complexity

- medium

### Definition of Done

- generated topics are auto-classified
- classification is stored and exposed in APIs
- score explanations are inspectable for debugging


## Milestone 4: Topic Creation UX Upgrade

### Outcome

- users can see topic mode and viability before entering topic detail

### Scope

- add mode/viability badges to topic generation UI
- show reasoning snippets
- surface topic mode in research listings and topic detail

### Dependencies

- Milestone 3

### Complexity

- medium

### Definition of Done

- topic cards and generation screens show mode and viability
- topic detail explains expected workflow path


## Milestone 5: Mode-Aware Topic Detail Routing

### Outcome

- Topic Detail no longer forces every topic into the strict keyword path

### Scope

- branch behavior by `topic_mode`
- `keyword_first`: current keyword pipeline
- `editorial_first`: direct editorial idea path
- `hybrid`: keyword first with fallback support

### Dependencies

- Milestone 4

### Complexity

- medium-high

### Definition of Done

- topic detail shows the right workflow based on mode
- dead-end keyword flows are reduced for weak informational topics


## Milestone 6: Editorial-First Idea Generation

### Outcome

- informational/editorial topics can still generate ideas even with weak keyword demand

### Scope

- add backend endpoint for editorial idea generation
- persist ideas into `content_ideas`
- preserve `topic_id` and `idea_metadata.category_context`
- add frontend review/publish support for editorial-origin ideas

### Dependencies

- Milestone 5

### Complexity

- high

### Definition of Done

- editorial-first topics can produce article ideas end to end
- ideas can be reviewed and published into `Titles`
- downstream category/WordPress contracts remain intact


## Milestone 7: Hybrid Fallback

### Outcome

- hybrid topics recover automatically when keyword demand is weak

### Scope

- detect weak keyword runs
- surface or auto-trigger editorial fallback
- keep idea review coherent in the UI

### Dependencies

- Milestone 6

### Complexity

- medium

### Definition of Done

- hybrid topics no longer strand users after a weak keyword run
- fallback behavior is logged and explainable


## Milestone 8: Manual Seed Rescue

### Outcome

- users can rescue a strategically good topic when automation fails

### Scope

- add manual seed editor
- allow rerun from manual seeds
- preserve automation-first default

### Dependencies

- Milestone 5

### Complexity

- medium

### Definition of Done

- manual seeds appear only after failure/weak runs or explicit user request
- manual reruns replace research artifacts safely without affecting `Titles`


## Milestone 9: Topic Brief and Prompt Enrichment

### Outcome

- topic generation receives better upstream guidance and produces stronger topics

### Scope

- enrich topic-generation brief with:
  - generation goal
  - intent preference
  - topic mix
  - optional user jobs to be done
- tune prompts using real outcome data

### Dependencies

- Milestones 2 through 8 provide the data needed for tuning

### Complexity

- medium

### Definition of Done

- topic creation produces fewer abstract dead-end topics
- topic-mode distribution aligns better with downstream success


## Milestone 10: QA, Tuning, and Cutover

### Outcome

- new mode split is stable enough to become the default workflow

### Scope

- compare before/after baseline results
- validate Content Studio and WordPress compatibility
- tune thresholds and prompts
- retire or demote old assumptions in the UI

### Dependencies

- Milestones 1 through 9

### Complexity

- medium

### Definition of Done

- measurable improvement in:
  - keyword-run success rate
  - informational-topic idea generation success
  - reduced junk clusters
- rollout decision made for full default adoption


## Dependency Summary

- Milestone 1 is the foundation for all later routing and UX work.
- Milestones 2 and 3 are the core upstream fixes.
- Milestones 5, 6, and 7 are the core downstream workflow changes.
- Milestone 8 is valuable but not on the critical path for the automated default.
- Milestone 10 should happen only after real usage across all three topic modes.


## Recommended Build Order

### Phase A: Upstream Intelligence

- Milestone 0
- Milestone 1
- Milestone 2
- Milestone 3

### Phase B: Workflow Routing

- Milestone 4
- Milestone 5
- Milestone 6
- Milestone 7

### Phase C: Recovery and Tuning

- Milestone 8
- Milestone 9
- Milestone 10


## Critical Path

The highest-value critical path is:

1. add topic mode schema
2. improve topic generation
3. classify topics automatically
4. route topic detail by mode
5. implement editorial-first idea generation
6. add hybrid fallback

If time is constrained, manual seeds should be delayed until after this path is working.


## Suggested Engineering Split

### Backend Track

- schema and serializer changes
- topic generation refactor
- viability scoring
- editorial-first idea endpoint
- hybrid fallback logic
- manual seed rerun support

### Frontend Track

- topic generation UI badges
- research list/topic detail mode display
- mode-specific workflow panels
- editorial idea review path
- hybrid fallback UX
- manual seed rescue UI

### QA / Product Validation Track

- baseline comparison set
- topic-mode calibration
- prompt tuning
- downstream Content Studio / WordPress verification


## Sprint Delivery Plan

## Sprint 1: Foundation and Metadata

### Goal

- make topic mode and viability first-class concepts in the data model and API

### Suggested Tickets

1. Add topic mode schema fields
- add `topic_mode`
- add `keyword_viability_score`
- add `keyword_viability_label`
- add `topic_generation_reasoning`
- optional `topic_generation_metadata`

2. Expose topic mode fields in API responses
- update topic list/get/create/update serializers
- preserve backward compatibility

3. Update frontend types for topic mode metadata
- update research topic interfaces
- ensure existing pages compile cleanly

4. Create baseline fixtures and evaluation set
- capture 10-15 real topics
- document current candidate/cluster/idea outcomes

### Exit Criteria

- schema applied
- CRUD works end to end
- baseline sample set documented


## Sprint 2: Topic Generation Refactor

### Goal

- generate better topics upstream and classify them automatically

### Suggested Tickets

5. Refactor topic generation service to support prompt modes
- `keyword_first`
- `editorial_first`
- `mixed`

6. Add keyword-first topic generation prompt
- concrete, search-shaped, plain-language topics
- category-aware without becoming keyword lists

7. Keep editorial-first topic generation prompt but tighten it
- preserve strategic/editorial strength
- reduce overly abstract titles

8. Add automated topic viability scoring
- score search-likeness
- score commercial/tool potential
- score concreteness
- map to `keyword_first` / `hybrid` / `editorial_first`

9. Persist topic mode and viability with generated topics
- AI-generated topics
- optionally extend to trend-generated topics in same sprint if feasible

### Exit Criteria

- topic generation produces mode-tagged topics
- viability metadata is stored and inspectable


## Sprint 3: Topic Creation and Topic Detail UX

### Goal

- make topic mode visible and route users correctly before they hit dead ends

### Suggested Tickets

10. Show topic mode and viability in topic generation UI
- badges
- reasoning snippet

11. Show topic mode and viability in research list cards
- make keyword/editorial differences visible at a glance

12. Add topic mode summary to Topic Detail
- explain expected path:
  - keyword-first
  - editorial-first
  - hybrid

13. Add Topic Detail routing shell
- branch visible sections by topic mode
- keep current keyword UI for keyword-first

### Exit Criteria

- users can tell which workflow a topic belongs to
- Topic Detail is mode-aware at the UI level


## Sprint 4: Editorial-First Idea Generation

### Goal

- let informational topics generate useful ideas without strong keyword demand

### Suggested Tickets

14. Add backend editorial idea generation endpoint
- topic-based article ideas
- optional software ideas when tool potential exists

15. Persist editorial-first ideas into `content_ideas`
- preserve `topic_id`
- preserve `idea_metadata.category_context`
- preserve downstream compatibility for `Titles`

16. Add frontend editorial idea generation action in Topic Detail
- `Generate Editorial Ideas`
- loading, error, and success states

17. Add editorial-origin review and publish support
- reuse existing review/publish path where possible
- label origin as `Editorial Pipeline`

### Exit Criteria

- editorial-first topics can generate, review, and publish ideas end to end


## Sprint 5: Hybrid Fallback

### Goal

- recover automatically when a mixed topic has weak keyword demand

### Suggested Tickets

18. Detect weak keyword-run outcomes
- zero active candidates
- zero usable clusters
- weak candidate density threshold

19. Add hybrid fallback CTA in Topic Detail
- `Generate Editorial Ideas Instead`

20. Add optional auto-fallback behavior
- trigger only for `hybrid`
- log fallback reason

21. Make review panel origin-aware
- `Keyword Pipeline`
- `Editorial Pipeline`
- `Hybrid Fallback`

### Exit Criteria

- hybrid topics no longer dead-end after weak keyword runs


## Sprint 6: Manual Seed Rescue and Brief Enrichment

### Goal

- add controlled rescue paths and improve automation quality further

### Suggested Tickets

22. Add manual seed editor UI
- hidden by default
- available after failure or user request

23. Add rerun-from-manual-seeds backend path
- replace topic research artifacts safely
- never touch `Titles`

24. Enrich topic generation brief inputs
- generation goal
- intent preference
- topic mix
- optional jobs-to-be-done

25. Tune prompts with real outcome data
- refine mode prompts
- refine viability scoring thresholds

### Exit Criteria

- manual rescue exists without becoming the default workflow
- topic-generation quality improves from observed usage


## Sprint 7: QA, Cutover, and Cleanup

### Goal

- validate quality and make the new mode-split workflow the primary experience

### Suggested Tickets

26. Run baseline before/after comparisons
- candidate counts
- cluster quality
- idea usefulness
- editorial fallback frequency

27. Verify downstream compatibility
- Content Studio category path
- `Titles.topic_id`
- `Titles.source_idea_id`
- WordPress export categories

28. Clean up old topic-detail assumptions
- reduce misleading keyword-only messaging
- demote irrelevant UX for editorial-first topics

29. Final threshold tuning and rollout decision
- adjust topic-mode thresholds
- decide whether hybrid auto-fallback stays default

### Exit Criteria

- measurable improvement documented
- rollout approved
- UX reflects the new default model


## Estimated Critical Path

If we need the minimum high-value path, the shortest route is:

1. Sprint 1
2. Sprint 2
3. Sprint 3
4. Sprint 4
5. Sprint 5

That gets us:

- smarter topics
- automatic classification
- route-aware topic detail
- editorial-first idea generation
- hybrid fallback

Manual seeds and deeper prompt enrichment can come afterward.


## Suggested Work Package Sizes

Use these rough sizes to help planning:

- `S`
  - serializer/type/UI badge updates
- `M`
  - single endpoint additions
  - prompt refactors
  - topic detail routing changes
- `L`
  - editorial-first idea generation path
  - hybrid fallback orchestration
  - manual seed rescue flow


## Subsystem Checklist

## Schema and Data Model

Primary area:

- [migrations](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/migrations)

Checklist:

- [ ] Add `topic_mode` to `research_topics`
- [ ] Add `keyword_viability_score` to `research_topics`
- [ ] Add `keyword_viability_label` to `research_topics`
- [ ] Add `topic_generation_reasoning` to `research_topics`
- [ ] Add `topic_generation_metadata` JSON field if needed
- [ ] Decide whether editorial idea generation needs any new persistence metadata in `content_ideas`
- [ ] Confirm no migration touches or deletes `Titles`


## Topic Generation Backend

Primary files:

- [src/services/editorial_topic_generation_service.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/services/editorial_topic_generation_service.py)
- [src/services/topic_generation_brief_service.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/services/topic_generation_brief_service.py)
- [src/api/endpoints/ai.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/ai.py)

Checklist:

- [ ] Add prompt support for `keyword_first`
- [ ] Add prompt support for `editorial_first`
- [ ] Add `mixed` or routing-aware topic generation mode if desired
- [ ] Add automated viability scoring after generation
- [ ] Add topic mode classification logic
- [ ] Persist topic mode and viability into saved topic candidates
- [ ] Enrich brief inputs with:
- `generation_goal`
- `intent_preference`
- `topic_mix`
- optional jobs-to-be-done
- [ ] Reduce generation of abstract topics that are poor keyword candidates


## Topic Management API

Primary file:

- [src/api/endpoints/research_topics.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/research_topics.py)

Checklist:

- [ ] Expose new topic mode fields in list/get/create/update responses
- [ ] Support mode-aware topic detail responses if needed
- [ ] Add editorial-first idea generation endpoint
- [ ] Add hybrid fallback support in endpoint contracts if needed
- [ ] Preserve `topic_id`, `source_idea_id`, and `idea_metadata.category_context`
- [ ] Keep downstream Content Studio and WordPress compatibility intact


## Keyword Research Backend

Primary file:

- [src/services/topic_keyword_research_service.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/services/topic_keyword_research_service.py)

Checklist:

- [ ] Respect `topic_mode` when deciding whether keyword research is primary, optional, or fallback-only
- [ ] Add mode-aware thresholds for weak keyword runs
- [ ] Surface signals needed by hybrid fallback
- [ ] Support rerun from manual seeds
- [ ] Preserve automation-first behavior for keyword-first topics
- [ ] Continue logging raw seed generation diagnostics for debugging


## Idea Persistence and Publishing

Primary files:

- [src/api/endpoints/content_ideas.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/content_ideas.py)
- [src/api/endpoints/research_topics.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/research_topics.py)

Checklist:

- [ ] Persist editorial-first ideas into `content_ideas`
- [ ] Add `generation_origin = "editorial_topic_pipeline_v1"` metadata
- [ ] Preserve `topic_id` on saved ideas
- [ ] Preserve `idea_metadata.category_context`
- [ ] Ensure ideas still publish correctly into `Titles`
- [ ] Verify `Titles` records still contain topic lineage and WordPress mapping context


## Frontend Types and Services

Primary files:

- [frontend/src/types/research.ts](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/types/research.ts)
- [frontend/src/services/research-topics.service.ts](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/services/research-topics.service.ts)
- [frontend/src/services/topic-keyword-research.service.ts](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/services/topic-keyword-research.service.ts)

Checklist:

- [ ] Add topic mode and viability fields to research topic types
- [ ] Add service methods for editorial-first idea generation
- [ ] Add service methods for manual-seed reruns if needed
- [ ] Add any new response types for fallback/editorial idea generation


## Topic Creation UI

Primary file:

- [frontend/src/pages/Landing.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/pages/Landing.tsx)

Checklist:

- [ ] Show topic mode badges in AI-generated topic candidate UI
- [ ] Show keyword viability badges
- [ ] Show short reasoning snippets
- [ ] Optionally allow generation-goal controls:
- `keyword_opportunity`
- `editorial_authority`
- `balanced`
- [ ] Keep manual topic creation compatible with new topic metadata defaults


## Research Topic List UI

Primary file:

- [frontend/src/pages/Research.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/pages/Research.tsx)

Checklist:

- [ ] Show topic mode in topic cards
- [ ] Show viability at-a-glance
- [ ] Make it clear which topics are likely to use keyword-first vs editorial-first paths


## Topic Detail UI

Primary file:

- [frontend/src/pages/TopicDetail.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/pages/TopicDetail.tsx)

Checklist:

- [ ] Add topic mode strategy summary
- [ ] Route topic detail sections by `topic_mode`
- [ ] Keep keyword-first UI as primary for keyword-first topics
- [ ] Add editorial-first generation CTA for editorial topics
- [ ] Add hybrid fallback CTA or auto-fallback UX
- [ ] Add manual seed rescue UI only after failure or by user request
- [ ] Avoid showing lack of keyword metrics as failure for editorial-first topics


## Idea Review UI

Primary files:

- [frontend/src/components/GeneratedIdeasPanel.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/components/GeneratedIdeasPanel.tsx)
- [frontend/src/pages/TopicDetail.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/pages/TopicDetail.tsx)

Checklist:

- [ ] Show origin badges:
- `Keyword Pipeline`
- `Editorial Pipeline`
- `Hybrid Fallback`
- [ ] Add review support for editorial-origin ideas
- [ ] Keep publish-to-`Titles` flow shared where possible
- [ ] Make metrics-heavy review optional for editorial-first ideas


## QA and Validation

Cross-cutting areas:

- topic generation
- keyword research
- idea generation
- Content Studio
- WordPress export

Checklist:

- [ ] Compare baseline before/after for sample topics
- [ ] Confirm keyword-first topics produce stronger candidate runs
- [ ] Confirm editorial-first topics can still generate ideas with weak keyword demand
- [ ] Confirm hybrid topics recover via fallback
- [ ] Confirm Content Studio still resolves category path correctly
- [ ] Confirm WordPress export still preserves multi-category handoff
- [ ] Confirm no path deletes `Titles`


## Implementation Checklist

- [ ] Add `topic_mode`, `keyword_viability_score`, `keyword_viability_label`, and `topic_generation_metadata` to `research_topics`
- [ ] Update backend serializers and frontend types for new topic fields
- [ ] Refactor topic generation service to support `keyword_first`, `editorial_first`, and `mixed` prompt modes
- [ ] Add automated topic viability scoring after topic generation
- [ ] Persist topic mode and viability with generated topics
- [ ] Show topic mode and viability in topic generation UI
- [ ] Show topic mode strategy summary in Topic Detail
- [ ] Add routing logic in Topic Detail for `keyword_first`, `editorial_first`, and `hybrid`
- [ ] Add `editorial_first` idea-generation endpoint and persistence path
- [ ] Preserve `topic_id`, `source_idea_id`, and `idea_metadata.category_context` through new idea-generation paths
- [ ] Add hybrid fallback from failed/weak keyword runs into editorial idea generation
- [ ] Add manual seed creation UI only for failed/weak runs or explicit user request
- [ ] Add telemetry/logging for:
- topic mode chosen
- viability score
- keyword candidate count
- fallback activation
- manual seed usage
- [ ] Validate Content Studio and WordPress export still preserve category context and topic lineage
- [ ] Compare before/after quality on a fixed topic sample set
