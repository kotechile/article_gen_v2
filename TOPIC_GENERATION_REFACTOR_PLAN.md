# Topic Generation Refactor Plan

## Goal

Refactor the upper half of the pipeline so:

- topic generation is purely editorial and category-constrained
- AI topic generation does not spend DataForSEO credits
- Hot-in-the-news topic generation can keep DataForSEO for freshness discovery
- trend-aware topics remain fresh without pretending to be SEO-validated
- SEO keyword mining is concentrated where it matters most: content idea generation
- every content idea includes a short plain-language explanation of what the article is about


## Executive Summary

The pipeline currently mixes two different jobs:

1. editorial discovery
2. keyword validation

That mix is useful at the content-idea stage, but it is wasteful at the topic stage.

Topics are not final article targets. They are editorial clusters that help organize:

- subtopics
- article angles
- internal linking
- site architecture

Because downstream subtopic generation and content idea generation already run their own keyword mining, topic-stage DataForSEO adds cost without creating durable value.

Recommended direction:

- keep topic generation editorial-first
- remove only topic-stage keyword survey/validation work when those keyword outputs are not reused downstream
- keep DataForSEO in the Hot-in-the-news path for news, Quora, Reddit, and SERP discovery
- use category/sub-category names and descriptions as hard constraints
- keep trend signals as freshness inputs only
- move all serious keyword competition/volume optimization to content idea generation
- make content idea descriptions required and consistently persisted/rendered


## Key Findings

### 1. AI topic generation is already editorial-first

Current `AI generated` topics already flow through:

- `src/api/endpoints/ai.py`
- `src/services/topic_generation_brief_service.py`
- `src/services/editorial_topic_generation_service.py`

This path already:

- uses category and sub-category descriptions
- does not call DataForSEO
- produces editorial topic candidates, not keyword dossiers

Conclusion:

- this path should be preserved and tightened, not re-keyworded


### 2. The likely waste is in trend/news-informed topic generation

Current `Hot in the news` topic generation flows through:

- `src/api/trends.py`
- `src/services/trend_engine.py`

This path currently uses external trend/search collection, including DataForSEO retrieval, to synthesize topics.

But the resulting topics are still editorial objects:

- title
- rationale
- source_signals
- related_terms

Those topic-level SEO signals are not the source of truth for later article keyword selection.

Conclusion:

- news/trend topic generation should stay freshness-aware
- it should keep DataForSEO for discovery inputs such as news, Quora, Reddit, and SERP movement
- it should stop doing keyword survey/validation work whose outputs are not reused downstream
- topic-stage DataForSEO in this path should be treated as freshness and audience-signal discovery, not as downstream keyword truth


### 3. Real keyword leverage happens later

The most valuable keyword work happens in:

- subtopic keyword handoff
- content idea enrichment
- article keyword selection / refresh

Especially important:

- content idea generation already has the best article-angle context
- DataForSEO at this stage is tied to actual article candidates
- this is where volume, difficulty, CPC, and commercial viability matter

Conclusion:

- keep DataForSEO budget concentrated at content idea generation and idea keyword refresh


### 4. Content ideas already support a short explanation

Current code already references and renders content idea descriptions:

- `src/api/endpoints/research_topics.py`
- `src/api/endpoints/content_ideas.py`
- `frontend/src/types/idea-burst.ts`
- `frontend/src/components/IdeaBurstModal.tsx`

Conclusion:

- we probably do not need a new field
- the immediate fix is to make `description` mandatory, high quality, and consistently filled
- if database verification later shows a missing or unreliable schema in production, then a migration becomes a follow-up task rather than the starting assumption


## Problem Statement

We currently risk solving the wrong problem too early.

At the topic stage, we want:

- category fit
- editorial usefulness
- decision relevance
- freshness
- room for multiple subtopics and articles

At the topic stage, we do not need:

- keyword competition filtering
- search-volume gating
- keyword survey/validation that will be recomputed later anyway and is not handed off downstream

In the Hot-in-the-news path, we do still want:

- fresh news discovery
- forum/question discovery
- SERP-adjacent signal collection
- real-world evidence of emerging discussion

This early SEO effort creates three problems:

- wasted DataForSEO spend
- slower topic generation
- false confidence that topics are “validated” when downstream article keyword mining still has to happen independently


## Recommended Target Architecture

Target flow:

`project context + category/sub-category strategy + optional trend freshness -> editorial topics -> editorial subtopics -> DataForSEO-backed content ideas -> article keyword selection`

This means:

- Topics own editorial architecture.
- Subtopics own decision clusters and user-problem framing.
- Content ideas own keyword opportunity discovery.
- Final article generation owns the exact keyword strategy handoff.


## Design Principles

1. Topics are editorial clusters, not keyword targets.
2. Category and sub-category descriptions are hard constraints, not optional hints.
3. Trend signals are freshness inputs, not SEO validators.
4. DataForSEO should be used selectively:
   - allowed in Hot-in-the-news for discovery
   - avoided in AI-generated topics
   - concentrated on content ideas for keyword opportunity scoring
5. Every content idea should explain the article in one or two plain-language sentences.
6. Avoid consultant-speak at every level:
   - topics
   - subtopics
   - content ideas


## Proposed Changes

### Phase 1. Remove topic-stage SEO validation from the mental model and code path

#### A. AI-generated topics

Keep the current editorial architecture, but tighten prompt language so topics remain:

- category-aware
- non-technical in title
- broad enough to support subtopics
- specific enough to fit the site’s strategy

No DataForSEO should be added here.

#### B. Hot-in-the-news topics

Refactor `trend_engine` topic synthesis so it produces:

- trend-aware editorial topics
- not keyword-validated topics

Recommended behavior:

- use category/sub-category names and descriptions directly in prompt context
- use trend titles, DataForSEO news inputs, Quora/Reddit/forum signals, and SERP-style discovery to identify fresh angles
- do not keyword-score or keyword-gate topic candidates
- do not reject topics because they are not SEO-validated yet

Possible implementation options:

1. Best option:
   - keep DataForSEO where it helps discover fresh public discussion and search/news signals
   - remove only the keyword survey/validation parts whose outputs are not reused downstream
   - preserve topic outputs as editorial objects rather than keyword-scored objects

2. Acceptable transitional option:
   - keep existing trend ingestion for now
   - stop using non-reused DataForSEO keyword survey results as a validation mechanism for topic acceptance
   - treat those signals as optional “source_signals” only

Recommendation:

- choose option 1 if change surface is manageable
- choose option 2 if we want lower rollout risk and a faster refactor


### Phase 2. Make topic outputs explicitly editorial

Topic candidates should continue to persist fields like:

- `title`
- `rationale`
- `intent_bucket`
- `decision_focus`
- `angle_question`
- `value_layer_tags`
- `related_terms`
- `source_signals`

But they should not try to carry:

- keyword volume
- keyword difficulty
- topic-level SEO validation state

Reason:

- those fields are not the lasting source of truth for downstream content decisions


### Phase 3. Keep subtopics editorial-first

Subtopic generation should continue to use:

- topic title
- topic description
- project description
- category path
- primary category description
- secondary category description

Subtopics may still generate seed phrases for later use, but subtopics should not become the main owner of SEO truth.

Subtopic purpose should remain:

- cluster shape
- problem framing
- decision framing
- article family structure


### Phase 4. Concentrate keyword scoring DataForSEO on content idea generation

This is the most important shift.

Content idea generation should be the primary place where we:

- mine realistic keywords
- score volume
- score difficulty
- evaluate CPC/commercial paths
- choose lower-competition opportunities when possible

Why this stage is right:

- the article angle is concrete
- the user intent is clearer
- the site/category context is already known
- the keyword query can be simpler and more realistic

This also matches the current user goal:

- find the best keywords, ideally lower competition with meaningful traffic

Important distinction:

- Hot-in-the-news topic generation may still use DataForSEO for discovery
- content idea generation should be the primary place where DataForSEO decides keyword opportunity


### Phase 5. Make content idea description required

Every content idea should include a short explanation of what the article is about.

Recommended format:

- 1-2 sentences
- plain language
- focused on the article promise and reader value

This should be required in:

- LLM prompt instructions
- parser validation
- persistence payload
- frontend display

If parsing fails to provide a description:

- derive one deterministically from title + primary keyword + decision/outcome context

Example fallback:

- “This article explains [reader problem] and shows how to use [keyword/topic] to make a better decision.”


### Phase 6. Verify whether a migration is actually needed

Because `content_ideas.description` already appears in code, the migration decision should be based on schema verification, not assumption.

Verification tasks:

- confirm `content_ideas.description` exists in production schema
- confirm it is returned by list endpoints
- confirm publish/enrich flows do not strip it
- confirm null/empty descriptions are currently possible

Only add a migration if one of these is true:

- the column is missing in production
- the column exists but is too small / wrong type
- we decide to introduce a separate field such as `article_summary`

Default recommendation:

- do not start with a migration
- first make `description` required everywhere


## Concrete Refactor Plan

### Track A. Topic generation cleanup

- [ ] Audit `src/services/trend_engine.py` and list every place where DataForSEO is used specifically to help generate topic candidates.
- [ ] Classify those DataForSEO calls into:
  - freshness support
  - news/forum/serp discovery
  - keyword survey not reused downstream
  - SEO validation
  - redundant/not reused downstream
- [ ] Keep DataForSEO calls that contribute to Hot-in-the-news discovery quality.
- [ ] Remove or bypass only topic-stage keyword survey/validation logic whose outputs are not reused downstream.
- [ ] Ensure AI-generated topics and Hot-in-the-news topics both inject:
  - primary category name
  - primary category description
  - secondary category name
  - secondary category description
- [ ] Update topic-generation prompts to explicitly reject consultant-speak and over-technical titles.
- [ ] Ensure trend-aware topics are evaluated on:
  - category fit
  - editorial usefulness
  - freshness
  - downstream subtopic potential
  not on keyword metrics.

- [ ] Ensure AI-generated topics remain DataForSEO-free.
- [ ] Ensure Hot-in-the-news topics still benefit from DataForSEO discovery inputs without being keyword-gated.


### Track B. Content idea SEO ownership

- [ ] Review content idea generation prompt and make keyword mining the canonical SEO stage.
- [ ] Explicitly instruct keyword generation to prefer:
  - real search behavior
  - simpler phrases
  - lower competition when reasonable
  - meaningful volume over vanity complexity
- [ ] Verify that DataForSEO enrichment at the idea stage ranks candidate keywords by:
  - search volume
  - keyword difficulty
  - competition
  - commercial intent
- [ ] Ensure no upstream topic-level keyword score is overriding idea-level keyword selection.
- [ ] Add logging that makes it clear which stage produced the final SEO metrics shown on content ideas.


### Track C. Content idea description quality

- [ ] Make `DESCRIPTION` mandatory in the content idea generation prompt.
- [ ] Reject or repair parsed ideas that have empty descriptions.
- [ ] Add deterministic fallback description generation when LLM output omits the field.
- [ ] Confirm `content_ideas.description` is persisted on insert and retained on update/publish/enrich flows.
- [ ] Confirm the UI renders description consistently in all content idea views.
- [ ] Verify whether a migration is needed only after schema inspection.


### Track D. Observability and rollout safety

- [ ] Add logs showing whether topic generation used:
  - editorial-only context
  - trend-freshness context
  - any remaining optional external signals
- [ ] Add logs showing whether content idea SEO metrics came from:
  - DataForSEO exact keyword match
  - related keyword match
  - fallback estimate
- [ ] Add QA notes for topics/subtopics/content ideas so we can compare:
  - topical relevance
  - title clarity
  - keyword realism
  - SEO usefulness


## Suggested Rollout Order

1. Topic generation refactor
2. Trend-engine topic cleanup that preserves DataForSEO discovery inputs
3. Content idea prompt/description hardening
4. Content idea SEO logging improvements
5. Schema verification for `content_ideas.description`
6. Migration only if schema verification proves it is needed


## Success Criteria

The refactor is successful when:

- AI-generated topics no longer spend DataForSEO credits
- Hot-in-the-news topics still use DataForSEO for discovery and freshness
- topic generation no longer spends credits on keyword survey work that does not feed downstream stages
- AI-generated and trend-generated topics both stay tightly aligned to category/sub-category descriptions
- topics feel editorially useful instead of keyword-shaped
- content ideas remain the primary owner of keyword opportunity discovery
- content idea keywords show meaningful differentiation by idea
- every content idea includes a short explanation of what the article is about
- the system is faster and cheaper at the topic stage without reducing downstream SEO quality


## Files Most Likely to Change During Implementation

- `src/api/endpoints/ai.py`
- `src/api/trends.py`
- `src/services/topic_generation_brief_service.py`
- `src/services/editorial_topic_generation_service.py`
- `src/services/trend_engine.py`
- `src/services/editorial_subtopic_service.py`
- `src/api/endpoints/research_topics.py`
- `src/api/endpoints/content_ideas.py`
- `frontend/src/services/command-center.service.ts`
- `frontend/src/components/IdeaBurstModal.tsx`
- `frontend/src/types/idea-burst.ts`


## Recommendation

Best next implementation move:

- remove topic-stage SEO validation first while preserving Hot-in-the-news discovery inputs
- keep topic generation editorial and category-aware
- then harden content idea generation as the canonical SEO/keyword stage

That gives us the highest impact with the cleanest conceptual model:

- topics decide what we should cover
- content ideas decide how we can win search traffic
