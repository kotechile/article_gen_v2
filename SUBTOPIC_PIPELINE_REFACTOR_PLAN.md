# Subtopic Pipeline Refactor Plan

## Goal

Refactor subtopic generation so the system produces article titles that are:

- SEO-friendly: based on real keyword evidence with measurable volume, CPC, and manageable competition
- GEO-friendly: rich in entities, comparisons, factual claims, decision support, and citation potential
- Editorially strong: centered on user problems, decisions, frameworks, comparisons, calculators, and scenarios

The current pipeline over-relies on keyword expansion to define subtopics. That works for obvious keyword-driven niches, but it is brittle for abstract or strategic decision spaces where the user intent is real but the direct keyword recall is sparse.


## Current Problem

The current flow is effectively:

`topic -> LLM seed keywords -> DataForSEO related keywords -> filter -> cluster -> subtopics -> article ideas`

This makes DataForSEO and keyword recall responsible for generating the subtopics themselves.

That creates a few structural issues:

- Valuable editorial subtopics can disappear if keyword recall is weak.
- Abstract strategic topics become distorted into awkward keyword-like labels.
- Strong decision-oriented themes are rejected simply because the subtopic title is not itself a good search term.
- Missing metrics can flatten the entire pipeline.


## Recommended Target Flow

The refactored flow should be:

`topic -> editorial subtopics -> keyword mining per subtopic -> evidence scoring -> validated subtopics -> article ideas`

This keeps subtopics grounded in actual user decisions and uses DataForSEO for validation and enrichment rather than as the primary generator.


## Design Principles

1. Subtopics are editorial structures, not keyword phrases.
2. Keywords should support a subtopic, not define it.
3. DataForSEO should validate and enrich, not control the whole pipeline.
4. Missing SEO evidence should weaken confidence, not destroy the subtopic.
5. GEO value should be measured separately from classic SEO value.


## Proposed Architecture

### 1. Topic Brief

Build a normalized topic brief before any decomposition.

Fields should include:

- `topic_title`
- `topic_description`
- `project_name`
- `project_description`
- `category_path`
- `intent_bucket`
- `decision_focus`
- `angle_question`
- `value_layer_tags`
- `target_audience`
- `evidence_sources`
- `signal_terms`
- `trend_titles`
- `autocomplete_suggestions`

Purpose:

- give the LLM a stable, structured brief
- reduce drift toward generic keyword generation
- keep the output aligned with the project/site lens


### 2. Editorial Subtopic Generation

Generate subtopics first, without depending on DataForSEO.

Each subtopic should include:

- `title`
- `summary`
- `user_problem`
- `decision_type`
- `target_audience`
- `geo_entity_hints`
- `commercial_paths`
- `seed_search_phrases`
- `content_type_hint`

Preferred subtopic types:

- `comparison`
- `framework`
- `checklist`
- `audit`
- `calculator`
- `scenario`
- `decision`
- `problem`

Rules:

- titles should be human-readable and useful for downstream ideation
- titles should reflect user intent, not keyword-tool syntax
- avoid broad generic buckets unless there is a specific decision angle


### 3. Keyword Mining Per Subtopic

For each editorial subtopic, generate a set of compact search formulations and query DataForSEO.

Each subtopic should produce several keyword variants:

- direct phrase
- compact phrase
- head term
- geo/entity variant
- comparison variant
- commercial variant

Example:

Subtopic:
`Portugal NHR vs Greece Golden Visa Yield Audit`

Possible keyword variants:

- `Portugal NHR tax benefits`
- `Greece golden visa real estate`
- `Portugal vs Greece tax residency`
- `best tax residency Europe`
- `golden visa property yield`

Purpose:

- let DataForSEO validate a subtopic through nearby search demand
- avoid forcing the subtopic title itself to be the exact keyword


### 4. Evidence Scoring

Each subtopic should be scored from supporting evidence, not from the subtopic label alone.

Recommended score dimensions:

- `editorial_value_score`
- `seo_support_score`
- `serp_fit_score`
- `monetization_score`
- `geo_readiness_score`
- `final_subtopic_score`

Recommended states:

- `validated`
- `weak_seo_support`
- `editorial_only`

Interpretation:

- `validated`: strong editorial + SEO evidence
- `weak_seo_support`: strong editorial value, weak SEO support
- `editorial_only`: useful idea, but not yet backed by search evidence


### 5. Article Idea Generation

Generate article ideas from:

- editorial subtopic
- best supporting keyword cluster
- monetization path
- GEO/entity opportunities

Each article idea should include:

- `title`
- `primary_keyword`
- `secondary_keywords`
- `search_intent`
- `content_type`
- `affiliate_angle`
- `geo_entities`
- `supporting_subtopic_id`

Title generation should favor:

- clear entities
- concrete comparisons
- scenario framing
- framework/checklist structure
- titles that satisfy search intent but remain readable


## DataForSEO Recommendation

Do not remove DataForSEO.

Use it differently:

- keep it as the SEO validator and keyword miner
- stop using it as the main generator of subtopics
- allow it to strengthen, rerank, or weaken subtopics instead of deciding whether they exist

Summary:

- Keep `DataForSEO`
- Move it one stage later
- Reduce its power over editorial structure


## Database / Model Changes

The data model should separate editorial structure from SEO evidence.

Recommended entities:

- `research_topics`
- `subtopics`
- `subtopic_keyword_candidates`
- `subtopic_affiliate_evidence`
- `article_ideas`

### `subtopics`

Should store:

- editorial title
- summary
- user problem
- decision type
- intent bucket
- GEO hints
- final validation state
- final aggregate scores

### `subtopic_keyword_candidates`

Should store:

- `subtopic_id`
- `keyword`
- `variant_type`
- `search_volume`
- `cpc`
- `keyword_difficulty`
- `competition`
- `serp_intent`
- `is_selected_primary`
- `selection_reason`

### `subtopic_affiliate_evidence`

Should store:

- `subtopic_id`
- affiliate offers/programs
- confidence
- source
- monetization rationale


## Refactor Strategy

### Phase 1. Introduce Subtopics-First Pipeline

Objective:

- Generate editorial subtopics independently from DataForSEO.

Checklist:

- Add `generate_editorial_subtopics()` service
- Return strong subtopics even when SEO evidence is missing
- Stop treating the subtopic title as the primary search term
- Preserve current endpoint contract as much as possible during transition


### Phase 2. Add Keyword Evidence Per Subtopic

Objective:

- Attach keyword sets to each subtopic rather than using a single global keyword pool.

Checklist:

- Add `mine_keywords_for_subtopic()` service
- Generate multiple compact keyword variants per subtopic
- Query DataForSEO for each subtopic independently
- Save keyword evidence in a dedicated structure/table
- Select a best supporting keyword cluster per subtopic


### Phase 3. Add Proper Validation States and Scores

Objective:

- Distinguish editorial quality from SEO support.

Checklist:

- Add `score_subtopic_evidence()` service
- Compute separate scores for editorial value, SEO support, monetization, and GEO readiness
- Add `validated`, `weak_seo_support`, and `editorial_only` states
- Remove any misleading hardcoded confidence defaults where possible


### Phase 4. Refactor Article Idea Generation

Objective:

- Generate titles from subtopic + supporting keyword evidence.

Checklist:

- Update idea generation input contract
- Require `primary_keyword` and secondary keyword evidence
- Add GEO/entity-aware title heuristics
- Support comparison, framework, checklist, audit, scenario, and calculator title formats


### Phase 5. UI / Observability Improvements

Objective:

- Make it obvious why a subtopic is or is not SEO-backed.

Checklist:

- Show validation state on each subtopic
- Show selected primary keyword separately from subtopic title
- Show keyword evidence count
- Show SEO support score and GEO readiness score
- Show fallback reason when a subtopic is editorial-only
- Add structured logs for each pipeline stage


## Suggested Service Boundaries

Recommended services/modules:

- `topic_brief_builder`
- `editorial_subtopic_service`
- `subtopic_keyword_mining_service`
- `subtopic_scoring_service`
- `article_idea_generation_service`

Suggested responsibilities:

- `topic_brief_builder`: collect and normalize decomposition context
- `editorial_subtopic_service`: generate decision/problem subtopics
- `subtopic_keyword_mining_service`: generate query variants and fetch DataForSEO evidence
- `subtopic_scoring_service`: merge editorial and SEO evidence into final scores/states
- `article_idea_generation_service`: generate SEO/GEO-friendly titles from validated evidence


## Implementation Checklist

- Define the new subtopic-first pipeline contract
- Create `generate_editorial_subtopics()` service
- Build a reusable topic brief object
- Generate structured editorial subtopics from topic brief
- Add `mine_keywords_for_subtopic()` service
- Generate 3-8 compact keyword variants per subtopic
- Query DataForSEO per subtopic instead of globally only
- Add support for storing keyword evidence separately from subtopic record
- Add support for storing affiliate evidence per subtopic
- Add `score_subtopic_evidence()` service
- Add `seo_support_score`
- Add `geo_readiness_score`
- Add `editorial_value_score`
- Add `validated`, `weak_seo_support`, `editorial_only` status
- Refactor `/subtopics/generate` to use the new orchestration flow
- Refactor article idea generation to use the validated keyword cluster
- Update UI to distinguish subtopic title from primary keyword
- Add logs for seed generation, keyword mining, scoring, and final validation state
- Remove synthetic confidence defaults where they create misleading output
- Add tests for abstract decision topics where direct keyword recall is sparse


## What Success Looks Like

A strong result should look like this:

- the app generates subtopics that make sense to a human editor
- each subtopic carries keyword evidence, not just a title
- titles are generated from subtopic + keyword support
- low keyword recall no longer collapses the editorial structure
- GEO-friendly and SEO-friendly outputs can coexist instead of competing


## Final Recommendation

Keep DataForSEO, but demote it from "subtopic generator" to "subtopic validator and keyword miner."

Best approach:

- Use heuristics + LLM + project context to define the right subtopics first
- Use DataForSEO to validate and enrich those subtopics
- Use the best supporting keyword cluster to generate article ideas and titles

This architecture is more stable, more explainable, and much closer to how a strong editorial SEO strategist would actually work.
