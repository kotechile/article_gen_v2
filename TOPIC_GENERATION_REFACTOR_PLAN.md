# Topic Generation Refactor Plan

## Goal

Refactor topic generation so research topics are created as category-aware editorial opportunities instead of early-stage keyword guesses.

The new topic pipeline should:

- use project, category, and sub-category context as the primary constraint
- generate editorial topics that can support multiple subtopics and articles
- avoid DataForSEO at the topic stage
- preserve trend awareness without letting trends override category strategy
- produce topics that are better inputs for downstream subtopic and content-idea generation


## Current Problem

The current `propose-topics` flow is still too thin and too generic:

- it receives a flattened `niche_description`
- it only lightly uses category names
- it treats topics as broad seed ideas instead of structured editorial opportunities
- it does not consistently use category and sub-category descriptions
- it can drift into interesting but weakly aligned topics

This makes topics less useful as the top-level editorial architecture for the site.


## Recommended Target Flow

The target flow should be:

`project context + category strategy + trend signals -> editorial topic brief -> editorial topic candidates -> research topics`

This means:

- topics are generated from editorial strategy first
- subtopics are generated from topics later
- keyword validation is deferred to the content-idea stage where it creates the most value


## Design Principles

1. Topics are editorial clusters, not keywords.
2. Category and sub-category act as hard strategic constraints.
3. Topics should describe a decision space, framework, comparison path, audit path, or recurring problem.
4. Trend data is a directional signal, not the primary generator.
5. DataForSEO should not be used for topic generation.


## Proposed Architecture

### 1. Topic Brief

Build a normalized brief before calling the LLM.

Fields should include:

- `project_name`
- `project_description`
- `primary_category_name`
- `primary_category_description`
- `secondary_category_name`
- `secondary_category_description`
- `category_path`
- `category_strategy_hint`
- `trend_titles`
- `count`

Purpose:

- keep the generator tightly aligned with the site and its taxonomy
- reduce generic brainstorming
- make the topic generator reusable and testable


### 2. Editorial Topic Generation

Generate topics as structured editorial opportunities.

Each topic candidate should include:

- `title`
- `rationale`
- `intent_bucket`
- `decision_focus`
- `angle_question`
- `value_layer_tags`
- `related_terms`
- `source_signals`

Topic titles should be:

- broad enough to support multiple subtopics
- narrow enough to clearly fit one category/sub-category lane
- useful as a research cluster, not just as a phrase

Preferred topic patterns:

- frameworks
- comparisons
- timing decisions
- acquisition strategy
- cost/value audits
- decision systems
- scenario planning


### 3. Category Guardrails

The generator must use category descriptions directly, not only names.

Example:

- primary category = `Real Assets and Living`
- sub-category = `Acquisition Strategy`

This should bias topic generation toward:

- acquisition timing
- cost of ownership
- entry criteria
- hold vs buy tradeoffs
- valuation frameworks
- downside protection

It should explicitly avoid drifting into:

- generic lifestyle admiration topics
- broad news commentary with weak decision value
- keyword-shaped but strategically weak ideas


### 4. Trend Usage

Trend signals should remain available, but only as supporting context.

Trend titles should:

- inspire topical freshness
- influence examples and framing
- help identify new decision contexts

Trend titles should not:

- override category fit
- turn the pipeline into a news-topic generator


### 5. Downstream Relationship

The topic pipeline should hand off stronger editorial structure to:

- subtopic generation
- content idea generation

Recommended division of labor:

- `Topics`: editorial architecture
- `Subtopics`: article-cluster structure
- `Content ideas`: final SEO/GEO optimization


## DataForSEO Recommendation

Do not use DataForSEO in topic generation.

Reason:

- it is too early in the workflow
- topics are an editorial organization problem, not a keyword-validation problem
- calling DataForSEO here adds cost without proportionate value
- exact keyword quality matters much more at the content-idea level

Use DataForSEO mainly in:

- content-idea generation
- optional later-stage keyword enrichment


## Implementation Checklist

- [ ] Add a dedicated backend service to build a topic-generation brief
- [ ] Add a dedicated backend service to generate editorial topics from the brief
- [ ] Refactor `/api/ai/propose-topics` to use the new service
- [ ] Pass category descriptions from the frontend topic-generation flow
- [ ] Pass project/site description explicitly instead of flattening everything into `niche_description`
- [ ] Pass recent trend titles as optional supporting context
- [ ] Add deterministic editorial fallback topics if LLM parsing fails
- [ ] Verify backend compile
- [ ] Verify frontend build
- [ ] Push and deploy


## Expected Outcome

After this refactor, generated topics should:

- feel more aligned with the project taxonomy
- be more useful as top-level research clusters
- create better downstream subtopics
- reduce wasted DataForSEO usage
- make the whole pipeline more editorially coherent
