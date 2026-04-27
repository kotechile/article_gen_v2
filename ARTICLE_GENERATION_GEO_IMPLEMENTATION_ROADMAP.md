# Article Generation and GEO Implementation Roadmap

## Purpose

This document is the implementation source of truth for improving the article pipeline so it consistently produces:

- human-sounding, non-robotic writing
- well-informed, source-grounded content
- high-quality articles across many topic types
- GEO-friendly articles that perform well for generative answer engines

This roadmap focuses especially on the current deep research and article generation flow.

## Why this roadmap exists

The current system has strong building blocks, but there are several structural gaps:

- deep research is not tightly coupled to the article-writing path
- evidence collection is stronger than evidence reasoning
- section generation favors throughput over coherence
- refinement improves tone, but not enough for true humanization
- GEO optimization is partial and mostly SEO-adjacent
- the pipeline allows too many low-evidence articles to complete successfully

This roadmap converts those observations into a phased implementation plan.

## Current-state summary

### What works today

- The app can move ideas into the Content Library and create article records.
- The main generation pipeline already has stages for claims, evidence, structure, writing, citations, refinement, and finalization.
- RAG and Linkup fallback are present.
- Section-specific evidence lookup exists.
- Deep research can gather external information and upload a report into RAG.
- WordPress publishing already supports post creation, categories, featured media, and SEO metadata.

### What is missing

- Deep research outputs are not treated as a required input to structure generation and section writing.
- Evidence ranking is relatively shallow and does not strongly model contradiction, freshness, source authority, or claim coverage.
- The writer can still produce generic LLM phrasing and repetitive structure.
- Parallel section generation weakens narrative flow.
- GEO requirements are not modeled explicitly in the article assembly process.
- There are not enough quality gates before an article is considered complete.

## Goals

### Primary goals

- Make research quality deterministically influence writing quality.
- Make the article sound like a human editor wrote it, not a generic assistant.
- Make outputs more trustworthy, nuanced, and grounded in evidence.
- Make the article structure retrieval-friendly for answer engines and LLM summarizers.
- Prevent low-evidence or generic articles from quietly passing as complete.

### Non-goals for the first implementation phases

- Full autonomous editorial agent orchestration
- Fine-tuning custom models
- Replacing the entire current pipeline in one release
- Building a complete CMS workflow redesign

## Success criteria

We will consider this roadmap successful when the system can generate articles that consistently meet these standards:

- fewer generic or repetitive phrases
- higher source coverage per major claim
- better section-to-section coherence
- stronger answer-first formatting for GEO
- fewer unsupported statements
- better editorial voice consistency
- higher evaluation scores in automated and human review

## Delivery principles

- Integrate with existing pipeline stages instead of rewriting everything at once.
- Prefer additive changes with quality gates over risky big-bang replacements.
- Make research outputs structured and reusable.
- Add evaluation and observability early so improvements are measurable.
- Keep backward compatibility where reasonable during rollout.

## Phase plan

## Phase 0: Baseline, instrumentation, and safety rails

### Objective

Create enough visibility to measure quality before changing behavior.

### Scope

- Add article-quality diagnostics to generation outputs.
- Track evidence counts, source types, freshness, and uncited sections.
- Add humanization diagnostics such as repetition, filler density, and sentence monotony.
- Add GEO diagnostics such as answer-first summary presence, entity clarity, FAQ presence, and passage structure.

### Implementation tasks

- Add a `quality_report` object to final generation metadata.
- Add logs for:
  - evidence count by section
  - average source freshness
  - percentage of paragraphs with evidence support
  - repeated phrases detected
  - average sentence length variance
- Add a lightweight evaluator service or utility module for post-generation checks.

### Candidate files

- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/tasks.py`
- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/topic_analysis/backend/src/services/content_generator/content_generator.py`
- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/topic_analysis/backend/src/services/content_generator/article_structure_generator.py`

### Deliverables

- quality diagnostics in generated article metadata
- structured logs for article-quality signals
- first-pass evaluation rubric

### Acceptance criteria

- Every generated article includes a machine-readable quality report.
- We can compare two article outputs using the same evaluation dimensions.

### Status

- [x] Baseline evaluator utility implemented (`quality_report`)
- [x] Quality report persisted in generation metadata/finalization
- [x] Evaluation script added for repeatable scoring
- [ ] Additional diagnostics expansion pending

## Phase 1: Make deep research a first-class upstream artifact

### Objective

Turn deep research into a structured dossier that directly feeds article generation.

### Problem being solved

Today deep research uploads a markdown report to RAG and marks the title as `Research Complete`, but the main writer is not guaranteed to consume that output in a controlled way.

### Scope

- Replace or augment the markdown-only deep research output with a structured `research dossier`.
- Bind the dossier to the article record and generation payload.
- Require article generation to consume the dossier if it exists.

### Research dossier schema

The dossier should include:

- article id
- topic
- research date
- primary claims
- supporting sources by claim
- counterpoints by claim
- key entities
- important statistics
- examples and case studies
- unresolved questions
- freshness summary
- source quality summary
- citation map

### Implementation tasks

- Extend deep research service to emit both:
  - a markdown report
  - a structured JSON dossier
- Store the dossier in the DB and optionally also index it into RAG.
- Update the generation pipeline to load dossier data before structure generation.
- Add a rule: if dossier exists and passes validation, use it as the primary research context.

### Candidate files

- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/topic_analysis/backend/src/services/research/deep_research_service.py`
- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/topic_analysis/backend/src/api/deep_research_routes.py`
- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/tasks.py`
- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/topic_analysis/backend/src/services/rag_service.py`

### Data model changes

- Add a `research_dossier` field or related table keyed by article/title id.
- Add dossier validity fields:
  - `dossier_status`
  - `dossier_source_count`
  - `dossier_last_updated_at`
  - `dossier_quality_score`

### Deliverables

- deep research produces structured research assets
- article generation consumes dossier context
- title status reflects dossier readiness, not just markdown generation success

### Acceptance criteria

- A completed deep research run produces a dossier object with claim-level source coverage.
- Structure generation can explicitly reference dossier claims and source groups.
- Articles generated after deep research show higher evidence coverage in diagnostics.

### Status

- [x] Deep research now emits structured dossier JSON
- [x] Dossier persisted and loaded by generation pipeline
- [x] Dossier validation gate implemented (`ready` vs `needs_review`)
- [x] Dossier context injected into outline + section generation prompts

## Phase 2: Upgrade evidence modeling from pooled evidence to claim-centered evidence

### Objective

Move from general evidence collection to explicit claim-to-evidence mapping.

### Problem being solved

The system collects evidence, but reasoning over evidence is still shallow. It does not strongly represent disagreement, source quality, or coverage by claim.

### Scope

- Create claim bundles for major assertions.
- Attach evidence, freshness, authority, and confidence to each bundle.
- Detect contradictory and mixed evidence.

### Implementation tasks

- Replace or extend `_extract_claims()` to output structured claims with ids.
- Update evidence collection to map evidence into claim bundles.
- Add contradiction detection and confidence scoring.
- Add source diversity scoring:
  - expert source
  - primary source
  - secondary source
  - commercial source
  - community source

### Candidate files

- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/tasks.py`
- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/core/models/evidence.py`
- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/topic_analysis/backend/src/services/content_generator/citation_generator.py`

### Deliverables

- claim bundle model
- claim-to-evidence mapping
- contradiction and confidence scoring

### Acceptance criteria

- Every important article section can point to a claim bundle.
- Evidence ranking reflects authority, freshness, and claim support.
- The system can mark a claim as high-confidence, mixed, or weakly supported.

### Status

- [x] `_extract_claims()` now emits structured claims with deterministic `claim_id`
- [x] Claim keyword extraction added for downstream claim-evidence matching
- [x] `_rank_evidence()` now builds `claim_bundles` with per-claim support mapping
- [x] Contradiction and mixed-signal detection added to claim bundles
- [x] Claim confidence and coverage scoring added
- [x] Source-type diversity taxonomy scoring added (expert/primary/secondary/commercial/community)
- [x] Claim-bundle summaries wired into final article metadata/reporting

## Phase 3: Make the outline evidence-aware and intent-aware

### Objective

Generate better outlines based on search intent, evidence shape, and editorial format.

### Problem being solved

The current structure generator is solid, but it still relies too heavily on the brief and generic article-type classification.

### Scope

- Classify article intent more precisely.
- Build outlines that reflect available evidence and topic type.
- Support GEO-aware blocks in the outline from the start.

### Implementation tasks

- Expand article framing taxonomy:
  - explainer
  - decision guide
  - comparison
  - tutorial
  - strategic analysis
  - controversy or debate
  - review
- Add outline generation rules based on:
  - user intent
  - topic freshness
  - evidence distribution
  - entity complexity
- Add optional sections such as:
  - short answer
  - key takeaways
  - tradeoffs
  - FAQ
  - who this is for
  - when this advice does not apply

### Candidate files

- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/topic_analysis/backend/src/services/content_generator/article_structure_generator.py`
- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/tasks.py`

### Deliverables

- improved outline classification
- evidence-aware section planning
- GEO-friendly outline blocks

### Acceptance criteria

- Outlines differ meaningfully by content type and intent.
- Answer-first and retrieval-friendly sections can be toggled or inserted automatically.

### Status

- [x] Structure generation now consumes `claim_bundles` from `tasks.py`
- [x] Search-intent inference added to structure generation (explicit + heuristic fallback)
- [x] Section planning prompt now includes claim-bundle confidence/mixed-signal context
- [x] Outline prompt rules now enforce intent-aware section shaping
- [x] Added explicit GEO block insertion rules (Short Answer, Key Takeaways, FAQ, Tradeoffs) with `geo_blocks_enabled` toggle

## Phase 4: Replace throughput-first drafting with hybrid sequential composition

### Objective

Preserve speed where it helps, but restore editorial coherence across the article.

### Problem being solved

Parallel section generation is efficient, but sections can feel isolated and repetitive.

### Scope

- Keep section evidence gathering parallel.
- Draft the body in a controlled sequence using memory of prior sections.
- Write the intro and conclusion after the core body exists.

### Implementation tasks

- Separate `research parallelism` from `writing order`.
- Build a lightweight section-memory object containing:
  - covered claims
  - covered examples
  - used sources
  - repeated themes
  - promised but not yet covered points
- Generate:
  - body sections first
  - intro second-to-last
  - conclusion last
- Add anti-repetition checks between adjacent sections.

### Candidate files

- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/tasks.py`
- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/topic_analysis/backend/src/services/content_generator/content_generator.py`

### Deliverables

- hybrid writing orchestration
- section memory propagation
- intro and conclusion generated from full article context

### Acceptance criteria

- Repetition across sections drops in evaluation.
- Introductions feel more aligned with the actual article body.
- Conclusions synthesize instead of repeating boilerplate.

### Status

- [x] Separated section evidence prefetch parallelism from drafting order
- [x] Implemented hybrid drafting order: body first, intro later, conclusion last
- [x] Added section-memory object (covered claims, used sources, repeated themes, promised points)
- [x] Passed section-memory context into per-section generation prompt
- [x] Added adjacent-section anti-repetition similarity diagnostics

## Phase 5: Add a dedicated humanization pass

### Objective

Make the article read like it was editing by a thoughtful human writer.

### Problem being solved

Tone refinement exists, but it does not yet enforce enough editorial texture or anti-robot safeguards.

### Scope

- Build a dedicated pass for humanization, separate from factual drafting.
- Add style diagnostics and repair loops.

### Humanization rules

- vary sentence length and cadence
- prefer concrete wording over abstract filler
- use examples before abstractions where appropriate
- remove repetitive assistant phrases
- use sharper verbs and cleaner transitions
- allow nuance and selective caveats
- avoid fake personality or forced friendliness

### Implementation tasks

- Add a banned phrase and robotic-pattern detector.
- Add heuristics for:
  - monotony
  - hedging overload
  - abstraction density
  - filler density
- Add rewrite prompts that only target weak sections instead of full article rewrites.
- Replace the current one-size-fits-all refinement with:
  - factual integrity pass
  - humanization pass
  - final cleanup pass

### Candidate files

- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/tasks.py`
- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/topic_analysis/backend/src/services/content_generator/content_generator.py`

### Deliverables

- robotic phrase detector
- section-level rewrite triggers
- improved refinement prompts and repair flow

### Acceptance criteria

- Human review reports noticeably less generic phrasing.
- The evaluator shows lower repetition and filler density.
- Only weak sections are regenerated when possible.

### Status

- [x] Added robotic-pattern diagnostics service (`src/services/humanization_service.py`)
- [x] Added section-level weak-content detection and targeted rewrite triggers
- [x] Replaced one-size-fits-all refinement with 3-pass flow in `tasks.py`:
  - factual integrity pass
  - targeted humanization pass
  - final cleanup/tone consistency pass
- [x] Added Phase 5 stage metrics (`weak_sections_detected`, per-pass counters)

## Phase 6: Add explicit GEO optimization

### Objective

Design articles to be easier for generative engines to retrieve, summarize, and quote correctly.

### Problem being solved

The current system does some SEO work, but GEO needs retrieval-friendly content structure, not just metadata.

### Scope

- Add answer-engine-friendly structures into article assembly.
- Improve passage design for LLM summarization.

### GEO components

- short answer block near the top
- key takeaways section
- explicit concept definitions
- clean entity naming
- comparison tables when useful
- tradeoff summaries
- FAQ blocks for branching intent
- concise quotable fact patterns

### Implementation tasks

- Add GEO block generation rules based on article type.
- Add a passage-quality scorer.
- Ensure every article includes at least one answer-first summary block when appropriate.
- Make entity disambiguation explicit in technical and ambiguous topics.

### Candidate files

- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/tasks.py`
- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/topic_analysis/backend/src/services/content_generator/article_structure_generator.py`
- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/services/wordpressService.ts`

### Deliverables

- GEO-aware structural blocks
- answer-first formatting support
- entity and passage optimization rules

### Acceptance criteria

- Articles contain clear answer-oriented passages for major queries.
- GEO diagnostics show improved entity clarity and passage extractability.

### Status

- [x] GEO block rules added at outline stage (Phase 3 dependency)
- [x] Passage-quality scorer added to `article_quality_evaluator.py` (`extractability_score`)
- [x] Entity clarity/disambiguation scorer added (`entity_clarity_score`)
- [x] Finalization now enforces answer-first block when missing (`geo_enforce_answer_first`)
- [x] GEO optimization metadata persisted in final article (`geo_optimization`)

## Phase 7: Add evidence quality gates and confidence mapping

### Objective

Prevent weakly supported articles from being treated as complete.

### Problem being solved

The current pipeline is resilient, but too permissive when evidence is poor or missing.

### Scope

- Add minimum quality thresholds before completion.
- Add confidence labels to claims and sections.

### Implementation tasks

- Add per-topic evidence thresholds.
- Require stronger evidence on sensitive categories.
- Add `confidence_map` output with:
  - high-confidence claims
  - mixed-evidence claims
  - low-confidence claims
  - uncited paragraphs
- Change final status behavior:
  - `Created` only if thresholds pass
  - otherwise `Needs Review`

### Candidate files

- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/tasks.py`
- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/core/models/evidence.py`

### Deliverables

- quality gates before finalization
- confidence map metadata
- stricter completion criteria

### Acceptance criteria

- Weakly sourced articles are flagged automatically.
- Confidence map is available for UI review and future edits.

### Status

- [x] Added confidence map output (`high`, `mixed`, `low`, `uncited_paragraphs`) in final article metadata
- [x] Added per-topic quality gates with stricter thresholds for sensitive topics
- [x] Added completion decisioning (`Created` vs `Needs Review`) via `quality_gate`
- [x] Persistence now writes gated final status to `Titles.status`
- [x] Added backward-compatible DB fallback when new JSON columns are missing

## Phase 8: Build an evaluation harness and release process

### Objective

Make improvements measurable and safe to roll out.

### Scope

- Build test sets across article categories.
- Add automated scoring and sampling review workflows.

### Evaluation dimensions

- factual grounding
- claim coverage
- source quality
- section coherence
- repetition
- humanization
- GEO readiness
- citation precision

### Implementation tasks

- Create benchmark prompts for:
  - evergreen informational topics
  - technical topics
  - product comparisons
  - B2B topics
  - timely topics
  - sensitive or high-stakes topics
- Add evaluation scripts and snapshot outputs.
- Define rollout criteria before enabling new pipeline behavior by default.

### Candidate files

- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/tests`
- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/tests/manual`
- `/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/topic_analysis/backend/tests`

### Deliverables

- article-quality benchmark suite
- comparison rubric
- rollout and rollback rules

### Acceptance criteria

- We can compare old vs new pipeline outputs on a stable benchmark set.
- New pipeline is only promoted after measurable improvement.

### Status

- [x] Added benchmark topic set for Phase 8 (`.benchmarks/phase8/benchmark_topics.json`)
- [x] Added comparison harness (`scripts/run_phase8_benchmark.py`) for baseline vs candidate snapshots
- [x] Added snapshot outputs (JSON + Markdown) with aggregate deltas
- [x] Reused evaluator rubric in benchmark workflow (`build_article_quality_report`)
- [x] Added rollout/rollback policy with explicit go/no-go thresholds (`PHASE8_ROLLOUT_ROLLBACK_POLICY.md`)

## Cross-cutting design decisions

These decisions should guide implementation across phases.

### Decision 1: Structured research beats markdown-only research

- Always prefer structured dossier data for generation-time reasoning.
- Markdown reports remain useful for inspection and export, but should not be the only artifact.

### Decision 2: Fallbacks should be explicit

- The app may still support low-evidence generation for some use cases.
- However, that should be visible and intentional, not silent.

### Decision 3: Humanization is an editing function, not a tone toggle

- A simple tone string is not enough.
- The program needs voice profiles, anti-pattern detection, and selective rewrites.

### Decision 4: GEO is a structural requirement

- GEO should be built into outline and final formatting, not treated as post-hoc metadata.

### Decision 5: Research providers should be routed by task type

- Keep Linkup as default for fast, broad evidence collection during normal article generation.
- Use Tavily for deep research workflows and escalation when evidence is insufficient.
- Standardize both providers into one internal evidence and dossier schema.
- Choose defaults based on measured quality, latency, and cost, not preference.

## Research provider strategy (Linkup + Tavily)

### Why this matters

- Linkup and Tavily have different strengths.
- Deep research quality goals require stronger iterative retrieval than broad fallback search alone.
- The pipeline should support provider-specific strengths while keeping one downstream writing path.

### Proposed routing model

- `standard article flow`: Linkup-first when RAG is insufficient.
- `deep research flow`: Tavily-first with agentic query refinement.
- `auto escalation`: if Linkup evidence quality is below threshold, run Tavily enrichment pass.
- `provider-agnostic output`: both providers map to the same evidence shape and dossier fields.

### Configuration model

- Add `research_provider_strategy` option:
  - `linkup_only`
  - `tavily_only`
  - `hybrid`
  - `auto`
- Add `deep_research_provider` option:
  - `tavily`
  - `linkup`
  - `auto`
- Add evaluation flag:
  - `research_provider_experiment_enabled`

### A/B evaluation plan

- Compare `linkup_only` vs `hybrid` vs `tavily_first_for_deep_research`.
- Evaluate on:
  - source quality and authority mix
  - contradiction coverage
  - claim coverage depth
  - GEO score impact
  - humanization score impact
  - latency
  - cost per article

### Acceptance criteria for provider decision

- Hybrid routing is considered successful if it improves quality metrics with acceptable latency and cost.
- Default strategy should be selected only after benchmark runs across multiple topic classes.
- Provider selection should remain configurable per environment.

## Suggested rollout order

This is the recommended sequence for implementation:

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7
9. Phase 8

## MVP slice

If we want the highest-leverage MVP before a broader rollout, implement this subset first:

1. Phase 1 research dossier
2. claim-to-evidence mapping from Phase 2
3. hybrid section composition from Phase 4
4. humanization pass from Phase 5
5. answer-first GEO block from Phase 6

This MVP should deliver the largest quality jump without requiring a full pipeline rewrite.

## Open questions

- Should deep research run automatically for all long-form articles, or only when requested?
- Should the dossier live in `Titles`, in a separate table, or both?
- Should article completion be blocked when evidence is insufficient, or only flagged for review?
- Which topic categories need stricter evidence policies first?
- Do we want one universal voice profile, or user-selectable editorial styles?

## Tracking checklist

### Phase 0

- [x] Add post-generation quality report
- [x] Add evidence and style diagnostics
- [x] Add first evaluation rubric

### Phase 1

- [x] Define research dossier schema
- [x] Store dossier in DB
- [x] Bind dossier into generation payload
- [x] Validate dossier before use
- [x] Add provider strategy flags (`linkup_only`, `tavily_only`, `hybrid`, `auto`)
- [x] Normalize Tavily and Linkup evidence into one schema

### Phase 2

- [ ] Add claim ids and claim bundles
- [ ] Map evidence to claims
- [ ] Add contradiction and confidence scoring

### Phase 3

- [ ] Expand article framing taxonomy
- [ ] Make outlines evidence-aware
- [ ] Add GEO-aware structural blocks

### Phase 4

- [ ] Separate evidence parallelism from writing order
- [ ] Add section memory
- [ ] Generate intro and conclusion from final body

### Phase 5

- [ ] Add robotic-pattern detector
- [ ] Add humanization pass
- [ ] Add selective rewrite loop for weak sections

### Phase 6

- [ ] Add short answer block
- [ ] Add key takeaways
- [ ] Add entity clarity rules
- [ ] Add passage extractability scoring

### Phase 7

- [ ] Add evidence quality thresholds
- [ ] Add confidence map
- [ ] Add `Needs Review` final state

### Phase 8

- [ ] Build benchmark topic set
- [ ] Add evaluation scripts
- [ ] Define rollout criteria
- [ ] Run research-provider A/B comparison and document default strategy decision

## Notes

- Update this document as implementation decisions change.
- Keep the phase statuses current.
- Link PRs, issue references, or migration notes under the relevant phase as work begins.
