# Keyword Handoff Implementation Checklist

Purpose: make keyword strategy a first-class part of the article pipeline.

Target outcome:
- Research performs broad keyword discovery and validation.
- Content Generation performs final keyword selection for the exact article.
- The selected keyword set is persisted on `Titles` and used by deep research, outlining, writing, SEO metadata, and dashboards.

## Progress Snapshot

- [x] Added migration scaffold for keyword handoff fields on `Titles`.
- [x] Added Research -> `Titles` handoff writes in content idea publish flow.
- [x] Added baseline keyword strategy persistence in generation finalization (`tasks.py`).
- [x] Added backward-compatible fallback logic when new `Titles` columns are not yet present.
- [x] Added generation-start recovery of keyword dossier from `source_idea_id` (`content_ideas`) when `Titles` is missing structured keyword fields.
- [x] Added explicit `KEYWORD_INTELLIGENCE` pipeline stage before evidence collection.
- [x] Updated evidence collection query formation to prioritize selected keyword strategy (primary + secondary + intent).
- [x] Added derivation + persistence of `supporting_entities_json` and `priority_questions_json` from research dossier in `KEYWORD_INTELLIGENCE`.
- [x] Updated finalization payload to reuse selected keyword strategy artifacts from `research_data` instead of writing empty GEO context arrays.
- [x] Switched generation-complete lifecycle status to `Written` while keeping quality gate decision in `quality_gate`.
- [x] Updated Content Library/Generation modal status handling for `New` -> `Written` -> `WP Published`.
- [x] Added keyword metric UI fallbacks (selected keyword metrics first, DataForSEO aggregate fallback second).

## Principles

- Research owns keyword discovery.
- Content Generation owns keyword selection.
- DataForSEO-backed keyword signals must be visible, persisted, and auditable.
- Fallbacks to LLM-only keywords must be explicit, not silent.
- Article statuses should reflect the real lifecycle: `New` -> `Written` -> `WP Published`.

## Phase 1: Audit Current Flow

- [ ] Trace where Research currently generates or enriches keywords.
- [ ] Trace where DataForSEO metrics are computed in Research and Idea enrichment.
- [ ] Trace what keyword data, if any, is persisted to `Titles`.
- [ ] Trace what keyword data is read by `tasks.py` during article generation.
- [ ] Identify every place where keyword metrics can be dropped, overwritten, or left null.
- [ ] Document current source-of-truth tables/fields for:
- `candidate keywords`
- `search volume`
- `keyword difficulty`
- `search intent`
- `cluster/topic mapping`
- [ ] Confirm how `ContentStudio`, `ArticleEditor`, and `Content Library` currently read keyword-related fields.

## Phase 2: Define Data Model

- [ ] Add/confirm `Titles` columns for research-stage keyword dossier.
- [ ] Add/confirm `Titles` columns for final selected article keyword strategy.
- [ ] Proposed research-stage fields:
- `keyword_candidates_json`
- `keyword_clusters_json`
- `keyword_research_status`
- `keyword_research_source`
- `keyword_research_confidence`
- `keyword_research_generated_at`
- [ ] Proposed final-selection fields:
- `primary_keyword`
- `secondary_keywords_json`
- `supporting_entities_json`
- `priority_questions_json`
- `selected_keyword_search_volume`
- `selected_keyword_difficulty`
- `selected_keyword_intent`
- `keyword_selection_reason`
- `keyword_strategy_version`
- `keyword_selection_source`
- [ ] Add migration SQL for any missing fields.
- [ ] Ensure old fields remain backward-compatible during rollout.

## Phase 3: Research-Side Keyword Dossier

- [ ] Persist the output of DataForSEO-backed keyword discovery into the article/title handoff path.
- [ ] Store whether keyword data came from:
- `dataforseo`
- `hybrid`
- `llm_fallback`
- [ ] Attach a confidence/status flag when DataForSEO returns weak or empty results.
- [ ] Prevent silent null handoff when keyword enrichment fails.
- [ ] Ensure dossier includes:
- candidate keywords
- cluster/group labels
- volume
- difficulty
- intent
- entities
- relevant questions
- [ ] Add logs/observability for dossier creation success and failure.

## Phase 4: Content Generation Keyword Intelligence Stage

- [ ] Add a new pipeline stage before deep research in `tasks.py`.
- [ ] Name the stage something explicit, e.g. `KEYWORD_SELECTION` or `KEYWORD_INTELLIGENCE`.
- [ ] Load:
- article title
- brief
- topic/site context
- research keyword dossier
- [ ] Re-score candidate keywords for the exact article angle.
- [ ] Select:
- one `primary_keyword`
- several `secondary_keywords`
- supporting entities
- answerable search questions
- [ ] Save final keyword strategy back to `Titles`.
- [ ] Mark whether the selection was:
- `research_dossier_reused`
- `re-ranked_with_dataforseo`
- `llm_fallback`

## Phase 5: Deep Research Integration

- [ ] Change deep research query building to use the selected keyword strategy, not just raw brief text.
- [ ] Include `primary_keyword`, top secondary keywords, entities, and question set in research prompts.
- [ ] Improve RAG/web research query formation using the final keyword set.
- [ ] Add keyword coverage diagnostics for the selected strategy.
- [ ] Log which selected keywords were actually used during research.

## Phase 6: Outline and Writing Integration

- [ ] Feed selected keywords into structure generation.
- [ ] Use `primary_keyword` to influence:
- title refinement
- deck
- thesis
- meta description
- section headings
- [ ] Use secondary keywords/entities for semantic coverage, not stuffing.
- [ ] Add validation that the final article naturally covers selected entities/questions.
- [ ] Add explicit anti-stuffing checks.

## Phase 7: UI and Dashboard

- [ ] Show the selected `primary_keyword` in Content Studio / Article Editor.
- [ ] Show keyword source and confidence:
- `DataForSEO`
- `Hybrid`
- `LLM fallback`
- [ ] Show selected keyword metrics:
- search volume
- keyword difficulty
- intent
- [ ] Keep generation quality metrics separate from keyword opportunity metrics.
- [ ] Update dashboard cards so users can distinguish:
- article quality
- keyword opportunity
- publication state
- [ ] Make status display consistent everywhere:
- `New`
- `Written`
- `WP Published`

## Phase 8: Failure Modes and Fallbacks

- [ ] Define behavior when DataForSEO returns no useful keyword data.
- [ ] Define behavior when Research has dossier data but Content Generation cannot re-score it.
- [ ] Ensure fallback paths are visible in UI and logs.
- [ ] Never present LLM-only keyword guesses as if they were validated by DataForSEO.
- [ ] Add safe defaults so generation still works if keyword intelligence fails.

## Phase 9: QA and Verification

- [ ] Test article with strong DataForSEO-backed dossier.
- [ ] Test article with partial DataForSEO result.
- [ ] Test article with no DataForSEO result and explicit fallback.
- [ ] Verify selected keyword strategy persists on `Titles`.
- [ ] Verify Article Editor shows both:
- quality metrics
- keyword opportunity metrics
- [ ] Verify Content Library statuses transition correctly.
- [ ] Verify WordPress publish updates `WP Published`.
- [ ] Verify deep research prompts include selected keywords.
- [ ] Verify final article metadata uses selected keywords.

## Phase 10: Rollout Order

- [ ] Ship schema changes first.
- [ ] Ship Research dossier persistence second.
- [ ] Ship Content Generation keyword intelligence third.
- [ ] Ship UI visibility fourth.
- [ ] Backfill a sample of recent articles to validate the new flow.
- [ ] Benchmark output quality and keyword completeness before full rollout.

## Success Criteria

- [ ] Every generated article has an explicit keyword strategy record.
- [ ] Every article clearly shows whether keyword selection was DataForSEO-backed or fallback.
- [ ] Deep research uses selected keywords rather than only the raw brief.
- [ ] Article Editor displays meaningful keyword opportunity metrics.
- [ ] Content Library statuses reflect actual lifecycle state.
- [ ] No silent null keyword handoff between Research and Content Generation.

## Recommended Build Order

1. Schema + persistence
2. Research dossier handoff
3. Content Generation keyword intelligence
4. Deep research query upgrade
5. UI visibility
6. QA + backfill
