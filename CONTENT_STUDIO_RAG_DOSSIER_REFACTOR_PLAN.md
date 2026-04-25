# Content Studio RAG + Dossier Refactor Plan

## Goal

Refactor Content Studio and the article-generation pipeline so that:

1. Deep Research (`research_dossier`) is the primary external research backbone.
2. RAG is an optional complementary source of private knowledge.
3. Live web research during article generation is explicit, not automatic.
4. Gap Analysis becomes an optional upstream enrichment workflow, not a hidden side effect of normal generation.

This plan is based on the current implementation in:

- `frontend/src/pages/ContentStudio.tsx`
- `src/api/endpoints/research.py`
- `tasks.py`
- `content_generator.py`
- `article_structure_generator.py`


## Problem Summary

The current `rag_enabled: true` behavior reflects an older product model where RAG was closer to the main research engine. Today the system has a stronger Deep Research workflow, and the dossier should be treated as the main planning artifact.

Current issues:

1. `rag_enabled` does two jobs at once:
   - Enables private knowledge retrieval
   - Indirectly influences whether the pipeline performs more live web research

2. `claims_research_enabled` is overloaded:
   - In the UI it reads like "claims validation"
   - In the backend it acts as a permission switch for Linkup/Tavily evidence collection

3. RAG currently participates in a fallback tree:
   - RAG runs first
   - If coverage is insufficient, live web research may run
   - Section-level RAG may also trigger section-level Linkup fallback

4. This creates the wrong mental model for the current product:
   - Users want "Dossier + optional RAG"
   - They do not want "RAG mode that may or may not trigger more web research"

5. The current Query Type UI is not trustworthy enough for product-facing semantics:
   - The task pipeline currently rewrites non-hybrid endpoints toward `/query_hybrid_enhanced`
   - That means the dropdown does not cleanly map to actual runtime behavior


## Target Product Workflow

### Primary workflow

The default article-generation workflow should be:

1. Use the validated Deep Research dossier as the primary external-research source.
2. Optionally retrieve complementary evidence from a selected RAG collection.
3. Build the table of contents and section content from dossier claims + dossier summary + dossier questions + optional RAG evidence.
4. Do not run live web research during generation unless the user explicitly requested it.

### Mental model

Use these product definitions:

- `Dossier`: The Deep Research output stored in `research_dossier`
- `RAG`: User-curated private knowledge added before generation
- `Live Web Refresh`: Optional real-time search during generation for freshness-sensitive topics
- `Gap Analysis`: Optional upstream research/enrichment step, separate from article generation

### Recommended generation modes

Introduce explicit source modes:

- `dossier_only`
- `dossier_plus_rag`
- `dossier_plus_rag_plus_live_web`
- `rag_only`

Default mode:

- `dossier_plus_rag` when a RAG collection is selected
- `dossier_only` when no RAG collection is selected

Do not default to `dossier_plus_rag_plus_live_web`.


## Desired Runtime Rules

### Rule 1: Dossier is the primary source of truth

If a valid `research_dossier` exists, the pipeline should use it as the default basis for:

- claim extraction
- keyword intelligence
- article structure generation
- section-writing prompts

This is already partially true and should remain true.

### Rule 2: RAG is additive

If a RAG collection is selected:

- retrieve RAG evidence globally
- retrieve section-specific RAG evidence where useful
- merge that evidence into the dossier-driven generation flow

RAG should enrich planning and writing, not replace the dossier by default.

### Rule 3: Live web refresh must be explicit

Live web research during article generation must only happen when the selected source mode includes live web.

That means:

- `dossier_only`: no Linkup/Tavily
- `dossier_plus_rag`: no Linkup/Tavily
- `dossier_plus_rag_plus_live_web`: Linkup/Tavily allowed
- `rag_only`: no Linkup/Tavily unless a separate `rag_only_plus_live_web` mode is ever added later

### Rule 4: No automatic fallback from RAG to live web

Remove the current logic that says:

- if RAG coverage is insufficient, automatically use live web

Replace it with:

- if source mode does not allow live web, continue with dossier + whatever RAG evidence was found
- optionally log low coverage for diagnostics, but do not switch sources automatically

### Rule 5: Gap Analysis is upstream only

Gap Analysis should not be bundled into article generation implicitly.

It should be an explicit preparatory action that:

- enriches a RAG collection
- enriches a dossier
- or creates reusable research assets before generation


## Proposed UI Refactor

## Section rename

Replace `RAG & Research` with `Sources` or `Research Sources`.

Reason:

- The current label groups concepts that now need clearer separation.

## New UI model

Replace the current coupling of:

- `RAG Collection`
- `Query Type`
- `Emphasis`
- `Enable Claims Validation (Web Search)`

with a clearer source strategy UI.

### Recommended controls

#### 1. Source Mode

Add a required dropdown:

- `Dossier only`
- `Dossier + RAG`
- `Dossier + RAG + Live Web Refresh`
- `RAG only`

Default:

- `Dossier only` if no RAG collection is selected
- auto-switch to `Dossier + RAG` when a collection is selected, unless the user explicitly chooses another mode

#### 2. RAG Collection

Keep the existing collection selector.

Behavior:

- Enabled for modes containing RAG
- Disabled/hidden for `dossier_only`
- Required for `dossier_plus_rag` and `rag_only`

Helper text:

- "Use private notes, uploaded documents, and manually added research as a complement to the dossier."

#### 3. RAG Retrieval Settings

Keep advanced RAG controls, but move them under an expandable `Advanced RAG Settings` group:

- Query Type
- Emphasis

These should be treated as retrieval tuning, not as source-strategy controls.

#### 4. Live Web Refresh

Remove the current `Enable Claims Validation (Web Search)` checkbox from the main UI.

Replace it with either:

- a mode choice (`Dossier + RAG + Live Web Refresh`), or
- a checkbox labeled `Include live web refresh during generation`

Preferred approach:

- keep it embedded in `Source Mode`

Reason:

- It is easier for users to understand a source strategy than a low-level boolean.

#### 5. Gap Analysis

Add a separate optional action or panel:

- `Run Gap Analysis before generation`

Behavior:

- not part of the default Generate button payload
- can create or enrich a RAG collection
- can produce reusable notes/artifacts

### UX copy recommendation

Use copy like this:

- `Dossier only`: Use the existing Deep Research dossier as the article's research foundation.
- `Dossier + RAG`: Use the dossier plus your private knowledge collection. No live web refresh during generation.
- `Dossier + RAG + Live Web Refresh`: Use the dossier, your private collection, and real-time web refresh during generation.
- `RAG only`: Use only the selected private collection. Best for internal or highly curated content.


## Proposed API Contract Refactor

### New request field

Add a single explicit field:

```json
"source_strategy": "dossier_only" | "dossier_plus_rag" | "dossier_plus_rag_plus_live_web" | "rag_only"
```

### Continue supporting legacy fields temporarily

For backward compatibility during rollout:

- keep accepting `rag_enabled`
- keep accepting `claims_research_enabled`
- keep accepting `rag_collection_name`
- keep accepting `rag_balance_emphasis`

### Backend normalization

Inside `src/api/endpoints/research.py`, normalize requests like this:

1. If `source_strategy` is provided, it becomes the source of truth.
2. If `source_strategy` is missing, infer it from legacy fields:
   - `rag_enabled=false` -> `dossier_only`
   - `rag_enabled=true` and `claims_research_enabled=false` -> `dossier_plus_rag`
   - `rag_enabled=true` and `claims_research_enabled=true` -> `dossier_plus_rag_plus_live_web`
   - if a future caller explicitly asks for RAG-only, allow `rag_only`
3. Pass the normalized strategy into `tasks.py`.

### Legacy compatibility note

Keep legacy booleans available in the payload during migration, but make `source_strategy` the only field used by new decision logic.


## tasks.py Decision Tree Refactor

## Current behavior to remove

The following behaviors should be removed from article generation:

1. Auto-enabling Linkup when RAG is disabled
2. Auto-enabling Linkup when RAG returns no evidence
3. Auto-enabling Linkup when RAG coverage is insufficient
4. Section-level automatic Linkup fallback when section RAG is insufficient, unless source mode explicitly allows live web

## New decision tree

Implement a normalized helper early in `process_research_task`:

```python
source_strategy = _normalize_source_strategy(research_data)
```

Recommended helper outputs:

```python
{
    "strategy": "dossier_plus_rag",
    "use_dossier": True,
    "use_rag": True,
    "use_live_web": False,
    "allow_gap_analysis": False,
}
```

### Stage-by-stage behavior

#### 1. Claim Extraction

Behavior:

- If `use_dossier` and dossier is valid, extract claims from dossier first.
- If `rag_only`, fall back to LLM claim extraction from brief when no dossier exists.

No live web decision belongs here.

#### 2. Keyword Intelligence

Behavior:

- Continue using dossier-driven keyword intelligence when dossier exists.
- RAG may later enrich evidence and section writing, but it should not replace keyword selection logic by default.

#### 3. Evidence Collection

Behavior:

- If `use_rag`, collect global RAG evidence.
- If `use_live_web`, collect Linkup/Tavily evidence.
- If both are enabled, merge them.
- If only one is enabled, do not silently switch to the other.

Important:

- RAG coverage scoring may still be useful for diagnostics and logging.
- It must no longer control whether live web research runs.

Refactor `_collect_evidence()` accordingly.

#### 4. Evidence Ranking

Behavior:

- Rank merged evidence from the enabled sources.
- Preserve source labels so downstream prompts know whether evidence came from:
  - dossier-derived claims
  - RAG
  - live web

#### 5. Structure Generation

Behavior:

- Generate TOC using:
  - dossier summary
  - dossier claims
  - dossier unresolved questions
  - ranked evidence from enabled sources

For `dossier_plus_rag`, this ensures RAG contributes to TOC and section design without requiring live web search.

#### 6. Section Evidence Collection

Behavior:

- If `use_rag`, collect section-specific RAG evidence.
- If `use_live_web`, collect section-specific live-web evidence.
- Do not run section-level Linkup fallback just because section RAG is weak unless the source strategy allows live web.

This is one of the highest-value logic changes.

#### 7. Content Generation

Behavior:

- Keep prioritizing RAG evidence when present.
- Keep dossier summary/claims/questions in the writing context.
- Make the prompts explicitly state the enabled source policy, for example:
  - "Use the Deep Research dossier as the primary research foundation."
  - "Use the RAG evidence as complementary private knowledge."
  - "Do not infer unsupported current events unless live web evidence is provided."


## Query Type Refactor

The current backend behavior should be corrected so the UI's RAG query type maps cleanly to runtime behavior.

### Current issue

The task pipeline currently forces endpoints toward `/query_hybrid_enhanced`, which can override or corrupt the selected query type.

### Target behavior

Use one of these approaches:

#### Preferred

Pass a normalized `rag_query_type` value like:

- `simple`
- `hybrid_enhanced`
- `agentic_iterative`
- `agentic_focused`
- `truly_agentic`

Then resolve the final endpoint in one place on the backend.

#### Acceptable interim fix

Keep sending the endpoint path from the frontend, but remove the hardcoded rewriting in `tasks.py`.

Recommendation:

- move endpoint construction into a dedicated backend helper
- do not mutate it later in the task pipeline


## Data Model and Naming Recommendations

### Keep

- `research_dossier`
- `rag_collection_name`
- `rag_balance_emphasis`

### Add

- `source_strategy`
- `live_web_refresh_enabled` only if the team prefers a separate boolean internally
- `rag_usage_mode` only if a future need arises for planning-only vs writing-only RAG

### Avoid

- Continuing to use `claims_research_enabled` as the product-level control for live web behavior

Reason:

- It no longer reflects user intent clearly.


## Suggested Implementation Phases

## Phase 1: Decision-tree cleanup

Goal:

- Make backend behavior match the new product semantics before polishing UI.

Tasks:

1. Add `_normalize_source_strategy(research_data)` in `tasks.py`
2. Stop using RAG coverage as a trigger for live web fallback
3. Gate Linkup/Tavily entirely behind `use_live_web`
4. Apply the same rule to `_collect_section_evidence()`
5. Keep coverage logs for observability only

Acceptance criteria:

- `dossier_plus_rag` never calls Linkup/Tavily
- `dossier_only` never calls Linkup/Tavily
- `dossier_plus_rag_plus_live_web` may call both RAG and Linkup/Tavily

## Phase 2: API normalization

Goal:

- Add `source_strategy` without breaking existing callers.

Tasks:

1. Extend `src/api/endpoints/research.py`
2. Infer `source_strategy` from legacy booleans when missing
3. Pass normalized strategy to the task payload
4. Add logging to show normalized strategy per generation request

Acceptance criteria:

- Old clients continue working
- New clients can fully control source behavior through `source_strategy`

## Phase 3: Content Studio UI refactor

Goal:

- Make source behavior understandable to users.

Tasks:

1. Replace `RAG & Research` with `Sources`
2. Add `Source Mode` control
3. Keep `RAG Collection`
4. Move `Query Type` and `Emphasis` under advanced RAG settings
5. Remove the ambiguous `Enable Claims Validation (Web Search)` checkbox
6. Add helper text clarifying that RAG is complementary to the dossier

Acceptance criteria:

- A user can clearly choose:
  - dossier only
  - dossier + RAG
  - dossier + RAG + live web refresh

## Phase 4: Gap Analysis separation

Goal:

- Decouple Gap Analysis from article generation.

Tasks:

1. Introduce a standalone Gap Analysis action or modal
2. Let it write output to a chosen RAG collection or dossier attachment flow
3. Keep article generation unaware of Gap Analysis unless its outputs have been intentionally added upstream

Acceptance criteria:

- Gap Analysis is optional and explicit
- Generation no longer implies gap-filling behavior

## Phase 5: Prompt and evidence polish

Goal:

- Improve how the writer uses dossier + RAG together.

Tasks:

1. Update section-writing prompts to explicitly label evidence source roles
2. Encourage use of RAG for proprietary/process/internal nuance
3. Encourage use of dossier for external synthesized research backbone
4. Add guardrails against inventing "latest news" when live web evidence is absent

Acceptance criteria:

- Output reflects both source layers clearly
- Freshness-sensitive claims are not implied unless current evidence exists


## Concrete Backend Refactor Checklist

### `src/api/endpoints/research.py`

1. Add support for `source_strategy`
2. Normalize legacy fields into `source_strategy`
3. Preserve legacy fields temporarily for compatibility

### `tasks.py`

1. Add `_normalize_source_strategy()`
2. Update `_collect_evidence()`:
   - remove auto-enable logic for Linkup
   - remove coverage-driven live-web switching
   - use `use_live_web` only
3. Update `_collect_section_evidence()`:
   - same gating change
4. Add source-strategy logging at task start
5. Keep RAG coverage assessment for diagnostics only

### `content_generator.py`

1. Update prompt language to explicitly distinguish:
   - dossier foundation
   - RAG complement
   - live web evidence when enabled
2. Preserve RAG prioritization in relevant evidence ordering

### `article_structure_generator.py`

1. Ensure structure generation prompt explicitly references:
   - dossier claims/questions as primary planning inputs
   - optional RAG evidence as complementary context
2. Verify TOC generation remains dossier-first even when RAG exists

### `frontend/src/pages/ContentStudio.tsx`

1. Add `sourceMode` form field
2. Replace checkbox-driven source behavior with source-mode-driven payload
3. Keep collection and advanced RAG settings
4. Submit `source_strategy` in the generation payload
5. Maintain backward-compatible fields during rollout


## Acceptance Tests

### Test 1: Dossier only

Input:

- valid dossier
- no RAG collection
- source mode = `dossier_only`

Expected:

- no RAG query
- no Linkup/Tavily
- TOC and article use dossier-backed claims/questions only

### Test 2: Dossier + RAG

Input:

- valid dossier
- RAG collection selected
- source mode = `dossier_plus_rag`

Expected:

- RAG queried globally and per section
- no Linkup/Tavily
- TOC and sections reflect dossier + RAG

### Test 3: Dossier + RAG + Live Web Refresh

Input:

- valid dossier
- RAG collection selected
- source mode = `dossier_plus_rag_plus_live_web`

Expected:

- RAG queried
- Linkup/Tavily allowed
- evidence set merged and ranked across both sources

### Test 4: RAG only

Input:

- no dossier
- RAG collection selected
- source mode = `rag_only`

Expected:

- RAG queried
- no Linkup/Tavily
- generation proceeds from brief + RAG

### Test 5: Weak RAG in dossier + RAG mode

Input:

- valid dossier
- weak or empty RAG results
- source mode = `dossier_plus_rag`

Expected:

- no live web fallback
- generation still proceeds using dossier + any available RAG
- low RAG coverage is logged only


## Final Recommendation

The best workflow for the current product is:

1. Use Deep Research to create the dossier.
2. Let users manually add timely or proprietary material into RAG right before generation.
3. Generate with `Dossier + RAG` by default.
4. Use `Dossier + RAG + Live Web Refresh` only for freshness-sensitive articles.
5. Keep Gap Analysis as a separate optional enrichment workflow.

This aligns the product with the user's mental model:

- The dossier is the research backbone.
- RAG is the private memory layer.
- Live web refresh is an explicit freshness decision.

