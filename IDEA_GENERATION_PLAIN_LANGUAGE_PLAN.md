# Idea Generation Plain-Language + Context-Pack Plan

## Goal
Improve first-pass content-idea generation so ideas are phrased like real Google searches, reducing consultant jargon and improving downstream DataForSEO keyword enrichment quality.

## Problem
Current first-pass ideas can be too abstract (for example, "framework/playbook/architecture" language). That lowers keyword match quality in the next enrichment step.

## Strategy (Summary)
1. Use a compact **Context Pack** from subtopic fields to keep ideas aligned with Decision + Outcome without overloading the LLM.
2. Force the LLM to emit a realistic `SEARCH_PHRASE` first, then require the title to include it.
3. Apply strict anti-jargon constraints in prompt + parser/post-processing guardrails.
4. Keep DataForSEO enrichment as the second pass, but feed it better search-shaped ideas.

---

## Context Pack Design (Low Token, High Signal)
Use this per-idea generation request:

- `topic_title` (max 80 chars)
- `category_path` (max 80 chars)
- `subtopic_title` (max 80 chars)
- `intent_bucket` (enum)
- `cluster_type` (enum)
- `decision_focus` (max 140 chars)
- `primary_user_outcome` (max 140 chars)
- `angle_question` (max 140 chars)
- `top_keywords` (3-5 only, deduped)

Rules:
- Do not include full historical context, full trend blobs, or long category descriptions in prompt body.
- Keep entire context section under ~1,000 characters.
- If a field is missing, pass an empty value and continue.

---

## Prompt Contract Changes
Update both blog and software idea prompts in `idea_burst` to require:

Required output fields per idea:
- `SEARCH_PHRASE: ...` (2-5 words, plain-language, realistic query)
- `TITLE: ...` (must include exact `SEARCH_PHRASE`)
- `DESCRIPTION: ...` (one short sentence, practical)
- existing fields (`KEYWORDS`, `MONETIZATION`, etc.)

Hard constraints:
- Keep language plain; avoid consultant/corporate jargon.
- Must align to `decision_focus` and `primary_user_outcome`.
- Reject generic fluff titles.

Style constraints:
- Prefer one of these search-friendly patterns:
  - `how to ...`
  - `X vs Y`
  - `when to ...`
  - `should I ...`
  - `cost of ...`
  - `best ...`
  - `risks of ...`
  - `signs of ...`

Anti-jargon blocked terms (unless exact niche requirement):
- `framework`, `playbook`, `methodology`, `architecture`, `paradigm`, `strategic lens`, `optimization matrix`

---

## Parser + Guardrails (Deterministic)
Extend parsing and normalization in `parse_idea_response()` and `create_idea_dict()`:

1. Parse `SEARCH_PHRASE` if present.
2. Validate phrase quality:
   - 2-5 words
   - no banned jargon
   - no empty/placeholder text
3. Ensure title contains phrase:
   - if missing, prefix/reshape title with `SEARCH_PHRASE`.
4. Rewrite jargon-heavy titles:
   - map jargon to simpler terms (existing map + expanded blocked-term handling).
5. Fallback behavior:
   - if `SEARCH_PHRASE` missing, derive from `keywords[0]` or title tokens.
   - always produce usable title/description.

---

## Implementation Steps (Checklist)

### Phase 1: Context Pack + Prompt
- [x] Add helper function in `src/api/endpoints/research_topics.py` to build compact context pack.
- [x] Inject context pack in blog/software prompts.
- [x] Update output format docs in prompt with required `SEARCH_PHRASE`.
- [x] Add anti-jargon and 2 AM Google test instructions.

### Phase 2: Parsing + Rewrite Guard
- [x] Extend `parse_idea_response()` to capture `SEARCH_PHRASE`.
- [x] Add `_normalize_search_phrase()` helper.
- [x] Enforce title contains phrase in `create_idea_dict()`.
- [x] Expand `_normalize_idea_title()` replacements + banned-term rewrite.
- [x] Keep existing short-description fallback/capping behavior.

### Phase 3: DataForSEO Readiness Improvements
- [x] Ensure `primary_keywords` starts with chosen search phrase.
- [x] Keep keywords list deduped and plain-language.
- [x] Log reason when fallback phrase/title rewrite triggers.

### Phase 4: Observability + QA
- [x] Add structured logs for:
  - phrase extracted/generated
  - title rewritten (`true/false`)
  - banned-term hit count
- [ ] Add metrics counters:
  - `ideas_with_phrase_pct`
  - `titles_rewritten_pct`
  - `banned_term_rewrite_pct`
- [ ] Verify DFS enrichment non-zero coverage trend improves over baseline.

---

## Files to Change
- `src/api/endpoints/research_topics.py`
  - `idea_burst()` prompt templates
  - `parse_idea_response()`
  - `_normalize_idea_title()`
  - `create_idea_dict()`

Optional (if needed later):
- `src/api/endpoints/content_ideas.py`
  - add extra fallback if any legacy publish paths bypass first-pass normalization.

---

## Acceptance Criteria
- At least 95% of generated ideas include a valid `SEARCH_PHRASE` (2-5 words).
- At least 95% of titles include the exact `SEARCH_PHRASE`.
- Banned consultant jargon appears in less than 5% of generated titles.
- Description is present for 100% of new ideas.
- DataForSEO enrichment quality improves vs baseline:
  - higher `% ideas with >=1 non-zero keyword metric`
  - lower `% zero-metric idea sets`

---

## Rollout Plan
1. Ship behind feature flag: `IDEA_SEARCH_LANGUAGE_MODE=true`.
2. Run A/B for 3-5 days:
   - Control: existing prompts
   - Variant: context pack + search phrase contract
3. Compare enrichment coverage and title quality.
4. Promote variant to default if acceptance criteria are met.

---

## Risks and Mitigations
- Risk: Over-constraining prompt reduces creativity.
  - Mitigation: keep 5 ideas with varied formats, only constrain phrase realism.
- Risk: Phrase stuffing in titles.
  - Mitigation: enforce natural-language title rewrite rules.
- Risk: Missing context fields break alignment.
  - Mitigation: safe defaults + required minimal pack only.

---

## Notes
- This plan keeps Hot-in-the-news DataForSEO usage intact.
- This plan does not re-enable DataForSEO keyword mining at subtopic-generation stage.
- Implemented in code on April 21, 2026:
  - prompt contract now requires `SEARCH_PHRASE`
  - compact context pack is injected into both blog and software prompts
  - parser captures `SEARCH_PHRASE`
  - title + keyword guardrails enforce search phrase inclusion
