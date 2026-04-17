# Article Idea Workflow Improvement Checklist

This checklist is the implementation plan for improving article and application idea generation end to end.

Target workflow:

`Project/Site -> Category -> Intent -> Decision Focus -> Angle -> Validated Keywords -> Angle Cluster -> Article Ideas / App Ideas`

## Phase 0: Baseline and Safety

- [ ] Capture the current end-to-end flow from category selection to `Idea Burst`.
- [ ] Document current prompt inputs and outputs for:
- `Hot in the News`
- `AI propose-topics`
- `Decompose Topic`
- `Keyword expansion`
- `Idea Burst`
- [ ] Save 3-5 real example topics and outputs as baseline fixtures.
- [ ] Define success metrics:
- Higher relevance of subtopics to category/subcategory
- Better article specificity
- Better application idea quality
- Lower topic drift across stages
- [ ] Decide whether rollout will be behind a feature flag or replace the current flow directly.

## Phase 1: Data Model for Intent and Angles

- [x] Add structured fields to research topics or a related table for:
- `intent_bucket`
- `decision_focus`
- `angle_question`
- `value_layer_tags`
- `target_audience`
- `evidence_sources`
- `related_terms`
- [ ] Decide whether to store one `research_topic` per angle or keep one topic with multiple child angles.
- [x] Update backend models and response serializers to expose the new fields.
- [x] Update frontend types to support the new structured metadata.
- [x] Preserve backward compatibility for existing topics that only have `title` and `description`.

## Phase 2: Category Intent Layer

- [ ] Define 3-5 reusable intent buckets per category/subcategory.
- [ ] Add a rule or prompt layer that converts category/subcategory into:
- primary content intent
- likely decision types
- likely business value patterns
- [ ] Add value-layer tagging options such as:
- `roi-focused`
- `cost-vs-value`
- `hidden-cost-audit`
- `timing-decision`
- `location-decision`
- `appraisal-impact`
- `workflow-automation`
- `tool-builder`
- [ ] Ensure category intent is available to all later steps, not just topic creation.

## Phase 3: Replace Broad Topics with Angles

- [x] Update topic generation prompts so they output structured angles instead of only broad titles.
- [ ] Require each generated angle to include:
- short UI title
- angle question
- decision focus
- intent bucket
- rationale
- source signals
- related terms
- [x] Keep the UI-friendly short title, but store the full angle metadata behind it.
- [x] Ensure angle generation stays tightly aligned to site description and selected category/subcategory.
- [ ] Verify that trend-driven topics and manually created topics can both use the same angle structure.

## Phase 4: Generate Seed Keywords per Angle

- [x] Move seed keyword generation earlier in the flow so it starts from each angle.
- [x] Generate seed keywords grouped by intent type:
- question intent
- comparison intent
- ROI intent
- tool/application intent
- [x] Ensure seed keywords stay anchored to the angle question and decision focus.
- [x] Store seed keywords so later stages can explain why a cluster exists.
- [x] Preserve category lens during keyword expansion to reduce drift.

## Phase 5: Validate and Expand with Data

- [ ] Enrich each angle with external data using DataForSEO and existing providers.
- [ ] Treat keyword research as angle validation, not just keyword collection.
- [ ] Expand with:
- related queries
- question variants
- SERP variations
- trend indicators
- [ ] Add filtering that keeps:
- strong intent match
- strong business value
- relevant homeowner/investor/operator context
- [ ] Reject or down-rank keywords that are semantically related but off-lens for the site.

## Phase 6: Cluster Within the Angle

- [x] Refactor decomposition so clustering happens inside each angle, not across the whole topic globally.
- [x] Update prompts to produce cluster names that are article-useful, not generic theme labels.
- [ ] Prefer cluster names that imply:
- a problem
- a decision
- a comparison
- a checklist
- a framework
- an audit
- a calculator/tool
- [ ] Store cluster-level metadata:
- `cluster_type`
- `primary_user_outcome`
- `serp_intent_match`
- `tool_potential_score`
- [x] Ensure decomposition can still reuse trend context, project context, and category path.

## Phase 7: Upgrade Subtopics into Angle Clusters

- [ ] Decide whether to evolve the existing `subtopics` table or add a new `angle_clusters` concept.
- [x] If reusing `subtopics`, add fields for:
- `angle_id`
- `intent_bucket`
- `decision_focus`
- `value_layer_tags`
- `angle_question`
- `cluster_type`
- `primary_user_outcome`
- `tool_potential_score`
- [x] Preserve compatibility with existing UI components that expect `subtopics`.
- [ ] Update list and detail APIs to expose both legacy and new structured fields.

## Phase 8: Rewrite Article Idea Generation

- [x] Update `Idea Burst` article prompts to use:
- category/subcategory
- intent bucket
- decision focus
- angle question
- cluster keywords
- value-layer tags
- monetization context
- trend evidence
- [x] Require every article idea to output:
- title
- target intent
- primary keyword
- article format
- user decision helped
- monetization angle
- internal link hooks
- [x] Prefer titles that are specific, outcome-oriented, and decision-relevant.
- [ ] Reduce generic listicles unless they are truly the best fit for the angle.

## Phase 9: Rewrite Application Idea Generation

- [x] Separate application idea generation from article idea generation at the prompt and scoring level.
- [ ] Ask explicitly what repeated user action can be productized:
- calculator
- planner
- evaluator
- comparison tool
- dashboard
- workflow helper
- [x] Require every application idea to output:
- product name
- user job to be done
- key inputs
- output/result
- monetization path
- build complexity estimate
- search/distribution angle
- [x] Prefer software ideas that emerge naturally from repeated decision friction in the angle clusters.

## Phase 10: Scoring and Ranking

- [x] Replace simple viability-only ranking with a multi-factor opportunity score.
- [x] Include:
- search opportunity
- intent match
- business value
- user decision value
- software potential
- trend relevance
- internal-link fit
- [x] Make ranking explainable in the UI or API for debugging and trust.
- [x] Add logging to understand why one idea outranked another.

## Phase 11: UX and Internal Linking

- [x] Update the UI so users can see:
- category/subcategory
- intent bucket
- decision focus
- angle question
- value tags
- [x] Show why each subtopic or idea exists, not just the label.
- [x] Add internal linking hooks so related ideas can be grouped by value layer and decision type.
- [x] Ensure article ideas and application ideas can both map back to:
- project
- category
- angle
- cluster

## Phase 12: Migration and Rollout

- [x] Add migrations for any new columns or tables.
- [x] Backfill existing topics where reasonable using heuristics:
- derive intent from category
- derive decision focus from title/description
- [x] Leave legacy topics functional if no backfill is available.
- [ ] Roll out in phases:
- internal testing
- limited user-facing testing
- full release
- [ ] Compare output quality before and after on the same sample topics.

## Phase 13: Validation and QA

- [ ] Build a test set with representative categories and subcategories.
- [ ] Validate outputs for:
- relevance
- specificity
- category alignment
- user decision usefulness
- software idea usefulness
- [ ] Add automated tests for:
- angle generation shape
- decomposition context shape
- prompt parsing
- fallback behavior
- [ ] Add a manual review checklist for qualitative evaluation.

## Suggested Implementation Order

- [x] Step 1: Add structured angle fields to backend + frontend types.
- [x] Step 2: Update topic generation to emit angles instead of plain titles.
- [x] Step 3: Move seed keyword generation earlier and tie it to angles.
- [x] Step 4: Refactor decomposition to cluster within angle context.
- [x] Step 5: Rewrite `Idea Burst` prompts for article and app generation.
- [x] Step 6: Add ranking improvements and value-layer tagging.
- [x] Step 7: Update UI to expose intent, angle, and value metadata.

## Files to Start With

- [x] `src/api/endpoints/ai.py`
- [x] `src/api/endpoints/research_topics.py`
- [x] `src/services/semantic_expansion_service.py`
- [x] `src/services/enhanced_topic_decomposition_service.py`
- [x] `frontend/src/types/research.ts`
- [ ] `frontend/src/services/research-topics.service.ts`
- [x] `frontend/src/services/content-ideas.service.ts`

## Definition of Done

- [ ] Category selection reliably shapes all downstream steps.
- [ ] Every generated topic has clear intent and decision focus.
- [ ] Keyword generation is anchored to angles, not vague subtopics.
- [ ] Decomposition produces tight, article-useful clusters.
- [ ] Article ideas are specific, relevant, and strategically aligned.
- [ ] Application ideas emerge from real workflow friction and decision patterns.
- [ ] Outputs are easier to explain, rank, link, and monetize.
