# Idea Enrichment Migration Plan

This plan tracks moving SEO/offer enrichment from subtopics to content ideas.

## Goal

Make enrichment reliable by running it on content ideas (which already include keywords), and stop depending on long-tail subtopic phrases for SEO metrics.

## Backend First

- [x] Add idea-level enrichment endpoint in `src/api/endpoints/content_ideas.py`
- [x] Accept `idea_ids` and enforce user ownership
- [x] For each selected idea, compute keyword-based SEO metrics and offer signals
- [x] Persist enrichment fields on `content_ideas` rows
- [x] Return structured per-idea status (`enriched` / `failed`) with reason
- [x] Add safe fallbacks for schemas missing optional columns

## Frontend Second

- [x] Add idea-level `Run SEO/Offers` action in idea workflow UI
- [x] Trigger backend idea enrichment endpoint from `content-ideas.service.ts`
- [x] Show per-idea enrichment status and errors in the modal/list
- [x] Remove/de-emphasize subtopic-level enrichment trigger and failure copy
- [ ] Keep subtopics focused on discovery (intent/angle), not SEO metrics

## Validation

- [ ] Verify endpoint with valid idea IDs updates DB rows
- [ ] Verify partial-failure response surfaces in UI
- [ ] Verify no regressions in generate ideas / publish / in-library flow
- [ ] Verify Topic Detail no longer relies on subtopic SEO enrichment to appear healthy

## Rollout

- [ ] Deploy backend + frontend together
- [ ] Smoke test in production topic with existing ideas
- [ ] Confirm performance and error logs are acceptable
