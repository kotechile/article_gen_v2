# Phase 12 Rollout + Backfill Runbook

This runbook covers migration, safe rollout, and backfill for the angle-first workflow.

## 1) Apply schema migrations

Run these SQL scripts in Supabase SQL editor (or migration runner) in order:

1. `migrations/add_research_topic_angle_fields.sql`
2. `migrations/add_project_topic_candidate_angle_fields.sql`
3. `migrations/add_subtopics_angle_cluster_fields.sql`

## 2) Dry-run metadata backfill

Use service role credentials in environment:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`

Dry run:

```bash
python3 scripts/backfill_angle_metadata.py --dry-run --limit 1000
```

Apply:

```bash
python3 scripts/backfill_angle_metadata.py --apply --limit 1000
```

## 3) Rollout stages

1. **Internal validation**  
   Run decomposition + idea burst on known topics and review:
   - topic context visibility
   - internal-link grouping
   - ranking breakdown chips
   - ranking logs in backend (`idea_ranking_detail`, `idea_ranking_summary`)

2. **Limited user rollout**  
   Enable for a small set of projects/users and compare:
   - title specificity
   - decision relevance
   - software idea actionability
   - topic drift rate

3. **Full rollout**  
   Ship to all users once quality and reliability remain stable.

## 4) Legacy behavior

Legacy topics remain functional because backend derives fallback metadata when explicit angle fields are missing.

## 5) Compare before/after quality

For the same sample topics, capture:

- old generated ideas (baseline)
- new generated ideas

Then score manually for:
- category alignment
- decision usefulness
- article specificity
- software idea usefulness
