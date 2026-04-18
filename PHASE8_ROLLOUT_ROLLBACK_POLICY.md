# Phase 8 Rollout and Rollback Policy

## Purpose

Define explicit go/no-go rules for promoting the new article pipeline behavior, based on measurable benchmark outcomes.

This policy is designed to:

- reduce subjective release decisions
- protect production quality during rollout
- make rollback fast and deterministic

## Scope

Applies to pipeline changes that affect:

- deep research ingestion and dossier usage
- claim/evidence ranking and confidence mapping
- structure generation and GEO blocks
- content generation orchestration
- humanization/refinement passes
- final quality gating behavior (`Created` vs `Needs Review`)

## Required Inputs

Before any rollout decision:

1. Run benchmark comparison using:
   - `python3 scripts/run_phase8_benchmark.py ...`
2. Produce a fresh snapshot pair (JSON + Markdown).
3. Confirm sample size:
   - minimum `n=30` evaluated articles per cohort for decision-grade confidence
4. Include at least:
   - evergreen informational
   - technical
   - comparison/commercial
   - sensitive/high-stakes

## Primary Go/No-Go Thresholds

Candidate can be promoted only if ALL thresholds pass:

1. `avg_overall_score_delta >= +3.0`
2. `avg_humanization_score_delta >= +2.0`
3. `avg_geo_score_delta >= +2.0`
4. `avg_grounding_score_delta >= 0.0` (no regression allowed)
5. `below_60_overall_delta <= -20%` relative improvement or better
6. Sensitive-topic subset:
   - no decrease in grounding score
   - no increase in low-confidence-claim rate

If any threshold fails, decision is `NO-GO`.

## Guardrail Thresholds (Hard Stops)

Rollout is blocked immediately if any of the following occur in benchmark results:

1. Grounding regression:
   - `avg_grounding_score_delta <= -2.0`
2. Quality collapse:
   - `avg_overall_score_delta <= -2.0`
3. GEO regression:
   - `avg_geo_score_delta <= -2.0`
4. Safety regression (sensitive bucket):
   - uncited paragraph count increases by more than 15%
   - low-confidence claim count increases by more than 15%

## Rollout Stages

### Stage A: Internal Dry Run (0%)

- Run benchmark and manual review.
- Verify `Needs Review` gating behavior is expected.
- Confirm no schema/update failures in logs.

### Stage B: Canary (10%)

- Enable new behavior for ~10% of generation jobs.
- Monitor for at least 24 hours:
  - quality gate outcomes
  - failure rates
  - timeout/latency anomalies
  - user-visible regressions

Promotion to next stage requires:

- no hard-stop trigger
- stable task success rate (no more than 2% relative drop)

### Stage C: Expanded Canary (30%)

- Run for 24-48 hours.
- Re-run benchmark snapshot against recent outputs.
- Validate sensitive-topic metrics separately.

### Stage D: Broad Rollout (60%)

- Continue monitoring and drift checks.
- Ensure `Needs Review` volume remains operationally manageable.

### Stage E: Full Rollout (100%)

- Promote to default.
- Keep rollback path active for one release window.

## Rollback Triggers

Rollback immediately if any production signal crosses thresholds:

1. Task failure rate increases by >5% relative to baseline.
2. Average grounding score drops by >= 2 points.
3. `Needs Review` rate spikes above expected capacity by >25%.
4. Critical user-facing regression is confirmed (formatting, citations, coherence, or safety issue).
5. Hard-stop benchmark criteria are met in post-deploy sample.

## Rollback Procedure

1. Disable new pipeline behavior flags (or revert deployment commit).
2. Confirm incoming jobs are routed to prior stable behavior.
3. Record incident snapshot:
   - timestamp
   - active config/flags
   - affected metrics
   - rollback reason
4. Re-run benchmark against rollback outputs to confirm recovery.
5. Open follow-up remediation task before next rollout attempt.

## Monitoring Checklist

During canary and rollout, track:

- average overall/humanization/grounding/GEO scores
- low-confidence claim distribution
- uncited paragraph count
- generation latency by stage
- task success/failure rate
- `Created` vs `Needs Review` distribution
- sensitive-topic performance split

## Decision Log Template

Use this lightweight template for each promotion decision:

- Date:
- Stage:
- Snapshot files:
- Sample size:
- Overall delta:
- Humanization delta:
- Grounding delta:
- GEO delta:
- Below-60 delta:
- Sensitive-topic result:
- Decision (`GO`/`NO-GO`):
- Notes:

## Ownership

- Engineering owner: pipeline reliability + implementation
- Content/Editorial owner: qualitative review + language quality
- Product owner: rollout timing + risk acceptance

A rollout decision is valid only when all three roles sign off.

