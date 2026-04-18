# Phase 13 QA Test Matrix (Angle-First Workflow)

Use this matrix to validate end-to-end quality from category selection to article/app ideas.

## How to use

- Run each test on production-like data.
- Capture evidence (screenshots + API response snippets + logs).
- Mark each case: `Pass`, `Fail`, or `Partial`.
- A release is ready when all **P0** tests pass and no **P1** has a blocking defect.

---

## Test Cases

| ID | Priority | Area | Scenario | Steps | Expected Result | Evidence |
|---|---|---|---|---|---|---|
| QA-01 | P0 | Topic creation | Category/subcategory selected topic includes angle metadata | Create a topic from category/subcategory flow | Topic has `intent_bucket`, `decision_focus`, `angle_question`, `value_layer_tags` populated | Topic JSON/API + UI screenshot |
| QA-02 | P0 | Backward compatibility | Legacy topic without metadata still decomposes correctly | Open older topic and click `Decompose Topic` | Decomposition succeeds; fallback metadata is applied; no crash | Worker logs + subtopics API |
| QA-03 | P0 | Decomposition quality | Subtopics stay aligned to category lens | Pick a narrow category and decompose | Subtopics remain on-category; low drift/generic output | Subtopic list screenshot |
| QA-04 | P0 | Cluster metadata | Subtopics contain cluster/intent fields | Run decomposition + expansion | Subtopics include `cluster_type`, `primary_user_outcome`, `serp_intent_match`, `tool_potential_score` | Subtopics API payload |
| QA-05 | P0 | Idea Burst context | Idea generation uses angle + decision context | Open subtopic and run Idea Burst | Modal shows context chips (intent/tags/decision/angle) and generates ideas | Modal screenshot |
| QA-06 | P0 | Article idea structure | Blog ideas include new structured fields | Generate blog ideas | Each idea includes `FORMAT`, `INTENT`, `USER_DECISION_HELPED`, `INTERNAL_LINK_HOOK` and stored fields | DB row/API response |
| QA-07 | P0 | Software idea structure | Software ideas include product fields | Generate software ideas | Each idea includes `PRODUCT_TYPE`, `USER_JOB`, `KEY_INPUTS`, `OUTPUT_RESULT`, `BUILD_COMPLEXITY`, `DISTRIBUTION_ANGLE` | DB row/API response |
| QA-08 | P1 | Ranking explainability | Opportunity ranking is visible + coherent | Generate ideas with mixed viability/volume | `opportunity_score` present, factor chips visible, top ideas explainable by breakdown | Modal screenshot + API |
| QA-09 | P1 | Internal linking | Related ideas grouped by internal hooks | Generate multiple blog ideas with similar hooks | Internal Link Groups panel appears and groups related hooks/counts | Modal screenshot |
| QA-10 | P1 | Map-back traceability | Ideas map back to project/category/angle/cluster | Open generated ideas cards | Cards show map-back badges and align with selected topic/subtopic | Modal screenshot |
| QA-11 | P1 | Logging observability | Ranking logs emitted for diagnostics | Run Idea Burst once | Logs include `idea_ranking_detail` and `idea_ranking_summary` entries | Backend/worker logs |
| QA-12 | P1 | Error handling | Missing provider/key path returns clean user error | Temporarily set invalid key/provider in test env | User sees clear error; no unhandled exception loops | UI error + worker log |
| QA-13 | P2 | Performance | Decompose + Idea Burst latency acceptable | Run on 3 representative topics | Response times stay within team thresholds (define target) | Timing notes |
| QA-14 | P2 | Security/tenant scope | User can only affect own project data | Run flow with User A and User B | No cross-project leakage in topics/subtopics/ideas | API checks |

---

## Defect Severity Guidance

- **Blocker**: Prevents decomposition/idea generation or causes data corruption.
- **High**: Wrong category alignment or missing required structured fields.
- **Medium**: UI explainability gaps, non-critical ranking mismatch.
- **Low**: Cosmetic issues, wording, non-blocking layout.

---

## Release Gate

Ship only when:

1. All `P0` tests pass.
2. No open Blocker/High defects.
3. At least one successful end-to-end run is documented with:
   - topic creation,
   - decomposition,
   - idea burst,
   - published/saved idea action.
