# Excerpt ↔ article alignment

## Problem

The excerpt (standfirst) is written at outline time from the **brief**. The finished
article follows its own **outline**, so the two drift apart: the excerpt advertises
sections, numbers and outcomes the article never contains. WordPress prints it under
the title, social cards and feeds reuse it, and it is the first paragraph a reader
sees — the most-read sentence in the article describes a piece that was not written.

Measured on 14 recent production articles, one carried an excerpt whose support by
its own body was 34% ("accelerate", "adoption", "advantage", "ai-driven" — none of
them in the text). The other 13 were grounded.

## Contract

The excerpt must describe the article that is being saved. Enforced in two places,
both deterministic-first:

1. **Outline generator** (`article_structure_generator.py`)
   - The excerpt is generated **after** the section plan and its prompt carries the
     planned sections (titles, subtitles, key points) as the only permitted source of
     claims and numbers — not the brief alone.
   - `_enforce_outline_alignment()` rejects an excerpt the plan does not support:
     two retries that name the failing terms, then a rebuild from the outline's own
     key points. Nothing is synthesised.
2. **Finalization / final validator** (`_reconcile_excerpt` in `tasks.py`, runs with
   the `final_review` role)
   - Compares the stored excerpt with the finished document (references excluded).
   - On a mismatch: one rewrite requested against the finished text, accepted only if
     it clears the same bar; otherwise an excerpt cut from the article's own lede.

## The check

`src/services/excerpt_alignment.py`

- `evaluate_excerpt_alignment(excerpt, body)` →
  `matches`, `support`, `unsupported_terms`, `missing_numbers`, `reason`.
  - **Numbers first**: every figure in the excerpt must exist in the article
    (`41%` / `41 percent` / `41` compare equal).
  - Then **support**: ≥50% of the excerpt's content words must appear in the article.
    Function words and standfirst framing words (`guide`, `practical`, `insight`,
    `learn`, …) are excluded — otherwise a brief-driven excerpt clears the bar on
    "practical guide" alone.
- `derive_excerpt_from_article(body)` — first lede paragraph, labels and citation
  markers stripped, capped at 70 words / 300 chars.
- `derive_excerpt_from_outline(sections)` — the planner's own key points; returns `""`
  rather than composing a sentence that says nothing.

## Log lines

```
Excerpt check: kept (support 0.79, 30 words, 24 terms): excerpt is supported by the article
Excerpt rejected against the outline (attempt 1/2): only 21% of the excerpt's terms appear …
Excerpt check: rewritten by <provider>/<model> (41 -> 46 words, support 0.31 -> 0.86): …
Excerpt check: rebuilt from the article's lede (41 -> 38 words, support 1.0): …
Excerpt check: no grounded replacement found, keeping the unmatched excerpt: …
```

`result['excerpt_report']` and `quality_report['excerpt_alignment']` carry the same
data for the UI (`source` is `kept` / `model` / `derived`).

## Flags

- `ARTICLE_EXCERPT_GUARD=false` — skip the finalization reconciliation.
- `ARTICLE_EXCERPT_TIMEOUT` (default 75s) — rewrite call timeout.

## Tests

`tests/unit/test_excerpt_alignment.py` — the deterministic check (paraphrase passes,
off-topic and dropped-number excerpts fail, framing words alone do not carry), both
repairs, and the outline-stage prompt/retry/fallback behaviour. All offline; no model
calls.
