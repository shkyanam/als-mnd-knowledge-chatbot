# Evaluation baseline and target

## Audience and problem

The product is for MND learners and research users who need to locate a precise finding across the project’s open-access PMC literature corpus. The current pain is evidence discovery: a search can identify a relevant article while still failing to surface the passage that contains the answer.

## Metric

Use **top-5 evidence task completion** as the initial baseline unit. For each fixed question, mark the task complete (`1`) when at least one of the five excerpts supplied to the answer step contains:

1. the expected source article; and
2. all evidence terms required for the reference answer.

Otherwise mark it incomplete (`0`). This measures the retrieval bottleneck directly and uses the same unit before and after deployment. It does not measure clinical correctness or replace human evaluation of the generated wording.

## Seed baseline

The first seeded question is:

> In the ALS speech BCI study, how did active-period spectral signal-to-noise ratio (SNR) change over time?

The expected evidence is in `PMC13042181.xml` and includes approximately `1 dB`, `6 dB`, and `3 dB`. On the current 1,907-chunk semantic index, measured on 2026-08-31:

| Retriever | Evidence result | Top-5 task completion |
| --- | --- | ---: |
| Dense-only baseline | Exact evidence ranked 8th, outside the five excerpts | 0/1 (0%) |
| Hybrid deployment | Exact evidence retained in the five excerpts | 1/1 (100%) |

The dense-only error rate for this seed task is therefore **100%**, and the hybrid smoke-test error rate is **0%**. Because this is one diagnostic question (`n=1`), these figures are a baseline demonstration, not a statistically general retrieval or answer-accuracy claim.

Run the check after the corpus has been downloaded and indexed:

```bash
uv run python evaluate_retrieval.py
```

The script reports both modes. `--mode dense` measures the baseline; `--mode hybrid` measures the current retriever.

## Deployment target

Create a fixed 20–30-question evaluation set covering factual, numeric, terminology, source-specific, and unanswerable questions. Apply the same binary rubric and report:

- top-5 evidence task completion: target at least **90%**;
- source-link accuracy: target **100%** for answered questions; and
- answer faithfulness: manually review every material claim for support in the supplied excerpts.

The 90% target is the first meaningful post-deployment comparison. The seed result should remain in the set as a regression test, but it should not be treated as the full evaluation set.

## Repository access note

The canonical repository is [shkyanam/als-mnd-knowledge-chatbot](https://github.com/shkyanam/als-mnd-knowledge-chatbot), on the `main` branch. Source code, dependency files, and architecture documents are tracked in Git. The downloaded XML corpus, prepared burden export, and local Chroma index are ignored generated/local data; a reviewer can recreate them using the setup commands in `README.md`.
