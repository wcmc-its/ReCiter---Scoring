# ReCiter External-Validation Harness

Scripts to run ReCiter's CARE scoring model against an institution's
gold-standard publication assertions and evaluate the results — discrimination
(AUC-ROC), probability calibration (ECE), and manual-review burden.

This harness produced the multi-site external validation reported for the CARE
methodology: six institutions in the University of California system (UC
Irvine, UCLA, UC San Diego, UC Davis, UCSF) plus the University of Southern
California, and Fred Hutchinson Cancer Center. The WCM-trained scoring model is
applied to each institution's researchers without retraining.

It runs ReCiter as a fully local stack — DynamoDB Local, a local PubMed
retrieval service, and a local scoring service — so no AWS account is required.

## Layout

```
validation/
  scripts/              run + evaluation + audit scripts
  src/                  preprocessing, evaluation, calibration modules
  external_validation/
    uc_system/          per-institution spring configs (*.json) + properties
    fredhutch_*.json    Fred Hutch spring configs
    results/            run outputs (gitignored, created at runtime)
```

## Prerequisites

| Component | Requirement |
|---|---|
| Java | 17 (Lombok requires it) |
| Docker | for DynamoDB Local |
| Python | 3.12; `pip install -r requirements.txt` |
| ReCiter (Java) | `wcmc-its/ReCiter`, `development` branch at or after PR #625 (the `DynamoDbS3Operations` null-guard; merged 2026-05-16). The run depends on this fix. |
| ReCiter---Scoring | this repo, `dev_v2` — the scoring models in `app/models/` (XGBoost 3.2.0) |
| ReCiter-PubMed-Retrieval-Tool | `wcmc-its/ReCiter-PubMed-Retrieval-Tool` |
| NCBI API key | optional, `PUBMED_API_KEY` — raises esummary rate limits |

The harness locates the scoring repo automatically (it lives inside it). Point
it at the other two repos via env vars if they are not at the default
`~/Dropbox/GitHub/` locations:

```
export RECITER_DIR=/path/to/ReCiter
export PUBMED_DIR=/path/to/ReCiter-PubMed-Retrieval-Tool
```

## Input data

Researcher data is **not included** — it is identifiable and gitignored. Each
institution needs a CSV at `external_validation/uc_system/data/<inst>_data.csv`
with columns: `PersonID, FirstName, MiddleName, LastName, PMID, Assertion`
(`Assertion` ∈ `ACCEPTED` / `REJECTED`). `build_uc_validation_data.py` derives
these from a source spreadsheet.

## Reproduce a run

```bash
pip install -r requirements.txt

# Score all six UC-system institutions (sequential; multi-hour).
#   FILTER_BY_FEEDBACK=ALL also scores PENDING articles (new-match discovery);
#   omit it to score only curated ACCEPTED/REJECTED articles.
FILTER_BY_FEEDBACK=ALL scripts/run_uc_validation_all.sh

# Single institution (also the Fred Hutch path — institution-generic):
scripts/run_external_validation.py \
    --institution fredhutch \
    --data-file external_validation/uc_system/data/fredhutch_data.csv \
    --config external_validation/fredhutch_spring_config.json \
    --uid-prefix "fredhutch_" --base-url http://localhost:8081 --non-interactive

# Evaluate already-scored results (no API needed — reads score files):
scripts/run_external_validation.py --institution <inst> ... --evaluate-only
scripts/aggregate_uc_evaluation.py        # cross-institution metrics
scripts/recency_matched_evaluation.py     # recency-matched re-evaluation
scripts/build_uc_review_export.py         # per-article reviewer CSV/workbook
```

`run_external_validation.py --help` documents every flag.

## Pinned versions

The published results were produced with:
- `wcmc-its/ReCiter` — `development`, post-PR-625
- `wcmc-its/ReCiter---Scoring` — `dev_v2`, scoring models from commit `2bcb707`
- harness scripts — extracted from the research working tree at `d9191b5`

XGBoost must be 3.2.0; the scoring models were trained with it and the isotonic
calibrator amplifies cross-version drift. This is enforced by the inherited
`../requirements.txt`.
