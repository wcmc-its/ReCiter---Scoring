# ReCiter External-Validation Harness

This harness measures how well ReCiter's publication-scoring model works at
institutions it was never trained on.

## What this is, for someone new to ReCiter

**ReCiter** is an author name disambiguation system: it works out which
publications belong to a given researcher. The hard part is name collisions —
telling *your* "J. Smith" apart from the hundreds of other J. Smiths in PubMed.

**CARE** (Composite Author Recognition Engine) is the scoring model inside
ReCiter. For each candidate article it produces one number from 0 to 100: the
calibrated confidence that the article belongs to the researcher. A 99 means
"almost certainly theirs"; a 5 means "almost certainly not."

A **gold standard** is the ground truth for one institution: for each
researcher, a list of articles a human curator has confirmed (`ACCEPTED`) or
ruled out (`REJECTED`). It is what the model is scored against.

**External validation** is the question this harness answers. CARE was trained
on Weill Cornell Medicine data. Does it still work somewhere else — a different
institution, different researchers, different curation habits — *without being
retrained*? The only way to know is to run it against another institution's
gold standard and measure. This harness has done that for seven institutions:
the six in the UC-system run (UC Irvine, UCLA, USC, UC San Diego, UC Davis,
UCSF) and Fred Hutchinson Cancer Center.

## What it measures

Scoring every article is only step one. The harness then computes three things:

- **Discrimination (AUC-ROC)** — does the model reliably score real articles
  above wrong-person articles? 1.0 is perfect ranking.
- **Calibration (ECE)** — when the model says 90, is the article really theirs
  about 90% of the time? A well-calibrated score can be trusted as a
  probability, which is what lets it drive automated accept/reject decisions.
- **Review burden** — what fraction of articles land in the uncertain middle
  (score 10–95), where a human still has to look? Lower is better.

## How it works

```
  Institution's gold standard
  (researchers + their PMIDs,
   each ACCEPTED or REJECTED)
            |
            v
  +-----------------------------+
  |  ReCiter, run locally       |   <- the "local stack": no AWS needed
  |  - retrieve candidate       |
  |    articles from PubMed     |
  |  - compute evidence         |
  |    features per article     |
  |  - CARE model -> 0-100      |
  +-----------------------------+
            |
            v
  +---------------+   +-------------------------+
  | Evaluation    |   | Review export           |
  | AUC, ECE,     |   | per-article spreadsheet  |
  | review burden |   | with a suggested action  |
  +---------------+   +-------------------------+
```

1. **Prepare** — a per-institution data builder turns a curator's spreadsheet
   into one CSV per institution (`PersonID, name, PMID, Assertion`):
   `build_uc_validation_data.py` for the UC-system sites,
   `build_fredhutch_data.py` for Fred Hutch.
2. **Score** — `run_uc_validation_all.sh` (or `run_external_validation.py` for
   a single institution) runs every researcher through ReCiter and records each
   article's 0–100 score.
3. **Evaluate** — `run_external_validation.py --evaluate-only` and
   `aggregate_uc_evaluation.py` compute AUC, calibration, and review burden,
   per institution and pooled.
4. **Export for review** — `build_review_export.py` writes a per-article
   spreadsheet a curator can act on, each row tagged with a plain-English
   suggested action (e.g. "Add: very likely missing").

## The local stack, and ReCiter Desktop

ReCiter normally runs in the cloud — on AWS, across Kubernetes (EKS), a scoring
Lambda, DynamoDB, and separate retrieval services. That is fine for a hosted
service, but it makes "just run ReCiter on this data" a heavy lift, and it is a
non-starter for an external institution that has no ReCiter infrastructure.

So this harness runs ReCiter as a **standalone local stack**.
`start_local_reciter.sh` brings up DynamoDB Local (in Docker), a local PubMed
retrieval service, a local scoring service, and the ReCiter Java app — all on
one machine, no AWS account required.

That local stack is the **proof-of-concept for ReCiter Desktop**, the planned
effort to make ReCiter an installable desktop application anyone can run
without cloud infrastructure. The relationship is proof-of-concept to product:

- The scripts here — `start_local_reciter.sh`, `stop_local_reciter.sh`,
  `local_scoring_service.py` — are the first working "ReCiter with no cloud."
  They were built for this validation work and are driven by shell scripts, not
  packaged as an application.
- ReCiter Desktop would productize that: a real installer, a user interface,
  and no manual stack management.

When ReCiter Desktop matures, this harness would call it instead of managing
the stack itself — the local-stack scripts here are the seed it grows from. For
now the two are kept separate: this harness is self-contained so it can be
cited as one unit by the CARE methodology paper. A reader who only wants to
reproduce the validation needs nothing from ReCiter Desktop, and a future
ReCiter Desktop can adopt these scripts without disturbing the paper's frozen
reference.

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
(`Assertion` is `ACCEPTED` or `REJECTED`). `build_uc_validation_data.py` (UC
system) and `build_fredhutch_data.py` (Fred Hutch) derive these from a source
spreadsheet.

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
scripts/build_review_export.py            # per-article reviewer CSV/workbook
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
