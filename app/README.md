# Developer Reference: ReCiter 4.0 — CART Scoring System

**Version:** ReCiter 4.0
**Component:** CART (Calibrated Author Recognition Technique)

This folder contains everything needed to understand, test, and integrate the CART scoring system.

## Contents

| File | Description |
|------|-------------|
| `sample_input_ajg9004.json` | Raw input file (1,216 articles for one researcher) |
| `expected_outputs_ajg9004.json` | Expected scores for all articles (both models) |
| `preprocessing.py` | Feature computation and derived feature logic |
| `feedbackIdentityModel.joblib` | Feedback+Identity XGBoost model |
| `feedbackIdentityCalibrator.joblib` | Feedback+Identity isotonic calibrator |
| `feedbackIdentityScaler.joblib` | Feedback+Identity feature scaler |
| `identityOnlyModel.joblib` | Identity-Only XGBoost model |
| `identityOnlyCalibrator.joblib` | Identity-Only isotonic calibrator |
| `identityOnlyScaler.joblib` | Identity-Only feature scaler |
| `verify_setup.py` | Verification script (run to confirm setup) |

## Quick Start

### 0. Verify Your Setup

```bash
python verify_setup.py           # Standard output (authorshipLikelihoodScore only)
python verify_setup.py --debug   # Include intermediate values for troubleshooting
```

### 1. Load and Score Articles

```python
import json
import joblib
import pandas as pd
import numpy as np
from preprocessing import (
    FEEDBACK_IDENTITY_FEATURES, FEEDBACK_IDENTITY_BASE_FEATURES,
    IDENTITY_ONLY_FEATURES, IDENTITY_ONLY_BASE_FEATURES,
    compute_derived_features_feedback_identity,
    compute_derived_features_identity_only
)

# Load models
fb_model = joblib.load("feedbackIdentityModel.joblib")
fb_cal = joblib.load("feedbackIdentityCalibrator.joblib")
fb_scaler = joblib.load("feedbackIdentityScaler.joblib")

io_model = joblib.load("identityOnlyModel.joblib")
io_cal = joblib.load("identityOnlyCalibrator.joblib")
io_scaler = joblib.load("identityOnlyScaler.joblib")

# Load input data
with open("sample_input_ajg9004.json") as f:
    articles = json.load(f)
df = pd.DataFrame(articles)

# Preprocess for Feedback+Identity model
for feat in FEEDBACK_IDENTITY_BASE_FEATURES:
    if feat not in df.columns:
        df[feat] = 0
    df[feat] = df[feat].fillna(0)
df_fb = compute_derived_features_feedback_identity(df.copy())

# Scale and predict
X_fb = fb_scaler.transform(df_fb[FEEDBACK_IDENTITY_FEATURES].values)
raw_probs = fb_model.predict_proba(X_fb)[:, 1]
calibrated_probs = fb_cal.predict(raw_probs)

# Final output: authorshipLikelihoodScore (0-100 scale)
authorshipLikelihoodScore = calibrated_probs * 100

print(f"First article: authorshipLikelihoodScore={authorshipLikelihoodScore[0]:.2f}")
# Expected: 100.00
```

### 2. Verify Against Expected Outputs

```python
with open("expected_outputs_ajg9004.json") as f:
    expected = json.load(f)

# Check first article
exp_art = expected["articles"][0]
print(f"Article {exp_art['articleId']}:")
print(f"  Expected FB score: {exp_art['scores']['feedback_identity']['authorshipLikelihoodScore']}")
print(f"  Expected IO score: {exp_art['scores']['identity_only']['authorshipLikelihoodScore']}")
```

## Two Model Types

### Identity-Only Model
- **Use case**: New researchers with no feedback history (cold-start)
- **Features**: 18 base + 1 derived (identityStrength)
- **No access to**: countAccepted, countRejected, feedbackScore* features

### Feedback+Identity Model
- **Use case**: Researchers with curation history
- **Features**: 31 base + 4 derived
- **Leverages**: Historical acceptance/rejection patterns via feedback synthesis

## Input Schema

Each article in the input JSON should have:

```json
{
  "articleId": 31625646,
  "feedbackScoreCites": 0.25,
  "feedbackScoreCoAuthorName": 0.11,
  "feedbackScoreEmail": 0.0,
  "feedbackScoreInstitution": 0.50,
  "feedbackScoreJournal": 0.12,
  "feedbackScoreJournalSubField": -0.55,
  "feedbackScoreKeyword": 1.16,
  "feedbackScoreOrcid": 0.0,
  "feedbackScoreOrcidCoAuthor": 4.0,
  "feedbackScoreOrganization": 0.50,
  "feedbackScoreTargetAuthorName": -0.49,
  "feedbackScoreYear": 0.0,
  "articleCountScore": -2.06,
  "authorCountScore": -0.44,
  "discrepancyDegreeYearScore": 0.0,
  "emailMatchScore": 0.0,
  "genderScoreIdentityArticleDiscrepancy": 0.24,
  "grantMatchScore": 0.0,
  "journalSubfieldScore": -0.47,
  "nameMatchFirstScore": 1.852,
  "nameMatchLastScore": 0.664,
  "nameMatchMiddleScore": 0.794,
  "nameMatchModifierScore": 0.0,
  "organizationalUnitMatchingScore": 0.0,
  "scopusNonTargetAuthorInstitutionalAffiliationScore": 0.3,
  "targetAuthorInstitutionalAffiliationMatchTypeScore": 1.8,
  "pubmedTargetAuthorInstitutionalAffiliationMatchTypeScore": 0.0,
  "relationshipPositiveMatchScore": 0.21,
  "relationshipNegativeMatchScore": 0.48,
  "relationshipIdentityCount": 46,
  "countAccepted": 234,
  "countRejected": 513,
  "userAssertion": "ACCEPTED"
}
```

**Note**: The `feedbackScore*` features are **pre-computed** upstream (sigmoid transformation applied in the Java application). The Python pipeline receives these values as-is.

## Output Schema

The final output is `authorshipLikelihoodScore`: calibrated probability × 100 (0-100 scale).

```json
{
  "articleId": 31625646,
  "scores": {
    "identity_only": {
      "authorshipLikelihoodScore": 91.11,
      "_debug": {
        "raw_probability": 0.919451,
        "calibrated_probability": 0.911129
      }
    },
    "feedback_identity": {
      "authorshipLikelihoodScore": 100.0,
      "_debug": {
        "raw_probability": 0.99986,
        "calibrated_probability": 1.0
      }
    }
  }
}
```

**Interpretation:**
- `authorshipLikelihoodScore = 100` → Article almost certainly belongs to this researcher
- `authorshipLikelihoodScore = 0` → Article almost certainly does NOT belong to this researcher
- `authorshipLikelihoodScore = 50` → Maximum uncertainty

**Note:** The `_debug` section contains intermediate values for troubleshooting only.

## Derived Features

The preprocessing module computes these derived features:

| Feature | Model | Description |
|---------|-------|-------------|
| `identityStrength` | Both | Max of email/ORCID/affiliation signals (0-1) |
| `acceptanceRateLowerBound` | FB only | Wilson score lower bound on acceptance rate |
| `feedbackConfidence` | FB only | log(1 + countAccepted + countRejected) |
| `uncertainRejectionRisk` | FB only | Risk score for uncertain high-rejection cases |

## Expected Results for ajg9004

| Metric | Identity-Only | Feedback+Identity |
|--------|---------------|-------------------|
| Mean authorshipLikelihoodScore | 22.4 | 20.8 |
| Auto-accept (≥95) | 104 | 247 |
| Needs review (30-95) | 127 | 4 |
| Auto-reject (<30) | 985 | 965 |
| Accuracy on labeled (n=748) | 94.4% | 99.7% |

## Tolerance for Verification

When comparing your implementation against expected outputs:
- **authorshipLikelihoodScore**: ±0.1 tolerance (0-100 scale)
- **Calibrated probabilities**: ±0.001 tolerance (0-1 scale)
- **Raw probabilities**: ±0.0001 tolerance
- **Derived features**: ±0.000001 tolerance

Larger discrepancies indicate implementation issues (feature order, scaling, etc.).

## Troubleshooting

### Scores don't match expected values
1. Verify feature order matches `FEEDBACK_IDENTITY_FEATURES` or `IDENTITY_ONLY_FEATURES`
2. Ensure NaN values are filled with 0 before scaling
3. Check that derived features are computed after base features are filled

### Calibrated scores saturate at 1.0
This is expected behavior. The isotonic calibrator maps high-confidence predictions to 1.0 when training data supports it.

### Cold-start (no feedback history)
For new researchers with `countAccepted=0` and `countRejected=0`:
- Use the Identity-Only model
- Or use Feedback+Identity but expect `acceptanceRateLowerBound=0.5` (maximum uncertainty)

## Model Provenance

**ReCiter 4.0** models trained using the **clean-split protocol**:
- Hash-based person-level splits (SHA-256)
- Train: buckets 0-69 (70%)
- Calibration: buckets 70-84 (15%)
- Test: buckets 85-99 (15%)
- No person appears in multiple splits (prevents data leakage)

See `RELEASE_NOTES.md` in the parent directory for full release documentation.
