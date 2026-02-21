#!/usr/bin/env python3
"""
preprocessing.py - Shared feature computation for ReCiter scoring models.

This module contains:
- Feature column definitions for both model types
- Wilson score interval calculation
- Derived feature computation (confidence-aware features)

Used by: feedbackIdentityCreateModel_*.py, identityOnlyCreateModel_*.py,
         and their corresponding inference scripts.
"""

import os
import re
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# CONFIGURATION
# =============================================================================

# Thresholds for identity signal strength (environment-configurable)
STRONG_EMAIL = float(os.getenv("STRONG_EMAIL", "0.90"))
STRONG_ORCID = float(os.getenv("STRONG_ORCID", "0.90"))
STRONG_AFFIL = float(os.getenv("STRONG_AFFIL", "0.95"))

_log = logging.getLogger(__name__)


# =============================================================================
# NAME FREQUENCY DATA (loaded once at import time)
# =============================================================================

def _load_name_frequency():
    """Load name frequency table from data/name_frequency.json.
    Returns (table_dict, median_score) or ({}, 0.0) if unavailable."""
    freq_path = Path(__file__).parent.parent / 'data' / 'name_frequency.json'
    if freq_path.exists():
        with open(freq_path, 'r') as f:
            table = json.load(f)
        scores = [v['score'] for v in table.values()]
        median = sorted(scores)[len(scores) // 2] if scores else 0.0
        _log.info(f"Loaded name frequency table: {len(table):,} names (median score={median:.4f})")
        return table, median
    return {}, 0.0


_NAME_FREQ_TABLE, _NAME_FREQ_MEDIAN = _load_name_frequency()


def _name_frequency_score(first_name):
    """Look up IDF-like frequency score for a first name.

    For compound names (e.g., "Jean-Pierre", "Sae hee"), splits into tokens,
    discards single-char initials, and averages the scores.
    Returns median score for unknown names or 0.0 if no frequency table loaded.
    """
    if not _NAME_FREQ_TABLE or not first_name or not isinstance(first_name, str):
        return 0.0

    first_name = first_name.strip().lower().replace('.', '')
    tokens = [t for t in re.split(r'[\s\-]+', first_name) if len(t) > 1]

    if not tokens:
        return _NAME_FREQ_MEDIAN

    scores = [_NAME_FREQ_TABLE[t]['score'] if t in _NAME_FREQ_TABLE else _NAME_FREQ_MEDIAN
              for t in tokens]
    return sum(scores) / len(scores)


# =============================================================================
# FEATURE COLUMN DEFINITIONS
# =============================================================================

# Feedback + Identity model: 31 base features
FEEDBACK_IDENTITY_BASE_FEATURES = [
    # 12 feedback score features (per-person learned patterns)
    'feedbackScoreCites', 'feedbackScoreCoAuthorName', 'feedbackScoreEmail',
    'feedbackScoreInstitution', 'feedbackScoreJournal', 'feedbackScoreJournalSubField',
    'feedbackScoreKeyword', 'feedbackScoreOrcid', 'feedbackScoreOrcidCoAuthor',
    'feedbackScoreOrganization', 'feedbackScoreTargetAuthorName', 'feedbackScoreYear',
    # 19 identity/matching features
    'articleCountScore', 'authorCountScore', 'discrepancyDegreeYearScore', 'emailMatchScore',
    'genderScoreIdentityArticleDiscrepancy', 'grantMatchScore', 'journalSubfieldScore',
    'nameMatchFirstScore', 'nameMatchLastScore', 'nameMatchMiddleScore', 'nameMatchModifierScore',
    'organizationalUnitMatchingScore', 'scopusNonTargetAuthorInstitutionalAffiliationScore',
    'targetAuthorInstitutionalAffiliationMatchTypeScore', 'pubmedTargetAuthorInstitutionalAffiliationMatchTypeScore',
    'relationshipPositiveMatchScore', 'relationshipNegativeMatchScore', 'relationshipIdentityCount',
    'countAccepted', 'countRejected'
]

# Identity-Only model: 18 base features (no feedback scores, no countAccepted/countRejected)
IDENTITY_ONLY_BASE_FEATURES = [
    'articleCountScore', 'authorCountScore', 'discrepancyDegreeYearScore', 'emailMatchScore',
    'genderScoreIdentityArticleDiscrepancy', 'grantMatchScore', 'journalSubfieldScore',
    'nameMatchFirstScore', 'nameMatchLastScore', 'nameMatchMiddleScore', 'nameMatchModifierScore',
    'organizationalUnitMatchingScore',
    'scopusNonTargetAuthorInstitutionalAffiliationScore',
    'targetAuthorInstitutionalAffiliationMatchTypeScore',
    'pubmedTargetAuthorInstitutionalAffiliationMatchTypeScore',
    'relationshipPositiveMatchScore', 'relationshipNegativeMatchScore', 'relationshipIdentityCount'
    # NOTE: countAccepted and countRejected excluded - this model is blind to feedback history
]

# Derived identity features shared by both models
DERIVED_FEATURES_IDENTITY_SHARED = [
    'identityStrength',            # Combined strength of identity signals (continuous)
    'netEvidenceCount',            # Positive identity signals minus negative ones
    'ambiguityRisk',               # articleCountScore * (1 - identityStrength) — common name + weak identity
    'nameInstitutionInteraction',  # nameMatchFirst * bestAffiliation — name AND institution agree
    'worstSingleEvidence',         # Min of key identity features — no damning evidence against
    'nameQualityMin',              # Min of first/last/middle name scores — all name parts match
    'firstNameFrequencyScore',     # IDF-like score: rare names → high, common names → low (person-level)
]

# Derived features for Feedback+Identity model (uses feedback counts)
DERIVED_FEATURES_FEEDBACK = [
    'acceptanceRateLowerBound',   # Wilson score interval LB - confidence-adjusted
    'feedbackConfidence',          # How much feedback data we have (log-scaled)
    'uncertainRejectionRisk',      # Continuous risk score for uncertain high-rejection cases
    'feedbackDensity'              # Fraction of 12 feedback features that are non-zero
] + DERIVED_FEATURES_IDENTITY_SHARED

# Derived features for Identity-Only model (no feedback-based features)
DERIVED_FEATURES_IDENTITY_ONLY = list(DERIVED_FEATURES_IDENTITY_SHARED)
    # NOTE: No acceptanceRateLowerBound, feedbackConfidence, uncertainRejectionRisk
    # because those require countAccepted/countRejected

# Complete feature lists
FEEDBACK_IDENTITY_FEATURES = FEEDBACK_IDENTITY_BASE_FEATURES + DERIVED_FEATURES_FEEDBACK
IDENTITY_ONLY_FEATURES = IDENTITY_ONLY_BASE_FEATURES + DERIVED_FEATURES_IDENTITY_ONLY

# Backward compatibility alias
DERIVED_FEATURES = DERIVED_FEATURES_FEEDBACK


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def wilson_lower_bound(successes: float, failures: float, confidence: float = 0.95) -> float:
    """
    Wilson score interval lower bound for success rate.

    Key properties:
    - With no data (0,0): returns 0.5 (maximum uncertainty)
    - With small samples: pulls toward 0.5 (low confidence)
    - With large samples: approaches raw proportion (high confidence)
    - No arbitrary thresholds or cliff effects

    Examples:
        >>> wilson_lower_bound(0, 0)      # No data
        0.5
        >>> wilson_lower_bound(4, 1)      # 80% but small sample
        0.376...  # Pulled toward 0.5
        >>> wilson_lower_bound(100, 25)   # 80% with large sample
        0.717...  # Close to true 0.80

    Args:
        successes: Number of successes (acceptances)
        failures: Number of failures (rejections)
        confidence: Confidence level for interval (default 95%)

    Returns:
        Lower bound of confidence interval for success rate [0, 1]
    """
    n = successes + failures
    if n == 0:
        return 0.5  # No data = maximum uncertainty

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n

    denominator = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)

    lower = (centre - spread) / denominator
    return max(0.0, min(1.0, lower))


# =============================================================================
# DERIVED FEATURE COMPUTATION
# =============================================================================

# Features used for evidence counting and worst-single-evidence
_POSITIVE_EVIDENCE_FEATURES = [
    'nameMatchFirstScore', 'emailMatchScore', 'grantMatchScore',
    'journalSubfieldScore', 'organizationalUnitMatchingScore',
    'targetAuthorInstitutionalAffiliationMatchTypeScore',
    'pubmedTargetAuthorInstitutionalAffiliationMatchTypeScore',
    'scopusNonTargetAuthorInstitutionalAffiliationScore',
    'relationshipPositiveMatchScore',
    'genderScoreIdentityArticleDiscrepancy',
]

_NEGATIVE_EVIDENCE_FEATURES = [
    'nameMatchFirstScore', 'nameMatchMiddleScore', 'nameMatchModifierScore',
    'targetAuthorInstitutionalAffiliationMatchTypeScore',
    'discrepancyDegreeYearScore', 'genderScoreIdentityArticleDiscrepancy',
    'relationshipNegativeMatchScore',
    'articleCountScore',
]

_WORST_EVIDENCE_FEATURES = [
    'nameMatchFirstScore', 'nameMatchMiddleScore',
    'targetAuthorInstitutionalAffiliationMatchTypeScore',
    'discrepancyDegreeYearScore', 'genderScoreIdentityArticleDiscrepancy',
]


def _compute_identity_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 5 engineered identity features shared by both models.

    These features synthesize raw identity signals into higher-level concepts:
    - netEvidenceCount: breadth of supporting vs contradicting evidence
    - ambiguityRisk: common name + weak identity = danger zone
    - nameInstitutionInteraction: name AND institution agree
    - worstSingleEvidence: no single feature is strongly against
    - nameQualityMin: all name parts match (weakest-link)

    Requires 'identityStrength' to already be computed on df.
    """
    # 1. netEvidenceCount: positive signals minus negative signals
    pos_count = sum(
        (df[f] > 0).astype(int) for f in _POSITIVE_EVIDENCE_FEATURES
    )
    neg_count = sum(
        (df[f] < 0).astype(int) for f in _NEGATIVE_EVIDENCE_FEATURES
    )
    df['netEvidenceCount'] = pos_count - neg_count

    # 2. ambiguityRisk: high articleCountScore + low identityStrength
    df['ambiguityRisk'] = df['articleCountScore'] * (1 - df['identityStrength'])

    # 3. nameInstitutionInteraction: name * best affiliation
    best_affil = np.maximum(
        df['targetAuthorInstitutionalAffiliationMatchTypeScore'],
        df['pubmedTargetAuthorInstitutionalAffiliationMatchTypeScore']
    )
    df['nameInstitutionInteraction'] = df['nameMatchFirstScore'] * best_affil

    # 4. worstSingleEvidence: min of key identity features
    df['worstSingleEvidence'] = np.minimum.reduce([
        df[f] for f in _WORST_EVIDENCE_FEATURES
    ])

    # 5. nameQualityMin: weakest name component
    df['nameQualityMin'] = np.minimum.reduce([
        df['nameMatchFirstScore'],
        df['nameMatchLastScore'],
        df['nameMatchMiddleScore']
    ])

    # 6. firstNameFrequencyScore: IDF-like score from name frequency table (person-level)
    #    Requires 'identityFirstName' column (added by Java Feature Generator).
    #    Rare names get high scores (strong identity signal), common names get low scores.
    if _NAME_FREQ_TABLE and 'identityFirstName' in df.columns:
        df['firstNameFrequencyScore'] = df['identityFirstName'].map(_name_frequency_score)
    else:
        df['firstNameFrequencyScore'] = 0.0

    return df


def compute_derived_features_feedback_identity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features for Feedback + Identity model.

    Uses email, ORCID feedback score, and affiliation for identity strength.

    Args:
        df: DataFrame with base features already filled (NaN -> 0)

    Returns:
        DataFrame with derived features added
    """
    df = df.copy()

    # 1. Acceptance Rate Lower Bound (Wilson score)
    df['acceptanceRateLowerBound'] = df.apply(
        lambda row: wilson_lower_bound(
            row['countAccepted'],
            row['countRejected'],
            confidence=0.95
        ), axis=1
    )

    # 2. Feedback Confidence (log-scaled total feedback)
    total_feedback = df['countAccepted'] + df['countRejected']
    df['feedbackConfidence'] = np.log1p(total_feedback)

    # 3. Identity Strength (continuous, 0 to 1)
    # Uses three signals: email, ORCID feedback, affiliation
    email_strength = (df['emailMatchScore'].fillna(0).clip(0, 1) / STRONG_EMAIL).clip(0, 1)
    orcid_strength = (df['feedbackScoreOrcid'].fillna(0).clip(0, 1) / STRONG_ORCID).clip(0, 1)
    affil_strength = np.maximum(
        df['targetAuthorInstitutionalAffiliationMatchTypeScore'].fillna(0).clip(0, 1),
        df['pubmedTargetAuthorInstitutionalAffiliationMatchTypeScore'].fillna(0).clip(0, 1)
    ) / STRONG_AFFIL
    affil_strength = affil_strength.clip(0, 1)

    # Take max of the three signals (any strong signal is good)
    df['identityStrength'] = np.maximum.reduce([email_strength, orcid_strength, affil_strength])

    # Compute shared identity engineered features
    df = _compute_identity_engineered_features(df)

    # 4. Uncertain Rejection Risk (continuous, 0 to 1)
    confidence_factor = (1 - np.exp(-total_feedback / 10))
    df['uncertainRejectionRisk'] = (
        (1 - df['acceptanceRateLowerBound']) *
        (1 - df['identityStrength']) *
        confidence_factor
    )

    # 5. Feedback Density (fraction of 12 feedback features that are non-zero)
    feedback_score_cols = [
        'feedbackScoreCites', 'feedbackScoreCoAuthorName', 'feedbackScoreEmail',
        'feedbackScoreInstitution', 'feedbackScoreJournal', 'feedbackScoreJournalSubField',
        'feedbackScoreKeyword', 'feedbackScoreOrcid', 'feedbackScoreOrcidCoAuthor',
        'feedbackScoreOrganization', 'feedbackScoreTargetAuthorName', 'feedbackScoreYear'
    ]
    df['feedbackDensity'] = (df[feedback_score_cols] != 0).sum(axis=1) / len(feedback_score_cols)

    return df


def compute_derived_features_identity_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features for Identity-Only model.

    This model is BLIND to feedback history (countAccepted, countRejected).
    Only computes identityStrength using email and affiliation signals.

    Args:
        df: DataFrame with base features already filled (NaN -> 0)

    Returns:
        DataFrame with identityStrength added
    """
    df = df.copy()

    # Identity Strength (continuous, 0 to 1)
    # Uses two signals: email, affiliation (no ORCID or feedback data)
    email_strength = (df['emailMatchScore'].fillna(0).clip(0, 1) / STRONG_EMAIL).clip(0, 1)
    affil_strength = np.maximum(
        df['targetAuthorInstitutionalAffiliationMatchTypeScore'].fillna(0).clip(0, 1),
        df['pubmedTargetAuthorInstitutionalAffiliationMatchTypeScore'].fillna(0).clip(0, 1)
    ) / STRONG_AFFIL
    affil_strength = affil_strength.clip(0, 1)

    # Take max of the two signals
    df['identityStrength'] = np.maximum(email_strength, affil_strength)

    # Compute shared identity engineered features
    df = _compute_identity_engineered_features(df)

    # NOTE: No acceptanceRateLowerBound, feedbackConfidence, or uncertainRejectionRisk
    # because this model is blind to feedback history (countAccepted/countRejected)

    return df


# =============================================================================
# PREPROCESSING PIPELINES
# =============================================================================

def preprocess_feedback_identity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline for Feedback + Identity model.

    Args:
        df: Raw DataFrame from database with required columns

    Returns:
        Preprocessed DataFrame ready for model training/inference
    """
    df = df.copy()

    required = set(FEEDBACK_IDENTITY_BASE_FEATURES + ["articleId", "personIdentifier", "pmid", "userAssertion"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Create label from userAssertion
    df["label"] = df["userAssertion"].map({"ACCEPTED": 1, "REJECTED": 0})
    df = df[~df["label"].isna()]

    # Fill missing base features with 0
    df[FEEDBACK_IDENTITY_BASE_FEATURES] = df[FEEDBACK_IDENTITY_BASE_FEATURES].fillna(0)

    # Compute derived features
    df = compute_derived_features_feedback_identity(df)

    return df


def preprocess_identity_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline for Identity-Only model.

    Args:
        df: Raw DataFrame from database with required columns

    Returns:
        Preprocessed DataFrame ready for model training/inference
    """
    df = df.copy()

    required = set(IDENTITY_ONLY_BASE_FEATURES + ["articleId", "personIdentifier", "pmid", "userAssertion"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Create label from userAssertion
    df["label"] = df["userAssertion"].map({"ACCEPTED": 1, "REJECTED": 0})
    df = df[~df["label"].isna()]

    # Fill missing base features with 0
    df[IDENTITY_ONLY_BASE_FEATURES] = df[IDENTITY_ONLY_BASE_FEATURES].fillna(0)

    # Compute derived features
    df = compute_derived_features_identity_only(df)

    return df


# =============================================================================
# INFERENCE PREPROCESSING (no label required)
# =============================================================================

def preprocess_for_inference_feedback_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess for inference (no userAssertion/label required)."""
    df = df.copy()
    df[FEEDBACK_IDENTITY_BASE_FEATURES] = df[FEEDBACK_IDENTITY_BASE_FEATURES].fillna(0)
    df = compute_derived_features_feedback_identity(df)
    return df


def preprocess_for_inference_identity_only(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess for inference (no userAssertion/label required)."""
    df = df.copy()
    df[IDENTITY_ONLY_BASE_FEATURES] = df[IDENTITY_ONLY_BASE_FEATURES].fillna(0)
    df = compute_derived_features_identity_only(df)
    return df
