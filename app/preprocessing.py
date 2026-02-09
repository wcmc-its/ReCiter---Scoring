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

# Derived features for Feedback+Identity model (uses feedback counts)
DERIVED_FEATURES_FEEDBACK = [
    'acceptanceRateLowerBound',   # Wilson score interval LB - confidence-adjusted
    'feedbackConfidence',          # How much feedback data we have (log-scaled)
    'identityStrength',            # Combined strength of identity signals (continuous)
    'uncertainRejectionRisk'       # Continuous risk score for uncertain high-rejection cases
]

# Derived features for Identity-Only model (no feedback-based features)
DERIVED_FEATURES_IDENTITY_ONLY = [
    'identityStrength'             # Combined strength of identity signals (continuous)
    # NOTE: No acceptanceRateLowerBound, feedbackConfidence, uncertainRejectionRisk
    # because those require countAccepted/countRejected
]

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

    # 4. Uncertain Rejection Risk (continuous, 0 to 1)
    confidence_factor = (1 - np.exp(-total_feedback / 10))
    df['uncertainRejectionRisk'] = (
        (1 - df['acceptanceRateLowerBound']) *
        (1 - df['identityStrength']) *
        confidence_factor
    )

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
