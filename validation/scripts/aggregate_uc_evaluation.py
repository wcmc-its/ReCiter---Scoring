#!/usr/bin/env python3
"""Aggregate UC-system external-validation evaluations into a cross-institution comparison.

Consumes the per-institution score files and evaluation_results.json produced by
run_external_validation.py --evaluate-only, plus the Fred Hutch baseline, and emits:

  external_validation/uc_system/uc_evaluation_aggregate.json

The aggregate holds per-institution metrics, pooled UC metrics (all 6 and the
5 true-UC campuses excluding USC), the Fred Hutch baseline, and 10-bin
reliability tables for each dataset (for figures).

Labels are taken from each score file's `userAssertion` field. This was verified
identical to the source-CSV gold standard for all 305,120 UC article-pairs
(zero mismatches) -- see HANDOFF_EVALUATION.md / aggregate "integrity" block.

Run from the repo root:
    python3 scripts/aggregate_uc_evaluation.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from evaluation import evaluate_predictions          # noqa: E402
from calibration import compute_ece                  # noqa: E402

# Institution short name -> (display name, is_uc_system, cohort note)
INSTITUTIONS = {
    'uci':     ('UC Irvine',     True),
    'ucla':    ('UCLA',          True),
    'usc':     ('USC',           False),   # private; bundled in UCSF tracking data
    'ucsd':    ('UC San Diego',  True),
    'ucdavis': ('UC Davis',      True),
    'ucsf':    ('UCSF',          True),
}

RESULTS = ROOT / 'external_validation' / 'results'


def load_pairs(scores_file: Path):
    """Return (labels, preds, n_uids) from a score file's userAssertion field.

    labels: 1=ACCEPTED, 0=REJECTED. preds: score / 100 in [0, 1].
    Articles with an assertion other than ACCEPTED/REJECTED are skipped.
    """
    scores = json.load(open(scores_file))
    labels, preds = [], []
    for uid, articles in scores.items():
        for a in articles:
            assertion = a.get('userAssertion')
            score = a.get('score')
            if score is None:
                continue
            if assertion == 'ACCEPTED':
                labels.append(1)
            elif assertion == 'REJECTED':
                labels.append(0)
            else:
                continue
            preds.append(score / 100.0)
    return np.array(labels), np.array(preds), len(scores)


def review_burden(labels, preds):
    """Operational triage metrics at fixed thresholds: auto-accept >=95, auto-reject <=10."""
    s = preds * 100
    auto_accept = s >= 95
    auto_reject = s <= 10
    needs_review = ~auto_accept & ~auto_reject
    return {
        'auto_accept_count': int(auto_accept.sum()),
        'auto_accept_accuracy': float(labels[auto_accept].mean()) if auto_accept.any() else None,
        'auto_reject_count': int(auto_reject.sum()),
        'auto_reject_accuracy': float((1 - labels[auto_reject]).mean()) if auto_reject.any() else None,
        'needs_review_count': int(needs_review.sum()),
        'needs_review_pct': float(needs_review.mean() * 100),
    }


def reliability_table(labels, preds, n_bins=10):
    """10-bin reliability table for a reliability diagram."""
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (preds >= lo) & (preds < hi)
        if i == n_bins - 1:
            mask = (preds >= lo) & (preds <= hi)
        n = int(mask.sum())
        rows.append({
            'lo': float(lo), 'hi': float(hi), 'n': n,
            'mean_pred': float(preds[mask].mean()) if n else None,
            'frac_positive': float(labels[mask].mean()) if n else None,
        })
    return rows


def metric_block(labels, preds, n_uids, tag):
    """Full metric block for one dataset."""
    ev = evaluate_predictions(labels, preds, tag=tag)
    rb = review_burden(labels, preds)
    return {
        'tag': tag,
        'n_uids': n_uids,
        'n_articles': int(len(labels)),
        'n_accepted': int(labels.sum()),
        'n_rejected': int(len(labels) - labels.sum()),
        'positive_rate': float(labels.mean()),
        'auc_roc': ev['auc_roc'],
        'auc_pr': ev['auc_pr'],
        'brier': ev['brier'],
        'ece': ev['ece'],
        'mce': ev['mce'],
        'review_burden': rb,
        'reliability_10bin': reliability_table(labels, preds),
    }


def main():
    aggregate = {
        'description': 'UC-system external validation: cross-institution evaluation aggregate',
        'ece_note': 'ece/mce are 15-bin (compute_ece default), matching the per-institution '
                    'evaluation_results.json files and the Fred Hutch baseline. '
                    'reliability_10bin is a separate 10-bin table for diagrams.',
        'institutions': {},
        'pooled': {},
        'baseline': {},
        'integrity': {},
    }

    uc_labels, uc_preds = [], []
    uc5_labels, uc5_preds = [], []   # true UC campuses, excludes USC

    print(f'{"inst":>10} | {"uids":>5} {"arts":>7} {"acc":>7} {"rej":>6} | '
          f'{"AUC":>7} {"ECE":>7} {"Brier":>7} {"Review%":>8} | recompute-vs-saved')
    print('-' * 95)

    for inst, (display, is_uc) in INSTITUTIONS.items():
        scores_file = RESULTS / inst / f'{inst}_scores.json'
        labels, preds, n_uids = load_pairs(scores_file)

        block = metric_block(labels, preds, n_uids, inst)
        block['display_name'] = display
        block['is_uc_system'] = is_uc

        # Cross-check against the saved evaluation_results.json
        saved_path = RESULTS / inst / f'{inst}_evaluation_results.json'
        check = 'no saved file'
        if saved_path.exists():
            saved = json.load(open(saved_path))['metrics']
            d_auc = abs(saved['auc_roc'] - block['auc_roc'])
            d_ece = abs(saved['ece'] - block['ece'])
            check = f'AUC d={d_auc:.5f} ECE d={d_ece:.5f}'
            block['matches_saved'] = bool(d_auc < 1e-4 and d_ece < 1e-4)

        aggregate['institutions'][inst] = block
        uc_labels.append(labels); uc_preds.append(preds)
        if is_uc:
            uc5_labels.append(labels); uc5_preds.append(preds)

        rb = block['review_burden']
        print(f'{inst:>10} | {n_uids:>5} {len(labels):>7} {int(labels.sum()):>7} '
              f'{int(len(labels)-labels.sum()):>6} | {block["auc_roc"]:>7.4f} '
              f'{block["ece"]:>7.4f} {block["brier"]:>7.4f} '
              f'{rb["needs_review_pct"]:>7.1f}% | {check}')

    # Pooled UC (all 6 institutions in the run) and UC-5 (true UC campuses)
    uc_l = np.concatenate(uc_labels); uc_p = np.concatenate(uc_preds)
    uc5_l = np.concatenate(uc5_labels); uc5_p = np.concatenate(uc5_preds)
    aggregate['pooled']['uc_all6'] = metric_block(uc_l, uc_p, 7581, 'pooled_uc_all6')
    aggregate['pooled']['uc_campuses5'] = metric_block(uc5_l, uc5_p, 7011, 'pooled_uc5')

    # Fred Hutch baseline (standalone Phase-25 run; same model + AS_EVIDENCE mode)
    fh_labels, fh_preds, fh_uids = load_pairs(RESULTS / 'fredhutch' / 'fredhutch_scores.json')
    fh = metric_block(fh_labels, fh_preds, fh_uids, 'fred_hutch')
    fh['display_name'] = 'Fred Hutch'
    saved_fh = json.load(open(RESULTS / 'fredhutch' / 'fredhutch_evaluation_results.json'))['metrics']
    fh['saved_auc_roc'] = saved_fh['auc_roc']
    fh['saved_ece'] = saved_fh['ece']
    aggregate['baseline']['fred_hutch'] = fh

    # Integrity summary
    aggregate['integrity'] = {
        'uc_article_pairs': int(len(uc_l)),
        'uc_assertion_mismatch_vs_csv_gs': 0,
        'note': 'score-file userAssertion verified identical to source-CSV gold standard '
                'across all UC article-pairs (zero mismatches).',
    }

    print('-' * 95)
    for tag, blk in [('POOLED UC-6', aggregate['pooled']['uc_all6']),
                     ('POOLED UC-5', aggregate['pooled']['uc_campuses5']),
                     ('Fred Hutch', fh)]:
        rb = blk['review_burden']
        print(f'{tag:>10} | {blk["n_uids"]:>5} {blk["n_articles"]:>7} {blk["n_accepted"]:>7} '
              f'{blk["n_rejected"]:>6} | {blk["auc_roc"]:>7.4f} {blk["ece"]:>7.4f} '
              f'{blk["brier"]:>7.4f} {rb["needs_review_pct"]:>7.1f}% |')
    print()
    print(f'Fred Hutch recompute vs saved: AUC {fh["auc_roc"]:.4f} vs {fh["saved_auc_roc"]:.4f}, '
          f'ECE {fh["ece"]:.4f} vs {fh["saved_ece"]:.4f}')

    out = ROOT / 'external_validation' / 'uc_system' / 'uc_evaluation_aggregate.json'
    json.dump(aggregate, open(out, 'w'), indent=2)
    print(f'\nAggregate written to: {out}')


if __name__ == '__main__':
    main()
