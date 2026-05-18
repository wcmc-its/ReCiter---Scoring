#!/usr/bin/env python3
"""
NEEDS REVIEWED fold-in — completing the recency-matched comparison.

Fred Hutch curators marked 965 articles NEEDS REVIEWED and dropped them before
evaluation; the UC gold standards kept every uncertain article. Fred Hutch's
review burden (3.2%) is therefore measured on its easier articles only. This
script scores the 965 excluded articles with the deployed model (they survive in
the FH feature vectors as userAssertion=PENDING) and folds them back into the
Fred Hutch baseline — the fully matched cross-institution comparison.

Scope: review burden only. NEEDS REVIEWED articles have no ACCEPTED/REJECTED
label, so they cannot enter ECE or AUC. Review burden = score in (10, 95),
matching scripts/run_external_validation.py.
"""
import sys
import os
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from run_external_validation import parse_data_file          # noqa: E402
from theory_identity_rejects_heatmap import load_models, score_file  # noqa: E402
from recency_matched_evaluation import fetch_years            # noqa: E402

FV_DIR = os.path.expanduser(os.environ.get(
    'RECITER_FV_DIR', '~/Dropbox/GitHub/ReCiter/src/main/resources/scripts'))
FH_XLSX = ROOT / 'external_validation/Fred Hutch - Reciter_data_20210101-20240630.xlsx'
FH_SCORES = ROOT / 'external_validation/results/fredhutch/fredhutch_scores.json'
OUT = ROOT / 'external_validation/uc_system/needs_reviewed_foldin.json'
CUTOFF = 2021
UC_POOLED_2021PLUS_REVIEW = 6.15   # from recency_matched_evaluation.json


def review_band(s):
    s = np.asarray(s, float)
    return (s > 10) & (s < 95)


def burden(scores, years=None):
    """Return (review_band_count, n, review_pct) over full and 2021+ sets."""
    scores = list(scores)
    s = np.asarray([sc for sc, _ in scores], float)
    full = {'n': len(s), 'review_n': int(review_band(s).sum())}
    full['review_pct'] = round(100 * full['review_n'] / full['n'], 2) if full['n'] else None
    s21 = np.asarray([sc for sc, pm in scores if (years.get(pm) or 0) >= CUTOFF], float)
    rec = {'n': len(s21), 'review_n': int(review_band(s21).sum())}
    rec['review_pct'] = round(100 * rec['review_n'] / rec['n'], 2) if rec['n'] else None
    return full, rec


def main():
    # 1. NEEDS REVIEWED (uid, pmid) pairs from the FH gold standard
    _, rows = parse_data_file(str(FH_XLSX))
    nr = {}
    for r in rows:
        if str(r.get('Assertion Status')).strip() == 'NEEDS REVIEWED':
            try:
                uid = str(r.get('Uid')).strip()
                nr.setdefault(uid, set()).add(int(float(str(r.get('Pmid')))))
            except (ValueError, TypeError):
                pass
    n_nr = sum(len(v) for v in nr.values())
    print(f'NEEDS REVIEWED: {n_nr} pairs across {len(nr)} researchers')

    # 2. score them from the FH feature vectors with the deployed model
    print('Scoring NEEDS REVIEWED articles ...')
    m = load_models()
    nr_scores = {}          # (uid, pmid) -> deployed final score
    missing_fv, missing_art = 0, 0
    for uid, pmids in nr.items():
        fp = f'{FV_DIR}/fr_{uid}-feedbackIdentityScoringInput.json'
        if not os.path.exists(fp):
            missing_fv += len(pmids)
            continue
        df = score_file(fp, m)
        if df is None:
            missing_art += len(pmids)
            continue
        found = set()
        hit = df[df.articleId.isin(pmids)]
        for _, row in hit.iterrows():
            nr_scores[(uid, int(row.articleId))] = float(row.score_fb)
            found.add(int(row.articleId))
        missing_art += len(pmids - found)
    print(f'  scored {len(nr_scores)}/{n_nr} '
          f'({100*len(nr_scores)/n_nr:.1f}% coverage; '
          f'{missing_fv} no feature-vector file, {missing_art} pmid not in vector)')

    # 3. publication years for the NEEDS REVIEWED pmids
    nr_years = fetch_years({p for _, p in nr_scores})

    # 4. Fred Hutch evaluated set (ACCEPTED + REJECTED) + its years
    fh = json.load(open(FH_SCORES))
    eval_scores = []        # (score, pmid)
    for uid, arts in fh.items():
        for a in arts:
            eval_scores.append((float(a['score']), int(a['pmid'])))
    eval_years = fetch_years({p for _, p in eval_scores})

    # 5. burdens
    nr_pairs = [(sc, pm) for (uid, pm), sc in nr_scores.items()]
    nr_full, nr_rec = burden(nr_pairs, nr_years)
    ev_full, ev_rec = burden(eval_scores, eval_years)

    # where the model puts the curator-uncertain articles
    s_nr = np.asarray([sc for sc, _ in nr_pairs], float)
    nr_dist = {
        'auto_accept_pct': round(100 * np.mean(s_nr >= 95), 1),
        'review_pct': round(100 * np.mean(review_band(s_nr)), 1),
        'auto_reject_pct': round(100 * np.mean(s_nr <= 10), 1),
    }

    # folded-in Fred Hutch review burden
    fold_full = 100 * (ev_full['review_n'] + nr_full['review_n']) / \
        (ev_full['n'] + nr_full['n'])
    fold_rec = 100 * (ev_rec['review_n'] + nr_rec['review_n']) / \
        (ev_rec['n'] + nr_rec['n'])

    print(f'\n=== Where the deployed model places the 965 NEEDS REVIEWED articles ===')
    print(f'  auto-accept (>=95): {nr_dist["auto_accept_pct"]}%   '
          f'review band: {nr_dist["review_pct"]}%   '
          f'auto-reject (<=10): {nr_dist["auto_reject_pct"]}%')

    print(f'\n=== Fred Hutch review burden — before vs after folding in NEEDS REVIEWED ===')
    print(f'{"":>16} | {"evaluated only":>16} | {"+ NEEDS REVIEWED":>18}')
    print(f'{"full set":>16} | {ev_full["review_pct"]:>14.2f}%  | '
          f'{fold_full:>16.2f}%  (n {ev_full["n"]:,}+{nr_full["n"]:,})')
    print(f'{"2021+ only":>16} | {ev_rec["review_pct"]:>14.2f}%  | '
          f'{fold_rec:>16.2f}%  (n {ev_rec["n"]:,}+{nr_rec["n"]:,})')

    print(f'\n=== Recency-matched, fully matched residual ===')
    print(f'  UC pooled 2021+       : {UC_POOLED_2021PLUS_REVIEW:.2f}%')
    print(f'  Fred Hutch 2021+ (raw): {ev_rec["review_pct"]:.2f}%   '
          f'-> residual {UC_POOLED_2021PLUS_REVIEW - ev_rec["review_pct"]:+.2f} pp')
    print(f'  Fred Hutch 2021+ (+NR): {fold_rec:.2f}%   '
          f'-> residual {UC_POOLED_2021PLUS_REVIEW - fold_rec:+.2f} pp')

    payload = {
        'needs_reviewed_total': n_nr,
        'needs_reviewed_scored': len(nr_scores),
        'coverage_pct': round(100 * len(nr_scores) / n_nr, 1),
        'needs_reviewed_score_distribution': nr_dist,
        'fred_hutch': {
            'evaluated_only': {'full': ev_full, 'recency_2021plus': ev_rec},
            'with_needs_reviewed': {
                'full_review_pct': round(fold_full, 2),
                'recency_2021plus_review_pct': round(fold_rec, 2),
            },
        },
        'residual_vs_uc_pooled_2021plus': {
            'uc_pooled_2021plus_review_pct': UC_POOLED_2021PLUS_REVIEW,
            'fred_hutch_2021plus_evaluated_only': ev_rec['review_pct'],
            'fred_hutch_2021plus_with_needs_reviewed': round(fold_rec, 2),
            'residual_pp_evaluated_only': round(
                UC_POOLED_2021PLUS_REVIEW - ev_rec['review_pct'], 2),
            'residual_pp_with_needs_reviewed': round(
                UC_POOLED_2021PLUS_REVIEW - fold_rec, 2),
        },
    }
    with open(OUT, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\n  json -> {OUT}')


if __name__ == '__main__':
    main()
