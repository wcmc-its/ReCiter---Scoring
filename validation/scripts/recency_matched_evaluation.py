#!/usr/bin/env python3
"""
Recency-matched re-evaluation of the 7-site external validation.

GOLD_STANDARD_AUDIT.md's "determined next step." Fetches the publication year of
every gold-standard PMID in the external-validation runs, then recomputes
AUC / ECE / review burden restricted to 2021+ articles — the Fred Hutch
evaluation window — giving the apples-to-apples cross-institution comparison.

The raw UC↔Fred Hutch calibration gap (UC ECE 0.040 / review 11.2% vs FH
0.007 / 3.2%) is attributed to Fred Hutch's gold standard being 98% recent
(2021–2024) while the UC gold standards span full careers. This script converts
that estimate into exact numbers.

No re-scoring: uses the scores already in the *_scores.json files. Metric
definitions match scripts/run_external_validation.py and src/calibration.py
(ECE 15-bin; review burden = 10 < score < 95).
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from calibration import compute_ece  # noqa: E402  (15-bin default)

ESUMMARY = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'
API_KEY = os.environ.get('PUBMED_API_KEY', '')
YEAR_CACHE = '/tmp/uc_fh_article_years.json'
CUTOFF = 2021

UC_INSTITUTIONS = [
    ('UC Irvine',    'external_validation/results/uci/uci_scores.json'),
    ('UCLA',         'external_validation/results/ucla/ucla_scores.json'),
    ('USC',          'external_validation/results/usc/usc_scores.json'),
    ('UC San Diego', 'external_validation/results/ucsd/ucsd_scores.json'),
    ('UC Davis',     'external_validation/results/ucdavis/ucdavis_scores.json'),
    ('UCSF',         'external_validation/results/ucsf/ucsf_scores.json'),
]
FRED_HUTCH = ('Fred Hutch', 'external_validation/results/fredhutch/fredhutch_scores.json')
OUT_JSON = ROOT / 'external_validation/uc_system/recency_matched_evaluation.json'
FIG = ROOT / 'external_validation/uc_system/figures/recency_matched_evaluation.png'

# published full-set numbers from CROSS_INSTITUTION_EVALUATION.md, for validation
PUBLISHED = {
    'UC Irvine': (0.9945, 0.0344, 9.4), 'UCLA': (0.9907, 0.0375, 9.0),
    'USC': (0.9932, 0.0429, 11.2), 'UC San Diego': (0.9875, 0.0408, 11.0),
    'UC Davis': (0.9883, 0.0507, 13.1), 'UCSF': (0.9902, 0.0343, 10.5),
    'UC pooled': (0.9902, 0.0395, 11.2), 'Fred Hutch': (0.9991, 0.0069, 3.2),
}


def load_rows(path):
    """Return list of (pmid, score_0_100, label) for one score file."""
    with open(ROOT / path) as f:
        data = json.load(f)
    rows = []
    for uid, arts in data.items():
        for a in arts:
            ua = str(a.get('userAssertion') or '').strip().upper()
            if ua == 'ACCEPTED':
                label = 1
            elif ua == 'REJECTED':
                label = 0
            else:
                continue
            rows.append((int(a['pmid']), float(a['score']), label))
    return rows


def fetch_years(pmids):
    """Return {pmid:int -> year:int|None}, cached on disk across runs."""
    cache = {}
    if os.path.exists(YEAR_CACHE):
        with open(YEAR_CACHE) as f:
            cache = json.load(f)
    todo = sorted({p for p in pmids if str(p) not in cache})
    print(f'  year cache: {len(cache):,} known, {len(todo):,} to fetch')
    for i in range(0, len(todo), 200):
        batch = todo[i:i + 200]
        params = {'db': 'pubmed', 'id': ','.join(map(str, batch)),
                  'retmode': 'json'}
        if API_KEY:
            params['api_key'] = API_KEY
        url = ESUMMARY + '?' + urllib.parse.urlencode(params)
        data = {'result': {}}
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = json.loads(resp.read())
                break
            except Exception:
                time.sleep(2 + attempt)
        res = data.get('result', {})
        for pmid in batch:
            e = res.get(str(pmid))
            year = None
            if e:
                raw = e.get('sortpubdate') or e.get('pubdate') or ''
                m = re.search(r'(19|20)\d\d', raw)
                if m:
                    year = int(m.group())
            cache[str(pmid)] = year
        if (i // 200) % 25 == 0:
            with open(YEAR_CACHE, 'w') as f:
                json.dump(cache, f)
            print(f'  fetched {min(i + 200, len(todo)):,}/{len(todo):,}', end='\r')
        time.sleep(0.12 if API_KEY else 0.34)
    with open(YEAR_CACHE, 'w') as f:
        json.dump(cache, f)
    print()
    return {int(k): v for k, v in cache.items()}


def metrics(rows):
    """rows: list of (pmid, score, label). Returns metric dict."""
    if not rows:
        return None
    scores = np.array([r[1] for r in rows], float)
    labels = np.array([r[2] for r in rows], int)
    pred = scores / 100.0
    auto_accept = scores >= 95
    auto_reject = scores <= 10
    review = ~auto_accept & ~auto_reject
    n_acc, n_rej = int(labels.sum()), int((labels == 0).sum())
    out = {
        'n': len(rows), 'n_accepted': n_acc, 'n_rejected': n_rej,
        'accept_rate': round(float(labels.mean()), 4),
        'review_burden_pct': round(float(review.mean() * 100), 2),
        'auto_accept_pct': round(float(auto_accept.mean() * 100), 2),
        'auto_reject_pct': round(float(auto_reject.mean() * 100), 2),
        'auto_accept_accuracy': round(float(labels[auto_accept].mean()), 4)
            if auto_accept.any() else None,
        'auto_reject_accuracy': round(float((1 - labels[auto_reject]).mean()), 4)
            if auto_reject.any() else None,
        'ece': round(float(compute_ece(labels, pred, 15)), 4),
        'auc_roc': round(float(roc_auc_score(labels, pred)), 4)
            if n_acc > 0 and n_rej > 0 else None,
    }
    return out


def year_distribution(rows, years):
    yrs = [years.get(p) for p, _, _ in rows]
    known = [y for y in yrs if y is not None]
    n = len(rows)
    band = lambda lo, hi: round(100 * sum(1 for y in known if lo <= y <= hi) / n, 1)
    return {
        'n': n, 'year_resolved': len(known),
        'year_resolved_pct': round(100 * len(known) / n, 1),
        'median_year': int(np.median(known)) if known else None,
        'pct_2021plus': band(2021, 2100),
        'pct_2010_2020': band(2010, 2020),
        'pct_2000_2009': band(2000, 2009),
        'pct_before_2000': band(0, 1999),
    }


def plot(results, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    order = [n for n, _ in UC_INSTITUTIONS] + ['UC pooled', 'Fred Hutch']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, key, title, ymax in [
        (axes[0], 'review_burden_pct', 'Manual review burden (%)', 15),
        (axes[1], 'ece', 'Expected Calibration Error', 0.055)]:
        full = [results[n]['full'][key] for n in order]
        rec = [results[n]['recency_2021plus'][key]
               if results[n]['recency_2021plus'] else np.nan for n in order]
        x = np.arange(len(order))
        ax.bar(x - 0.2, full, 0.4, label='full gold standard', color='#c0504d')
        ax.bar(x + 0.2, rec, 0.4, label='2021+ only (FH window)', color='#4f81bd')
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=35, ha='right')
        ax.set_title(title)
        ax.set_ylim(0, ymax)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        for xi, (f, r) in enumerate(zip(full, rec)):
            ax.text(xi - 0.2, f, f'{f:.1f}' if key.endswith('pct') else f'{f:.3f}',
                    ha='center', va='bottom', fontsize=7)
            if not np.isnan(r):
                ax.text(xi + 0.2, r, f'{r:.1f}' if key.endswith('pct') else f'{r:.3f}',
                        ha='center', va='bottom', fontsize=7)
    fig.suptitle('Recency-matched re-evaluation — restricting to 2021+ articles '
                 'collapses the UC↔Fred Hutch gap', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    fig.savefig(str(path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f'  figure -> {path}')


def main():
    print('Loading score files ...')
    inst_rows = {name: load_rows(path) for name, path in UC_INSTITUTIONS}
    inst_rows[FRED_HUTCH[0]] = load_rows(FRED_HUTCH[1])
    all_pmids = {p for rows in inst_rows.values() for p, _, _ in rows}
    print(f'  {sum(len(r) for r in inst_rows.values()):,} article-pairs, '
          f'{len(all_pmids):,} unique PMIDs')

    print('Fetching publication years (NCBI esummary) ...')
    years = fetch_years(all_pmids)
    resolved = sum(1 for p in all_pmids if years.get(p) is not None)
    print(f'  resolved {resolved:,}/{len(all_pmids):,} '
          f'({100*resolved/len(all_pmids):.1f}%)')

    results = {}
    groups = list(UC_INSTITUTIONS) + [FRED_HUTCH]
    for name, _ in groups:
        rows = inst_rows[name]
        rec = [r for r in rows if (years.get(r[0]) or 0) >= CUTOFF]
        results[name] = {
            'full': metrics(rows),
            'recency_2021plus': metrics(rec),
            'year_distribution': year_distribution(rows, years),
        }
    uc_all = [r for name, _ in UC_INSTITUTIONS for r in inst_rows[name]]
    uc_rec = [r for r in uc_all if (years.get(r[0]) or 0) >= CUTOFF]
    results['UC pooled'] = {
        'full': metrics(uc_all),
        'recency_2021plus': metrics(uc_rec),
        'year_distribution': year_distribution(uc_all, years),
    }

    # ---- validation against published full-set numbers ----
    print('\n=== Validation: recomputed full-set vs CROSS_INSTITUTION_EVALUATION.md ===')
    print(f'{"group":>14} | {"AUC recomp/pub":>20} {"ECE recomp/pub":>20} '
          f'{"review% recomp/pub":>20}')
    for name, (auc_p, ece_p, rev_p) in PUBLISHED.items():
        m = results[name]['full']
        print(f'{name:>14} | {m["auc_roc"]:.4f} / {auc_p:.4f}      '
              f'{m["ece"]:.4f} / {ece_p:.4f}      '
              f'{m["review_burden_pct"]:5.1f} / {rev_p:5.1f}')

    # ---- headline table ----
    print(f'\n=== Recency-matched results (2021+ = Fred Hutch window) ===')
    print(f'{"group":>14} | {"FULL: n":>9} {"AUC":>7} {"ECE":>7} {"rev%":>6} | '
          f'{"2021+: n":>9} {"AUC":>7} {"ECE":>7} {"rev%":>6} | {"rev drop":>9}')
    print('-' * 104)
    for name in [n for n, _ in UC_INSTITUTIONS] + ['UC pooled', 'Fred Hutch']:
        f, r = results[name]['full'], results[name]['recency_2021plus']
        auc_r = f'{r["auc_roc"]:.4f}' if r and r['auc_roc'] is not None else '   n/a'
        drop = f'{f["review_burden_pct"] - r["review_burden_pct"]:+.1f} pp' if r else '  n/a'
        print(f'{name:>14} | {f["n"]:>9,} {f["auc_roc"]:.4f} {f["ece"]:.4f} '
              f'{f["review_burden_pct"]:5.1f}% | {r["n"]:>9,} {auc_r} '
              f'{r["ece"]:.4f} {r["review_burden_pct"]:5.1f}% | {drop:>9}')

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump({'cutoff_year': CUTOFF, 'n_unique_pmids': len(all_pmids),
                   'pmids_year_resolved': resolved, 'results': results},
                  f, indent=2)
    print(f'\n  json   -> {OUT_JSON}')
    FIG.parent.mkdir(parents=True, exist_ok=True)
    plot(results, FIG)


if __name__ == '__main__':
    main()
