#!/usr/bin/env python3
"""Cross-tabulate ACCEPTED-article scores by publication year.

The article-year check showed Fred Hutch's gold standard is 98% from 2021+
while the UC gold standards span full careers (back to the 1960s). This script
tests whether that matters: does an old ACCEPTED article actually score
mid-range (because pre-2000 PubMed records carry sparse metadata — initials-only
authors, no affiliations, no ORCID — so ReCiter's identity evidence fires
weakly)?

It pools the UC ACCEPTED article-pairs from the score files, samples them,
fetches each PMID's publication year from PubMed, and reports the score
distribution per year bucket. If the auto-accept share collapses for older
articles, the FH-vs-UC calibration gap is substantially a time-window artifact.

Run from the repo root:
    python3 scripts/score_by_year_check.py
"""
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
ESUMMARY = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'
API_KEY = os.environ.get('PUBMED_API_KEY', '')
random.seed(42)

UC = ['uci', 'ucla', 'usc', 'ucsd', 'ucdavis', 'ucsf']
N_UC = 2400
N_FH = 1200


def load_accepted(scores_file):
    """Return [(pmid, score)] for ACCEPTED articles in a score file."""
    d = json.load(open(scores_file))
    out = []
    for articles in d.values():
        for a in articles:
            if a.get('userAssertion') == 'ACCEPTED' and a.get('score') is not None:
                out.append((a['pmid'], a['score']))
    return out


def fetch_years(pmids):
    years = {}
    pmids = list(pmids)
    for i in range(0, len(pmids), 200):
        batch = pmids[i:i + 200]
        params = {'db': 'pubmed', 'id': ','.join(map(str, batch)), 'retmode': 'json'}
        if API_KEY:
            params['api_key'] = API_KEY
        url = ESUMMARY + '?' + urllib.parse.urlencode(params)
        data = {'result': {}}
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = json.loads(resp.read())
                break
            except Exception:
                time.sleep(2)
        for pmid in batch:
            e = data.get('result', {}).get(str(pmid))
            if not e:
                continue
            m = re.search(r'(19|20)\d\d', e.get('sortpubdate') or e.get('pubdate') or '')
            if m:
                years[pmid] = int(m.group())
        print(f'  fetched {min(i+200, len(pmids))}/{len(pmids)}', end='\r')
        time.sleep(0.12 if API_KEY else 0.34)
    print()
    return years


BUCKETS = [('2021+', 2021, 2100), ('2015-2020', 2015, 2020), ('2010-2014', 2010, 2014),
           ('2000-2009', 2000, 2009), ('1990-1999', 1990, 1999), ('<1990', 0, 1989)]


def tabulate(label, pairs, years):
    print(f'\n=== {label} — ACCEPTED-article score distribution by publication year ===')
    print(f'{"year bucket":>12} | {"n":>6} | {"auto-accept ≥95":>16} {"review 10-95":>13} '
          f'{"auto-rej ≤10":>13} | {"mean score":>10}')
    print('-' * 80)
    rows = {}
    for name, lo, hi in BUCKETS:
        sub = [s for p, s in pairs if p in years and lo <= years[p] <= hi]
        if not sub:
            continue
        n = len(sub)
        hi_pct = 100 * sum(1 for s in sub if s >= 95) / n
        mid_pct = 100 * sum(1 for s in sub if 10 < s < 95) / n
        lo_pct = 100 * sum(1 for s in sub if s <= 10) / n
        mean = sum(sub) / n
        rows[name] = {'n': n, 'auto_accept_pct': hi_pct, 'review_pct': mid_pct,
                      'auto_reject_pct': lo_pct, 'mean_score': mean}
        print(f'{name:>12} | {n:>6} | {hi_pct:>15.1f}% {mid_pct:>12.1f}% '
              f'{lo_pct:>12.1f}% | {mean:>10.1f}')
    return rows


def main():
    uc_pairs = []
    for inst in UC:
        uc_pairs += load_accepted(ROOT / f'external_validation/results/{inst}/{inst}_scores.json')
    fh_pairs = load_accepted(ROOT / 'external_validation/results/fredhutch/fredhutch_scores.json')
    print(f'UC ACCEPTED pairs: {len(uc_pairs)} | Fred Hutch ACCEPTED pairs: {len(fh_pairs)}')

    random.shuffle(uc_pairs)
    random.shuffle(fh_pairs)
    uc_s, fh_s = uc_pairs[:N_UC], fh_pairs[:N_FH]
    pmids = {p for p, _ in uc_s} | {p for p, _ in fh_s}
    print(f'Fetching years for {len(pmids)} PMIDs...')
    years = fetch_years(pmids)
    print(f'  got {len(years)}/{len(pmids)}')

    uc_rows = tabulate('UC SYSTEM (pooled, 6 institutions)', uc_s, years)
    fh_rows = tabulate('FRED HUTCH', fh_s, years)

    out = ROOT / 'external_validation' / 'uc_system' / 'score_by_year_check.json'
    json.dump({'n_uc': N_UC, 'n_fh': N_FH, 'uc': uc_rows, 'fred_hutch': fh_rows},
              open(out, 'w'), indent=2)
    print(f'\nWritten: {out}')


if __name__ == '__main__':
    main()
