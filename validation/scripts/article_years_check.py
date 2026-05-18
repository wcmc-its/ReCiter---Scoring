#!/usr/bin/env python3
"""Compare the publication-year distribution of each gold standard, via PubMed.

The name-collision check surfaced that UC Davis ACCEPTED articles are 41%
initials-only (vs Fred Hutch 2.8%) — a strong signal the UC gold standard spans
much older articles. Fred Hutch's data file is an explicit recent-window extract
(filename: ...20210101-20240630...). This script confirms the time-window
mismatch directly: it samples ACCEPTED articles per institution, fetches the
publication year from PubMed (esummary), and reports the distribution.

A time-window mismatch matters: pre-2000 articles carry sparse metadata
(initials-only authors, no affiliations, no ORCID), so ReCiter's identity
features fire weakly and such articles score mid-range — inflating review burden
and ECE for a gold standard that includes them.

Run from the repo root:
    python3 scripts/article_years_check.py
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
sys.path.insert(0, str(ROOT / 'scripts'))
from run_external_validation import parse_data_file, detect_column_mapping  # noqa: E402

random.seed(42)
ESUMMARY = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'
API_KEY = os.environ.get('PUBMED_API_KEY', '')
N = 350  # ACCEPTED articles sampled per institution

DATASETS = {
    'UC Irvine':    'external_validation/uc_system/data/uci_data.csv',
    'UCLA':         'external_validation/uc_system/data/ucla_data.csv',
    'USC':          'external_validation/uc_system/data/usc_data.csv',
    'UC San Diego': 'external_validation/uc_system/data/ucsd_data.csv',
    'UC Davis':     'external_validation/uc_system/data/ucdavis_data.csv',
    'UCSF':         'external_validation/uc_system/data/ucsf_data.csv',
    'Fred Hutch':   'external_validation/Fred Hutch - Reciter_data_20210101-20240630.xlsx',
}


def load_accepted_pmids(path):
    headers, rows = parse_data_file(str(ROOT / path))
    mapping, _ = detect_column_mapping(headers)
    pmid_col = assert_col = None
    for h, (f, s) in mapping.items():
        if f == '_gold_standard' and s == 'pmid':
            pmid_col = h
        elif f == '_gold_standard' and s == 'assertion':
            assert_col = h
    out = []
    for r in rows:
        if str(r.get(assert_col) or '').strip().upper() != 'ACCEPTED':
            continue
        try:
            out.append(int(float(str(r.get(pmid_col)))))
        except (ValueError, TypeError):
            continue
    return out


def fetch_years(pmids):
    """Return {pmid: year} via NCBI esummary."""
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
        res = data.get('result', {})
        for pmid in batch:
            e = res.get(str(pmid))
            if not e:
                continue
            raw = e.get('sortpubdate') or e.get('pubdate') or ''
            m = re.search(r'(19|20)\d\d', raw)
            if m:
                years[pmid] = int(m.group())
        print(f'  fetched {min(i+200, len(pmids))}/{len(pmids)}', end='\r')
        time.sleep(0.12 if API_KEY else 0.34)
    print()
    return years


def pct(vals, predicate):
    return 100 * sum(1 for v in vals if predicate(v)) / len(vals) if vals else 0


def main():
    samples = {}
    for name, path in DATASETS.items():
        acc = load_accepted_pmids(path)
        random.shuffle(acc)
        samples[name] = acc[:N]
        print(f'  {name}: {len(acc)} ACCEPTED rows, sampled {len(samples[name])}')

    all_pmids = {p for s in samples.values() for p in s}
    print(f'\nFetching publication years for {len(all_pmids)} PMIDs...')
    years = fetch_years(all_pmids)
    print(f'  got years for {len(years)}/{len(all_pmids)}')

    print(f'\n{"institution":>13} | {"n":>4} {"min":>5} {"median":>7} | '
          f'{"≥2021":>7} {"2010-20":>8} {"2000-09":>8} {"<2000":>7}')
    print('-' * 72)
    results = {}
    for name in DATASETS:
        ys = sorted(years[p] for p in samples[name] if p in years)
        if not ys:
            continue
        med = ys[len(ys) // 2]
        r = {
            'n': len(ys), 'min_year': ys[0], 'median_year': med,
            'pct_2021plus': pct(ys, lambda y: y >= 2021),
            'pct_2010_2020': pct(ys, lambda y: 2010 <= y <= 2020),
            'pct_2000_2009': pct(ys, lambda y: 2000 <= y <= 2009),
            'pct_before_2000': pct(ys, lambda y: y < 2000),
        }
        results[name] = r
        print(f'{name:>13} | {r["n"]:>4} {r["min_year"]:>5} {r["median_year"]:>7} | '
              f'{r["pct_2021plus"]:>6.1f}% {r["pct_2010_2020"]:>7.1f}% '
              f'{r["pct_2000_2009"]:>7.1f}% {r["pct_before_2000"]:>6.1f}%')

    out = ROOT / 'external_validation' / 'uc_system' / 'article_years_check.json'
    json.dump({'n_sampled_per_institution': N, 'results': results}, open(out, 'w'), indent=2)
    print(f'\nWritten: {out}')


if __name__ == '__main__':
    main()
