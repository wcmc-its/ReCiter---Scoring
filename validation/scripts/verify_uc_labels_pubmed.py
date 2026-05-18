#!/usr/bin/env python3
"""Independent verification of gold-standard label accuracy against PubMed.

Model-independent. For a sample of (researcher, PMID) pairs from the gold
standard, fetch the article's real author list from PubMed (NCBI esummary) and
check whether the curator-named researcher's surname actually appears among the
authors. An ACCEPTED article whose author list contains no matching surname is a
near-certain label error (modulo name changes / transliteration).

The same check is run on the UC institutions and on Fred Hutch, so the
*difference* in no-match rate is the signal even if the absolute rate carries
some name-normalization noise.

REJECTED articles are a control: they were retrieved because the name collided,
so they should surname-match at a high rate regardless of the curator's call.

Run from the repo root:
    python3 scripts/verify_uc_labels_pubmed.py
"""
import json
import os
import random
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from run_external_validation import parse_data_file, detect_column_mapping, parse_composite_name  # noqa: E402

random.seed(42)
ESUMMARY = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'
API_KEY = os.environ.get('PUBMED_API_KEY', '')

DATASETS = {
    'UC Irvine':    ('uc', 'external_validation/uc_system/data/uci_data.csv'),
    'UCLA':         ('uc', 'external_validation/uc_system/data/ucla_data.csv'),
    'USC':          ('uc', 'external_validation/uc_system/data/usc_data.csv'),
    'UC San Diego': ('uc', 'external_validation/uc_system/data/ucsd_data.csv'),
    'UC Davis':     ('uc', 'external_validation/uc_system/data/ucdavis_data.csv'),
    'UCSF':         ('uc', 'external_validation/uc_system/data/ucsf_data.csv'),
    'Fred Hutch':   ('fh', 'external_validation/Fred Hutch - Reciter_data_20210101-20240630.xlsx'),
}
N_ACCEPTED = 600    # sampled ACCEPTED pairs per group (UC pooled / Fred Hutch)
N_REJECTED = 300    # sampled REJECTED pairs per group (control)


def norm(s):
    """Lowercase, strip diacritics, keep letters only."""
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()
    return ''.join(c for c in s.lower() if c.isalpha())


def load_dataset(path):
    """Return list of (lastName, firstName, pmid, assertion) from a GS file."""
    headers, rows = parse_data_file(str(ROOT / path))
    mapping, _ = detect_column_mapping(headers)
    pmid_col = assert_col = first_col = last_col = comp_col = comp_fmt = None
    for h, (f, s) in mapping.items():
        if f == '_gold_standard' and s == 'pmid':
            pmid_col = h
        elif f == '_gold_standard' and s == 'assertion':
            assert_col = h
        elif f == 'primaryName' and s == 'firstName':
            first_col = h
        elif f == 'primaryName' and s == 'lastName':
            last_col = h
        elif f == '_composite_name':
            comp_col, comp_fmt = h, s
    out = []
    for r in rows:
        pmid_raw, a = r.get(pmid_col), r.get(assert_col)
        if not pmid_raw or not a:
            continue
        if last_col:
            ln = str(r.get(last_col) or '').strip()
            fn = str(r.get(first_col) or '').strip()
        elif comp_col:
            p = parse_composite_name(str(r.get(comp_col) or ''), comp_fmt)
            ln, fn = p['lastName'], p['firstName']
        else:
            continue
        if not ln:
            continue
        try:
            pmid = int(float(str(pmid_raw)))
        except (ValueError, TypeError):
            continue
        out.append((ln, fn, pmid, str(a).strip().upper()))
    return out


def fetch_authors(pmids):
    """Return {pmid: [author name strings]} via NCBI esummary, batched."""
    authors = {}
    pmids = list(pmids)
    for i in range(0, len(pmids), 200):
        batch = pmids[i:i + 200]
        params = {'db': 'pubmed', 'id': ','.join(map(str, batch)), 'retmode': 'json'}
        if API_KEY:
            params['api_key'] = API_KEY
        url = ESUMMARY + '?' + urllib.parse.urlencode(params)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = json.loads(resp.read())
                break
            except Exception as e:
                if attempt == 2:
                    print(f'  batch {i//200+1}: failed ({e})')
                    data = {'result': {}}
                else:
                    time.sleep(2)
        res = data.get('result', {})
        for pmid in batch:
            entry = res.get(str(pmid))
            if entry and 'authors' in entry:
                authors[pmid] = [a.get('name', '') for a in entry['authors']
                                 if a.get('name')]
        print(f'  fetched {min(i+200, len(pmids))}/{len(pmids)} PMIDs', end='\r')
        time.sleep(0.12 if API_KEY else 0.34)
    print()
    return authors


def surname_match(last_name, first_name, author_names):
    """(surname_match, surname_plus_initial_match) for a researcher vs an author list.

    esummary author names are 'Surname Initials' (e.g. 'Tarlock K')."""
    rl = norm(last_name)
    rf_init = norm(first_name)[:1]
    if not rl:
        return False, False
    sm = im = False
    for name in author_names:
        toks = name.split()
        if not toks:
            continue
        # last token is the initials block; the rest is the surname
        surname = norm(' '.join(toks[:-1])) if len(toks) > 1 else norm(toks[0])
        initials = norm(toks[-1]) if len(toks) > 1 else ''
        whole = norm(name)
        hit = bool(rl) and (rl == surname or rl in surname or surname in rl
                            or rl in whole)
        if hit:
            sm = True
            if not rf_init or rf_init in initials:
                im = True
    return sm, im


def sample(pairs, assertion, n):
    pool = [p for p in pairs if p[3] == assertion]
    random.shuffle(pool)
    return pool[:n]


def main():
    print(f'PubMed API key: {"set" if API_KEY else "not set (slower, still fine)"}')
    uc_pairs, fh_pairs = [], []
    for name, (grp, path) in DATASETS.items():
        ds = load_dataset(path)
        (uc_pairs if grp == 'uc' else fh_pairs).extend(ds)
        print(f'  loaded {name}: {len(ds)} rows')

    groups = {
        'UC ACCEPTED':  sample(uc_pairs, 'ACCEPTED', N_ACCEPTED),
        'FH ACCEPTED':  sample(fh_pairs, 'ACCEPTED', N_ACCEPTED),
        'UC REJECTED':  sample(uc_pairs, 'REJECTED', N_REJECTED),
        'FH REJECTED':  sample(fh_pairs, 'REJECTED', N_REJECTED),
    }
    all_pmids = {p[2] for g in groups.values() for p in g}
    print(f'\nFetching author lists for {len(all_pmids)} unique PMIDs from PubMed...')
    authors = fetch_authors(all_pmids)
    print(f'  got author lists for {len(authors)}/{len(all_pmids)} PMIDs')

    print(f'\n{"group":>13} | {"n":>5} {"found":>6} | {"surname":>8} {"+initial":>9} '
          f'{"NO-MATCH":>9}')
    print('-' * 62)
    results = {}
    examples = []
    for gname, pairs in groups.items():
        n = found = sm = im = 0
        for ln, fn, pmid, _ in pairs:
            if pmid not in authors:
                continue
            found += 1
            n += 1
            s, i = surname_match(ln, fn, authors[pmid])
            sm += s
            im += i
            if not s and gname == 'UC ACCEPTED' and len(examples) < 12:
                examples.append((ln, fn, pmid, authors[pmid][:6]))
        nomatch = n - sm
        results[gname] = {'n': n, 'surname_pct': 100*sm/n if n else 0,
                          'initial_pct': 100*im/n if n else 0,
                          'nomatch_pct': 100*nomatch/n if n else 0}
        print(f'{gname:>13} | {len(pairs):>5} {found:>6} | '
              f'{100*sm/n if n else 0:>7.1f}% {100*im/n if n else 0:>8.1f}% '
              f'{100*nomatch/n if n else 0:>8.1f}%')

    print(f'\nHeadline: of sampled ACCEPTED articles, the curator-named researcher\'s '
          f'surname is\nNOT among the PubMed authors for '
          f'{results["UC ACCEPTED"]["nomatch_pct"]:.1f}% of UC vs '
          f'{results["FH ACCEPTED"]["nomatch_pct"]:.1f}% of Fred Hutch.')
    print('\nSample UC ACCEPTED articles with no surname match (researcher | pmid | authors):')
    for ln, fn, pmid, auth in examples:
        print(f'  {fn} {ln} | {pmid} | {auth}')

    out = ROOT / 'external_validation' / 'uc_system' / 'label_verification_pubmed.json'
    json.dump({'results': results, 'n_accepted_sampled': N_ACCEPTED,
               'n_rejected_sampled': N_REJECTED,
               'uc_accepted_nomatch_examples': [
                   {'researcher': f'{fn} {ln}', 'pmid': pmid, 'authors': auth}
                   for ln, fn, pmid, auth in examples]},
              open(out, 'w'), indent=2)
    print(f'\nWritten: {out}')


if __name__ == '__main__':
    main()
