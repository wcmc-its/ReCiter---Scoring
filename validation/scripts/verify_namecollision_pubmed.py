#!/usr/bin/env python3
"""Test ACCEPTED gold standards for name-collision contamination, via PubMed.

A near-zero rejection rate (UC Davis: 0.42/researcher) is the signature of an
ACCEPTED set never disambiguated against same-surname authors. The surname check
in verify_uc_labels_pubmed.py cannot catch "right surname, wrong person" — this
does. For sampled ACCEPTED articles it pulls the FULL PubMed record (efetch) and
matches the curator-named researcher's full FIRST NAME and the author
AFFILIATION against the real author list.

UC Davis (near-zero rejections) is compared against UCLA (a normally-curated UC
school) and Fred Hutch. If UC Davis ACCEPTED articles fail first-name matching
markedly more than UCLA's, its ACCEPTED set is contaminated with name
collisions; if it matches UCLA, the low rejection count is just an export gap.

Run from the repo root:
    python3 scripts/verify_namecollision_pubmed.py
"""
import json
import os
import random
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from run_external_validation import parse_data_file, detect_column_mapping, parse_composite_name  # noqa: E402

random.seed(42)
EFETCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
API_KEY = os.environ.get('PUBMED_API_KEY', '')
N = 400  # ACCEPTED articles sampled per group

GROUPS = {
    'UC Davis':   ('external_validation/uc_system/data/ucdavis_data.csv', ['davis', 'sacramento']),
    'UCLA':       ('external_validation/uc_system/data/ucla_data.csv',   ['losangeles', 'ucla']),
    'Fred Hutch': ('external_validation/Fred Hutch - Reciter_data_20210101-20240630.xlsx',
                   ['hutchinson', 'fredhutch']),
}


def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()
    return ''.join(c for c in s.lower() if c.isalpha())


def load_accepted(path):
    """Return [(lastName, firstName, pmid)] for ACCEPTED rows only."""
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
        if str(r.get(assert_col) or '').strip().upper() != 'ACCEPTED':
            continue
        if last_col:
            ln = str(r.get(last_col) or '').strip()
            fn = str(r.get(first_col) or '').strip()
        else:
            p = parse_composite_name(str(r.get(comp_col) or ''), comp_fmt)
            ln, fn = p['lastName'], p['firstName']
        try:
            pmid = int(float(str(r.get(pmid_col))))
        except (ValueError, TypeError):
            continue
        if ln and fn:
            out.append((ln, fn, pmid))
    return out


def efetch_records(pmids):
    """Return {pmid: [{'last','fore','affils'}]} from full PubMed XML records."""
    out = {}
    pmids = list(pmids)
    for i in range(0, len(pmids), 150):
        batch = pmids[i:i + 150]
        params = {'db': 'pubmed', 'id': ','.join(map(str, batch)), 'retmode': 'xml'}
        if API_KEY:
            params['api_key'] = API_KEY
        url = EFETCH + '?' + urllib.parse.urlencode(params)
        xml = b'<x/>'
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=90) as r:
                    xml = r.read()
                break
            except Exception as e:
                if attempt == 2:
                    print(f'  batch {i//150+1} failed: {e}')
                else:
                    time.sleep(3)
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            continue
        for art in root.findall('.//PubmedArticle'):
            pmid = art.findtext('.//MedlineCitation/PMID')
            authors = []
            for au in art.findall('.//Article/AuthorList/Author'):
                ln = au.findtext('LastName')
                if not ln:
                    continue
                authors.append({
                    'last': ln,
                    'fore': au.findtext('ForeName') or '',
                    'affils': [a.text for a in au.findall('AffiliationInfo/Affiliation') if a.text],
                })
            if pmid:
                out[int(pmid)] = authors
        print(f'  fetched {min(i+150, len(pmids))}/{len(pmids)} PMIDs', end='\r')
        time.sleep(0.12 if API_KEY else 0.34)
    print()
    return out


def classify(r_last, r_first, authors, inst_kw):
    """Return (firstname_status, affil_status) for one ACCEPTED (researcher, article).

    firstname_status: no_surname | match | mismatch | unknown
    affil_status:     match | nomatch | absent | (None if no_surname)
    """
    rl, rf = norm(r_last), norm(r_first)
    cands = [a for a in authors
             if rl and (rl == norm(a['last']) or rl in norm(a['last']) or norm(a['last']) in rl)]
    if not cands:
        return 'no_surname', None

    fn_status = 'unknown'
    has_real_forename = False
    for a in cands:
        ft = a['fore'].strip().split()
        aft = norm(ft[0]) if ft else ''
        if len(aft) >= 2:                      # a real first name, not an initial
            has_real_forename = True
            if rf and (aft == rf or aft.startswith(rf) or rf.startswith(aft)):
                fn_status = 'match'
                break
    if fn_status != 'match' and has_real_forename:
        fn_status = 'mismatch'

    affil_blob = norm(' '.join(af for a in cands for af in a['affils']))
    if not affil_blob:
        af_status = 'absent'
    else:
        af_status = 'match' if any(k in affil_blob for k in inst_kw) else 'nomatch'
    return fn_status, af_status


def main():
    print(f'PubMed API key: {"set" if API_KEY else "not set"}')
    samples = {}
    for name, (path, _) in GROUPS.items():
        acc = load_accepted(path)
        random.shuffle(acc)
        samples[name] = acc[:N]
        print(f'  {name}: {len(acc)} ACCEPTED rows, sampled {len(samples[name])}')

    all_pmids = {p for s in samples.values() for _, _, p in s}
    print(f'\nFetching full PubMed records for {len(all_pmids)} PMIDs (efetch)...')
    records = efetch_records(all_pmids)
    print(f'  got {len(records)}/{len(all_pmids)} records')

    print(f'\n{"group":>11} | {"n":>4} | {"surname":>8} {"FN match":>9} '
          f'{"FN MISMATCH":>12} {"FN unk":>7} | {"affil match*":>12}')
    print('-' * 76)
    results = {}
    examples = []
    for name, (_, inst_kw) in GROUPS.items():
        fn = Counter()
        af = Counter()
        n = 0
        for ln, fnm, pmid in samples[name]:
            if pmid not in records:
                continue
            n += 1
            fs, afs = classify(ln, fnm, records[pmid], inst_kw)
            fn[fs] += 1
            if afs:
                af[afs] += 1
            if fs == 'mismatch' and name == 'UC Davis' and len(examples) < 12:
                auth = [f"{a['fore']} {a['last']}".strip() for a in records[pmid]][:6]
                examples.append((fnm, ln, pmid, auth))
        sm = n - fn['no_surname']                       # surname-matched articles
        af_known = af['match'] + af['nomatch']
        results[name] = {
            'n': n,
            'no_surname_pct': 100 * fn['no_surname'] / n if n else 0,
            'fn_match_pct': 100 * fn['match'] / sm if sm else 0,
            'fn_mismatch_pct': 100 * fn['mismatch'] / sm if sm else 0,
            'fn_unknown_pct': 100 * fn['unknown'] / sm if sm else 0,
            'affil_match_pct': 100 * af['match'] / af_known if af_known else 0,
        }
        r = results[name]
        print(f'{name:>11} | {n:>4} | {100-r["no_surname_pct"]:>7.1f}% '
              f'{r["fn_match_pct"]:>8.1f}% {r["fn_mismatch_pct"]:>11.1f}% '
              f'{r["fn_unknown_pct"]:>6.1f}% | {r["affil_match_pct"]:>11.1f}%')

    print('\n* affil match = of surname-matched articles where an affiliation is on '
          'record,\n  the share where a same-surname author’s affiliation names the institution.')
    print('\nFN MISMATCH = a same-surname author exists but their full first name '
          'clearly differs\nfrom the curator-named researcher = a likely name-collision (wrong person).')
    print('\nSample UC Davis ACCEPTED articles flagged FN-MISMATCH (researcher | pmid | authors):')
    for fnm, ln, pmid, auth in examples:
        print(f'  {fnm} {ln} | {pmid} | {auth}')

    out = ROOT / 'external_validation' / 'uc_system' / 'namecollision_check_pubmed.json'
    json.dump({'results': results, 'n_sampled_per_group': N,
               'uc_davis_mismatch_examples': [
                   {'researcher': f'{fnm} {ln}', 'pmid': pmid, 'authors': auth}
                   for fnm, ln, pmid, auth in examples]},
              open(out, 'w'), indent=2)
    print(f'\nWritten: {out}')


if __name__ == '__main__':
    main()
