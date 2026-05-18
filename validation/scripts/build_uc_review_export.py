#!/usr/bin/env python3
"""Build per-article review exports for the 6 UC-system institutions.

For each institution, joins the scored articles (pmid, score, userAssertion)
against PubMed article metadata (title, journal, pub date, DOI) fetched from
NCBI esummary, and writes a reviewer-facing CSV. Also emits a single combined
.xlsx workbook (one sheet per institution) as the email deliverable.

Mirrors external_validation/results/fredhutch/fredhutch_all_articles_for_review.csv,
minus the target_author / target_orcid columns (no ORCID inference run for UC).

Output:
  external_validation/results/<inst>/<inst>_all_articles_for_review.csv  (6 files)
  external_validation/uc_system/uc_articles_for_review.xlsx              (6 sheets)
"""
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ESUMMARY = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'
API_KEY = os.environ.get('PUBMED_API_KEY', '')
META_CACHE = '/tmp/uc_article_meta.json'

INSTITUTIONS = [
    ('uci',     'UC Irvine'),
    ('ucla',    'UCLA'),
    ('usc',     'USC'),
    ('ucsd',    'UC San Diego'),
    ('ucdavis', 'UC Davis'),
    ('ucsf',    'UCSF'),
]
COLUMNS = ['institution', 'person_id', 'name', 'pmid', 'score', 'assertion',
           'suggested_action', 'flag',
           'title', 'journal', 'pub_date', 'doi', 'pubmed_link']


def load_names(slug):
    """Return {person_id -> (first, middle, last)} from the institution data CSV."""
    path = ROOT / f'external_validation/uc_system/data/{slug}_data.csv'
    names = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            pid = row['PersonID']
            if pid not in names:
                names[pid] = (row.get('FirstName', '') or '',
                              row.get('MiddleName', '') or '',
                              row.get('LastName', '') or '')
    return names


def load_scores(slug):
    """Return list of (uid, pmid, score, assertion) for one institution."""
    path = ROOT / f'external_validation/results/{slug}/{slug}_scores_all.json'
    with open(path) as f:
        data = json.load(f)
    rows = []
    for uid, arts in data.items():
        for a in arts:
            ua = str(a.get('userAssertion') or '').strip().upper()
            if ua in ('', 'NULL', 'NONE', 'PENDING'):
                ua = 'PENDING'   # ReCiter labels uncurated articles 'NULL'
            rows.append((uid, int(a['pmid']), float(a['score']), ua))
    return rows


def fetch_metadata(pmids):
    """Return {pmid:int -> {title, journal, pub_date, doi}}, disk-cached."""
    cache = {}
    if os.path.exists(META_CACHE):
        with open(META_CACHE) as f:
            cache = json.load(f)
    todo = sorted({p for p in pmids if str(p) not in cache})
    print(f'  metadata cache: {len(cache):,} known, {len(todo):,} to fetch')
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
            e = res.get(str(pmid)) or {}
            doi = ''
            for aid in e.get('articleids', []):
                if aid.get('idtype') == 'doi':
                    doi = aid.get('value', '')
                    break
            raw_date = e.get('sortpubdate') or e.get('pubdate') or ''
            m = re.match(r'(\d{4})[/-](\d{2})[/-](\d{2})', raw_date)
            pub_date = f'{m.group(1)}-{m.group(2)}-{m.group(3)}' if m else raw_date
            cache[str(pmid)] = {
                'title': e.get('title', ''),
                'journal': e.get('fulljournalname') or e.get('source', ''),
                'pub_date': pub_date,
                'doi': doi,
            }
        if (i // 200) % 25 == 0:
            with open(META_CACHE, 'w') as f:
                json.dump(cache, f)
            print(f'  fetched {min(i + 200, len(todo)):,}/{len(todo):,}', end='\r')
        time.sleep(0.12 if API_KEY else 0.34)
    with open(META_CACHE, 'w') as f:
        json.dump(cache, f)
    print()
    return cache


def flag_for(score, assertion):
    """Review-priority flag (compact token, for filtering/pivots)."""
    if assertion == 'ACCEPTED' and score < 50:
        return 'LOW_SCORE_ACCEPT'
    if assertion == 'REJECTED' and score >= 50:
        return 'HIGH_SCORE_REJECT'
    if assertion == 'PENDING' and score >= 99:
        return 'NEW_HIGH_CONF'
    if assertion == 'PENDING' and score >= 95:
        return 'NEW_PROBABLE'
    return ''


def suggested_action(score, assertion):
    """Plain-English action for a curator, derived from score + assertion."""
    if assertion == 'PENDING':
        if score >= 99:
            return 'Add: very likely missing'
        if score >= 95:
            return 'Add: likely (quick check)'
        if score >= 50:
            return 'Review: uncertain match'
        return 'Skip: unlikely match'
    if assertion == 'ACCEPTED':
        if score < 50:
            return 'Review: model disputes this acceptance'
        return 'OK: acceptance confirmed'
    if assertion == 'REJECTED':
        if score >= 50:
            return 'Review: model disputes this rejection'
        return 'OK: rejection confirmed'
    return ''


def build_rows(slug, label, meta, names):
    out = []
    prefix = f'{slug}_'
    for uid, pmid, score, assertion in load_scores(slug):
        person_id = uid[len(prefix):] if uid.startswith(prefix) else uid
        fn, mn, ln = names.get(person_id, ('', '', ''))
        given = ' '.join(x for x in (fn, mn) if x)
        name = f'{ln}, {given}'.strip() if ln else given
        m = meta.get(str(pmid), {})
        out.append({
            'institution': label,
            'person_id': person_id,
            'name': name,
            'pmid': pmid,
            'score': round(score, 2),
            'assertion': assertion,
            'suggested_action': suggested_action(score, assertion),
            'flag': flag_for(score, assertion),
            'title': m.get('title', ''),
            'journal': m.get('journal', ''),
            'pub_date': m.get('pub_date', ''),
            'doi': m.get('doi', ''),
            'pubmed_link': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
        })
    out.sort(key=lambda r: (r['name'].lower(), r['person_id'], -r['score']))
    return out


def main():
    all_pmids = set()
    scored = {}
    for slug, label in INSTITUTIONS:
        rows = load_scores(slug)
        scored[slug] = rows
        all_pmids.update(p for _, p, _, _ in rows)
    print(f'Total article-rows: {sum(len(v) for v in scored.values()):,}; '
          f'unique PMIDs: {len(all_pmids):,}')

    print('Fetching PubMed metadata (NCBI esummary) ...')
    meta = fetch_metadata(all_pmids)

    try:
        import openpyxl
        wb = openpyxl.Workbook()
        wb.remove(wb.active)
    except ImportError:
        wb = None

    grand = 0
    for slug, label in INSTITUTIONS:
        rows = build_rows(slug, label, meta, load_names(slug))
        grand += len(rows)
        csv_path = ROOT / f'external_validation/results/{slug}/{slug}_all_articles_for_review.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        flagged = sum(1 for r in rows if r['flag'])
        new_hc = sum(1 for r in rows if r['flag'] in ('NEW_HIGH_CONF', 'NEW_PROBABLE'))
        size_mb = csv_path.stat().st_size / 1e6
        print(f'  {label:14s} {len(rows):>7,} rows  {flagged:>6,} flagged  '
              f'{new_hc:>6,} new >=95  {size_mb:5.1f} MB')
        if wb is not None:
            ws = wb.create_sheet(title=label[:31])
            ws.append(COLUMNS)
            for r in rows:
                ws.append([r[c] for c in COLUMNS])

    if wb is not None:
        xlsx_path = ROOT / 'external_validation/uc_system/uc_articles_for_review.xlsx'
        wb.save(xlsx_path)
        size_mb = xlsx_path.stat().st_size / 1e6
        print(f'\nCombined workbook: {grand:,} rows, {size_mb:.1f} MB '
              f'-> {xlsx_path.relative_to(ROOT)}')


if __name__ == '__main__':
    sys.exit(main())
