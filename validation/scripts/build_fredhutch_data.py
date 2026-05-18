#!/usr/bin/env python3
"""Convert the Fred Hutch assertion xlsx into the standard harness data CSV.

Source: external_validation/Fred Hutch - Reciter_data_20210101-20240630.xlsx
  sheet "User Assertion Details" — columns: Tracked Author, Uid,
  Date Publication Added To Entrez, Article Title, Assertion Status, Pmid.

Output: external_validation/uc_system/data/fredhutch_data.csv
  columns: PersonID, FirstName, MiddleName, LastName, PMID, Assertion
  (one row per ACCEPTED / REJECTED assertion — the gold standard).

This makes Fred Hutch a uniform institution alongside the six UC-system sites,
so the same scoring and review-export scripts handle all seven without
special-casing. NEEDS REVIEWED / blank statuses are dropped from the gold
standard, matching how Fred Hutch was treated in the cross-institution
evaluation.
"""
import csv
from collections import Counter
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'external_validation/Fred Hutch - Reciter_data_20210101-20240630.xlsx'
OUT = ROOT / 'external_validation/uc_system/data/fredhutch_data.csv'
KEEP = {'ACCEPTED', 'REJECTED'}


def split_name(tracked_author):
    """'Last, First M' -> (first, middle, last)."""
    s = (tracked_author or '').strip()
    if ',' in s:
        last, given = s.split(',', 1)
    else:
        last, given = s, ''
    parts = given.split()
    first = parts[0] if parts else ''
    middle = ' '.join(parts[1:])
    return first, middle, last.strip()


def main():
    wb = openpyxl.load_workbook(SRC, read_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    header = [str(h or '').strip() for h in next(rows)]
    idx = {name: header.index(name) for name in
           ('Tracked Author', 'Uid', 'Assertion Status', 'Pmid')}

    seen = set()
    out_rows = []
    status_counts = Counter()
    for row in rows:
        status = str(row[idx['Assertion Status']] or '').strip().upper()
        status_counts[status] += 1
        if status not in KEEP:
            continue
        uid = str(row[idx['Uid']] or '').strip()
        pmid = str(row[idx['Pmid']] or '').strip()
        if not uid or not pmid:
            continue
        key = (uid, pmid, status)
        if key in seen:
            continue
        seen.add(key)
        first, middle, last = split_name(row[idx['Tracked Author']])
        out_rows.append((uid, first, middle, last, pmid, status))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['PersonID', 'FirstName', 'MiddleName', 'LastName',
                    'PMID', 'Assertion'])
        w.writerows(out_rows)

    persons = len({r[0] for r in out_rows})
    print('Source assertion-status distribution:')
    for s, c in status_counts.most_common():
        mark = ' (kept)' if s in KEEP else ' (dropped)'
        print(f'  {s or "(blank)":20s} {c:>7,}{mark}')
    print(f'\nWrote {len(out_rows):,} rows, {persons:,} researchers -> '
          f'{OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
