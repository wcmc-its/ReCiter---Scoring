#!/usr/bin/env python3
"""Audit gold-standard quality: UC institutions vs the Fred Hutch baseline.

Model-independent. Reads only the source gold-standard files (UC per-institution
CSVs and the Fred Hutch xlsx) and measures internal-consistency signals that do
not depend on any ReCiter score:

  - assertion-value cleanliness (is the data binary, or is there a NEEDS-REVIEWED
    escape hatch the curators could use for uncertain articles?)
  - duplicate (researcher, PMID) rows
  - curator self-contradictions: the same (researcher, PMID) asserted both
    ACCEPTED and REJECTED  -> a hard, model-free label-error floor
  - per-researcher assertion patterns (rejections per person, all-accepted share)
  - cross-person PMID collisions: one PMID claimed as ACCEPTED by many distinct
    researchers (a long tail is implausible without consortium authorship)

Writes external_validation/uc_system/gold_standard_audit.json and prints a table.

Run from the repo root:
    python3 scripts/audit_uc_gold_standard.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from run_external_validation import parse_data_file, detect_column_mapping  # noqa: E402

DATASETS = {
    'uci':       ('UC Irvine',    'external_validation/uc_system/data/uci_data.csv'),
    'ucla':      ('UCLA',         'external_validation/uc_system/data/ucla_data.csv'),
    'usc':       ('USC',          'external_validation/uc_system/data/usc_data.csv'),
    'ucsd':      ('UC San Diego', 'external_validation/uc_system/data/ucsd_data.csv'),
    'ucdavis':   ('UC Davis',     'external_validation/uc_system/data/ucdavis_data.csv'),
    'ucsf':      ('UCSF',         'external_validation/uc_system/data/ucsf_data.csv'),
    'fredhutch': ('Fred Hutch',   'external_validation/Fred Hutch - Reciter_data_20210101-20240630.xlsx'),
}


def find_cols(mapping):
    """Return (uid_col, pmid_col, assertion_col) from a detected column mapping."""
    uid = pmid = assertion = None
    for header, (field, sub) in mapping.items():
        if field == 'uid':
            uid = header
        elif field == '_gold_standard' and sub == 'pmid':
            pmid = header
        elif field == '_gold_standard' and sub == 'assertion':
            assertion = header
    return uid, pmid, assertion


def audit(path):
    headers, rows = parse_data_file(str(ROOT / path))
    mapping, _ = detect_column_mapping(headers)
    uid_col, pmid_col, assertion_col = find_cols(mapping)
    if not (uid_col and pmid_col and assertion_col):
        raise SystemExit(f'could not map uid/pmid/assertion columns in {path}: {headers}')

    assertion_counts = Counter()        # raw assertion value -> count
    pair_assertions = defaultdict(set)  # (uid, pmid) -> set of assertions seen
    pair_rowcount = Counter()           # (uid, pmid) -> number of rows
    per_person = defaultdict(Counter)   # uid -> Counter(assertion)
    pmid_accepters = defaultdict(set)   # pmid -> set of uids asserting ACCEPTED

    n_rows = 0
    for row in rows:
        uid = row.get(uid_col)
        pmid = row.get(pmid_col)
        araw = row.get(assertion_col)
        if uid is None or pmid is None or araw is None:
            continue
        uid = str(uid).strip()
        pmid = str(pmid).strip()
        a = str(araw).strip().upper()
        if not uid or not pmid or not a:
            continue
        n_rows += 1
        assertion_counts[a] += 1
        pair_assertions[(uid, pmid)].add(a)
        pair_rowcount[(uid, pmid)] += 1
        per_person[uid][a] += 1
        if a == 'ACCEPTED':
            pmid_accepters[pmid].add(uid)

    n_acc = assertion_counts.get('ACCEPTED', 0)
    n_rej = assertion_counts.get('REJECTED', 0)
    n_other = sum(v for k, v in assertion_counts.items() if k not in ('ACCEPTED', 'REJECTED'))

    # curator self-contradictions: same (uid, pmid) asserted both ways
    contradictions = sum(1 for s in pair_assertions.values()
                         if 'ACCEPTED' in s and 'REJECTED' in s)
    # duplicate rows: a (uid, pmid) pair appearing in more than one row
    dup_pairs = sum(1 for c in pair_rowcount.values() if c > 1)

    # per-researcher assertion patterns
    n_people = len(per_person)
    all_acc = all_rej = mixed = 0
    rej_per_person = []
    for uid, c in per_person.items():
        a, r = c.get('ACCEPTED', 0), c.get('REJECTED', 0)
        rej_per_person.append(r)
        if a and not r:
            all_acc += 1
        elif r and not a:
            all_rej += 1
        elif a and r:
            mixed += 1
    rej_per_person.sort()
    median_rej = rej_per_person[n_people // 2] if n_people else 0
    zero_rej_people = sum(1 for r in rej_per_person if r == 0)

    # cross-person PMID collisions (distinct researchers asserting ACCEPTED)
    accepters_per_pmid = sorted((len(s) for s in pmid_accepters.values()), reverse=True)
    collisions = {f'>={n}': sum(1 for c in accepters_per_pmid if c >= n)
                  for n in (2, 5, 10, 20, 50)}

    return {
        'n_rows': n_rows,
        'n_researchers': n_people,
        'n_distinct_pairs': len(pair_assertions),
        'assertion_values': dict(assertion_counts),
        'n_accepted': n_acc,
        'n_rejected': n_rej,
        'n_other_assertions': n_other,
        'accept_rate_of_decided': n_acc / (n_acc + n_rej) if (n_acc + n_rej) else None,
        'duplicate_pairs': dup_pairs,
        'self_contradictions': contradictions,
        'self_contradiction_rate': contradictions / len(pair_assertions) if pair_assertions else None,
        'rejections_per_researcher_mean': (n_rej / n_people) if n_people else None,
        'rejections_per_researcher_median': median_rej,
        'researchers_with_zero_rejections_pct': 100 * zero_rej_people / n_people if n_people else None,
        'pct_all_accepted': 100 * all_acc / n_people if n_people else None,
        'pct_all_rejected': 100 * all_rej / n_people if n_people else None,
        'pct_mixed': 100 * mixed / n_people if n_people else None,
        'max_accepters_one_pmid': accepters_per_pmid[0] if accepters_per_pmid else 0,
        'pmid_collisions': collisions,
    }


def main():
    results = {}
    for key, (name, path) in DATASETS.items():
        print(f'auditing {name} ...', flush=True)
        results[key] = {'display_name': name, **audit(path)}

    out = ROOT / 'external_validation' / 'uc_system' / 'gold_standard_audit.json'
    json.dump(results, open(out, 'w'), indent=2)

    print()
    hdr = (f'{"dataset":>13} | {"people":>7} {"pairs":>8} | {"accept%":>7} '
           f'{"rej/pers":>8} {"0-rej%":>7} {"allAcc%":>8} {"mixed%":>7} | '
           f'{"contra":>6} {"dup":>6} {"other":>6} | {"maxClaim":>8}')
    print(hdr)
    print('-' * len(hdr))
    for key in DATASETS:
        r = results[key]
        print(f'{r["display_name"]:>13} | {r["n_researchers"]:>7} '
              f'{r["n_distinct_pairs"]:>8} | {r["accept_rate_of_decided"]*100:>6.1f}% '
              f'{r["rejections_per_researcher_mean"]:>8.2f} '
              f'{r["researchers_with_zero_rejections_pct"]:>6.1f}% '
              f'{r["pct_all_accepted"]:>7.1f}% {r["pct_mixed"]:>6.1f}% | '
              f'{r["self_contradictions"]:>6} {r["duplicate_pairs"]:>6} '
              f'{r["n_other_assertions"]:>6} | {r["max_accepters_one_pmid"]:>8}')
    print()
    print('assertion values per dataset:')
    for key in DATASETS:
        r = results[key]
        print(f'  {r["display_name"]:>13}: {r["assertion_values"]}')
    print()
    print('PMID collisions (one PMID asserted ACCEPTED by N distinct researchers):')
    for key in DATASETS:
        r = results[key]
        print(f'  {r["display_name"]:>13}: {r["pmid_collisions"]}')
    print(f'\nWritten: {out}')


if __name__ == '__main__':
    main()
