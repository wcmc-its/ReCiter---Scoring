#!/usr/bin/env python3
"""Cross-institution figures for the UC-system external validation.

Reads uc_evaluation_aggregate.json (run aggregate_uc_evaluation.py first) and the
per-institution score files, and writes three figures to
external_validation/uc_system/figures/:

  fig_uc1_discrimination_vs_calibration  -- AUC near-ceiling vs ECE divergence
  fig_uc2_reliability_uc_vs_fredhutch     -- reliability diagrams, pooled UC vs Fred Hutch
  fig_uc3_review_burden                   -- triage-band composition per institution

Visual style matches paper/figures/ (seaborn whitegrid, dpi 300, PNG + PDF).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from aggregate_uc_evaluation import load_pairs, RESULTS  # noqa: E402

OUT = ROOT / 'external_validation' / 'uc_system' / 'figures'
AGG = ROOT / 'external_validation' / 'uc_system' / 'uc_evaluation_aggregate.json'

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

C_UC = '#1f77b4'        # UC campuses (blue)
C_USC = '#9ecae1'       # USC -- private, not a UC campus (light blue)
C_POOL = '#08306b'      # pooled UC (navy)
C_FH = '#2ca02c'        # Fred Hutch baseline (green)
C_REVIEW = '#ff7f0e'    # needs-review band (orange)
C_REJECT = '#7f7f7f'    # auto-reject band (grey)

# display order: 5 UC campuses by cohort size, USC, pooled, Fred Hutch
ORDER = ['uci', 'ucla', 'ucsd', 'ucdavis', 'ucsf', 'usc']
LABELS = {'uci': 'UC Irvine', 'ucla': 'UCLA', 'ucsd': 'UC San Diego',
          'ucdavis': 'UC Davis', 'ucsf': 'UCSF', 'usc': 'USC*'}


def wilson_ci(successes, n, z=1.96):
    """Wilson score 95% interval for a binomial proportion."""
    if n == 0:
        return 0.5, 0.0, 1.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, max(0, center - spread), min(1, center + spread)


def adaptive_bins(scores, n_initial=20, min_size=800):
    """Equal-width bins merged left-to-right until each holds >= min_size points."""
    edges = list(np.linspace(0, 1, n_initial + 1))
    merged = True
    while merged:
        merged = False
        new_edges = [edges[0]]
        i = 0
        while i < len(edges) - 1:
            lo, hi = edges[i], edges[i + 1]
            mask = ((scores >= lo) & (scores <= hi)) if i == len(edges) - 2 \
                else ((scores >= lo) & (scores < hi))
            if mask.sum() < min_size and i < len(edges) - 2:
                merged = True
                i += 1
            else:
                new_edges.append(hi)
                i += 1
        edges = new_edges
    return np.array(edges)


def fig1_discrimination_vs_calibration(agg):
    """Panel A: AUC per institution (near ceiling). Panel B: ECE (diverges)."""
    insts = ORDER
    aucs = [agg['institutions'][i]['auc_roc'] for i in insts]
    eces = [agg['institutions'][i]['ece'] for i in insts]
    colors = [C_USC if i == 'usc' else C_UC for i in insts]

    pool_auc = agg['pooled']['uc_all6']['auc_roc']
    pool_ece = agg['pooled']['uc_all6']['ece']
    fh_auc = agg['baseline']['fred_hutch']['auc_roc']
    fh_ece = agg['baseline']['fred_hutch']['ece']

    names = [LABELS[i] for i in insts] + ['UC pooled', 'Fred Hutch']
    auc_vals = aucs + [pool_auc, fh_auc]
    ece_vals = eces + [pool_ece, fh_ece]
    bar_colors = colors + [C_POOL, C_FH]
    x = np.arange(len(names))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5))

    axA.bar(x, auc_vals, color=bar_colors, edgecolor='white', linewidth=0.8)
    axA.axhline(fh_auc, color=C_FH, ls='--', lw=1.2, alpha=0.7)
    axA.set_ylim(0.97, 1.0)
    axA.set_ylabel('AUC-ROC')
    axA.set_title('A. Discrimination — near ceiling at every site\n'
                  '(AUC-ROC, ACCEPTED vs REJECTED)', fontsize=11)
    for xi, v in zip(x, auc_vals):
        axA.text(xi, v + 0.0012, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    axB.bar(x, ece_vals, color=bar_colors, edgecolor='white', linewidth=0.8)
    axB.axhline(fh_ece, color=C_FH, ls='--', lw=1.2, alpha=0.7,
                label=f'Fred Hutch ECE = {fh_ece:.4f}')
    axB.axhline(0.02, color='black', ls=':', lw=1.0, alpha=0.6,
                label='Well-calibrated target (0.02)')
    axB.set_ylabel('Expected Calibration Error (15-bin)')
    axB.set_title('B. Calibration — measured ECE runs 5–7× higher at UC\n'
                  '(lower is better)', fontsize=11)
    for xi, v in zip(x, ece_vals):
        axB.text(xi, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    axB.legend(loc='upper left', fontsize=8)
    axB.set_ylim(0, max(ece_vals) * 1.25)

    for ax in (axA, axB):
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right')

    fig.suptitle('ReCiter external validation: discrimination is consistent across sites; '
                 'measured calibration error is higher at UC\n'
                 '7,581 researchers · 6 institutions · useGoldStandard=AS_EVIDENCE '
                 '(production mode) · *USC is private, not a UC campus',
                 fontsize=10, y=1.04)
    plt.tight_layout()
    _save(fig, 'fig_uc1_discrimination_vs_calibration')


def fig2_reliability(agg):
    """Reliability diagrams: pooled UC vs Fred Hutch, with score-distribution insets."""
    uc_l, uc_p = [], []
    for inst in ORDER:
        labels, preds, _ = load_pairs(RESULTS / inst / f'{inst}_scores.json')
        uc_l.append(labels)
        uc_p.append(preds)
    uc_l, uc_p = np.concatenate(uc_l), np.concatenate(uc_p)
    fh_l, fh_p, _ = load_pairs(RESULTS / 'fredhutch' / 'fredhutch_scores.json')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, scores, labels, title, color, ece in [
        (axes[0], uc_p, uc_l, 'A. UC system (pooled, 6 institutions)', C_POOL,
         agg['pooled']['uc_all6']['ece']),
        (axes[1], fh_p, fh_l, 'B. Fred Hutch (baseline)', C_FH,
         agg['baseline']['fred_hutch']['ece']),
    ]:
        # 10 equal-width bins; marker area encodes bin population (scores cluster
        # on discrete calibration plateaus, so bin counts vary by ~1000x).
        edges = np.linspace(0, 1, 11)
        xp, yt, lo, hi, ns = [], [], [], [], []
        for i in range(10):
            mask = ((scores >= edges[i]) & (scores <= edges[i + 1])) if i == 9 \
                else ((scores >= edges[i]) & (scores < edges[i + 1]))
            n = int(mask.sum())
            if n < 50:                       # drop near-empty bins
                continue
            p, l, h = wilson_ci(labels[mask].sum(), n)
            xp.append(scores[mask].mean())
            yt.append(p)
            lo.append(p - l)
            hi.append(h - p)
            ns.append(n)
        # shade the under-confident region (observed accuracy above the diagonal)
        ax.fill_between([0, 1], [0, 1], [1, 1], color=color, alpha=0.06)
        ax.text(0.35, 0.86, 'observed accept rate\nexceeds predicted score here',
                fontsize=8.5, style='italic', color='#555555', ha='center')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect calibration')
        ax.errorbar(xp, yt, yerr=[lo, hi], fmt='none', ecolor=color,
                    capsize=3, capthick=1.3, alpha=0.8)
        ax.plot(xp, yt, '-', color=color, lw=1.2, alpha=0.45)
        sizes = 40 + 360 * np.sqrt(np.array(ns) / max(ns))
        ax.scatter(xp, yt, s=sizes, color=color, edgecolor='white', linewidth=0.8,
                   zorder=5, label='Observed accept rate (marker ∝ bin n)')

        # score-distribution inset (log count)
        ins = ax.inset_axes([0.54, 0.13, 0.42, 0.32])
        ins.hist(scores, bins=50, range=(0, 1), color=color, alpha=0.75,
                 edgecolor='white', linewidth=0.4)
        ins.set_yscale('log')
        ins.set_xlabel('Score', fontsize=8)
        ins.set_ylabel('Count (log)', fontsize=8)
        ins.tick_params(labelsize=7)
        ins.set_xlim(0, 1)
        mid = ((scores > 0.10) & (scores < 0.95)).mean() * 100
        ins.text(0.96, 0.95, f'mid-range\n(10–95): {mid:.1f}%', transform=ins.transAxes,
                 fontsize=7, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives (ACCEPTED)')
        ax.set_title(f'{title}\nECE = {ece:.4f}', fontsize=11)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle('Calibration reliability: UC mid-range observed accept rate exceeds '
                 'the predicted score', fontsize=10, y=1.01)
    plt.tight_layout()
    _save(fig, 'fig_uc2_reliability_uc_vs_fredhutch')


def fig3_review_burden(agg):
    """Stacked bars: auto-accept / needs-review / auto-reject share per institution."""
    rows = ORDER + ['_pool', '_fh']
    names, accept, review, reject = [], [], [], []
    for r in rows:
        if r == '_pool':
            blk = agg['pooled']['uc_all6']; nm = 'UC pooled'
        elif r == '_fh':
            blk = agg['baseline']['fred_hutch']; nm = 'Fred Hutch'
        else:
            blk = agg['institutions'][r]; nm = LABELS[r]
        rb = blk['review_burden']
        n = blk['n_articles']
        names.append(nm)
        accept.append(rb['auto_accept_count'] / n * 100)
        review.append(rb['needs_review_pct'])
        reject.append(rb['auto_reject_count'] / n * 100)

    y = np.arange(len(names))[::-1]   # top-to-bottom
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.barh(y, accept, color=C_FH, edgecolor='white', label='Auto-accept (score ≥ 95)')
    ax.barh(y, review, left=accept, color=C_REVIEW, edgecolor='white',
            label='Needs manual review (10 < score < 95)')
    ax.barh(y, reject, left=np.array(accept) + np.array(review), color=C_REJECT,
            edgecolor='white', label='Auto-reject (score ≤ 10)')
    # needs-review % as a clean right-side column (band itself is too thin to label)
    for yi, rv in zip(y, review):
        ax.text(103, yi, f'{rv:.1f}%', ha='left', va='center', fontsize=9,
                color=C_REVIEW, fontweight='bold')
    ax.text(103, max(y) + 0.85, 'needs\nreview', ha='left', va='center', fontsize=8,
            color=C_REVIEW, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel('Share of evaluated articles (%)')
    ax.set_xlim(0, 113)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_title('Triage-band composition — UC review burden runs 9–13% vs 3.2% at Fred Hutch',
                 fontsize=11, pad=28)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=3, fontsize=8,
              framealpha=0.95)
    ax.grid(axis='y', visible=False)
    plt.tight_layout()
    _save(fig, 'fig_uc3_review_burden')


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        kw = {'dpi': 300} if ext == 'png' else {}
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', facecolor='white', **kw)
    plt.close(fig)
    print(f'  wrote {name}.png / .pdf')


def main():
    agg = json.load(open(AGG))
    print('Generating UC external-validation figures...')
    fig1_discrimination_vs_calibration(agg)
    fig2_reliability(agg)
    fig3_review_burden(agg)
    print(f'Done -> {OUT}')


if __name__ == '__main__':
    main()
