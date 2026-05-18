#!/usr/bin/env python3
"""
Test 1 for the two-channel theory of ReCiter scores
====================================================
Theory (user-endorsed):
  "A high, decisive score requires sufficient evidence on at least one channel
   -- identity-match strength or feedback richness. Rejects are the half of the
   feedback channel that arms decisive rejection. Where both channels are thin,
   scores collapse into the review band."

Builds a 2x2 map of review-band rate over:
  Arm A axis  = identity-only model score (0-100)  -- what the safety net uses
  Arm B axis  = countAccepted  OR  countRejected per researcher

The theory is asymmetric, so each label is cut against BOTH feedback axes:
  - ACCEPTED articles escape the review band UPWARD via identity OR accepts.
  - REJECTED articles escape the review band DOWNWARD via identity OR rejects.
The diagonal (ACCEPTED x accepts, REJECTED x rejects) should show the OR; the
off-diagonal should show the channel does NOT transfer.

Re-scores the surviving per-researcher feature vectors locally with the deployed
model (ReCiter---Scoring app/models: 72-feature feedback + 47-feature identity-
only), mirroring verify_setup.py exactly, including the safety net
final = min(fb, io*33). No standalone stack needed.
"""
import json
import sys
import glob
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# This harness lives inside the ReCiter---Scoring repo; derive both roots.
HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORING_REPO = os.environ.get("SCORING_REPO_DIR", os.path.dirname(HARNESS_ROOT))
APP_DIR = os.path.join(SCORING_REPO, "app")
MODELS_DIR = os.path.join(APP_DIR, "models")
VECTOR_DIR = os.path.expanduser(
    os.environ.get("RECITER_FV_DIR",
                   "~/Dropbox/GitHub/ReCiter/src/main/resources/scripts"))
OUT_DIR = os.path.join(HARNESS_ROOT, "external_validation", "uc_system")
FIG_DIR = os.path.join(OUT_DIR, "figures")
FH_SCORES = os.path.join(HARNESS_ROOT, "external_validation", "results",
                         "fredhutch", "fredhutch_scores.json")
TABLE_OUT = "/tmp/theory_scored_labelled.csv.gz"   # PII -> scratch, not committed

sys.path.insert(0, APP_DIR)
import joblib  # noqa: E402
from preprocessing import (  # noqa: E402
    FEEDBACK_IDENTITY_FEATURES, IDENTITY_ONLY_FEATURES,
    preprocess_for_inference_feedback_identity,
    preprocess_for_inference_identity_only,
)

REVIEW_LO, REVIEW_HI = 10.0, 95.0   # review band, per score_by_year_check.py

IO_BINS = [0, 1, 10, 50, 90, 99, 100.0001]
IO_LABELS = ["<1", "1-10", "10-50", "50-90", "90-99", "99+"]
FB_BINS = [0, 1, 6, 21, 76, 1e12]
FB_LABELS = ["0", "1-5", "6-20", "21-75", "76+"]


def load_models():
    m = {
        "fb_model": joblib.load(f"{MODELS_DIR}/feedbackIdentityModel.joblib"),
        "fb_scaler": joblib.load(f"{MODELS_DIR}/feedbackIdentityScaler.joblib"),
        "fb_cal": joblib.load(f"{MODELS_DIR}/feedbackIdentityCalibrator.joblib"),
        "io_model": joblib.load(f"{MODELS_DIR}/identityOnlyModel.joblib"),
        "io_scaler": joblib.load(f"{MODELS_DIR}/identityOnlyScaler.joblib"),
        "io_cal": joblib.load(f"{MODELS_DIR}/identityOnlyCalibrator.joblib"),
    }
    cfg = json.loads(m["fb_model"].get_booster().save_config())
    print(f"  feedback model XGBoost {cfg['version']}  "
          f"(scaler n_features={m['fb_scaler'].n_features_in_})")
    return m


def score_file(path, m):
    with open(path) as f:
        articles = json.load(f)
    if not articles:
        return None
    df = pd.DataFrame(articles)
    if "articleId" not in df.columns:
        return None

    df_fb = preprocess_for_inference_feedback_identity(df.copy())
    df_io = preprocess_for_inference_identity_only(df.copy())

    X_fb = m["fb_scaler"].transform(df_fb[FEEDBACK_IDENTITY_FEATURES].values)
    raw_fb = m["fb_model"].predict_proba(X_fb)[:, 1]
    score_fb = np.asarray(m["fb_cal"].predict(raw_fb.reshape(-1, 1))).ravel() * 100

    X_io = m["io_scaler"].transform(df_io[IDENTITY_ONLY_FEATURES].values)
    raw_io = m["io_model"].predict_proba(X_io)[:, 1]
    score_io = np.asarray(m["io_cal"].predict(raw_io)).ravel() * 100

    score_fb_final = np.minimum(score_fb, score_io * 33.0)

    uid = os.path.basename(path).replace("-feedbackIdentityScoringInput.json", "")
    return pd.DataFrame({
        "uid": uid,
        "institution": uid.split("_")[0],
        "articleId": pd.to_numeric(df["articleId"], errors="coerce"),
        "userAssertion": df.get("userAssertion", pd.Series([None] * len(df))),
        "countAccepted": pd.to_numeric(df.get("countAccepted", 0), errors="coerce"),
        "countRejected": pd.to_numeric(df.get("countRejected", 0), errors="coerce"),
        "articleCountScore": pd.to_numeric(
            df.get("articleCountScore", np.nan), errors="coerce"),
        "score_io": score_io,
        "score_fb_raw": score_fb,
        "score_fb": score_fb_final,
    })


def cross_check_fred_hutch(scored):
    if not os.path.exists(FH_SCORES):
        print("  (fredhutch_scores.json not found - skipping cross-check)")
        return
    with open(FH_SCORES) as f:
        fh = json.load(f)
    ref = {(uid, int(a["pmid"])): a["score"]
           for uid, arts in fh.items() for a in arts}
    fr = scored[scored.institution == "fr"]
    diffs = [abs(r.score_fb - ref[(r.uid, int(r.articleId))])
             for _, r in fr.iterrows()
             if (r.uid, int(r.articleId)) in ref and pd.notna(r.articleId)]
    if diffs:
        d = np.array(diffs)
        print(f"  FH cross-check: {len(d)} articles matched | "
              f"mean|diff|={d.mean():.4f} p99={np.percentile(d,99):.3f} "
              f"max={d.max():.2f} | >0.5pt: {(d>0.5).sum()} "
              f"({100*(d>0.5).mean():.2f}%)")


def review_rate(s):
    s = np.asarray(s, dtype=float)
    return np.nan if len(s) == 0 else 100.0 * np.mean((s >= REVIEW_LO) & (s < REVIEW_HI))


def grid(df, value_col, aggfunc):
    return df.pivot_table(index="io_band", columns="fb_band", values=value_col,
                          aggfunc=aggfunc, observed=False).reindex(
        index=IO_LABELS, columns=FB_LABELS)


def make_panel(df, label, fb_col):
    sub = df[df.userAssertion == label].copy()
    sub["io_band"] = pd.cut(sub.score_io, bins=IO_BINS, labels=IO_LABELS,
                            right=False, include_lowest=True)
    sub["fb_band"] = pd.cut(sub[fb_col], bins=FB_BINS, labels=FB_LABELS,
                            right=False, include_lowest=True)
    rate = grid(sub, "score_fb", review_rate)
    n = grid(sub, "score_fb", "size")
    # decisive-wrong: ACCEPTED scored <10 ; REJECTED scored >=95
    if label == "ACCEPTED":
        sub["wrong"] = (sub.score_fb < REVIEW_LO).astype(float) * 100
    else:
        sub["wrong"] = (sub.score_fb >= REVIEW_HI).astype(float) * 100
    wrong = grid(sub, "wrong", "mean")
    return {"rate": rate, "n": n, "wrong": wrong}


def evidence_decomposition(lab):
    """Identity-floor analysis + REJECTED reject-count dose-response."""
    acc = lab[lab.userAssertion == "ACCEPTED"]
    rej = lab[lab.userAssertion == "REJECTED"]
    floor = REVIEW_HI / 33.0          # io needed so cap (io*33) reaches 95
    band = lambda s: (s >= REVIEW_LO) & (s < REVIEW_HI)

    below = acc[acc.score_io < floor]
    rb_acc = acc[band(acc.score_fb)]
    rb_below = rb_acc[rb_acc.score_io < floor]
    d = {
        "identity_floor_io": round(floor, 4),
        "accepted": {
            "n": int(len(acc)),
            "below_floor_n": int(len(below)),
            "below_floor_pct": round(100 * len(below) / len(acc), 2),
            "below_floor_auto_accept_pct": round(100 * np.mean(below.score_fb >= REVIEW_HI), 2),
            "below_floor_review_pct": round(100 * np.mean(band(below.score_fb)), 2),
            "below_floor_max_score": round(float(below.score_fb.max()), 1),
            "above_floor_auto_accept_pct": round(
                100 * np.mean(acc[acc.score_io >= floor].score_fb >= REVIEW_HI), 2),
            "review_burden_n": int(len(rb_acc)),
            "review_burden_pct": round(100 * len(rb_acc) / len(acc), 2),
            "review_burden_below_floor_share_pct": round(
                100 * len(rb_below) / len(rb_acc), 1),
        },
        "rejected": {
            "n": int(len(rej)),
            "review_burden_pct": round(100 * np.mean(band(rej.score_fb)), 2),
            "by_reject_count": {},
        },
    }
    for lo, hi, lbl in [(0, 1, "0"), (1, 6, "1-5"), (6, 21, "6-20"),
                        (21, 76, "21-75"), (76, 1e12, "76+")]:
        s = rej[(rej.countRejected >= lo) & (rej.countRejected < hi)]
        d["rejected"]["by_reject_count"][lbl] = {
            "n": int(len(s)),
            "auto_reject_pct": round(100 * np.mean(s.score_fb < REVIEW_LO), 1),
            "review_pct": round(100 * np.mean(band(s.score_fb)), 1),
            "wrong_auto_accept_pct": round(100 * np.mean(s.score_fb >= REVIEW_HI), 1),
        }
    d["below_floor_pct_by_institution"] = {
        inst: round(100 * np.mean(acc[acc.institution == inst].score_io < floor), 1)
        for inst in sorted(acc.institution.unique())
    }

    a = d["accepted"]
    print(f"\n=== Identity floor (io >= {floor:.2f} required for auto-accept) ===")
    print(f"  ACCEPTED below floor: {a['below_floor_n']:,} "
          f"({a['below_floor_pct']}%) -- auto-accept "
          f"{a['below_floor_auto_accept_pct']}%, review "
          f"{a['below_floor_review_pct']}%, max score {a['below_floor_max_score']}")
    print(f"  ACCEPTED above floor auto-accept: {a['above_floor_auto_accept_pct']}%")
    print(f"  ACCEPTED review burden {a['review_burden_n']:,} "
          f"({a['review_burden_pct']}%) -- "
          f"{a['review_burden_below_floor_share_pct']}% is below-floor articles")
    print("  below-floor % by institution:", d["below_floor_pct_by_institution"])
    print("\n=== REJECTED: reject-count dose-response ===")
    for lbl, v in d["rejected"]["by_reject_count"].items():
        print(f"  rejects {lbl:>6}: auto-reject {v['auto_reject_pct']:5.1f}%  "
              f"review {v['review_pct']:5.1f}%  "
              f"wrong-accept {v['wrong_auto_accept_pct']:4.1f}%  (n={v['n']:,})")
    return d


def confound_recut(lab):
    """Name-ambiguity re-cut on the REJECTED dose-response.

    The observational dose-response has one confound: a 0-reject researcher may
    have a well-separated (rare) name -- genuinely nothing to reject -- or a
    common name that was never curated. Split the REJECTED set by name
    ambiguity (articleCountScore = log PubMed hit-count for the name) crossed
    with reject count. If rejects causally matter, the common-name penalty
    should be large at 0 rejects and small once the researcher is well-armed.
    """
    rej = lab[lab.userAssertion == "REJECTED"].dropna(
        subset=["articleCountScore"]).copy()
    med = float(rej.articleCountScore.median())
    rej["name_ambiguity"] = np.where(rej.articleCountScore >= med,
                                     "common-name", "rare-name")
    band = lambda s: (s >= REVIEW_LO) & (s < REVIEW_HI)
    out = {"articleCountScore_median": round(med, 3), "cells": {}}
    print(f"\n=== Confound re-cut: REJECTED by name ambiguity x reject count "
          f"(articleCountScore median={med:.2f}) ===")
    for rlabel, rmask in [("0 rejects", rej.countRejected == 0),
                          ("21+ rejects", rej.countRejected >= 21)]:
        for nlabel in ("rare-name", "common-name"):
            s = rej[rmask & (rej.name_ambiguity == nlabel)]
            cell = {
                "n": int(len(s)),
                "review_pct": round(100 * np.mean(band(s.score_fb)), 1)
                    if len(s) else None,
                "wrong_auto_accept_pct": round(
                    100 * np.mean(s.score_fb >= REVIEW_HI), 1) if len(s) else None,
                "auto_reject_pct": round(100 * np.mean(s.score_fb < REVIEW_LO), 1)
                    if len(s) else None,
            }
            out["cells"][f"{rlabel} | {nlabel}"] = cell
            print(f"  {rlabel:>12} | {nlabel:>12}: review {cell['review_pct']:>5}%  "
                  f"wrong-accept {cell['wrong_auto_accept_pct']:>4}%  "
                  f"(n={cell['n']:,})")
    return out


def plot_2x2(panels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    layout = [("ACCEPTED", "countAccepted"), ("ACCEPTED", "countRejected"),
              ("REJECTED", "countAccepted"), ("REJECTED", "countRejected")]
    for ax, (label, fb_col) in zip(axes.ravel(), layout):
        p = panels[(label, fb_col)]
        rate = p["rate"].reindex(index=IO_LABELS[::-1])
        n = p["n"].reindex(index=IO_LABELS[::-1])
        im = ax.imshow(rate.values, cmap="RdYlGn_r", vmin=0, vmax=50,
                       aspect="auto")
        ax.set_xticks(range(len(FB_LABELS)))
        ax.set_xticklabels(FB_LABELS)
        ax.set_yticks(range(len(IO_LABELS)))
        ax.set_yticklabels(IO_LABELS[::-1])
        matched = (label == "ACCEPTED" and fb_col == "countAccepted") or \
                  (label == "REJECTED" and fb_col == "countRejected")
        ax.set_xlabel(f"{fb_col}  (Arm B)")
        ax.set_ylabel("identity-only score  (Arm A)")
        tag = "  <- channel match" if matched else "  (off-diagonal)"
        ax.set_title(f"{label}  x  {fb_col}{tag}", fontsize=10,
                     fontweight="bold" if matched else "normal")
        for i in range(rate.shape[0]):
            for j in range(rate.shape[1]):
                v, cnt = rate.values[i, j], n.values[i, j]
                if not np.isnan(v) and not np.isnan(cnt):
                    ax.text(j, i, f"{v:.0f}%\nn={int(cnt)}", ha="center",
                            va="center", fontsize=7.5,
                            color="black" if v < 30 else "white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="% in review band (10-95)")
    fig.suptitle("Two-channel theory -- review band concentrates where BOTH "
                 "arms are thin; feedback channel is label-specific",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  figure -> {path}")


def main():
    print("Loading deployed models ...")
    m = load_models()

    files = sorted(glob.glob(f"{VECTOR_DIR}/*-feedbackIdentityScoringInput.json"))
    print(f"Scoring {len(files)} researcher feature-vector files ...")
    parts, errs = [], 0
    for i, path in enumerate(files):
        try:
            r = score_file(path, m)
            if r is not None:
                parts.append(r)
        except Exception as e:
            errs += 1
            if errs <= 5:
                print(f"  ! {os.path.basename(path)}: {e}")
        if (i + 1) % 2000 == 0:
            print(f"  ... {i + 1}/{len(files)}")
    scored = pd.concat(parts, ignore_index=True)
    print(f"Scored {len(scored):,} article-rows from {len(parts)} researchers "
          f"({errs} file errors)")

    cross_check_fred_hutch(scored)

    lab = scored[scored.userAssertion.isin(["ACCEPTED", "REJECTED"])].copy()
    lab.to_csv(TABLE_OUT, index=False)
    print(f"  labelled table -> {TABLE_OUT}  ({len(lab):,} rows)")
    print(f"  ACCEPTED {(lab.userAssertion=='ACCEPTED').sum():,} / "
          f"REJECTED {(lab.userAssertion=='REJECTED').sum():,}")
    for col in ("countAccepted", "countRejected"):
        print(f"  {col} pctiles:",
              {p: int(np.percentile(lab[col].dropna(), p))
               for p in (25, 50, 75, 90, 99)})

    panels = {}
    for label in ("ACCEPTED", "REJECTED"):
        for fb_col in ("countAccepted", "countRejected"):
            panels[(label, fb_col)] = make_panel(lab, label, fb_col)

    for (label, fb_col), p in panels.items():
        matched = (label == "ACCEPTED" and fb_col == "countAccepted") or \
                  (label == "REJECTED" and fb_col == "countRejected")
        print(f"\n=== {label} x {fb_col}"
              f"{'  [channel match]' if matched else '  [off-diagonal]'} "
              f"-- review-band % (rows=identity, cols=feedback) ===")
        print(p["rate"].round(1).to_string())

    decomp = evidence_decomposition(lab)
    recut = confound_recut(lab)

    os.makedirs(FIG_DIR, exist_ok=True)
    plot_2x2(panels, f"{FIG_DIR}/theory_identity_rejects_heatmap.png")

    payload = {
        "review_band": [REVIEW_LO, REVIEW_HI],
        "n_researchers": int(scored.uid.nunique()),
        "n_article_rows": int(len(scored)),
        "n_labelled": int(len(lab)),
        "io_bins": IO_LABELS, "fb_bins": FB_LABELS,
        "decomposition": decomp,
        "confound_recut": recut,
        "panels": {
            f"{label}|{fb_col}": {
                metric: p[metric].round(2)
                    .where(pd.notna(p[metric]), None).to_dict()
                for metric in ("rate", "wrong", "n")
            }
            for (label, fb_col), p in panels.items()
        },
    }
    out_json = f"{OUT_DIR}/theory_identity_rejects_heatmap.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  json   -> {out_json}")


if __name__ == "__main__":
    main()
