"""End-to-end run: load -> clean -> split -> train -> evaluate -> write results/.

    python -m nids.pipeline            # expects data/MachineLearningCVE/*.csv
    python -m nids.plots            # regenerate figures/ from results/

Every number in the README comes from the files this script writes.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning

from . import data, evaluate, models
from .data import BENIGN, LABEL_COL

log = logging.getLogger("nids")


def _xy(df: pd.DataFrame, feats: list[str]):
    return df[feats].to_numpy(dtype=np.float32), df[LABEL_COL]


def _fit(name: str, X, y, seed: int):
    model = models.make_model(name, seed=seed)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)
    return model, time.perf_counter() - t0


def run_binary(train, test, feats, model_names, seed, out: Path):
    X_tr, lab_tr = _xy(train, feats)
    X_te, lab_te = _xy(test, feats)
    y_tr, y_te = data.binary_target(lab_tr), data.binary_target(lab_te)

    metrics, detection, rocs, fitted = [], [], [], {}
    for name in model_names:
        log.info("binary  | fitting %s on %s rows", name, f"{len(y_tr):,}")
        model, secs = _fit(name, X_tr, y_tr, seed)
        proba = model.predict_proba(X_te)[:, 1]
        m = evaluate.binary_metrics(y_te, proba)
        m.update(model=name, fit_seconds=round(secs, 1))
        metrics.append(m)
        log.info(
            "binary  | %-13s acc=%.5f F1=%.4f DR=%.4f FPR=%.5f (%.0fs)",
            name, m["accuracy"], m["f1"], m["detection_rate"], m["false_positive_rate"], secs,
        )
        det = evaluate.per_attack_detection(lab_te, proba >= 0.5)
        detection.append(det["detection_rate"].rename(name))
        n_test = det["n_test"]
        roc = evaluate.roc_points(y_te, proba)
        roc.insert(0, "model", name)
        rocs.append(roc)
        fitted[name] = model

    metrics_df = pd.DataFrame(metrics).set_index("model")
    metrics_df.to_csv(out / "binary_metrics.csv", float_format="%.6f")
    det_df = pd.concat([n_test] + detection, axis=1).sort_values(model_names[-1])
    det_df.index.name = "attack"
    det_df.to_csv(out / "per_attack_detection.csv", float_format="%.6f")
    pd.concat(rocs).to_csv(out / "roc_curves.csv", index=False, float_format="%.7f")

    if "random_forest" in fitted:
        imp = pd.Series(fitted["random_forest"].feature_importances_, index=feats, name="importance")
        imp.sort_values(ascending=False).to_csv(out / "feature_importance_rf.csv", float_format="%.6f")
    return metrics_df


def run_multiclass(train, test, feats, model_names, seed, out: Path):
    X_tr, y_tr = _xy(train, feats)
    X_te, y_te = _xy(test, feats)
    classes = [BENIGN] + sorted(c for c in y_tr.unique() if c != BENIGN)

    per_class_all, summaries, cms = [], [], {}
    for name in model_names:
        log.info("multi   | fitting %s", name)
        model, secs = _fit(name, X_tr, y_tr.to_numpy(), seed)
        pred = model.predict(X_te)
        per_class, cm, summary = evaluate.multiclass_tables(y_te.to_numpy(), pred, classes)
        per_class.insert(0, "model", name)
        per_class_all.append(per_class)
        summary.update(model=name, fit_seconds=round(secs, 1))
        summaries.append(summary)
        cms[name] = cm
        log.info("multi   | %-13s acc=%.5f macroF1=%.4f (%.0fs)", name, summary["accuracy"], summary["macro_f1"], secs)

    summary_df = pd.DataFrame(summaries).set_index("model")
    summary_df.to_csv(out / "multiclass_summary.csv", float_format="%.6f")
    pd.concat(per_class_all).to_csv(out / "multiclass_per_class.csv", index=False, float_format="%.6f")
    best = summary_df["macro_f1"].idxmax()
    cms[best].to_csv(out / "multiclass_confusion_best.csv")
    (out / "multiclass_best_model.txt").write_text(best + "\n")
    return summary_df


def _row_keys(df: pd.DataFrame, feats: list[str]) -> pd.Series:
    """One hash per row of the feature vector, for overlap / conflict counting."""
    return pd.util.hash_pandas_object(df[feats], index=False)


def run_ablation(raw: pd.DataFrame, test_size, seed, out: Path):
    """How much do the port feature and deduplication change the headline numbers?

    Uses the gradient-boosting detector only (fastest). Each variant is cleaned and
    split separately, because deduplication changes which rows exist. The strict
    variant reproduces the main hist_gb binary result exactly (same data, seed).
    """
    variants = [
        ("strict (default): no port, deduplicated", dict(drop_shortcut_features=True, deduplicate=True)),
        ("with destination port, deduplicated", dict(drop_shortcut_features=False, deduplicate=True)),
        ("with destination port, NOT deduplicated (common tutorial setup)",
         dict(drop_shortcut_features=False, deduplicate=False)),
    ]
    rows = []
    for label, kw in variants:
        df, rep = data.clean(raw, **kw)
        feats = data.feature_columns(df)
        keys = _row_keys(df, feats)
        n_labels = df[LABEL_COL].groupby(keys.to_numpy()).transform("nunique")
        conflicting = int((n_labels > 1).sum())
        train, test = data.split(df, test_size=test_size, seed=seed)
        del df, keys, n_labels
        seen = _row_keys(test, feats).isin(set(_row_keys(train, feats))).mean()

        X_tr, lab_tr = _xy(train, feats)
        X_te, lab_te = _xy(test, feats)
        log.info("ablation| %s: %s rows", label, f"{rep.clean_rows:,}")
        model, secs = _fit("hist_gb", X_tr, data.binary_target(lab_tr), seed)
        proba = model.predict_proba(X_te)[:, 1]
        m = evaluate.binary_metrics(data.binary_target(lab_te), proba)
        det = evaluate.per_attack_detection(lab_te, proba >= 0.5)["detection_rate"]
        m.update(
            variant=label,
            rows=rep.clean_rows,
            test_rows_with_exact_copy_in_train=float(seen),
            rows_with_conflicting_labels=conflicting,
            fit_seconds=round(secs, 1),
        )
        m.update({f"dr_{k}": v for k, v in det.items()})
        rows.append(m)
        log.info("ablation| F1=%.4f FPR=%.5f test-in-train=%.1f%% conflicting=%s",
                 m["f1"], m["false_positive_rate"], 100 * seen, f"{conflicting:,}")
        del train, test, X_tr, X_te
    pd.DataFrame(rows).set_index("variant").to_csv(out / "ablation_binary_hist_gb.csv", float_format="%.6f")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/MachineLearningCVE",
                   help="folder with the 8 CIC-IDS2017 CSVs (what MachineLearningCSV.zip unzips to)")
    p.add_argument("--out", default="results")
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", default=",".join(models.MODEL_NAMES))
    p.add_argument("--skip-multiclass", action="store_true")
    p.add_argument("--skip-ablation", action="store_true")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run_log.txt", mode="w")],
    )
    model_names = args.models.split(",")
    t_start = time.perf_counter()

    log.info("loading %s", args.raw_dir)
    raw = data.load_raw(args.raw_dir)
    df, rep = data.clean(raw)
    log.info("clean   | %s -> %s rows, %d features", f"{rep.raw_rows:,}", f"{rep.clean_rows:,}", rep.clean_features)
    train, test = data.split(df, test_size=args.test_size, seed=args.seed)
    feats = data.feature_columns(df)

    summary = {
        "run": {
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": args.seed,
            "test_size": args.test_size,
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "cleaning": rep.as_dict(),
        "raw_label_counts": data.normalise_labels(raw[LABEL_COL]).value_counts().to_dict(),
        "clean_label_counts": df[LABEL_COL].value_counts().to_dict(),
        "train_label_counts": train[LABEL_COL].value_counts().to_dict(),
        "test_label_counts": test[LABEL_COL].value_counts().to_dict(),
        "features": feats,
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    del df

    run_binary(train, test, feats, model_names, args.seed, out)
    if not args.skip_multiclass:
        run_multiclass(train, test, feats, model_names, args.seed, out)
    del train, test

    if not args.skip_ablation:
        run_ablation(raw, args.test_size, args.seed, out)

    log.info("done in %.1f min", (time.perf_counter() - t_start) / 60)


if __name__ == "__main__":
    main()
