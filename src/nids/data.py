"""Loading, cleaning and splitting the CIC-IDS2017 MachineLearningCSV files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_COL = "Label"
DAY_COL = "Day"
BENIGN = "BENIGN"

# Destination port lets a model shortcut on service identity (FTP-Patator is
# always port 21, SSH-Patator port 22) instead of learning traffic behaviour,
# so it is excluded by default and evaluated separately as an ablation.
SHORTCUT_FEATURES = ["Destination Port"]

META_COLS = [LABEL_COL, DAY_COL]


@dataclass
class CleaningReport:
    raw_rows: int = 0
    raw_features: int = 0
    duplicate_columns_dropped: list[str] = field(default_factory=list)
    shortcut_features_dropped: list[str] = field(default_factory=list)
    constant_columns_dropped: list[str] = field(default_factory=list)
    rows_with_nan_or_inf: int = 0
    duplicate_rows_dropped: int = 0
    conflicting_rows_dropped: int = 0
    clean_rows: int = 0
    clean_features: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def normalise_labels(labels: pd.Series) -> pd.Series:
    """Fix the mis-encoded dash in web-attack labels and stray whitespace.

    The raw files contain e.g. ``"Web Attack \\x96 Brute Force"`` (a Windows-1252
    en dash), which shows up as a replacement character in most tools.
    """
    return (
        labels.astype(str)
        .str.strip()
        .str.replace(r"^Web Attack\s+\S+\s+", "Web Attack - ", regex=True)
    )


def day_from_filename(path: Path) -> str:
    """'Tuesday-WorkingHours.pcap_ISCX.csv' -> 'Tuesday'."""
    return path.name.split("-")[0].capitalize()


def load_raw(raw_dir: str | Path) -> pd.DataFrame:
    """Read every CSV in ``raw_dir`` into one DataFrame with stripped column names."""
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files in {raw_dir}. Download MachineLearningCSV.zip from "
            "https://www.unb.ca/cic/datasets/ids-2017.html and unzip it there."
        )
    frames = []
    for f in files:
        header = pd.read_csv(f, nrows=0, encoding="latin-1").columns
        # float32 keeps the full dataset under ~1 GB in memory; "Infinity"/"NaN"
        # strings in Flow Bytes/s parse to inf/NaN and are removed in clean().
        dtypes = {c: np.float32 for c in header if c.strip() != LABEL_COL}
        # latin-1 so the odd bytes in web-attack labels never raise a decode error.
        df = pd.read_csv(f, encoding="latin-1", dtype=dtypes)
        df.columns = [c.strip() for c in df.columns]
        df[DAY_COL] = day_from_filename(f)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out[LABEL_COL] = out[LABEL_COL].astype(str)
    return out


def clean(
    df: pd.DataFrame, drop_shortcut_features: bool = True, deduplicate: bool = True
) -> tuple[pd.DataFrame, CleaningReport]:
    """Return a model-ready copy of ``df`` plus a record of everything removed.

    Steps, in order:
      1. drop the duplicated ``Fwd Header Length`` column (it appears twice in the raw files)
      2. optionally drop shortcut features (destination port)
      3. normalise labels
      4. drop rows containing NaN or +/-inf (``Flow Bytes/s`` and ``Flow Packets/s``)
      5. drop columns that are constant across the whole dataset
      6. drop exact duplicate rows (same features and label)
      7. drop rows whose feature vector appears with more than one label
    Dedup runs after feature selection so that no identical feature vector can
    land in both the train and test split. ``deduplicate=False`` skips steps 6-7
    (used only for the ablation that reproduces the common, leakier setup).
    """
    rep = CleaningReport(raw_rows=len(df), raw_features=df.shape[1] - len(META_COLS))

    # 1. duplicate columns (pandas renames the second copy to "<name>.1")
    to_drop = []
    for col in [c for c in df.columns if c.endswith(".1")]:
        base = col[:-2]
        if base in df.columns and df[base].equals(df[col]):
            to_drop.append(col)
    rep.duplicate_columns_dropped = to_drop

    # 2. shortcut features
    if drop_shortcut_features:
        rep.shortcut_features_dropped = [c for c in SHORTCUT_FEATURES if c in df.columns]
    df = df.drop(columns=to_drop + rep.shortcut_features_dropped)  # returns a new frame; input untouched

    # 3. labels
    df[LABEL_COL] = normalise_labels(df[LABEL_COL])

    feature_cols = [c for c in df.columns if c not in META_COLS]
    if any(df[c].dtype == object for c in feature_cols):
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # 4. NaN / inf
    bad = ~np.isfinite(df[feature_cols].to_numpy(dtype=np.float32)).all(axis=1)
    rep.rows_with_nan_or_inf = int(bad.sum())
    df = df.loc[~bad]

    # 5. constant columns
    constant = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=constant)
    rep.constant_columns_dropped = constant
    feature_cols = [c for c in feature_cols if c not in constant]

    if deduplicate:
        # 6. exact duplicates (features + label; Day is metadata, not part of the key)
        before = len(df)
        df = df.drop_duplicates(subset=feature_cols + [LABEL_COL])
        rep.duplicate_rows_dropped = before - len(df)

        # 7. same features, different labels -> irreducible noise; remove
        conflicted = df.duplicated(subset=feature_cols, keep=False)
        rep.conflicting_rows_dropped = int(conflicted.sum())
        df = df.loc[~conflicted]

    df[feature_cols] = df[feature_cols].astype(np.float32)
    df = df.reset_index(drop=True)
    rep.clean_rows = len(df)
    rep.clean_features = len(feature_cols)
    return df, rep


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def split(df: pd.DataFrame, test_size: float = 0.3, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split on the fine-grained attack label so rare classes appear in both halves."""
    train, test = train_test_split(df, test_size=test_size, stratify=df[LABEL_COL], random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def binary_target(labels: pd.Series) -> np.ndarray:
    """1 = attack, 0 = benign."""
    return (labels != BENIGN).to_numpy(dtype=np.int8)
