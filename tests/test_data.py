import numpy as np
import pandas as pd
import pytest

from nids import data, evaluate


def _toy_frame() -> pd.DataFrame:
    """Small frame that exercises every cleaning step."""
    rows = [
        # port, duration, bytes/s, constant, dup-col, label
        (80, 10.0, 1.0, 0.0, 10.0, "BENIGN"),
        (80, 10.0, 1.0, 0.0, 10.0, "BENIGN"),  # exact duplicate
        (443, 10.0, 1.0, 0.0, 10.0, "BENIGN"),  # duplicate once port is dropped
        (21, 20.0, np.inf, 0.0, 20.0, "FTP-Patator"),  # inf -> dropped
        (21, 30.0, np.nan, 0.0, 30.0, "FTP-Patator"),  # NaN -> dropped
        (22, 40.0, 4.0, 0.0, 40.0, "SSH-Patator"),
        (80, 50.0, 5.0, 0.0, 50.0, "Web Attack \x96 XSS"),
        (80, 60.0, 6.0, 0.0, 60.0, "BENIGN"),
        (80, 60.0, 6.0, 0.0, 60.0, "DoS Hulk"),  # same features as previous, different label
    ]
    df = pd.DataFrame(
        rows,
        columns=["Destination Port", "Flow Duration", "Flow Bytes/s", "Bwd PSH Flags", "Flow Duration.1", "Label"],
    )
    df[data.DAY_COL] = "Monday"
    return df


def test_normalise_labels_fixes_mangled_dash():
    raw = pd.Series(["Web Attack \x96 Brute Force", "Web Attack � XSS", "Web Attack ï¿½ Sql Injection", " BENIGN "])
    assert data.normalise_labels(raw).tolist() == [
        "Web Attack - Brute Force",
        "Web Attack - XSS",
        "Web Attack - Sql Injection",
        "BENIGN",
    ]


def test_day_from_filename():
    from pathlib import Path

    assert data.day_from_filename(Path("Wednesday-workingHours.pcap_ISCX.csv")) == "Wednesday"


def test_clean_removes_bad_rows_and_columns():
    df, rep = data.clean(_toy_frame())

    assert "Destination Port" not in df.columns
    assert "Flow Duration.1" not in df.columns  # identical duplicate column
    assert "Bwd PSH Flags" not in df.columns  # constant
    assert rep.rows_with_nan_or_inf == 2
    # rows 0,1,2 collapse to one BENIGN row once port is gone
    assert rep.duplicate_rows_dropped == 2
    # BENIGN vs DoS Hulk with identical features are both removed
    assert rep.conflicting_rows_dropped == 2
    assert sorted(df["Label"]) == ["BENIGN", "SSH-Patator", "Web Attack - XSS"]
    assert np.isfinite(df[data.feature_columns(df)].to_numpy()).all()
    assert all(df[c].dtype == np.float32 for c in data.feature_columns(df))


def test_clean_keeping_port_keeps_port_distinct_rows():
    df, rep = data.clean(_toy_frame(), drop_shortcut_features=False)
    assert "Destination Port" in df.columns
    assert rep.duplicate_rows_dropped == 1  # only the exact copy


def test_clean_without_dedup_keeps_duplicates():
    df, rep = data.clean(_toy_frame(), deduplicate=False)
    assert rep.duplicate_rows_dropped == 0
    assert len(df) == 7  # 9 rows minus the NaN and inf rows


def test_clean_does_not_mutate_input():
    raw = _toy_frame()
    before = raw.copy()
    data.clean(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_split_is_stratified_and_disjoint():
    labels = ["BENIGN"] * 80 + ["DoS Hulk"] * 20
    df = pd.DataFrame({"f": np.arange(100, dtype=np.float32), "Label": labels, "Day": "Monday"})
    train, test = data.split(df, test_size=0.3, seed=0)
    assert len(test) == 30
    assert (test["Label"] == "DoS Hulk").sum() == 6
    assert set(train["f"]).isdisjoint(set(test["f"]))


def test_binary_metrics_hand_computed():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.7, 0.1, 0.9, 0.8, 0.3, 0.6])
    m = evaluate.binary_metrics(y, p)
    assert m["tp"] == 3 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 3
    assert m["false_positive_rate"] == pytest.approx(0.25)
    assert m["detection_rate"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(0.75)


def test_per_attack_detection_excludes_benign():
    labels = pd.Series(["BENIGN", "Bot", "Bot", "DDoS"])
    out = evaluate.per_attack_detection(labels, np.array([1, 0, 1, 1]))
    assert "BENIGN" not in out.index
    assert out.loc["Bot", "detection_rate"] == pytest.approx(0.5)
    assert out.loc["DDoS", "detection_rate"] == pytest.approx(1.0)
