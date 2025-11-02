# tests/test_excel_io.py
import os
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from ML.src.convert_labels import (
    convert_to_float,
    convert_to_bool,
    has_split_columns,
    normalize_label_column,
    find_label_column_name,
    parse_compact_row,
    parse_compact_sheet,
    fill_missing_columns,
    parse_excel,
)

# ---------------- convert_to_float ----------------

def test_convert_to_float_numbers():
    assert convert_to_float("3.14") == 3.14
    assert convert_to_float(2) == 2.0


def test_convert_to_float_invalid_input():
    assert convert_to_float("abc") is None
    assert convert_to_float(None) is None

# ---------------- convert_to_bool ----------------

def test_convert_to_bool_true():
    for v in ("true", "True", "1", "t", "yes", " YES "):
        assert convert_to_bool(v) is True


def test_convert_to_bool_false():
    for v in ("false", "False", "0", "f", "no", "  NO"):
        assert convert_to_bool(v) is False


def test_convert_to_bool_unknown_value():
    assert convert_to_bool("maybe") is None
    assert convert_to_bool("") is None

# ---------------- has_split_columns ----------------

def test_has_split_columns_all_present():
    cols = [
        "name_1", "email_1", "name_2", "email_2",
        "c1", "c2", "c3.1", "c3.2", "c4", "c5", "c6", "c7"
    ]
    df = pd.DataFrame(columns=cols)
    assert has_split_columns(df) is True


def test_has_split_columns_missing():
    df = pd.DataFrame(columns=["name_1", "email_1"])
    assert has_split_columns(df) is False

# ---------------- normalize_label_column ----------------

def test_normalize_label_column_lowercase():
    df = pd.DataFrame({"label": ["tp", "fp", "Tp"]})
    out = normalize_label_column(df)
    assert list(out["label"]) == ["TP", "FP", "TP"]


def test_normalize_label_column_rename_capitalized():
    df = pd.DataFrame({"Label": ["tp", "fp"]})
    out = normalize_label_column(df)
    assert "label" in out.columns
    assert list(out["label"]) == ["TP", "FP"]


def test_normalize_label_column_missing_raises():
    df = pd.DataFrame({"other": [1, 2]})
    with pytest.raises(ValueError):
        normalize_label_column(df)

# ---------------- find_label_column_name ----------------

def test_find_label_column_name():
    assert find_label_column_name(pd.DataFrame({"label": [1]})) == "label"
    assert find_label_column_name(pd.DataFrame({"Label": [1]})) == "Label"
    assert find_label_column_name(pd.DataFrame({"x": [1]})) is None

# ---------------- parse_compact_row ----------------

def test_parse_compact_row_basic():
    s = "Alice,alice@example.com,Bob,bob@example.com,0.1,0.2,0.3,0.4,true,false,1,0"
    row = parse_compact_row(s)
    assert row["name_1"] == "Alice"
    assert row["email_1"] == "alice@example.com"
    assert row["name_2"] == "Bob"
    assert row["email_2"] == "bob@example.com"
    assert row["c1"] == 0.1
    assert row["c2"] == 0.2
    assert row["c3.1"] == 0.3
    assert row["c3.2"] == 0.4
    assert row["c4"] is True
    assert row["c5"] is False
    assert row["c6"] is True
    assert row["c7"] is False


def test_parse_compact_row_commas_in_names():
    s = "Alice, A.,alice@example.com,Bob, B.,bob@example.com,1,2,3,4,yes,no,t,f"
    row = parse_compact_row(s)
    # 这里保留你当前实现的期望（无空格的“Alice,A.”、“Bob,B.”）
    assert row["name_1"] == "Alice,A."
    assert row["name_2"] == "Bob,B."
    assert row["c1"] == 1.0
    assert row["c2"] == 2.0
    assert row["c4"] is True
    assert row["c5"] is False
    assert row["c6"] is True
    assert row["c7"] is False

# ---------------- parse_compact_sheet ----------------

def test_parse_compact_sheet_maps_labels():
    data = {
        "data": [
            "Alice,alice@example.com,Bob,bob@example.com,0.1,0.2,0.3,0.4,true,false,1,0",
            "Carl,carl@example.com,Dana,dana@example.com,0.5,0.6,0.7,0.8,0,1,false,true",
        ],
        "label": ["1", "0"],
    }
    df = pd.DataFrame(data)
    out = parse_compact_sheet(df)
    assert set(out.columns) >= {"name_1", "email_1", "name_2", "email_2", "label"}
    assert list(out["label"]) == ["TP", "FP"]


def test_parse_compact_sheet_missing_label():
    df = pd.DataFrame({"data": ["Alice,a@x.com,Bob,b@x.com,0,0,0,0,1,1,1,1"]})
    with pytest.raises(ValueError):
        parse_compact_sheet(df)

# ---------------- fill_missing_columns ----------------

def test_fill_missing_columns_add_order():
    base = pd.DataFrame({"a": [1], "c": [3]})
    cols = ["a", "b", "c"]
    out = fill_missing_columns(base, cols)
    assert list(out.columns) == cols
    assert out.loc[0, "b"] is None

# ---------------- parse_excel (I/O) ----------------

def test_parse_excel_split_columns(tmp_path):
    df = pd.DataFrame(
        [
            {
                "name_1": "Alice Smith",
                "email_1": "alice@example.com",
                "name_2": "Bob Brown",
                "email_2": "bob@example.com",
                "c1": 0.1,
                "c2": 0.2,
                "c3.1": 0.3,
                "c3.2": 0.4,
                "c4": True,
                "c5": False,
                "c6": True,
                "c7": False,
                "label": "TP",
            }
        ]
    )

    xlsx = tmp_path / "input.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        df.to_excel(w, index=False)

    labels_csv = tmp_path / "labels.csv"
    candidates_csv = tmp_path / "candidates.csv"

    parse_excel(str(xlsx), str(labels_csv), str(candidates_csv))

    assert labels_csv.exists()
    assert candidates_csv.exists()

    labels = pd.read_csv(labels_csv)
    assert list(labels.columns) == ["name_1", "email_1", "name_2", "email_2", "label"]
    assert labels.shape[0] == 1

    candidates = pd.read_csv(candidates_csv)
    expected_cols = [
        "name_1", "email_1", "name_2", "email_2",
        "c1", "c2", "c3.1", "c3.2", "c4", "c5", "c6", "c7", "method"
    ]
    assert list(candidates.columns) == expected_cols
    assert candidates.loc[0, "method"] == "bird_sheet"


def test_parse_excel_compact_sheet(tmp_path):
    df = pd.DataFrame(
        {
            "data": [
                "Alice,alice@example.com,Bob,bob@example.com,0.1,0.2,0.3,0.4,true,false,1,0",
                "Carl,carl@example.com,Dana,dana@example.com,0.5,0.6,0.7,0.8,0,1,false,true",
            ],
            "Label": ["tp", "fp"],
        }
    )

    xlsx = tmp_path / "input_compact.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        df.to_excel(w, index=False)

    labels_csv = tmp_path / "labels_compact.csv"
    candidates_csv = tmp_path / "candidates_compact.csv"

    parse_excel(str(xlsx), str(labels_csv), str(candidates_csv))

    labels = pd.read_csv(labels_csv)
    assert labels.shape == (2, 5)
    assert set(labels["label"]) <= {"TP", "FP"}

    candidates = pd.read_csv(candidates_csv)
    assert "method" in candidates.columns
    assert candidates["method"].eq("bird_sheet").all()
    assert candidates.shape[0] == 2
