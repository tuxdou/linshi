# tests/test_trainer.py
import os
import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path

from ML.src.ml_train import load_dataset, train_and_eval, FEAT_COLS


# ------------------------------------------------
# Helpers
# ------------------------------------------------

def make_dataset_df(n=40, use_label=True, seed=0):
    """Create a tiny synthetic, balanced dataset with required feature columns."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, len(FEAT_COLS)))
    df = pd.DataFrame(X, columns=FEAT_COLS)
    y = np.array([0, 1] * (n // 2))
    if use_label:
        df["label"] = np.where(y == 1, "TP", "FP")
    else:
        df["y"] = y
    return df


# ------------------------------------------------
# load_dataset
# ------------------------------------------------

def test_load_dataset_label_column(tmp_path):
    """Maps TP/FP -> 1/0 and returns X, y with correct shapes."""
    csv_path = tmp_path / "data_label.csv"
    make_dataset_df(use_label=True).to_csv(csv_path, index=False)

    X, y = load_dataset(csv_path)
    assert X.shape[1] == len(FEAT_COLS)
    assert set(np.unique(y)) <= {0, 1}
    assert y.sum() == len(y) // 2  # half are TP


def test_load_dataset_numeric_y(tmp_path):
    """Reads numeric y column correctly."""
    csv_path = tmp_path / "data_y.csv"
    make_dataset_df(use_label=False).to_csv(csv_path, index=False)

    X, y = load_dataset(csv_path)
    assert X.shape[0] == len(y)
    assert set(np.unique(y)) == {0, 1}


def test_load_dataset_missing_label_raises(tmp_path):
    """Raises ValueError when neither label nor y exists."""
    df = make_dataset_df(use_label=True)
    df = df.drop(columns=["label"])
    csv_path = tmp_path / "bad.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_dataset(csv_path)


# ------------------------------------------------
# train_and_eval
# ------------------------------------------------

def test_train_and_eval_saves_model(tmp_path, capsys):
    """Fits model, prints metrics, and saves joblib file."""
    csv_path = tmp_path / "train.csv"
    make_dataset_df(n=40, use_label=True, seed=42).to_csv(csv_path, index=False)

    model_out = tmp_path / "logreg.pkl"
    train_and_eval(str(csv_path), model_out=str(model_out), test_size=0.25, random_state=123)

    # Model file should exist and be loadable
    assert model_out.exists()
    clf = joblib.load(model_out)
    assert hasattr(clf, "predict_proba")

    # Check printed metrics
    out = capsys.readouterr().out
    assert "ROC-AUC:" in out
    assert "PR-AUC" in out
    assert "Recommended threshold" in out
