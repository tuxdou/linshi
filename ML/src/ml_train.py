import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, precision_recall_curve
)
import joblib

FEAT_COLS = [
    "name_jw", "name_tfidf", "prefix_jw", "first_jw", "last_jw",
    "phone_first", "phone_last", "same_domain", "firstname_equal",
    "lastname_equal", "initials_equal", "prefix_has_fl",
    "prefix_has_fl_rev", "len_sim_name", "len_sim_prefix"
]


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    # label（TP/FP）or y（0/1）
    if "y" in df.columns:
        y = df["y"].astype(int).values
    elif "label" in df.columns:
        y = (df["label"].astype(str).str.upper() == "TP").astype(int).values
    else:
        raise ValueError("The data is missing the label (TP/FP) or y (0/1) column")

    X = df[FEAT_COLS].values
    return X, y


def train_and_eval(train_csv, model_out="logreg.pkl", test_size=0.25, random_state=42):
    X, y = load_dataset(train_csv)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # LogisticRegression
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(x_train, y_train)

    proba = clf.predict_proba(x_test)[:, 1]

    threshold = 0.916
    pred = (proba >= threshold).astype(int)

    roc = roc_auc_score(y_test, proba)
    pr  = average_precision_score(y_test, proba)
    print("ROC-AUC:", round(roc, 3))
    print("PR-AUC :", round(pr, 3))
    print(classification_report(y_test, pred, digits=3, zero_division=0))


   
    p, r, thr = precision_recall_curve(y_test, proba)
    f1s = 2 * p * r / np.clip(p + r, 1e-9, None)
    best_i = int(np.nanargmax(f1s))
    best_thr = thr[best_i-1] if best_i > 0 and best_i-1 < len(thr) else 0.5
    print("Recommended threshold (based on F1 maximum):", round(float(best_thr), 3))

    joblib.dump(clf, model_out)


if __name__ == "__main__":
    train_and_eval(
        train_csv="train_dataset.csv",  
        model_out="logreg.pkl"
    )
