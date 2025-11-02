import pandas as pd
import numpy as np
import joblib
from src.features import build_features  

def score_candidates(candidates_csv, model_pkl, out_csv, threshold=None, topk=None):
    df = pd.read_csv(candidates_csv).copy()

    feats = []
    for _, r in df.iterrows():
        a = (r["name_1"], r["email_1"])
        b = (r["name_2"], r["email_2"])
        feats.append(build_features(a, b))
    X = np.vstack(feats)

    model = joblib.load(model_pkl)
    proba = model.predict_proba(X)[:, 1]
    df["proba"] = proba

    df = df.sort_values("proba", ascending=False)

    if topk is not None:
        df_out = df.head(int(topk))
    elif threshold is not None:
        df_out = df[df["proba"] >= float(threshold)]
    else:
        df_out = df

    df_out.to_csv(out_csv, index=False)
    print(f"output: {out_csv}  rows={len(df_out)}")

if __name__ == "__main__":
    score_candidates(
        candidates_csv="devs_similarity.csv",
        model_pkl="logreg.pkl",
        out_csv="22ml_scored_p085.csv",
        threshold=0.916
    )

