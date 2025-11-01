import pandas as pd
import numpy as np
from features import build_features

def build_dataset(candidates_csv, labels_csv, out_csv):
    cands = pd.read_csv(candidates_csv)
    labels = pd.read_csv(labels_csv)

    df = pd.merge(
        cands,
        labels,
        on=["name_1", "email_1", "name_2", "email_2"],
        how="left",
        validate="many_to_one"
    )


    feat_list = []
    for _, row in df.iterrows():
        pair1 = (row["name_1"], row["email_1"])
        pair2 = (row["name_2"], row["email_2"])
        feats = build_features(pair1, pair2)
        feat_list.append(feats)

    feat_array = np.vstack(feat_list)
    feat_df = pd.DataFrame(feat_array, columns=[
        "name_jw", "name_tfidf", "prefix_jw", "first_jw", "last_jw",
        "phone_first", "phone_last", "same_domain", "firstname_equal",
        "lastname_equal", "initials_equal", "prefix_has_fl",
        "prefix_has_fl_rev", "len_sim_name", "len_sim_prefix"
    ])

    feat_df["label"] = df["label"]

    feat_df.to_csv(out_csv, index=False)
    print("Output:", out_csv)

if __name__ == "__main__":
    build_dataset(
        candidates_csv="candidates_from_excel.csv",
        labels_csv="labels_from_excel.csv",
        out_csv="train_dataset.csv"
    )
