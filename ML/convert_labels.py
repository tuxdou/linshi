import pandas as pd
from pathlib import Path

def parse_excel(in_xlsx, out_labels_csv, out_cands_csv):
    df = pd.ExcelFile(in_xlsx).parse(0)

    expected_cols = {"name_1","email_1","name_2","email_2","c1","c2","c3.1","c3.2","c4","c5","c6","c7"}
    has_split_cols = expected_cols.issubset(set(map(str, df.columns)))

    if has_split_cols:
        base = df.copy()
        if "label" in base.columns:
            base["label"] = base["label"].astype(str).str.upper()
        elif "Label" in base.columns:
            base = base.rename(columns={"Label":"label"})
            base["label"] = base["label"].astype(str).str.upper()
        else:
            raise ValueError("The label column is missing in Excel.")
    else:
        first_col = str(df.columns[0])

        def parse_row(s):
            parts = str(s).split(",")
            metrics = parts[-8:]
            left = parts[:-8]

            name1_tokens, name2_tokens = [], []
            email1, email2 = "", ""
            i = 0
            while i < len(left) and "@" not in left[i]:
                name1_tokens.append(left[i]); i += 1
            if i < len(left):
                email1 = left[i].strip(); i += 1
            while i < len(left) and "@" not in left[i]:
                name2_tokens.append(left[i]); i += 1
            if i < len(left):
                email2 = left[i].strip(); i += 1

            name1 = ",".join([t.strip() for t in name1_tokens]).strip()
            name2 = ",".join([t.strip() for t in name2_tokens]).strip()

            def to_float(x):
                try: return float(x)
                except: return None
            def to_bool(x):
                s = str(x).strip().lower()
                if s in ("true","1","t","yes"): return True
                if s in ("false","0","f","no"): return False
                return None

            c1, c2, c31, c32, c4, c5, c6, c7 = metrics
            return {
                "name_1": name1, "email_1": email1,
                "name_2": name2, "email_2": email2,
                "c1": to_float(c1), "c2": to_float(c2),
                "c3.1": to_float(c31), "c3.2": to_float(c32),
                "c4": to_bool(c4), "c5": to_bool(c5),
                "c6": to_bool(c6), "c7": to_bool(c7),
            }

        parsed = df[first_col].apply(parse_row).apply(pd.Series)

        if "label" in df.columns:
            lab_col = "label"
        elif "Label" in df.columns:
            lab_col = "Label"
        else:
            raise ValueError("The label column is missing in Excel.")

        base = parsed.copy()
        base["label"] = df[lab_col].map(lambda v: "TP" if str(v).strip() in ("1","TP","tp") else "FP").astype(str)

    labels = base[["name_1","email_1","name_2","email_2","label"]].copy()
    Path(out_labels_csv).parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_labels_csv, index=False)

    cands_cols = ["name_1","email_1","name_2","email_2","c1","c2","c3.1","c3.2","c4","c5","c6","c7"]
    for c in cands_cols:
        if c not in base.columns:
            base[c] = None
    cands = base[cands_cols].copy()
    cands["method"] = "bird_sheet"
    Path(out_cands_csv).parent.mkdir(parents=True, exist_ok=True)
    cands.to_csv(out_cands_csv, index=False)

    print(f"output: {out_labels_csv}")
    print(f"output: {out_cands_csv}")

if __name__ == "__main__":
    parse_excel(
        in_xlsx="devs_similarity_t=0.65.xlsx",      
        out_labels_csv="labels_from_excel.csv",     
        out_cands_csv="candidates_from_excel.csv"   
    )
