import pandas as pd
from pathlib import Path

def convert_to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def convert_to_bool(value):
    s = str(value).strip().lower()
    if s in ("true", "1", "t", "yes"):
        return True
    if s in ("false", "0", "f", "no"):
        return False
    return None


def has_split_columns(df):
    expected = {
        "name_1", "email_1", "name_2", "email_2",
        "c1", "c2", "c3.1", "c3.2", "c4", "c5", "c6", "c7"
    }
    return expected.issubset(set(map(str, df.columns)))


def normalize_label_column(df):
    base = df.copy()
    if "label" in base.columns:
        base["label"] = base["label"].astype(str).str.upper()
        return base
    if "Label" in base.columns:
        base = base.rename(columns={"Label": "label"})
        base["label"] = base["label"].astype(str).str.upper()
        return base
    raise ValueError("The label column is missing in Excel.")


def find_label_column_name(df):
    if "label" in df.columns:
        return "label"
    if "Label" in df.columns:
        return "Label"
    return None


def parse_compact_row(cell_value):

    parts = str(cell_value).split(",")
    metrics = parts[-8:]
    left = parts[:-8]

    name1_list, name2_list = [], []
    email1, email2 = "", ""
    i = 0

    # name_1
    while i < len(left) and "@" not in left[i]:
        name1_list.append(left[i])
        i += 1

    # email_1
    if i < len(left):
        email1 = left[i].strip()
        i += 1

    # name_2
    while i < len(left) and "@" not in left[i]:
        name2_list.append(left[i])
        i += 1

    # email_2
    if i < len(left):
        email2 = left[i].strip()
        i += 1

    name1 = ",".join([x.strip() for x in name1_list]).strip()
    name2 = ",".join([x.strip() for x in name2_list]).strip()

    c1, c2, c31, c32, c4, c5, c6, c7 = metrics
    return {
        "name_1": name1,
        "email_1": email1,
        "name_2": name2,
        "email_2": email2,
        "c1": convert_to_float(c1),
        "c2": convert_to_float(c2),
        "c3.1": convert_to_float(c31),
        "c3.2": convert_to_float(c32),
        "c4": convert_to_bool(c4),
        "c5": convert_to_bool(c5),
        "c6": convert_to_bool(c6),
        "c7": convert_to_bool(c7),
    }


def parse_compact_sheet(df):
    first_col = str(df.columns[0])
    parsed = df[first_col].apply(parse_compact_row).apply(pd.Series)

    label_col = find_label_column_name(df)
    if label_col is None:
        raise ValueError("The label column is missing in Excel.")

    base = parsed.copy()
    base["label"] = df[label_col].map(
        lambda v: "TP" if str(v).strip() in ("1", "TP", "tp") else "FP"
    ).astype(str)
    return base


def fill_missing_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns].copy()



def parse_excel(input_xlsx, output_labels_csv, output_candidates_csv):
    df = pd.ExcelFile(input_xlsx).parse(0)


    if has_split_columns(df):
        base = normalize_label_column(df)
    else:
        base = parse_compact_sheet(df)

    # labels
    labels = base[["name_1", "email_1", "name_2", "email_2", "label"]].copy()
    Path(output_labels_csv).parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_labels_csv, index=False)

    # candidates
    candidate_columns = [
        "name_1", "email_1", "name_2", "email_2",
        "c1", "c2", "c3.1", "c3.2", "c4", "c5", "c6", "c7"
    ]
    candidates = fill_missing_columns(base.copy(), candidate_columns)
    candidates["method"] = "bird_sheet"

    Path(output_candidates_csv).parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_candidates_csv, index=False)

    print(f"Output saved: {output_labels_csv}")
    print(f"Output saved: {output_candidates_csv}")


if __name__ == "__main__":
    parse_excel(
        input_xlsx="devs_similarity_t=0.65.xlsx",
        output_labels_csv="labels_from_excel.csv",
        output_candidates_csv="candidates_from_excel.csv",
    )
