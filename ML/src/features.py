import numpy as np
import jellyfish
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocess import normalize_name, split_name, normalize_email


def jaro_winkler_sim(a, b):
    if not a:
        a = ""
    if not b:
        b = ""
    return float(JaroWinkler.normalized_similarity(a, b))


def tfidf_similarity(a, b):
    if not a:
        a = ""
    if not b:
        b = ""

    if not a.strip() and not b.strip():
        return 0.0
    
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    X = vectorizer.fit_transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0, 0])


def phonetic_similarity(a, b):
    if not a:
        a = ""
    if not b:
        b = ""
    same_soundex = jellyfish.soundex(a) == jellyfish.soundex(b)
    same_metaphone = jellyfish.metaphone(a) == jellyfish.metaphone(b)
    if same_soundex and same_metaphone:
        return 1.0
    elif same_soundex or same_metaphone:
        return 0.5
    else:
        return 0.0


def get_initials(name):
    name = normalize_name(name)
    parts = name.split()
    initials = ""
    for p in parts:
        initials += p[0]
    return initials


def prefix_contains_name(first, last, prefix):
    if not first or not last:
        return 0
    if first[0] in prefix and last in prefix:
        return 1
    else:
        return 0


def build_features(pair1, pair2):
    
    name1, email1 = pair1
    name2, email2 = pair2

    # clean and split the data
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    _, p1, d1 = normalize_email(email1)
    _, p2, d2 = normalize_email(email2)
    f1, l1 = split_name(name1)
    f2, l2 = split_name(name2)

    feats = {}

    feats["name_jw"] = jaro_winkler_sim(n1, n2)
    feats["name_tfidf"] = tfidf_similarity(n1, n2)

    feats["prefix_jw"] = jaro_winkler_sim(p1, p2)
    feats["first_jw"] = jaro_winkler_sim(f1, f2)
    feats["last_jw"] = jaro_winkler_sim(l1, l2)

    feats["phone_first"] = phonetic_similarity(f1, f2)
    feats["phone_last"] = phonetic_similarity(l1, l2)

    feats["same_domain"] = int(d1 == d2 and d1 != "")
    feats["firstname_equal"] = int(f1 == f2 and f1 != "")
    feats["lastname_equal"] = int(l1 == l2 and l1 != "")
    feats["initials_equal"] = int(get_initials(name1) == get_initials(name2) and get_initials(name1) != "")

    feats["prefix_has_fl"] = prefix_contains_name(f1, l1, p2)
    feats["prefix_has_fl_rev"] = prefix_contains_name(f2, l2, p1)

    if max(len(n1), len(n2)) > 0:
        feats["len_sim_name"] = 1 - abs(len(n1) - len(n2)) / max(len(n1), len(n2))
    else:
        feats["len_sim_name"] = 0
    if max(len(p1), len(p2)) > 0:
        feats["len_sim_prefix"] = 1 - abs(len(p1) - len(p2)) / max(len(p1), len(p2))
    else:
        feats["len_sim_prefix"] = 0

    feat_order = [
        "name_jw",
        "name_tfidf",
        "prefix_jw",
        "first_jw",
        "last_jw",
        "phone_first",
        "phone_last",
        "same_domain",
        "firstname_equal",
        "lastname_equal",
        "initials_equal",
        "prefix_has_fl",
        "prefix_has_fl_rev",
        "len_sim_name",
        "len_sim_prefix",
    ]

    values = []
    for key in feat_order:
        values.append(feats[key])

    return np.array(values, dtype=float)
