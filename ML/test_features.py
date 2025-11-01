import numpy as np
import pytest

# 如果 features.py 或其依赖（jellyfish / rapidfuzz / scikit-learn）没装好，则自动跳过测试
features = pytest.importorskip("features", reason="无法导入 features.py 或其依赖")

# ---- 小工具：保持和源码里的顺序一致，便于按名称取特征 ----
FEAT_ORDER = [
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

@pytest.fixture
def feat_index():
    return {k: i for i, k in enumerate(FEAT_ORDER)}


# -----------------------------
# 基础相似度函数
# -----------------------------

def test_jaro_winkler_sim_basic():
    s = "john smith"
    assert features.jaro_winkler_sim(s, s) == pytest.approx(1.0, rel=0, abs=1e-7)

    a, b = "alice", "zzz"
    assert features.jaro_winkler_sim(a, b) < 1.0

    # None / 空串不应报错
    assert features.jaro_winkler_sim(None, "") >= 0.0


def test_tfidf_similarity_identical_and_diff():
    s = "engineering"
    # 完全相同 -> 1.0
    assert features.tfidf_similarity(s, s) == pytest.approx(1.0, rel=0, abs=1e-9)
    # 差异较大 -> 明显小一些（阈值留余地，避免偶发抖动）
    assert features.tfidf_similarity("abc", "xyz") < 0.5


def test_phonetic_similarity_simple():
    # 常见近似发音：Smith vs Smyth，通常 >= 0.5
    assert features.phonetic_similarity("Smith", "Smyth") >= 0.5
    # 明显不同的词应为 0.0（两种算法都不匹配）
    assert features.phonetic_similarity("apple", "zebra") == 0.0
    # 空值安全
    assert features.phonetic_similarity("", None) >= 0.0


def test_get_initials():
    assert features.get_initials("John Ronald Tolkien") == "jrt"
    assert features.get_initials("  john   SMITH ") == "js"


def test_prefix_contains_name():
    assert features.prefix_contains_name("john", "smith", "jsmith") == 1
    assert features.prefix_contains_name("john", "doe", "jsmith") == 0
    assert features.prefix_contains_name("", "doe", "jdoe") == 0


# -----------------------------
# build_features 主流程
# -----------------------------

def test_build_features_shape_and_bounds():
    pair1 = ("John Smith", "john.smith@example.com")
    pair2 = ("John Smith", "jsmith@example.com")

    v = features.build_features(pair1, pair2)

    # 形状与类型
    assert isinstance(v, np.ndarray)
    assert v.shape == (15,)
    assert v.dtype == float

    # 范围（大部分是 [0,1] 的相似度或 0/1 标志）
    assert np.all(v >= 0.0)
    assert np.all(v <= 1.0)


def test_build_features_expected_flags(feat_index):
    pair1 = ("John Smith", "john.smith@example.com")
    pair2 = ("John Smith", "jsmith@example.com")
    v = features.build_features(pair1, pair2)

    idx = feat_index

    # 域名相同
    assert v[idx["same_domain"]] == 1.0
    # 名/姓完全一致
    assert v[idx["firstname_equal"]] == 1.0
    assert v[idx["lastname_equal"]] == 1.0
    # 名/姓的 Jaro-Winkler 相似度为 1
    assert v[idx["first_jw"]] == pytest.approx(1.0, abs=1e-9)
    assert v[idx["last_jw"]] == pytest.approx(1.0, abs=1e-9)
    # initials 一致
    assert v[idx["initials_equal"]] == 1.0
    # 全名级别相似度为 1
    assert v[idx["name_jw"]] == pytest.approx(1.0, abs=1e-9)
    assert v[idx["name_tfidf"]] == pytest.approx(1.0, abs=1e-9)
    # 前缀 jsmith vs john.smith 有一定相似度（给个保守下界）
    assert v[idx["prefix_jw"]] > 0.6


def test_build_features_different_person(feat_index):
    pair1 = ("Alice Johnson", "alice.j@example.com")
    pair2 = ("Bob Smith", "b.smith@sample.org")
    v = features.build_features(pair1, pair2)
    idx = feat_index

    # 域名不同
    assert v[idx["same_domain"]] == 0.0
    # 名/姓不相等
    assert v[idx["firstname_equal"]] == 0.0
    assert v[idx["lastname_equal"]] == 0.0
    # 名称级别相似度显著小于 1
    assert v[idx["name_jw"]] < 1.0
    assert v[idx["name_tfidf"]] < 1.0
