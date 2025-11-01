# tests/test_blocking.py
from blocking import (
    parse_gh_handle,
    bucket_key,
    make_candidates,
    merge_candidates,
)

# ---------- 小工具 ----------
def mk(id, name, email):
    return {"id": id, "name": name, "email": email}


# ---------- 单元测试 ----------

def test_parse_gh_handle():
    # GitHub noreply 邮箱应该能提取用户名
    assert parse_gh_handle("12345+alice", "users.noreply.github.com") == "alice"
    # 普通邮箱返回空
    assert parse_gh_handle("bob+xyz", "gmail.com") == ""


def test_bucket_key_domain_and_lastname():
    r1 = mk(1, "John Smith", "john@company.com")
    r2 = mk(2, "J. Smíth", "js@company.com")  # 有重音和缩写，结果应相同
    k1 = bucket_key(r1, key=("domain", "lastname_initial"))
    k2 = bucket_key(r2, key=("domain", "lastname_initial"))
    assert k1 == "company.com|s"
    assert k2 == "company.com|s"


def test_bucket_key_ignore_common_domains_default_true():
    r = mk(1, "Alice Zhang", "alice@gmail.com")
    # 默认会忽略常见邮箱域
    k = bucket_key(r, key=("domain", "lastname_initial"))
    assert k == "|z"


def test_bucket_key_keep_common_domain_when_disabled():
    r = mk(1, "Alice Zhang", "alice@gmail.com")
    k = bucket_key(r, key=("domain", "lastname_initial"), ignore_common_domains=False)
    assert k == "gmail.com|z"


def test_bucket_key_prefix_initial_prefers_gh_handle():
    r = mk(1, "Alice", "12345+alice@users.noreply.github.com")
    k = bucket_key(r, key=("prefix_initial",))
    assert k == "a"  # handle 首字母

    r2 = mk(2, "Bob", "bob.s@company.com")
    k2 = bucket_key(r2, key=("prefix_initial",))
    assert k2 == "b"  # 普通邮箱首字母


def test_bucket_key_gh_handle_field():
    r = mk(1, "Alice", "12345+alice@users.noreply.github.com")
    assert bucket_key(r, key=("gh_handle",)) == "alice"

    r2 = mk(2, "Bob", "bob@company.com")
    assert bucket_key(r2, key=("gh_handle",)) == ""


def test_make_candidates_and_max_bucket_limit():
    recs = [
        mk(1, "John Smith", "john@company.com"),
        mk(2, "Jim Smith", "jim@company.com"),
        mk(3, "Jack Smith", "jack@company.com"),
        mk(4, "Alice Zhang", "alice@company.com"),
    ]
    pairs = list(make_candidates(recs, key=("domain", "lastname_initial"), max_bucket=100))
    # Smith 三人 -> C(3,2)=3 对
    assert len(pairs) == 3
    got_ids = sorted([sorted([a["id"], b["id"]]) for a, b in pairs])
    assert got_ids == [[1, 2], [1, 3], [2, 3]]

    # max_bucket 限制生效（桶太大被跳过）
    pairs_small = list(make_candidates(recs, key=("domain", "lastname_initial"), max_bucket=2))
    assert len(pairs_small) == 0


def test_merge_candidates_union_and_dedup():
    recs = [
        mk(1, "John Smith", "12345+john@users.noreply.github.com"),  # gh handle: john
        mk(2, "J. Smith", "jsmith@company.com"),
        mk(3, "Jane Smith", "jane@company.com"),
        mk(4, "Alice Sun", "alice@company.com"),
        mk(5, "Bob Sun", "bob@company.com"),
    ]
    pairs = list(merge_candidates(recs))
    got = {tuple(sorted([a["email"].lower(), b["email"].lower()])) for a, b in pairs}

    # 至少应该包含 Smith 和 Sun 两组配对
    must = {
        tuple(sorted(["jsmith@company.com", "jane@company.com"])),
        tuple(sorted(["alice@company.com", "bob@company.com"])),
    }
    assert must.issubset(got)


def test_merge_candidates_use_email_for_dedup():
    recs = [
        mk(1, "John Smith", "john@company.com"),
        mk(2, "Jim Smith", "jim@company.com"),
        mk(3, "Jack Smith", "jack@company.com"),
    ]
    pairs = list(merge_candidates(recs, ignore_common_domains=False))
    # Smith 三人 -> 3 对；去重后仍应是 3 对
    assert len(pairs) == 3
    uniq = {tuple(sorted([a["email"].lower(), b["email"].lower()])) for a, b in pairs}
    assert len(uniq) == 3
