from ML.src.blocking import (
    parse_gh_handle,
    bucket_key,
    make_candidates,
    merge_candidates,
    COMMON_DOMAINS,
)


# ------------------------------------------------
# parse_gh_handle
# ------------------------------------------------

def test_parse_gh_handle_from_noreply():
    result = parse_gh_handle("123+octocat", "users.noreply.github.com")
    assert result == "octocat"


def test_parse_gh_handle_non_github_domain():
    result = parse_gh_handle("user+test", "gmail.com")
    assert result == ""


def test_parse_gh_handle_no_plus():
    result = parse_gh_handle("user", "users.noreply.github.com")
    assert result == ""


# ------------------------------------------------
# bucket_key
# ------------------------------------------------

def test_bucket_key_domain_and_last_initial():
    record = {"name": "Alice Smith", "email": "alice.smith@example.com"}
    key = bucket_key(record, key=("domain", "lastname_initial"))
    assert key == "example.com|s"


def test_bucket_key_ignores_common_domains():
    record = {"name": "Alice Smith", "email": "alice.smith@gmail.com"}
    key = bucket_key(record, key=("domain", "lastname_initial"))
    assert key == "|s"


def test_bucket_key_prefix_initial():
    record = {"name": "Bob Brown", "email": "bob@example.com"}
    key = bucket_key(record, key=("prefix_initial",))
    assert key == "b"


def test_bucket_key_github_handle():
    record = {"name": "Cat Coder", "email": "123+catcoder@users.noreply.github.com"}
    key = bucket_key(record, key=("gh_handle",))
    assert key == "catcoder"


def test_bucket_key_empty_fields():
    record = {"name": "", "email": ""}
    key = bucket_key(record, key=("domain", "lastname_initial"))
    assert key == "|"


# ------------------------------------------------
# make_candidates
# ------------------------------------------------

def test_make_candidates_same_domain():
    records = [
        {"name": "Alice Smith", "email": "alice@example.com"},
        {"name": "Adam Stone", "email": "adam@example.com"},
        {"name": "Brian Lee", "email": "brian@gmail.com"},
    ]
    pairs = list(make_candidates(records, key=("domain",)))
    assert len(pairs) == 1
    a, b = pairs[0]
    assert {a["email"], b["email"]} == {"alice@example.com", "adam@example.com"}


def test_make_candidates_bucket_limit():
    records = [{"name": f"User{i}", "email": f"user{i}@example.com"} for i in range(5)]
    pairs = list(make_candidates(records, key=("domain",), max_bucket=2))
    assert pairs == []


def test_make_candidates_skip_single():
    records = [{"name": "Solo", "email": "solo@example.com"}]
    pairs = list(make_candidates(records))
    assert pairs == []


# ------------------------------------------------
# merge_candidates
# ------------------------------------------------

def test_merge_candidates_combines_and_dedupes():
    records = [
        {"name": "Alice Smith", "email": "alice@example.com"},
        {"name": "Adam Stone", "email": "adam@example.com"},
        {"name": "Amy Snow", "email": "amy@example.com"},
    ]
    pairs = list(merge_candidates(records, max_bucket=10))
    emails = sorted([tuple(sorted([a["email"], b["email"]])) for a, b in pairs])
    assert len(pairs) == 3
    assert emails[0] == ("adam@example.com", "alice@example.com")


def test_merge_candidates_bucket_limit():
    records = [{"name": f"User{i}", "email": f"user{i}@example.com"} for i in range(6)]
    pairs = list(merge_candidates(records, max_bucket=2))
    assert pairs == []


def test_merge_candidates_empty_input():
    pairs = list(merge_candidates([]))
    assert pairs == []
