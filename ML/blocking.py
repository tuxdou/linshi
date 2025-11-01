# src/blocking.py
from collections import defaultdict
from preprocess import split_name, normalize_email, normalize_name

COMMON_DOMAINS = {
    "gmail.com", "googlemail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "qq.com", "163.com", "proton.me", "protonmail.com", "icloud.com",
    "gmx.com", "gmx.de", "yandex.ru", "yandex.com"
}

def parse_gh_handle(local, domain):
    if domain == "users.noreply.github.com" and "+" in local:
        return local.split("+", 1)[1]
    return ""

def bucket_key(record, key=("domain", "lastname_initial"), ignore_common_domains=True):
    """
    Generate a bucket key for grouping records based on specified fields.

    The `key` argument defines which components to include. Supported values:
      - "domain"           Use the email domain.
      - "lastname_initial" Use the first letter of the last name.
      - "prefix_initial"   Use the first letter of the email prefix (or GitHub handle if applicable).
      - "gh_handle"        Use the GitHub username extracted from a noreply address.

    Common email domains can be ignored by setting `ignore_common_domains=True`.
    """

    _, local, domain = normalize_email(record["email"])
    first, last = split_name(normalize_name(record["name"]))

    parts = []
    for k in key:
        if k == "domain":
            if ignore_common_domains and domain in COMMON_DOMAINS:
                parts.append("")
            else:
                parts.append(domain)
        elif k == "lastname_initial":
            parts.append(last[:1] if last else "")
        elif k == "prefix_initial":
            gh_user = parse_gh_handle(local, domain)
            base = gh_user or local
            parts.append(base[:1] if base else "")
        elif k == "gh_handle":
            parts.append(parse_gh_handle(local, domain))
        else:
            parts.append("")
    return "|".join(parts)

def make_candidates(records, key=("domain", "lastname_initial"),
                    max_bucket=1000, ignore_common_domains=True):
    buckets = defaultdict(list)
    for r in records:
        k = bucket_key(r, key=key, ignore_common_domains=ignore_common_domains)
        buckets[k].append(r)

    for items in buckets.values():
        n = len(items)
        if n < 2 or n > max_bucket:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                yield items[i], items[j]
                

def merge_candidates(records, max_bucket=1000, ignore_common_domains=True):
    seen = set()
    passes = [
        ("domain", "lastname_initial"),
        ("gh_handle",),
        ("domain", "prefix_initial"),
        ("lastname_initial",),
    ]
    for key in passes:
        for a, b in make_candidates(records, key=key,
                                    max_bucket=max_bucket,
                                    ignore_common_domains=ignore_common_domains):
            ea, eb = a["email"].lower(), b["email"].lower()
            pair = tuple(sorted([ea, eb]))
            if pair not in seen:
                seen.add(pair)
                yield a, b
