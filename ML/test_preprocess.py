import pytest

from preprocess import (
    normalize_name,
    split_name,
    normalize_email,
)

# -----------------------
# normalize_name
# -----------------------

@pytest.mark.parametrize("raw, expected", [
    # 去重音、小写、去点、去特殊符号、合并空格
    ("  José.L. Núñez  ", "jose l nunez"),
    ("Jean-Luc   Picard!!!", "jean-luc picard"),     # 保留连字符
    ("Hello! I'm @ChatGPT-5.", "hello im chatgpt-5"),# 去掉标点但保留 -
    ("", ""),
    (None, ""),
])
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


# -----------------------
# split_name
# -----------------------

@pytest.mark.parametrize("full, expected", [
    ("  Madonna  ", ("Madonna", "")),            # 单词 → (词, "")
    ("Jean Luc Picard", ("Jean", "Picard")),     # 多词 → (第一个, 最后一个)
    (" 王 小明 ", ("王", "小明")),                # 中文词间空格
    ("", ("", "")),                              # 空
])
def test_split_name(full, expected):
    assert split_name(full) == expected


# -----------------------
# normalize_email
# -----------------------

@pytest.mark.parametrize("raw, email_norm, local, domain", [
    # 基础：小写、去空格
    ("  SOMEONE@EXAMPLE.COM  ", "someone@example.com", "someone", "example.com"),

    # 兼容 "Name <addr>" 形式
    ("Alice <Alice@Example.org>", "alice@example.org", "alice", "example.org"),

    # Gmail 规则：googlemail 合并、去点、去 +tag
    ("John.Doe+spam@Gmail.COM", "johndoe@gmail.com", "johndoe", "gmail.com"),
    ("x.y.z+promo@googlemail.com", "xyz@gmail.com", "xyz", "gmail.com"),

    # 非 Gmail：保留 + 与点
    ("abk+1233@en.ok.bk", "abk+1233@en.ok.bk", "abk+1233", "en.ok.bk"),

    # 没有 @ 的输入：尽力返回
    ("noatsign", "noatsign", "noatsign", ""),

    # 极端：有人把空格打在 local @ domain 里了
    ("  bob  @  example.com ", "bob@example.com", "bob", "example.com"),

    # Unicode 域名（本实现不做 IDNA 转换，只小写）
    ("dev@例子.公司", "dev@例子.公司", "dev", "例子.公司"),
])
def test_normalize_email(raw, email_norm, local, domain):
    out_email, out_local, out_domain = normalize_email(raw)
    assert out_email == email_norm
    assert out_local == local
    assert out_domain == domain
