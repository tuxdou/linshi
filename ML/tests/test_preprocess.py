from ML.src.preprocess import normalize_name, split_name, normalize_email

# Tests for normalize_name

def test_normalize_name_basic():
    result = normalize_name("  Dr. John Smith ")
    assert result == "dr john smith"


def test_normalize_name_with_accents():
    result = normalize_name("José María-López")
    assert result == "jose maria-lopez"


def test_normalize_name_empty():
    assert normalize_name("") == ""
    assert normalize_name(None) == ""


def test_normalize_name_with_symbols():
    result = normalize_name("A_B*C@D!")
    assert result == "a_bcd"

# Tests for split_name

def test_split_name_two_words():
    first, last = split_name("Alice Smith")
    assert first == "Alice"
    assert last == "Smith"


def test_split_name_one_word():
    first, last = split_name("Alice")
    assert first == "Alice"
    assert last == ""


def test_split_name_with_spaces():
    first, last = split_name("  Alice   B.  Smith  ")
    assert first == "Alice"
    assert last == "Smith"


def test_split_name_empty():
    first, last = split_name("")
    assert (first, last) == ("", "")


# Tests for normalize_email

def test_normalize_email_standard():
    full, local, domain = normalize_email("John.Smith@Example.COM")
    assert full == "john.smith@example.com"
    assert local == "john.smith"
    assert domain == "example.com"


def test_normalize_email_with_display_name():
    full, local, domain = normalize_email("John Smith <john.smith@example.com>")
    assert full == "john.smith@example.com"
    assert local == "john.smith"
    assert domain == "example.com"


def test_normalize_email_gmail_rules():
    full, local, domain = normalize_email("user.name+spam@gmail.com")
    assert full == "username@gmail.com"
    assert local == "username"
    assert domain == "gmail.com"


def test_normalize_email_googlemail_alias():
    full, local, domain = normalize_email("user.name@googlemail.com")
    assert full == "username@gmail.com"
    assert local == "username"
    assert domain == "gmail.com"


def test_normalize_email_missing_at():
    full, local, domain = normalize_email("invalidemail")
    assert (full, local, domain) == ("invalidemail", "invalidemail", "")


def test_normalize_email_empty_or_none():
    assert normalize_email("") == ("", "", "")
    assert normalize_email(None) == ("", "", "")


def test_normalize_email_multiple_at():
    full, local, domain = normalize_email("a@b@c.com")
    assert full == "a@b@c.com"
    assert local == "a"
    assert domain == "b@c.com"
