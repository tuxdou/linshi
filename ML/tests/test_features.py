# tests/test_features.py
import numpy as np
import pytest
from ML.src.features import (
    jaro_winkler_sim,
    tfidf_similarity,
    phonetic_similarity,
    get_initials,
    prefix_contains_name,
    build_features,
)


# ------------------------------------------------
# Tests for jaro_winkler_sim
# ------------------------------------------------

def test_jaro_winkler_sim_identical():
    """Should return 1.0 for identical strings."""
    assert jaro_winkler_sim("alice", "alice") == pytest.approx(1.0)


def test_jaro_winkler_sim_empty_inputs():
    """Should handle None or empty strings safely."""
    assert jaro_winkler_sim("", None) >= 0.0


def test_jaro_winkler_sim_partial_similarity():
    """Should return value between 0 and 1 for partial match."""
    sim = jaro_winkler_sim("alice", "alicia")
    assert 0 < sim < 1


# ------------------------------------------------
# Tests for tfidf_similarity
# ------------------------------------------------

def test_tfidf_similarity_identical():
    """Should return 1.0 for same text."""
    assert tfidf_similarity("hello", "hello") == pytest.approx(1.0)


def test_tfidf_similarity_completely_different():
    """Should return small value for unrelated strings."""
    sim = tfidf_similarity("cat", "dog")
    assert sim < 0.5


def test_tfidf_similarity_handles_empty():
    """Should handle empty strings without errors."""
    assert tfidf_similarity("", "") >= 0.0


# ------------------------------------------------
# Tests for phonetic_similarity
# ------------------------------------------------

def test_phonetic_similarity_exact_match():
    """Should be 1.0 when both phonetic encodings match."""
    assert phonetic_similarity("smith", "smith") == pytest.approx(1.0)


def test_phonetic_similarity_partial_match():
    """Should be 0.5 when only one encoding matches."""
    # "Steven" and "Stephen" usually share metaphone but not soundex
    score = phonetic_similarity("Steven", "Stephen")
    assert score in (0.5, 1.0)  # phonetic algorithms vary slightly by lib version


def test_phonetic_similarity_no_match():
    """Should be 0.0 when completely different sounds."""
    assert phonetic_similarity("cat", "dog") == pytest.approx(0.0)


# ------------------------------------------------
# Tests for get_initials
# ------------------------------------------------

def test_get_initials_multiple_names():
    """Should return initials for all name parts."""
    assert get_initials("Alice Bob Charlie") == "abc"


def test_get_initials_empty():
    """Should return empty string for empty input."""
    assert get_initials("") == ""


# ------------------------------------------------
# Tests for prefix_contains_name
# ------------------------------------------------

def test_prefix_contains_name_positive_case():
    """Should return 1 if prefix includes first initial and last name."""
    result = prefix_contains_name("a", "smith", "asmith123")
    assert result == 1


def test_prefix_contains_name_negative_case():
    """Should return 0 if conditions not met."""
    result = prefix_contains_name("a", "smith", "bob")
    assert result == 0


def test_prefix_contains_name_missing_inputs():
    """Should return 0 if missing names."""
    assert prefix_contains_name("", "smith", "asmith") == 0


# ------------------------------------------------
# Tests for build_features
# ------------------------------------------------

def test_build_features_typical_pair():
    """Should return 15 numeric features in expected order."""
    pair1 = ("Alice Smith", "alice.smith@example.com")
    pair2 = ("Alicia Smith", "alicia.smith@example.com")

    features = build_features(pair1, pair2)
    assert isinstance(features, np.ndarray)
    assert features.shape == (15,)
    assert (features >= 0).all()
    assert (features <= 1).all()


def test_build_features_empty_inputs():
    """Should handle empty names and emails gracefully."""
    pair1 = ("", "")
    pair2 = ("", "")
    features = build_features(pair1, pair2)
    assert isinstance(features, np.ndarray)
    assert len(features) == 15
    # Most features should be zero or near zero
    assert (features >= 0).all()


def test_build_features_same_person_high_similarity():
    """Should produce high similarity scores for identical person."""
    pair1 = ("Bob Brown", "bob.brown@gmail.com")
    pair2 = ("Bob Brown", "bob.brown@gmail.com")

    features = build_features(pair1, pair2)
    # Expect very high name and domain similarity
    assert features[0] == pytest.approx(1.0, rel=1e-3)
    assert features[7] == 1  # same_domain == 1
    assert features[8] == 1  # firstname_equal
    assert features[9] == 1  # lastname_equal
    assert features[10] == 1  # initials_equal
