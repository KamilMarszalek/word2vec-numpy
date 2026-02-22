from collections import Counter

import numpy as np
import pytest

from word2vec.data import iter_train_data, preprocess


def test_preprocess_basic_properties():
    text = "Hello, world! Hello test."
    corpus, word_to_id, id_to_word, unigram_probs = preprocess(text)

    # Lowercased, punctuation removed, order preserved by first occurrence
    assert id_to_word == ["hello", "world", "test"]
    assert word_to_id == {"hello": 0, "world": 1, "test": 2}
    assert corpus == [0, 1, 0, 2]

    assert isinstance(unigram_probs, np.ndarray)
    assert unigram_probs.shape == (3,)
    assert np.isclose(unigram_probs.sum(), 1.0)
    # hello occurs twice, should have highest probability after smoothing
    assert unigram_probs[0] > unigram_probs[1]
    assert unigram_probs[0] > unigram_probs[2]


def test_preprocess_empty_after_cleanup_raises():
    with pytest.raises(ValueError, match="empty after preprocessing"):
        preprocess("!!! ... ,,,")


def test_iter_train_data_yields_expected_pairs_as_multiset():
    corpus = [10, 20, 30]
    rng = np.random.default_rng(123)
    out = list(iter_train_data(corpus, window_size=1, rng=rng))

    # Order is randomized, so compare as a multiset of pairs.
    expected = [
        (10, 20),
        (20, 10),
        (20, 30),
        (30, 20),
    ]
    assert Counter(out) == Counter(expected)


def test_iter_train_data_window_two_on_short_corpus_counts_all_contexts():
    corpus = [0, 1, 2, 3]
    rng = np.random.default_rng(0)
    out = list(iter_train_data(corpus, window_size=2, rng=rng))

    # Count expected number of (center, context) pairs
    # i=0 ->2, i=1 ->3, i=2 ->3, i=3 ->2 => total 10
    assert len(out) == 10
    # no self-pairs
    assert all(center != ctx for center, ctx in out)
