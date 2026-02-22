import numpy as np
import pytest

from word2vec.embeddings import WordEmbeddings, cosine_similarity


def make_simple_embeddings() -> WordEmbeddings:
    # 4 words, 2D embeddings crafted for predictable similarity/analogy behavior
    # man=(1,0), king=(2,0), woman=(0,1), queen=(1,1)
    w_in = np.array(
        [
            [1.0, 0.0],  # man
            [2.0, 0.0],  # king
            [0.0, 1.0],  # woman
            [1.0, 1.0],  # queen
        ],
        dtype=np.float64,
    )
    w_out = w_in.copy()
    word_to_id = {"man": 0, "king": 1, "woman": 2, "queen": 3}
    id_to_word = ["man", "king", "woman", "queen"]
    return WordEmbeddings(w_in, w_out, word_to_id, id_to_word)


def test_cosine_similarity_basic_and_zero_vector_safe():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.0, 1.0])
    vz = np.array([0.0, 0.0])

    assert np.isclose(cosine_similarity(v1, v2), 1.0)
    assert np.isclose(cosine_similarity(v1, v3), 0.0)
    assert np.isfinite(cosine_similarity(v1, vz))


def test_most_similar_excludes_query_word_and_sorts():
    emb = make_simple_embeddings()
    result = emb.most_similar("queen", topn=2)
    words = [w for w, _ in result]

    assert "queen" not in words
    assert len(result) == 2
    # man and king both have strong similarity to queen; exact order can vary slightly,
    # but top result should be one of them.
    assert words[0] in {"man", "king"}


def test_most_similar_unknown_word_raises():
    emb = make_simple_embeddings()
    with pytest.raises(KeyError, match="Word not in vocabulary"):
        emb.most_similar("prince")


def test_find_closest_topn_validation():
    emb = make_simple_embeddings()
    with pytest.raises(ValueError, match="topn must be >= 1"):
        emb._find_closest(np.array([1.0, 0.0]), topn=0)


def test_analogy_returns_expected_word():
    emb = make_simple_embeddings()
    # man : king :: woman : ?  -> queen
    result = emb.analogy("man", "king", "woman", topn=1)
    assert result[0][0] == "queen"


def test_analogy_missing_word_raises():
    emb = make_simple_embeddings()
    with pytest.raises(KeyError, match="Word not in vocabulary"):
        emb.analogy("man", "king", "missing", topn=1)
