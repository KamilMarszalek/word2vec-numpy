import numpy as np


def get_similarity_between_words(
    word1: str,
    word2: str,
    W_in: np.ndarray,
    word_to_id: dict[str, int],
) -> float:
    ind1, ind2 = word_to_id[word1], word_to_id[word2]
    h1, h2 = W_in[ind1], W_in[ind2]

    return cosine_similarity(h1, h2)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def find_closest(
    vector: np.ndarray,
    W_in: np.ndarray,
    id_to_word: list[str],
) -> str:
    best = -1
    best_ind = -1
    for ind, elem in enumerate(W_in):
        score = cosine_similarity(vector, elem)
        if score > best:
            best = score
            best_ind = ind
    return id_to_word[best_ind]
