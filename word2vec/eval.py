import numpy as np

EPS = 1e-8


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
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + EPS)


def find_closest(
    vector: np.ndarray,
    W_in: np.ndarray,
    id_to_word: list[str],
    topn: int = 1,
    exclude_ids: list[int] | None = None,
) -> list[tuple[str, float]]:
    dot_products = W_in @ vector

    norm_vec = np.linalg.norm(vector)
    norms_W = np.linalg.norm(W_in, axis=1)

    scores = dot_products / ((norms_W * norm_vec) + EPS)

    if exclude_ids is not None:
        scores[exclude_ids] = -np.inf

    topn = min(topn, len(scores))
    unsorted_best_indices = np.argpartition(scores, -topn)[-topn:]

    sorted_indices = unsorted_best_indices[
        np.argsort(scores[unsorted_best_indices])[::-1]
    ]

    results = []
    for idx in sorted_indices:
        word = id_to_word[idx]
        score = scores[idx]
        results.append((word, float(score)))
    return results
