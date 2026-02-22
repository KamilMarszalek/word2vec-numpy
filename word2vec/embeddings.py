import numpy as np

EPS = 1e-8


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + EPS)


class WordEmbeddings:
    def __init__(
        self,
        W_in: np.ndarray,
        W_out: np.ndarray,
        word_to_id: dict[str, int],
        id_to_word: list[str],
    ):
        self.W_in = W_in
        self.W_out = W_out
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word

    def get_similarity_between_words(
        self,
        word1: str,
        word2: str,
    ) -> float:
        ind1, ind2 = self.word_to_id[word1], self.word_to_id[word2]
        h1, h2 = self.W_in[ind1], self.W_in[ind2]

        return cosine_similarity(h1, h2)

    def _find_closest(
        self,
        vector: np.ndarray,
        topn: int = 1,
        exclude_ids: list[int] | None = None,
    ) -> list[tuple[str, float]]:
        if topn < 1:
            raise ValueError("topn must be >= 1")
        dot_products = self.W_in @ vector

        norm_vec = np.linalg.norm(vector)
        norms_W = np.linalg.norm(self.W_in, axis=1)

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
            word = self.id_to_word[idx]
            score = scores[idx]
            results.append((word, float(score)))
        return results

    def most_similar(
        self,
        word: str,
        topn: int = 10,
    ) -> list[tuple[str, float]]:
        if word not in self.word_to_id:
            raise KeyError(f"Word not in vocabulary: {word}")
        idx = self.word_to_id[word]
        return self._find_closest(
            self.W_in[idx],
            topn=topn,
            exclude_ids=[idx],
        )

    def analogy(
        self,
        a: str,
        b: str,
        c: str,
        topn: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Solve analogy: a : b :: c : ?  => vec(b) - vec(a) + vec(c)
        Example: man : king :: woman : ? -> queen
        """
        for word in (a, b, c):
            if word not in self.word_to_id:
                raise KeyError(f"Word not in vocabulary: {word}")

        a_idx = self.word_to_id[a]
        b_idx = self.word_to_id[b]
        c_idx = self.word_to_id[c]

        target = self.W_in[b_idx] - self.W_in[a_idx] + self.W_in[c_idx]
        return self._find_closest(
            target,
            topn=topn,
            exclude_ids=[a_idx, b_idx, c_idx],
        )
