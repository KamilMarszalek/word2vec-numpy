import string
from collections import Counter
from collections.abc import Iterator

import numpy as np


def preprocess(
    text: str,
) -> tuple[
    list[int],
    dict[str, int],
    list[str],
    np.ndarray,
]:
    lowercase_text = text.lower()
    lowercase_no_punctuation = lowercase_text.translate(
        str.maketrans("", "", string.punctuation)
    )
    words = lowercase_no_punctuation.split()
    if not words:
        raise ValueError("Input text is empty after preprocessing")

    words_counter = Counter(words)

    id_to_word = list(dict.fromkeys(words))
    freqs = []
    word_to_id = {}
    for i, word in enumerate(id_to_word):
        word_to_id[word] = i
        freqs.append(words_counter[word])

    freqs_arr = np.array(freqs)
    unigram_probs = freqs_arr**0.75 / np.sum(freqs_arr**0.75)

    corpus = [word_to_id[word] for word in words]

    return corpus, word_to_id, id_to_word, unigram_probs


def iter_train_data(
    corpus: list[int],
    window_size: int,
    rng: np.random.Generator,
) -> Iterator[tuple[int, int]]:
    n = len(corpus)
    for i in rng.permutation(n):
        center = corpus[i]
        start = max(i - window_size, 0)
        end = min(i + window_size, n - 1)
        ctx_positions = [pos for pos in range(start, end + 1) if pos != i]
        rng.shuffle(ctx_positions)

        for j in ctx_positions:
            yield center, corpus[j]
