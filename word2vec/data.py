import string
from collections import Counter

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


def generate_training_data(
    corpus: list[int], window_size: int
) -> list[tuple[int, int]]:
    output = []
    for i, center in enumerate(corpus):
        start = max(i - window_size, 0)
        end = min(i + window_size, len(corpus) - 1)
        for j in range(start, end + 1):
            if j == i:
                continue
            output.append((center, corpus[j]))
    return output
