import numpy as np


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
