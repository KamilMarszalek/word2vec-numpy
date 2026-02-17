import string

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


def init_model(vocab_size: int, emb_dim: int) -> tuple[np.ndarray, np.ndarray]:
    W_in = np.random.uniform(
        -0.5 / emb_dim,
        0.5 / emb_dim,
        size=(vocab_size, emb_dim),
    )
    W_out = np.random.uniform(
        -0.5 / emb_dim,
        0.5 / emb_dim,
        size=(vocab_size, emb_dim),
    )
    return W_in, W_out


def sigmoid(x: np.ndarray | int) -> int:
    return 1 / (1 + np.exp(-x))


def train_step(
    center_idx: int,
    context_idx: int,
    W_in: np.ndarray,
    W_out: np.ndarray,
    learning_rate: float,
    k_neg: int,
) -> None:
    h = W_in[center_idx]
    v_pos = W_out[context_idx]

    grad_h = np.zeros_like(h)

    z = h @ v_pos

    prob = sigmoid(z)
    error = prob - 1
    grad_h += error * v_pos
    W_out[context_idx] -= learning_rate * error * h

    for _ in range(k_neg):
        noise_idx = np.random.randint(0, W_in.shape[0])
        # OPTIONAL: check if noise_idx == context_idx
        v_neg = W_out[noise_idx]
        z = h @ v_neg
        error = sigmoid(z)
        grad_h += error * v_neg
        W_out[noise_idx] -= learning_rate * error * h

    W_in[center_idx] -= learning_rate * grad_h


def train(
    corpus: list[int],
    window_size: int,
    emb_dim: int,
    epochs: int,
    learning_rate: float,
    k_neg: int,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    W_in, W_out = init_model(vocab_size, emb_dim)
    training_samples = generate_training_data(corpus, window_size)

    for _ in range(epochs):
        np.random.shuffle(training_samples)
        for center, context in training_samples:
            train_step(center, context, W_in, W_out, learning_rate, k_neg)

    return W_in, W_out


def preprocess(text: str) -> tuple[list[int], dict[str, int], list[str]]:
    lowercase_text = text.lower()
    lowercase_no_punctuation = lowercase_text.translate(
        str.maketrans("", "", string.punctuation)
    )
    words = lowercase_no_punctuation.split()
    id_to_word = list(set(words))
    word_to_id = {}
    for i, word in enumerate(id_to_word):
        word_to_id[word] = i

    corpus = [word_to_id[word] for word in words]

    return corpus, word_to_id, id_to_word


def get_similarity(
    word1: str,
    word2: str,
    W_in: np.ndarray,
    word_to_id: dict[str, int],
) -> float:
    ind1, ind2 = word_to_id[word1], word_to_id[word2]
    h1, h2 = W_in[ind1], W_in[ind2]

    return np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2))


if __name__ == "__main__":
    text = "Król lubi złoto i królowa lubi złoto"
    corpus, word_to_id, id_to_word = preprocess(text)

    window_size = 2
    emb_dim = 5
    epochs = 100
    learning_rate = 0.1
    k_neg = 3

    W_in, W_out = train(
        corpus,
        window_size,
        emb_dim,
        epochs,
        learning_rate,
        k_neg,
        len(word_to_id),
    )

    print(W_in)
    print(W_out)

    print("W_in shape:", W_in.shape)
    print("W_out shape:", W_out.shape)

    print(
        'Similarity between "król" and "królowa"',
        get_similarity("król", "królowa", W_in, word_to_id),
    )

    print(
        'Similarity between "król" and "lubi"',
        get_similarity("król", "lubi", W_in, word_to_id),
    )
