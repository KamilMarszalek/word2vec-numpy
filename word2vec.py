import random
import string
from dataclasses import dataclass

import numpy as np


@dataclass
class Word2VecConfig:
    window_size: int
    emb_dim: int
    epochs: int
    learning_rate: float
    num_of_neg_samples: int
    vocab_size: int


class Word2VecAlgo:
    def __init__(
        self,
        config: Word2VecConfig,
        corpus: list[int],
        word_to_id: dict[str, int],
        id_to_word: list[str],
    ) -> None:
        self.config = config
        self.corpus = corpus
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.W_in = np.random.uniform(
            -0.5 / self.config.emb_dim,
            0.5 / self.config.emb_dim,
            size=(self.config.vocab_size, self.config.emb_dim),
        )
        self.W_out = np.random.uniform(
            -0.5 / self.config.emb_dim,
            0.5 / self.config.emb_dim,
            size=(self.config.vocab_size, self.config.emb_dim),
        )

    def train(self) -> tuple[np.ndarray, np.ndarray]:
        training_samples = self._generate_training_data()

        for _ in range(self.config.epochs):
            np.random.shuffle(training_samples)
            for center, context in training_samples:
                self._train_step(center, context)

        return self.W_in, self.W_out

    def _generate_training_data(self) -> list[tuple[int, int]]:
        output = []
        for i, center in enumerate(self.corpus):
            start = max(i - self.config.window_size, 0)
            end = min(i + self.config.window_size, len(self.corpus) - 1)
            for j in range(start, end + 1):
                if j == i:
                    continue
                output.append((center, self.corpus[j]))
        return output

    def _train_step(self, center_idx: int, context_idx: int) -> None:
        h = self.W_in[center_idx]
        v_pos = self.W_out[context_idx]

        grad_h = np.zeros_like(h)

        z = h @ v_pos

        prob = sigmoid(z)
        error = prob - 1
        grad_h += error * v_pos
        self.W_out[context_idx] -= self.config.learning_rate * error * h

        for _ in range(self.config.num_of_neg_samples):
            noise_idx = np.random.randint(0, self.W_in.shape[0])
            # OPTIONAL: check if noise_idx == context_idx
            v_neg = self.W_out[noise_idx]
            z = h @ v_neg
            error = sigmoid(z)
            grad_h += error * v_neg
            self.W_out[noise_idx] -= self.config.learning_rate * error * h

        self.W_in[center_idx] -= self.config.learning_rate * grad_h


def sigmoid(x: np.ndarray | int) -> float:
    return 1 / (1 + np.exp(-x))


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


def create_synthetic_corpus() -> string:
    templates = [
        "król rządzi królestwem",
        "królowa rządzi królestwem",
        "król jest mądrym mężczyzną",
        "królowa jest mądrą kobietą",
        "mężczyzna to król",
        "kobieta to królowa",
        "król nosi złotą koronę",
        "królowa nosi złotą koronę",
        "władca to inaczej król",
        "władczyni to inaczej królowa",
        "król kocha swój lud",
        "królowa kocha swój lud",
        "książę to syn króla",
        "księżniczka to córka królowej",
        "mężczyzna silny jak król",
        "kobieta piękna jak królowa",
    ]

    corpus_text = []
    for _ in range(2000):
        corpus_text.append(random.choice(templates))

    return " ".join(corpus_text)


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


if __name__ == "__main__":
    text = create_synthetic_corpus()
    corpus, word_to_id, id_to_word = preprocess(text)

    window_size = 2
    emb_dim = 10
    epochs = 500
    learning_rate = 0.05
    k_neg = 5

    algo = Word2VecAlgo(
        Word2VecConfig(
            window_size,
            emb_dim,
            epochs,
            learning_rate,
            k_neg,
            len(word_to_id),
        ),
        corpus,
        word_to_id,
        id_to_word,
    )

    W_in, W_out = algo.train()

    print(W_in)
    print(W_out)

    print("W_in shape:", W_in.shape)
    print("W_out shape:", W_out.shape)

    print(
        'Similarity between "król" and "królowa"',
        get_similarity_between_words("król", "królowa", W_in, word_to_id),
    )

    print(
        'Similarity between "król" and "mężczyzna"',
        get_similarity_between_words("król", "mężczyzna", W_in, word_to_id),
    )

    target_vector = (
        W_in[word_to_id["król"]]
        - W_in[word_to_id["mężczyzna"]]
        + W_in[word_to_id["kobieta"]]
    )

    print('"król" - "mężczyzna" + "kobieta"')
    print("Expected: królowa")
    print("Result:", find_closest(target_vector, W_in, id_to_word))
