from dataclasses import dataclass

import numpy as np
from data import generate_training_data


def sigmoid(x: np.ndarray | int) -> float:
    return 1 / (1 + np.exp(-x))


@dataclass
class Word2VecSGNSConfig:
    window_size: int
    emb_dim: int
    epochs: int
    learning_rate: float
    num_of_neg_samples: int
    vocab_size: int
    seed: int


class Word2VecSGNS:
    def __init__(
        self,
        config: Word2VecSGNSConfig,
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
        training_samples = generate_training_data(
            self.corpus,
            self.config.window_size,
        )

        for _ in range(self.config.epochs):
            np.random.shuffle(training_samples)
            for center, context in training_samples:
                self._train_step(center, context)

        return self.W_in, self.W_out

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
