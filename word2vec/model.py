from dataclasses import dataclass

import numpy as np

from .data import iter_train_data


def sigmoid(x: np.ndarray | int) -> float | np.ndarray:
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
        unigram_probs: np.ndarray,
    ) -> None:
        self.config = config
        self.rng = np.random.default_rng(self.config.seed)
        self.corpus = corpus
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.loss_history = []
        self.W_in = self.rng.uniform(
            -0.5 / self.config.emb_dim,
            0.5 / self.config.emb_dim,
            size=(self.config.vocab_size, self.config.emb_dim),
        )
        self.W_out = self.rng.uniform(
            -0.5 / self.config.emb_dim,
            0.5 / self.config.emb_dim,
            size=(self.config.vocab_size, self.config.emb_dim),
        )
        self.unigram_probs = unigram_probs

    def train(self) -> tuple[np.ndarray, np.ndarray]:
        for epoch in range(self.config.epochs):
            epoch_loss_sum = 0.0
            steps = 0
            for center, context in iter_train_data(
                self.corpus,
                self.config.window_size,
                self.rng,
            ):
                loss = self._train_step(center, context)
                self.loss_history.append(loss)
                epoch_loss_sum += loss
                steps += 1
            print(f"Epoch {epoch + 1} loss: {epoch_loss_sum / steps:.4f}")

        return self.W_in, self.W_out

    def _train_step(self, center_idx: int, context_idx: int) -> float:
        h = self.W_in[center_idx]
        v_pos = self.W_out[context_idx]

        grad_h = np.zeros_like(h)

        z_pos = h @ v_pos

        prob = sigmoid(z_pos)
        loss_pos = np.logaddexp(0.0, -z_pos)
        error = prob - 1
        grad_h += error * v_pos
        self.W_out[context_idx] -= self.config.learning_rate * error * h

        loss_neg = 0.0
        for _ in range(self.config.num_of_neg_samples):
            noise_idx = self.rng.choice(
                self.unigram_probs.shape[0], p=self.unigram_probs
            )
            v_neg = self.W_out[noise_idx]
            z_neg = h @ v_neg
            error = sigmoid(z_neg)
            loss_neg += np.logaddexp(0.0, z_neg)
            grad_h += error * v_neg
            self.W_out[noise_idx] -= self.config.learning_rate * error * h

        self.W_in[center_idx] -= self.config.learning_rate * grad_h
        return loss_pos + loss_neg
