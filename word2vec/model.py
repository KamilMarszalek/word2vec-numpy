from dataclasses import dataclass

import numpy as np

from .data import iter_train_data


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)

    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))

    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)

    return out.item() if out.ndim == 0 else out


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
        self.loss_history: list[float] = []
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
                epoch_loss_sum += loss
                steps += 1
            mean_loss = epoch_loss_sum / steps
            self.loss_history.append(mean_loss)
            print(f"Epoch {epoch + 1} loss: {mean_loss:.4f}")

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
        noise_indices = self._get_k_neg_samples(context_idx, center_idx)
        for noise_idx in noise_indices:
            v_neg = self.W_out[noise_idx]
            z_neg = h @ v_neg
            error = sigmoid(z_neg)
            loss_neg += np.logaddexp(0.0, z_neg)
            grad_h += error * v_neg
            self.W_out[noise_idx] -= self.config.learning_rate * error * h

        self.W_in[center_idx] -= self.config.learning_rate * grad_h
        return loss_pos + loss_neg

    def _get_k_neg_samples(self, context_idx: int, center_idx: int) -> np.ndarray:
        output = []
        while len(output) < self.config.num_of_neg_samples:
            noise_idx = self.rng.choice(
                self.unigram_probs.shape[0], p=self.unigram_probs
            )
            if noise_idx in {context_idx, center_idx}:
                continue
            output.append(noise_idx)
        return np.array(output)
