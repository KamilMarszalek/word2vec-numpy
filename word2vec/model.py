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

    def __post_init__(self) -> None:
        if (
            self.window_size < 1
            or self.emb_dim < 1
            or self.epochs < 1
            or self.learning_rate <= 0
            or self.num_of_neg_samples < 1
            or self.vocab_size < 1
        ):
            raise ValueError("invalid model initialization params")


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
            if steps == 0:
                raise ValueError("No training pairs generated")
            mean_loss = epoch_loss_sum / steps
            self.loss_history.append(mean_loss)
            print(f"Epoch {epoch + 1} loss: {mean_loss:.4f}")

        return self.W_in, self.W_out

    def _train_step(self, center_idx: int, context_idx: int) -> float:
        h = self.W_in[center_idx].copy()
        v_pos = self.W_out[context_idx].copy()

        z_pos = h @ v_pos

        prob_pos = sigmoid(z_pos)
        loss_pos = np.logaddexp(0.0, -z_pos)
        error = prob_pos - 1
        grad_h = error * v_pos
        self.W_out[context_idx] -= self.config.learning_rate * error * h

        noise_indices = self._get_k_neg_samples()
        V_neg = self.W_out[noise_indices]

        z_neg = V_neg @ h
        error = sigmoid(z_neg)
        loss_neg = np.logaddexp(0.0, z_neg).sum()
        grad_h += error @ V_neg
        neg_updates = -self.config.learning_rate * error[:, None] * h[None, :]
        np.add.at(self.W_out, noise_indices, neg_updates)

        self.W_in[center_idx] -= self.config.learning_rate * grad_h

        return float(loss_pos + loss_neg)

    def _get_k_neg_samples(self) -> np.ndarray:
        return self.rng.choice(
            self.unigram_probs.shape[0],
            size=self.config.num_of_neg_samples,
            p=self.unigram_probs,
        )
