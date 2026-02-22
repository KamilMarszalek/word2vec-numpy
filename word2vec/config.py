from dataclasses import dataclass


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
