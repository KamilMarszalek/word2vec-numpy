import pytest

from word2vec.config import Word2VecSGNSConfig


def test_config_valid_initialization():
    cfg = Word2VecSGNSConfig(
        window_size=2,
        emb_dim=10,
        epochs=3,
        learning_rate=0.05,
        num_of_neg_samples=5,
        vocab_size=100,
        seed=42,
    )
    assert cfg.emb_dim == 10
    assert cfg.learning_rate == 0.05


@pytest.mark.parametrize(
    "field,value",
    [
        ("window_size", 0),
        ("emb_dim", 0),
        ("epochs", 0),
        ("learning_rate", 0.0),
        ("learning_rate", -1.0),
        ("num_of_neg_samples", 0),
        ("vocab_size", 0),
    ],
)
def test_config_invalid_values_raise(field, value):
    kwargs = dict(
        window_size=2,
        emb_dim=10,
        epochs=3,
        learning_rate=0.05,
        num_of_neg_samples=5,
        vocab_size=100,
        seed=42,
    )
    kwargs[field] = value
    with pytest.raises(ValueError, match="invalid model initialization params"):
        Word2VecSGNSConfig(**kwargs)
