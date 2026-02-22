import numpy as np
import pytest

from word2vec.config import Word2VecSGNSConfig
from word2vec.embeddings import WordEmbeddings
from word2vec.model import Word2VecSGNS, sigmoid


def make_tiny_training_problem():
    # corpus of word ids: a b a c
    corpus = [0, 1, 0, 2]
    word_to_id = {"a": 0, "b": 1, "c": 2}
    id_to_word = ["a", "b", "c"]
    # any valid probability vector
    unigram_probs = np.array([0.5, 0.25, 0.25], dtype=np.float64)
    cfg = Word2VecSGNSConfig(
        window_size=1,
        emb_dim=4,
        epochs=2,
        learning_rate=0.1,
        num_of_neg_samples=3,
        vocab_size=3,
        seed=123,
    )
    return cfg, corpus, word_to_id, id_to_word, unigram_probs


def test_sigmoid_scalar_and_vector_are_stable():
    assert np.isclose(sigmoid(0.0), 0.5)
    assert np.isclose(sigmoid(1000.0), 1.0)
    assert np.isclose(sigmoid(-1000.0), 0.0)

    x = np.array([-1000.0, 0.0, 1000.0])
    y = sigmoid(x)
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    assert np.all((y >= 0.0) & (y <= 1.0))


def test_get_k_neg_samples_shape_and_range():
    cfg, corpus, word_to_id, id_to_word, probs = make_tiny_training_problem()
    model = Word2VecSGNS(cfg, corpus, word_to_id, id_to_word, probs)
    neg = model._get_k_neg_samples()
    assert neg.shape == (cfg.num_of_neg_samples,)
    assert np.issubdtype(neg.dtype, np.integer)
    assert np.all(neg >= 0)
    assert np.all(neg < cfg.vocab_size)


def test_train_step_returns_finite_loss_and_updates_params():
    cfg, corpus, word_to_id, id_to_word, probs = make_tiny_training_problem()
    model = Word2VecSGNS(cfg, corpus, word_to_id, id_to_word, probs)

    w_in_before = model.W_in.copy()
    w_out_before = model.W_out.copy()

    loss = model._train_step(center_idx=0, context_idx=1)

    assert isinstance(loss, float)
    assert np.isfinite(loss)
    assert not np.allclose(model.W_in, w_in_before)
    assert not np.allclose(model.W_out, w_out_before)


def test_train_returns_embeddings_and_records_epoch_losses(capsys):
    cfg, corpus, word_to_id, id_to_word, probs = make_tiny_training_problem()
    model = Word2VecSGNS(cfg, corpus, word_to_id, id_to_word, probs)

    embeddings = model.train()

    assert isinstance(embeddings, WordEmbeddings)
    assert embeddings.W_in.shape == (cfg.vocab_size, cfg.emb_dim)
    assert embeddings.W_out.shape == (cfg.vocab_size, cfg.emb_dim)
    assert len(model.loss_history) == cfg.epochs
    assert all(np.isfinite(v) for v in model.loss_history)

    captured = capsys.readouterr()
    assert "Epoch 1 loss:" in captured.out


def test_train_raises_when_no_training_pairs_generated():
    cfg = Word2VecSGNSConfig(
        window_size=2,
        emb_dim=4,
        epochs=1,
        learning_rate=0.1,
        num_of_neg_samples=2,
        vocab_size=1,
        seed=0,
    )
    model = Word2VecSGNS(
        cfg,
        corpus=[0],
        word_to_id={"only": 0},
        id_to_word=["only"],
        unigram_probs=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="No training pairs generated"):
        model.train()
