from pathlib import Path

import numpy as np
import pytest

import main as main_module
from word2vec.config import Word2VecSGNSConfig
from word2vec.embeddings import WordEmbeddings


def make_embeddings() -> WordEmbeddings:
    W_in = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    W_out = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float64)
    word_to_id = {"foo": 0, "bar": 1}
    id_to_word = ["foo", "bar"]
    return WordEmbeddings(W_in=W_in, W_out=W_out, word_to_id=word_to_id, id_to_word=id_to_word)


def make_config() -> Word2VecSGNSConfig:
    return Word2VecSGNSConfig(
        window_size=2,
        emb_dim=2,
        epochs=3,
        learning_rate=0.05,
        num_of_neg_samples=5,
        vocab_size=2,
        seed=42,
    )


def test_save_and_load_model_artifacts_roundtrip(tmp_path: Path):
    out_dir = tmp_path / "model_artifacts"
    emb = make_embeddings()
    cfg = make_config()
    loss_history = [1.23, 0.98, 0.76]

    main_module.save_model_artifacts(out_dir, emb, cfg, loss_history)

    loaded = main_module.load_embeddings_from_dir(out_dir)
    assert np.allclose(loaded.W_in, emb.W_in)
    assert np.allclose(loaded.W_out, emb.W_out)
    assert loaded.word_to_id == emb.word_to_id
    assert loaded.id_to_word == emb.id_to_word

    assert (out_dir / "config.json").exists()
    assert (out_dir / "training_meta.json").exists()
    assert (out_dir / "loss_history.npy").exists()


def test_load_embeddings_missing_directory_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="Model directory does not exist"):
        main_module.load_embeddings_from_dir(missing)


def test_load_embeddings_missing_required_files_raises(tmp_path: Path):
    out_dir = tmp_path / "model"
    out_dir.mkdir()
    # create only one file, leave others missing
    np.save(out_dir / "W_in.npy", np.zeros((2, 2)))

    with pytest.raises(FileNotFoundError, match="Missing model files"):
        main_module.load_embeddings_from_dir(out_dir)


def test_load_embeddings_shape_validation_raises(tmp_path: Path):
    out_dir = tmp_path / "model"
    out_dir.mkdir()

    np.save(out_dir / "W_in.npy", np.zeros((2, 2)))
    np.save(out_dir / "W_out.npy", np.zeros((2, 3)))  # shape mismatch
    (out_dir / "word_to_id.json").write_text('{"foo": 0, "bar": 1}', encoding="utf-8")
    (out_dir / "id_to_word.json").write_text('["foo", "bar"]', encoding="utf-8")

    with pytest.raises(ValueError, match="shape mismatch"):
        main_module.load_embeddings_from_dir(out_dir)


def test_build_parser_parses_train_and_use_subcommands():
    parser = main_module.build_parser()

    args_train = parser.parse_args(["train", "dataset.txt"])
    assert args_train.mode == "train"
    assert args_train.dataset == "dataset.txt"
    assert callable(args_train.func)

    args_use = parser.parse_args(["use", "model_dir", "similarity", "king", "queen"])
    assert args_use.mode == "use"
    assert args_use.command == "similarity"
    assert args_use.word1 == "king"
    assert args_use.word2 == "queen"
    assert callable(args_use.func)
