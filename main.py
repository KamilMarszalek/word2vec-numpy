import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from word2vec.config import Word2VecSGNSConfig
from word2vec.data import preprocess
from word2vec.embeddings import WordEmbeddings
from word2vec.model import Word2VecSGNS


def _json_dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_model_artifacts(
    out_dir: Path,
    embeddings: WordEmbeddings,
    config: Word2VecSGNSConfig,
    loss_history: list[float],
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "W_in.npy", embeddings.W_in)
    np.save(out_dir / "W_out.npy", embeddings.W_out)
    np.save(out_dir / "loss_history.npy", np.asarray(loss_history, dtype=np.float64))

    _json_dump(out_dir / "word_to_id.json", embeddings.word_to_id)
    _json_dump(out_dir / "id_to_word.json", embeddings.id_to_word)
    _json_dump(out_dir / "config.json", asdict(config))

    training_meta = {
        "vocab_size": len(embeddings.id_to_word),
        "emb_dim": int(embeddings.W_in.shape[1]),
        "num_epochs": int(config.epochs),
    }
    _json_dump(out_dir / "training_meta.json", training_meta)


def load_embeddings_from_dir(model_dir: Path) -> WordEmbeddings:
    """
    Load WordEmbeddings object from saved artifacts directory.
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    w_in_path = model_dir / "W_in.npy"
    w_out_path = model_dir / "W_out.npy"
    word_to_id_path = model_dir / "word_to_id.json"
    id_to_word_path = model_dir / "id_to_word.json"

    required = [w_in_path, w_out_path, word_to_id_path, id_to_word_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")

    W_in = np.load(w_in_path)
    W_out = np.load(w_out_path)
    word_to_id = _json_load(word_to_id_path)
    id_to_word = _json_load(id_to_word_path)

    if not isinstance(word_to_id, dict):
        raise ValueError("word_to_id.json must contain a JSON object")
    if not isinstance(id_to_word, list):
        raise ValueError("id_to_word.json must contain a JSON array")

    if W_in.ndim != 2 or W_out.ndim != 2:
        raise ValueError("W_in and W_out must be 2D matrices")
    if W_in.shape != W_out.shape:
        raise ValueError(
            f"W_in and W_out shape mismatch: {W_in.shape} vs {W_out.shape}"
        )
    if len(id_to_word) != W_in.shape[0]:
        raise ValueError("id_to_word length does not match embedding vocab size")

    return WordEmbeddings(
        W_in=W_in,
        W_out=W_out,
        word_to_id=word_to_id,
        id_to_word=id_to_word,
    )


def cmd_train(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not dataset_path.is_file():
        raise ValueError(f"Dataset path is not a file: {dataset_path}")

    text = dataset_path.read_text(encoding=args.encoding)

    corpus, word_to_id, id_to_word, unigram_probs = preprocess(text)

    config = Word2VecSGNSConfig(
        window_size=args.window_size,
        emb_dim=args.emb_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_of_neg_samples=args.neg_samples,
        vocab_size=len(word_to_id),
        seed=args.seed,
    )

    model = Word2VecSGNS(
        config=config,
        corpus=corpus,
        word_to_id=word_to_id,
        id_to_word=id_to_word,
        unigram_probs=unigram_probs,
    )

    embeddings = model.train()

    if args.output_dir is not None:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = (Path.cwd() / dataset_path.stem).resolve()

    save_model_artifacts(
        out_dir=out_dir,
        embeddings=embeddings,
        config=config,
        loss_history=model.loss_history,
    )

    print("\nTraining finished.")
    print(f"Dataset: {dataset_path}")
    print(f"Saved model to: {out_dir}")
    print(f"Vocab size: {len(id_to_word)}")
    print(f"Embedding dim: {embeddings.W_in.shape[1]}")
    if model.loss_history:
        print(f"Final epoch loss: {model.loss_history[-1]:.6f}")


def cmd_use(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir).expanduser().resolve()
    embeddings = load_embeddings_from_dir(model_dir)

    if args.command == "similarity":
        score = embeddings.get_similarity_between_words(args.word1, args.word2)
        print(f"similarity({args.word1}, {args.word2}) = {score:.6f}")
        return

    if args.command == "most-similar":
        results = embeddings.most_similar(args.word, topn=args.topn)
        print(f"Most similar to '{args.word}' (top {args.topn}):")
        for i, (word, score) in enumerate(results, start=1):
            print(f"{i:>2}. {word:<20} {score:.6f}")
        return

    if args.command == "analogy":
        results = embeddings.analogy(args.a, args.b, args.c, topn=args.topn)
        print(f"Analogy: {args.a} : {args.b} :: {args.c} : ? (top {args.topn})")
        for i, (word, score) in enumerate(results, start=1):
            print(f"{i:>2}. {word:<20} {score:.6f}")
        return

    raise ValueError(f"Unknown use subcommand: {args.command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="word2vec-cli",
        description="Train and use a pure NumPy Word2Vec SGNS model.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Train model on a text dataset file"
    )
    train_parser.add_argument("dataset", help="Path to text file dataset")
    train_parser.add_argument(
        "--encoding", default="utf-8", help="File encoding (default: utf-8)"
    )
    train_parser.add_argument(
        "--output-dir", default=None, help="Directory to save model artifacts"
    )
    train_parser.add_argument("--window-size", type=int, default=2)
    train_parser.add_argument("--emb-dim", type=int, default=50)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=0.05)
    train_parser.add_argument("--neg-samples", type=int, default=5)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(func=cmd_train)

    use_parser = subparsers.add_parser("use", help="Load trained model and use it")
    use_parser.add_argument("model_dir", help="Path to saved model directory")

    use_sub = use_parser.add_subparsers(dest="command", required=True)

    sim_parser = use_sub.add_parser(
        "similarity", help="Compute similarity between two words"
    )
    sim_parser.add_argument("word1")
    sim_parser.add_argument("word2")
    sim_parser.set_defaults(func=cmd_use)

    ms_parser = use_sub.add_parser("most-similar", help="Find most similar words")
    ms_parser.add_argument("word")
    ms_parser.add_argument("--topn", type=int, default=10)
    ms_parser.set_defaults(func=cmd_use)

    ana_parser = use_sub.add_parser("analogy", help="Solve analogy a:b::c:?")
    ana_parser.add_argument("a")
    ana_parser.add_argument("b")
    ana_parser.add_argument("c")
    ana_parser.add_argument("--topn", type=int, default=5)
    ana_parser.set_defaults(func=cmd_use)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
