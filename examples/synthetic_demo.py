import random

from word2vec.data import preprocess
from word2vec.eval import find_closest, get_similarity_between_words
from word2vec.model import Word2VecSGNS, Word2VecSGNSConfig


def create_synthetic_corpus() -> str:
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


if __name__ == "__main__":
    text = create_synthetic_corpus()
    corpus, word_to_id, id_to_word = preprocess(text)

    window_size = 2
    emb_dim = 10
    epochs = 500
    learning_rate = 0.05
    k_neg = 5
    seed = 42

    random.seed(seed)

    algo = Word2VecSGNS(
        Word2VecSGNSConfig(
            window_size,
            emb_dim,
            epochs,
            learning_rate,
            k_neg,
            len(word_to_id),
            seed,
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
