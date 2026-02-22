import random

from word2vec.data import preprocess
from word2vec.eval import find_closest, get_similarity_between_words
from word2vec.model import Word2VecSGNS, Word2VecSGNSConfig


def create_synthetic_corpus() -> str:
    templates = [
        "king rules kingdom",
        "queen rules kingdom",
        "king is wise man",
        "queen is wise woman",
        "man is king",
        "woman is queen",
        "king wears golden crown",
        "queen wears golden crown",
        "ruler means king",
        "female ruler means queen",
        "king loves his people",
        "queen loves her people",
        "prince is son of king",
        "princess is daughter of queen",
        "man strong like king",
        "woman beautiful like queen",
    ]

    corpus_text = []
    for _ in range(2000):
        corpus_text.append(random.choice(templates))

    return " ".join(corpus_text)


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    text = create_synthetic_corpus()
    corpus, word_to_id, id_to_word, unigram_probs = preprocess(text)

    window_size = 2
    emb_dim = 10
    epochs = 100
    learning_rate = 0.05
    k_neg = 5

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
        unigram_probs,
    )

    W_in, W_out = algo.train()

    print(W_in)
    print(W_out)

    print("W_in shape:", W_in.shape)
    print("W_out shape:", W_out.shape)

    print(
        'Similarity between "king" and "queen"',
        get_similarity_between_words("king", "queen", W_in, word_to_id),
    )

    print(
        'Similarity between "king" and "man"',
        get_similarity_between_words("king", "man", W_in, word_to_id),
    )

    target_vector = (
        W_in[word_to_id["king"]] - W_in[word_to_id["man"]] + W_in[word_to_id["woman"]]
    )

    print('"king" - "man" + "woman"')
    print("Expected: queen")
    print(
        "Result:",
        find_closest(
            target_vector,
            W_in,
            id_to_word,
            topn=2,
            exclude_ids=[
                word_to_id["king"],
                word_to_id["man"],
                word_to_id["woman"],
            ],
        ),
    )
