import random

from word2vec.config import Word2VecSGNSConfig
from word2vec.data import preprocess
from word2vec.model import Word2VecSGNS


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
    epochs = 50
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

    embeddings = algo.train()

    print("W_in shape:", embeddings.W_in.shape)
    print("W_out shape:", embeddings.W_out.shape)

    print(
        'Similarity between "king" and "queen"',
        embeddings.get_similarity_between_words("king", "queen"),
    )

    print(
        'Similarity between "king" and "man"',
        embeddings.get_similarity_between_words("king", "man"),
    )

    target_vector = (
        embeddings.W_in[word_to_id["king"]]
        - embeddings.W_in[word_to_id["man"]]
        + embeddings.W_in[word_to_id["woman"]]
    )

    print('"king" - "man" + "woman"')
    print("Expected: queen")
    print(
        "Result:",
        embeddings.find_closest(
            target_vector,
            topn=1,
            exclude_ids=[
                word_to_id["king"],
                word_to_id["man"],
                word_to_id["woman"],
            ],
        )[0][0],
    )
