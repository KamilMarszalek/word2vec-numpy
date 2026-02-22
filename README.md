# Word2Vec (SGNS) in Pure NumPy

A from-scratch implementation of the **core Word2Vec training loop** in **pure NumPy** (no PyTorch / TensorFlow), using the **Skip-Gram with Negative Sampling (SGNS)** variant.

This project focuses on:

- implementing the **forward pass**
- computing the **SGNS loss**
- deriving and applying **gradients**
- updating parameters manually with **SGD**
- building a small but practical **CLI** for training, saving, loading, and querying embeddings

---

## Features

- ✅ Pure NumPy implementation (no ML frameworks)
- ✅ Skip-Gram with Negative Sampling (SGNS)
- ✅ Numerically stable sigmoid implementation
- ✅ Stable loss computation using `np.logaddexp` (softplus form)
- ✅ Negative sampling with **unigram distribution^0.75**
- ✅ Training pairs generated via an **iterator/generator** (memory-friendly)
- ✅ Vectorized gradient computation for all negative samples in a step
- ✅ Correct repeated-index updates via `np.add.at(...)`
- ✅ CLI for:
  - training on a text file
  - saving model artifacts
  - loading and using a trained model
- ✅ Embedding utilities:
  - cosine similarity
  - nearest neighbors (`most_similar`)
  - analogies (`a : b :: c : ?`)

---

## Project Structure

```text
.
├── README.md
├── examples
│   └── synthetic_demo.py
├── main.py
├── pyproject.toml
├── uv.lock
└── word2vec
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── embeddings.py
    └── model.py
```

---

## Installation (using `uv`)

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

### 1. Clone the repository

```bash
git clone "https://github.com/KamilMarszalek/word2vec-numpy"
cd word2vec-numpy
```

### 2. Sync dependencies

```bash
uv sync
```

### 3. Run commands

```bash
uv run python ...
```

---

## Quick Start (Synthetic Demo)

A small synthetic example is provided to verify that training works and to demonstrate similarity / analogy behavior.

```bash
uv run python -m examples.synthetic_demo
```

Typical demo flow:
- creates a synthetic corpus (e.g. `king`, `queen`, `man`, `woman`, etc.)
- trains SGNS embeddings
- prints selected similarities
- runs nearest-neighbor and analogy queries

---

## CLI Usage

The main CLI entry point is `main.py`.

### Show help

```bash
uv run python main.py --help
```

### 1. Train a model on a text file

```bash
uv run python main.py train path/to/dataset.txt
```

Example with custom hyperparameters:

```bash
uv run python main.py train data/corpus.txt \
  --window-size 2 \
  --emb-dim 100 \
  --epochs 20 \
  --learning-rate 0.05 \
  --neg-samples 5 \
  --seed 42
```

### 2. Load and use a trained model

```bash
uv run python main.py use path/to/saved_model_dir --help
```

#### Similarity between two words

```bash
uv run python main.py use models/corpus similarity king queen
```

#### Most similar words

```bash
uv run python main.py use models/corpus most-similar king --topn 10
```

#### Analogy

Example: `man : king :: woman : ?`

```bash
uv run python main.py use models/corpus analogy man king woman --topn 5
```

---

## Saved Model Format

A trained model is stored in a directory (typically named after the dataset), using NumPy arrays and JSON files.

Depending on your current `main.py` version, artifacts may include:

- `W_in.npy` — input embeddings matrix
- `W_out.npy` — output embeddings matrix
- `word_to_id.json` — vocabulary mapping (`word -> index`)
- `id_to_word.json` — reverse mapping (`index -> word`)
- `config.json` — training configuration
- `loss_history.npy` — epoch loss history (optional)
- `training_meta.json` — additional metadata (optional)

---

## Mathematical Background (SGNS)

This section describes the exact objective and gradients implemented in this project.

### 1. Skip-Gram setup

For each position in the corpus, let:

- $`w_c`$ = center word
- $`w_o`$ = observed (positive) context word
- $`w_{n_1}, \dots, w_{n_K}`$ = negative samples

Word2Vec SGNS maintains **two embedding matrices**:

- $`W_{in} \in \mathbb{R}^{|V| \times d}`$ — input embeddings (center words)
- $`W_{out} \in \mathbb{R}^{|V| \times d}`$ — output embeddings (context/target words)

For a training pair:
- $`h = W_{in}[w_c]`$ (center embedding, shape $d$)
- $`v_o = W_{out}[w_o]`$ (positive context embedding)
- $`v_{n_i} = W_{out}[w_{n_i}]`$ (negative context embeddings)

Define scores:
- Positive score: $`z_{pos} = v_o^\top h`$
- Negative scores: $`z_i = v_{n_i}^\top h`$

---

### 2. SGNS loss for one training pair

For one positive pair and $K$ negatives:

```math
\mathcal{L}
=
-\log \sigma(z_{pos})
-
\sum_{i=1}^{K}\log \sigma(-z_i)
```

where $`\sigma(x)`$ is the sigmoid function:

```math
\sigma(x)=\frac{1}{1+e^{-x}}
```

This objective:
- pushes $`z_{pos}`$ **up** (positive pair should match)
- pushes $`z_i`$ **down** for negative samples

---

### 3. Numerically stable loss (softplus / `logaddexp`)

Instead of computing the logs of sigmoids directly, the implementation uses stable identities:

```math
-\log \sigma(z) = \log(1 + e^{-z}) = \text{softplus}(-z)
```

```math
-\log \sigma(-z) = \log(1 + e^{z}) = \text{softplus}(z)
```

In NumPy:

- positive term: `np.logaddexp(0.0, -z_pos)`
- negative term: `np.logaddexp(0.0, z_neg)`

This avoids numerical issues for large positive/negative scores.

---

### 4. Gradients (core of the training loop)

#### Positive term

Let

```math
\mathcal{L}_{pos} = -\log \sigma(z_{pos}), \quad z_{pos}=v_o^\top h
```

Then:

```math
\frac{\partial \mathcal{L}_{pos}}{\partial z_{pos}} = \sigma(z_{pos}) - 1
```

Define:

```math
e_{pos} = \sigma(z_{pos}) - 1
```

Gradients:

```math
\frac{\partial \mathcal{L}_{pos}}{\partial h} = e_{pos} \, v_o
```

```math
\frac{\partial \mathcal{L}_{pos}}{\partial v_o} = e_{pos} \, h
```

---

#### Negative terms

For one negative sample:

```math
\mathcal{L}_{neg,i} = -\log \sigma(-z_i), \quad z_i=v_{n_i}^\top h
```

Then:

```math
\frac{\partial \mathcal{L}_{neg,i}}{\partial z_i} = \sigma(z_i)
```

Define:

```math
e_i = \sigma(z_i)
```

Gradients:

```math
\frac{\partial \mathcal{L}_{neg,i}}{\partial h} = e_i \, v_{n_i}
```

```math
\frac{\partial \mathcal{L}_{neg,i}}{\partial v_{n_i}} = e_i \, h
```

Summed over all negatives:

```math
\frac{\partial \mathcal{L}_{neg}}{\partial h}
=
\sum_{i=1}^{K} e_i v_{n_i}
```

---

### 5. Final gradient for the center embedding

The total gradient w.r.t. the center embedding is:

```math
\frac{\partial \mathcal{L}}{\partial h}
=
\frac{\partial \mathcal{L}_{pos}}{\partial h}
+
\frac{\partial \mathcal{L}_{neg}}{\partial h}
=
(\sigma(z_{pos}) - 1)v_o
+
\sum_{i=1}^{K}\sigma(z_i)v_{n_i}
```

This is exactly what the code accumulates in `grad_h`.

---

### 6. SGD parameter updates

With learning rate $`\eta`$, parameters are updated by standard SGD:

```math
h \leftarrow h - \eta \frac{\partial \mathcal{L}}{\partial h}
```

```math
v_o \leftarrow v_o - \eta \frac{\partial \mathcal{L}}{\partial v_o}
```

```math
v_{n_i} \leftarrow v_{n_i} - \eta \frac{\partial \mathcal{L}_{neg,i}}{\partial v_{n_i}}
```

In implementation terms:
- positive context row in `W_out` gets one update
- negative context rows in `W_out` get $K$ updates
- center row in `W_in` gets one accumulated update

---

### 7. Vectorized negative-sample gradients (implemented optimization)

Instead of looping through negative samples one by one, the project computes them in a vectorized way.

Let:

- $`V_{neg} \in \mathbb{R}^{K \times d}`$ be the stacked negative output embeddings
- $`h \in \mathbb{R}^{d}`$

Then:

```math
z_{neg} = V_{neg}h \in \mathbb{R}^{K}
```

```math
e_{neg} = \sigma(z_{neg}) \in \mathbb{R}^{K}
```

The negative contribution to the center gradient becomes:

```math
\frac{\partial \mathcal{L}_{neg}}{\partial h}
=
e_{neg}^{\top} V_{neg}
```

which is implemented as:

```python
grad_h += err_neg @ V_neg
```

The negative updates for output embeddings are computed as an outer-product-like batch:

```python
neg_updates = -lr * err_neg[:, None] * h[None, :]
```

Because negative samples are drawn **with replacement**, duplicate indices can occur.  
To accumulate repeated-row updates correctly, the implementation uses:

```python
np.add.at(self.W_out, noise_indices, neg_updates)
```

This is an important detail—simple advanced indexing assignment would be incorrect in the presence of duplicate indices.

---

### 8. Negative sampling distribution (smoothed unigram, power 0.75)

Negative words are sampled from a smoothed unigram distribution:

```math
P(w) \propto \text{count}(w)^{0.75}
```

This is the standard choice used in Word2Vec and usually works better than:
- raw unigram distribution ($`\alpha = 1`$)
- uniform sampling ($`\alpha = 0`$)

It reduces over-sampling of very frequent words while still preserving corpus frequency information.

---

## Numerical Stability Notes

### Stable sigmoid
The sigmoid is implemented in a numerically stable piecewise form:
- one branch for $`x \ge 0`$
- another branch for $`x < 0`$

This avoids overflow in expressions like `exp(1000)`.

### Stable log-sigmoid terms
Using `np.logaddexp` avoids unstable patterns such as:
- `-np.log(sigmoid(z))`
- `-np.log(sigmoid(-z))`

which can underflow / overflow for large values.

---

## Limitations

This project focuses on the **core SGNS training loop** and educational clarity.

Potential improvements:
- subsampling of very frequent words
- dynamic window size (as in the original Word2Vec)
- mini-batch training
- more aggressive vectorization / performance optimizations
- richer evaluation scripts and benchmarks
- larger dataset support / streaming pipelines
