# Prob-AE robustness experiments

Research-oriented implementation of a probabilistic-weighting linear
autoencoder and the SVD-AE, EASE, and Mult-VAE baselines.  The evaluator uses
clean leave-one-out targets, full-catalog ranking, and masks the (possibly
noisy) training interactions.

## Methodological definition

For fixed user features `Z`, Prob-AE minimizes, independently for every item
`j`,

```text
sum_u w[u,j] (R[u,j] - z[u]^T beta[j])^2 + lambda ||beta[j]||^2.
```

The E-step fits a configurable `K`-component Gaussian mixture to log squared
errors on observed interactions.  The posterior probabilities of the
`signal_components` lowest-error components are summed into the confidence
weight.  `K=1` is an unweighted-ridge ablation.  Unobserved entries receive a configurable small
`zero_weight`; setting it to zero makes positive-only implicit reconstruction
degenerate.  The exact M-step is

```text
beta[j] = (Z^T W[j] Z + lambda I)^(-1) Z^T W[j] r[j].
```

By default the solve also enforces a zero self-reconstruction coefficient,
`projection[j]^T beta[j] = 0`, through the analytic equality-constrained ridge
correction.  This prevents the random encoder from leaking item `j` directly
into its own reconstruction error.  It can be disabled with
`zero_diagonal=False` for an ablation.

Arbitrary element-wise weights imply a different system for each item.  A
single shared inverse would *not* be exact weighted ridge.  To keep the itemwise
closed forms practical, the default encoder is a fixed random projection of the
sparse user-item matrix.  It is a computational sketch rather than SVD-based
noise removal; `encoder="svd"` is available as an ablation.

Set `--gmm-components K` and `--signal-components S` (where `1 <= S <= K`)
in the experiment CLI, or pass `gmm_components=K, signal_components=S` to
`ProbAE` in Python.  `--gmm-iterations` controls the inner mixture EM loop.

`SVD_AE` follows Eq. 15 of Hong et al., IJCAI 2024.  With
`R_tilde = Q S V^T`, its reconstruction is evaluated efficiently as
`Q (Q^T R)`, without materializing an item-item matrix.

## Run

Tune ProbAE on the default synthetic dataset with a stratified, validation-only
random search over `K=1,2,3,4`, signal components, rank, ridge strength, zero
weight, and outer EM iterations:

```bash
python -m RecSys.tune_probae --trials 32
```

The selection metric is mean validation Recall@20 across clean, 10% Noisy 1,
and 10% Noisy 0 conditions (NDCG@20 breaks ties).  Test is evaluated only for
the selected configuration.  The bounded search saves trials and a JSON
summary under `RecSys/results/`; it does not establish a global optimum.

With seed 42 and the default 2,000 × 1,000 synthetic dataset, the best of 32
sampled configurations was `K=4`, `signal_components=1`, `rank=64`,
`regularization=100`, `zero_weight=0.05`, and `em_iterations=2`.  Its mean
validation Recall@20 was 0.0933 across the three conditions.  Reproduce this
configuration in the robustness sweep with:

```bash
python -m RecSys.main --format synthetic --models ProbAE \
  --rank 64 --prob-regularization 100 --zero-weight 0.05 \
  --gmm-components 4 --signal-components 1 --em-iterations 2 \
  --output RecSys/results/robustness_dirichlet_tuned.csv
```

Generic model defaults remain unchanged: synthetic-tuned values should not be
silently reused on ML-1M or Yelp.

Synthetic data now uses a **Dirichlet–Categorical generative model**, not
latent dot-product scores.  For each user, draw a distribution over topics;
for each topic, draw a distribution over items.  Then repeatedly sample a
topic and an item, rejecting duplicate user-item pairs.  The defaults are
2,000 users, 1,000 items, 30 positives per user, 24 topics,
`user_concentration=0.3`, and `item_concentration=0.05`.  Smaller
concentrations produce more concentrated preferences.  Adjust them with
`--n-topics`, `--user-concentration`, and `--item-concentration`.

**Provenance:** Existing `RecSys/results/robustness.csv` and its PNG were
generated with the previous dot-product generator.  Rerun the sweep to obtain
results for the new generator, preferably into a new filename such as
`robustness_dirichlet.csv`.

From the repository root (module mode is the supported script entry point):

```bash
python -m RecSys.main \
  --format synthetic \
  --noise-levels 0,0.05,0.10,0.15,0.20 \
  --noise-types noisy_1,noisy_0 \
  --output RecSys/results/robustness_dirichlet.csv
```

Quick smoke run:

```bash
python -m RecSys.main --n-users 100 --n-items 150 \
  --interactions-per-user 10 --noise-levels 0,0.1 \
  --noise-types noisy_1 --rank 8 --em-iterations 2 --vae-epochs 1
```

Real datasets:

```bash
# MovieLens 1M ratings.dat (ratings >= 4 become implicit positives)
python -m RecSys.main --format ml1m --data data/ml-1m/ratings.dat --min-rating 4

# Yelp review JSONL
python -m RecSys.main --format yelp --data data/yelp_academic_dataset_review.json --min-rating 4

# A pre-built scipy sparse matrix
python -m RecSys.main --format npz --data data/train.npz
```

In Jupyter, import `load_dataset`, `inject_noise`, models, and `evaluate` from
the `RecSys` package.  CSV output is replaced atomically after every completed
run, so long grids retain partial results if interrupted.

## Visualize results

```bash
python -m RecSys.plot_robustness \
  --input RecSys/results/robustness_dirichlet.csv \
  --output RecSys/results/robustness_dirichlet.png
```

The figure compares Recall@20, NDCG@20, and training time across noise levels,
with separate columns for `Noisy 1` and `Noisy 0`.  Use `--no-time` to omit the
timing row, or change the output extension to `.pdf` for a vector figure.  In
Jupyter:

```python
from RecSys.plot_robustness import plot_robustness
display(plot_robustness("RecSys/results/robustness_dirichlet.csv"))
```

## Experimental cautions

- Tune each model on the clean validation split, then freeze hyperparameters
  before the robustness sweep.  The supplied sweep evaluates the test split.
- Report both training and inference wall-clock time on identical hardware,
  including the SVD/Gram computation.
- `EASE` necessarily stores a dense item-item matrix and may be impractical for
  very large item catalogs.  Prob-AE and SVD-AE avoid that `I x I` storage.
- `Noisy 1` adds random false-positive edges; `Noisy 0` removes true training
  edges.  The ratio is relative to the original number of training edges.

Run tests with `python -m pytest RecSys/tests -q`.
