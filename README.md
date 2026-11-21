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

Create an environment and install the package from the repository root.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Run the following examples from the repository root.

Tune ProbAE on the default synthetic dataset with a stratified, validation-only
random search over `K=1,2,3,4`, signal components, rank, ridge strength, zero
weight, and outer EM iterations:

```bash
python -m RecSys.tune_probae --trials 32
```

The selection metric is mean validation Recall@20 across clean, 10% Noisy 1,
and 10% Noisy 0 conditions (NDCG@20 breaks ties).  Test is evaluated only for
the selected configuration.  The bounded search saves trials and a JSON
summary under `results/`; it does not establish a global optimum.

With seed 42 and the default 2,000 × 1,000 synthetic dataset, the best of 32
sampled configurations was `K=4`, `signal_components=1`, `rank=64`,
`regularization=100`, `zero_weight=0.05`, and `em_iterations=2`.  Its mean
validation Recall@20 was 0.0933 across the three conditions.  Reproduce this
configuration in the robustness sweep with:

```bash
python -m RecSys.main --format synthetic --models ProbAE \
  --rank 64 --prob-regularization 100 --zero-weight 0.05 \
  --gmm-components 4 --signal-components 1 --em-iterations 2 \
  --output results/robustness_dirichlet_tuned.csv
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

**Provenance:** Older local copies of `results/robustness.csv` and its PNG were
generated with the previous dot-product generator.  Rerun the sweep to obtain
results for the new generator, preferably into a new filename such as
`robustness_dirichlet.csv`.

Run the robustness sweep with the synthetic dataset.

```bash
python -m RecSys.main \
  --format synthetic \
  --noise-levels 0,0.05,0.10,0.15,0.20 \
  --noise-types noisy_1,noisy_0 \
  --output results/robustness_dirichlet.csv
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
`--top-k` sets the metric cutoff and the CSV column suffix, for example
`recall_at_3` and `ndcg_at_3` for `--top-k 3`.

## Visualize results

```bash
python -m RecSys.plot_robustness \
  --input results/robustness_dirichlet.csv \
  --output results/robustness_dirichlet.png
```

The figure compares Recall and NDCG at the cutoff recorded in the CSV alongside
training time, with separate columns for `Noisy 1` and `Noisy 0`. Use
`--no-time` to omit the
timing row, or change the output extension to `.pdf` for a vector figure.  In
Jupyter:

```python
from RecSys.plot_robustness import plot_robustness
display(plot_robustness("results/robustness_dirichlet.csv"))
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

Install `pytest` in the project environment and run `python -m pytest -q`.

## KT-2 controlled experiments

The new `ResearchAE` in `kt2_models.py` is an explicitly target-excluded
linear decoder. It removes the target from both the random projection and
mean genre profile for each itemwise training system. This implementation
is separate from the original `ProbAE`; results for the two implementations
must not be silently combined.

Implemented controls:

- unit weights, hard trimming, Gaussian, Student-t (df=3), Cauchy (df=1);
- two-component mixtures of `log(error**2 + 1e-8)`, 40 inner updates,
  minimum posterior weight 0.02; lower-location component designated signal
  as a hypothesis, not an identified truth;
- genres × weights, shuffled genre rows, matched total feature dimension,
  genre-cosine baseline, and nested histories of 5/10/20 events;
- quantile versus two random mixture starts, 0/1/2/5/10 outer iterations;
- three-fold error control: auxiliary models exclude held-out events from
  features and loss, including their potential contribution as negatives;
- per-popularity and per-history weight distributions and ranking metrics.

The decoder uses weighted squared loss and ridge regularization. The t-mixture
latent precision only updates mixture parameters; decoder weights remain
component-membership posteriors. The outer loop is not claimed to be EM for
a single joint likelihood. Heavy tails need not improve either detection or ranking.

### Reproduce the pilot

The saved equal-budget pilot in [`output/kt2/pilot_v2/`](output/kt2/pilot_v2/)
contains 426 evaluations across the three seeds listed in its manifest. To
reproduce it, run the script in the project environment.

```bash
sh scripts/run_kt2.sh /path/to/ml-1m
```

The script creates a fresh directory under `results/kt2/` by default. Pass a new
or empty directory as the second argument to choose the output location. The
script uses `python3` from `PATH`. Set `RECSYS_PYTHON` to select another interpreter.

The pilot selects the 600 most popular positive-rated movies and 240 randomly
sampled eligible users (selection seed 1729). Eligibility requires 22 positive
ratings after the item filter. Thus the pilot excludes the global item tail
and very short natural histories; its groups are relative to this subset.
It is not a full MovieLens-1M benchmark. Histories 5/10/20 simulate sparse
inputs on a fixed eligible population, not naturally cold-start users.

Run the full eligible MovieLens cohort with the following commands.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m RecSys.kt2_experiments \
  --data /path/to/ml-1m --output results/kt2/full \
  --seeds 42 43 44 45 46
python -m RecSys.plot_kt2 --input results/kt2/full
```

Full runs are computationally expensive: every outer update solves an
item-specific system; cross-fitting fits additional models. Partial tables
are written after every completed evaluation. Each run starts from scratch and
requires a new or empty output directory. `manifest.json` records sample IDs and
arguments.

### Evaluation protocol

Ratings >=4; random per-user leave-one-out validation and test; noise injected
only after splitting. Additions cannot overlap validation/test. Train-time
features use only available training interactions. The evaluator's separate
`seen` mask is original train UNION observed noisy train. Consequently deleted
Noisy 0 events remain excluded and added Noisy 1 events are also masked.
For shortened histories, the original train remains masked, keeping a common
candidate set across history lengths and models. These evaluation-only masks
are never passed as model input. No validation-to-train refit is performed.

For each seed and variant, hyperparameters are selected on CLEAN validation
NDCG@20 and frozen across noise levels. Each ResearchAE variant gets three
trials per lambda, using zero weights 0.005/0.01/0.05. For trimming these are
paired with trim fractions 0.05/0.10/0.20 (a bounded search, not a Cartesian
grid). Thus all five weighting methods receive the same trial count. Stability and cross-fit diagnostics
inherit that variant's chosen hyperparameters; they are not additional searches
on test. The old `tune_probae` continues to use its separately documented
synthetic three-condition validation objective.

Detection positive labels mean artificially added events, not verified real
noise. AP uses threshold-grouped ties; its random-reference prevalence is
approximately rho/(1+rho), not rho. At 0% additions or for Noisy 0, detection
is undefined and omitted. Unit weights give AUROC=0.5 and AP=prevalence.

Figures contain means and observed min/max over seeds, NOT 95% confidence
intervals. `paired_deltas.csv` contains matched-seed differences versus unit
weights. The pilot supports exploratory conclusions only. Five-seed full-data
experiments and paired user-level/hierarchical bootstrap remain final-report work.
No transfer claim is made beyond this target-excluded linear decoder; EASE and
SVD-AE are external performance baselines, not weight-transfer experiments.
LightFM and Mult-VAE expansion are deferred; original Mult-VAE remains available
through `main.py`.

### Report artifacts

Each run directory contains raw metrics, validation trials, a manifest,
grouped metrics and weight arrays. The script also generates summaries and
PDF figures. Regenerate charts from saved metrics with
`python -m RecSys.plot_kt2 --input <run-directory>`.

### Literature and scope

- [ADT, Wang et al., WSDM 2021](https://arxiv.org/abs/2006.04153): truncates or
  reweights high-loss observations during training. Our fixed-quantile trimming
  control is not a reproduction of the complete ADT procedure.
- [DeCA, Wang et al., WWW 2022](https://arxiv.org/abs/2105.09605): uses agreement
  between recommendation models. Our single-decoder mixture does not use this
  independent source of evidence.
- [DCF, He et al., KDD 2024](https://arxiv.org/abs/2405.11272): examines loss
  trajectories and label correction to distinguish difficult samples from noise.
  A single large reconstruction error does not resolve this distinction.
- [Peel and McLachlan, 2000](https://people.smp.uq.edu.au/GeoffMcLachlan/pm_sc00.pdf):
  t-mixtures reduce atypical observations' influence on component parameter
  estimation. This motivates the mixture family, but does not establish that
  its lower-location component represents genuine recommendation interests.
