# E09 — Representational velocity spectrum

PCA on the per-step latent deltas δ_t = z_{t+1} - z_t across the
corpus. The spectrum tells you how many directions narrative is
actually moving in; the principal components are inspectable
directions in the latent space.

Companion to E08 (linear transition operator). Where E08 asks "what
does the operator amplify?", E09 asks "where does the corpus actually
go?". They coincide for shift-invariant dynamics and diverge
interestingly otherwise.

Full hypothesis is in `experiments/EXPERIMENTS.md`.

## Run

```bash
python experiments/e09_velocity_spectrum/compute.py
```

## Dependencies

`numpy` only. Thin SVD on the centered (n × d) delta matrix.

## What it adds to the JSON

In `metadata`:

```json
"velocity_pca": {
  "components":     [[..., ..., ...], ...],  // PCs as rows, descending variance
  "variance":       [..., ..., ...],         // raw variances
  "variance_ratio": [..., ..., ...],         // fraction of total
  "mean_delta":     [..., ..., ...],         // average drift in the latent
  "n_deltas":       4379
}
```

## ICA upgrade

To get statistically independent components in place of orthogonal
ones, swap the SVD for `sklearn.decomposition.FastICA`. Independence
is sometimes a better prior for "narrative operations" than
orthogonality, especially when the underlying mechanisms are
non-Gaussian (which they almost always are for text).
