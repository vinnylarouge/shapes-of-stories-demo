# E08 — Linear narrative transition operator (SVD / DMD)

Fits a single linear operator T : ℝᵈ → ℝᵈ across all (z_t, z_{t+1})
pairs in the corpus, then takes the SVD to get an interpretable
spectrum of input/output directions and amplification factors.

This is the simplest, cheapest member of the spectral family from
`DIRECTIONS.md` — exact for purely linear dynamics, a useful first
approximation otherwise. Companion to HGR, CKA, and the full
T*T-on-L² operator (which are nonlinear / higher-order generalisations).

Full hypothesis is in `experiments/EXPERIMENTS.md`.

## Run

```bash
python experiments/e08_transition_svd/compute.py
```

## Dependencies

`numpy` only. Closed-form least-squares + SVD on a 3×3 matrix; runs
in milliseconds on the canonical-books corpus.

## What it adds to the JSON

In `metadata`:

```json
"transition_svd": {
  "T":     [[..., ..., ...], ...],   // 3×3 fitted operator
  "U":     [[..., ..., ...], ...],   // 3×3 left singular vectors (output directions)
  "sigma": [..., ..., ...],          // 3 singular values, descending
  "V":     [[..., ..., ...], ...],   // 3×3 right singular vectors (input directions)
  "n_pairs": 4379,
  "r2":     0.92
}
```

`r2` is the coefficient of determination on the (z_t → z_{t+1})
prediction. Values close to 1 say the linear operator is a good fit;
values much below 0.9 say nonlinear effects dominate and the next step
in this track is to upgrade to E11 / E12 (HGR / nonlinear T*T spectrum).

## How to read the spectrum

Largest σ tells you the dominant mode of single-step amplification:
the input direction `V[:, 0]` is the latent direction along which the
narrative reliably advances; `U[:, 0]` is where it lands one step
later. Smallest σ tells you the most-suppressed direction: structure
that the corpus systematically *forgets* between adjacent passages.

Sign-flipping a singular vector flips both its U and V counterparts;
the spectrum is determined up to that gauge.

## Reinterpretation note

`DIRECTIONS.md` frames these spectra as *between consecutive
transformer layers* (the (Z_l, Z_{l+1}) joint distribution). The
existing JSON has only the post-UMAP 3D latent and no per-layer
activations, so this script computes the same operator but along the
*narrative-time* axis instead — z_{t+1} given z_t. When E07 produces
per-layer data, swap the data loader and the spectrum becomes the
literal layer-to-layer object from the directions document.
