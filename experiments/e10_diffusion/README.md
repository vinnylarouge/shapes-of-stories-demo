# E10 — Diffusion eigenmaps over the passage cloud

Builds a k-NN graph over all passages in the latent, computes the
symmetric normalised Laplacian, and stores the bottom non-trivial
eigenvectors per passage. Each eigenvector is a smooth scalar function
on the corpus reflecting one slice of intrinsic geometry — the spectral
decomposition of the *manifold the corpus traces*, independent of any
particular book.

Lifts the graph-Laplacian work in `backend/story_shapes_full.ipynb`
Part 2 from the notebook into the rendered visualization, where the
eigenvectors become new colour modes (`eig1`, `eig2`, `eig3`,
`eig4`).

Full hypothesis is in `experiments/EXPERIMENTS.md`.

## Run

```bash
python experiments/e10_diffusion/compute.py \
    --k 15 --n-eig 4
```

## Dependencies

`numpy` only. The eigendecomposition step uses `np.linalg.eigh` on a
dense (n × n) matrix; for n ≈ 4400 this takes about 30 s and 600 MB
of peak memory. Switch to `scipy.sparse.linalg.eigsh` if you need
faster turn-around or larger n.

## What it adds to the JSON

Per passage:

```json
"eig": [0.0123, -0.0451, 0.0089, 0.0211]
```

In `metadata`:

```json
"diffusion": {
  "k_neighbors": 15,
  "n_eig": 4,
  "eigenvalues": [0.0021, 0.0044, 0.0079, 0.0102],
  "trivial_eigenvalue": 0.0
}
```

`trivial_eigenvalue` is the constant mode at the bottom of the
spectrum (always ~0 for a connected graph). The non-trivial
`eigenvalues` are the smallest informative ones; the gaps between
them reveal the natural number of clusters (the "spectral gap").

## Frontend coupling

Each non-trivial eigenvector becomes a new colour mode in the
sidebar (`eig1`, `eig2`, ..., up to `n_eig`). Toggling one paints
every passage by its value on that eigenvector, using a diverging
blue↔red gradient around 0. Smooth eigenvectors look like gradients
across the cloud; rougher eigenvectors look like blocky
partitions — that visual difference is the spectral gap rendered.

## Notes

- Connectivity matters. If the k-NN graph is disconnected, the
  bottom eigenvalues are all near 0 with no information; bump
  `--k` until the spectrum is clean.
- This is currently run on the post-UMAP 3D latent so the diffusion
  geometry is filtered through UMAP. When E07 produces the higher-
  dimensional AE latent, run E10 on that for cleaner spectra.
- The same script works for *any* point cloud — including the future
  per-layer trajectories from E07. Each transformer layer would get
  its own diffusion spectrum, and comparing the spectra (e.g., with
  CKA) recovers the spirit of the layer-comparison spectral track in
  `DIRECTIONS.md`.
