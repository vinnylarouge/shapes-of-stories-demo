# E03 — Motif dictionary (VQ)

Clusters all passage latents into K motifs via k-means and assigns
each passage a code in `[0, K)`. Each book also gets a histogram of
its motif usage.

Full hypothesis is in `experiments/EXPERIMENTS.md`.

## Run

```bash
python experiments/e03_motifs/compute.py \
    --in public/story_shapes.json \
    --out public/story_shapes.json \
    --k 8
```

## Dependencies

`numpy` only. K-means with k-means++ init is implemented inline.

## What it adds to the JSON

Per passage:

```json
"code": 3
```

Per book:

```json
"code_histogram": [12, 4, 0, 41, 0, 18, 5, 7]
```

In `metadata`:

```json
"vq_k": 8,
"vq_centroids": [[8.21, 9.04, 7.18], ...]
```

## Notes

- This is the **placeholder version** that runs on the post-UMAP 3D
  coordinates. The codes are recognisable as spatial regions of the
  visualization, which is fine for a frontend demo but is *not* a real
  motif dictionary in the §3 sense.
- The real version runs on the raw AE latents (typically 32–64-d) that
  E07 will produce. The `compute.py` here will run unchanged on that
  richer input — only the dimensionality of the centroids changes.
- The per-book `code_histogram` is the natural input for cross-book
  similarity in motif space (Jensen-Shannon divergence on the
  normalised histograms). This gives an alternative to E02's shape
  signature for clustering books.
