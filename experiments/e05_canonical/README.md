# E05 — Position-aligned canonical axes (CCA)

Pairs passages across books by position fraction and runs canonical
correlation analysis on each pair. Averages the resulting canonical
loadings (with sign alignment) into a global set of axes whose
projections vary *similarly across books* — the operational handle on
"content-independent narrative axes" from `DIRECTIONS.md` §5.

## Run

```bash
python experiments/e05_canonical/compute.py \
    --in public/story_shapes.json \
    --out public/story_shapes.json \
    --samples 20
```

## Dependencies

`numpy` only. CCA is implemented via SVD on the whitened cross-covariance.

## What it adds to the JSON

Per passage:

```json
"canonical": [0.412, -0.139, 0.078]
```

The three numbers are the projections onto the top-3 averaged canonical
axes.

In `metadata`:

```json
"canonical_axes": {
  "samples_per_book": 20,
  "n_pairs": 406,
  "mean_canonical_corrs": [0.842, 0.611, 0.392]
}
```

The `mean_canonical_corrs` are the headline number: how strongly does
each canonical axis correlate across book pairs on average? Anything
above ~0.5 on the first axis is meaningful evidence of cross-book
shared structure.

## Notes

- This runs on the post-UMAP 3D coordinates, so the canonical axes live
  in 3D and are limited to 3 components. On the raw AE latent (32–64-d)
  from E07 the same script will produce a 32+-component canonical
  basis, of which the top few are the relevant ones.
- The frontend uses `passage.canonical` to drive a "canonical view"
  toggle in `Sidebar.tsx`: when on, `Canvas.tsx` swaps `(x3d, y3d, z3d)`
  for `canonical[0..2]`. Same books, same passages, *content-removed*
  embedding — A/B comparison with the default view.
