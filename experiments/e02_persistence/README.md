# E02 — Per-book persistence signatures

Each book's passage trajectory is treated as a point cloud in the 3D
latent. Single-linkage gives the H₀ death times (the merge heights of
the Vietoris-Rips filtration), the top-N deaths form a fixed-length
shape signature, and k-means clusters those signatures into
`shape_archetype` labels.

Full hypothesis is in `experiments/EXPERIMENTS.md`.

## Run

```bash
python experiments/e02_persistence/compute.py \
    --in public/story_shapes.json \
    --out public/story_shapes.json \
    --top 8 \
    --archetypes 4
```

## Dependencies

`numpy` only. Single-linkage and k-means are implemented inline so the
script runs on any Python with numpy. No scipy, no sklearn, no ripser.

## What it adds to the JSON

Per book:

```json
"persistence": [
  {"d": 0, "birth": 0.0, "death": 1.42},
  {"d": 0, "birth": 0.0, "death": 0.91},
  ...
],
"shape_archetype": 2
```

`d=0` is the homological dimension (H₀ only in this version; H₁ is a
later extension that needs `ripser`). The list is the top-N most
persistent components sorted by lifespan (descending). `shape_archetype`
is an integer in `[0, archetypes)` from k-means on the signatures.

In `metadata`:

```json
"persistence_top": 8,
"shape_archetypes": 4
```

## Notes

- The Wasserstein-1 proxy (sorted-deaths-as-vector + Euclidean
  distance) used here gives the same clustering intuition as full
  bottleneck distance for our 3D, ~150-points-per-book regime, with
  zero dependencies. If you want true bottleneck distances, install
  `persim` and add a per-pair `persim.bottleneck` call inside
  `compute.py`.
- H₁ persistence (loops in the trajectory — recurrent stories that
  return to earlier states) requires `ripser` or `gudhi`. Worth adding
  once you've decided whether the H₀ archetypes are interesting.
- The `shape_archetype` field is independent of the existing
  `archetype` field on `Book` (which comes from the original notebook's
  k-means on raw passage means). Both are kept; the frontend can show
  either or both.
