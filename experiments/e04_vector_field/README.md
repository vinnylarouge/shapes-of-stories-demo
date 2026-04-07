# E04 — Local-kernel vector field + per-passage surprise

Fits a Nadaraya-Watson kernel-weighted vector field to all observed
passage transitions in the corpus, samples the field on a regular 3D
grid (for canvas visualization), and stores a per-passage *surprise
residual* — how far the actual next step departs from the field's
prediction.

Full hypothesis and connections in `experiments/EXPERIMENTS.md` §6.

## Run

```bash
python experiments/e04_vector_field/compute.py \
    --in public/story_shapes.json \
    --out public/story_shapes.json \
    --grid 12
```

`--grid 12` gives 12³ = 1728 grid nodes; the JSON gains roughly 200 KB.
For finer field overlays use `--grid 16` (4096 nodes).

## Dependencies

`numpy` only. The kernel sums are O(Q × M) and run in seconds for the
canonical-books corpus (M ≈ 4400 transitions, Q ≈ 1700 grid points).

## What it adds to the JSON

Per passage:

```json
"surprise": 0.142
```

In `metadata`:

```json
"field": {
  "grid_n": 12,
  "bandwidth": 0.213,
  "grid": [
    {"x":  4.21, "y": 5.10, "z": 6.04, "vx": 0.018, "vy": -0.022, "vz": 0.041},
    ...
  ]
}
```

`grid` contains `grid_n^3` nodes spanning the data bounding box (with
5% padding). Each node carries the predicted drift vector at that
location.

## Notes

- The bandwidth is set to 0.75× the median pairwise displacement length
  in a 100-point sample, which gives a locally smooth field without
  smearing across unrelated regions. Tweak inside `compute.py` if the
  streamlines look too tight or too washed out.
- The last passage of each book has no observed next-step, so its
  surprise is filled with the previous value rather than dropped.
- For a Neural ODE upgrade (per `DIRECTIONS.md` §6.2.2): swap
  `kernel_field` for a small MLP trained on the same `(sources,
  deltas)` pairs and call it the same way. The grid-sampling and
  surprise-residual code stay the same.
