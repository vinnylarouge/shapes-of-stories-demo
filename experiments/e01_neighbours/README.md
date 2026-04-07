# E01 — Cross-book nearest passages

Attaches the K nearest *cross-book* passages to every passage in
`public/story_shapes.json`. The frontend's `PassagePanel` reads the
optional `neighbours` field and shows the matches under the existing
feature row.

Full hypothesis, schema, and frontend integration are in
`experiments/EXPERIMENTS.md`.

## Run

From the repo root:

```bash
python experiments/e01_neighbours/enrich.py \
    --in public/story_shapes.json \
    --out public/story_shapes.json \
    --k 5
```

Defaults are `--in public/story_shapes.json --out public/story_shapes.json
--k 5`, so a bare `python experiments/e01_neighbours/enrich.py` does the
right thing.

## Dependencies

`numpy` only. No model load, no GPU. Runs in well under a second on the
existing 29-book corpus.

## What it adds to the JSON

Per passage:

```json
"neighbours": [
  {"b": 12, "p": 47, "d": 0.184},
  {"b":  3, "p":  9, "d": 0.221},
  ...
]
```

`b` is the book index (matches the index in the top-level `books` array),
`p` is the passage index inside that book, `d` is the Euclidean distance
in 3D latent space. The list is sorted ascending by `d` and has length `k`.

The script also stamps `metadata.neighbours_k = k` so the frontend can
display "k=5 nearest cross-book passages" in a tooltip if desired.

## Notes

- Distances are computed on the post-UMAP `(x3d, y3d, z3d)` columns
  because that is what the visualization shows; nearest-in-the-picture
  matches the user's expectation. When the §1 Mac-mini extraction lands
  with raw AE latents, swap the coordinate columns inside `enrich.py`
  for those without changing anything else.
- The script overwrites the input JSON in place by default. If you want
  to keep the unenriched version, pass an explicit `--out` path.
- Brute-force pairwise distance is O(n²) but fine for n ≈ 2k. If the
  corpus grows past ~50k passages, switch to `scipy.spatial.cKDTree`
  inside `enrich.py` (one-line change).
