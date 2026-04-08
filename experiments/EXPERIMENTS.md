# Experiments

This file is the running experiment program for `shapes-of-stories`. It is the
operational counterpart to `DIRECTIONS.md`: every experiment here ties to one
or more directions there, and every experiment is shaped so that its output
flows back into `public/story_shapes.json` and renders in the existing
React/Three.js frontend with the smallest possible UI delta.

## Design rules

These rules apply to every experiment in this directory:

1. **Inputs.** Every experiment reads `public/story_shapes.json` as its
   primary input. (Once the §1 Mac-mini extraction lands, the same scripts
   will read its richer output unchanged — the format is a strict
   superset.)
2. **Outputs are additive.** Experiments enrich the JSON with new
   *optional* fields. The frontend treats every new field as
   `field?: T | undefined` and degrades silently when it's missing. This
   means the existing frontend never breaks, and an experiment can be
   tried, reverted, or A/B tested without coordination.
3. **Re-runnable.** Each experiment is a single Python script (or
   notebook) that can be invoked from the repo root with no arguments
   beyond an input/output path. No notebook state, no Colab assumptions.
4. **Forward-compatible.** Experiments do not assume GPT-2 medium or
   the existing W=64 passage size. When `public/story_shapes.json` is
   regenerated from Llama-3.2-3B (per §1), the same scripts run unchanged.
5. **Schema-first.** Every experiment specifies the exact JSON keys it
   adds, what type they are, and what frontend code reads them. This is
   in the per-experiment README.
6. **Cheap before expensive.** Run order is by cost, not by §-number in
   `DIRECTIONS.md`. Experiments that need only the existing 3D
   coordinates run first; experiments that need the full activation
   tensor wait for §1.

## Mapping to directions

| Exp | Title | Track | Cost | Status |
|---|---|---|---|---|
| **E01** | Cross-book nearest passages | §4 (interpolation prelude), §5 (canonicity prelude) | trivial — numpy on existing JSON | **implemented** |
| **E02** | Per-book persistence signatures | §2 (TDA) | small — pure numpy (single-linkage H₀) | **implemented** |
| **E03** | Motif dictionary (VQ) | §3 (inverse protein-folding) | small — pure-numpy k-means | **implemented** (3D placeholder; raw-latent version once E07 lands) |
| **E04** | Local-kernel vector field + surprise track | §6 (dynamical systems) | small — pure numpy | **implemented** |
| **E05** | Position-aligned canonical axes (CCA) | §5 (canonicity) | small — pure numpy via SVD | **implemented** |
| **E06** | Story interpolation slider | §4 (interpolation) | medium — frontend-only | **implemented** (lerp version; OT/slerp upgrade pending) |
| **E07** | Mac-mini extraction CLI + Llama-3.2-3B | §1 (technical) | large — needs hardware | **scripts ready, not yet run** |
| **E08** | Linear narrative transition operator (SVD / DMD) | §6 (dynamical systems), spectral track | trivial — numpy lstsq + 3×3 SVD | **implemented** |
| **E09** | Representational velocity spectrum (PCA on deltas) | §6 (dynamical systems), spectral track | trivial — thin SVD on (n × d) | **implemented** |
| **E10** | Diffusion eigenmaps over the passage cloud | §5 (canonicity), spectral track | small — dense Laplacian eigh, ~30 s | **implemented** |
| E11 | HGR maximal correlation spectrum | spectral track | medium — needs binned joint distribution or per-layer data | spec only |
| E12 | CKA cross-layer / cross-book alignment | spectral track | small — needs per-layer data from E07 | spec only |
| E13 | Spectral analysis of T*T on L² | spectral track | large — fully nonlinear, kernel methods | spec only |

E01–E10 have working implementations. E08–E10 form a *spectral track*
that computes interpretable spectra of the latent dynamics: the linear
transition operator (E08), the velocity covariance (E09), and the
diffusion-operator eigenfunctions on the passage cloud (E10). The
richer cousins E11–E13 (HGR, CKA, T*T on L²) need either per-layer
activation data from E07 or expensive kernel machinery; specs are below.

## Spectral track (E08–E13)

`DIRECTIONS.md` originally framed these spectra as "between consecutive
transformer layers" (the joint distribution of (Z_l, Z_{l+1}) in the
autoencoded latent). The published JSON has only the post-UMAP 3D
latent and no per-layer activations, so the implemented members of
this track (E08–E10) apply the same machinery to the
**narrative-time** axis instead — Z_l = z_t, Z_{l+1} = z_{t+1}, with
the transition operator now being the corpus's drift between adjacent
passages. When E07 produces per-layer data, swap the data loaders and
the same spectra become the literal layer-to-layer objects from the
directions document.

### E11 — HGR maximal correlation

The HGR decomposition of the joint distribution of (Z_l, Z_{l+1})
gives orthogonal "channels" of information flow, each with a maximal
correlation value playing the role of a singular value. Architecture-
invariant by construction; directly generalises SVD to the nonlinear
setting; connects to the principal inertia components framework.

Computation, two paths:

1. **Discretised** — bin (Z_l, Z_{l+1}) into a joint histogram, build
   the *Q matrix* `Q[i,j] = P_{ll+1}[i,j] / sqrt(P_l[i] · P_{l+1}[j])`,
   take its SVD. Singular values are the HGR spectrum, singular
   functions are step functions on the bins. Tractable for low-d (≤ 3)
   but blows up combinatorially with dimension — best run after E07
   produces a small, well-chosen latent rather than the post-UMAP 3D
   coordinates that have already lost most of the conditional
   structure.
2. **Kernel** — approximate the HGR functions in an RKHS; reduces to
   kernel CCA. More expressive but more expensive; needs hours of
   compute on the canonical-books corpus even for the simple
   passage-to-passage case.

### E12 — CKA (centered kernel alignment) cross-layer

CKA between two representations X and Y is a normalised scalar
saying how similarly they cluster the data. Lifted to a *spectral*
view via Gram-matrix eigendecomposition, it gives shared modes
between layers (or between books, or between models).

Trivial to implement once E07 produces multi-layer activations:

```python
def cka(X, Y):
    Kx = X @ X.T; Ky = Y @ Y.T
    n = len(X)
    H = np.eye(n) - np.ones((n, n)) / n
    KxC, KyC = H @ Kx @ H, H @ Ky @ H
    return (KxC * KyC).sum() / (np.linalg.norm(KxC) * np.linalg.norm(KyC))
```

For the existing 3D-only data, the cross-*book* CKA still works and
gives a 29×29 book similarity matrix that complements the E02 shape
clustering — TODO follow-on.

### E13 — Spectral analysis of T*T on L²

The fully general nonlinear version: keep the layer-to-layer map T
nonlinear and study the eigendecomposition of its adjoint composition
T*T as an operator on L²(latent). HGR is the finite-dim approximation
of this; the full operator-theoretic version is the gold standard but
requires either Markov chain Monte Carlo or kernel approximations
that are hard to converge. Best run on the per-layer activations from
E07, where the operator has a short-range structure that makes the
estimation tractable.

---

## E01 — Cross-book nearest passages

**Direction.** §4.3.1 (nearest-neighbour decoding) and §5 (canonicity:
"do passages from different books that share latent space share prose
character?").

**Hypothesis.** For most passages, the five nearest non-same-book passages
in the latent space are *recognisably similar* in prose mode (dialogue
density, narration style, register). If true, this is the simplest direct
evidence that the latent encodes content-independent narrative structure
and not just book fingerprinting.

**Procedure.** For each passage `p` in book `b`:

1. Compute the Euclidean distance from `p`'s `(x3d, y3d, z3d)` to every
   passage in every book `b' ≠ b`.
2. Keep the 5 smallest.
3. Attach them to `p` as `p.neighbours`.

We use the post-UMAP 3D coordinates already in the JSON because (a) they
are what the visualization shows, so neighbour-of-a-point on hover is
visually consistent, and (b) it requires no extra extraction. When the
richer §1 JSON lands with raw AE latents, swap the coordinate columns
for those without changing the script structure.

**Output schema** (additive):

```ts
interface PassageNeighbour {
  b: number;  // bookIndex (matches Book[] index in the JSON)
  p: number;  // passageIndex within that book
  d: number;  // Euclidean distance in the latent (rounded)
}

interface Passage {
  ...existing fields...
  neighbours?: PassageNeighbour[];  // length K, sorted ascending by d
}
```

Short keys (`b`, `p`, `d`) keep the JSON compact. For 29 books × ~80
passages × 5 neighbours × 3 numbers ≈ 35k numbers, ~250 KB additional
on disk.

**Frontend changes.**

- `src/utils/types.ts` — add `PassageNeighbour` interface, mark
  `Passage.neighbours` as optional.
- `src/components/PassagePanel.tsx` — when the hovered passage has
  `neighbours`, render them under the existing feature row as a small
  list of book + position. Read-only in this first pass.
- No changes to `Canvas.tsx`, `Sidebar.tsx`, or any other component.

**Follow-ons (not in this pass).**

- Click a neighbour → select that book + jump playback to that passage.
  This is the natural next UI gesture but requires plumbing
  `singleActiveBook` + `playback.currentIndex` callbacks down to
  `PassagePanel`. Worth doing, but a separate PR.
- Cross-book pairing matrix as a sidebar panel: "books most often
  appearing as neighbours of book X" → a quick proxy for
  `story_shapes_full.ipynb` Part 2's cross-book mode-sharing test.

**Run.** See `e01_neighbours/README.md`.

---

## E02 — Per-book persistence signatures

**Direction.** §2.2 (persistent homology of a story).

**Hypothesis.** Books cluster into a small number of *shape archetypes*
when measured by the bottleneck distance between their persistence
diagrams. The clustering should not coincide with author or genre — it
should pick up structural features (linear chronicle vs. recurrent return
vs. branching ensemble).

**Procedure.** For each book:

1. Take its passage trajectory as a point cloud in 3D.
2. Compute the H₀ and H₁ persistence diagrams via `ripser`.
3. Reduce each diagram to a fixed-length signature: top-N persistence
   pairs by lifespan (`death − birth`).

Then across the corpus:

4. Compute pairwise bottleneck distance between all (book, book′) pairs.
5. Hierarchical-cluster the distance matrix; assign each book a
   `shape_archetype ∈ {0,1,2,...}`.

**Output schema** (additive):

```ts
interface PersistencePair {
  d: 0 | 1;       // homological dimension
  birth: number;
  death: number;
}

interface Book {
  ...existing fields...
  persistence?: PersistencePair[];
  shape_archetype?: number;  // replaces or supplements existing `archetype`
}
```

**Frontend changes.**

- Sidebar: a tiny inline persistence-diagram glyph (5×5 px SVG with H₁
  pairs as dots) next to each book. This is a reusable
  `<PersistenceGlyph>` component.
- Optional new colour mode `'shape'` that colours points by their book's
  `shape_archetype`.

**Dependencies.** `ripser` (or `gudhi`). Installable via pip on macOS.

---

## E03 — Motif dictionary (VQ)

**Direction.** §3 (inverse protein-folding / discrete narrative alphabet).

**Hypothesis.** A vocabulary of ~64–256 motifs is sufficient to
reconstruct the corpus's passage latents within an MSE budget. Motifs
recur across books and correspond to recognisable prose modes.

**Procedure.**

1. Cluster the (latent) passage vectors into K codes via mini-batch
   k-means. K is a sweep parameter; start with 64.
2. Each passage gets its nearest code's index.
3. Per book, compute the histogram of codes used (a length-K vector).
4. Per code, compute reconstruction MSE (from the cluster centroid to
   the original latent).

For the existing JSON we have only the post-UMAP 3D coordinates, which
is too compressed for a meaningful motif dictionary. E03 should run for
real on the §1 output. As a placeholder demo for the frontend, k-means
on the 3D coordinates with K=8 still gives a recognisable colouring.

**Output schema.**

```ts
interface Passage {
  ...
  code?: number;  // 0..K-1
}

interface Book {
  ...
  code_histogram?: number[];  // length K
}

interface Metadata {
  ...
  vq_k?: number;
  vq_centroids?: number[][];  // K × D, optional, for the canvas overlay
}
```

**Frontend changes.**

- New colour mode `'motif'` in `Sidebar.tsx` and `colours.ts`.
- Optional sidebar panel: per-book code histograms as a small sparkline.
- Optional canvas overlay: render the K centroids as larger
  semi-transparent spheres in the 3D view (the "amino acid alphabet"
  visualised in place).

---

## E04 — Local-kernel vector field + surprise track

**Direction.** §6 (stories as dynamical systems).

**Hypothesis.** The passage transitions across the corpus reveal a
non-trivial drift field on the latent. The field has identifiable
fixed points (lingering states) and the per-passage residual against
the field aligns with chapter breaks and climactic moments.

**Procedure.** Cheap baseline first; Neural ODE later.

1. Build a flat array of (point, displacement) pairs by walking each
   book's passage trajectory and recording `(z_t, z_{t+1} − z_t)`.
2. For any query point `q`, the predicted drift is the kernel-weighted
   average of nearby displacements:
   `v(q) = Σᵢ wᵢ Δzᵢ` with `wᵢ = exp(-‖q − zᵢ‖²/h²)`.
   Pick `h` from the median pairwise distance.
3. Sample `v` on a regular 3D grid (16³ is enough for visualization).
4. For each actual passage, compute the *surprise residual*
   `s_t = ‖(z_{t+1} − z_t) − v(z_t)‖`.

**Output schema.**

```ts
interface FieldGridPoint {
  x: number; y: number; z: number;     // grid node
  vx: number; vy: number; vz: number;  // predicted drift at this node
}

interface Passage {
  ...
  surprise?: number;  // ||observed Δ - predicted Δ||
}

interface Metadata {
  ...
  field?: { grid: FieldGridPoint[]; bandwidth: number };
}
```

**Frontend changes.**

- A toggle in the colour-mode row: "Field". When on, render the field
  grid in `Canvas.tsx` as small line segments (or small cones via
  three's `ConeGeometry`) at each grid node, oriented along `(vx,vy,vz)`.
- New colour mode `'surprise'` that colours passages by `s_t`. This is
  the most legible use of the surprise signal.
- Optional sidebar sub-panel: per-book surprise track as a line plot
  along position fraction.

**Why this is the headline visual.** Streamlines + actual story
trajectories overlaid + colour-by-surprise in the same view *is* the
"shapes of stories" concept made literal — the corpus shows you
where stories tend to flow, and each book is a particle being pushed
around in that flow.

---

## E05 — Position-aligned canonical axes (CCA)

**Direction.** §5.2.2 (cross-book CCA on positionally aligned passages).

**Hypothesis.** Pairing passages across books by *position fraction*
(10% into book A with 10% into book B, etc.) and running CCA reveals
axes in the latent that vary similarly across books. These are
candidates for content-independent narrative axes.

**Procedure.**

1. For each book, sample 20 passages at evenly spaced position
   fractions [0, 0.05, 0.10, ..., 0.95].
2. Stack across books: `X ∈ ℝ^{n_books × 20 × d}`.
3. For each pair of books `(A, B)`, run CCA on `X[A]` and `X[B]` and
   collect the top canonical correlation axes.
4. Average the resulting axes across all pairs (with sign alignment).
5. Project all passages onto the top 3 averaged axes.

**Output schema.**

```ts
interface Passage {
  ...
  canonical?: [number, number, number];  // top 3 canonical projections
}

interface Metadata {
  ...
  canonical_axes?: { variance_explained: number[]; corr_per_pair: number };
}
```

**Frontend changes.**

- New "Canonical view" toggle that, when on, replaces the 3D coordinates
  shown in the canvas with the 3D `canonical` projections. This is a
  one-line swap in `Canvas.tsx` (`p.x3d` → `p.canonical[0]`, etc.) and
  gives an A/B comparison: same books, same passages, *content-removed*
  embedding.
- Three new colour modes (`canonical_0`, `canonical_1`, `canonical_2`)
  if the per-axis colouring is more legible.

---

## E06 — Story interpolation slider

**Direction.** §4 (interpolation).

**Depends on.** E01 (for nearest-neighbour decoding) and E03 (for code
assignment) — best run after both.

**Hypothesis.** Slerp between two books' position-aligned latents
produces a coherent intermediate trajectory whose nearest-passage
retrievals are recognisable as a blend.

**Procedure.**

1. User picks book A and book B (in the UI).
2. For each `t ∈ {0%, 5%, ..., 100%}`, position-align both books at
   `t` and slerp the latents with `α ∈ [0, 1]`.
3. For each interpolated point at the current `α`, retrieve the nearest
   corpus passage (use the precomputed E01 neighbours as the index).
4. Render the interpolated trajectory as a third polyline in 3D.

**Output schema.** None precomputed — interpolation is computed in the
browser from the existing `x3d/y3d/z3d` and the E01 neighbour index.
Optionally precompute a grid of (A, B, α) → retrieved passage triples
for instant rendering.

**Frontend changes.**

- New "Interpolate" mode in `Sidebar.tsx` with two book selectors and
  an α slider.
- `Canvas.tsx` renders the interpolated polyline.
- `PassagePanel.tsx` shows the current retrieved passage.

---

## E07 — Mac-mini extraction CLI + Llama-3.2-3B

**Direction.** §1 (technical track).

This is the foundational work that everything else benefits from. It is
*not* needed to run E01–E06 on the existing data, but the moment it
lands, all of E01–E06 should be re-run unchanged on its output and the
visual should *get better*.

The full design is in §1 of `DIRECTIONS.md`. Concrete deliverables here:

1. `experiments/e07_extract/extract_activations.py` — the CLI described
   in §1.3.
2. `experiments/e07_extract/configs/canonical_books.yaml` — the corpus
   file (port the existing book list from `story_shapes_demo.ipynb`).
3. `experiments/e07_extract/build_story_shapes_json.py` — the
   post-processing script that turns raw activations into the existing
   `public/story_shapes.json` schema (PCA → AE → UMAP → passage features).

The output is a drop-in replacement for the current
`public/story_shapes.json`, at which point E01 just re-runs against it
and the frontend keeps working without any code change.

---

## Suggested implementation order

1. **E01** — done in this commit. Demonstrates the integration pattern
   and gives the frontend its first new behaviour.
2. **E04 (cheap version)** — local-kernel vector field on the existing
   JSON. The streamline overlay is the most visually striking single
   addition to the canvas.
3. **E02** — persistence signatures. Adds the sidebar glyph and a
   shape-based archetype that may be more meaningful than the current
   `archetype` field.
4. **E07** — Mac-mini extraction CLI. Once running, re-do E01 and E04
   on its output for the headline result.
5. **E03 + E05** in parallel after E07.
6. **E06** last, on top of E01/E03/E04.
