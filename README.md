# shapes of stories

A demo of *concept-mining for channels*, applied to narrative analysis and
generation.

The core idea: a transformer is a channel. As it processes text, its hidden
states (and per-layer residual deltas) trace out a trajectory in activation
space. An autoencoder over those activations gives a low-dimensional latent
in which a narrative becomes a *sequence of points* — beads on a string.
The shape of the string is the shape of the story.

The frontend in this repo loads a corpus of canonical books processed
through that pipeline and renders the resulting trajectories as a 3D point
cloud you can fly around, colour by various features, and play back one
passage at a time.

## What's in here

```
backend/                          original Colab notebooks (GPT-2 medium)
public/story_shapes.json          the data the frontend reads
src/                              React + Three.js frontend
DIRECTIONS.md                     research directions writeup
experiments/                      experiment program — Python scripts that
                                  enrich story_shapes.json + a Mac-mini CLI
  EXPERIMENTS.md                  the experiment plan
  e01_neighbours/                 cross-book nearest passages
  e02_persistence/                per-book persistence signatures (TDA)
  e03_motifs/                     motif dictionary (vector quantisation)
  e04_vector_field/               local-kernel vector field + surprise
                                  (Barnes-Hut-flavoured adaptive octree)
  e05_canonical/                  position-aligned canonical axes (CCA)
  e07_extract/                    Mac-mini local extraction CLI for any
                                  HuggingFace causal LM (default
                                  Llama-3.2-3B), drop-in replacement for
                                  the existing story_shapes.json
```

`E06` (story interpolation) is a frontend-only mode and lives in the React
code, not under `experiments/`.

## Quickstart

```bash
# 1. install
npm install

# 2. enrich the JSON with all five experiments (cheap, ~seconds, pure numpy)
python experiments/e01_neighbours/enrich.py
python experiments/e02_persistence/compute.py
python experiments/e03_motifs/compute.py
python experiments/e04_vector_field/compute.py
python experiments/e05_canonical/compute.py

# 3. run the visualization
npm run dev
```

Open the printed URL (default <http://localhost:5173/shapes-of-stories/>).
The new features all degrade gracefully if the enrichment scripts have
not been run — you'll just see the original five colour modes and no
field overlay.

## What the frontend shows

Each book is a polyline of passages in a shared 3D latent space. Per
passage you can hover to read its text + features; per book you can
toggle it on, click to select, play back its trajectory, or pair it up
with another book in interpolate mode.

| Sidebar control | What it does | Comes from |
|---|---|---|
| **View** Default / Canonical / Interpolate | Default: post-UMAP latent. Canonical: same passages projected onto cross-book canonical axes (content-removed). Interpolate: pick two books, slide between them. | E05, E06 |
| **Show vector field overlay** | Renders an adaptive-octree drift field over the latent. Colour = magnitude (navy → vermilion), tail-to-head gradient = direction. Cell size adapts to local field variation. | E04 |
| **Colouring** Book / Dialogue / Entropy / Sent. Len / Position / **Motif** / **Surprise** / **Shape** / **Spec 1–4** | Bold modes are the new ones. Motif = k-means code. Surprise = residual against the field. Shape = persistence-based archetype. Spec 1–4 = top non-trivial diffusion eigenvectors of the passage cloud, rendered with a diverging blue↔red colour map. | E02, E03, E04, E10 |
| **Persistence glyph** next to each book | Tiny vertical-stroke H₀ persistence signature of the book's trajectory. | E02 |
| **Spectral metrics** sub-panel | Linear-T singular values + R², velocity-PCA variance ratios. | E08, E09 |
| **Field influence** slider in interpolate mode | Bends the interpolated polyline along the local v(z). 0 = pure lerp, 1 = strongly bent by the corpus's drift field. | E04 ↔ E06 coupling |
| **Hover** any passage | Text snippet, dialogue %, entropy, sentence length, motif code, surprise, and the 5 nearest cross-book passages. | E01, E03, E04 |

## Experiments

All experiments under `experiments/` write to `public/story_shapes.json`
in place, additively. Each one is a single Python script with no
arguments needed for the defaults. They are:

| Exp | What it computes | Cost |
|---|---|---|
| **E01** Cross-book nearest passages | per-passage 5-nearest-neighbour list across books, prepares E04/E05/E06 | numpy only, ~1 s |
| **E02** Persistence signatures | per-book H₀ via single-linkage, top-N + k-means to assign `shape_archetype` | numpy only, ~1 s |
| **E03** Motif dictionary | k-means with k-means++ init, per-passage `code`, per-book histogram | numpy only, ~1 s |
| **E04** Adaptive vector field | kernel-fit drift v(z) sampled by adaptive octree refinement (Barnes-Hut-flavoured), per-passage surprise residual | numpy only, ~5 s |
| **E05** Canonical axes (CCA) | position-aligned cross-book CCA via SVD on whitened cross-covariance | numpy only, ~1 s |
| **E07** Mac-mini extraction CLI | two-step CLI: load any HF causal LM (default Llama-3.2-3B), extract per-token activations to mmap, then PCA → AE → UMAP → JSON | torch + transformers; not yet run |
| **E08** Linear narrative transition operator | least-squares fit of T : ℝᵈ → ℝᵈ across (z_t, z_{t+1}) pairs, SVD to get input/output direction spectrum and amplification factors | numpy only, < 1 s |
| **E09** Velocity spectrum | PCA on the per-step latent deltas — how many directions narrative is actually moving in | numpy only, < 1 s |
| **E10** Diffusion eigenmaps | k-NN graph Laplacian eigendecomposition on the passage cloud; bottom non-trivial eigenvectors become new colour modes | numpy only, ~5 s |

E01–E10 run on the existing GPT-2 medium output. E07 produces a fresh
`public/story_shapes.json` from a larger model, after which all of
E01–E10 should be re-run unchanged.

E08–E10 form a *spectral track* that lifts the layer-to-layer spectral
analyses sketched in `DIRECTIONS.md` to the narrative-time axis (since
we don't yet have per-layer activations). E11–E13 — HGR maximal
correlation, CKA, and the full T*T-on-L² operator — are spec'd in
`experiments/EXPERIMENTS.md` and will run unchanged on the per-layer
data once E07 lands.

## Sample run output

After running E01–E05 against the canonical-books corpus (29 books, 4408
passages):

- **E01**: 4408 passages × 5 cross-book neighbours, ~250 KB
- **E02**: 29 books → 4 shape archetypes via single-linkage H₀ + k-means
- **E03**: 8 motifs balanced across 298–827 cells each
- **E04**: kernel field at h=1.92, **3676 adaptive leaf cells** distributed
  across 4 octree levels (9 / 249 / 1258 / 2160) — ~93% at the finest
  two levels, where the field has the most local variation
- **E05**: mean canonical correlations **[0.75, 0.41, 0.13]** across 406
  book pairs — the first axis correlates at 0.75 across books on average,
  which is real evidence for the §5 *content-independent narrative axes*
  hypothesis

Final enriched JSON: 1.06 MB → ~2.81 MB.

## Deeper reading

- `DIRECTIONS.md` — the full research direction writeup: technical
  (Mac-mini extraction at scale), TDA across multiple chunk scales,
  inverse protein-folding for narrative motifs, interpolation,
  canonicity / conceptual bases, and stories as dynamical systems
  (vector field via Neural ODE / flow matching).
- `experiments/EXPERIMENTS.md` — the operational experiment plan
  mapping each direction to a concrete experiment with its hypothesis,
  procedure, output schema, and frontend integration spec.
- `backend/` — the original Colab notebooks the frontend data was
  produced from. `concept_shapes_v2` is the per-layer ARI sweep + delta
  stack analysis; `story_shapes_full` adds graph-Laplacian spectral
  decomposition to the passage similarity graph.

## Tech stack

Frontend: Vite + React 19 + TypeScript + Three.js, Tailwind for styling.

Backend (data extraction + experiments): Python 3, numpy. The Mac-mini
extraction CLI (E07) additionally uses torch + transformers + datasets,
with MPS preferred over CUDA over CPU. The five enrichment scripts
(E01–E05) are pure numpy and run on any Python with numpy installed.
