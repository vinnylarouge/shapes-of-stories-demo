# Directions: Channel Mining for Narrative Analysis & Generation

This document elaborates the next phase of the *shapes-of-stories* project. The
existing notebooks (`concept_shapes_v2`, `story_shapes_demo`, `story_shapes_full`)
establish the core technique:

> A transformer is a channel. As it processes text, its hidden states (and the
> per-layer residual deltas) sketch out a trajectory in activation space. An
> autoencoder over those activations gives us a low-dimensional latent in which
> a narrative is a *sequence of points* — beads on a string. The shape of the
> string is the shape of the story.

The aim of the next phase is to make this technique (a) runnable locally on
Apple Silicon against a substantially larger open-source model, (b) topologically
aware so that we can compare narrative shapes across multiple chunk scales, and
(c) **generative**, by treating the latent as a manipulable substrate rather
than a passive read-out.

The directions below are split into one technical track and three theoretical
tracks. Each section names the concrete artefacts to produce.

---

## 0. Where we are

| Notebook | Model | Granularity | Latent | Output |
|---|---|---|---|---|
| `concept_shapes_v2` | GPT-2 (124M, 13 layers, d=768) | per-token, free generation | AE on PCA(128) of normed delta stack, k∈{16,32,64} | per-layer ARI curve, delta-stack UMAP, trajectory dynamics |
| `story_shapes_demo` / `_3d` | GPT-2 medium (355M) | passage (W=64 tokens) | AE on per-passage mean hidden state | UMAP of canonical books, prose-mode heuristics, JSON export |
| `story_shapes_full` (Part 2) | as above | passage | k-NN graph on AE latent, normalised Laplacian | spectral clusters, eigenvector–prose correlations, trajectory frequency decomposition |

Two key empirical results from `concept_shapes_v2` are load-bearing for
everything that follows:

1. **Final-layer hidden states do not separate topic.** Topic information lives
   in *cross-layer structure*, recoverable via the delta stack δᵢ = hᵢ − hᵢ₋₁.
2. **The delta stack is a composition of morphisms.** Each per-layer 768-d
   block is what one transformer block *added*. Treating the stack as the
   primary object — rather than any single layer — is what unlocks the latent.

Both of these justify scaling up: a larger model has more layers and richer
per-block contributions, and we expect the delta-stack signal to grow with
depth.

---

## 1. Technical track — Mac-mini local extraction at scale

**Goal.** Replace the Colab/CUDA assumptions in the existing extraction
notebooks with a local, scriptable pipeline that runs on a Mac-mini (M-series)
against an open-source model substantially larger than GPT-2 medium.

### 1.1 Target models

Pick one of these as the default; all of them have public weights and run
under `transformers` with `output_hidden_states=True`:

| Model | Params | Layers | d_model | Notes |
|---|---|---|---|---|
| Llama-3.2-1B | 1.2B | 16 | 2048 | smallest jump, fits comfortably in 16 GB unified memory in fp16 |
| Llama-3.2-3B | 3.2B | 28 | 3072 | sweet spot for an M2/M4-Pro Mac-mini in fp16 |
| Mistral-7B-v0.3 | 7.2B | 32 | 4096 | needs 4-bit quantisation to fit; only viable on 32 GB+ unified memory |
| Qwen2.5-3B | 3.1B | 36 | 2048 | more layers per parameter — good for the delta-stack thesis |

The right default is **Llama-3.2-3B** or **Qwen2.5-3B**: both give us
~3× more layers than GPT-2 (so the delta stack is correspondingly richer)
without requiring quantisation.

### 1.2 MPS / Metal pipeline

Concrete code-level changes from the existing notebooks:

- Replace `device = 'cuda' if torch.cuda.is_available() else 'cpu'` with
  ```python
  if torch.backends.mps.is_available():
      device = 'mps'
  elif torch.cuda.is_available():
      device = 'cuda'
  else:
      device = 'cpu'
  ```
  and gate any `torch.cuda.empty_cache()` / `synchronize()` calls behind a
  device check.
- Load the model with `torch_dtype=torch.float16` (MPS supports fp16; bf16
  support is partial as of torch 2.4 — verify before defaulting).
- Replace `GPT2LMHeadModel` / `GPT2Tokenizer` with `AutoModelForCausalLM` /
  `AutoTokenizer` so the same script can switch models by name.
- The current scripts allocate hidden states with `torch.HalfStorage.from_file`
  to map a single contiguous file. That still works on macOS but the file
  paths (`/content/shapes_v2`) are Colab-specific. Make the output directory
  configurable via an env var (default `./runs/<model>-<timestamp>/`).

### 1.3 From notebooks to a script

Pull the extraction logic out of the notebooks into a CLI:

```
extract_activations.py \
    --model meta-llama/Llama-3.2-3B \
    --corpus configs/canonical_books.yaml \
    --chunk-size 1024 --chunk-overlap 512 \
    --layers all --store delta-stack \
    --out runs/llama32-3b-canon/
```

Outputs to `runs/<id>/`:

- `hidden.bin` — memory-mapped fp16 tensor `(N, L, D)` where `N` is total
  retained tokens, `L` is the number of layers, `D` is `d_model`. Use the
  same `from_file` trick to keep memory usage flat.
- `meta.parquet` — one row per stored token with `seq_id`, `chunk_id`,
  `position`, `book`, `category`, `token_id`, `token_str`, `entropy`,
  `top1_logprob`. Parquet (rather than the current `torch.save(meta.pt)`)
  keeps the schema explicit and lets DuckDB / Polars query the metadata
  without loading the activations.
- `config.json` — model name, tokenizer hash, chunk policy, git SHA, the
  exact corpus file. Reproducibility.

### 1.4 Memory budget on a Mac-mini

A worked example for Llama-3.2-3B (L=28, D=3072) on the canonical-books
corpus from `story_shapes_demo` (~12 books × 10k tokens ≈ 120k tokens
retained after the second-half filter):

- Activations stored: 120 000 × 28 × 3072 × 2 bytes ≈ **20 GB**.

That's already too large to comfortably keep in unified memory alongside
the model. Two options, both worth implementing:

1. **Stream-and-project.** Fit an `IncrementalPCA` to a target rank (e.g.
   1024) on the fly per layer, and only persist the projected activations.
   The full delta stack then occupies 120 000 × 28 × 1024 × 2 ≈ 6.7 GB.
2. **Layer subset.** The per-layer ARI curve in `concept_shapes_v2` already
   shows that not every layer is informative. Pick the top-k layers by ARI
   on a small calibration run, then re-extract only those. This is the
   quickest path to a usable corpus.

### 1.5 Validation: reproduce the v2 result on a bigger model

Before any of the theoretical tracks, confirm that the *shape of the result*
survives the model upgrade:

1. Run the per-layer ARI sweep on the same Wikipedia category set, but with
   Llama-3.2-3B in place of GPT-2.
2. Plot the concept-localisation curve. We expect a peak somewhere in the
   middle layers and a higher absolute ARI than GPT-2 achieved.
3. Compute the delta-stack ARI. This is the headline number — if it does
   not exceed the best single-layer ARI (as it does for GPT-2), the
   "transformer-as-channel" framing has not transferred and we need to
   diagnose before scaling further.

---

## 2. Multi-scale / topological track — chunks of varying sizes

**Goal.** A narrative is not at one scale. A scene is a few hundred tokens,
a chapter a few thousand, an arc a few tens of thousands. Right now the
project picks a single window size (`W=64` in the demo notebooks, per-token
in `concept_shapes_v2`) and lives at that scale. The TDA hooks below are
about pinning the *same* trajectory across scales so we can talk about its
persistent features.

### 2.1 Scale-stack extraction

Extend the chunking logic so that each book yields a *pyramid*:

- Level 0: per-token deltas (the `concept_shapes_v2` regime)
- Level 1: passage means at W=32
- Level 2: passage means at W=128
- Level 3: passage means at W=512
- Level 4: passage means at W=2048

Concretely, each level reuses the same underlying activations — only the
pooling window changes — so the extra cost is a few mean-reductions, not
re-running the model.

### 2.2 Persistent homology of a story

For each story, compute persistent homology (H₀, H₁) of the point cloud at
each level using `ripser` (or `gudhi` for cubical complexes). The deliverable
is a *persistence diagram per scale per book*. From these we can read off:

- The number and lifespan of connected components — how many "movements"
  the story has at this scale.
- The number and lifespan of 1-dimensional holes — does the trajectory
  *return*? Recurrent stories (a hero leaving and returning) should show
  long-lived H₁ classes; linear stories (a chronicle) should not.
- A **bottleneck** or **Wasserstein** distance between two stories'
  diagrams gives a scale-aware similarity that doesn't depend on a shared
  embedding — useful for comparing across models or across runs.

This is the cleanest first deliverable in the topological track because it
requires no new models, only `ripser` over the existing latents. A single
notebook `topology_persistence.ipynb` can demonstrate the pipeline end-to-end
on the canonical-books corpus.

### 2.3 Mapper for narrative skeletons

`ripser` gives numbers; **Mapper** (via `kepler-mapper`) gives a graph.
Apply Mapper to the latent point cloud of a corpus (not a single story) with
the lens function set to either:

- the time index along each story (so the Mapper graph reflects narrative
  position), or
- the first Laplacian eigenvector from `story_shapes_full.ipynb` Part 2 (so
  it reflects the dominant prose-mode axis).

The output is a graph whose nodes are clusters of similar passages and
whose edges connect overlapping clusters along the lens. This is the
*topological skeleton* of the corpus and is the natural object to search
when generating: "find me a path from a node like X to a node like Y".

### 2.4 Multiscale consistency check

For each book, compute the trajectory at every scale and ask whether the
*coarse* trajectory (level 4) is approximately the *moving average* of the
*fine* trajectory (level 0). If yes, the scales agree and we can use the
coarse one as a navigation aid; if not, we have evidence that the model's
representation changes character at some scale, which is itself interesting
and a candidate for a paper figure.

---

## 3. Theoretical track A — Inverse protein-folding for narratives

**Goal.** Given the corpus of trajectories produced by §1, identify a small
inventory of *building blocks* such that every trajectory in the corpus can
be approximately reconstructed as a sequence (or weighted superposition) of
these blocks. The blocks are the "amino acids" of the narrative analogy.

### 3.1 Why this is the right framing

In protein folding, the primary structure (a sequence of amino acids)
determines the tertiary structure (a 3-D fold) determines the function.
*Inverse* folding asks the opposite: given a desired fold, find a sequence
that produces it. The narrative analogue:

| Protein term | Narrative term |
|---|---|
| primary sequence | sequence of latent vectors (or block IDs) |
| amino acid | a recurring motif in latent space |
| tertiary structure (3D fold) | the geometric shape of the trajectory |
| function | the experienced narrative effect (a turn, a reveal, a lull) |

The forward problem is what `story_shapes_demo` already does: feed text in,
get a fold out. The inverse problem is what makes the work *generative*:
prescribe a fold, recover a plausible sequence of motifs.

### 3.2 Building the block dictionary

Three candidate methods, in order of increasing structure:

1. **Vector quantisation.** Train a VQ-VAE (or its simpler cousin, k-means
   with a learned codebook of size 256–1024) on the per-passage latents
   from §1. Each passage is then assigned a discrete code. A book becomes
   a string over a small alphabet — exactly the *primary sequence*.
2. **Sparse dictionary learning.** Learn an overcomplete dictionary D such
   that each latent z ≈ Dα with α sparse (via `scikit-learn`'s
   `MiniBatchDictionaryLearning` or a lightweight sparse autoencoder).
   This relaxes the "exactly one block at a time" assumption: a passage
   can be a *blend* of two or three motifs, which matches reality better
   than a single discrete code.
3. **HMM / segmental HMM over the trajectory.** Train an HMM where the
   hidden states are the motifs and the observation is the latent. This
   adds an explicit *transition* model — some motifs follow other motifs
   far more often than chance — which is exactly what we need for §4.

Validate the dictionary by checking that:

- The same code/motif occurs across books with semantically related
  passages (e.g. dialogue scenes from different novels should share codes,
  not be book-fingerprinted). The cross-book mode-sharing test from
  `story_shapes_full.ipynb` Part 2 is the right rubric.
- Reconstruction MSE plateaus at a codebook size compatible with our
  intuitions (somewhere in the tens to low hundreds — not thousands).

### 3.3 Deliverables

- `dictionary_vq.ipynb` — VQ-VAE on the §1 latents, code distribution
  histogram, examples of the top-N passages assigned to each code, the
  book-to-code-string view.
- `dictionary_eval.md` — a short writeup with the cross-book sharing
  numbers, the reconstruction–size curve, and a qualitative discussion
  of which motifs are recognisable as known prose modes.

---

## 4. Theoretical track B — Interpolation between narrative-proteins

**Goal.** Given two narratives A and B (each represented as a trajectory or
a sequence of motifs from §3), produce a *third* trajectory that is
recognisably "between" them, and decode that trajectory back into text.

### 4.1 The two halves of the problem

Interpolation is two distinct sub-problems:

1. **Geometric.** Construct a path in latent space that starts at A's
   latent shape and ends at B's. The naive choice is linear interpolation;
   the better choice respects the manifold of plausible trajectories.
2. **Decoding.** Turn a latent trajectory back into text. The original
   model is a one-way street (text → activations); we need a route from
   activations back to text.

### 4.2 Geometric interpolation

Three options of increasing fidelity:

- **Slerp on the latent.** Spherical interpolation between the per-passage
  latents of A and B, after normalising. Cheap, surprisingly effective.
- **Optimal transport.** Treat each story's latent points as a probability
  measure, compute the OT plan between them (`POT` library), and interpolate
  along the McCann displacement. This respects the *distribution* of
  passages, not just their centroid.
- **Geodesic on a learned manifold.** Train a normalising flow or a
  diffusion model on the latents from §1. Interpolation is then a geodesic
  in the flow's base space, or a denoising trajectory between the two
  endpoints. This is the heaviest option but the only one that guarantees
  on-manifold interpolations.

### 4.3 Decoding back to text

Three options, in order of sophistication:

1. **Nearest-neighbour retrieval.** For each point on the interpolated
   trajectory, retrieve the corpus passage with the closest latent. Output
   is a *collage* — not a generated narrative, but a quick sanity check
   that the trajectory is meaningful.
2. **Activation steering.** Condition the original LM on a prompt and at
   each step *push* its hidden state toward the target latent (Turner et
   al.-style activation patching, or the contrastive activation addition
   approach). The model still produces fluent text but its trajectory is
   nudged onto our chosen path.
3. **Train an inverse decoder.** Train a small model whose input is a
   sequence of latents and whose output is text. This is the most ambitious
   option and the most rewarding: it makes the latent space a true
   generative substrate. A reasonable architecture is a frozen LM with a
   prefix conditioner that takes the latent as input.

Start with (1) for the demo, then move to (2) for the headline result.
(3) is a follow-on project on its own.

### 4.4 Deliverables

- `interpolation_demo.ipynb` — slerp + nearest-neighbour decoding between
  two canonical books, side-by-side text output for the original endpoints
  and the midpoint.
- A figure showing the interpolated trajectory drawn over the §1 UMAP, so
  the reader can see it traversing the latent.
- A failure-mode section: what does the latent look like when it leaves
  the manifold? Are there "uncanny valleys"?

---

## 5. Theoretical track C — Canonicity and conceptual bases

**Goal.** Find a small set of axes — *conceptual basis vectors* — in the
latent space such that any narrative can be described as its trajectory
along those axes, content-independently. These are the analogue of the
"conceptual bases" mentioned in the brief, and they are what would let us
say *Pride and Prejudice* and *Persuasion* trace similar shapes despite
different content.

### 5.1 Why this is hard and why it might work

The risk is that the principal axes of the latent space are dominated by
genre, register, vocabulary — content. The bet is that *after* removing
content (e.g. by projecting out the codes of §3 or by averaging over many
books in the same genre), the residual axes correspond to *narrative
operations*: rising tension, dialogue density, narrator distance, scene
vs. summary, and so on.

The graph-Laplacian decomposition in `story_shapes_full.ipynb` Part 2 is
already a crude version of this: the low-frequency Laplacian eigenvectors
are by construction the smoothest functions on the passage graph, which is
exactly what content-independent narrative axes ought to be. Push this
further:

### 5.2 Construction

Three complementary methods:

1. **Content-orthogonalised PCA.** For each passage, regress out its book
   identity (or genre embedding) from the latent. PCA the residuals. The
   top components are by construction not "this is Austen vs. Tolstoy"
   axes.
2. **Cross-book CCA.** Pair passages across books by *narrative position*
   (e.g. 10% into the book) and run canonical correlation analysis. The
   shared axes are the ones that vary similarly across books — these are
   the narrative-shape axes.
3. **Spectral mode interpretation.** Extend the existing Laplacian work
   from `story_shapes_full.ipynb` Part 2 by computing each eigenvector's
   correlation with hand-labelled features (entropy, quote density, mean
   sentence length) on a held-out corpus. Eigenvectors that survive
   labelling on a *different* corpus are candidate canonical axes.

### 5.3 The canonical-axis test

A direction in latent space is *canonical* iff:

- it correlates with a recognisable feature (so it has meaning),
- the correlation reproduces on a corpus the basis was not fit on (so the
  meaning generalises), and
- moving along it changes the *narrative* of a generated passage in a
  recognisable way (so it is causally implicated, not just descriptively).

The third condition is the hardest and is the natural connection point to
§4: once we have a candidate canonical axis, we use the activation-steering
technique from §4.3.2 to push a generated passage along it and see whether
the change is the change we expected.

### 5.4 Deliverables

- `canonical_bases.ipynb` — the three constructions above, side by side,
  on the canonical-books corpus.
- A short table: top 5 canonical axes, the feature each best correlates
  with, the cross-corpus correlation, and a generated example of moving
  along it.

---

## 6. Theoretical track D — Stories as dynamical systems

**Goal.** Treat each book's latent trajectory not as a static curve but as
the *integral curve* of an underlying vector field on the latent space.
That is, learn a function v : ℝᵈ → ℝᵈ such that

> dz/dt ≈ v(z(t))

holds across the corpus, where `z(t)` is the per-passage (or per-token)
latent at narrative position `t`. Once we have `v`, narrative becomes a
dynamical system, and the full toolkit of dynamical-systems analysis —
fixed points, basins, Lyapunov exponents, Poincaré sections — becomes
available for narrative.

### 6.1 Why this is the right object

The "beads on a string" picture from `concept_shapes_v2` is purely
descriptive: each story is its own polyline and the polylines exist in
isolation. A learned vector field is the *generative law* behind those
polylines. It says: from any point in the latent, here is the direction
narrative *tends to* move next.

This buys us four things at once that the other tracks only get partially:

- **A test for narrative regularity.** If `v` fits the corpus well, then
  there really is a "physics" of narrative in this representation. If it
  fits poorly, narratives are too idiosyncratic to be modelled as a single
  dynamical system, and we should ask whether they cluster into
  *families* of dynamical systems instead (one per genre? per author?).
- **Attractors as archetypes.** Fixed points of `v` (places where
  v(z) ≈ 0) are passages where narrative "lingers"; stable manifolds and
  basins of attraction are recurring narrative situations. This is a
  *continuous* and *causal* version of the discrete motif dictionary in
  §3 — and the two should be cross-checked against each other.
- **A surprise signal.** For any actual passage, the residual
  `r(t) = (z(t+1) − z(t)) − v(z(t))` is what the field *did not predict*.
  Large residuals are the moments where the story departs from the
  ambient flow — the turn, the reveal, the scene break. This is a
  precise, model-agnostic definition of *narrative novelty*.
- **A generative engine.** Sampling = pick an initial condition, integrate
  forward. This sidesteps the decoder problem in §4 by keeping the whole
  generative loop in latent space, then handing the trajectory to the
  decoder of §4.3 only at the end.

### 6.2 Methods to learn the field

Four candidate methods, in roughly increasing sophistication:

1. **Local regression.** The crudest version: for each point `z` in the
   corpus, fit a local linear or kernel regression of the next-step
   displacement `Δz` on the surrounding cloud. No training, immediate.
   Useful as a baseline and as a sanity check that there *is* a field
   to learn (i.e. that nearby latents tend to flow in similar directions).
2. **Neural ODE.** Parameterise `v` as a small MLP `v_θ(z)` and train
   with the adjoint method (`torchdiffeq`) so that integrating `v_θ`
   from `z(0)` reproduces the observed trajectory of each story. The
   loss is per-step or per-trajectory MSE in latent space. This is the
   canonical formulation and is well-supported by tooling.
3. **Flow matching.** A modern alternative to Neural ODE that avoids
   solving an ODE during training: regress `v_θ(z, t)` directly onto a
   target velocity defined by a chosen probability path between an
   easy distribution and the data distribution (Lipman et al. 2023).
   Faster to train, more stable, and gives a vector field by construction.
4. **Stochastic / Koopman.** Two extensions worth trying after the
   deterministic field is in place:
   - **Stochastic.** Replace the ODE with an SDE
     `dz = v(z) dt + σ(z) dW`. The drift `v` is the deterministic part
     and `σ` captures the *unpredictability* of narrative at each point.
     This is the principled way to put a number on "how much branching
     is possible from here?" — useful for §4 (interpolation) and for
     surprise quantification.
   - **Koopman.** Lift the latent into a higher-dimensional space of
     observables (e.g. polynomial features, or learned via a deep
     Koopman autoencoder) where the dynamics become *linear*. The
     Koopman operator's eigenfunctions are the dynamical analogue of
     §5's canonical bases — coordinates in which narrative evolves as
     a simple linear flow.

### 6.3 Conditioning and content-independence

A single global field probably overfits to the dominant content of the
corpus. Two ways to handle this:

- **Conditional fields.** Make `v` a function of `(z, c)` where `c` is a
  book or genre embedding, and train as before. The *family* of fields
  indexed by `c` is then itself an object to study — distances between
  fields, shared fixed points across `c`, etc.
- **Quotient fields.** Project out the §5 content axes from `z` before
  fitting `v`. The resulting field lives on the *content-independent*
  subspace and is by construction what we'd want to call the "narrative
  field" rather than the "Austen field" or the "physics-article field".

The two are complementary: conditional fields tell us what is
*book-specific*, quotient fields tell us what is *book-invariant*.

### 6.4 Analysis once the field exists

Concrete things to compute and plot:

- **Streamline plot.** Project `v` to the §1 UMAP and draw streamlines.
  Overlay actual story trajectories. Visually obvious when the fit is
  good; visually obvious when it isn't.
- **Fixed-point catalogue.** Run a root-finder on `v_θ(z) = 0` from many
  random initialisations. For each fixed point, classify by Jacobian
  spectrum (sink, source, saddle, spiral) and retrieve the nearest
  corpus passages. Compare against the §3 dictionary codes and against
  the §5 canonical-axis features.
- **Lyapunov exponent along each story.** Approximate the largest
  Lyapunov exponent along each book's trajectory. High → narrative is
  chaotic / sensitive to small perturbations; low → predictable. This
  is a per-book scalar that can be cross-checked against literary
  judgement (we expect *Tristram Shandy* high, a procedural mystery low).
- **Residual surprise track.** Plot `‖z(t+1) − z(t) − v(z(t))‖` along
  each book. The peaks should align with chapter breaks, climaxes,
  and POV shifts. If they do, this is one of the most *legible*
  outputs the project produces.

### 6.5 Connections to the other tracks

This track is the integration point for everything else:

- **§2 (TDA).** Persistent homology of the *vector field* (rather than
  the point cloud) gives a topologically sound count of attractors and
  basins. Tools: discrete Morse theory on the field's gradient.
- **§3 (dictionary).** Fixed points and limit cycles of `v` are the
  continuous analogue of discrete motifs. They should *agree* with the
  VQ codes from §3 — that is the cross-check.
- **§4 (interpolation).** The geodesic between two stories is now
  *defined*: it is the integral curve of `v` connecting them, or in the
  stochastic case, the most-likely path between them under the SDE.
  This replaces the ad-hoc slerp/OT choices in §4.2 with a learned,
  data-driven interpolation.
- **§5 (canonicity).** The Jacobian of `v` at a typical point is a
  linear operator on the latent space. Its eigenvectors are the *local*
  canonical axes — the directions narrative is locally pulling along.
  Averaging over the corpus gives a candidate global basis to compare
  with §5's constructions.

### 6.6 Deliverables

- `vector_field_neuralode.ipynb` — fit a Neural ODE to the canonical-books
  trajectories from §1, show the streamline plot over the §1 UMAP, plot
  the per-book residual surprise track.
- `vector_field_analysis.ipynb` — fixed-point catalogue, Lyapunov
  exponents per book, comparison with the §3 dictionary codes.
- A figure: streamlines + actual book trajectories + retrieved passages
  at each fixed point. This is the most striking single image the
  project can produce — it makes "stories as dynamical systems"
  immediately legible.

---

## 7. Suggested ordering

The tracks above are independent but they enable each other in a specific
order. The path of least resistance is:

1. **§1 first** — without a Mac-mini extraction script we can't run
   anything new, and reproducing the v2 result on a larger model is the
   single most informative experiment we can do.
2. **§2.2 next** — persistent homology on the §1 latents is one notebook
   of work and produces the first new figure (persistence diagrams of
   canonical books).
3. **§3 dictionary** — the discrete-motif view that everything else
   builds on. VQ-VAE first, HMM only if the discrete view holds up.
4. **§5 canonical bases** — needs the larger-model latents from §1 but
   not the dictionary from §3, so it can run in parallel with §3.
5. **§6 vector field (Neural ODE / flow matching)** — the integration
   point for everything else. Best run after §3 and §5 because the
   dictionary codes and canonical axes are needed for the cross-checks
   in §6.5, but the streamline plot in §6.4 is striking enough on its
   own that a first pass can happen as soon as §1 is done.
6. **§4 interpolation** — the headline generative result. Once §6 exists,
   the interpolation problem is reframed as integrating `v` between two
   endpoints, which is cleaner than the slerp/OT options in §4.2.

Each of §2 through §6 is a self-contained notebook plus a short writeup.
None of them require new infrastructure beyond §1.

---

## 8. Open questions

A few things this document doesn't resolve and that should be confronted
explicitly before committing to any of the tracks above:

- **Free generation vs. teacher forcing.** `concept_shapes_v2` extracts
  hidden states during *free generation* from a prefix; the
  `story_shapes_*` notebooks extract them during a *forward pass over
  fixed text*. These produce different latents and the difference matters
  for §4 (interpolation). Pick one regime as canonical for the new
  pipeline, or extract both and label them.
- **Tokenizer drift.** Moving from GPT-2's BPE to Llama's SentencePiece
  changes what a "token" is. The W=64 passage choice in
  `story_shapes_demo` is calibrated to GPT-2 tokens; recalibrate for the
  new tokenizer or convert to character-level windows.
- **Where does the channel end?** We have been treating the residual
  stream as the channel, but the attention patterns are also part of the
  computation. A small experiment: include a per-layer attention summary
  (e.g. attention entropy) as an extra column in the delta stack and see
  whether it changes the ARI curve.
- **Ground truth for "narrative shape".** We don't have one. The
  cross-book mode-sharing test from `story_shapes_full.ipynb` Part 2 is
  the closest thing to a quantitative anchor; everything else is
  qualitative. Worth thinking about whether there is any external label
  set (TV-Tropes, Vonnegut's "shapes of stories" categories, beat-sheet
  annotations) that we could pin against.
