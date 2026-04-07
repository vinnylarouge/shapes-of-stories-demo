# E07 — Mac-mini local activation extraction

Replaces the Colab/CUDA assumptions in `backend/story_shapes_demo.ipynb`
with a scriptable two-step CLI that runs locally on Apple Silicon (or
CUDA, or CPU) and produces a drop-in replacement for the existing
`public/story_shapes.json`.

The split is:

1. `extract_activations.py` — heavy step. Loads a HuggingFace causal
   LM, processes the corpus in overlapping chunks, writes per-token
   hidden states for the chosen layers to a memory-mapped fp16 file.
2. `build_story_shapes_json.py` — light step. Pools tokens into
   passages, runs PCA → AE → UMAP, computes prose-mode features,
   emits the frontend JSON in the existing schema.

Both steps are independent: re-pool / re-train the AE without
re-extracting; swap models without rewriting the post-processing.

## Dependencies

```bash
pip install torch transformers datasets pyyaml tqdm \
            numpy scikit-learn umap-learn pyarrow
```

For 4-bit quantised loading on big models add `bitsandbytes` (Linux/CUDA
only — on Mac use bf16/fp16).

## Step 1 — extract activations

```bash
python experiments/e07_extract/extract_activations.py \
    --model meta-llama/Llama-3.2-3B \
    --config experiments/e07_extract/configs/canonical_books.yaml \
    --layers all \
    --chunk 1024 --overlap 512 \
    --out runs/llama32-3b-canon
```

`--layers` accepts:

- `all` — every hidden state including the embedding
- `lastK` — the K final layers (e.g. `last4`)
- comma-separated indices: `0,8,16,24`

The default `--keep-second-half` flag keeps only tokens from the second
half of each chunk so every stored token has full left context (no
chunk-boundary artefacts) — same logic as the existing notebook.

### Outputs in `--out/`

| File | Format | What |
|---|---|---|
| `hidden.bin` | fp16 mmap | shape `(N, L, D)` |
| `meta.parquet` (or `meta.jsonl`) | row-per-token | `seq_id`, `title`, `chunk_pos`, `in_chunk_pos`, `abs_pos`, `token_id`, `token_str`, `entropy` |
| `config.json` | JSON | model name, layers stored, chunk policy, git SHA |

### Memory budget worked example — Llama-3.2-3B

- L = 29 (28 transformer layers + embedding)
- D = 3072
- 30 books × ~5000 kept tokens/book ≈ 150k tokens
- Activations: `150k × 29 × 3072 × 2B ≈ 26.7 GB`

Too large for unified memory + the model. Two mitigations, both
straightforward additions to `extract_activations.py`:

1. **Layer subset.** `--layers last8` cuts to 8 layers, ~7.4 GB. Run
   the per-layer ARI sweep from `concept_shapes_v2.ipynb` against the
   §1 result first to identify which layers to keep.
2. **Online PCA.** Add an `IncrementalPCA(n_components=1024)` per layer
   in `extract_activations.py` that's `partial_fit`-ed on each chunk
   and only stores the projection. Reduces D from 3072 to 1024 (3.3×
   reduction) and is the cleanest way to fit Llama-3.2-3B comfortably
   in 32 GB.

## Step 2 — build the frontend JSON

```bash
python experiments/e07_extract/build_story_shapes_json.py \
    --run runs/llama32-3b-canon \
    --layer last \
    --window 64 \
    --ae-k 32 \
    --out public/story_shapes.json
```

`--layer` accepts an integer (relative to the layers stored in step 1)
or `last`.

The output JSON has exactly the same shape as `public/story_shapes.json`
today, so the existing frontend renders it without code changes — and
all of E01–E05 can be re-run on it unchanged.

## Then re-run E01–E05

After E07 lands a new `story_shapes.json`, the right sequence is:

```bash
python experiments/e01_neighbours/enrich.py
python experiments/e02_persistence/compute.py
python experiments/e03_motifs/compute.py
python experiments/e04_vector_field/compute.py
python experiments/e05_canonical/compute.py
```

Each script enriches the JSON additively. Order is independent except
that they all read the same file in place.

## Notes

- The script does **not** assume a particular tokenizer family — Llama,
  Qwen, Mistral, Gemma all work via `AutoModelForCausalLM` /
  `AutoTokenizer`. The `seg_len` and `chunk` defaults are calibrated to
  GPT-2 BPE; for SentencePiece tokenizers you may want to bump
  `seg_len` proportionally to keep the same character count.
- The MPS path uses fp16. bf16 on MPS is partial as of torch 2.4 — if
  the model misbehaves, force fp16 explicitly with
  `--torch-dtype float16` (TODO: surface as a CLI flag).
- For multi-billion-parameter models that don't fit in unified memory,
  add `device_map="auto"` and `load_in_8bit=True` (Linux/CUDA) or use
  `mlx-lm` as a separate backend. Both are clean swap-ins for the
  `from_pretrained` line.
- This script is the §1 deliverable from `DIRECTIONS.md`. The §1.5
  validation experiment (per-layer ARI on Wikipedia, expecting a higher
  peak than GPT-2) should be run before any of the other experiments
  on Llama output.
