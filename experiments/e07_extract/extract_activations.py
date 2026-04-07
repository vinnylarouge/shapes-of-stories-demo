"""
E07 — Mac-mini local activation extraction CLI.

Loads a HuggingFace causal LM (default: Llama-3.2-3B), processes a
corpus of books in overlapping chunks, and writes per-token hidden
states for a configurable set of layers to a memory-mapped fp16 file.

Designed to run on Apple Silicon with the MPS backend, but transparently
falls back to CUDA or CPU. Replaces the Colab assumptions in
`backend/story_shapes_demo.ipynb` with a scriptable, reproducible CLI.

Usage:
    python experiments/e07_extract/extract_activations.py \\
        --model meta-llama/Llama-3.2-3B \\
        --config experiments/e07_extract/configs/canonical_books.yaml \\
        --layers all \\
        --chunk 1024 --overlap 512 \\
        --out runs/llama32-3b-canon

Outputs (under --out):
    hidden.bin       — fp16 (N, L, D) memory-mapped activations
    meta.parquet     — per-stored-token row with seq_id, position,
                       book, token, entropy, and other features
                       (or meta.jsonl if pyarrow is unavailable)
    config.json      — model name, tokenizer info, chunk policy,
                       layer indices stored, git SHA

Pair with `build_story_shapes_json.py` to turn the run into a
`public/story_shapes.json` drop-in replacement.

Dependencies:
    pip install torch transformers datasets pyyaml pyarrow tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

# Heavy deps imported lazily inside main() so `--help` works without them.


@dataclass
class RunConfig:
    model: str
    chunk: int
    overlap: int
    layers: list[int]
    seg_len: int
    min_book_len: int
    books: list[str]
    out: str
    started_at: float
    git_sha: str | None


def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def git_sha(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def load_corpus(config_path: Path) -> tuple[dict, list[tuple[str, str]]]:
    """Returns (cfg_dict, [(title, text), ...]). Streams pg19 once."""
    import yaml
    from datasets import load_dataset

    cfg = yaml.safe_load(config_path.read_text())
    wanted = [w for w in cfg["books"]]
    wanted_lower = [w.lower() for w in wanted]
    found: dict[str, str] = {}

    pg = load_dataset("emozilla/pg19", split="train", streaming=True)
    for ex in pg:
        title = ex.get("short_book_title") or ex.get("title", "")
        title_l = title.lower()
        for wi, w in enumerate(wanted_lower):
            if w in title_l and wanted[wi] not in found:
                if len(ex["text"]) >= cfg["min_book_len"]:
                    found[wanted[wi]] = ex["text"]
                break
        if len(found) == len(wanted):
            break

    out: list[tuple[str, str]] = []
    for w in wanted:
        if w in found:
            out.append((w, found[w]))
    return cfg, out


def chunk_token_window(token_ids: list[int], chunk: int, overlap: int) -> Iterable[tuple[int, list[int]]]:
    stride = chunk - overlap
    pos = 0
    while pos + 1 < len(token_ids):
        ids = token_ids[pos : pos + chunk]
        if len(ids) < 8:
            break
        yield pos, ids
        if pos + chunk >= len(token_ids):
            break
        pos += stride


def parse_layers(spec: str, n_layers: int) -> list[int]:
    if spec == "all":
        return list(range(n_layers))
    if spec.startswith("last"):
        k = int(spec[4:]) if len(spec) > 4 else 1
        return list(range(max(0, n_layers - k), n_layers))
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--config", type=Path, default=Path("experiments/e07_extract/configs/canonical_books.yaml"))
    ap.add_argument("--layers", default="all", help="'all', 'lastK', or comma-separated indices")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=512)
    ap.add_argument("--keep-second-half", action="store_true", default=True,
                    help="Only keep tokens from the second half of each chunk (full-context only)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    args.out.mkdir(parents=True, exist_ok=True)
    device, dtype = pick_device()
    print(f"Device: {device}, dtype: {dtype}")

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers + 1  # includes embedding
    d_model = model.config.hidden_size
    layer_idx = parse_layers(args.layers, n_layers)
    print(f"Model has {n_layers} hidden states (incl. embedding), d={d_model}. Storing layers {layer_idx}.")

    print(f"Loading corpus from {args.config}...")
    cfg, corpus = load_corpus(args.config)
    print(f"Found {len(corpus)} books in pg19 (out of {len(cfg['books'])} requested).")

    # Tokenise + take middle segment
    book_tokens: list[tuple[str, list[int]]] = []
    seg_len = cfg["seg_len"]
    for title, text in corpus:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < seg_len:
            book_tokens.append((title, ids))
        else:
            mid = len(ids) // 2
            half = seg_len // 2
            book_tokens.append((title, ids[mid - half : mid + half]))

    # Estimate total stored tokens to preallocate hidden.bin
    half = args.chunk // 2 if args.keep_second_half else 0
    total_kept = 0
    for _, ids in book_tokens:
        for pos, ch in chunk_token_window(ids, args.chunk, args.overlap):
            keep = len(ch) - half
            total_kept += max(0, keep)

    L = len(layer_idx)
    print(f"Allocating hidden.bin: ({total_kept}, {L}, {d_model}) fp16 = "
          f"{total_kept * L * d_model * 2 / 1e9:.2f} GB")
    hidden_path = args.out / "hidden.bin"
    storage = np.memmap(hidden_path, dtype=np.float16, mode="w+",
                        shape=(total_kept, L, d_model))

    meta_rows: list[dict] = []
    write_idx = 0
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    pbar = tqdm(total=total_kept, desc="tokens")
    for seq_id, (title, ids) in enumerate(book_tokens):
        token_pos_in_book = 0
        for chunk_pos, chunk_ids in chunk_token_window(ids, args.chunk, args.overlap):
            input_ids = torch.tensor([chunk_ids], device=device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)
            hs = out.hidden_states  # tuple length n_layers, each (1, T, D)

            T = input_ids.shape[1]
            start = T // 2 if args.keep_second_half else 0
            kept = T - start

            # Per-token entropy from logits at each kept position
            logits = out.logits[0, start:, :]
            lp = F.log_softmax(logits, dim=-1)
            ent = (-(lp.exp() * lp).sum(-1)).cpu().float().numpy()

            stack = np.stack(
                [hs[li][0, start:, :].cpu().float().numpy().astype(np.float16) for li in layer_idx],
                axis=1,
            )  # (kept, L, D)

            storage[write_idx : write_idx + kept] = stack
            for k in range(kept):
                tid = int(chunk_ids[start + k])
                meta_rows.append({
                    "seq_id": seq_id,
                    "title": title,
                    "chunk_pos": chunk_pos,
                    "in_chunk_pos": start + k,
                    "abs_pos": chunk_pos + start + k,
                    "token_id": tid,
                    "token_str": tokenizer.decode([tid]),
                    "entropy": float(ent[k]),
                })
            write_idx += kept
            pbar.update(kept)

            del out, hs, logits, lp, stack
        token_pos_in_book += 1
    pbar.close()

    storage.flush()

    # Write meta — try pyarrow parquet, fall back to jsonl.
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pylist(meta_rows)
        pq.write_table(table, args.out / "meta.parquet")
        meta_kind = "parquet"
    except ImportError:
        with (args.out / "meta.jsonl").open("w") as f:
            for row in meta_rows:
                f.write(json.dumps(row) + "\n")
        meta_kind = "jsonl"

    cfg_out = RunConfig(
        model=args.model,
        chunk=args.chunk,
        overlap=args.overlap,
        layers=layer_idx,
        seg_len=cfg["seg_len"],
        min_book_len=cfg["min_book_len"],
        books=[t for t, _ in book_tokens],
        out=str(args.out),
        started_at=time.time(),
        git_sha=git_sha(Path(__file__).resolve().parent.parent.parent),
    )
    (args.out / "config.json").write_text(json.dumps(asdict(cfg_out), indent=2))

    print(f"\nDone. Stored {write_idx} tokens × {L} layers × {d_model} dims to {hidden_path}.")
    print(f"Metadata: {args.out}/meta.{meta_kind}")
    print(f"Config:   {args.out}/config.json")


if __name__ == "__main__":
    main()
