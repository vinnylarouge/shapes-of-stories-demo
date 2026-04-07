"""
E07 — Build the frontend's `story_shapes.json` from an extraction run.

Reads the (hidden.bin, meta) pair produced by `extract_activations.py`
and turns it into a `public/story_shapes.json` drop-in replacement
following the same passage-windowed pipeline as the existing
`backend/story_shapes_demo.ipynb`:

  1. Pool per-token hidden states into passage windows of W tokens.
  2. PCA → 128-d.
  3. Train a small autoencoder to K-d (default K=32).
  4. UMAP to 2D and 3D for the canvas.
  5. Compute prose-mode features (dialogue %, mean sentence length,
     mean entropy) on the same passage windows.
  6. Emit JSON in the existing schema; the frontend reads it without
     code changes.

Usage:
    python experiments/e07_extract/build_story_shapes_json.py \\
        --run runs/llama32-3b-canon \\
        --layer last \\
        --window 64 \\
        --ae-k 32 \\
        --out public/story_shapes.json

Dependencies:
    pip install torch numpy scikit-learn umap-learn pyarrow tqdm
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Heavy deps lazy-imported in main().


SENT_BOUNDARY = re.compile(r"[.!?]+")
DIALOGUE_QUOTE = re.compile(r"[\"\u201c\u201d]")


def passage_features(text: str) -> tuple[float, float]:
    """Returns (dialogue_fraction, mean_sentence_length)."""
    if not text.strip():
        return 0.0, 0.0
    quotes = len(DIALOGUE_QUOTE.findall(text))
    # Approximate dialogue fraction by even-odd quote spans.
    inside = False
    in_chars = 0
    for ch in text:
        if ch in '"\u201c\u201d':
            inside = not inside
        elif inside:
            in_chars += 1
    diag = in_chars / max(1, len(text))

    sents = [s for s in SENT_BOUNDARY.split(text) if s.strip()]
    sent_words = [len(s.split()) for s in sents]
    msl = sum(sent_words) / max(1, len(sent_words))
    return diag, msl


def load_meta(run: Path):
    """Load meta as a list of dicts. Tries parquet then jsonl."""
    pq_path = run / "meta.parquet"
    if pq_path.exists():
        import pyarrow.parquet as pq
        return pq.read_table(pq_path).to_pylist()
    js_path = run / "meta.jsonl"
    if js_path.exists():
        return [json.loads(line) for line in js_path.read_text().splitlines() if line.strip()]
    raise FileNotFoundError(f"No meta.parquet or meta.jsonl found in {run}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--layer", default="last", help="layer index, or 'last' for the final stored layer")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--ae-k", type=int, default=32)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("public/story_shapes.json"))
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.decomposition import PCA
    import umap

    cfg = json.loads((args.run / "config.json").read_text())
    meta = load_meta(args.run)
    if not meta:
        raise SystemExit("Empty metadata.")

    n_tokens = len(meta)
    layer_idx = cfg["layers"]
    L = len(layer_idx)
    d_model = None  # determined from disk

    hidden_path = args.run / "hidden.bin"
    # We do not know d_model from cfg alone — read it back from file size.
    n_bytes = hidden_path.stat().st_size
    d_model = n_bytes // (n_tokens * L * 2)
    print(f"Loaded run: {n_tokens} tokens, {L} layers, d={d_model}")

    hidden = np.memmap(hidden_path, dtype=np.float16, mode="r",
                       shape=(n_tokens, L, d_model))

    if args.layer == "last":
        sel = L - 1
    else:
        sel = int(args.layer)
        if sel not in range(L):
            raise SystemExit(f"--layer {sel} not in stored layers {list(range(L))}")

    # Group tokens by book and pool into passage windows.
    book_titles: list[str] = []
    book_to_idx: dict[str, int] = {}
    for row in meta:
        if row["title"] not in book_to_idx:
            book_to_idx[row["title"]] = len(book_titles)
            book_titles.append(row["title"])

    # Bucket token rows by book in original order
    rows_by_book: list[list[int]] = [[] for _ in book_titles]
    for ri, row in enumerate(meta):
        rows_by_book[book_to_idx[row["title"]]].append(ri)

    print(f"Pooling into W={args.window} passages...")
    passage_vecs: list[np.ndarray] = []
    passage_meta: list[dict] = []
    for bi, rows in enumerate(rows_by_book):
        title = book_titles[bi]
        n_pass = len(rows) // args.window
        if n_pass == 0:
            continue
        for k in range(n_pass):
            window_rows = rows[k * args.window : (k + 1) * args.window]
            block = np.stack([hidden[r, sel, :].astype(np.float32) for r in window_rows])
            mean_vec = block.mean(0)
            mean_ent = float(np.mean([meta[r]["entropy"] for r in window_rows]))
            text = " ".join(meta[r]["token_str"] for r in window_rows)
            diag, msl = passage_features(text)
            passage_vecs.append(mean_vec)
            passage_meta.append(
                {
                    "book": bi,
                    "title": title,
                    "passage_idx": k,
                    "n_passages": n_pass,
                    "pos": (k + 0.5) / n_pass,
                    "text": text[:200],
                    "dialogue": float(diag),
                    "entropy": mean_ent,
                    "sent_len": float(msl),
                }
            )

    X = np.stack(passage_vecs)
    print(f"Passages: {X.shape}")

    print(f"PCA → {args.pca} ...")
    pca = PCA(n_components=min(args.pca, X.shape[1], X.shape[0]), random_state=42)
    X_pca = pca.fit_transform(X)

    print(f"Training AE: {X_pca.shape[1]} → {args.ae_k} ...")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    class AE(nn.Module):
        def __init__(self, ind: int, hid: int, lat: int):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(ind, hid), nn.ReLU(), nn.Linear(hid, lat))
            self.dec = nn.Sequential(nn.Linear(lat, hid), nn.ReLU(), nn.Linear(hid, ind))

        def forward(self, x):
            z = self.enc(x)
            return self.dec(z), z

    ae = AE(X_pca.shape[1], 256, args.ae_k).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(torch.tensor(X_pca, dtype=torch.float32)),
                    batch_size=256, shuffle=True)
    for ep in range(50):
        for (b,) in dl:
            b = b.to(device)
            r, _ = ae(b)
            loss = F.mse_loss(r, b)
            opt.zero_grad()
            loss.backward()
            opt.step()
    ae.eval()
    with torch.no_grad():
        latents = ae.enc(torch.tensor(X_pca, dtype=torch.float32, device=device)).cpu().numpy()

    print("UMAP to 2D and 3D ...")
    umap2 = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(latents)
    umap3 = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(latents)

    # Build the JSON in the existing schema.
    by_book: dict[int, list[int]] = {}
    for j, m in enumerate(passage_meta):
        by_book.setdefault(m["book"], []).append(j)

    books_out: list[dict] = []
    for bi, title in enumerate(book_titles):
        idxs = by_book.get(bi, [])
        if not idxs:
            continue
        passages = []
        for j in idxs:
            m = passage_meta[j]
            passages.append(
                {
                    "x": round(float(umap2[j, 0]), 4),
                    "y": round(float(umap2[j, 1]), 4),
                    "x3d": round(float(umap3[j, 0]), 4),
                    "y3d": round(float(umap3[j, 1]), 4),
                    "z3d": round(float(umap3[j, 2]), 4),
                    "pos": round(m["pos"], 3),
                    "text": m["text"],
                    "dialogue": round(m["dialogue"], 3),
                    "entropy": round(m["entropy"], 2),
                    "sent_len": round(m["sent_len"], 1),
                }
            )
        full_length = len(idxs) * args.window
        books_out.append(
            {
                "title": title,
                "full_length": int(full_length),
                "archetype": -1,
                "passages": passages,
            }
        )

    out_data = {
        "metadata": {
            "model": cfg["model"],
            "layer": int(sel),
            "window": int(args.window),
            "ae_k": int(args.ae_k),
            "n_books": len(books_out),
            "n_passages": int(sum(len(b["passages"]) for b in books_out)),
        },
        "books": books_out,
    }
    args.out.write_text(json.dumps(out_data))
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB).")


if __name__ == "__main__":
    main()
