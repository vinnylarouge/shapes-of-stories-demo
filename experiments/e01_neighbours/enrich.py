"""
E01 — Cross-book nearest passages.

For each passage, find the K nearest passages from *other* books in the
3D latent (the `(x3d, y3d, z3d)` columns) and attach them to the passage
as `neighbours`. Writes the enriched JSON in place.

Usage:
    python experiments/e01_neighbours/enrich.py \\
        --in public/story_shapes.json \\
        --out public/story_shapes.json \\
        --k 5

The output schema is documented in experiments/EXPERIMENTS.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def enrich(in_path: Path, out_path: Path, k: int) -> None:
    data = json.loads(in_path.read_text())
    books = data["books"]

    # Flatten into one (N, 3) array plus a parallel index of (book, passage).
    coords: list[tuple[float, float, float]] = []
    owners: list[tuple[int, int]] = []
    for bi, book in enumerate(books):
        for pi, p in enumerate(book["passages"]):
            coords.append((p["x3d"], p["y3d"], p["z3d"]))
            owners.append((bi, pi))

    pts = np.asarray(coords, dtype=np.float32)
    book_of = np.asarray([o[0] for o in owners], dtype=np.int32)
    n = len(pts)

    if n == 0:
        raise SystemExit("No passages found in input JSON.")
    if k >= n:
        raise SystemExit(f"k={k} is too large for n={n} passages.")

    # Pairwise squared distances. n is small (~2k), so the (n,n) matrix
    # fits comfortably in memory and brute force is faster than building
    # a tree.
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = (diff * diff).sum(-1)

    # Mask out same-book and self.
    same_book = book_of[:, None] == book_of[None, :]
    d2_masked = np.where(same_book, np.inf, d2)

    # Top-k smallest per row. argpartition then sort the K winners.
    part = np.argpartition(d2_masked, kth=k, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    order = np.argsort(d2_masked[rows, part], axis=1)
    nearest = part[rows, order]
    nearest_d = np.sqrt(d2_masked[rows, nearest])

    # Write back into the JSON in original (book, passage) order.
    flat_idx = 0
    for bi, book in enumerate(books):
        for pi, p in enumerate(book["passages"]):
            neigh = []
            for j in range(k):
                global_idx = int(nearest[flat_idx, j])
                nb_book, nb_passage = owners[global_idx]
                neigh.append(
                    {
                        "b": int(nb_book),
                        "p": int(nb_passage),
                        "d": round(float(nearest_d[flat_idx, j]), 4),
                    }
                )
            p["neighbours"] = neigh
            flat_idx += 1

    # Stamp the metadata so the frontend can show what it's reading.
    data.setdefault("metadata", {})["neighbours_k"] = k

    out_path.write_text(json.dumps(data))
    print(
        f"Enriched {n} passages with k={k} cross-book neighbours. "
        f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=Path("public/story_shapes.json"))
    ap.add_argument("--out", type=Path, default=Path("public/story_shapes.json"))
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    enrich(args.inp, args.out, args.k)


if __name__ == "__main__":
    main()
