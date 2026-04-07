"""
E02 — Per-book persistence signatures.

For each book, treat its passage trajectory as a point cloud in 3D
latent space. Compute the H₀ persistence diagram via single-linkage
(equivalent to Vietoris-Rips H₀), reduce it to a fixed-length signature
of the top-N most persistent components, then cluster all books in
signature space to assign each one a `shape_archetype` label.

Pure numpy + stdlib. No scipy, no sklearn, no ripser. The signature is
a Wasserstein-1 proxy on the H₀ death values, which gives the same
clustering intuition as the full bottleneck distance for our purposes.

Usage:
    python experiments/e02_persistence/compute.py \\
        --in public/story_shapes.json \\
        --out public/story_shapes.json \\
        --top 8 \\
        --archetypes 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def h0_deaths(points: np.ndarray) -> np.ndarray:
    """Single-linkage merge heights = H₀ death times for the Vietoris-Rips
    filtration. All H₀ classes are born at filtration value 0."""
    n = len(points)
    if n < 2:
        return np.zeros(0, dtype=np.float32)

    diff = points[:, None, :] - points[None, :, :]
    d = np.sqrt((diff * diff).sum(-1))
    iu, ju = np.triu_indices(n, k=1)
    pair_d = d[iu, ju]
    order = np.argsort(pair_d)

    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    deaths: list[float] = []
    for k in order:
        i, j = int(iu[k]), int(ju[k])
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            deaths.append(float(pair_d[k]))
            if len(deaths) == n - 1:
                break
    return np.asarray(deaths, dtype=np.float32)


def signature(deaths: np.ndarray, top: int) -> np.ndarray:
    """Top-N most persistent H₀ deaths, padded to fixed length with 0s."""
    sig = np.zeros(top, dtype=np.float32)
    if deaths.size > 0:
        sorted_d = np.sort(deaths)[::-1]
        m = min(top, sorted_d.size)
        sig[:m] = sorted_d[:m]
    return sig


def kmeans_simple(X: np.ndarray, k: int, n_iter: int = 50, seed: int = 42) -> np.ndarray:
    """Lloyd's k-means in pure numpy. Returns integer labels of length len(X)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    if k >= n:
        return np.arange(n) % k
    centroids = X[rng.choice(n, size=k, replace=False)].copy()
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        labels = np.argmin(d2, axis=1)
        new_c = np.stack(
            [
                X[labels == c].mean(0) if (labels == c).any() else centroids[c]
                for c in range(k)
            ]
        )
        if np.allclose(new_c, centroids):
            break
        centroids = new_c
    return labels


def compute(in_path: Path, out_path: Path, top: int, archetypes: int) -> None:
    data = json.loads(in_path.read_text())
    books = data["books"]

    sigs = []
    for book in books:
        pts = np.asarray(
            [(p["x3d"], p["y3d"], p["z3d"]) for p in book["passages"]],
            dtype=np.float32,
        )
        deaths = h0_deaths(pts)
        sig = signature(deaths, top)
        sigs.append(sig)

        # Persistence pairs in the schema described in EXPERIMENTS.md.
        # Births are 0 for H₀; we keep the top `top` deaths.
        sorted_d = np.sort(deaths)[::-1][:top].tolist() if deaths.size else []
        book["persistence"] = [
            {"d": 0, "birth": 0.0, "death": round(float(x), 4)}
            for x in sorted_d
        ]

    sig_matrix = np.stack(sigs)  # (n_books, top)
    labels = kmeans_simple(sig_matrix, archetypes)
    for book, lbl in zip(books, labels):
        book["shape_archetype"] = int(lbl)

    data.setdefault("metadata", {})
    data["metadata"]["persistence_top"] = top
    data["metadata"]["shape_archetypes"] = int(archetypes)

    out_path.write_text(json.dumps(data))
    print(
        f"Wrote {len(books)} per-book persistence signatures "
        f"(top={top}, archetypes={archetypes}) to {out_path} "
        f"({out_path.stat().st_size / 1e6:.2f} MB)."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=Path("public/story_shapes.json"))
    ap.add_argument("--out", type=Path, default=Path("public/story_shapes.json"))
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--archetypes", type=int, default=4)
    args = ap.parse_args()
    compute(args.inp, args.out, args.top, args.archetypes)


if __name__ == "__main__":
    main()
