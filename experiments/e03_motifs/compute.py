"""
E03 — Motif dictionary (vector-quantised passage codes).

Cluster all passage latents into K codes via Lloyd's k-means in pure
numpy. Each passage is assigned its nearest code; each book gets a
length-K histogram of code usage.

This is the placeholder version (k-means on the post-UMAP 3D
coordinates). The real version of E03 in `EXPERIMENTS.md` runs on the
raw AE latent that the §1 (E07) extraction will produce; the same
script then runs unchanged on that richer input.

Usage:
    python experiments/e03_motifs/compute.py \\
        --in public/story_shapes.json \\
        --out public/story_shapes.json \\
        --k 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def kmeans(X: np.ndarray, k: int, n_iter: int = 100, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's k-means with k-means++ init. Returns (labels, centroids)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    if k >= n:
        raise ValueError(f"k={k} must be < n_samples={n}")

    # k-means++ init
    centroids = np.empty((k, d), dtype=X.dtype)
    centroids[0] = X[rng.integers(n)]
    closest_d2 = ((X - centroids[0]) ** 2).sum(-1)
    for c in range(1, k):
        probs = closest_d2 / closest_d2.sum()
        idx = rng.choice(n, p=probs)
        centroids[c] = X[idx]
        new_d2 = ((X - centroids[c]) ** 2).sum(-1)
        closest_d2 = np.minimum(closest_d2, new_d2)

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
    return labels, centroids


def compute(in_path: Path, out_path: Path, k: int) -> None:
    data = json.loads(in_path.read_text())
    books = data["books"]

    pts = []
    owners = []
    for bi, book in enumerate(books):
        for pi, p in enumerate(book["passages"]):
            pts.append((p["x3d"], p["y3d"], p["z3d"]))
            owners.append((bi, pi))
    X = np.asarray(pts, dtype=np.float32)

    labels, centroids = kmeans(X, k)

    # Per-passage code
    flat_idx = 0
    for bi, book in enumerate(books):
        hist = np.zeros(k, dtype=np.int32)
        for pi, p in enumerate(book["passages"]):
            code = int(labels[flat_idx])
            p["code"] = code
            hist[code] += 1
            flat_idx += 1
        book["code_histogram"] = hist.tolist()

    data.setdefault("metadata", {})
    data["metadata"]["vq_k"] = int(k)
    data["metadata"]["vq_centroids"] = [
        [round(float(x), 4) for x in c] for c in centroids
    ]

    out_path.write_text(json.dumps(data))
    print(
        f"Assigned {len(X)} passages into {k} motifs. "
        f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=Path("public/story_shapes.json"))
    ap.add_argument("--out", type=Path, default=Path("public/story_shapes.json"))
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    compute(args.inp, args.out, args.k)


if __name__ == "__main__":
    main()
