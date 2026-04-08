"""
E10 — Diffusion eigenmaps over the passage cloud.

Builds a k-NN graph over all passages in the latent, computes the
symmetric normalised Laplacian L = I − D^(−½) W D^(−½), and takes the
bottom eigenpairs. The non-trivial eigenvectors are smooth functions
on the corpus that pick up its intrinsic geometry — they are the
spectral analogue of `story_shapes_full.ipynb` Part 2 lifted from
the notebook into the visualisation pipeline.

The first eigenvector is the constant (eigenvalue 0); we drop it.
The next few are the genuinely informative axes. Each one assigns a
real value to every passage; the frontend exposes them as new colour
modes ('eig1', 'eig2', 'eig3'), so a single toggle paints the corpus
with a smooth scalar field reflecting one slice of its intrinsic
structure.

Pure numpy. The dense Laplacian is (n × n) double; for the canonical
corpus n ≈ 4400 → ~155 MB. We keep the weight matrix in float32 and
only promote to float64 for the eigh call.

Usage:
    python experiments/e10_diffusion/compute.py \\
        --in public/story_shapes.json \\
        --out public/story_shapes.json \\
        --k 15 --n-eig 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def diffusion_eig(
    coords: np.ndarray,
    k_neighbors: int,
    n_eig: int,
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (eigenvalues, eigenvectors) for the symmetric normalised
    Laplacian of the k-NN graph on `coords`. Returns the bottom (n_eig+1)
    pairs (the first being the constant mode at eigenvalue 0).

    Adds a tiny uniform background `eps` to the weight matrix to guarantee
    one connected component. Without it, gaps in the (post-UMAP) cloud
    can split the graph into pieces, in which case the bottom of the
    spectrum is dominated by zero eigenvalues — one per component — and
    the non-trivial structure gets pushed off the bottom.
    """
    n = len(coords)
    coords = coords.astype(np.float32)

    # Pairwise squared distances, chunked to bound peak memory.
    d2 = np.zeros((n, n), dtype=np.float32)
    chunk = 256
    for i in range(0, n, chunk):
        diff = coords[i : i + chunk, None, :] - coords[None, :, :]
        d2[i : i + chunk] = (diff * diff).sum(-1)

    # k-NN per row, sorted by distance with self at index 0.
    nn_idx = np.argpartition(d2, kth=k_neighbors, axis=1)[:, : k_neighbors + 1]
    rows = np.arange(n)[:, None]
    nn_d2 = d2[rows, nn_idx]
    sort_order = np.argsort(nn_d2, axis=1)
    nn_idx = np.take_along_axis(nn_idx, sort_order, axis=1)
    nn_d2 = np.take_along_axis(nn_d2, sort_order, axis=1)
    # Drop self (column 0).
    nn_idx = nn_idx[:, 1:]
    nn_d2 = nn_d2[:, 1:]

    # Bandwidth from the median k-NN distance.
    sigma2 = float(np.median(nn_d2)) or 1.0

    # Build a dense weight matrix from the k-NN edges, then symmetrise.
    W = np.zeros((n, n), dtype=np.float32)
    weights = np.exp(-nn_d2 / sigma2).astype(np.float32)
    rows_b = np.broadcast_to(rows, nn_idx.shape).ravel()
    cols_b = nn_idx.ravel()
    W[rows_b, cols_b] = weights.ravel()
    W = np.maximum(W, W.T)

    # Connectivity floor: tiny uniform background to merge any stranded
    # components. eps is small enough to leave the local k-NN structure
    # dominant but large enough to push isolated-island eigenvalues away
    # from zero so the informative eigenvectors land at the bottom.
    if eps > 0:
        W = W + eps
        np.fill_diagonal(W, 0.0)

    deg = W.sum(axis=1)
    deg_inv_sqrt = (1.0 / np.sqrt(deg + 1e-12)).astype(np.float32)
    L = np.eye(n, dtype=np.float32) - (deg_inv_sqrt[:, None] * W * deg_inv_sqrt[None, :])

    # Promote to double for the eig solve; eigh on a 4400² matrix is
    # the time-dominant step (~30 s in pure numpy).
    eigvals, eigvecs = np.linalg.eigh(L.astype(np.float64))
    return eigvals[: n_eig + 1], eigvecs[:, : n_eig + 1]


def compute(in_path: Path, out_path: Path, k: int, n_eig: int) -> None:
    data = json.loads(in_path.read_text())
    books = data["books"]

    coords: list[tuple[float, float, float]] = []
    owners: list[tuple[int, int]] = []
    for bi, book in enumerate(books):
        for pi, p in enumerate(book["passages"]):
            coords.append((p["x3d"], p["y3d"], p["z3d"]))
            owners.append((bi, pi))
    pts = np.asarray(coords, dtype=np.float32)
    n = len(pts)
    if n < n_eig + 2:
        raise SystemExit(f"Need at least {n_eig + 2} passages.")

    print(f"Computing diffusion eigenmaps on {n} passages with k={k}...")
    # Ask for a few extra eigenvalues so we can show the spectral gap.
    eigvals, eigvecs = diffusion_eig(pts, k_neighbors=k, n_eig=max(n_eig, 8))
    print(f"Bottom 9 eigenvalues: {[round(float(v), 6) for v in eigvals[:9]]}")
    eigvals = eigvals[: n_eig + 1]
    eigvecs = eigvecs[:, : n_eig + 1]

    # Drop the trivial constant mode (column 0). Keep the next n_eig.
    nontrivial = eigvecs[:, 1 : n_eig + 1]
    nontrivial_eigvals = eigvals[1 : n_eig + 1]

    # Per-passage eigenvalues stored as a fixed-length list.
    flat_idx = 0
    for bi, book in enumerate(books):
        for pi, p in enumerate(book["passages"]):
            p["eig"] = [round(float(nontrivial[flat_idx, k]), 5) for k in range(n_eig)]
            flat_idx += 1

    data.setdefault("metadata", {})
    data["metadata"]["diffusion"] = {
        "k_neighbors": int(k),
        "n_eig": int(n_eig),
        "eigenvalues": [round(float(v), 6) for v in nontrivial_eigvals],
        "trivial_eigenvalue": round(float(eigvals[0]), 6),
    }

    out_path.write_text(json.dumps(data))
    print(
        f"Stored top-{n_eig} non-trivial eigenvectors per passage. "
        f"Eigenvalues: {[round(float(v), 5) for v in nontrivial_eigvals]}. "
        f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=Path("public/story_shapes.json"))
    ap.add_argument("--out", type=Path, default=Path("public/story_shapes.json"))
    ap.add_argument("--k", type=int, default=30, help="number of nearest neighbours")
    ap.add_argument("--n-eig", type=int, default=4, help="non-trivial eigenvectors to store per passage")
    args = ap.parse_args()
    compute(args.inp, args.out, args.k, args.n_eig)


if __name__ == "__main__":
    main()
