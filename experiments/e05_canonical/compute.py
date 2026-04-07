"""
E05 — Position-aligned canonical axes (CCA across book pairs).

Pairs passages across books by *position fraction* (10% into book A
with 10% into book B, etc.) and runs canonical correlation analysis on
each (A, B) pair. Averages the resulting canonical loadings across all
pairs (with sign alignment), giving a global set of axes that vary
similarly across books — candidates for content-independent narrative
axes per `DIRECTIONS.md` §5.

Pure numpy. CCA is implemented via SVD on the whitened cross-covariance.

Usage:
    python experiments/e05_canonical/compute.py \\
        --in public/story_shapes.json \\
        --out public/story_shapes.json \\
        --samples 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def whiten(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centered, whitened X and the whitening matrix W (Σ^{-1/2})."""
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, len(Xc) - 1)
    cov += 1e-6 * np.eye(cov.shape[0])  # ridge for stability
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-12, None)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return Xc @ W, W


def cca(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (A, B, corrs) where A, B are loading matrices in the original
    feature space and corrs are the canonical correlations."""
    Xw, Wx = whiten(X)
    Yw, Wy = whiten(Y)
    M = Xw.T @ Yw / max(1, len(Xw) - 1)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    A = Wx @ U
    B = Wy @ Vt.T
    return A, B, S


def sample_at_positions(book_passages: list[dict], n: int) -> tuple[np.ndarray, list[int]]:
    """Pick `n` passages by closest match to evenly spaced position fractions."""
    positions = np.array([p["pos"] for p in book_passages], dtype=np.float32)
    targets = np.linspace(0.0, 1.0, n)
    chosen_idx = [int(np.argmin(np.abs(positions - t))) for t in targets]
    coords = np.asarray(
        [
            (book_passages[i]["x3d"], book_passages[i]["y3d"], book_passages[i]["z3d"])
            for i in chosen_idx
        ],
        dtype=np.float32,
    )
    return coords, chosen_idx


def compute(in_path: Path, out_path: Path, samples: int) -> None:
    data = json.loads(in_path.read_text())
    books = data["books"]

    # Per-book aligned samples
    aligned = []
    for book in books:
        if len(book["passages"]) < samples:
            continue
        coords, _ = sample_at_positions(book["passages"], samples)
        aligned.append(coords)
    if len(aligned) < 2:
        raise SystemExit("Need at least two books with enough passages.")

    # Run CCA on every pair, accumulate loadings with sign alignment.
    d = aligned[0].shape[1]
    A_sum = np.zeros((d, d), dtype=np.float32)
    n_pairs = 0
    corrs_acc = []
    ref = None
    for i in range(len(aligned)):
        for j in range(i + 1, len(aligned)):
            A, _, S = cca(aligned[i], aligned[j])
            if ref is None:
                ref = A
            else:
                # Sign-align each axis to the reference (flip column if needed).
                for c in range(d):
                    if float(np.dot(A[:, c], ref[:, c])) < 0:
                        A[:, c] = -A[:, c]
            A_sum += A
            corrs_acc.append(S)
            n_pairs += 1
    A_avg = A_sum / max(1, n_pairs)

    # Project all passages onto the averaged top-3 canonical axes.
    axes = A_avg[:, :3]
    for book in books:
        for p in book["passages"]:
            v = np.asarray([p["x3d"], p["y3d"], p["z3d"]], dtype=np.float32)
            proj = v @ axes
            p["canonical"] = [round(float(proj[k]), 4) for k in range(3)]

    mean_corrs = np.mean(np.stack(corrs_acc), axis=0).tolist()
    data.setdefault("metadata", {})
    data["metadata"]["canonical_axes"] = {
        "samples_per_book": int(samples),
        "n_pairs": int(n_pairs),
        "mean_canonical_corrs": [round(float(c), 4) for c in mean_corrs],
    }

    out_path.write_text(json.dumps(data))
    print(
        f"Fit canonical axes from {n_pairs} book pairs, "
        f"mean canonical correlations = {[round(c, 3) for c in mean_corrs]}. "
        f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=Path("public/story_shapes.json"))
    ap.add_argument("--out", type=Path, default=Path("public/story_shapes.json"))
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()
    compute(args.inp, args.out, args.samples)


if __name__ == "__main__":
    main()
