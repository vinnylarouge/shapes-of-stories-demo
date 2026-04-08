"""
E09 — Representational velocity spectrum (PCA / ICA on deltas).

For every (z_t, z_{t+1}) pair in the corpus, compute the latent
displacement δ_t = z_{t+1} - z_t. Treat the cloud of δ's as the
empirical distribution of *narrative motion*, and PCA it.

The spectrum tells you how many directions narrative is actually
moving in: a single dominant component means motion is
one-dimensional (a single arc); a flat spectrum means motion is
isotropic. The components themselves are inspectable directions in
the latent.

This is the cheapest cousin of the spectral track in DIRECTIONS.md
(companion to E08's transition operator and to HGR / CKA). Where E08
asks "what direction does the operator amplify?", E09 asks "in which
directions does the corpus actually go?". The two coincide for purely
shift-invariant dynamics and diverge interestingly otherwise.

ICA upgrade: replace `np.linalg.svd` below with FastICA from sklearn
to get statistically independent components in place of orthogonal
ones. Independence is sometimes a better prior for "narrative
operations" than orthogonality.

Usage:
    python experiments/e09_velocity_spectrum/compute.py \\
        --in public/story_shapes.json \\
        --out public/story_shapes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compute(in_path: Path, out_path: Path) -> None:
    data = json.loads(in_path.read_text())
    books = data["books"]

    deltas: list[tuple[float, float, float]] = []
    for book in books:
        coords = np.asarray(
            [(p["x3d"], p["y3d"], p["z3d"]) for p in book["passages"]],
            dtype=np.float64,
        )
        if len(coords) < 2:
            continue
        for i in range(len(coords) - 1):
            deltas.append(tuple(coords[i + 1] - coords[i]))

    D = np.asarray(deltas, dtype=np.float64)
    if len(D) == 0:
        raise SystemExit("No transitions found.")

    mean_delta = D.mean(0)
    Dc = D - mean_delta
    # PCA via thin SVD: rows are samples, columns are dimensions.
    _, S, Vt = np.linalg.svd(Dc, full_matrices=False)
    # Variance per component (sample covariance).
    var = (S ** 2) / max(1, len(D) - 1)
    var_ratio = var / var.sum() if var.sum() > 0 else var

    data.setdefault("metadata", {})
    data["metadata"]["velocity_pca"] = {
        "components": [[round(float(x), 4) for x in row] for row in Vt],  # PCs as rows
        "variance": [round(float(v), 6) for v in var],
        "variance_ratio": [round(float(v), 4) for v in var_ratio],
        "mean_delta": [round(float(x), 4) for x in mean_delta],
        "n_deltas": int(len(D)),
    }

    out_path.write_text(json.dumps(data))
    print(
        f"Velocity PCA on {len(D)} deltas. "
        f"Variance ratios = {[round(float(v), 4) for v in var_ratio]}. "
        f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=Path("public/story_shapes.json"))
    ap.add_argument("--out", type=Path, default=Path("public/story_shapes.json"))
    args = ap.parse_args()
    compute(args.inp, args.out)


if __name__ == "__main__":
    main()
