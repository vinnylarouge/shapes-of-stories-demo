"""
E08 — Linear narrative transition operator (SVD / DMD).

Treats each book's passage trajectory as a sequence z_0, z_1, ..., z_n in
the shared latent and fits a single linear operator T : ℝᵈ → ℝᵈ such that

    z_{t+1} ≈ T z_t

across the whole corpus, in a least-squares sense. The SVD of T gives an
interpretable spectrum of "input" and "output" directions:

    T = U Σ Vᵀ
    – V columns: input directions (what the operator reads)
    – U columns: output directions (what it writes)
    – σ values: how much each mode is amplified or suppressed in one step

This is the simplest, cheapest member of the spectral family discussed in
DIRECTIONS.md (companion to HGR maximal correlation, CKA, and the
T*T-on-L² operator). It's also the linear baseline of dynamic mode
decomposition (DMD): exact for purely linear dynamics, a useful first
approximation otherwise.

Note: in this version we operate on the post-UMAP 3D latent (d=3), so T
is a 3×3 matrix with three singular values. The same script runs
unchanged on the higher-dimensional AE latent that E07 will produce —
the spectrum just becomes longer.

Reinterpretation note: the directions document originally framed these
spectra as "between consecutive transformer layers". We currently lack
per-layer activations in the published JSON, so this script applies the
same machinery to the *narrative-time* axis instead — Z_l = z_t,
Z_{l+1} = z_{t+1}. When E07 produces per-layer data, swap the data
loader to walk layers instead of time and the rest of the script is
unchanged.

Usage:
    python experiments/e08_transition_svd/compute.py \\
        --in public/story_shapes.json \\
        --out public/story_shapes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fit_transition_operator(sources: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Returns T such that targets ≈ sources @ T.T (rows are samples)."""
    # Centre both with the source mean (consistent shift for the operator).
    mu = sources.mean(0, keepdims=True)
    Xc = sources - mu
    Yc = targets - mu
    # Least-squares: solve Xc @ M = Yc for M ∈ ℝ^{d×d}, then T = M.T.
    M, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)
    return M.T


def compute(in_path: Path, out_path: Path) -> None:
    data = json.loads(in_path.read_text())
    books = data["books"]

    sources: list[tuple[float, float, float]] = []
    targets: list[tuple[float, float, float]] = []
    for book in books:
        coords = np.asarray(
            [(p["x3d"], p["y3d"], p["z3d"]) for p in book["passages"]],
            dtype=np.float64,
        )
        if len(coords) < 2:
            continue
        for i in range(len(coords) - 1):
            sources.append(tuple(coords[i]))
            targets.append(tuple(coords[i + 1]))

    X = np.asarray(sources, dtype=np.float64)
    Y = np.asarray(targets, dtype=np.float64)
    if len(X) == 0:
        raise SystemExit("No transitions found.")

    T = fit_transition_operator(X, Y)
    U, sigma, Vt = np.linalg.svd(T)
    V = Vt.T

    # Quality of fit: residual norm vs target norm. Compute in centered
    # coordinates to keep the formula simple. Use np.dot rather than `@`
    # because macOS Accelerate's BLAS raises a spurious FPE warning on
    # `@` for some matmul shapes; np.dot goes through a different path
    # and gives the same result without the warning.
    mu = X.mean(0)
    Xc = X - mu
    Yc = Y - mu
    pred_c = np.dot(Xc, T.T)
    residual_norm2 = float(((Yc - pred_c) ** 2).sum())
    target_norm2 = float((Yc ** 2).sum())
    r2 = 1.0 - residual_norm2 / max(target_norm2, 1e-12)

    data.setdefault("metadata", {})
    data["metadata"]["transition_svd"] = {
        "T": [[round(float(x), 4) for x in row] for row in T],
        "U": [[round(float(x), 4) for x in row] for row in U],
        "sigma": [round(float(x), 4) for x in sigma],
        "V": [[round(float(x), 4) for x in row] for row in V],
        "n_pairs": int(len(X)),
        "r2": round(float(r2), 4),
    }

    out_path.write_text(json.dumps(data))
    print(
        f"Fitted linear T on {len(X)} (z_t, z_t+1) pairs. "
        f"σ = {[round(float(s), 4) for s in sigma]}, "
        f"R² = {r2:.4f}. Wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=Path("public/story_shapes.json"))
    ap.add_argument("--out", type=Path, default=Path("public/story_shapes.json"))
    args = ap.parse_args()
    compute(args.inp, args.out)


if __name__ == "__main__":
    main()
