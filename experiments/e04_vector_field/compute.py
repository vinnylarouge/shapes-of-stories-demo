"""
E04 — Local-kernel vector field + per-passage surprise.

Treats the corpus passage transitions as samples of a drift field on
the latent space. Fits a Nadaraya-Watson kernel-weighted average:

    v(q) = Σᵢ wᵢ(q) Δzᵢ / Σᵢ wᵢ(q)
    wᵢ(q) = exp(-‖q - zᵢ‖² / h²)

where (zᵢ, Δzᵢ) are observed (passage, next-displacement) pairs across
all books.

Sampling — adaptive octree (Barnes-Hut-flavoured).

    Rather than evaluating v on a uniform grid, we recursively subdivide
    cells in the spirit of Barnes-Hut tree decomposition, but applied to
    the *sampling* of the field rather than the n-body sum. Starting
    from a coarse init_n³ grid, each cell is split into 8 children only
    when v varies significantly across them; smooth/uniform regions
    stay coarse. The result is an irregular point cloud whose density
    follows where the field has structure.

    The kernel sum at each query is computed directly: M ≈ 4k
    transitions, far below the threshold where the Barnes-Hut far-field
    approximation would matter.

Surprise: per-passage, ‖observed Δ - predicted Δ‖ at every interior
passage. Spikes are candidates for chapter breaks and climactic shifts.

Pure numpy.

Usage:
    python experiments/e04_vector_field/compute.py \\
        --in public/story_shapes.json \\
        --out public/story_shapes.json \\
        --init-n 4 --max-depth 3 --threshold 0.35
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def kernel_field(query: np.ndarray, sources: np.ndarray, deltas: np.ndarray, bandwidth: float) -> np.ndarray:
    """Predict v(q) for each row of `query` from (sources, deltas)."""
    diff = query[:, None, :] - sources[None, :, :]
    d2 = (diff * diff).sum(-1)
    w = np.exp(-d2 / (bandwidth * bandwidth))
    wsum = w.sum(-1, keepdims=True)
    wsum = np.where(wsum < 1e-12, 1.0, wsum)
    return (w[:, :, None] * deltas[None, :, :]).sum(1) / wsum


def kernel_field_chunked(
    query: np.ndarray, sources: np.ndarray, deltas: np.ndarray, bandwidth: float, chunk: int = 1024
) -> np.ndarray:
    """Same as kernel_field, but bounds peak memory by chunking the queries."""
    n = len(query)
    out = np.empty((n, 3), dtype=np.float32)
    for i in range(0, n, chunk):
        out[i : i + chunk] = kernel_field(query[i : i + chunk], sources, deltas, bandwidth)
    return out


# 8-octant offsets used for child cell centers and for variation testing.
_CHILD_OFFSETS = np.array(
    [
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
    ],
    dtype=np.float32,
)


def adaptive_octree(
    bbox_lo: np.ndarray,
    bbox_hi: np.ndarray,
    init_n: int,
    max_depth: int,
    threshold: float,
    sources: np.ndarray,
    deltas: np.ndarray,
    bandwidth: float,
    max_leaves: int,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Returns a list of leaf cells `(center, v_center, half_size)`.

    A cell is refined into 8 children iff its magnitude is above an
    absolute floor *and* the relative spread of v across the 8 child
    samples is above `threshold`. The whole sweep is vectorised across
    all cells at each level.
    """
    init_step = (bbox_hi - bbox_lo) / init_n
    half_init = float(init_step.mean()) / 2

    # Level-0 cells: centres of an init_n³ regular partition of the bbox.
    ix, iy, iz = np.meshgrid(
        np.arange(init_n), np.arange(init_n), np.arange(init_n), indexing="ij"
    )
    grid_idx = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=-1).astype(np.float32)
    centers = bbox_lo + (grid_idx + 0.5) * init_step
    centers = centers.astype(np.float32)
    half_sizes = np.full(len(centers), half_init, dtype=np.float32)

    # Evaluate v at level-0 centres; the maximum magnitude here sets the
    # absolute floor used to skip dead-zone refinement.
    cur_vs = kernel_field_chunked(centers, sources, deltas, bandwidth)
    init_norms = np.linalg.norm(cur_vs, axis=1)
    global_max = float(init_norms.max()) if init_norms.size else 1.0
    abs_mag_floor = 0.08 * global_max

    leaves: list[tuple[np.ndarray, np.ndarray, float]] = []

    for level in range(max_depth + 1):
        n = len(centers)
        if n == 0:
            break

        # Stop conditions: max depth reached, or budget exhausted.
        if level == max_depth or len(leaves) + n > max_leaves:
            for i in range(n):
                leaves.append((centers[i].copy(), cur_vs[i].copy(), float(half_sizes[i])))
            break

        # Sample v at the 8 sub-octant centres of every cell.
        quarters = (half_sizes / 2).reshape(-1, 1, 1)
        child_centers = centers[:, None, :] + _CHILD_OFFSETS[None, :, :] * quarters
        child_centers_flat = child_centers.reshape(-1, 3)
        child_vs_flat = kernel_field_chunked(child_centers_flat, sources, deltas, bandwidth)
        child_vs = child_vs_flat.reshape(n, 8, 3)

        mean_v = child_vs.mean(1)
        max_dev = np.linalg.norm(child_vs - mean_v[:, None, :], axis=2).max(1)
        magnitudes = np.linalg.norm(mean_v, axis=1)
        denom = np.maximum(magnitudes, abs_mag_floor)
        rel_dev = max_dev / denom
        # Refine when the cell carries non-trivial flow AND the local
        # variation is large enough to justify subdivision.
        refine = (magnitudes >= abs_mag_floor) & (rel_dev >= threshold)

        stop_idx = np.where(~refine)[0]
        for i in stop_idx:
            leaves.append((centers[i].copy(), cur_vs[i].copy(), float(half_sizes[i])))

        refine_idx = np.where(refine)[0]
        if refine_idx.size == 0:
            break

        # The children of a refined cell *are* the next level's cells —
        # reuse their v values from the variation test (no extra work).
        new_centers = child_centers[refine_idx].reshape(-1, 3).astype(np.float32)
        new_half_sizes = np.repeat(half_sizes[refine_idx] / 2, 8).astype(np.float32)
        new_vs = child_vs[refine_idx].reshape(-1, 3).astype(np.float32)

        centers = new_centers
        half_sizes = new_half_sizes
        cur_vs = new_vs

    return leaves


def compute(
    in_path: Path,
    out_path: Path,
    init_n: int,
    max_depth: int,
    threshold: float,
    max_leaves: int,
) -> None:
    data = json.loads(in_path.read_text())
    books = data["books"]

    # Build (point, delta) pairs from each book's trajectory.
    sources: list[tuple[float, float, float]] = []
    deltas: list[tuple[float, float, float]] = []
    for book in books:
        coords = np.asarray(
            [(p["x3d"], p["y3d"], p["z3d"]) for p in book["passages"]],
            dtype=np.float32,
        )
        if len(coords) < 2:
            continue
        for i in range(len(coords) - 1):
            sources.append(tuple(coords[i]))
            deltas.append(tuple(coords[i + 1] - coords[i]))

    src = np.asarray(sources, dtype=np.float32)
    dlt = np.asarray(deltas, dtype=np.float32)
    if len(src) == 0:
        raise SystemExit("No transitions found.")

    # Bandwidth: median of a small subsample of pairwise displacements.
    pair_norms = np.linalg.norm(src[None, :100] - src[:100, None], axis=-1)
    median_d = float(np.median(pair_norms[pair_norms > 0]))
    bandwidth = median_d * 0.75

    # Per-passage surprise residual.
    pred_for_sources = kernel_field_chunked(src, src, dlt, bandwidth)
    residual = np.linalg.norm(dlt - pred_for_sources, axis=-1)
    src_lookup = 0
    for book in books:
        n = len(book["passages"])
        for i in range(n):
            if i < n - 1:
                book["passages"][i]["surprise"] = round(float(residual[src_lookup]), 4)
                src_lookup += 1
            else:
                prev = book["passages"][i - 1].get("surprise") if i > 0 else 0.0
                book["passages"][i]["surprise"] = float(prev or 0.0)

    # Adaptive octree sampling of the field over the data bounding box.
    pts_all = np.concatenate([src, src + dlt])
    lo = pts_all.min(0)
    hi = pts_all.max(0)
    pad = 0.05 * (hi - lo)
    bbox_lo = (lo - pad).astype(np.float32)
    bbox_hi = (hi + pad).astype(np.float32)

    leaves = adaptive_octree(
        bbox_lo, bbox_hi,
        init_n=init_n, max_depth=max_depth, threshold=threshold,
        sources=src, deltas=dlt, bandwidth=bandwidth,
        max_leaves=max_leaves,
    )

    grid_export = []
    for c, v, h in leaves:
        grid_export.append(
            {
                "x": round(float(c[0]), 3),
                "y": round(float(c[1]), 3),
                "z": round(float(c[2]), 3),
                "vx": round(float(v[0]), 4),
                "vy": round(float(v[1]), 4),
                "vz": round(float(v[2]), 4),
                "h": round(float(h), 4),
            }
        )

    data.setdefault("metadata", {})
    data["metadata"]["field"] = {
        "grid": grid_export,
        "bandwidth": round(float(bandwidth), 4),
        "adaptive": True,
        "init_n": int(init_n),
        "max_depth": int(max_depth),
        "threshold": float(threshold),
        "n_leaves": int(len(leaves)),
        # grid_n is no longer meaningful for an adaptive grid; keep the
        # field for backward-compat readers but report the initial size.
        "grid_n": int(init_n),
    }

    out_path.write_text(json.dumps(data))
    print(
        f"Adaptive field on {len(src)} transitions: h={bandwidth:.3f}, "
        f"init_n={init_n}, max_depth={max_depth}, threshold={threshold}, "
        f"{len(leaves)} leaf cells. "
        f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=Path("public/story_shapes.json"))
    ap.add_argument("--out", type=Path, default=Path("public/story_shapes.json"))
    ap.add_argument("--init-n", type=int, default=4, help="initial coarse grid resolution")
    ap.add_argument("--max-depth", type=int, default=3, help="maximum recursive subdivisions per cell")
    ap.add_argument("--threshold", type=float, default=0.35, help="relative variation threshold for refining a cell")
    ap.add_argument("--max-leaves", type=int, default=8000, help="hard cap on the number of output cells")
    args = ap.parse_args()
    compute(args.inp, args.out, args.init_n, args.max_depth, args.threshold, args.max_leaves)


if __name__ == "__main__":
    main()
