"""
Poisson disk sampling on triangle meshes via sample elimination.
Note that points are sampled according to distance in 3D space, and not the geodesic distance on the surface.
In practice this does not seem to be an issue.

Based on:
    Yuksel, "Sample Elimination for Generating Poisson Disk Sample Sets",
    EUROGRAPHICS 2015.
"""

import igl
import numba as nb
import numpy as np
from scipy.spatial import KDTree
from typing import Optional


@nb.njit(cache=True)
def _eliminate(
    offsets: np.ndarray,  # (n_pool+1,) int64
    nb_idx: np.ndarray,  # (M,)        int64
    nb_w: np.ndarray,  # (M,)        float64
    n_samples: int,
) -> np.ndarray:  # (n_pool,)   bool
    n_pool = len(offsets) - 1

    # Initialise weights from CSR
    weights = np.zeros(n_pool, dtype=np.float64)
    for i in range(n_pool):
        for k in range(offsets[i], offsets[i + 1]):
            weights[i] += nb_w[k]

    # Max-heap over (weight, point_index)
    # hi[h] = point index at heap slot h
    # pos[i] = heap slot currently occupied by point i
    hi = np.arange(n_pool, dtype=np.int64)
    pos = np.arange(n_pool, dtype=np.int64)
    heap_size = n_pool

    def swap(a: int, b: int) -> None:
        hi[a], hi[b] = hi[b], hi[a]
        pos[hi[a]], pos[hi[b]] = a, b

    def sift_up(h: int) -> None:
        while h > 0:
            parent = (h - 1) >> 1
            if weights[hi[h]] > weights[hi[parent]]:
                swap(h, parent)
                h = parent
            else:
                break

    def sift_down(h: int) -> None:
        while True:
            s = h
            left = 2 * h + 1
            right = left + 1
            if left < heap_size and weights[hi[left]] > weights[hi[s]]:
                s = left
            if right < heap_size and weights[hi[right]] > weights[hi[s]]:
                s = right
            if s == h:
                break
            swap(h, s)
            h = s

    # Heapify
    for h in range(n_pool // 2 - 1, -1, -1):
        sift_down(h)

    alive = np.ones(n_pool, dtype=nb.boolean)

    for _ in range(n_pool - n_samples):
        # Pop the most-crowded point
        idx = hi[0]
        heap_size -= 1
        swap(0, heap_size)
        sift_down(0)

        alive[idx] = False

        # Decrease key for each live neighbour
        for k in range(offsets[idx], offsets[idx + 1]):
            j = nb_idx[k]
            if alive[j]:
                weights[j] -= nb_w[k]
                sift_down(pos[j])  # weight decreases: can only move down

    return alive


def sample_points_poisson_disk(
    v: np.ndarray,
    f: np.ndarray,
    n: np.ndarray,
    n_samples: int,
    pool_factor: int = 10,
) -> np.ndarray:
    """Sample `n_samples` near-equidistant points on a mesh.

    Uses the sample elimination strategy: draw a large pool of uniformly
    random points, then greedily remove the most-crowded point (highest
    neighbor weight) until `n_samples` remain.

    Parameters
    ----------
    v: (V, 3) vertex positions.
    f: (F, 3) triangle indices.
    f: (V, 3) vertex normals.
    n_samples: Desired number of output points.
    pool_factor: How many times to oversample before elimination (default 10).

    Returns
    -------
    pts     : (n_samples, 3) array of sampled positions.
    normals : (n_samples, 3) array of interpolated normals.
    r       : Packing radius estimate.
    """

    # Sample initial pool
    n_pool = int(pool_factor * n_samples)

    b, fi, pts = igl.random_points_on_mesh(n_pool, v, f)

    # Radius estimate based on ideal (hexagonal) packing
    area = igl.doublearea(v, f).sum() * 0.5
    r = np.sqrt(area / (2.0 * np.sqrt(3.0) * n_samples))
    print(f"Packing radius estimate: {r}")

    tree = KDTree(pts)
    pairs = tree.query_pairs(2.0 * r, output_type="ndarray")  # (M, 2)

    assert len(pairs) > 0

    # Compute weights
    i_idx, j_idx = pairs[:, 0], pairs[:, 1]
    dists = np.linalg.norm(pts[i_idx] - pts[j_idx], axis=1)
    x = dists / (2.0 * r)
    w = (1.0 - x * x) ** 2

    # Symmetrize
    all_i = np.concatenate([i_idx, j_idx])
    all_j = np.concatenate([j_idx, i_idx])
    all_w = np.concatenate([w, w])

    order = np.argsort(all_i, kind="stable")
    all_i = all_i[order]
    all_j = all_j[order].astype(np.int64)
    all_w = all_w[order]

    # Compute CSR indices
    counts = np.bincount(all_i, minlength=n_pool)
    offsets = np.empty(n_pool + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    # Run elimination
    alive = _eliminate(offsets, all_j, all_w, n_samples)

    pts = pts[alive]

    # Interpolate mesh normals
    nrm = b[:, 0:1] * n[f[fi, 0]] + b[:, 1:2] * n[f[fi, 1]] + b[:, 2:3] * n[f[fi, 2]]
    nrm = nrm[alive]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return pts, nrm, r
