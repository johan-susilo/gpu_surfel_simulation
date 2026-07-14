# pyright: reportInvalidTypeForm=false
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import scipy  # type: ignore[import]
from scipy.spatial.transform import Rotation  # type: ignore[import]
import time

from core.surfels import Surfels
from core.logger import Logger, EventType
from core.signal import Signal
from core.parameters import Parameters
from core.model import Model
from core.types import Array, ArrayIndex
from core.warp_kernels import compute_density_kernel, compute_forces_kernel, compute_centroids_kernel, normalize_centroids_kernel, compute_volumes_kernel, apply_volume_pressure_kernel, integrate_positions_kernel, downcast_positions_kernel
import warp as wp

from core.profiler import SimulationProfiler


wp.init()
if wp.is_cuda_available():
    DEVICE = wp.get_preferred_device()
else:
    DEVICE = wp.get_device("cpu")

# Default adhesion value
ADHESION = 0.2


class Neighbors:
    def __init__(self, _n: int):
        self.dist: Array
        self.i1: ArrayIndex
        self.i2: ArrayIndex
        self.count: np.ndarray[Tuple[int], np.dtype[np.int64]]

    def update(self, surfels: Surfels, max_dist: float) -> None:
        tree = scipy.spatial.kdtree.KDTree(surfels.x)
        pairs = tree.query_pairs(max_dist, output_type="ndarray")

        self.i1 = pairs[:, 0]
        self.i2 = pairs[:, 1]
        self.dist = np.sqrt(np.sum((surfels.x[self.i1] - surfels.x[self.i2]) ** 2, axis=1))

        n_surfels = surfels.x.shape[0]
        self.count = np.zeros((n_surfels), dtype=np.int64)

        # only count neighbors of the same color — clip -1 (fixed) to 0, same as CPU
        colors = np.maximum(0, surfels.cell_index)
        color_mask = colors[self.i1] == colors[self.i2]
        self.count += np.bincount(self.i1, weights=color_mask, minlength=n_surfels).astype(np.int64)
        self.count += np.bincount(self.i2, weights=color_mask, minlength=n_surfels).astype(np.int64)


def project(x: Array, n: Array) -> Array:
    return np.einsum("ij, ij->i", x, n)[:, np.newaxis] * n


def compute_curvature(surfels, neighbors, h, color_mask):
    points = surfels.x
    N = points.shape[0]

    i = neighbors.i1
    j = neighbors.i2

    delta = points[j] - points[i]
    dist = np.sqrt(np.sum(delta**2, axis=1))
    weight = poly6(dist, h)

    weighted_diffs = delta * weight[:, None]
    weighted_diffs *= color_mask[:, None] * 0.9 + 0.1

    numerator = np.zeros((N, 3))
    np.add.at(numerator, i, weighted_diffs)
    np.add.at(numerator, j, -weighted_diffs)

    denominator = np.zeros(N)
    np.add.at(denominator, i, weight)
    np.add.at(denominator, j, weight)
    denominator += 1.0

    valid_mask = denominator > 1e-10
    v_weighted = np.zeros((N, 3))
    v_weighted[valid_mask] = numerator[valid_mask] / denominator[valid_mask, None]

    result = (20.0 / 3.0) * v_weighted / (h**2)
    return result


class Fluid(Model):
    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.timings = {
            "neighbors": 0,
            "phase": 0,
            "warp_setup": 0,
            "warp_density": 0,
            "gpu_transfer": 0,
            "volume_pressure": 0,
            "curvature": 0,
            "pairwise_pressure": 0,
            "signal_force": 0,
            "position_update": 0,
        }

        self.grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128, device=DEVICE)

    def integrate(self, surfels: Surfels, n: int, signal: Optional[Signal], logger: Logger):
        dt = self.parameters.dt
        d0 = surfels.d0
        h = d0

        n_surfels = surfels.x.shape[0]
        total_steps = 0

        if n_surfels == 0:
            print("No surfels!")
            return

        wp_cell_index = wp.array(surfels.cell_index.astype(np.int32), dtype=int, device=DEVICE)
        _last_cell_index = surfels.cell_index.copy()

        # Pre-allocate all VRAM outside the loop.
        wp_positions = wp.zeros(n_surfels, dtype=wp.vec3d, device=DEVICE)
        wp_positions_32 = wp.zeros(n_surfels, dtype=wp.vec3, device=DEVICE)
        wp_densities = wp.zeros(n_surfels, dtype=float, device=DEVICE)
        wp_neighbor_counts = wp.zeros(n_surfels, dtype=int, device=DEVICE)
        wp_phase_counts = wp.zeros(n_surfels, dtype=int, device=DEVICE)
        wp_dens_scaled = wp.zeros(n_surfels, dtype=float, device=DEVICE)
        wp_normals = wp.zeros(n_surfels, dtype=wp.vec3, device=DEVICE)
        wp_out_forces = wp.zeros(n_surfels, dtype=wp.vec3, device=DEVICE)
        wp_out_curvature = wp.zeros(n_surfels, dtype=wp.vec3, device=DEVICE)
        wp_cpu_forces = wp.zeros(n_surfels, dtype=wp.vec3, device=DEVICE)
        
        # Pre-allocate cell structures so they exist on step 0
        max_cells = max(1, int(np.max(surfels.cell_index)) + 1)
        wp_cell_counts = wp.zeros(max_cells, dtype=int, device=DEVICE)
        
        # Use float64 for global accumulators to prevent atomic truncation
        wp_cell_centroids = wp.zeros(max_cells, dtype=wp.vec3d, device=DEVICE)
        wp_cell_volumes = wp.zeros(max_cells, dtype=wp.float64, device=DEVICE)
        
        # initialize the profiler with 5 warmup steps to hide LLVM compilation
        profiler = SimulationProfiler(device=DEVICE, warmup_steps=5)

        # fast copy CPU -> GPU (Creates a zero-copy CPU view, then streams to VRAM)
        wp.copy(wp_positions, wp.array(surfels.x, dtype=wp.vec3d, device="cpu"))
        wp.copy(wp_normals, wp.array(surfels.n, dtype=wp.vec3, device="cpu"))

        for step in range(n + profiler.warmup_steps):
            profiler.start_step()
            
            # log step overhead before the compute timer starts
            logger.log_event(EventType.TIMESTEP)
            total_steps += 1

            if not np.array_equal(surfels.cell_index, _last_cell_index):
                wp_cell_index = wp.array(surfels.cell_index.astype(np.int32), dtype=int, device=DEVICE)
                _last_cell_index = surfels.cell_index.copy()
                max_cells = max(1, int(np.max(surfels.cell_index)) + 1)
                wp_cell_centroids = wp.zeros(max_cells, dtype=wp.vec3d, device=DEVICE)
                wp_cell_counts = wp.zeros(max_cells, dtype=int, device=DEVICE)
                wp_cell_volumes = wp.zeros(max_cells, dtype=wp.float64, device=DEVICE)

            profiler.start_compute()
            
            t0 = time.perf_counter()
            wp_densities.zero_()
            wp_neighbor_counts.zero_()
            wp_phase_counts.zero_()
            wp_out_forces.zero_()
            wp_out_curvature.zero_()
            
            # --- Convert 64-bit state to 32-bit for grid queries (GPU) ---
            wp.launch(downcast_positions_kernel, dim=n_surfels, inputs=[wp_positions, wp_positions_32], device=DEVICE)
            
            self.grid.build(wp_positions_32, h)
            self.timings["warp_setup"] += time.perf_counter() - t0

            # --- Density (GPU) ---
            t0 = time.perf_counter()
            wp.launch(
                compute_density_kernel, 
                dim=n_surfels, 
                inputs=[wp_positions, wp_positions_32, self.grid.id, h, wp_densities, wp_neighbor_counts, wp_phase_counts, wp_cell_index], 
                device=DEVICE
            )
            wp.synchronize_device(DEVICE)
            self.timings["warp_density"] += time.perf_counter() - t0

            # --- Volume pressure (GPU) ---
            t0 = time.perf_counter()
            # rest_volume = self.parameters.rest_radius
            rest_volume = (4.0 / 3.0) * np.pi * (self.parameters.rest_radius) ** 3
            wp_cell_centroids.zero_()
            wp_cell_counts.zero_()
            wp_cell_volumes.zero_()
            
            wp.launch(compute_centroids_kernel, dim=n_surfels, inputs=[wp_positions, wp_cell_index, wp_cell_centroids, wp_cell_counts], device=DEVICE)
            wp.launch(normalize_centroids_kernel, dim=max_cells, inputs=[wp_cell_centroids, wp_cell_counts], device=DEVICE)
            wp.launch(compute_volumes_kernel, dim=n_surfels, inputs=[wp_positions, wp_normals, wp_densities, wp_cell_index, wp_cell_centroids, wp_cell_volumes], device=DEVICE)
            wp.launch(apply_volume_pressure_kernel, dim=n_surfels, inputs=[wp_normals, wp_cell_index, wp_cell_volumes, rest_volume, self.parameters.k_pressure, wp_out_forces], device=DEVICE)
            wp.synchronize_device(DEVICE)
            self.timings["volume_pressure"] += time.perf_counter() - t0
            
            # --- Force (GPU) ---
            t0 = time.perf_counter()  
            wp.launch(
                compute_forces_kernel, 
                dim=n_surfels, 
                inputs=[
                    wp_positions,
                    wp_positions_32,
                    wp_densities, 
                    self.grid.id, 
                    h, 
                    self.parameters.k_plane, 
                    self.parameters.k_dist, 
                    wp_cell_index, 
                    wp_normals, 
                    wp_out_forces, 
                    wp_out_curvature,
                    self.parameters.rho_target 
                ], 
                device=DEVICE
            )
            wp.synchronize_device(DEVICE)
            self.timings["curvature"] += time.perf_counter() - t0

            # --- Signal force (CPU) ---
            t0 = time.perf_counter()
            force = np.zeros_like(surfels.x)
            if signal is not None:
                f_signal = self.calc_f_signal(surfels, signal)
                force += self.parameters.k_signal * f_signal
            self.timings["signal_force"] += time.perf_counter() - t0

            # --- Position update (GPU) ---
            t0 = time.perf_counter()
            wp_cpu_forces = wp.array(force, dtype=wp.vec3, device=DEVICE)
            current_seed = int(time.perf_counter() * 1000000) % (2**31 - 1)
            
            wp.launch(integrate_positions_kernel, dim=n_surfels, inputs=[wp_positions, wp_out_forces, wp_cpu_forces, wp_cell_index, dt, current_seed], device=DEVICE)
            wp.synchronize_device(DEVICE)
            self.timings["position_update"] += time.perf_counter() - t0

            profiler.end_compute()

            # PCIE TRANSFERS 
            # fetch arrays required for JSON stats (MUST happen every step to avoid stale metrics)
            density = wp_densities.numpy()
            neighbor_counts = wp_neighbor_counts.numpy()
            phase_counts = wp_phase_counts.numpy()
            cpu_volumes = wp_cell_volumes.numpy()
            curvature_gpu = wp_out_curvature.numpy()
            forces_gpu = wp_out_forces.numpy()
            
            surfels.phase = (phase_counts > 8).astype(int)

            avg_n_neighbors = float(np.mean(neighbor_counts))
            if avg_n_neighbors > 40:
                print(f"Collapse detected: avg neighbors = {avg_n_neighbors:.1f}")
                logger.log_event(EventType.UPDATE_SURFELS, surf=surfels, force=None)
                break

            # heavy Rerun visualizer updates (File I/O - only on snapshots)
            is_logging_step = (logger.snapshot_every is not None) and ((logger.sim_ctr - 1) % logger.snapshot_every == 0)
            
            if is_logging_step or step == 0 or step == (n + profiler.warmup_steps - 1):
                surfels.x = wp_positions.numpy()
                curv_mag = np.linalg.norm(curvature_gpu, axis=1) 
                surfels.debug_force = curvature_gpu
                surfels.log_color = curv_mag**2 
                surfels.neighbor_count = neighbor_counts
                logger.log_event(EventType.UPDATE_SURFELS, surf=surfels, force=None)

            # compile metrics dictionary
            volumes = {}
            for c in np.unique(surfels.cell_index):
                if c >= 0 and c < len(cpu_volumes):
                    volumes[int(c)] = float(cpu_volumes[int(c)])
                else:
                    volumes[int(c)] = None

            total_forces_np = forces_gpu + force 
            force_mags = np.linalg.norm(total_forces_np, axis=1)
            curv_mag = np.linalg.norm(curvature_gpu, axis=1)

            metrics = {
                "sim_ctr": logger.sim_ctr,
                "time": logger.t,
                "density_stats": {
                    "min": float(np.min(density)),
                    "max": float(np.max(density)),
                    "mean": float(np.mean(density)),
                },
                "neighbor_stats": {
                    "min": int(np.min(neighbor_counts)),
                    "max": int(np.max(neighbor_counts)),
                    "mean": float(np.mean(neighbor_counts)),
                },
                "force_stats": {
                    "min": float(np.min(force_mags)),
                    "max": float(np.max(force_mags)),
                    "mean": float(np.mean(force_mags)),
                    "std": float(np.std(force_mags)),
                },
                "curvature_stats": {
                    "min": float(np.min(curv_mag)),
                    "max": float(np.max(curv_mag)),
                    "mean": float(np.mean(curv_mag)),
                    "std": float(np.std(curv_mag)),
                },
                "volumes": volumes,
            }

            profiler.end_step(metrics, logger)

    def calc_f_tilt(
        self, surfels: Surfels, neighbors: Neighbors, color_mask: Array, w: Array
    ) -> Array:
        f_tilt_pairs = -np.cross(surfels.n[neighbors.i2], surfels.n[neighbors.i1]) * w[:, None]
        f_tilt_pairs *= color_mask

        f_tilt = np.zeros_like(surfels.x)
        np.add.at(f_tilt, neighbors.i1, f_tilt_pairs)
        np.add.at(f_tilt, neighbors.i2, -f_tilt_pairs)

        return self.parameters.k_tilt * f_tilt

    def calc_f_signal(self, surfels: Surfels, signal: Signal) -> Array:
        f_signal = signal.interpolator(surfels.x)
        f_signal = project(f_signal, surfels.n)
        return self.parameters.k_signal * f_signal

    def repr(self, repr_level):
        if repr_level > 0:
            print("Fluid surfel model")

    def optimize_surfel_density(self, surfels: Surfels, logger: Logger):
        n_surfels = surfels.x.shape[0]
        h = surfels.d0 * 3

        neighbors = Neighbors(n_surfels)
        neighbors.update(surfels, h)
        self.estimate_normals(surfels)

        # FIXME: skip remodeling for now
        return

        fixed_ids = surfels.cell_index == -1

        to_add = np.logical_and(neighbors.count > 6, neighbors.count <= 20)
        to_keep = neighbors.count <= 32

        to_add[fixed_ids] = 0
        to_keep[fixed_ids] = 1

        max_delete_fraction = 0.05
        max_add_fraction = 0.1

        if 1.0 - np.average(to_keep) > max_delete_fraction:
            keep_random = max_delete_fraction / (1.0 - np.average(to_keep))
            to_keep = np.logical_or(to_keep, np.random.random(size=to_keep.shape) > keep_random)

        if np.average(to_add) > max_add_fraction:
            add_random = max_add_fraction / np.average(to_add)
            to_add = np.logical_and(to_add, np.random.random(size=to_add.shape) < add_random)

        new_positions = surfels.x[to_add]
        new_normals = surfels.n[to_add]
        new_colors = surfels.cell_index[to_add]
        new_phase = surfels.phase[to_add]

        offsets = np.random.normal(size=new_positions.shape)
        offsets -= project(offsets, new_normals)
        offsets /= np.linalg.norm(offsets, axis=1)[:, np.newaxis]
        offsets *= 0.5 * surfels.d0

        new_positions += offsets
        surfels.x[to_add] -= offsets

        surfels.x = surfels.x[to_keep]
        surfels.n = surfels.n[to_keep]
        surfels.cell_index = surfels.cell_index[to_keep]
        surfels.phase = surfels.phase[to_keep]

        n_removed = n_surfels - surfels.x.shape[0]

        surfels.x = np.append(surfels.x, new_positions, axis=0)
        surfels.n = np.append(surfels.n, new_normals, axis=0)
        surfels.cell_index = np.append(surfels.cell_index, new_colors, axis=0)
        surfels.phase = np.append(surfels.phase, new_phase, axis=0)

        print(f">>>Remodeling : added: {new_positions.shape[0]} removed: {n_removed} total: {surfels.x.shape[0]} ")
        logger.remodel_ctr += 1

    def estimate_normals(self, surfels: Surfels):
        n_surfels = surfels.x.shape[0]
        d0 = surfels.d0

        neighbors = Neighbors(n_surfels)
        neighbors.update(surfels, 3 * d0)

        i1 = neighbors.i1
        i2 = neighbors.i2

        color_mask = surfels.cell_index[i1] == surfels.cell_index[i2]
        r_ij = (surfels.x[i2] - surfels.x[i1]) / d0
        outer_prod = np.einsum("ki, kj -> kij", r_ij, r_ij)
        outer_prod *= color_mask[:, np.newaxis, np.newaxis]

        cov = np.zeros((n_surfels, 3, 3))
        np.add.at(cov, i1, outer_prod)
        np.add.at(cov, i2, outer_prod)

        new_normals = np.linalg.eigh(cov)[1][:, :, 0]
        new_normals *= np.sign(np.einsum("ij, ij->i", new_normals, surfels.n))[:, np.newaxis]

        surfels.n = new_normals