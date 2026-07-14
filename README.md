# Surfel_fluid_simulation

This code is a GPU-accelerated surfel-based fluid / membrane simulator used for research-quality experiments comparing a Warp/GPU implementation with a reference CPU implementation. The repository contains the simulation core, utilities for logging and visualization.

## Quick overview

- Core simulation code: `core/` (model implementations, surfel handling, logging, kernels)
- Example entrypoint: `main.py` (calls `Fluid.integrate()`)
- Parameters: `params/param_*.yml` — pick a testcase name like `d0`
- Outputs: `output/<testcase>/`
  - `metrics.jsonl` — one JSON line per timestep with per-step metrics
  - `run_summary.json` — run-level summary (total_time_ms, total_steps)
  - `surfels_x.npy`, `surfels_n.npy`, `surfels_i.npy` — final surfel state
  - optional `*.vtp` / `*.pkl` snapshots depending on settings

## Requirements

Recommended: Python 3.11+ in a virtual environment. Key Python dependencies (see `requirements.txt`) include:

- numpy
- scipy
- matplotlib
- tifffile (optional output)
- warp (Warp SDK) — for GPU acceleration (optional if running CPU implementation)
- rerun (optional, for realtime visualization)

Install into a virtualenv and then:

```bash
python -m venv .venv
source .venv/bin/activate    # or on Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Note: Warp (NVIDIA Warp) may require a CUDA-capable GPU and proper installation; if Warp is not available the GPU code paths will not run.

## Running a simulation

Choose a testcase name (matches files in `params/`, e.g. `d0`), then run:

```bash
python main.py d0
```

This will create `output/d0/` (or the output folder defined in the param file) and write:

- `metrics.jsonl` — per-timestep metrics (JSON-lines). Each entry contains keys such as `sim_ctr`, `time`, `density_stats`, `force_stats`, `curvature_stats`, `volumes`, and timing keys `step_time_ms` and `total_time_ms`.
- `run_summary.json` — top-level run summary with `total_time_ms` and `total_steps`.

## Metrics schema (per-timestep)

Each line in `metrics.jsonl` is a JSON object containing at least the following fields (keys may vary slightly between GPU/CPU, aliases are supported):

- `sim_ctr`, `time` — step counters and simulation time
- `density_stats` — dict with `min`, `max`, `mean` (and sometimes `std`)
- `force_stats` — dict with `min`, `max`, `mean`, `std`
- `curvature_stats` — dict with `min`, `max`, `mean`, `std`
- `volumes` — dict mapping cell id -> estimated volume (may contain nulls)
- `hashgrid_stats` / `kdt_stats` / `neighbor_stats` — neighbor count statistics
- `step_time_ms` — wall-clock time taken for the step (ms)
- `total_time_ms` — accumulated wall-clock time since run start (ms)

The logging code intentionally writes the metrics as simple JSON-lines so downstream tools can stream and compare them easily.

## Realtime visualization

If `rerun` is installed and enabled in the `Logger` (default behavior when `rerun_viewer=True`), the simulator will mirror selected scalar metrics (average density, avg force magnitude, step and total ms) and surfel geometry to the Rerun timeline. See `core/logger.py` for the exact keys used in the rerun timeline.

## Developer notes

- The GPU implementation lives in `core/model_fluid.py` and uses Warp kernels in `core/warp_kernels.py`.
- The CPU implementation in the other repo mirrors the same metrics schema for straightforward comparison.
- Per-step timing is collected per-phase on the GPU side and exposed via `step_time_ms` / `total_time_ms` in the metrics.
- A `run_summary.json` is written by `main.py` after the run completes (contains `total_time_ms` and `total_steps`) and is used by the plotting script for a simple run-time comparison.
