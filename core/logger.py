from __future__ import annotations

import pickle
import json
import re
import logging
from .signal import Signal

import rerun
from matplotlib import colormaps
import numpy as np
from enum import Enum
from typing import Any, Dict, Optional, TYPE_CHECKING
from pathlib import Path

from core.vtk_utils import simlog_to_vtp

if TYPE_CHECKING:
    from core.model import Model
    from core.surfels import Surfels
    from core.forces import Forces


# Scaling factor for drawing normals (only affects Rerun output)
# NORMAL_SCALE = 0.15
NORMAL_SCALE = 1.0


# Required top-level keys in every metrics dict
_REQUIRED_METRIC_KEYS = frozenset(
    [
        "sim_ctr",
        "time",
        "device",          
        "density_stats",
        "neighbor_stats",  
        "force_stats",
        "curvature_stats",
        "volumes",
        "compute_ms",      
        "step_time_ms",    
        "total_time_ms",   
    ]
)

# Sub-keys required inside each *_stats dict
_REQUIRED_STAT_KEYS = frozenset(["min", "max", "mean"])


cmap_index = colormaps.get_cmap("tab10")
# cmap_numeric = colormaps.get_cmap("viridis")
cmap_numeric = colormaps.get_cmap("summer")

def index_to_color(index: np.ndarray) -> np.ndarray:
    color = cmap_index((index % 10) / 10)

    # Convert to 8-bit
    color = (color[:, 0:3] * 255).astype(np.uint8)
    return color



def normalize(x: np.ndarray) -> np.ndarray:
    amin = np.min(x)
    amax = np.max(x)
    return (x - amin) / (amax - amin + 1e-6)


def float_to_color(x: np.ndarray, norm: bool = True) -> np.ndarray:
    if norm:
        x = normalize(x)

    color = cmap_numeric(np.clip(x, 0, 1))

    # Convert to 8-bit
    color = (color[:, 0:3] * 255).astype(np.uint8)
    return color


class EventType(Enum):
    INIT_MODEL = 0
    TIMESTEP = 1
    UPDATE_SURFELS = 2


class Logger:
    """Track variables and parameters during simulation, write outputs."""

    def __init__(
        self,
        dt: float,
        log_folder: str | Path,
        surfel_remodeling: int,
        time_between_remodeling: float,
        snapshot_every: Optional[int] = None,
        clear_pickle: bool = True,
        clear_vtp: bool = False,
        pickle_snapshot: bool = False,
        vtp_snapshot: bool = False,
        rerun_viewer: bool = True,
        testcase: str = "",
        repr_level: int = 1,
    ):
        self.t = 0.0        # simulation time
        self.sim_ctr = 0
        self.remodel_ctr = 0
        self.snapshot_every = snapshot_every
        self.pickle_ctr: Dict[str, int] = {}
        self.vtp_ctr: Dict[str, int] = {}
        self.pickle_snapshot = pickle_snapshot
        self.vtp_snapshot = vtp_snapshot
        self.dt = dt
        self.log_folder = Path(log_folder)
        self.log_folder.mkdir(parents=True, exist_ok=True)
        self.surfel_remodeling = surfel_remodeling
        self.time_between_remodeling = time_between_remodeling
        self.total_nb_simulations = int(
            (self.surfel_remodeling + 1) * int(self.time_between_remodeling / self.dt)
        )
        self.testcase = testcase
        self.repr_level = repr_level

        if clear_pickle:
            self.clear_log_folder(file_type="pkl")
        if clear_vtp:
            self.clear_log_folder(file_type="vtp")

        self.rerun_viewer = rerun_viewer
        if self.rerun_viewer:
            rerun.init("embryo_viewer", spawn=True)
            rerun.set_time("step", sequence=0)
            self.running_min: Optional[float] = None
            self.running_max: Optional[float] = None


    def init_text_logger(self) -> None:
        logging.basicConfig(
            filename=self.log_folder / "job.log",
            level=logging.DEBUG,
            format="%(asctime)s %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(console)

    def write_pickle(self, data: Any, file_stem: str = "sim_log") -> Path:
        ctr = self.pickle_ctr.setdefault(file_stem, 0)
        p_out = self.log_folder / f"{file_stem}_{str(ctr).zfill(3)}.pkl"
        with open(p_out, "wb") as fh:
            pickle.dump(data, fh)
        self.pickle_ctr[file_stem] += 1
        return p_out

    def log_event(
        self,
        event_type: EventType,
        model: Optional[Model] = None,
        surf: Optional[Surfels] = None,
        force: Optional[Forces] = None,
    ) -> None:
        if event_type is EventType.TIMESTEP:
            self.sim_ctr += 1
            self.t += self.dt
            if self.rerun_viewer:
                rerun.set_time("step", sequence=self.sim_ctr)

        elif event_type is EventType.UPDATE_SURFELS:
            if self.rerun_viewer:
                assert surf
                self._rerun_log_surfels(surf)

            if self.snapshot_every and not (self.sim_ctr - 1) % self.snapshot_every:
                self.repr()
                assert surf
                if force:
                    force.repr(self.repr_level)
                data = {
                    "x": surf.x,
                    "n": surf.n,
                    "cell_index": surf.cell_index,
                    "phase": surf.phase,
                    "force": force,
                    "neighbor_count": surf.neighbor_count,
                }
                if self.pickle_snapshot:
                    self.write_pickle(data)
                if self.vtp_snapshot:
                    self.save_vtp_snapshot(data=data)

        elif event_type is EventType.INIT_MODEL:
            assert model
            model.repr(self.repr_level)

    def _rerun_log_surfels(self, surf: Surfels) -> None:
        """Push surfel state to the Rerun viewer."""
        color_index = index_to_color(surf.cell_index)

        amin = float(np.min(surf.log_color))
        amax = float(np.max(surf.log_color))

        if self.running_min is None:
            self.running_min = amin
        if self.running_max is None:
            self.running_max = amax

        self.running_min = amin
        self.running_max = amax

        x = (surf.log_color - self.running_min) / (
            self.running_max - self.running_min + 1e-7
        )
        colors = float_to_color(x, norm=False)

        tc = self.testcase
        rerun.log(f"{tc}/surfels_index", rerun.Points3D(surf.x, colors=color_index))
        rerun.log(f"{tc}/surfels_color", rerun.Points3D(surf.x, colors=colors))
        rerun.log(
            f"{tc}/surfels_normal",
            rerun.Arrows3D(vectors=surf.n * NORMAL_SCALE, origins=surf.x, colors=color_index),
        )
        rerun.log(
            f"{tc}/force",
            rerun.Arrows3D(vectors=surf.debug_force * NORMAL_SCALE, origins=surf.x, colors=color_index),
        )

    def log_signal(self, signal: Signal) -> None:
        if self.rerun_viewer and signal.points is not None:
            rerun.log(f"{self.testcase}/signal_points", rerun.Points3D(signal.points))

    def clear_log_folder(self, file_type: str) -> None:
        for f in self.log_folder.glob(f"*.{file_type}"):
            f.unlink()

    def log_folder_to_vtp(self, file_stem: str = "sim_log") -> None:
        self.clear_log_folder(file_type="vtp")
        logging.info("Transforming pickles into vtp files:")
        for pickle_i in sorted(self.log_folder.glob(f"{file_stem}*.pkl")):
            pkl_ix = int(re.findall(f".*{file_stem}_(.*).pkl", str(pickle_i))[0])
            with open(pickle_i, "rb") as fh:
                data = pickle.load(fh)
            simlog_to_vtp(
                data["x"], data["n"], data["cell_index"], data["phase"],
                data["force"], data["neighbor_count"],
                output_vtp=self.log_folder / f"{file_stem}_{str(pkl_ix).zfill(3)}.vtp",
            )
            logging.info(f"|{pkl_ix}", end="")
        logging.info(f"\nVTP written to {self.log_folder.absolute()}")

    def save_vtp_snapshot(
        self,
        pickle_file: Optional[Path] = None,
        data: Optional[dict] = None,
        file_stem: str = "sim_log",
    ) -> None:
        self.vtp_ctr.setdefault(file_stem, 0)
        if pickle_file:
            with open(pickle_file, "rb") as fh:
                data = pickle.load(fh)
            t_ix = int(re.findall(f".*{file_stem}_(.*).pkl", str(pickle_file))[0])
        else:
            t_ix = self.vtp_ctr[file_stem]

        assert data is not None
        simlog_to_vtp(
            data["x"], data["n"], data["cell_index"], data["phase"],
            data["force"], data["neighbor_count"],
            output_vtp=self.log_folder / f"{file_stem}_{str(t_ix).zfill(3)}.vtp",
        )
        self.vtp_ctr[file_stem] += 1

    def repr(self) -> None:
        snap = self.vtp_ctr if self.vtp_snapshot else self.pickle_ctr
        print(
            f"---- t={self.t:.4f} | step={self.sim_ctr}/{self.total_nb_simulations}"
            f" | remodel_ctr={self.remodel_ctr} | snapshots={snap}",
            flush=True,
        )

    @staticmethod
    def _json_default(obj: Any) -> Any:
        try:
            import numpy as _np
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            if isinstance(obj, _np.integer):
                return int(obj)
            if isinstance(obj, _np.floating):
                return float(obj)
        except Exception:
            pass
        return str(obj)

    @staticmethod
    def _validate_metrics(metrics: Dict[str, Any]) -> None:
        """Warn (not raise) when required metric keys are missing."""
        missing_top = _REQUIRED_METRIC_KEYS - metrics.keys()
        if missing_top:
            logging.warning(
                "log_metrics: missing required keys %s — "
                "plot_compare_metrics.py may produce NaN series.",
                sorted(missing_top),
            )
        for stat_key in ("density_stats", "neighbor_stats", "force_stats", "curvature_stats"):
            stat = metrics.get(stat_key)
            if isinstance(stat, dict):
                missing_stat = _REQUIRED_STAT_KEYS - stat.keys()
                if missing_stat:
                    logging.warning(
                        "log_metrics: %s is missing sub-keys %s.",
                        stat_key, sorted(missing_stat),
                    )

    def log_metrics(self, metrics: Dict[str, Any], file_stem: str = "metrics") -> None:
        """Append one JSON-line per timestep to ``<log_folder>/<file_stem>.jsonl``.

        The file is consumed by ``plot_compare_metrics.py``.  Both GPU and CPU
        models must call this with the canonical schema documented at the top of
        this module.

        Parameters
        ----------
        metrics:
            Per-timestep metrics dict.  See module docstring for required keys.
        file_stem:
            Base name of the output file (without extension).  Defaults to
            ``"metrics"`` → ``metrics.jsonl``.
        """
        # Soft validation — emit warnings but never crash the simulation.
        self._validate_metrics(metrics)

        out = self.log_folder / f"{file_stem}.jsonl"
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics, default=self._json_default) + "\n")

        # Mirror simple scalars to Rerun for real-time visual inspection.
        if self.rerun_viewer:
            self._rerun_log_metrics(metrics)

    def _rerun_log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Push selected scalar metrics to the Rerun timeline."""
        try:
            tc = self.testcase

            # Density
            ds = metrics.get("density_stats")
            if isinstance(ds, dict) and ds.get("mean") is not None:
                rerun.log(f"{tc}/metrics/avg_density", rerun.Scalar(float(ds["mean"])))

            # Force magnitude
            fs = metrics.get("force_stats")
            if isinstance(fs, dict) and fs.get("mean") is not None:
                rerun.log(f"{tc}/metrics/avg_force_mag", rerun.Scalar(float(fs["mean"])))

            # Neighbour count (canonical key; fall back to aliases)
            ns = (
                metrics.get("neighbor_stats")
                or metrics.get("hashgrid_stats")
                or metrics.get("kdt_stats")
            )
            if isinstance(ns, dict) and ns.get("mean") is not None:
                rerun.log(f"{tc}/metrics/avg_neighbors", rerun.Scalar(float(ns["mean"])))

            # Total volume across all cells
            vols = metrics.get("volumes")
            if isinstance(vols, dict):
                total_vol = sum(v for v in vols.values() if v is not None)
                rerun.log(f"{tc}/metrics/total_volume", rerun.Scalar(float(total_vol)))

            # Step and total wall-clock time
            step_ms = metrics.get("step_time_ms")
            if step_ms is not None:
                rerun.log(f"{tc}/metrics/step_time_ms", rerun.Scalar(float(step_ms)))

            total_ms = metrics.get("total_time_ms")
            if total_ms is not None:
                rerun.log(f"{tc}/metrics/total_time_ms", rerun.Scalar(float(total_ms)))

        except Exception:
            pass  # Rerun failures must never crash the simulation