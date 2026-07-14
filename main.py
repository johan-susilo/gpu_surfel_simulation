"""
Created on 4 Nov 2022

@author: robjelier
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional
import warnings
import shutil
from typing import Tuple
import cProfile

import numpy as np
import tifffile  # type: ignore[import]

from core.types import Array
from core.surfels import Surfels
from core.logger import Logger, EventType
from core.signal import Signal
from core.model_fluid import Fluid
from core.model import Model
from core.parameters import Parameters
from core.vtk_utils import signal_to_vtp


def unreachable():
    assert False, "unreachable"


def draw_on_image(x, im, z_res, start_index, n_steps):
    size = 1
    for v in range(x.shape[1]):
        for i in range(n_steps):
            p = x[v, :].astype("int")
            z = int(np.floor(p[0]) / z_res)
            im[
                (z - 1) : (z + 1),
                (p[1] - size) : (p[1] + size),
                (p[2] - size) : (p[2] + size),
            ] = (
                start_index + i
            )


def handle_output(prm, model: Model, out_folder: Path, signal: Optional[Signal], sim_log: Logger):
    warnings.filterwarnings("ignore")

    if prm.write_tif:
        assert signal
        tifffile.imwrite(out_folder / "SurfelResult.tif", signal.im, photometric="minisblack")

    if prm.vtp_snapshot:
        logging.info(f"\nVTP written to {sim_log.log_folder.absolute()}")
        if not sim_log.vtp_snapshot:
            sim_log.log_folder_to_vtp()

    prm.save_to_yaml(out_folder / "param.yml")

    if prm.backup_vtp_folder:
        bu_folder = prm.paths["output_folder"] / "last_run"
        bu_folder.mkdir(parents=True, exist_ok=True)
        for file_i in bu_folder.glob("*.vtp"):
            file_i.unlink()
        for ix, file_i in enumerate(sorted(sim_log.log_folder.glob("*.vtp"))):
            if ix < sim_log.vtp_ctr.get("sim_log", 0):
                shutil.copy(file_i, bu_folder)
        print(f"vtp files copied to {bu_folder.absolute()}")


def fit_sphere(points) -> Tuple[Array, float]:
    # https://lucidar.me/en/mathematics/least-squares-fitting-of-sphere/
    A = np.hstack([points, np.ones((points.shape[0], 1))])
    B = np.sum(points**2, axis=1)
    sol = np.linalg.lstsq(A, B, rcond=None)[0]

    est_center = sol[0:3] / 2
    est_radius = np.sqrt(sol[0] ** 2 + sol[1] ** 2 + sol[2] ** 2 + 4 * sol[3]) / 2
    return (est_center, est_radius)


def check_bubble_radii(surfels: Surfels):
    points_a = surfels.x[surfels.phase == 0]
    points_b = surfels.x[surfels.phase == 1]
    points_c = surfels.x[surfels.phase == 2]

    r_a = fit_sphere(points_a)[1]
    r_b = fit_sphere(points_b)[1]
    r_c = fit_sphere(points_c)[1]

    print("radii:")
    print(f"{r_a}, {r_b}, {r_c}")
    curv_obs = 1 / r_c
    curv_pred = np.abs((1 / r_a) - (1 / r_b))
    print(f"curv_pred = {curv_pred}")
    print(f"curv_obs = {curv_obs}")
    print(f"r_pred = {1/curv_pred}")
    print(f"r_obs = {1/curv_obs}")


def main(testcase: str):
    prm = Parameters.new_from_yaml(f"params/param_{testcase}.yml")

    out_folder = prm.paths["output_folder"] / testcase
    out_folder.mkdir(parents=True, exist_ok=True)

    sim_log = Logger(
        prm.dt,
        out_folder,
        snapshot_every=prm.snapshot_every,
        surfel_remodeling=prm.surfel_remodeling,
        time_between_remodeling=prm.time_between_remodeling,
        pickle_snapshot=prm.pickle_snapshot,
        vtp_snapshot=prm.vtp_snapshot,
        clear_vtp=prm.clear_vtp,
        testcase=testcase,
        repr_level=prm.repr_level,
    )

    signal = None
    if prm.k_signal != 0:
        signal = Signal(prm)
        sim_log.log_signal(signal)

    surf = Surfels(prm.surfel_elements, prm)

    model: Model = Fluid(prm)
    sim_log.log_event(EventType.INIT_MODEL, model=model)

    logging.info(
        f"surfel_remodeling:{prm.surfel_remodeling}, time_between_remodeling:{prm.time_between_remodeling}"
    )

    sim_time = 0.0

    t0 = time.perf_counter()

    # add some noise
    surf.x += np.random.normal(size=surf.x.shape) * prm.initial_noise * surf.d0

    print(np.mean(surf.x, axis=0))

    skip_first = True

    # This is the main loop!
    for t_step in range(prm.surfel_remodeling + 1):
        n_steps = int(prm.time_between_remodeling / prm.dt)

        if skip_first:
            # model.estimate_normals(surf)
            pass
        else:
            # model.estimate_normals(surf)
            model.optimize_surfel_density(surf, sim_log)
        skip_first = False

        model.integrate(surf, n_steps, signal, sim_log)
        sim_time += n_steps * prm.dt

        # check_bubble_radii(surf)

        if prm.write_tif:
            assert signal
            # TODO: not sure how this was supposed to work
            draw_on_image(surf.x, signal.im, prm.z_res, 2 + t_step, n_steps)

    # check_bubble_radii(surf)

    np.save(out_folder / "surfels_x", surf.x)
    np.save(out_folder / "surfels_n", surf.n)
    np.save(out_folder / "surfels_i", surf.cell_index)

    print(f"Simulations took {(time.perf_counter() - t0):.3f}s")

    total_run_ms = float((time.perf_counter() - t0) * 1000)
    run_summary = {
        "backend": "gpu",
        "total_time_ms": total_run_ms,
        "total_steps": int(sim_log.sim_ctr),
    }
    # write run summary to output folder for later comparison
    with open(out_folder / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(run_summary, fh)

    handle_output(prm, model, out_folder, signal, sim_log)


if __name__ == "__main__":
    parameter_name = "d0"

    if len(sys.argv) > 1:
        parameter_name = sys.argv[1].lower()

    profile = False

    if profile:
        cProfile.run(
            "main(parameter_name)",
            f"profile_{parameter_name}.prof",
        )
    else:
        main(parameter_name)
