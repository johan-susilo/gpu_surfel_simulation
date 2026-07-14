"""
Created on 26 Dec 2022

@author: wimthiels
"""
from enum import Enum
import pprint
from pathlib import Path
from typing import List, Any, Dict, Optional, Literal
from pydantic import BaseModel
import yaml  # type: ignore[import]
from yaml.loader import SafeLoader  # type: ignore[import]


def load_yaml(filename: str | Path):
    with open(filename, encoding="utf-8") as f:
        prm = yaml.load(f, Loader=SafeLoader)

    if "default_params" in prm:
        with open(Path(filename).parent / prm["default_params"], encoding="utf-8") as f:
            default_parms = yaml.load(f, Loader=SafeLoader)
            default_parms.update(prm)
            prm = default_parms

    if prm.get("verbose"):
        pprint.pprint(prm)
    else:
        print(prm.get("testcase_description"))

    # TODO: this is stupid, just use strings instead of path objects
    if "paths" in prm:
        for k, v in prm.get("paths").items():
            prm["paths"][k] = Path(v)

    return prm


class Parameters(BaseModel):
    testcase: Optional[str]
    testcase_description: str
    default_params: Optional[str] = None
    paths: Dict[str, Path]

    # parameters
    dt: float = 1.0
    d0: float | Literal["auto"] = 1.0
    scale: float = 1.0

    k_dist: float = 0
    k_repulsion: float = 0
    k_plane: float = 0
    k_tilt: float = 0
    k_pressure: float = 0
    k_signal: float = 0
    
    rest_radius: float = 50.0
    rho_target: float = 0.5

    xy_res: float = 1.0
    z_res: float = 1.0
    signal_downsample: float = 2.0

    initial_noise: float = 0.0

    snapshot_every: int = 5
    surfel_elements: List[List[Any]] = []

    # TODO: should just have a total runtime instead of this
    surfel_remodeling: int = 0
    time_between_remodeling: float = 100

    # output options
    rerun_viewer: bool = True
    repr_level: int = 1
    verbose: bool = False
    pickle_snapshot: bool = False
    vtp_snapshot: bool = False
    clear_vtp: bool = True
    write_tif: bool = False
    # True is probably a better default for this, but it's mostly a waste during testing
    backup_vtp_folder: bool = False

    class Config:
        frozen = True
        extra = "forbid"

    @classmethod
    def new_from_yaml(cls, filename: str | Path):
        data = load_yaml(filename)
        new = cls(**data)

        return new

    def save_to_yaml(self, filename: str | Path):
        with open(filename, "w", encoding="utf-8") as file:
            yaml.dump(self.dict(), file)
