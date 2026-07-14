from abc import ABC, abstractmethod
from typing import Optional

from core.surfels import Surfels
from core.logger import Logger
from core.signal import Signal


# force model interface
# should not have any properties or variables, only methods
class Model(ABC):
    @abstractmethod
    def integrate(self, surfels: Surfels, n: int, signal: Optional[Signal], logger: Logger):
        pass

    @abstractmethod
    def optimize_surfel_density(self, surfels: Surfels, logger: Logger):
        pass

    @abstractmethod
    def estimate_normals(self, surfels: Surfels):
        pass

    @abstractmethod
    def repr(self, repr_level: int):
        pass
