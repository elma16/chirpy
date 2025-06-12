from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from ..function.base import Function


class Optimizer(ABC):
    @abstractmethod
    def solve(self, fun: Function, m0: np.ndarray) -> np.ndarray: ...
