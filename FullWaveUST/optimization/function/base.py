from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod


class Function(ABC):
    @abstractmethod
    def value(self, m: np.ndarray) -> float: ...

    @abstractmethod
    def gradient(self, m: np.ndarray) -> np.ndarray: ...
