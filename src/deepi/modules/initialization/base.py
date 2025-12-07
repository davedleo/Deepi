from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from deepi.modules import Module


class Initializer(ABC):

    def __init__(self, _type: str):
        self._type = f"initializer.{_type}"

    @abstractmethod
    def rule(self, shape: Tuple[int, ...]) -> np.ndarray:
        pass

    def init(self, module: Module):
        if module.has_params:
            params = module.get_params()
            for k, v in params.items():
                if isinstance(v, tuple):
                    params[k] = self.rule(v)

    def fan_in(self, shape: Tuple[int, ...]) -> int:
        if len(shape) == 2:
            fan_in = shape[0]
        elif len(shape) == 3:
            fan_in = shape[1] * shape[2]
        elif len(shape) == 4:
            fan_in = shape[1] * shape[2] * shape[3]
        else:
            raise ValueError()
        return fan_in

    def fan_out(self, shape: Tuple[int, ...]) -> int:
        if len(shape) == 2:
            fan_out = shape[1]
        elif len(shape) == 3:
            fan_out = shape[0] * shape[2]
        elif len(shape) == 4:
            fan_out = shape[0] * shape[2] * shape[3]
        else:
            raise ValueError()
        return fan_out

    def __str__(self) -> str:
        parts = self._type.split(".")
        return ".".join([part.capitalize() for part in parts])

    def __repr__(self) -> str:
        return self.__str__()
