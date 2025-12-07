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
            for k, shape in params.items(): 
                params[k] = self.rule(shape)

    def __str__(self) -> str:
        parts = self._type.split(".")
        return ".".join([part.capitalize() for part in parts])

    def __repr__(self) -> str:
        return self.__str__()
