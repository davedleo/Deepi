from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class Module(ABC):

    def __init__(
        self,
        _type: str,
        _has_params: bool,
    ):
        self._type = f"module.{_type}"

        self.next: List["Module"] = []
        self.prev: List["Module"] = []

        self.dx: np.ndarray = 0.0
        self._is_training: bool = False

        self._has_params = _has_params
        if self._has_params:
            self.params: Dict[str, np.ndarray] = dict()
            self.grads: Dict[str, np.ndarray] = dict()

    def __str__(self) -> str:
        parts = self._type.split('.')
        capitalized = [part.capitalize() for part in parts]
        return '.'.join(capitalized)

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def type(self):
        return self._type

    @property
    def has_params(self):
        return self._has_params

    @abstractmethod
    def forward(self, *x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def backward(self, dy: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def link(self, module: "Module") -> None:
        if module not in self.next:
            self.next.append(module)
        if self not in module.prev:
            module.prev.append(self)

    def train(self) -> None:
        self._is_training = True

    def eval(self) -> None:
        self._is_training = False

    def clear(self) -> None:
        self.dx = 0.0
        if self._has_params:
            self.grads = dict()
