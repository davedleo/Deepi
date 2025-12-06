from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

ArrayOrTuple = Union[np.ndarray, Tuple[np.ndarray, ...]]


class Module(ABC):
    
    def __init__(self, _type: str, _has_params: bool = False):
        self._type = f"module.{_type}"
        self.next: List["Module"] = []
        self.prev: List["Module"] = []

        self.x: Optional[ArrayOrTuple] = None
        self.y: Optional[ArrayOrTuple] = None
        self.dy: Optional[ArrayOrTuple] = None

        self._is_training: bool = False
        self._has_params = _has_params
        self.params: Dict[str, np.ndarray] = dict()
        self.grads: Dict[str, np.ndarray] = dict()

    @abstractmethod
    def transform(self, x: ArrayOrTuple) -> ArrayOrTuple:
        """Forward computation: x can be a single array or a tuple of arrays."""
        pass

    @abstractmethod
    def gradients(self, dy: ArrayOrTuple) -> ArrayOrTuple:
        """Backward computation: returns gradient(s) w.r.t inputs."""
        pass

    def forward(self, x: ArrayOrTuple) -> ArrayOrTuple: 
        y = self.transform(x)

        if self._is_training: 
            self.x = x # No inplace operations by design
            self.y = y # No inplace operations by design

        return y

    def backward(self, dy: ArrayOrTuple):
        """
        Backward pass with accumulated gradient stored in self.dy.
        dy can be a single array or a tuple of arrays.
        Important: make defensive copies so we never mutate user-provided arrays.
        """
        # Make defensive copy of incoming gradient(s) to avoid mutating caller's arrays
        if isinstance(dy, np.ndarray):
            dy_copy = np.array(dy, copy=True)
        else:
            # tuple of arrays
            dy_copy = tuple(np.array(d, copy=True) for d in dy)

        # Accumulate incoming gradient into self.dy
        if self.dy is None:
            self.dy = dy_copy
        else:
            # both self.dy and dy_copy must have the same structure (ndarray or tuple)
            if isinstance(self.dy, np.ndarray) and isinstance(dy_copy, np.ndarray):
                # in-place accumulation on self.dy is fine because self.dy is owned by the module
                self.dy = self.dy + dy_copy
            elif isinstance(self.dy, tuple) and isinstance(dy_copy, tuple):
                self.dy = tuple(a + b for a, b in zip(self.dy, dy_copy))
            else:
                # mismatched shapes/types — raise an informative error
                raise TypeError("Mismatched dy types when accumulating gradients")

        # Compute gradient(s) to propagate upstream
        dx = self.gradients(self.dy)

        # Propagate upstream according to dx shape
        if isinstance(dx, tuple):
            for prev_module, dx_i in zip(self.prev, dx):
                prev_module.backward(dx_i)
        else:
            for prev_module in self.prev:
                prev_module.backward(dx)

    def link(self, module: "Module"):
        """Link this module to the next module in the graph."""
        if module not in self.next:
            self.next.append(module)
        if self not in module.prev:
            module.prev.append(self)

    def get_params(self) -> Dict[str, np.ndarray]:
        return self.params

    def load_params(self, params: Dict[str, np.ndarray]):
        for k, v in params.items():
            self.params[k] = v

    def train(self):
        self._is_training = True

    def eval(self):
        self._is_training = False

    def clear(self):
        """Clear cached values and gradients."""
        self.x = None
        self.y = None
        self.dy = None
        if self._has_params:
            self.grads = dict()

    def __str__(self) -> str:
        parts = self._type.split(".")
        return ".".join([part.capitalize() for part in parts])

    def __repr__(self) -> str:
        return self.__str__()