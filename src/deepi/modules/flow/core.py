from typing import Tuple, Optional

import numpy as np 

from deepi.modules import Module 


class Flow(Module): 

    def __init__(
            self,
            _type: str
    ): 
        super().__init__(f"flow.{_type}", False)


class Input(Flow): 

    def __init__(
            self, 
            in_shape: Tuple[int, ...],
            store_gradient: bool = False
    ): 
        super().__init__("input")
        self.in_shape = in_shape 
        self.store_gradient = store_gradient

    def forward(self, x: Optional[np.ndarray] = None) -> np.ndarray: 
        return x if x is not None else np.empty(self.in_shape, dtype=float)
    
    def backward(self, dy: np.ndarray): 
        if self.store_gradient: 
            self.cache = dy 


class Flatten(Flow): 
    
    def __init__(self): 
        super().__init__("flatten")

    def forward(self, x: np.ndarray) -> np.ndarray: 
        if self._is_training: 
            self.cache = x.shape
        return x.reshape(len(x), -1)
    
    def backward(self, dy: np.ndarray) -> np.ndarray: 
        in_shape = self.cache 
        return dy.reshape(in_shape)
    

class Reshape(Flow):

    def __init__(
            self, 
            out_shape: Tuple[int, ...]
    ):
        super().__init__("reshape")
        self.out_shape = out_shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self._is_training:
            self.cache = x.shape 
        return x.reshape((len(x),) + self.out_shape)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        in_shape = self.cache
        return dy.reshape(in_shape)
    

class Concatenate(Flow): 

    def __init__(
            self,
            axis: int = -1
    ): 
        super().__init__("concatenate")
        self.axis = axis

    def forward(self, xs: Tuple[np.ndarray, ...]) -> np.ndarray:
        self.cache = [x.shape[self.axis] for x in xs]
        return np.concatenate(xs, axis=self.axis)

    def backward(self, dy: np.ndarray) -> Tuple[np.ndarray, ...]:
        sizes = self.cache
        indices = np.cumsum(sizes)[:-1] 
        dx_tuple = tuple(np.split(dy, indices, axis=self.axis))
        return dx_tuple