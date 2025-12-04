from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from deepi.modules import Module


class Initializer(ABC):

    def __init__(self, _type: str):
        self._type = f"initializer.{_type}"

    def __call__(self, module: Module):
        if module.has_params:
            params = module.get_params()
            for k, v in params.items():
                if (k != "b") or ("running" not in k.split(".")):
                    params[k] = self.initialize(v)

    def fans(self, shape: Tuple[int, ...]) -> Tuple[float, float]:
        n_axis = len(shape)
        if n_axis == 2:
            fan_in, fan_out = shape
        elif n_axis == 3:
            out_channels, in_channels, kernel_size = shape
            fan_in = in_channels * kernel_size
            fan_out = out_channels * kernel_size
        elif n_axis == 4:
            out_channels, in_channels, kernel_height, kernel_width = shape
            receptive_field_size = kernel_height * kernel_width
            fan_in = in_channels * receptive_field_size
            fan_out = out_channels * receptive_field_size
        else:
            raise NotImplementedError()

        return fan_in, fan_out

    @abstractmethod
    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError()


class Uniform(Initializer): 

    def __init__(
            self,
            low: float = 0.0, 
            high: float = 1.0
    ): 
        super().__init__("uniform")
        self.low = low 
        self.high = high

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray: 
        return np.random.uniform(self.low, self.high, shape)
    

class Normal(Initializer): 

    def __init__(
            self,
            mean: float = 0.0, 
            std: float = 1.0
    ): 
        super().__init__("normal")
        self.mean = mean 
        self.std = std

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray: 
        return np.random.normal(self.mean, self.std, shape)  