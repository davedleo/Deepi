from abc import ABC, abstractmethod
from math import sqrt 
from typing import Optional, Tuple, Union

import numpy as np

from deepi.modules import Module
from deepi.modules.initialization.utils import get_gain


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
    

class XavierUniform(Initializer): 

    def __init__(
            self,
            gain: Union[str, float] = 1.0,
            negative_slope: Optional[float] = None
    ): 
        super().__init__("xavier_uniform")
        if isinstance(gain, str): 
            gain = get_gain(gain, negative_slope)

        self.gain = gain

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        fan_in, fan_out = self.fans(shape)
        r = self.gain * sqrt(6.0 / (fan_in + fan_out)) 
        return  np.random.uniform(-r, r, shape)
    

class XavierNormal(Initializer): 

    def __init__(
            self,
            gain: Union[str, float] = 1.0,
            negative_slope: Optional[float] = None
    ): 
        super().__init__("xavier_normal")
        if isinstance(gain, str): 
            gain = get_gain(gain, negative_slope)

        self.gain = gain

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        fan_in, fan_out = self.fans(shape)
        std = self.gain * sqrt(2.0 / (fan_in + fan_out)) 
        return  np.random.normal(0.0, std, shape) 
    
    
class KaimingUniform(Initializer):
    def __init__(
        self,
        fan_mode: str = "in",
        gain: Union[str, float] = "leaky_relu",
        negative_slope: Optional[float] = None
    ):
        super().__init__("kaiming_uniform")
        if fan_mode not in ("in", "out"):
            raise ValueError("fan_mode must be 'in' or 'out'")
        self.fan_mode = fan_mode
        if isinstance(gain, str):
            self.gain = get_gain(gain, negative_slope)
        else:
            self.gain = gain
        self.negative_slope = negative_slope

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        fan_in, fan_out = self.fans(shape)
        fan = fan_in if self.fan_mode == "in" else fan_out
        bound = sqrt(6.0 / fan) * self.gain
        return np.random.uniform(-bound, bound, shape)


class KaimingNormal(Initializer):
    def __init__(
        self,
        fan_mode: str = "in",
        gain: Union[str, float] = "leaky_relu",
        negative_slope: Optional[float] = None
    ):
        super().__init__("kaiming_normal")
        if fan_mode not in ("in", "out"):
            raise ValueError("fan_mode must be 'in' or 'out'")
        self.fan_mode = fan_mode
        if isinstance(gain, str):
            self.gain = get_gain(gain, negative_slope)
        else:
            self.gain = gain
        self.negative_slope = negative_slope

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        fan_in, fan_out = self.fans(shape)
        fan = fan_in if self.fan_mode == "in" else fan_out
        std = sqrt(2.0 / fan) * self.gain
        return np.random.normal(0.0, std, shape)


class Orthogonal(Initializer):
    def __init__(
        self,
        gain: Union[str, float] = 1.0,
        negative_slope: Optional[float] = None
    ):
        super().__init__("orthogonal")
        if isinstance(gain, str):
            gain = get_gain(gain, negative_slope)
        self.gain = gain

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        if len(shape) < 2:
            raise ValueError("Orthogonal initializer requires at least 2 dimensions")
        flat_shape = (shape[0], int(np.prod(shape[1:])))
        a = np.random.normal(0.0, 1.0, flat_shape)
        q, r = np.linalg.qr(a)
        d = np.diag(r)
        ph = np.sign(d)
        q *= ph
        q = q.reshape(shape)
        return self.gain * q