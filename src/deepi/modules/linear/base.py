from abc import abstractmethod 
from typing import Dict, Tuple

import numpy as np  

from deepi.modules import Module 


class Linear(Module): 
    
    def __init__(
            self, 
            _type: str, 
            _has_bias: bool,
            _out_size: int
    ): 
        super().__init__(f"linear.{_type}", True)
        self._has_bias = _has_bias 
        if self._has_bias: 
            self.params["b"] = np.zeros((1, _out_size), dtype=np.float64)
        
    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray: 
        pass 

    @abstractmethod 
    def gradients(self, dy: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]: 
        pass