from abc import ABC, abstractmethod 
import numpy as np

class Regularizer(ABC): 

    def __init__(
            self, 
            gamma: float,
            _type: str
    ): 
        self._type = f"regularizer.{_type}"
        self.gamma = gamma

    @abstractmethod 
    def regularization(self, x: np.ndarray) -> np.ndarray: 
        pass
        
    def __call__(self, x: np.ndarray) -> np.ndarray: 
        return self.gamma * self.regularization(x)