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
    
    @property
    def type(self) -> str:
        return self._type

    def __str__(self) -> str:
        parts = self._type.split(".")
        return ".".join([part.capitalize() for part in parts])

    def __repr__(self) -> str:
        return self.__str__()