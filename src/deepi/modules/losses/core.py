from abc import abstractmethod
import numpy as np 

from deepi.modules import Module


class Loss(Module): 

    def __init__(
            self,
            _type: str,
    ): 
        super().__init__(f"loss.{_type}", False)

    @abstractmethod
    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> float: 
        raise NotImplementedError()
    
    @abstractmethod 
    def backward(self) -> np.ndarray: 
        raise NotImplementedError


class MAE(Loss): 

    def __init__(self): 
        super().__init__("mae")

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> float: 
        diff = y_hat - y
        
        if self._is_training: 
            self.dx = np.sign(diff, dtype=float) / diff.size
        
        return np.abs(diff).mean()

    def backward(self) -> np.ndarray: 
        return self.dx