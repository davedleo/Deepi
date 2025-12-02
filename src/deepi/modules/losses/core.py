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
    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray: 
        raise NotImplementedError()
    
    def backward(self) -> np.ndarray: 
        raise self.dx.copy()


class MAE(Loss): 

    def __init__(self): 
        super().__init__("mae")

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray: 
        diff = y_hat - y
        
        if self._is_training: 
            self.dx = np.sign(diff, dtype=float) / diff.size
        
        return np.abs(diff).mean()


class MSE(Loss):

    def __init__(self):
        super().__init__("mse")

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        diff = y_hat - y

        if self._is_training:
            self.dx = 2.0 * diff / diff.size

        return np.mean(diff ** 2)