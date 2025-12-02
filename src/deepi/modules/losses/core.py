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


class ElasticNet(Loss):

    def __init__(self, l1: float = 0.5, l2: float = 0.5):
        super().__init__("elasticnet")
        total = l1 + l2
        self.alpha_l1 = l1 / total
        self.alpha_l2 = l2 / total

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        diff = y_hat - y
        l1_term = np.abs(diff)
        l2_term = diff ** 2

        if self._is_training:
            self.dx = self.alpha_l1 * np.sign(diff) / diff.size + self.alpha_l2 * 2.0 * diff / diff.size

        return (self.alpha_l1 * l1_term + self.alpha_l2 * l2_term).mean()