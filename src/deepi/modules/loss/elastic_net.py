from typing import Optional
import numpy as np
from deepi.modules.loss import Loss 

class ElasticNet(Loss): 
    
    def __init__(
            self,
            alpha: float = 0.5, # smoothness
            reduction: Optional[str] = "mean",
            eps: float = 1e-12
    ): 
        super().__init__("elastic_net", reduction)
        if alpha < 0.0 or alpha > 1.0: 
            raise ValueError()
        self.alpha = alpha
        self.eps = eps

    def transform(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray: 
        l1 = np.abs(y_hat - y).sum(1)
        l2 = np.sqrt(((y_hat - y) ** 2.0).sum(1))
        return (1.0 - self.alpha) * l1 + self.alpha * l2
    
    def gradients(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        diff = y_hat - y
        l2_norm = np.sqrt(np.sum(diff**2, axis=1, keepdims=True)) + self.eps
        grad_l1 = np.sign(diff, dtype=np.float64)
        grad_l2 = diff / l2_norm
        return (1.0 - self.alpha) * grad_l1 + self.alpha * grad_l2