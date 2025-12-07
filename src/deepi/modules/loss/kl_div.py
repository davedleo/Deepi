from typing import Optional
import numpy as np
from deepi.modules.loss import Loss 

class KLDiv(Loss): 
    
    def __init__(self, eps: float = 1e-16, reduction: Optional[str] = "mean"): 
        super().__init__("kldiv", reduction)
        self.eps = eps

    def transform(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray: 
        loss = y * (np.log(y + self.eps) - np.log(y_hat + self.eps))
        return loss.sum(1)
    
    def gradients(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return - y / (y_hat + self.eps)