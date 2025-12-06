from typing import Optional
import numpy as np
from deepi.modules.loss import Loss 

class MSE(Loss): 
    
    def __init__(self, reduction: Optional[str] = "mean"): 
        super().__init__("mse", reduction)

    def transform(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray: 
        return ((y_hat - y) ** 2.0).mean(axis=1)
    
    def gradients(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2.0 * (y_hat - y)