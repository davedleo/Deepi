from typing import Optional
import numpy as np
from deepi.modules.loss import Loss 

class MAE(Loss): 
    
    def __init__(self, reduction: Optional[str] = "mean"): 
        super().__init__("rmse", reduction)

    def transform(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray: 
        return (np.abs(y_hat - y)).mean(1)
    
    def gradients(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sign(y_hat - y, dtype=np.float64) / y_hat.shape[1]