from abc import abstractmethod 
from typing import Optional

import numpy as np  

from deepi.modules import Module 


_REDUCTIONS = {None, "mean", "sum"}


class Loss(Module): 
    
    def __init__(
            self, 
            _type: str,
            reduction: Optional[str] = "mean"
    ): 
        super().__init__(f"loss.{_type}", False)
        if reduction not in _REDUCTIONS: 
            raise ValueError()
        
        self.reduction = reduction
        
    @abstractmethod
    def loss_transform(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray: 
        pass 

    @abstractmethod 
    def loss_gradient(self) -> np.ndarray: 
        pass 

    def apply_reduction(self, loss: np.ndarray) -> np.ndarray: 
        if self.reduction is None: 
            return loss
        elif self.reduction == "sum": 
            return loss.sum(keepdims=True)
        else: 
            return loss.mean(keepdims=True)

    def transform(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray: 
        loss = self.loss_transform(y, y_hat)
        return self.apply_reduction(loss)

    def gradients(self, dy: np.ndarray) -> np.ndarray: 
        gradient = self.loss_gradient() * dy
        if self.reduction and self.reduction == "mean": 
            gradient /= gradient.shape[0]
            
        return gradient