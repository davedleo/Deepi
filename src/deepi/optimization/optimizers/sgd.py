from typing import Dict, Optional
import numpy as np 

from deepi import Model
from deepi.optimization.optimizers import Optimizer
from deepi.optimization.regularization import Regularizer 


class SGD(Optimizer): 

    def __init__(
            self, 
            model: Model,
            lr: float = 0.01,
            momentum: float = 0.0,
            dampening: float = 0.0,
            nesterov: bool = False,
            regularizer: Optional[Regularizer] = None,
            maximize: bool = False
    ): 
        super().__init__(model, lr, regularizer, False, maximize, "sgd")
        self.mu = momentum 
        self.tau = dampening 
        self.nesterov = nesterov

        for module_buffer in self.buffer.values(): 
            for buffer in module_buffer.values(): 
                buffer["velocity"] = None

    def direction(
            self, 
            dw: np.ndarray,
            buffer: Dict[str, np.ndarray]
    ) -> np.ndarray: 
        if self.mu != 0.0: 
            v = buffer["velocity"]
            if v is None: 
                buffer["velocity"] = dw.copy()
            else:  
                v *= self.mu 
                v += (1.0 - self.tau) * dw 
                dw = dw + self.mu * v if self.nesterov else v

        return self.lr * dw