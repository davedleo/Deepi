from typing import Dict, Optional
import numpy as np 

from deepi import Model 
from deepi.optimization.optimizers import Optimizer 
from deepi.optimization.regularization import Regularizer 


class Adamax(Optimizer): 

    def __init__(
            self,
            model: Model, 
            lr: float = 0.002,
            beta1: float = 0.9,
            beta2: float = 0.999,
            eps: float = 1e-8,
            regularizer: Optional[Regularizer] = None,
            maximize: bool = False
    ): 
        super().__init__(model, lr, regularizer, False, maximize, "adamax")
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.eps = eps

        for module_id, module in self.modules.items(): 
            for k, buffer in self.buffer[module_id].items(): 
                buffer["t"] = 0
                buffer["velocity"] = np.zeros_like(module.params[k])
                buffer["dw_max"] = np.zeros_like(module.params[k])

    def direction(
            self, 
            dw: np.ndarray,
            buffer: Dict[str, np.ndarray]
    ) -> np.ndarray: 
        buffer["t"] += 1
        t = buffer["t"]

        velocity = buffer["velocity"]
        velocity *= self.beta1 
        velocity += (1.0 - self.beta1) * dw

        dw_max = buffer["dw_max"]
        dw_max = np.maximum(self.beta2 * dw_max, np.abs(dw) + self.eps)

        velocity_hat = velocity / (1.0 - (self.beta1 ** t))

        return self.lr * velocity_hat / dw_max