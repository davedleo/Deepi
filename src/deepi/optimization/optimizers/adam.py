from typing import Dict, Optional
import numpy as np 

from deepi import Model 
from deepi.optimization.optimizers import Optimizer 
from deepi.optimization.regularization import Regularizer 


class Adam(Optimizer): 

    def __init__(
            self,
            model: Model, 
            lr: float = 0.001,
            beta1: float = 0.9,
            beta2: float = 0.999,
            amsgrad: bool = False,
            decoupled_regularization: bool = False,
            eps: float = 1e-8,
            regularizer: Optional[Regularizer] = None,
            maximize: bool = False
    ): 
        super().__init__(model, lr, regularizer, decoupled_regularization, maximize, "adam")
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.amsgrad = amsgrad
        self.eps = eps

        for module_id, module in self.modules.items(): 
            for k, buffer in self.buffer[module_id].items(): 
                buffer["t"] = 0
                buffer["velocity"] = np.zeros_like(module.params[k])
                buffer["square_avg"] = np.zeros_like(module.params[k])
                if self.amsgrad: 
                    buffer["square_avg_max"] = np.zeros_like(module.params[k])

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

        square_avg = buffer["square_avg"]
        square_avg *= self.beta2 
        square_avg += (1.0 - self.beta2) * (dw ** 2)

        velocity_hat = velocity / (1.0 - (self.beta1 ** t))
        
        if self.amsgrad: 
            square_avg_max = buffer["square_avg_max"]
            square_avg_max = np.maximum(square_avg, square_avg_max)
            buffer["square_avg_max"] = square_avg_max
            square_avg_hat = square_avg_max / (1.0 - (self.beta2 ** t))
        else: 
            square_avg_hat = square_avg / (1.0 - (self.beta2 ** t))

        return self.lr * velocity_hat / (np.sqrt(square_avg_hat) + self.eps)