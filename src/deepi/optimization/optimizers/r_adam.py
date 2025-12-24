from typing import Dict, Optional
import numpy as np 

from deepi import Model 
from deepi.optimization.optimizers import Optimizer 
from deepi.optimization.regularization import Regularizer 


class RAdam(Optimizer): 

    def __init__(
            self,
            model: Model, 
            lr: float = 0.001,
            beta1: float = 0.9,
            beta2: float = 0.999,
            eps: float = 1e-8,
            decoupled_regularization: bool = False,
            regularizer: Optional[Regularizer] = None,
            maximize: bool = False
    ): 
        super().__init__(model, lr, regularizer, decoupled_regularization, maximize, "r_adam")
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.eps = eps

        for module_id, module in self.modules.items(): 
            for k, buffer in self.buffer[module_id].items(): 
                buffer["t"] = 0
                buffer["velocity"] = np.zeros_like(module.params[k])
                buffer["square_avg"] = np.zeros_like(module.params[k])
                buffer["rho_inf"] = 2.0 / (1.0 - self.beta2) - 1.0

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
        
        rho_inf = buffer["rho_inf"]
        rho_t = rho_inf - 2.0 * t * (self.beta2 ** t) / (1.0 - (self.beta2 ** t))

        if rho_t > 5.0: 
            l_t = np.sqrt(1.0 - (self.beta2 ** t)) / (np.sqrt(square_avg) + self.eps)
            r_t = np.sqrt((rho_t ** 2 - 6 * rho_t + 8) * rho_inf) / np.sqrt((rho_inf ** 2 - 6 * rho_inf + 8) * rho_t)
            return self.lr * velocity_hat * r_t * l_t 
        else: 
            return self.lr * velocity_hat