from typing import Dict, Optional
import numpy as np 

from deepi import Model 
from deepi.optimization.optimizers import Optimizer 
from deepi.optimization.regularization import Regularizer 


class NAdam(Optimizer): 

    def __init__(
            self,
            model: Model, 
            lr: float = 0.002,
            beta1: float = 0.9,
            beta2: float = 0.999,
            momentum_decay: float = 0.004,
            eps: float = 1e-8,
            decoupled_regularization: bool = False,
            regularizer: Optional[Regularizer] = None,
            maximize: bool = False
    ): 
        super().__init__(model, lr, regularizer, decoupled_regularization, maximize, "n_adam")
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.psi = momentum_decay
        self.eps = eps

        for module_id, module in self.modules.items(): 
            for k, buffer in self.buffer[module_id].items(): 
                buffer["t"] = 0
                buffer["velocity"] = np.zeros_like(module.params[k])
                buffer["square_avg"] = np.zeros_like(module.params[k])
                buffer["nesterov_momentum"] = self.beta1 * (1.0 - 0.5 * (0.96 ** self.psi))
                buffer["nesterov_momentum_prod"] = self.beta1 * (1.0 - 0.5 * (0.96 ** self.psi))

    def direction(
            self, 
            dw: np.ndarray,
            buffer: Dict[str, np.ndarray]
    ) -> np.ndarray: 
        buffer["t"] += 1
        t = buffer["t"]

        nesterov_momentum = buffer["nesterov_momentum"]
        nesterov_momentum_prod = buffer["nesterov_momentum_prod"]
        nesterov_momentum_next = self.beta1 * (1.0 - 0.5 * (0.96 ** ((t + 1) * self.psi)))
        nesterov_momentum_prod_next = nesterov_momentum_prod * nesterov_momentum_next

        velocity = buffer["velocity"]
        velocity *= self.beta1 
        velocity += (1.0 - self.beta1) * dw 
        
        velocity_hat_term1 = nesterov_momentum_next * velocity / (1.0 - nesterov_momentum_prod_next)
        velocity_hat_term2 = (1.0 - nesterov_momentum) * dw / (1.0 - nesterov_momentum_prod)
        velocity_hat = velocity_hat_term1 + velocity_hat_term2

        buffer["nesterov_momentum"] = nesterov_momentum_next
        buffer["nesterov_momentum_prod"] = nesterov_momentum_prod_next

        square_avg = buffer["square_avg"]
        square_avg *= self.beta2 
        square_avg += (1.0 - self.beta2) * (dw ** 2)
        square_avg_hat = square_avg / (1.0 - (self.beta2 ** t))

        return self.lr * velocity_hat / (np.sqrt(square_avg_hat) + self.eps)