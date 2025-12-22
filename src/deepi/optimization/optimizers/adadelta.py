from typing import Dict, Optional
import numpy as np 

from deepi import Model 
from deepi.optimization.optimizers import Optimizer 
from deepi.optimization.regularization import Regularizer 


class Adadelta(Optimizer): 

    def __init__(
            self,
            model: Model, 
            lr: float = 1.0,
            rho: float = 0.9,
            eps: float = 1e-6,
            regularizer: Optional[Regularizer] = None,
            maximize: bool = False
    ): 
        super().__init__(model, lr, regularizer, False, maximize, "adadelta")
        self.rho = rho
        self.eps = eps

        for module_id, module in self.modules.items(): 
            for k, buffer in self.buffer[module_id].items(): 
                buffer["square_dw_avg"] = np.zeros_like(module.params[k])
                buffer["square_delta_avg"] = np.zeros_like(module.params[k])

    def direction(
            self, 
            dw: np.ndarray,
            buffer: Dict[str, np.ndarray]
    ) -> np.ndarray: 
        square_dw_avg = buffer["square_dw_avg"]
        square_delta_avg = buffer["square_delta_avg"]

        square_dw_avg *= self.rho 
        square_dw_avg += (1.0 - self.rho) * (dw ** 2)

        delta = dw * np.sqrt(square_delta_avg + self.eps) / np.sqrt(square_dw_avg + self.eps) 
        square_delta_avg *= self.rho 
        square_delta_avg += (1.0 - self.rho) * (delta ** 2) 

        return self.lr * delta