from typing import Dict, Optional
import numpy as np 

from deepi import Model 
from deepi.optimization.optimizers import Optimizer 
from deepi.optimization.regularization import Regularizer 


class Adagrad(Optimizer): 

    def __init__(
            self,
            model: Model, 
            lr: float = 0.01,
            lr_decay: float = 0.0,
            square_sum_init: float = 0.0,
            eps: float = 1e-10,
            regularizer: Optional[Regularizer] = None,
            maximize: bool = False
    ): 
        super().__init__(model, lr, regularizer, False, maximize, "adagrad")
        self.lr_decay = lr_decay  
        self.square_sum_init = square_sum_init 
        self.eps = eps
        for module_id, module in self.modules.items(): 
            for k, buffer in self.buffer[module_id].items(): 
                buffer["square_sum"] = np.full_like(module.params[k], square_sum_init)
                buffer["lr"] = lr + 0.0 # Safe copy 
                buffer["t"] = 0

    def direction(
            self, 
            dw: np.ndarray,
            buffer: Dict[str, np.ndarray]
    ) -> np.ndarray: 
        buffer["t"] += 1
        buffer["lr"] = self.lr / (1.0 + self.lr_decay * (1.0 - buffer["t"]))

        square_sum = buffer["square_sum"]
        square_sum += (dw ** 2)

        dw = dw / (np.sqrt(square_sum) + self.eps)

        return buffer["lr"] * dw