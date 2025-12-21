from typing import Dict, Optional 

import numpy as np

from deepi import Model
from deepi.optimization.optimizers import Optimizer
from deepi.optimization.regularization import Regularizer


class RMSprop(Optimizer): 

    def __init__(
            self, 
            model: Model,
            lr: float = 0.001,  
            momentum: float = 0.0,          
            alpha: float = 0.99,
            centered: float = True,
            eps: float = 1e-8,
            regularizer: Optional[Regularizer] = None,
            maximize: bool = False
    ): 
        super().__init__(model, lr, regularizer, False, maximize, True, "rms_prop")    
        self._has_momentum = momentum != 0.0 
        if self._has_momentum: 
            self.mu = momentum 

        self.alpha = alpha 
        self.eps = eps
        self.centered = centered

        for module_id, module in self.modules.items(): 
            for k in self.buffer[module_id].keys(): 
                if self._has_momentum: 
                    self.buffer[module_id][k]["velocity"] = np.zeros_like(module.params[k])

                if self.centered: 
                    self.buffer[module_id][k]["avg"] = np.zeros_like(module.params[k])
                
                self.buffer[module_id][k]["square_avg"] = np.zeros_like(module.params[k])

    def direction(
            self, 
            dw: np.ndarray,
            buffer: Optional[Dict[str, np.ndarray]]
    ) -> np.ndarray: 
        square_avg = buffer["square_avg"]
        square_avg *= self.alpha
        square_avg += (1.0 - self.alpha) * (dw ** 2)

        if self.centered: 
            avg = buffer["avg"]
            avg *= self.alpha
            avg += (1.0 - self.alpha) * dw
            square_avg = square_avg - (avg ** 2)

        if self._has_momentum: 
            v = buffer["velocity"]
            v *= self.mu 
            v += dw / (np.sqrt(square_avg) + self.eps)
            dw = self.lr * v 
        else: 
            dw = self.lr * dw / (np.sqrt(square_avg) + self.eps)

        return dw