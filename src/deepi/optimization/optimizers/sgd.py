from typing import Dict, Optional 

import numpy as np

from deepi import Model
from deepi.optimization.optimizers import Optimizer
from deepi.optimization.regularization import Regularizer


class SGD(Optimizer): 

    def __init__(
            self, 
            model: Model,
            lr: float = 0.001,
            momentum: float = 0.0,
            dampening: float = 0.0,
            nesterov: bool = False,
            regularizer: Optional[Regularizer] = None,
            maximize: bool = False
    ): 
        super().__init__(model, lr, regularizer, False, maximize, momentum > 0.0, "sgd")    
        self._has_momentum = momentum != 0      
        
        if self._has_momentum: 
            self.mu = momentum 
            self.tau = dampening    
            self.nesterov = nesterov 

            for module_id, module in self.modules.items(): 
                for k in self.buffer[module_id].keys(): 
                    self.buffer[module_id][k]["velocity"] = np.zeros_like(module.params[k])

    def direction(
            self, 
            dw: np.ndarray,
            buffer: Optional[Dict[str, np.ndarray]]
    ) -> np.ndarray: 
        if self._has_momentum: 
            v = buffer["velocity"]
            v *= self.mu 
            v += (1.0 - self.tau) * dw 
            dw = dw + self.mu * v if self.nesterov else v

        return dw