from typing import Dict, Optional 

import numpy as np

from deepi import Model
from deepi.optimization.optimizers import Optimizer
from deepi.optimization.regularization import Regularizer


class Rprop(Optimizer): 

    def __init__(
            self, 
            model: Model,
            lr: float = 0.01,
            eta_minus: float = 0.5,
            eta_plus: float = 1.2,
            step_min: float = 1e-6,
            step_max: float = 50.0,
            maximize: bool = False
    ): 
        super().__init__(model, lr, None, False, maximize, True, "r_prop")    
        self.eta_minus = eta_minus 
        self.eta_plus = eta_plus 
        self.step_min = step_min 
        self.step_max = step_max 

        for module_id, module in self.modules.items(): 
            for k in self.buffer[module_id].keys(): 
                self.buffer[module_id][k]["dw"] = np.zeros_like(module.params[k])
                self.buffer[module_id][k]["eta"] = self.lr * np.ones_like(module.params[k])

    def direction(
            self,
            dw: np.ndarray,
            buffer: Optional[Dict[str, np.ndarray]]
    ) -> np.ndarray:
        dw_prev = buffer["dw"]
        eta = buffer["eta"]
        dw_prod = dw * dw_prev

        pos_mask = dw_prod > 0.0
        eta[pos_mask] = np.minimum(eta[pos_mask] * self.eta_plus, self.step_max)

        neg_mask = dw_prod < 0.0
        eta[neg_mask] = np.maximum(eta[neg_mask] * self.eta_minus, self.step_min)
        dw[neg_mask] = 0.0

        direction = np.sign(dw) * eta 
        buffer["dw"] = dw

        return direction 