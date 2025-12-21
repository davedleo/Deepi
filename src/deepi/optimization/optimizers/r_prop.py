from typing import Dict
import numpy as np 

from deepi import Model
from deepi.optimization.optimizers import Optimizer


class Rprop(Optimizer): 

    def __init__(
            self, 
            model: Model,
            lr: float = 0.01,
            eta_minus: float = 0.5,
            eta_plus: float = 1.2,
            min_step: float = 1e-6,
            max_step: float = 50.0,
            maximize: bool = False
    ): 
        super().__init__(model, lr, None, False, maximize, "r_prop")
        self.eta_minus = eta_minus 
        self.eta_plus = eta_plus 
        self.min_step = min_step 
        self.max_step = max_step

        for module_id, module in self.modules.items(): 
            for k, buffer in self.buffer["params"][module_id].items(): 
                buffer["dw_prev"] = np.zeros_like(module.params[k])
                buffer["eta"] = np.full_like(module.params[k], lr)

    def direction(
            self, 
            dw: np.ndarray,
            buffer: Dict[str, np.ndarray]
    ) -> np.ndarray: 
        dw_prod = dw * buffer["dw_prev"] 
        buffer["dw_prev"] = dw.copy()
        
        pos_mask = dw_prod > 0.0 
        neg_mask = dw_prod < 0.0

        eta = buffer["eta"]
        eta[pos_mask] = np.minimum(eta[pos_mask] * self.eta_plus, self.max_step)
        eta[neg_mask] = np.maximum(eta[neg_mask] * self.eta_minus, self.min_step)

        return np.sign(dw) * eta