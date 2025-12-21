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
            eps: float = 1e-10,
            regularizer: Optional[Regularizer] = None,
            maximize: bool = False
    ): 
        super().__init__(model, lr, regularizer, False, maximize, "adadelta")
        self.rho = rho
        self.eps = eps

        for module_buffer in self.buffer["params"].values(): 
            for buffer in module_buffer.values(): 
                buffer["square_dw_avg"] = None
                buffer["square_delta_avg"] = None

    def direction(
            self, 
            dw: np.ndarray,
            buffer: Dict[str, np.ndarray]
    ) -> np.ndarray: 
        if buffer["square_dw_avg"] is None: 
            ...


        return 