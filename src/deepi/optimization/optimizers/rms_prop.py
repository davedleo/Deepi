from typing import Dict, Optional
import numpy as np 

from deepi import Model 
from deepi.optimization.optimizers import Optimizer 
from deepi.optimization.regularization import Regularizer 


class RMSprop(Optimizer): 

    def __init__(
            self,
            model: Model, 
            lr: float = 0.01,
            momentum: float = 0.0,
            alpha: float = 0.99,
            centered: bool = False,
            eps: float = 1e-10,
            regularizer: Optional[Regularizer] = None,
            maximize: bool = False
    ): 
        super().__init__(model, lr, regularizer, False, maximize, "rms_prop")
        self.mu = momentum 
        self.alpha = alpha 
        self.centered = centered 
        self.eps = eps

        for module_buffer in self.buffer["params"].values(): 
            for buffer in module_buffer.values(): 
                buffer["velocity"] = None
                buffer["avg"] = None
                buffer["square_avg"] = None

    def direction(
            self, 
            dw: np.ndarray,
            buffer: Dict[str, np.ndarray]
    ) -> np.ndarray: 
        if buffer["square_avg"] is None: 
            buffer["square_avg"] = dw ** 2 
            
            if self.centered: 
                buffer["avg"] = dw.copy()
            if self.mu != 0.0: 
                buffer["velocity"] = dw.copy()
            
            return self.lr * dw
        
        square_avg = buffer["square_avg"]
        square_avg *= self.alpha 
        square_avg += (1.0 - self.alpha) * (dw ** 2)

        if self.centered: 
            avg = buffer["avg"] 
            avg *= self.alpha 
            avg += (1.0 - self.alpha) * dw 
            square_avg = square_avg - (avg ** 2)

        if self.mu != 0.0: 
            v = buffer["velocity"]
            v *= self.mu 
            v += dw / (np.sqrt(square_avg) + self.eps)
            return self.lr * v
        
        else: 
            return self.lr * dw / (np.sqrt(square_avg) + self.eps)