import numpy as np 
from deepi.optimization.regularization import Regularizer 


class L1(Regularizer): 

    def __init__(
            self, 
            gamma: float = 0.001
    ): 
        super().__init__(gamma, "l1")

    def regularization(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x, dtype=np.float64)