import numpy as np 
from deepi.optimization.regularization import Regularizer 


class L2(Regularizer): 

    def __init__(
            self, 
            gamma: float = 0.001
    ): 
        super().__init__(gamma, "l2")

    def regularization(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * x