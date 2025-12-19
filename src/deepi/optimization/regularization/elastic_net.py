import numpy as np 
from deepi.optimization.regularization import Regularizer 


class ElasticNet(Regularizer): 

    def ___init__(
            self, 
            alpha: float = 0.5,
            gamma: float = 0.001
    ): 
        super().__init__(gamma, "elastic_net")
        if alpha < 0.0 or alpha > 1.0: 
            raise ValueError()
        self.alpha = alpha

    def regularization(self, x: np.ndarray) -> np.ndarray:
        l1 = np.sign(x, dtype=np.float64)
        l2 = 2.0 * x
        return (1.0 * self.alpha) * l1 + self.alpha * l2