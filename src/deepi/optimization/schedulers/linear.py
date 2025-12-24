import numpy as np

from deepi.optimization.optimizers import Optimizer 
from deepi.optimization.schedulers import Scheduler 


class Linear(Scheduler): 

    def __init__(
            self, 
            optimizer: Optimizer, 
            start_factor: float = 0.33, 
            end_factor: float = 1.0,
            milestone: int = 5
    ): 
        super().__init__(optimizer, "linear")
        self.factors = np.linspace(start_factor, end_factor, milestone).tolist()
        self.milestone = milestone 

    def update(self) -> float: 
        if self.t > self.milestone: 
            return self.optimizer.get_lr()
        return self.lr * self.factors[self.t - 1]