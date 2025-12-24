from deepi.optimization.optimizers import Optimizer 
from deepi.optimization.schedulers import Scheduler 


class Exponential(Scheduler): 

    def __init__(
            self, 
            optimizer: Optimizer, 
            factor: float = 0.95
    ): 
        super().__init__(optimizer, "exponential")
        self.factor = factor 

    def update(self) -> float: 
        return self.lr * (self.factor ** (self.t - 1))