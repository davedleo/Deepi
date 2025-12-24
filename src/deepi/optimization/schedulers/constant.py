from deepi.optimization.optimizers import Optimizer 
from deepi.optimization.schedulers import Scheduler 


class Constant(Scheduler): 

    def __init__(
            self, 
            optimizer: Optimizer, 
            factor: 0.33, 
            milestone: 5
    ): 
        super().__init__(optimizer, "constant")
        self.factor = factor 
        self.milestone = milestone 
        self.t = 0 

    def update(self): 
        self.t += 1 
        return self.factor * self.lr if self.t < self.milestone else self.lr