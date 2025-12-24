from deepi.optimization.optimizers import Optimizer 
from deepi.optimization.schedulers import Scheduler 


class Polynomial(Scheduler): 

    def __init__(
            self, 
            optimizer: Optimizer, 
            power: float = 2, 
            milestone: int = 5
    ): 
        super().__init__(optimizer, "polynomial")
        self.power = power
        self.milestone = milestone 

    def update(self) -> float: 
        if self.t > self.milestone: 
            return self.optimizer.get_lr()
        
        factor = (1.0 - (self.t - 1) / self.milestone) ** self.power

        return factor * self.lr