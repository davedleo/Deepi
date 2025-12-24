from deepi.optimization.optimizers import Optimizer 
from deepi.optimization.schedulers import Scheduler 


class Step(Scheduler): 

    def __init__(
            self, 
            optimizer: Optimizer, 
            factor: 0.33, 
            step_size: 5
    ): 
        super().__init__(optimizer, "step")
        self.factor = factor 
        self.step_size = step_size 

    def update(self) -> float: 
        lr = self.optimizer.get_lr()
        return self.factor * lr if self.t % self.step_size == 0 else lr