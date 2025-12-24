from typing import Callable

from deepi.optimization.optimizers import Optimizer 
from deepi.optimization.schedulers import Scheduler 


class Multiplicative(Scheduler): 

    def __init__(
            self, 
            optimizer: Optimizer, 
            lmbd: Callable
    ): 
        super().__init__(optimizer, "multiplicative")
        self.lmbd = lmbd 

    def update(self): 
        factor = self.lmbd(self.t)
        lr = self.optimizer.get_lr()
        return factor * lr