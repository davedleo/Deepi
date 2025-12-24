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
        self.t = 0 

    def update(self, lr: float) -> float: 
        self.t += 1 
        gamma = self.lmbd(self.t)
        return gamma * lr