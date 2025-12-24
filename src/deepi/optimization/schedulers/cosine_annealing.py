from math import cos, pi

from deepi.optimization.optimizers import Optimizer 
from deepi.optimization.schedulers import Scheduler 


class CosineAnnealing(Scheduler): 

    def __init__(
            self, 
            optimizer: Optimizer, 
            lr_min: float = 0.0, 
            T_max: int = 5
    ): 
        super().__init__(optimizer, "cosine_annealing")
        self.lr_min = lr_min 
        self.T_max = T_max 

    def update(self) -> float: 
        lr = self.optimizer.get_lr()
        lr_new = self.lr_min + 0.5 * (lr - self.lr_min) * (1.0 + cos(pi * (self.t - 1) / self.T_max))
        return  lr_new