from abc import ABC, abstractmethod 
from deepi.optimization.optimizers import Optimizer


class Scheduler(ABC): 

    def __init__(
            self, 
            optimizer: Optimizer,
            _type: str
    ): 
        self._type = f"scheduler.{_type}"
        self.optimizer = optimizer 
    
    @abstractmethod
    def update(self, lr: float): 
        pass

    def step(self): 
        lr = self.optimizer.get_lr()
        lr_new = self.update(lr)
        self.optimizer.load_lr(lr_new)
