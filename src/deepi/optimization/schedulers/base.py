from abc import ABC, abstractmethod 
from deepi.optimization.optimizers import Optimizer 


class Scheduler(ABC): 

    def __init__(
            self,
            optimizer: Optimizer,
            _type: str
    ): 
        self.optimizer = optimizer 
        self._type = f"scheduler.{_type}"
        self.lr = self.optimizer.get_lr()

    @abstractmethod
    def update(self) -> float: 
        pass

    def step(self): 
        lr = self.update()
        self.optimizer.load_lr(lr)

    @property
    def type(self) -> str:
        return self._type

    def __str__(self) -> str:
        parts = self._type.split(".")
        return ".".join([part.capitalize() for part in parts])

    def __repr__(self) -> str:
        return self.__str__()