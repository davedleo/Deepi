from abc import ABC, abstractmethod 


class Optimizer(ABC): 

    def __init__(
            self, 
            _type: str
    ): 
        self._type = f"optimizer.{_type}"
        return