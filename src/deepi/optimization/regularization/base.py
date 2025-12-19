from abc import ABC, abstractmethod 


class Regularizer(ABC): 

    def __init__(
            self, 
            _type: str
    ): 
        self._type = f"regularizer.{_type}"
        return