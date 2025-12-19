from abc import ABC, abstractmethod 


class Scheduler(ABC): 

    def __init__(
            self, 
            _type: str
    ): 
        self._type = f"scheduler.{_type}"
        return