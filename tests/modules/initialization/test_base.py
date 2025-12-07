import typing


class Initializer:
    def __init__(self, name: str):
        self.name = name

    def rule(self, shape: typing.Tuple[int, ...]) -> typing.Any:
        raise NotImplementedError

    def init(self, module):
        params = module.get_params()
        for key, val in params.items():
            if isinstance(val, tuple):
                params[key] = self.rule(val)
            else:
                params[key] = self.rule(val.shape)

    def fan_in(self, shape: typing.Tuple[int, ...]) -> int:
        if len(shape) == 2:
            # Linear layer: (in_features, out_features)
            return shape[0]
        elif len(shape) == 3:
            # Conv1D: (out_channels, in_channels, kernel_size)
            return shape[1] * shape[2]
        elif len(shape) == 4:
            # Conv2D: (out_channels, in_channels, kernel_height, kernel_width)
            return shape[1] * shape[2] * shape[3]
        else:
            raise ValueError(f"Unsupported shape for fan_in: {shape}")

    def fan_out(self, shape: typing.Tuple[int, ...]) -> int:
        if len(shape) == 2:
            # Linear layer: (in_features, out_features)
            return shape[1]
        elif len(shape) == 3:
            # Conv1D: (out_channels, in_channels, kernel_size)
            return shape[0] * shape[2]
        elif len(shape) == 4:
            # Conv2D: (out_channels, in_channels, kernel_height, kernel_width)
            return shape[0] * shape[2] * shape[3]
        else:
            raise ValueError(f"Unsupported shape for fan_out: {shape}")

    def __str__(self) -> str:
        return f"Initializer.{self.__class__.__name__}"

    __repr__ = __str__