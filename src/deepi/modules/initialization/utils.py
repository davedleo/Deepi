from math import sqrt
from typing import Optional

_GAIN_IDS = {
    "linear",
    "identity",
    "convolution",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "selu"
}


def get_gain(gain: str, negative_slope: Optional[float] = None) -> float:
    if gain not in _GAIN_IDS:
        raise ValueError(f"Unsupported gain type '{gain}'. Must be one of {_GAIN_IDS}.")

    if gain in ("linear", "identity", "convolution"):
        return 1.0
    elif gain == "sigmoid":
        return 1.0
    elif gain == "tanh":
        return 5.0 / 3
    elif gain == "relu":
        return sqrt(2.0)
    elif gain == "leaky_relu":
        if negative_slope is None:
            negative_slope = 0.01
        return sqrt(2.0 / (1 + negative_slope ** 2))
    elif gain == "selu":
        return 3.0 / 4
    else:
        raise ValueError(f"Unsupported gain type '{gain}'.")
        