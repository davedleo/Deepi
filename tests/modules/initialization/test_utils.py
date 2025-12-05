import pytest
from math import sqrt

from deepi.modules.initialization.utils import get_gain


def test_gain_linear_identity_convolution():
    assert get_gain("linear") == 1.0
    assert get_gain("identity") == 1.0
    assert get_gain("convolution") == 1.0


def test_gain_sigmoid():
    assert get_gain("sigmoid") == 1.0


def test_gain_tanh():
    expected = 5.0 / 3.0
    assert get_gain("tanh") == expected


def test_gain_relu():
    expected = sqrt(2.0)
    assert get_gain("relu") == expected


def test_gain_leaky_relu_default_slope():
    # default slope=0.01:
    expected = sqrt(2.0 / (1 + 0.01 ** 2))
    assert get_gain("leaky_relu") == expected


def test_gain_leaky_relu_custom_slope():
    slope = 0.2
    expected = sqrt(2.0 / (1 + slope ** 2))
    assert get_gain("leaky_relu", negative_slope=slope) == expected


def test_gain_selu():
    expected = 3.0 / 4.0
    assert get_gain("selu") == expected


def test_gain_invalid():
    with pytest.raises(ValueError):
        get_gain("softmax")

    with pytest.raises(ValueError):
        get_gain("unknown")


def test_gain_invalid_type_message():
    """
    Ensure the error mentions supported types explicitly.
    """
    with pytest.raises(ValueError) as exc:
        get_gain("random")

    msg = str(exc.value)
    assert "Unsupported gain type" in msg
    assert "linear" in msg  # known valid type appears in the message