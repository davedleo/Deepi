import numpy as np
import pytest

from deepi.modules import Module
from deepi.modules.initialization.base import Initializer


# --------------------------------------------------------------------------
# Dummy Initializer
# --------------------------------------------------------------------------

class DummyInit(Initializer):
    """Simple initializer that returns ones for a given shape."""

    def rule(self, shape):
        return np.ones(shape)


# --------------------------------------------------------------------------
# Dummy Module with parameters
# --------------------------------------------------------------------------

class DummyParam(Module):
    def __init__(self, shape=(3,)):
        super().__init__("dummy.param", _has_params=True)
        self.params = {"w": shape}  # shape tuple before init
        self.grads = {"w": shape}

    def get_params(self):
        return self.params

    def transform(self, x):
        return x + self.params["w"]

    def gradients(self, dy):
        self.grads["w"] = np.sum(dy, axis=0)
        return dy


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_rule_returns_array_of_ones():
    init = DummyInit("dummy")
    shape = (2, 3)
    arr = init.rule(shape)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == shape
    assert np.all(arr == 1.0)


def test_init_replaces_shapes_with_arrays():
    init = DummyInit("dummy")
    m = DummyParam(shape=(2, 2))
    # Before init, params are tuples
    assert m.params["w"] == (2, 2)
    init.init(m)
    # After init, params are arrays of ones
    assert isinstance(m.params["w"], np.ndarray)
    assert m.params["w"].shape == (2, 2)
    assert np.all(m.params["w"] == 1.0)


def test_str_and_repr():
    init = DummyInit("dummy")
    s = str(init)
    r = repr(init)
    assert s == "Initializer.Dummy"
    assert r == "Initializer.Dummy"
    assert s == r


def test_fan_in_and_fan_out_linear():
    init = DummyInit("dummy")
    shape = (5, 10)
    assert init.fan_in(shape) == 5
    assert init.fan_out(shape) == 10


def test_fan_in_and_fan_out_conv1d():
    init = DummyInit("dummy")
    shape = (16, 3, 7)
    assert init.fan_in(shape) == 3 * 7
    assert init.fan_out(shape) == 16 * 7


def test_fan_in_and_fan_out_conv2d():
    init = DummyInit("dummy")
    shape = (32, 3, 5, 5)
    assert init.fan_in(shape) == 3 * 5 * 5
    assert init.fan_out(shape) == 32 * 5 * 5


def test_init_no_params_does_nothing():
    class NoParam(Module):
        def __init__(self):
            super().__init__("noparam")
            self.params = {}
            self.grads = {}

        def get_params(self):
            return self.params

        def transform(self, x):
            return x

        def gradients(self, dy):
            return dy

    init = DummyInit("dummy")
    m = NoParam()
    init.init(m)
    assert m.params == {}