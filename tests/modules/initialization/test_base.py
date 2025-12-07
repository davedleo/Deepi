import numpy as np
import pytest

from deepi.modules.base import Module
from deepi.modules.initialization.base import Initializer


# --------------------------------------------------------------------------
# Dummy Initializer
# --------------------------------------------------------------------------

class DummyInit(Initializer):
    """A simple initializer that returns all ones with the requested shape."""

    def rule(self, shape):
        return np.ones(shape)


# --------------------------------------------------------------------------
# Dummy Module with parameters
# --------------------------------------------------------------------------

class DummyParam(Module):
    def __init__(self, shape=(3,)):
        super().__init__("dummy.param", _has_params=True)
        # Initialize parameter arrays (actual NumPy arrays)
        self.params["w"] = shape
        self.grads["w"] = shape

    def get_params(self):
        # Return the actual arrays (so init can overwrite them)
        return self.params

    def transform(self, x):
        return x + self.params["w"]

    def gradients(self, dy):
        self.grads["w"] = np.sum(dy, axis=0)
        return dy


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_initializer_rule_returns_correct_array():
    init = DummyInit("dummy")
    shape = (2, 3)
    arr = init.rule(shape)
    assert arr.shape == shape
    assert np.all(arr == 1.0)


def test_initializer_init_sets_module_params():
    init = DummyInit("dummy")
    m = DummyParam(shape=(2, 2))
    init.init(m)
    assert np.all(m.params["w"] == 1.0)


def test_initializer_str_and_repr():
    init = DummyInit("dummy")
    s = str(init)
    r = repr(init)
    assert s == "Initializer.Dummy"
    assert r == "Initializer.Dummy"
    assert s == r


def test_initializer_with_different_shapes():
    init = DummyInit("dummy")
    shapes = [(1,), (2, 3), (4, 5, 6)]
    for s in shapes:
        arr = init.rule(s)
        assert arr.shape == s
        assert np.all(arr == 1.0)