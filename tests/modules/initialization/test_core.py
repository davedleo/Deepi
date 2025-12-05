import numpy as np
import pytest

from deepi.modules.initialization.core import (
    Uniform,
    Normal,
    XavierUniform,
    XavierNormal,
    KaimingUniform,
    KaimingNormal,
    Orthogonal,
)
from deepi.modules import Module


# Dummy module to test __call__
class DummyModule(Module):

    def __init__(self, shapes):
        super().__init__("dummy", True)
        # Create parameter tensors matching given shapes
        for i, s in enumerate(shapes):
            self.params[f"w{i}"] = s

    def forward(self, x):
        return x
    def backward(self, dy):
        return dy


@pytest.fixture
def shape_2d():
    return (4, 6)     # fan_in=4, fan_out=6


@pytest.fixture
def shape_4d():
    return (8, 4, 3, 3)   # conv kernel format


#
# Basic initializers
#

def test_uniform(shape_2d):
    ini = Uniform(low=-2.0, high=5.0)
    x = ini.initialize(shape_2d)

    assert isinstance(x, np.ndarray)
    assert x.shape == shape_2d
    assert (x >= -2.0).all()
    assert (x <= 5.0).all()


def test_normal(shape_2d):
    ini = Normal(mean=3.0, std=2.0)
    x = ini.initialize(shape_2d)

    assert isinstance(x, np.ndarray)
    assert x.shape == shape_2d
    # Centre of mass check
    assert abs(np.mean(x) - 3.0) < 1.0


#
# Xavier family
#

def test_xavier_uniform(shape_2d):
    fan_in, fan_out = shape_2d
    gain = 2.0
    ini = XavierUniform(gain=gain)
    x = ini.initialize(shape_2d)

    assert x.shape == shape_2d

    r = gain * np.sqrt(6.0 / (fan_in + fan_out))
    assert x.max() <= r + 1e-6
    assert x.min() >= -r - 1e-6


def test_xavier_normal(shape_2d):
    fan_in, fan_out = shape_2d
    gain = 2.0
    ini = XavierNormal(gain=gain)
    x = ini.initialize(shape_2d)

    assert x.shape == shape_2d

    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    assert abs(np.std(x) - std) < std  # distribution variability allowed


#
# Kaiming family
#

def test_kaiming_uniform_in(shape_2d):
    fan_in, fan_out = shape_2d
    gain = 0.5
    ini = KaimingUniform(fan_mode="in", gain=gain)
    x = ini.initialize(shape_2d)

    assert x.shape == shape_2d

    bound = np.sqrt(6.0 / fan_in) * gain
    assert x.max() <= bound + 1e-6
    assert x.min() >= -bound - 1e-6


def test_kaiming_uniform_out(shape_2d):
    fan_in, fan_out = shape_2d
    gain = 0.5
    ini = KaimingUniform(fan_mode="out", gain=gain)
    x = ini.initialize(shape_2d)

    assert x.shape == shape_2d

    bound = np.sqrt(6.0 / fan_out) * gain
    assert x.max() <= bound + 1e-6
    assert x.min() >= -bound - 1e-6


def test_kaiming_normal(shape_2d):
    fan_in, fan_out = shape_2d
    gain = 0.3
    ini = KaimingNormal(fan_mode="in", gain=gain)
    x = ini.initialize(shape_2d)

    assert x.shape == shape_2d

    expected_std = np.sqrt(2.0 / fan_in) * gain
    assert abs(np.std(x) - expected_std) < expected_std


#
# Orthogonal initializer
#

def test_orthogonal_square_matrix():
    ini = Orthogonal(gain=2.0)
    x = ini.initialize((4, 4))

    assert x.shape == (4, 4)

    # Check orthogonality: QᵀQ = g²I
    XT = x.T @ x
    assert np.allclose(XT, 4.0 * np.eye(4), atol=1e-5)


def test_orthogonal_rectangular():
    ini = Orthogonal(gain=1.5)
    x = ini.initialize((6, 3))

    assert x.shape == (6, 3)

    XT = x.T @ x
    assert np.allclose(XT, (1.5 ** 2) * np.eye(3), atol=1e-5)


def test_orthogonal_error():
    ini = Orthogonal()
    with pytest.raises(ValueError):
        ini.initialize((5,))


#
# initializer __call__ test
#

def test_initializer_call_updates_parameters():
    mod = DummyModule([(4, 6), (3, 3), (5,)])
    ini = Uniform(low=-1.0, high=1.0)

    ini(mod)
    params = mod.get_params()

    for name, v in params.items():
        assert isinstance(v, np.ndarray)
        assert (v >= -1.0).all()
        assert (v <= 1.0).all()