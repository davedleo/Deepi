import numpy as np
from deepi.modules import Module
from deepi.modules.initialization.normal import Normal


class DummyParam(Module):
    _has_params = True

    def __init__(self, shape):
        super().__init__("dummy")
        self.shape = shape
        self.params = {'weight': np.empty(shape)}
        self.grads = {'weight': np.empty(shape)}

    def get_params(self):
        return self.params

    def forward(self, params):
        self.params = params

    def gradients(self):
        return self.grads


def test_normal_rule_returns_correct_shape_and_range():
    mean = 0.0
    std = 1.0
    normal_init = Normal(mean=mean, std=std)
    shape = (1000, 1000)
    result = normal_init.init(shape)
    assert isinstance(result, np.ndarray)
    assert result.shape == shape
    assert abs(result.mean() - mean) < 0.01
    assert abs(result.std() - std) < 0.01


def test_normal_init_replaces_shapes_with_arrays():
    normal_init = Normal(mean=0.0, std=1.0)
    dummy = DummyParam((3, 3))
    normal_init(dummy)
    for param in dummy.get_params().values():
        assert isinstance(param, np.ndarray)
        assert param.shape == dummy.shape


def test_normal_str_and_repr():
    normal_init = Normal(mean=0.5, std=2.0)
    s = str(normal_init)
    r = repr(normal_init)
    assert "Initializer.Normal" in s
    assert "Initializer.Normal" in r
