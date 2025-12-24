import numpy as np
from deepi.modules import Module
from deepi.modules.initialization.kaiming_uniform import KaimingUniform


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


def test_kaiming_uniform_rule_returns_correct_shape_and_range():
    gain = 1.0
    ku = KaimingUniform(gain=gain)
    shape = (1000, 1000)
    result = ku.init(shape)
    assert isinstance(result, np.ndarray)
    assert result.shape == shape

    fan_in = shape[0]
    r = gain * np.sqrt(3.0 / fan_in)
    assert np.all(result >= -r) and np.all(result <= r)

    # Media e std teoriche di uniforme su [-r, r]
    expected_mean = 0.0
    expected_std = np.sqrt((2 * r)**2 / 12)  # Varianza uniforme = (b-a)^2 / 12
    assert abs(np.mean(result) - expected_mean) < 0.01
    assert abs(np.std(result) - expected_std) < 0.01


def test_kaiming_uniform_init_replaces_shapes_with_arrays():
    ku = KaimingUniform()
    dummy = DummyParam((3, 3))
    ku(dummy)
    for param in dummy.get_params().values():
        assert isinstance(param, np.ndarray)
        assert param.shape == dummy.shape


def test_kaiming_uniform_str_and_repr():
    ku = KaimingUniform()
    s = str(ku)
    r = repr(ku)
    assert "Initializer.Kaiming_uniform" in s
    assert "Initializer.Kaiming_uniform" in r