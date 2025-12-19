import numpy as np
from deepi.modules import Module
from deepi.modules.initialization.orthogonal import Orthogonal


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


def test_orthogonal_rule_returns_correct_shape_and_gain():
    gain = 2.0
    ortho = Orthogonal(gain=gain)
    shape = (100, 100)
    result = ortho.init(shape)
    
    # Forma e tipo
    assert isinstance(result, np.ndarray)
    assert result.shape == shape

    # Verifica range del gain
    max_abs = np.max(np.abs(result))
    assert max_abs <= gain + 1e-12  # piccolo margine numerico


def test_orthogonal_rule_is_orthogonal_2d():
    gain = 1.0
    ortho = Orthogonal(gain=gain)
    shape = (50, 50)
    result = ortho.init(shape)
    
    # Verifica ortogonalità
    q = result.reshape(shape)
    identity = np.eye(shape[0])
    qtq = np.dot(q.T, q)
    assert np.allclose(qtq, identity, atol=1e-6)


def test_orthogonal_init_replaces_shapes_with_arrays():
    ortho = Orthogonal(gain=1.5)
    dummy = DummyParam((3, 3))
    ortho(dummy)
    for param in dummy.get_params().values():
        assert isinstance(param, np.ndarray)
        assert param.shape == dummy.shape
        # Controllo del gain massimo
        assert np.max(np.abs(param)) <= 1.5 + 1e-12


def test_orthogonal_str_and_repr():
    ortho = Orthogonal(gain=1.0)
    s = str(ortho)
    r = repr(ortho)
    assert "Initializer.Orthogonal" in s
    assert "Initializer.Orthogonal" in r