import numpy as np
import pytest

from deepi.modules.base import Module, ArrayOrTuple


# --------------------------------------------------------------------------
# Helper dummy modules
# --------------------------------------------------------------------------

class DummyUnary(Module):
    """
    A module that applies y = 2*x
    And backward: dx = 2*dy
    """

    def __init__(self):
        super().__init__("dummy.unary")

    def transform(self, x: ArrayOrTuple):
        return 2 * x

    def gradients(self, dy: ArrayOrTuple):
        return 2 * dy


class DummyBinary(Module):
    """
    A module that takes two inputs (x1, x2) and returns y = x1 + x2
    And backward propagates dx1 = dy, dx2 = dy
    """

    def __init__(self):
        super().__init__("dummy.binary")

    def transform(self, xs: ArrayOrTuple):
        x1, x2 = xs
        return x1 + x2

    def gradients(self, dy: ArrayOrTuple):
        return (dy, dy)


class DummyParam(Module):
    """
    A module with parameters: y = x + w
    where w is stored in params['w']
    backward: dx = dy, dw = sum(dy)
    """

    def __init__(self, shape=(3,)):
        super().__init__("dummy.param", _has_params=True)
        self.params["w"] = np.ones(shape)
        self.grads["w"] = np.zeros(shape)

    def transform(self, x: ArrayOrTuple):
        return x + self.params["w"]

    def gradients(self, dy: ArrayOrTuple):
        grads = {"w": dy.sum(axis=0)}
        return dy, grads


class Identity(Module):
    """
    Simply forwards input unchanged.
    """

    def __init__(self):
        super().__init__("identity")

    def transform(self, x):
        return x

    def gradients(self, dy):
        return dy


# --------------------------------------------------------------------------
# Tests Below
# --------------------------------------------------------------------------

def test_forward_without_training():
    m = DummyUnary()
    x = np.array([1., 2., 3.])

    y = m.forward(x)

    assert np.allclose(y, np.array([2., 4., 6.]))
    assert m.x is None
    assert m.y is None


def test_forward_with_training_caching():
    m = DummyUnary()
    m.train()

    x = np.array([1., 3.])
    y = m.forward(x)

    assert np.allclose(y, np.array([2., 6.]))
    assert np.allclose(m.x, x)
    assert np.allclose(m.y, y)


def test_backward_local_computation():
    m = DummyUnary()
    m.train()
    x = np.array([1., 2., 3.])
    y = m.forward(x)

    dy = np.array([10., 11., 12.])
    m.backward(dy)

    assert np.allclose(m.dy, dy)
    assert np.allclose(m.gradients(dy), 2 * dy)


def test_backward_accumulation_single_input():
    m = DummyUnary()

    dy1 = np.array([1., 2., 3.])
    dy2 = np.array([2., 3., 4.])

    dy1_c = dy1.copy()  # verify no mutation occurs
    dy2_c = dy2.copy()

    m.backward(dy1)
    m.backward(dy2)

    assert np.allclose(dy1, dy1_c)
    assert np.allclose(dy2, dy2_c)
    assert np.allclose(m.dy, dy1 + dy2)


def test_backward_with_tuple_gradients():
    """
    Validate propagation through binary splitting and unary parents.
    """

    x1 = np.array([1., 1., 1.])
    x2 = np.array([2., 2., 2.])
    dy = np.array([10., 20., 30.])

    # Graph
    p1 = DummyUnary()
    p2 = DummyUnary()
    b = DummyBinary()
    p1.link(b)
    p2.link(b)

    # simulate forward
    f1 = p1.forward(x1)  # 2*x1
    f2 = p2.forward(x2)  # 2*x2

    y = b.forward((f1, f2))
    assert np.allclose(y, (2 * x1) + (2 * x2))

    # backward
    b.backward(dy)

    # each unary has dy split * 2 (local gradient)
    assert np.allclose(p1.dy, dy)
    assert np.allclose(p2.dy, dy)


def test_backward_fan_out_accumulation():
    """
    Graph:
         A(identity)
         |
        U1(unary)
       /       \
    U2(unary)  U3(unary)

    Gradients propagate back twice into U1
    """

    A = Identity()
    U1 = DummyUnary()
    U2 = DummyUnary()
    U3 = DummyUnary()

    A.train()
    U1.train()
    U2.train()
    U3.train()

    A.link(U1)
    U1.link(U2)
    U1.link(U3)

    x = np.array([1., 1.])
    U1.forward(A.forward(x))
    U2.forward(U1.y)
    U3.forward(U1.y)

    dy_U2 = np.array([2., 3.])
    dy_U3 = np.array([4., 1.])

    U2.backward(dy_U2)
    U3.backward(dy_U3)

    # U1 sees dy_U2 and dy_U3, backward multiplies both by 2
    expected = 2 * dy_U2 + 2 * dy_U3
    assert np.allclose(U1.dy, expected)


def test_params_collected_and_set():
    m = DummyParam()
    assert "w" in m.get_params()

    new_params = {"w": np.array([10., 10., 10.])}
    m.load_params(new_params)
    assert np.allclose(m.params["w"], new_params["w"])


def test_parameter_gradient_update():
    m = DummyParam(shape=(3,))
    x = np.array([[1., 1., 1.],
                  [2., 2., 2.]])
    y = m.forward(x)

    dy = np.array([[10., 10., 10.],
                   [5., 5., 5.]])
    m.backward(dy)

    assert np.allclose(m.grads["w"], np.array([15., 15., 15.]))


def test_train_and_eval_modes():
    m = DummyUnary()
    assert m._is_training is False

    m.train()
    assert m._is_training is True

    m.eval()
    assert m._is_training is False


def test_clear_resets_all():
    m = DummyUnary()
    m.train()

    x = np.array([1., 2.])
    m.forward(x)
    m.backward(np.array([10., 20.]))

    m.clear()

    assert m.x is None
    assert m.y is None
    assert m.dy is None
    assert len(m.grads) == 0


def test_string_and_repr():
    m = DummyUnary()
    assert str(m) == "Module.Dummy.Unary"
    assert repr(m) == "Module.Dummy.Unary"