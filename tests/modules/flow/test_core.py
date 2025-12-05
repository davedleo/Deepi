import pytest
import numpy as np
from deepi.modules.flow import Input, Flatten, Reshape, Concatenate


@pytest.fixture
def x():
    return np.arange(24, dtype=float).reshape(2, 3, 4)


@pytest.fixture
def dy():
    return 0.1 * np.arange(24, dtype=float).reshape(2, 3, 4)


def test_input_forward_backward(x, dy):
    # Test Input without storing gradient
    inp = Input(in_shape=x.shape)
    out = inp.forward(x)
    assert isinstance(out, np.ndarray)
    assert np.all(out == x)
    assert inp.cache is None

    # Test Input with store_gradient=True
    inp_grad = Input(in_shape=x.shape, store_gradient=True)
    inp_grad.forward(x)
    inp_grad.backward(dy)
    assert np.all(inp_grad.cache == dy)


def test_input_forward_default_empty():
    shape = (2, 3, 4)
    inp = Input(in_shape=shape)
    out = inp.forward(None)
    assert out.shape == shape
    assert np.all(out == np.empty(shape)) or out.shape == shape


def test_flatten_forward_backward(x, dy):
    flat = Flatten()
    flat.train()
    out = flat.forward(x)
    assert out.shape == (x.shape[0], np.prod(x.shape[1:]))
    assert flat.cache == x.shape

    dx = flat.backward(dy.reshape(out.shape))
    assert dx.shape == x.shape


def test_reshape_forward_backward(x, dy):
    new_shape = (x.shape[0], 12)
    resh = Reshape(out_shape=(12,))
    resh.train()
    out = resh.forward(x)
    assert out.shape == (x.shape[0], 12)
    assert resh.cache == x.shape

    dx = resh.backward(out)
    assert dx.shape == x.shape


def test_concatenate_forward_backward():
    a = np.arange(6, dtype=float).reshape(2, 3)
    b = np.arange(6, 12, dtype=float).reshape(2, 3)
    concat = Concatenate(axis=1)
    concat.train()
    out = concat.forward((a, b))
    assert out.shape == (2, 6)
    assert concat.cache == [a.shape[1], b.shape[1]]

    dy = np.ones_like(out)
    dx_tuple = concat.backward(dy)
    assert isinstance(dx_tuple, tuple)
    assert len(dx_tuple) == 2
    assert dx_tuple[0].shape == a.shape
    assert dx_tuple[1].shape == b.shape
    assert np.all(dx_tuple[0] == 1.0)
    assert np.all(dx_tuple[1] == 1.0)