import numpy as np
import pytest
from deepi.modules.linear import LowRank

def make_low_rank(input_size=3, out_size=2, rank=2, bias=True):
    m = LowRank(out_size=out_size, rank=rank, bias=bias)
    # Proper initialization for testing
    m.params["w1"] = np.arange(input_size * rank).reshape(input_size, rank).astype(np.float64)
    m.params["w2"] = np.arange(rank * out_size).reshape(rank, out_size).astype(np.float64)
    if bias:
        m.params["b"] = np.ones((1, out_size))
    m.x = None
    m.grads = {}
    return m

def test_forward_no_bias():
    m = make_low_rank(input_size=3, out_size=2, rank=2, bias=False)
    x = np.array([[1.0, 2.0, 3.0]])
    m.x = x
    y = m.transform(x)
    expected = x @ m.params["w1"] @ m.params["w2"]
    np.testing.assert_array_equal(y, expected)

def test_forward_with_bias():
    m = make_low_rank(input_size=3, out_size=2, rank=2, bias=True)
    x = np.array([[1.0, 2.0, 3.0]])
    m.x = x
    y = m.transform(x)
    expected = x @ m.params["w1"] @ m.params["w2"] + m.params["b"]
    np.testing.assert_array_equal(y, expected)

def test_backward_no_bias():
    m = make_low_rank(input_size=3, out_size=2, rank=2, bias=False)
    x = np.array([[1.0, 2.0, 3.0]])
    m.x = x
    dy = np.array([[1.0, -1.0]])
    dx, grads = m.gradients(dy)

    expected_dx = dy @ m.params["w2"].T @ m.params["w1"].T
    expected_dw1 = x.T @ dy @ m.params["w2"].T
    expected_dw2 = (x @ m.params["w1"]).T @ dy

    np.testing.assert_array_equal(dx, expected_dx)
    np.testing.assert_array_equal(grads["w1"], expected_dw1)
    np.testing.assert_array_equal(grads["w2"], expected_dw2)
    assert "b" not in grads

def test_backward_with_bias():
    m = make_low_rank(input_size=3, out_size=2, rank=2, bias=True)
    x = np.array([[1.0, 2.0, 3.0]])
    m.x = x
    dy = np.array([[1.0, -1.0]])
    dx, grads = m.gradients(dy)

    expected_dx = dy @ m.params["w2"].T @ m.params["w1"].T
    expected_dw1 = x.T @ dy @ m.params["w2"].T
    expected_dw2 = (x @ m.params["w1"]).T @ dy
    expected_db = dy.sum(axis=0, keepdims=True)

    np.testing.assert_array_equal(dx, expected_dx)
    np.testing.assert_array_equal(grads["w1"], expected_dw1)
    np.testing.assert_array_equal(grads["w2"], expected_dw2)
    np.testing.assert_array_equal(grads["b"], expected_db)

def test_backward_multiple_samples():
    m = make_low_rank(input_size=2, out_size=2, rank=2, bias=True)
    x = np.array([[1, 2], [3, 4]])
    m.x = x
    dy = np.array([[1, -1], [0.5, 0.5]])
    dx, grads = m.gradients(dy)

    expected_dx = dy @ m.params["w2"].T @ m.params["w1"].T
    expected_dw1 = x.T @ dy @ m.params["w2"].T
    expected_dw2 = (x @ m.params["w1"]).T @ dy
    expected_db = dy.sum(axis=0, keepdims=True)

    np.testing.assert_array_equal(dx, expected_dx)
    np.testing.assert_array_equal(grads["w1"], expected_dw1)
    np.testing.assert_array_equal(grads["w2"], expected_dw2)
    np.testing.assert_array_equal(grads["b"], expected_db)