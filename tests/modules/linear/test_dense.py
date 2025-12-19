import numpy as np
import pytest
from deepi.modules.linear import Dense

def make_dense(input_size=3, output_size=2, bias=True):
    m = Dense(out_size=output_size, bias=bias)
    # simulate external weight initialization
    m.params["w"] = np.arange(input_size * output_size).reshape(input_size, output_size).astype(np.float64)
    if bias:
        m.params["b"] = np.ones((1, output_size))
    m.x = None
    m.grads = {}
    return m

def test_forward_no_bias():
    m = make_dense(input_size=3, output_size=2, bias=False)
    x = np.array([[1.0, 2.0, 3.0]])
    m.x = x
    y = m.transform(x)
    expected = x @ m.params["w"]
    np.testing.assert_array_equal(y, expected)

def test_forward_with_bias():
    m = make_dense(input_size=3, output_size=2, bias=True)
    x = np.array([[1.0, 2.0, 3.0]])
    m.x = x
    y = m.transform(x)
    expected = x @ m.params["w"] + m.params["b"]
    np.testing.assert_array_equal(y, expected)

def test_backward_no_bias():
    m = make_dense(input_size=3, output_size=2, bias=False)
    x = np.array([[1.0, 2.0, 3.0]])
    m.x = x
    dy = np.array([[1.0, -1.0]])
    dx, grads = m.gradients(dy)
    expected_dx = dy @ m.params["w"].T
    expected_dw = x.T @ dy
    np.testing.assert_array_equal(dx, expected_dx)
    np.testing.assert_array_equal(grads["w"], expected_dw)
    assert "b" not in grads

def test_backward_with_bias():
    m = make_dense(input_size=3, output_size=2, bias=True)
    x = np.array([[1.0, 2.0, 3.0]])
    m.x = x
    dy = np.array([[1.0, -1.0]])
    dx, grads = m.gradients(dy)
    expected_dx = dy @ m.params["w"].T
    expected_dw = x.T @ dy
    expected_db = dy.sum(axis=0, keepdims=True)
    np.testing.assert_array_equal(dx, expected_dx)
    np.testing.assert_array_equal(grads["w"], expected_dw)
    np.testing.assert_array_equal(grads["b"], expected_db)

def test_backward_multiple_samples():
    m = make_dense(input_size=2, output_size=2, bias=True)
    x = np.array([[1, 2], [3, 4]])
    m.x = x
    dy = np.array([[1, -1], [0.5, 0.5]])
    dx, grads = m.gradients(dy)
    expected_dx = dy @ m.params["w"].T
    expected_dw = x.T @ dy
    expected_db = dy.sum(axis=0, keepdims=True)
    np.testing.assert_array_equal(dx, expected_dx)
    np.testing.assert_array_equal(grads["w"], expected_dw)
    np.testing.assert_array_equal(grads["b"], expected_db)