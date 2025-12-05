import numpy as np
import pytest

from deepi.modules.linear import Dense, LowRank


@pytest.fixture
def x():
    return np.arange(12, dtype=float).reshape(3, 4) - 5.0


@pytest.fixture
def dy():
    return 0.1 * (np.arange(12, dtype=float).reshape(3, 4) - 5.0)


def test_dense_initialization():

    dense = Dense(out_features=3, bias=True)

    # Initial attributes
    assert dense.type == "module.linear.dense"
    assert dense._is_training is False  # Module default
    assert dense._has_bias
    assert isinstance(dense.params["b"], np.ndarray)
    assert dense.params["b"].shape == (1, 3)

    # Before set_input it stores symbolic shape
    assert dense.params["w"] == (3,)


def test_dense_set_input(x):

    dense = Dense(out_features=3, bias=False)

    dense.set_input(x)

    # After set_input, actual shape is set
    assert isinstance(dense.params["w"], tuple)
    assert dense.params["w"][0] == x.shape[1]
    assert dense.params["w"][1] == (3,)


def test_dense_forward_no_training(x):

    dense = Dense(out_features=3, bias=True)
    dense.set_input(x)

    # define real weight
    in_features, out_features = x.shape[1], 3
    dense.params["w"] = np.arange(in_features * out_features).reshape(in_features, out_features)
    dense.params["b"] = np.ones((1, out_features))

    dense.eval()  # disable training mode

    y = dense.forward(x)

    expected = x @ dense.params["w"] + dense.params["b"]

    assert isinstance(y, np.ndarray)
    assert np.allclose(y, expected)
    assert dense.cache is None
    assert dense.grads == {}


def test_dense_forward_training_and_backward(x, dy):

    dense = Dense(out_features=3, bias=True)
    dense.set_input(x)

    in_features, out_features = x.shape[1], 3
    W = np.arange(in_features * out_features).reshape(in_features, out_features)
    B = np.arange(out_features).reshape(1, out_features)

    dense.params["w"] = W.copy()
    dense.params["b"] = B.copy()

    dense.train()

    out = dense.forward(x)

    # Forward explicit formula
    expected = x @ W + B

    assert np.allclose(out, expected)

    # Check cached values
    assert isinstance(dense.cache, np.ndarray)
    assert np.allclose(dense.cache, W.T)

    # Compute grads explicitly
    dy_copy = dy[:, :3].copy()

    expected_grad_w = x.T @ dy_copy
    expected_grad_b = dy_copy.sum(0, keepdims=True)

    dx = dense.backward(dy_copy)

    expected_dx = dy_copy @ W.T

    assert np.allclose(dx, expected_dx)

    # Check parameter grad lambdas
    assert np.allclose(dense.grads["w"](dy_copy), expected_grad_w)
    assert np.allclose(dense.grads["b"](dy_copy), expected_grad_b)


#
# LowRank tests
#

def test_lowrank_initialization():

    lr = LowRank(out_features=4, rank=2, bias=True)

    assert lr.type == "module.linear.low_rank"
    assert lr._has_bias
    assert isinstance(lr.params["b"], np.ndarray)
    assert lr.params["b"].shape == (1, 4)

    # symbolic shapes before input known
    assert lr.params["w1"] == (2,)
    assert lr.params["w2"] == (2, 4)


def test_lowrank_set_input(x):

    lr = LowRank(out_features=4, rank=2, bias=False)

    lr.set_input(x)

    assert isinstance(lr.params["w1"], tuple)
    assert lr.params["w1"][0] == x.shape[1]  # in_features
    assert lr.params["w1"][1] == 2          # rank remains fixed


def test_lowrank_forward_no_training(x):

    lr = LowRank(out_features=2, rank=2, bias=True)

    lr.set_input(x)

    W1 = np.arange(x.shape[1] * 2).reshape(x.shape[1], 2)
    W2 = np.arange(2 * 2).reshape(2, 2)
    B = np.ones((1, 2))

    lr.params["w1"] = W1.copy()
    lr.params["w2"] = W2.copy()
    lr.params["b"] = B.copy()

    lr.eval()

    y = lr.forward(x)

    expected = (x @ W1) @ W2 + B

    assert isinstance(y, np.ndarray)
    assert np.allclose(y, expected)
    assert lr.cache is None
    assert lr.grads == {}


def test_lowrank_forward_training_and_backward(x, dy):

    lr = LowRank(out_features=2, rank=2, bias=True)
    lr.set_input(x)

    W1 = np.arange(x.shape[1] * 2).reshape(x.shape[1], 2)
    W2 = np.arange(2 * 2).reshape(2, 2)
    B = np.arange(2).reshape(1, 2)

    lr.params["w1"] = W1.copy()
    lr.params["w2"] = W2.copy()
    lr.params["b"] = B.copy()

    lr.train()

    y = lr.forward(x)

    expected = (x @ W1) @ W2 + B
    xW1 = x @ W1

    assert np.allclose(y, expected)

    # cache correctness
    assert lr.cache[0].shape == W1.shape
    assert lr.cache[1].shape == W2.shape
    assert np.allclose(lr.cache[0], W1)
    assert np.allclose(lr.cache[1], W2)

    # gradient formulas
    dy_copy = dy[:, :2].copy()

    expected_grad_w1 = x.T @ dy_copy @ W2.T
    expected_grad_w2 = xW1.T @ dy_copy
    expected_grad_b = dy_copy.sum(0, keepdims=True)

    dx = lr.backward(dy_copy)

    # explicit dx: dy @ W2.T @ W1.T
    expected_dx = dy_copy @ W2.T
    expected_dx = expected_dx @ W1.T

    assert np.allclose(dx, expected_dx)

    assert np.allclose(lr.grads["w1"](dy_copy), expected_grad_w1)
    assert np.allclose(lr.grads["w2"](dy_copy), expected_grad_w2)
    assert np.allclose(lr.grads["b"](dy_copy), expected_grad_b)