import numpy as np
import pytest
from deepi.modules.loss import CrossEntropy

@pytest.mark.parametrize("weights", [None, {0: 1.0, 1: 2.0}])
def test_forward_exact(weights):
    loss_fn = CrossEntropy(weights=weights)
    loss_fn.train()

    y_hat = np.array([[2.0, 1.0, 0.1],
                      [0.5, 2.5, 0.0]])
    y = np.array([0, 1])

    shifted = y_hat - np.max(y_hat, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    if weights is None:
        expected = -np.log(probs[np.arange(len(y)), y]).mean()
    else:
        sample_weights = np.array([weights[label] for label in y], dtype=float)
        sample_weights /= sample_weights.sum()
        expected = (-sample_weights * np.log(probs[np.arange(len(y)), y])).mean()

    loss = loss_fn.forward(y_hat, y)
    assert np.allclose(loss, expected), "Forward output mismatch"

@pytest.mark.parametrize("weights", [None, {0: 1.0, 1: 2.0}])
def test_backward_exact(weights):
    loss_fn = CrossEntropy(weights=weights)
    loss_fn.train()

    y_hat = np.array([[2.0, 1.0, 0.1],
                      [0.5, 2.5, 0.0]])
    y = np.array([0, 1])

    loss_fn.forward(y_hat, y)
    grad = loss_fn.backward()

    shifted = y_hat - np.max(y_hat, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    n_samples = y.shape[0]
    expected = probs.copy()
    expected[np.arange(n_samples), y] -= 1.0

    if weights is not None:
        sample_weights = np.array([weights[label] for label in y], dtype=float)
        sample_weights /= sample_weights.sum()
        expected *= sample_weights[:, None]

    expected /= n_samples
    assert np.allclose(grad, expected), "Backward gradients mismatch"

def test_forward_eval_mode_no_cache():
    loss_fn = CrossEntropy()
    loss_fn.eval()

    y_hat = np.array([[1.0, 2.0, 3.0]])
    y = np.array([2])

    loss = loss_fn.forward(y_hat, y)
    assert loss_fn.x is None
    assert loss_fn.y is None

def test_forward_train_mode_caches():
    loss_fn = CrossEntropy()
    loss_fn.train()

    y_hat = np.array([[1.0, 2.0, 3.0]])
    y = np.array([2])

    loss = loss_fn.forward(y_hat, y)
    assert np.allclose(loss_fn.x[0], y_hat)
    assert np.allclose(loss_fn.x[1], y)
    assert np.allclose(loss_fn.y, loss)

def test_clear_resets():
    loss_fn = CrossEntropy()
    loss_fn.train()

    y_hat = np.array([[1.0, 2.0, 3.0]])
    y = np.array([2])

    loss_fn.forward(y_hat, y)
    loss_fn.backward()

    loss_fn.clear()
    assert loss_fn.x is None
    assert loss_fn.y is None
    assert loss_fn.dy is None