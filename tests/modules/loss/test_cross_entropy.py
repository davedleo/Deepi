import numpy as np
import pytest
from deepi.modules.loss import CrossEntropy

@pytest.mark.parametrize("weights", [None, {0: 1.0, 1: 2.0, 2: 1.5}])
def test_forward_large_batch(weights):
    loss_fn = CrossEntropy(weights=weights)
    loss_fn.train()

    y_hat = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.5, 0.0],
        [1.0, 0.0, 3.0],
        [0.1, 2.0, 1.5]
    ])
    y = np.array([0, 1, 2, 1])

    shifted = y_hat - np.max(y_hat, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    if weights is None:
        expected = -np.log(probs[np.arange(len(y)), y]).mean()
    else:
        labels_list = np.array(sorted(weights.keys()))
        weights_array = np.array([weights[label] for label in labels_list], dtype=float)
        weights_array /= weights_array.sum()
        sample_weights = weights_array[np.searchsorted(labels_list, y)]
        expected = (-sample_weights * np.log(probs[np.arange(len(y)), y])).mean()

    loss = loss_fn(y_hat, y)
    assert np.allclose(loss, expected), "Forward output mismatch on large batch"

@pytest.mark.parametrize("weights", [None, {0: 1.0, 1: 2.0, 2: 1.5}])
def test_backward_large_batch(weights):
    loss_fn = CrossEntropy(weights=weights)
    loss_fn.train()

    y_hat = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.5, 0.0],
        [1.0, 0.0, 3.0],
        [0.1, 2.0, 1.5]
    ])
    y = np.array([0, 1, 2, 1])

    loss_fn(y_hat, y)
    grad = loss_fn.backward()

    shifted = y_hat - np.max(y_hat, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    n_samples = y.shape[0]
    expected = probs.copy()
    expected[np.arange(n_samples), y] -= 1.0

    if weights is not None:
        labels_list = np.array(sorted(weights.keys()))
        weights_array = np.array([weights[label] for label in labels_list], dtype=float)
        weights_array /= weights_array.sum()
        sample_weights = weights_array[np.searchsorted(labels_list, y)]
        expected *= sample_weights[:, None]

    expected /= n_samples
    assert np.allclose(grad, expected), "Backward gradients mismatch on large batch"

def test_forward_multiple_classes_consistency():
    loss_fn = CrossEntropy()
    loss_fn.train()

    y_hat = np.array([
        [1.0, 2.0, 3.0, 0.5],
        [0.5, 1.5, 0.2, 2.0]
    ])
    y = np.array([2, 3])

    shifted = y_hat - np.max(y_hat, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    expected = -np.log(probs[np.arange(len(y)), y]).mean()

    loss = loss_fn(y_hat, y)
    assert np.allclose(loss, expected), "Forward mismatch with multiple classes"

def test_backward_multiple_classes_consistency():
    loss_fn = CrossEntropy()
    loss_fn.train()

    y_hat = np.array([
        [1.0, 2.0, 3.0, 0.5],
        [0.5, 1.5, 0.2, 2.0]
    ])
    y = np.array([2, 3])

    loss_fn(y_hat, y)
    grad = loss_fn.backward()

    shifted = y_hat - np.max(y_hat, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    n_samples = y.shape[0]
    expected = probs.copy()
    expected[np.arange(n_samples), y] -= 1.0
    expected /= n_samples

    assert np.allclose(grad, expected), "Backward mismatch with multiple classes"

def test_forward_eval_mode_no_cache_large_batch():
    loss_fn = CrossEntropy()
    loss_fn.eval()

    y_hat = np.array([[1.0, 2.0, 3.0], [0.5, 2.0, 1.5]])
    y = np.array([2, 1])

    loss = loss_fn(y_hat, y)
    assert loss_fn.x is None
    assert loss_fn.y is None