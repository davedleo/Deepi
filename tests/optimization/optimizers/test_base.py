import numpy as np
import pytest
from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers.base import Optimizer
from deepi.optimization.regularization import Regularizer

# --------------------------------------------------------------------------
# Dummy optimizer and regularizer
# --------------------------------------------------------------------------

class DummyOptimizer(Optimizer):
    def direction(self, w, dw, buffer):
        # simply return the gradient unchanged
        return dw.copy()


class DummyRegularizer(Regularizer):
    def __init__(self, gamma = 0.001): 
        super().__init__(gamma, "dummy")

    def regularization(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)  # simple constant for testing


# --------------------------------------------------------------------------
# Model builder for tests
# --------------------------------------------------------------------------

def build_test_model():
    inp = Input((4,))
    dense1 = Dense(8)
    relu = ReLU()
    dense2 = Dense(3)

    inp.link(dense1)
    dense1.link(relu)
    relu.link(dense2)

    model = Model(inp, dense2)
    return model, inp, dense1, relu, dense2


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_step_without_regularizer():
    """Step should apply sign * lr * gradient without regularizer"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    opt = DummyOptimizer(
        model,
        lr=0.1,
        regularizer=None,
        decoupled_regularization=False,
        maximize=False,
        has_buffer=True,
        _type="dummy"
    )

    print(opt.buffer)

    # fake gradients
    for module in model.modules:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules_map.items():
        if module.has_params:
            for k, v in module.params.items():
                assert np.allclose(v, orig_params[name][k] - 0.1 * np.ones_like(v))


def test_step_with_regularizer_decoupled():
    """Step should apply regularizer additively in decoupled mode"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    reg = DummyRegularizer(gamma=1.0)
    opt = DummyOptimizer(
        model,
        lr=0.1,
        regularizer=reg,
        decoupled_regularization=True,
        maximize=False,
        has_buffer=True,
        _type="dummy"
    )

    for module in model.modules:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.zeros_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules_map.items():
        if module.has_params:
            for k, v in module.params.items():
                # regularizer adds 1 * lr
                assert np.allclose(v, orig_params[name][k] - 0.1 * np.ones_like(v))


def test_step_with_regularizer_coupled():
    """Step should combine gradient + regularizer before direction"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    reg = DummyRegularizer(gamma=1.0)
    opt = DummyOptimizer(
        model,
        lr=0.1,
        regularizer=reg,
        decoupled_regularization=False,
        maximize=False,
        has_buffer=True,
        _type="dummy"
    )

    for module in model.modules:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.zeros_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules_map.items():
        if module.has_params:
            for k, v in module.params.items():
                # dw + reg = 0 + 1, * lr
                assert np.allclose(v, orig_params[name][k] - 0.1 * np.ones_like(v))


def test_maximize_sign():
    """Step should use +1 for maximization"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    opt = DummyOptimizer(
        model,
        lr=0.1,
        regularizer=None,
        decoupled_regularization=False,
        maximize=True,
        has_buffer=True,
        _type="dummy"
    )

    for module in model.modules:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules_map.items():
        if module.has_params:
            for k, v in module.params.items():
                assert np.allclose(v, orig_params[name][k] + 0.1 * np.ones_like(v))


def test_multiple_gradients_preserved():
    """Ensure original gradient arrays are not modified by step"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    opt = DummyOptimizer(
        model,
        lr=0.1,
        regularizer=None,
        decoupled_regularization=False,
        maximize=False,
        has_buffer=True,
        _type="dummy"
    )

    for module in model.modules:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    # store copies of gradients
    grad_copies = {}
    for module in model.modules:
        if module.has_params:
            grad_copies[module] = {k: v.copy() for k, v in module.grads.items()}

    opt.step()

    for module in model.modules:
        if module.has_params:
            for k, v in module.grads.items():
                assert np.allclose(v, grad_copies[module][k])