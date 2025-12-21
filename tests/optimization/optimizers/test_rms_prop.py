import numpy as np
import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers import RMSprop


# --------------------------------------------------------------------------
# Model builder
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
    return model


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_rmsprop_without_momentum():
    """RMSprop without momentum behaves like gradient descent with RMS scaling"""
    model = build_test_model()
    model.train()

    opt = RMSprop(model, lr=0.1, momentum=0.0, alpha=0.99, centered=False)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    # Since it's the first step, square_avg = dw^2 = 1 → sqrt = 1, update = lr * dw / sqrt = 0.1
    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                expected = orig_params[name][k] - 0.1 * np.ones_like(w)
                assert np.allclose(w, expected)


def test_rmsprop_with_momentum_initializes_buffers():
    """RMSprop with momentum should initialize velocity and square_avg buffers"""
    model = build_test_model()
    model.train()

    opt = RMSprop(model, lr=0.01, momentum=0.9)

    for module_buffer in opt.buffer["params"].values():
        for buf in module_buffer.values():
            assert "velocity" in buf
            assert "square_avg" in buf
            assert buf["velocity"] is None or isinstance(buf["velocity"], np.ndarray)
            assert buf["square_avg"] is None or isinstance(buf["square_avg"], np.ndarray)


def test_rmsprop_with_momentum_single_step():
    """First RMSprop step with momentum: v = dw / sqrt(square_avg)"""
    model = build_test_model()
    model.train()

    opt = RMSprop(model, lr=0.1, momentum=0.9)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    # First step: square_avg = 1, sqrt = 1, v = dw / sqrt = 1 → update = lr * v = 0.1
    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                expected = orig_params[name][k] - 0.1 * np.ones_like(w)
                assert np.allclose(w, expected)


def test_rmsprop_two_steps_no_momentum():
    """RMSprop accumulates square_avg across steps (without momentum)"""
    model = build_test_model()
    model.train()

    alpha = 0.5  # strong decay to see effect
    opt = RMSprop(model, lr=0.1, momentum=0.0, alpha=alpha)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()

    # Step 1: square_avg = 1, update = 0.1
    opt.step()
    # Step 2: square_avg = alpha*1 + (1-alpha)*1 = 1 → update = 0.1
    opt.step()

    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                total_update = 0.1 + 0.1
                expected = orig_params[name][k] - total_update * np.ones_like(w)
                assert np.allclose(w, expected)


def test_rmsprop_with_momentum_two_steps():
    """Momentum RMSprop accumulates velocity across steps"""
    model = build_test_model()
    model.train()

    mu = 0.5
    opt = RMSprop(model, lr=0.1, momentum=mu)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()

    # Step 1: v1 = dw / sqrt(square_avg) = 1 → update1 = 0.1
    opt.step()
    # Step 2: v2 = mu * v1 + dw / sqrt(square_avg) = 0.5*1 + 1 = 1.5 → update2 = 0.15
    opt.step()

    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                total_update = 0.1 + 0.15
                expected = orig_params[name][k] - total_update * np.ones_like(w)
                assert np.allclose(w, expected)


def test_rmsprop_maximize_flag():
    """maximize=True inverts gradient direction"""
    model = build_test_model()
    model.train()

    opt = RMSprop(model, lr=0.1, momentum=0.0, maximize=True)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                assert np.allclose(
                    w,
                    orig_params[name][k] + 0.1 * np.ones_like(w)
                )


def test_rmsprop_does_not_modify_gradients():
    """RMSprop must not modify gradients in-place"""
    model = build_test_model()
    model.train()

    opt = RMSprop(model, lr=0.1, momentum=0.9)

    grad_copies = {}
    for module in model.topology:
        if module.has_params:
            grad_copies[module] = {}
            for k in module.params:
                g = np.ones_like(module.params[k])
                module.grads[k] = g
                grad_copies[module][k] = g.copy()

    opt.step()

    for module in model.topology:
        if module.has_params:
            for k, g in module.grads.items():
                assert np.allclose(g, grad_copies[module][k])