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
    """RMSprop without momentum uses running square average scaling"""
    model = build_test_model()
    model.train()

    opt = RMSprop(model, lr=0.1, momentum=0.0, alpha=0.99, centered=False)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    # First step: square_avg = (1-alpha)*dw^2, update = lr * dw / sqrt(square_avg)
    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                square_avg = (1.0 - opt.alpha) * np.ones_like(w)
                expected = orig_params[name][k] - opt.lr * np.ones_like(w) / (np.sqrt(square_avg) + opt.eps)
                assert np.allclose(w, expected)


def test_rmsprop_with_momentum_initializes_buffers():
    """RMSprop with momentum should initialize velocity and square_avg buffers"""
    model = build_test_model()
    model.train()

    opt = RMSprop(model, lr=0.01, momentum=0.9)

    for module_buffer in opt.buffer["params"].values():
        for buf in module_buffer.values():
            assert "square_avg" in buf
            assert isinstance(buf["square_avg"], np.ndarray)
            if opt.mu > 0.0:
                assert "velocity" in buf
                assert isinstance(buf["velocity"], np.ndarray)


def test_rmsprop_with_momentum_single_step():
    """First RMSprop step with momentum uses scaled gradient"""
    model = build_test_model()
    model.train()

    opt = RMSprop(model, lr=0.1, momentum=0.9)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    # First step: v = dw / sqrt((1-alpha)*dw^2)
    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                square_avg = (1.0 - opt.alpha) * np.ones_like(w)
                v = np.ones_like(w) / (np.sqrt(square_avg) + opt.eps)
                expected = orig_params[name][k] - opt.lr * v
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
    # Step 2: square_avg remains constant for constant gradients
    opt.step()

    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                square_avg_1 = (1.0 - alpha) * np.ones_like(w)
                square_avg_2 = alpha * square_avg_1 + (1.0 - alpha) * np.ones_like(w)

                update1 = opt.lr / (np.sqrt(square_avg_1) + opt.eps)
                update2 = opt.lr / (np.sqrt(square_avg_2) + opt.eps)

                expected = orig_params[name][k] - (update1 + update2) * np.ones_like(w)
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

    # Step 1: v1 = dw / sqrt((1-alpha)*dw^2)
    opt.step()
    # Step 2: v2 = mu*v1 + dw / sqrt((1-alpha)*dw^2)
    opt.step()

    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                square_avg_1 = (1.0 - opt.alpha) * np.ones_like(w)
                square_avg_2 = opt.alpha * square_avg_1 + (1.0 - opt.alpha) * np.ones_like(w)

                v1 = 1.0 / (np.sqrt(square_avg_1) + opt.eps)
                v2 = mu * v1 + 1.0 / (np.sqrt(square_avg_2) + opt.eps)

                expected = orig_params[name][k] - opt.lr * (v1 + v2) * np.ones_like(w)
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
                square_avg = (1.0 - opt.alpha) * np.ones_like(w)
                expected = orig_params[name][k] + opt.lr * np.ones_like(w) / (np.sqrt(square_avg) + opt.eps)
                assert np.allclose(w, expected)


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