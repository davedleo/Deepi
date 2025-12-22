import numpy as np
import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers import Adamax


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

def test_adamax_initializes_buffers():
    """Adamax initializes velocity and dw_max buffers"""
    model = build_test_model()
    model.train()

    opt = Adamax(model)

    for module_buffer in opt.buffer.values():
        for buf in module_buffer.values():
            assert "t" in buf
            assert "velocity" in buf
            assert "dw_max" in buf
            assert isinstance(buf["velocity"], np.ndarray)
            assert isinstance(buf["dw_max"], np.ndarray)
            assert buf["t"] == 0


def test_adamax_single_step():
    """First Adamax step uses bias-corrected velocity and max-based scaling"""
    model = build_test_model()
    model.train()

    lr = 0.002
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    opt = Adamax(model, lr=lr, beta1=beta1, beta2=beta2, eps=eps)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                dw = np.ones_like(w)

                # velocity = (1-beta1)*dw
                velocity = (1.0 - beta1) * dw
                velocity_hat = velocity / (1.0 - beta1 ** 1)

                # dw_max = abs(dw) + eps
                dw_max = np.abs(dw) + eps

                expected = orig_params[name][k] - lr * velocity_hat / dw_max
                assert np.allclose(w, expected)


def test_adamax_two_steps():
    """Adamax accumulates velocity and dw_max over multiple steps"""
    model = build_test_model()
    model.train()

    lr = 0.002
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    opt = Adamax(model, lr=lr, beta1=beta1, beta2=beta2, eps=eps)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()

    # Step 1
    opt.step()
    # Step 2
    opt.step()

    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                dw = np.ones_like(w)

                # ---- step 1 ----
                v1 = (1.0 - beta1) * dw
                v_hat1 = v1 / (1.0 - beta1 ** 1)
                dw_max1 = np.abs(dw) + eps
                update1 = lr * v_hat1 / dw_max1

                # ---- step 2 ----
                v2 = beta1 * v1 + (1.0 - beta1) * dw
                v_hat2 = v2 / (1.0 - beta1 ** 2)
                dw_max2 = np.maximum(beta2 * dw_max1, np.abs(dw) + eps)
                update2 = lr * v_hat2 / dw_max2

                expected = orig_params[name][k] - (update1 + update2)
                assert np.allclose(w, expected)


def test_adamax_maximize_flag():
    """maximize=True inverts update direction"""
    model = build_test_model()
    model.train()

    opt = Adamax(model, maximize=True)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                dw = np.ones_like(w)
                velocity = (1.0 - opt.beta1) * dw
                velocity_hat = velocity / (1.0 - opt.beta1 ** 1)
                dw_max = np.abs(dw) + opt.eps
                expected = orig_params[name][k] + opt.lr * velocity_hat / dw_max
                assert np.allclose(w, expected)


def test_adamax_does_not_modify_gradients():
    """Adamax must not modify gradients in-place"""
    model = build_test_model()
    model.train()

    opt = Adamax(model)

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