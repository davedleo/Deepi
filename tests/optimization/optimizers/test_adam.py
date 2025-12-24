import numpy as np
import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers import Adam


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

def test_adam_initializes_buffers():
    """Adam initializes velocity and squared average"""
    model = build_test_model()
    model.train()

    opt = Adam(model)

    for module_buffer in opt.buffer.values():
        for buf in module_buffer.values():
            assert "velocity" in buf
            assert "square_avg" in buf
            assert "t" in buf
            assert isinstance(buf["velocity"], np.ndarray)
            assert isinstance(buf["square_avg"], np.ndarray)
            if opt.amsgrad:
                assert "square_avg_max" in buf
                assert isinstance(buf["square_avg_max"], np.ndarray)


def test_adam_single_step():
    """First Adam step uses bias-corrected velocity and square_avg"""
    model = build_test_model()
    model.train()

    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    opt = Adam(model, lr=lr, beta1=beta1, beta2=beta2, eps=eps)

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
                # bias correction
                velocity_hat = velocity / (1.0 - beta1)

                # square_avg = (1-beta2)*dw^2
                square_avg = (1.0 - beta2) * (dw ** 2)
                square_avg_hat = square_avg / (1.0 - beta2)

                expected = orig_params[name][k] - lr * velocity_hat / (np.sqrt(square_avg_hat) + eps)
                assert np.allclose(w, expected)


def test_adam_two_steps():
    """Adam accumulates velocity and squared gradients over steps"""
    model = build_test_model()
    model.train()

    lr = 0.01
    beta1 = 0.5
    beta2 = 0.9
    eps = 1e-8
    opt = Adam(model, lr=lr, beta1=beta1, beta2=beta2, eps=eps)

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
                s1 = (1.0 - beta2) * (dw ** 2)
                v_hat1 = v1 / (1.0 - beta1)
                s_hat1 = s1 / (1.0 - beta2)
                update1 = lr * v_hat1 / (np.sqrt(s_hat1) + eps)

                # ---- step 2 ----
                v2 = beta1 * v1 + (1.0 - beta1) * dw
                s2 = beta2 * s1 + (1.0 - beta2) * (dw ** 2)
                v_hat2 = v2 / (1.0 - beta1 ** 2)
                s_hat2 = s2 / (1.0 - beta2 ** 2)
                update2 = lr * v_hat2 / (np.sqrt(s_hat2) + eps)

                expected = orig_params[name][k] - (update1 + update2)
                assert np.allclose(w, expected)


def test_adam_amsgrad():
    """Adam with amsgrad keeps square_avg_max correctly"""
    model = build_test_model()
    model.train()

    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    opt = Adam(model, lr=lr, beta1=beta1, beta2=beta2, eps=eps, amsgrad=True)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    opt.step()

    for module_buffer in opt.buffer.values():
        for buf in module_buffer.values():
            assert np.all(buf["square_avg_max"] >= buf["square_avg"])


def test_adam_maximize_flag():
    """maximize=True inverts update direction"""
    model = build_test_model()
    model.train()

    opt = Adam(model, maximize=True)

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
                velocity_hat = velocity / (1.0 - opt.beta1)
                square_avg = (1.0 - opt.beta2) * (dw ** 2)
                square_avg_hat = square_avg / (1.0 - opt.beta2)
                expected = orig_params[name][k] + opt.lr * velocity_hat / (np.sqrt(square_avg_hat) + opt.eps)
                assert np.allclose(w, expected)


def test_adam_does_not_modify_gradients():
    """Adam must not modify gradients in-place"""
    model = build_test_model()
    model.train()

    opt = Adam(model)

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