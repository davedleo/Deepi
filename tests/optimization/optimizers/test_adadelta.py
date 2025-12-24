import numpy as np
import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers import Adadelta


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

def test_adadelta_initializes_buffers():
    """Adadelta initializes running averages"""
    model = build_test_model()
    model.train()

    opt = Adadelta(model)

    for module_buffer in opt.buffer.values():
        for buf in module_buffer.values():
            assert "square_dw_avg" in buf
            assert "square_delta_avg" in buf
            assert isinstance(buf["square_dw_avg"], np.ndarray)
            assert isinstance(buf["square_delta_avg"], np.ndarray)


def test_adadelta_single_step():
    """First Adadelta step uses eps-scaled update"""
    model = build_test_model()
    model.train()

    rho = 0.9
    eps = 1e-8
    opt = Adadelta(model, rho=rho, eps=eps)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                # square_dw_avg_1 = (1-rho) * dw^2, with dw=1
                square_dw = (1.0 - rho) * np.ones_like(w)

                # square_delta_avg_0 = 0
                dw = np.ones_like(w)
                delta = np.sqrt(0.0 + eps) / np.sqrt(square_dw + eps) * dw

                expected = orig_params[name][k] - opt.lr * delta
                assert np.allclose(w, expected)


def test_adadelta_two_steps():
    """Adadelta accumulates gradient and update history (PyTorch style)"""
    model = build_test_model()
    model.train()

    rho = 0.5
    eps = 1e-8
    lr = 1.0
    opt = Adadelta(model, rho=rho, eps=eps, lr=lr)

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
                square_dw_1 = (1.0 - rho) * (dw ** 2)
                delta_1 = (np.sqrt(0.0 + eps)) / (np.sqrt(square_dw_1 + eps)) * dw
                update_1 = lr * delta_1
                square_delta_1 = (1.0 - rho) * (update_1 ** 2)

                # ---- step 2 ----
                square_dw_2 = rho * square_dw_1 + (1.0 - rho) * (dw ** 2)
                delta_2 = np.sqrt(square_delta_1 + eps) / np.sqrt(square_dw_2 + eps) * dw
                update_2 = lr * delta_2

                expected = orig_params[name][k] - (update_1 + update_2)

                assert np.allclose(w, expected)
                

def test_adadelta_maximize_flag():
    """maximize=True inverts update direction"""
    model = build_test_model()
    model.train()

    opt = Adadelta(model, maximize=True)

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
                square_dw = (1.0 - opt.rho) * (dw ** 2)
                delta = np.sqrt(0.0 + opt.eps) / np.sqrt(square_dw + opt.eps) * dw

                expected = orig_params[name][k] + opt.lr * delta
                assert np.allclose(w, expected)


def test_adadelta_does_not_modify_gradients():
    """Adadelta must not modify gradients in-place"""
    model = build_test_model()
    model.train()

    opt = Adadelta(model)

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