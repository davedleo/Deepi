import numpy as np
import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers import Muon


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

def test_muon_initializes_buffers():
    """Muon initializes momentum and lr buffers"""
    model = build_test_model()
    model.train()

    lr = 1e-3
    opt = Muon(model, lr=lr)

    for module_buffer in opt.buffer.values():
        for buf in module_buffer.values():
            assert "momentum" in buf
            assert "lr" in buf
            assert isinstance(buf["momentum"], np.ndarray)
            assert isinstance(buf["lr"], float)
            assert buf["lr"] == lr


def test_muon_single_step_no_nesterov():
    """Muon single step without Nesterov matches explicit formula"""
    model = build_test_model()
    model.train()

    lr = 1e-3
    momentum = 0.9
    eps = 1e-7

    opt = Muon(
        model,
        lr=lr,
        momentum=momentum,
        nesterov=False,
        eps=eps,
    )

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules.items():
        if not module.has_params:
            continue

        for k, w in module.params.items():
            dw = np.ones_like(w)

            # momentum buffer
            B = dw

            # B_tilde = B (no nesterov)
            B_tilde = B

            if B_tilde.ndim == 1:
                norm = np.linalg.norm(B_tilde)
                O = B_tilde / norm if norm > eps else B_tilde
                expected = orig_params[name][k] - lr * O
                assert np.allclose(w, expected)


def test_muon_single_step_nesterov_vector():
    """Muon Nesterov update for vector parameters"""
    model = build_test_model()
    model.train()

    lr = 1e-3
    momentum = 0.95
    eps = 1e-7

    opt = Muon(
        model,
        lr=lr,
        momentum=momentum,
        nesterov=True,
        eps=eps,
    )

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules.items():
        if not module.has_params:
            continue

        for k, w in module.params.items():
            if w.ndim != 1:
                continue

            dw = np.ones_like(w)

            B = dw
            B_tilde = dw + momentum * B

            norm = np.linalg.norm(B_tilde)
            O = B_tilde / norm if norm > eps else B_tilde

            expected = orig_params[name][k] - lr * O
            assert np.allclose(w, expected)


def test_muon_lr_adjustment_matrix():
    """Muon applies Moonshot LR adjustment for 2D parameters"""
    model = build_test_model()
    model.train()

    lr = 1e-3
    opt = Muon(model, lr=lr)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    opt.step()

    for module_id, module in model.modules.items():
        if not module.has_params:
            continue

        for k, w in module.params.items():
            if w.ndim == 2:
                buf = opt.buffer[module_id][k]
                A, B = w.shape
                expected_lr = 0.2 * lr * max(A, B)
                assert np.isclose(buf["lr"], expected_lr)


def test_muon_two_steps_accumulates_momentum():
    """Muon accumulates momentum across steps"""
    model = build_test_model()
    model.train()

    lr = 1e-3
    momentum = 0.5

    opt = Muon(
        model,
        lr=lr,
        momentum=momentum,
        nesterov=False,
    )

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    # step 1
    opt.step()
    # step 2
    opt.step()

    for module_id, module in model.modules.items():
        if not module.has_params:
            continue

        for k, buf in opt.buffer[module_id].items():
            B = buf["momentum"]
            expected = momentum * np.ones_like(B) + np.ones_like(B)
            assert np.allclose(B, expected)


def test_muon_maximize_flag():
    """maximize=True inverts Muon update direction"""
    model = build_test_model()
    model.train()

    lr = 1e-3
    opt = Muon(model, lr=lr, maximize=True)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules.items():
        if not module.has_params:
            continue

        for k, w in module.params.items():
            assert np.any(w > orig_params[name][k])


def test_muon_does_not_modify_gradients():
    """Muon must not modify gradients in-place"""
    model = build_test_model()
    model.train()

    opt = Muon(model)

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