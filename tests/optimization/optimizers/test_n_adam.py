import numpy as np
import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers import NAdam


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

def test_nadam_initializes_buffers():
    """NAdam initializes all required buffers"""
    model = build_test_model()
    model.train()

    opt = NAdam(model)

    for module_buffer in opt.buffer.values():
        for buf in module_buffer.values():
            assert "t" in buf
            assert "velocity" in buf
            assert "square_avg" in buf
            assert "nesterov_momentum" in buf
            assert "nesterov_momentum_prod" in buf

            assert isinstance(buf["velocity"], np.ndarray)
            assert isinstance(buf["square_avg"], np.ndarray)
            assert isinstance(buf["nesterov_momentum"], float)
            assert isinstance(buf["nesterov_momentum_prod"], float)


def test_nadam_single_step():
    """First NAdam step matches explicit Nesterov-Adam equations"""
    model = build_test_model()
    model.train()

    lr = 0.002
    beta1 = 0.9
    beta2 = 0.999
    psi = 0.004
    eps = 1e-8

    opt = NAdam(
        model,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        momentum_decay=psi,
        eps=eps,
    )

    # set gradients = 1 everywhere
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

            # ---- time step ----
            t = 1

            # ---- momentum schedule ----
            mu_t = beta1 * (1.0 - 0.5 * (0.96 ** (t * psi)))
            mu_t_next = beta1 * (1.0 - 0.5 * (0.96 ** ((t + 1) * psi)))

            mu_prod = mu_t
            mu_prod_next = mu_prod * mu_t_next

            # ---- velocity ----
            v = (1.0 - beta1) * dw

            # ---- Nesterov velocity hat ----
            v_hat_term1 = mu_t_next * v / (1.0 - mu_prod_next)
            v_hat_term2 = (1.0 - mu_t) * dw / (1.0 - mu_prod)
            v_hat = v_hat_term1 + v_hat_term2

            # ---- squared average ----
            s = (1.0 - beta2) * (dw ** 2)
            s_hat = s / (1.0 - beta2)

            expected = orig_params[name][k] - lr * v_hat / (np.sqrt(s_hat) + eps)
            assert np.allclose(w, expected)


def test_nadam_two_steps():
    """NAdam accumulates velocity, momentum schedule, and square_avg correctly"""
    model = build_test_model()
    model.train()

    lr = 0.01
    beta1 = 0.5
    beta2 = 0.9
    psi = 0.004
    eps = 1e-8

    opt = NAdam(
        model,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        momentum_decay=psi,
        eps=eps,
    )

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()

    # ---------------- Step 1 ----------------
    opt.step()

    # ---------------- Step 2 ----------------
    opt.step()

    for name, module in model.modules.items():
        if not module.has_params:
            continue

        for k, w in module.params.items():
            dw = np.ones_like(w)

            # ===== STEP 1 =====
            t1 = 1
            mu1 = beta1 * (1.0 - 0.5 * (0.96 ** (t1 * psi)))
            mu2 = beta1 * (1.0 - 0.5 * (0.96 ** ((t1 + 1) * psi)))

            mu_prod1 = mu1
            mu_prod2 = mu_prod1 * mu2

            v1 = (1.0 - beta1) * dw

            vhat1 = (
                mu2 * v1 / (1.0 - mu_prod2)
                + (1.0 - mu1) * dw / (1.0 - mu_prod1)
            )

            s1 = (1.0 - beta2) * (dw ** 2)
            shat1 = s1 / (1.0 - beta2)

            update1 = lr * vhat1 / (np.sqrt(shat1) + eps)

            # ===== STEP 2 =====
            t2 = 2
            mu3 = beta1 * (1.0 - 0.5 * (0.96 ** ((t2 + 1) * psi)))

            mu_prod3 = mu_prod2 * mu3

            v2 = beta1 * v1 + (1.0 - beta1) * dw

            vhat2 = (
                mu3 * v2 / (1.0 - mu_prod3)
                + (1.0 - mu2) * dw / (1.0 - mu_prod2)
            )

            s2 = beta2 * s1 + (1.0 - beta2) * (dw ** 2)
            shat2 = s2 / (1.0 - beta2 ** 2)

            update2 = lr * vhat2 / (np.sqrt(shat2) + eps)

            expected = orig_params[name][k] - (update1 + update2)
            assert np.allclose(w, expected)


def test_nadam_maximize_flag():
    """maximize=True inverts NAdam update direction"""
    model = build_test_model()
    model.train()

    opt = NAdam(model, maximize=True)

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
            # sign should be inverted compared to minimize
            assert np.any(w > orig_params[name][k])


def test_nadam_does_not_modify_gradients():
    """NAdam must not modify gradients in-place"""
    model = build_test_model()
    model.train()

    opt = NAdam(model)

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