"""
Numerical gradient checks for every hand-derived gradient in src/models.py.
This is run before any real experiment (see main.py) -- if any of these
checks fail, the experiment aborts rather than reporting numbers computed
from a buggy gradient.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.models import (init_params, erm_grad, irm_grad, cisd_grad)

rng = np.random.RandomState(0)


def numerical_grad(f, p, key, eps=1e-5):
    g = np.zeros_like(p[key])
    it = np.nditer(p[key], flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = p[key][idx]
        p[key][idx] = orig + eps
        f_plus = f()
        p[key][idx] = orig - eps
        f_minus = f()
        p[key][idx] = orig
        g[idx] = (f_plus - f_minus) / (2 * eps)
    return g


def check(name, analytic, numeric, tol=1e-3):
    rel_err = np.max(np.abs(analytic - numeric) / (np.abs(numeric) + 1e-6))
    status = "OK" if rel_err < tol else "FAIL"
    print(f"  [{status}] {name:10s} max_rel_err={rel_err:.2e}")
    return rel_err < tol


def test_erm():
    print("ERM gradient check")
    N, D, H = 20, 5, 4
    X = rng.randn(N, D)
    y = (rng.rand(N) > 0.5).astype(float)
    p = init_params(D, H, n_env=1, seed=1)

    def f():
        loss, _, _ = erm_grad(X, y, p, l2=0.0)
        return loss

    _, _, grads = erm_grad(X, y, p, l2=0.0)
    ok = True
    for key in ["w_main", "b_main", "W1", "b1"]:
        num = numerical_grad(f, p, key)
        ok &= check(key, grads[key], num)
    return ok


def test_irm():
    print("IRM gradient check")
    N, D, H = 24, 5, 4
    X = rng.randn(N, D)
    y = (rng.rand(N) > 0.5).astype(float)
    envs = np.array([0] * 12 + [1] * 12)
    env_mask_list = [envs == 0, envs == 1]
    p = init_params(D, H, n_env=1, seed=2)

    def f():
        loss, pen, _, _ = irm_grad(X, y, p, env_mask_list, lambda_irm=2.0, l2=0.0)
        return loss + 2.0 * pen

    _, _, _, grads = irm_grad(X, y, p, env_mask_list, lambda_irm=2.0, l2=0.0)
    ok = True
    for key in ["w_main", "b_main", "W1", "b1"]:
        num = numerical_grad(f, p, key)
        ok &= check(key, grads[key], num)
    return ok


def test_cisd():
    print("CISD gradient check (loss_main + lambda_env*loss_env + lambda_orth*loss_orth part)")
    N, D, H = 24, 5, 6
    h1 = 3
    X = rng.randn(N, D)
    y = (rng.rand(N) > 0.5).astype(float)
    K = 2
    env_ids = rng.randint(0, K, size=N)
    env_onehot = np.eye(K)[env_ids]
    p = init_params(D, H, n_env=K, seed=3, split=(h1, H - h1))

    lambda_env, lambda_orth = 0.7, 0.3

    def f():
        total, _, _, _ = cisd_grad(X, y, env_onehot, p, h1,
                                    lambda_adv=0.0, lambda_env=lambda_env,
                                    lambda_orth=lambda_orth, l2=0.0)
        return total

    _, _, grads, _ = cisd_grad(X, y, env_onehot, p, h1, lambda_adv=0.0,
                                lambda_env=lambda_env, lambda_orth=lambda_orth, l2=0.0)
    ok = True
    for key in ["w_main", "b_main", "W1", "b1", "W_env", "b_env"]:
        num = numerical_grad(f, p, key)
        ok &= check(key, grads[key], num)
    return ok


def test_cisd_adv_own_params():
    """The adversarial classifier's OWN parameters (W_adv, b_adv) should
    still receive a normal (non-reversed) gradient w.r.t. loss_adv itself,
    since GRL only reverses the signal flowing to earlier layers."""
    print("CISD adversarial-head own-gradient check")
    N, D, H = 24, 5, 6
    h1 = 3
    X = rng.randn(N, D)
    y = (rng.rand(N) > 0.5).astype(float)
    K = 2
    env_ids = rng.randint(0, K, size=N)
    env_onehot = np.eye(K)[env_ids]
    p = init_params(D, H, n_env=K, seed=4, split=(h1, H - h1))

    def f():
        _, _, _, metrics = cisd_grad(X, y, env_onehot, p, h1, lambda_adv=1.0,
                                      lambda_env=0.0, lambda_orth=0.0, l2=0.0)
        return metrics["loss_adv"]

    _, _, grads, _ = cisd_grad(X, y, env_onehot, p, h1, lambda_adv=1.0,
                                lambda_env=0.0, lambda_orth=0.0, l2=0.0)
    ok = True
    for key in ["W_adv", "b_adv"]:
        num = numerical_grad(f, p, key)
        ok &= check(key, grads[key], num)
    return ok


if __name__ == "__main__":
    results = [test_erm(), test_irm(), test_cisd(), test_cisd_adv_own_params()]
    print()
    if all(results):
        print("ALL GRADIENT CHECKS PASSED")
        sys.exit(0)
    else:
        print("GRADIENT CHECK FAILURE -- DO NOT TRUST DOWNSTREAM RESULTS")
        sys.exit(1)
