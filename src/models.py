"""
Hand-derived NumPy implementations of three models trained on pooled
multi-environment data:

  ERM   -- standard empirical risk minimization (1-hidden-layer MLP).
  IRM   -- Invariant Risk Minimization (Arjovsky et al., 2019), IRMv1
           penalty computed analytically for a linear head (closed-form
           gradient of the per-environment loss w.r.t. a scalar dummy
           classifier fixed at w=1, which is algebraically equivalent to
           the commonly-used autograd version for a linear final layer).
  CISD  -- "Causal-Invariant Stable/spurious Decomposition" (proposed).
           The shared hidden representation Z is split into two disjoint
           blocks Z_stable, Z_spurious. Only Z_stable feeds the outcome
           predictor. An adversarial environment classifier (gradient-
           reversal) is attached to Z_stable to actively strip
           environment-identifying information out of it, while a
           second, non-adversarial environment classifier is attached to
           Z_spurious to encourage it to *absorb* environment-specific
           signal instead of leaving it to leak into Z_stable. An
           orthogonality penalty decorrelates the two blocks.

No autodiff framework is used: every gradient below was derived by hand
and is implemented directly. A numerical-gradient check for each model is
provided in tests/test_gradients.py and is run before every experiment.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def bce_loss(p, y, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class Adam:
    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def init_params(n_in, n_hidden, n_env, seed, split=None):
    rng = np.random.RandomState(seed)

    def glorot(fan_in, fan_out):
        s = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-s, s, size=(fan_in, fan_out))

    p = {
        "W1": glorot(n_in, n_hidden), "b1": np.zeros(n_hidden),
        "w_main": glorot(n_hidden if split is None else split[0], 1).ravel(),
        "b_main": np.zeros(1),
    }
    if split is not None:
        h1, h2 = split
        p["W_adv"] = glorot(h1, n_env)
        p["b_adv"] = np.zeros(n_env)
        p["W_env"] = glorot(h2, n_env)
        p["b_env"] = np.zeros(n_env)
    return p


def forward_encoder(X, p):
    Zpre = X @ p["W1"] + p["b1"]
    Z = np.tanh(Zpre)
    return Zpre, Z


def erm_forward(X, p):
    _, Z = forward_encoder(X, p)
    logit = Z @ p["w_main"] + p["b_main"][0]
    return Z, logit


def erm_grad(X, y, p, l2=1e-4):
    N = X.shape[0]
    Zpre, Z = forward_encoder(X, p)
    logit = Z @ p["w_main"] + p["b_main"][0]
    pred = sigmoid(logit)
    dlogit = (pred - y) / N  # dL/dlogit for BCE-with-logits
    grads = {}
    grads["w_main"] = Z.T @ dlogit + l2 * p["w_main"]
    grads["b_main"] = np.array([dlogit.sum()])
    dZ = np.outer(dlogit, p["w_main"])
    dZpre = dZ * (1 - Z ** 2)
    grads["W1"] = X.T @ dZpre + l2 * p["W1"]
    grads["b1"] = dZpre.sum(axis=0)
    loss = bce_loss(pred, y)
    return loss, pred, grads


def irm_penalty_and_grad(X, y, p, env_mask_list):
    """IRMv1 penalty: sum_e || d/dw L_e(w*f(x)) |_{w=1} ||^2, with analytic
    gradient of the penalty back into the encoder / main head.
    """
    Zpre, Z = forward_encoder(X, p)
    logit = Z @ p["w_main"] + p["b_main"][0]
    pred = sigmoid(logit)
    total_pen = 0.0
    dlogit_pen = np.zeros_like(logit)
    for mask in env_mask_list:
        n_e = mask.sum()
        if n_e == 0:
            continue
        g_e = ((pred[mask] - y[mask]) * logit[mask])  # per-sample d/dw term
        grad_w = g_e.mean()
        total_pen += grad_w ** 2
        # d(grad_w^2)/d(logit_i) for i in env e:
        # grad_w = (1/n_e) sum_i (sigmoid(w*logit_i)-y_i)*logit_i , at w=1
        # d(grad_w)/d(logit_i) = (1/n_e) * [pred_i*(1-pred_i)*logit_i + (pred_i - y_i)]
        dgrad_w_dlogit = (pred[mask] * (1 - pred[mask]) * logit[mask]
                          + (pred[mask] - y[mask])) / n_e
        dlogit_pen[mask] = 2 * grad_w * dgrad_w_dlogit
    return total_pen, dlogit_pen, Z, logit, pred


def irm_grad(X, y, p, env_mask_list, lambda_irm=1.0, l2=1e-4):
    N = X.shape[0]
    pen, dlogit_pen, Z, logit, pred = irm_penalty_and_grad(X, y, p, env_mask_list)
    dlogit_bce = (pred - y) / N
    dlogit = dlogit_bce + lambda_irm * dlogit_pen
    grads = {}
    grads["w_main"] = Z.T @ dlogit + l2 * p["w_main"]
    grads["b_main"] = np.array([dlogit.sum()])
    dZ = np.outer(dlogit, p["w_main"])
    dZpre = dZ * (1 - Z ** 2)
    grads["W1"] = X.T @ dZpre + l2 * p["W1"]
    grads["b1"] = dZpre.sum(axis=0)
    loss = bce_loss(pred, y)
    return loss, pen, pred, grads


def cisd_forward(X, p, h1):
    Zpre, Z = forward_encoder(X, p)
    Zs, Zp = Z[:, :h1], Z[:, h1:]
    logit_main = Zs @ p["w_main"] + p["b_main"][0]
    logit_adv = Zs @ p["W_adv"] + p["b_adv"]
    logit_env = Zp @ p["W_env"] + p["b_env"]
    return Z, Zs, Zp, logit_main, logit_adv, logit_env


def cisd_grad(X, y, env_onehot, p, h1, lambda_adv=1.0, lambda_env=0.5,
              lambda_orth=0.1, l2=1e-4):
    N = X.shape[0]
    Z, Zs, Zp, logit_main, logit_adv, logit_env = cisd_forward(X, p, h1)
    pred = sigmoid(logit_main)
    loss_main = bce_loss(pred, y)

    prob_adv = softmax(logit_adv)
    loss_adv = -np.mean(np.sum(env_onehot * np.log(np.clip(prob_adv, 1e-9, 1)), axis=1))
    prob_env = softmax(logit_env)
    loss_env = -np.mean(np.sum(env_onehot * np.log(np.clip(prob_env, 1e-9, 1)), axis=1))

    cross_cov = (Zs.T @ Zp) / N
    loss_orth = np.sum(cross_cov ** 2)

    total = loss_main + lambda_env * loss_env + lambda_orth * loss_orth  # adv handled via GRL below

    # ---- gradients ----
    grads = {}
    dlogit_main = (pred - y) / N
    grads["w_main"] = Zs.T @ dlogit_main + l2 * p["w_main"]
    grads["b_main"] = np.array([dlogit_main.sum()])
    dZs_main = np.outer(dlogit_main, p["w_main"])

    dlogit_adv = (prob_adv - env_onehot) / N          # normal grad for classifier's OWN params
    grads["W_adv"] = Zs.T @ dlogit_adv + l2 * p["W_adv"]
    grads["b_adv"] = dlogit_adv.sum(axis=0)
    dZs_adv_normal = dlogit_adv @ p["W_adv"].T         # normal backprop value into Zs
    dZs_adv_reversed = -lambda_adv * dZs_adv_normal    # GRL: reversed & scaled when feeding encoder

    dlogit_env = lambda_env * (prob_env - env_onehot) / N
    grads["W_env"] = Zp.T @ dlogit_env + l2 * p["W_env"]
    grads["b_env"] = dlogit_env.sum(axis=0)
    dZp_env = dlogit_env @ p["W_env"].T

    dorth_dZs = lambda_orth * (2.0 / N) * (Zp @ cross_cov.T)
    dorth_dZp = lambda_orth * (2.0 / N) * (Zs @ cross_cov)

    dZs_total = dZs_main + dZs_adv_reversed + dorth_dZs
    dZp_total = dZp_env + dorth_dZp

    dZ = np.concatenate([dZs_total, dZp_total], axis=1)
    Zpre_tanh_deriv = (1 - Z ** 2)
    dZpre = dZ * Zpre_tanh_deriv
    grads["W1"] = X.T @ dZpre + l2 * p["W1"]
    grads["b1"] = dZpre.sum(axis=0)

    metrics = {"loss_main": loss_main, "loss_adv": loss_adv,
               "loss_env": loss_env, "loss_orth": loss_orth}
    return total, pred, grads, metrics


def predict_cisd(X, p, h1):
    _, Zs, _, logit_main, _, _ = cisd_forward(X, p, h1)
    return sigmoid(logit_main)


def predict_erm(X, p):
    _, logit = erm_forward(X, p)
    return sigmoid(logit)
