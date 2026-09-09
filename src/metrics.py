import numpy as np


def auroc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # handle ties by average rank
    sorted_p = p[order]
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            avg_rank = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def auprc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    order = np.argsort(-p)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    recall = tp / n_pos
    precision = tp / (tp + fp + 1e-12)
    # trapezoid integration over recall
    ap = 0.0
    prev_recall = 0.0
    for r, prec in zip(recall, precision):
        ap += prec * (r - prev_recall)
        prev_recall = r
    return float(ap)


def f1_at_threshold(y, p, thresh=0.5):
    y = np.asarray(y)
    pred = (np.asarray(p) >= thresh).astype(int)
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def brier_score(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y, p, n_bins=10):
    y = np.asarray(y)
    p = np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / N) * abs(acc - conf)
    return float(ece)


def all_metrics(y, p):
    return {
        "auroc": auroc(y, p),
        "auprc": auprc(y, p),
        "f1": f1_at_threshold(y, p),
        "brier": brier_score(y, p),
        "ece": expected_calibration_error(y, p),
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
    }
