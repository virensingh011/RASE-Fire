import numpy as np


def per_event_mean(values, event_ids):
    values, event_ids = np.asarray(values, float), np.asarray(event_ids)
    return np.array([values[event_ids == e].mean() for e in np.unique(event_ids)])


def bootstrap_ci(per_event, n_boot=2000, seed=0, alpha=0.05):
    """Resample FIRES (events), not pixels: pixels of one fire are strongly correlated."""
    a = np.asarray(per_event, float)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(1)
    return float(a.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a, b, n_boot=5000, seed=0):
    """a, b: per-event scores of two models on the SAME events. Returns mean diff, 95% CI, two-sided p."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), float(min(1.0, p))
