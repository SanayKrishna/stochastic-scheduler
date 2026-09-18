"""PERT sampling helpers. No Flask, no AI in this module."""

from scipy.stats import beta


def pert_mean(optimistic, most_likely, pessimistic):
    return (optimistic + 4 * most_likely + pessimistic) / 6.0


def pert_alpha_beta(optimistic, most_likely, pessimistic):
    if pessimistic <= optimistic:
        return None, None
    denom = pessimistic - optimistic
    alpha = 1 + 4 * (most_likely - optimistic) / denom
    b = 1 + 4 * (pessimistic - most_likely) / denom
    return alpha, b


def sample_pert(optimistic, most_likely, pessimistic, size=None, rng=None):
    """Draw from a PERT (scaled Beta) distribution."""
    if pessimistic < most_likely or most_likely < optimistic:
        raise ValueError(
            f"bad three-point estimate O={optimistic} M={most_likely} P={pessimistic}, "
            "need O <= M <= P"
        )
    if pessimistic == optimistic:
        # zero spread, deterministic
        import numpy as np
        if size is None:
            return float(optimistic)
        return np.full(size, float(optimistic))

    alpha, b = pert_alpha_beta(optimistic, most_likely, pessimistic)
    vals = beta.rvs(alpha, b, size=size, random_state=rng)
    return optimistic + vals * (pessimistic - optimistic)
