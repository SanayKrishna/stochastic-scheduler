import numpy as np

from simulation_engine import pert_mean, sample_pert


def test_pert_mean_close_to_formula():
    # O=2, M=3, P=7 -> mean = (2 + 12 + 7) / 6 = 3.5
    assert pert_mean(2, 3, 7) == 3.5

    rng = np.random.default_rng(0)
    # scipy beta.rvs doesn't take Generator directly in all versions,
    # so draw in chunks with integer seeds derived from rng
    draws = []
    for _ in range(10):
        seed = int(rng.integers(0, 2**31 - 1))
        draws.append(sample_pert(2, 3, 7, size=10000, rng=seed))
    samples = np.concatenate(draws)  # 100k total
    assert abs(float(np.mean(samples)) - 3.5) < 0.05


def test_zero_spread_returns_constant():
    assert sample_pert(5, 5, 5) == 5.0
