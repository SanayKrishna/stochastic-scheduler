import numpy as np
import pytest

from simulation_engine import pert_mean, run_simulation, sample_pert


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


def _chain_data(n=5000):
    return {
        "project_name": "chain",
        "target_completion_days": 12,
        "simulations": n,
        "tasks": [
            {"id": "A", "name": "A", "optimistic": 2, "most_likely": 3, "pessimistic": 4, "dependencies": []},
            {"id": "B", "name": "B", "optimistic": 2, "most_likely": 3, "pessimistic": 4, "dependencies": ["A"]},
            {"id": "C", "name": "C", "optimistic": 2, "most_likely": 3, "pessimistic": 4, "dependencies": ["B"]},
        ],
    }


def test_linear_chain_p50_sensible():
    r = run_simulation(_chain_data(), seed=1)
    # each task averages 3 days, 3 in series -> ~9
    assert 8.0 < r["p50"] < 10.0
    assert 8.0 < r["mean"] < 10.0
    assert r["prob_on_time"] == pytest.approx(100.0, abs=1.0)


def test_parallel_paths_take_max():
    data = {
        "project_name": "parallel",
        "simulations": 3000,
        "target_completion_days": 10,
        "tasks": [
            {"id": "A", "name": "A", "optimistic": 2, "most_likely": 2, "pessimistic": 2, "dependencies": []},
            {"id": "B", "name": "B", "optimistic": 5, "most_likely": 5, "pessimistic": 5, "dependencies": ["A"]},
            {"id": "C", "name": "C", "optimistic": 1, "most_likely": 1, "pessimistic": 1, "dependencies": ["A"]},
        ],
    }
    r = run_simulation(data, seed=0)
    assert r["p50"] == pytest.approx(7.0, abs=0.01)


def test_bad_estimate_rejected():
    data = _chain_data(n=1000)
    data["tasks"][0]["optimistic"] = 9
    with pytest.raises(ValueError, match="optimistic"):
        run_simulation(data)


def test_cycle_rejected():
    data = _chain_data(n=1000)
    data["tasks"][0]["dependencies"] = ["C"]
    with pytest.raises(ValueError, match="cycle"):
        run_simulation(data)
