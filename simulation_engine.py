"""Monte Carlo schedule simulation with PERT sampling. No Flask, no AI here."""

import networkx as nx
import numpy as np
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
        if size is None:
            return float(optimistic)
        return np.full(size, float(optimistic))

    alpha, b = pert_alpha_beta(optimistic, most_likely, pessimistic)
    vals = beta.rvs(alpha, b, size=size, random_state=rng)
    return optimistic + vals * (pessimistic - optimistic)


def validate_input(data):
    if not isinstance(data, dict):
        raise ValueError("input must be a JSON object")
    for field in ("project_name", "tasks"):
        if field not in data:
            raise ValueError(f"missing required field: {field}")
    tasks = data["tasks"]
    if not isinstance(tasks, list) or len(tasks) == 0:
        raise ValueError("tasks must be a non-empty list")

    ids = set()
    for i, t in enumerate(tasks):
        for field in ("id", "name", "optimistic", "most_likely", "pessimistic", "dependencies"):
            if field not in t:
                raise ValueError(f"task at index {i} is missing '{field}'")
        tid = t["id"]
        if tid in ids:
            raise ValueError(f"duplicate task id: {tid}")
        ids.add(tid)
        try:
            o = float(t["optimistic"])
            m = float(t["most_likely"])
            p = float(t["pessimistic"])
        except (TypeError, ValueError):
            raise ValueError(f"task {tid}: O/M/P must be numbers")
        if not (o <= m <= p):
            raise ValueError(f"task {tid}: need optimistic <= most_likely <= pessimistic")
        if not isinstance(t["dependencies"], list):
            raise ValueError(f"task {tid}: dependencies must be a list")

    for t in tasks:
        for dep in t["dependencies"]:
            if dep not in ids:
                raise ValueError(f"task {t['id']}: unknown dependency '{dep}'")
            if dep == t["id"]:
                raise ValueError(f"task {t['id']}: cannot depend on itself")

    sims = data.get("simulations", 10000)
    try:
        sims = int(sims)
    except (TypeError, ValueError):
        raise ValueError("simulations must be an integer")
    if sims < 1000 or sims > 50000:
        raise ValueError("simulations must be between 1000 and 50000")
    data["simulations"] = sims

    if "target_completion_days" in data and data["target_completion_days"] is not None:
        try:
            data["target_completion_days"] = float(data["target_completion_days"])
        except (TypeError, ValueError):
            raise ValueError("target_completion_days must be a number")

    return data


def build_graph(tasks):
    g = nx.DiGraph()
    for t in tasks:
        g.add_node(t["id"])
    for t in tasks:
        for dep in t["dependencies"]:
            g.add_edge(dep, t["id"])
    if not nx.is_directed_acyclic_graph(g):
        try:
            cycle = nx.find_cycle(g, orientation="original")
            involved = sorted({u for u, v, _ in cycle} | {v for u, v, _ in cycle})
        except Exception:
            involved = sorted(g.nodes)
        raise ValueError(f"dependency cycle detected involving: {', '.join(involved)}")
    return g


def run_simulation(data, seed=None):
    data = validate_input(dict(data))
    tasks = data["tasks"]
    n_sims = data["simulations"]
    target = data.get("target_completion_days")

    g = build_graph(tasks)
    topo = list(nx.topological_sort(g))
    by_id = {t["id"]: t for t in tasks}
    leaf_nodes = [n for n in g.nodes if g.out_degree(n) == 0]

    rng = np.random.default_rng(seed)
    # precompute beta params per task in topo order
    params = []
    for tid in topo:
        t = by_id[tid]
        o, m, p = float(t["optimistic"]), float(t["most_likely"]), float(t["pessimistic"])
        a, b = pert_alpha_beta(o, m, p)
        params.append((tid, o, m, p, a, b))

    completions = np.empty(n_sims)
    task_samples = {tid: np.empty(n_sims) for tid in topo}

    for i in range(n_sims):
        finish = {}
        for tid, o, m, p, a, b in params:
            if a is None:
                dur = o
            else:
                dur = o + beta.rvs(a, b, random_state=rng) * (p - o)
            task_samples[tid][i] = dur
            preds = list(g.predecessors(tid))
            start = max((finish[pred] for pred in preds), default=0.0)
            finish[tid] = start + dur
        completions[i] = max(finish[n] for n in leaf_nodes)

    p50, p70, p80, p90, p95 = np.percentile(completions, [50, 70, 80, 90, 95])
    mean = float(np.mean(completions))
    std = float(np.std(completions))
    if target is not None:
        prob_on_time = float(np.mean(completions <= target) * 100.0)
    else:
        prob_on_time = None

    comp_std = float(np.std(completions))
    stats = []
    for tid in topo:
        t = by_id[tid]
        s = task_samples[tid]
        s_mean = float(np.mean(s))
        s_std = float(np.std(s))
        if comp_std == 0 or float(np.std(s)) == 0:
            corr = 0.0
        else:
            c = np.corrcoef(s, completions)[0, 1]
            corr = float(c) if not np.isnan(c) else 0.0
        stats.append({
            "id": tid,
            "name": t["name"],
            "optimistic": float(t["optimistic"]),
            "most_likely": float(t["most_likely"]),
            "pessimistic": float(t["pessimistic"]),
            "pert_mean": float(pert_mean(float(t["optimistic"]), float(t["most_likely"]), float(t["pessimistic"]))),
            "sim_mean": s_mean,
            "sim_std": s_std,
            "risk_contribution": corr,
        })

    ranked = sorted(stats, key=lambda d: d["risk_contribution"], reverse=True)
    risk_drivers = ranked[:5]

    return {
        "project_name": data["project_name"],
        "project_start_date": data.get("project_start_date"),
        "target_completion_days": target,
        "simulations": n_sims,
        "p50": float(p50),
        "p70": float(p70),
        "p80": float(p80),
        "p90": float(p90),
        "p95": float(p95),
        "mean": mean,
        "std": std,
        "prob_on_time": prob_on_time,
        "task_stats": stats,
        "risk_drivers": risk_drivers,
        "completions": completions.tolist(),
    }
