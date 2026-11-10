"""Scoring simulator with TRIGGERED leaks (cascading / clustered), on top of simulator.simulate's rules.

World leak times are PRIMARY leaks only (e.g. leak_risk_batchwave_primary.csv). Each year:
  1. work units run at the start of the year (same rules as simulator.simulate: strict order, money carries over,
     an all-polyurethane unit is skipped, 200 $/m only for pipes still non-poly),
  2. the year's primary leaks happen in time order; a primary leak at a pipe that is still non-poly costs its leak
     cost, and every other still-non-poly pipe j becomes a victim with probability TriggerModel.probs(s, j) and
     leaks at the same instant (victims do not trigger further leaks).
So replacing a pipe removes both its own leak and the leaks it would have triggered, and replacing its neighbours
removes the victims it could reach -- which an independent-leak model cannot credit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import contagion as ct
from .simulator import Prepared, sample_costs, sample_future
from .world import ANNUAL_BUDGET, LINEAR_RATE, N_YEARS, REPO_ROOT, World


class ContagionWorld:
    """Pipe attributes, components and the trigger model, in World pipe order."""

    def __init__(self, world: World, tm: ct.TriggerModel, root: Path | None = None, pipes: pd.DataFrame | None = None):
        if pipes is None:
            root = Path(root) if root else REPO_ROOT
            pipes = pd.read_csv(root / "data/raw/pipes.csv", parse_dates=["Lay date"])
        p = pipes.set_index("Pipe ID").loc[world.pipe_id].reset_index()
        self.A = ct.pipe_attributes(p)
        self.A["mat"] = np.where(self.A["mat"] < 0, 0, self.A["mat"])  # poly pipes are never candidates anyway
        self.comp = ct.components(p)
        self.tm = tm
        self.vf = tm.victim_factor(self.A, np.arange(world.n))  # per-pipe victim factor, precomputed

    def victims(self, s: int, alive: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        j = np.flatnonzero(alive)
        if not len(j):
            return j
        q = self.tm.probs(self.A, self.comp, s, j, vf=self.vf[j])
        return j[rng.random(len(j)) < q]

    def expected_victims(self, s: int, alive: np.ndarray) -> float:
        j = np.flatnonzero(alive)
        return float(self.tm.probs(self.A, self.comp, s, j, vf=self.vf[j]).sum()) if len(j) else 0.0


def load_trigger_model(root: Path | None = None) -> ct.TriggerModel:
    import json
    root = Path(root) if root else REPO_ROOT
    j = json.loads((root / "data/processed/trigger_model.json").read_text())
    kern = ct.Kernel(np.array(j["kernel_q"]), np.zeros((len(j["kernel_q"]), 2)), np.zeros((len(j["kernel_q"]), 2)))
    arr = lambda k: np.array(j[k])
    return ct.TriggerModel(kern, j["b_const"], arr("b_vmat"), arr("b_vsurf"), arr("b_vlay"), arr("b_smat"),
                           arr("b_slay"), j["b_same_mat"], j["b_same_year"], j["b_same_group"])


def simulate_events(world: World, prep: Prepared, cw: ContagionWorld, primary_time: np.ndarray, leak_cost: np.ndarray,
                    rng: np.random.Generator, annual_budget: float = ANNUAL_BUDGET, n_years: int = N_YEARS) -> dict:
    length, nonpoly0 = world.length, world.nonpoly0
    replaced = np.zeros(world.n, bool)
    leaked = np.zeros(world.n, bool)
    budget = spend = leak_total = 0.0
    n_primary = n_victims = 0
    idx = 0
    exec_year = np.full(prep.n, -1, dtype=np.int64)
    order = np.argsort(primary_time, kind="stable")
    pt_sorted = primary_time[order]
    for year in range(n_years):
        budget += annual_budget
        while idx < prep.n:
            m = prep.members(idx)
            alive_m = m[nonpoly0[m] & ~leaked[m] & ~replaced[m]]
            if not len(alive_m):
                idx += 1
                continue
            cost = round(prep.fixed[idx] + LINEAR_RATE * length[alive_m].sum(), 2)
            if cost > budget:
                break
            budget -= cost
            spend += cost
            replaced[alive_m] = True
            exec_year[idx] = year
            idx += 1
        lo, hi = np.searchsorted(pt_sorted, [year, year + 1])
        for s in order[lo:hi]:
            if not nonpoly0[s] or leaked[s] or replaced[s]:
                continue
            leaked[s] = True
            leak_total += leak_cost[s]
            n_primary += 1
            alive = nonpoly0 & ~leaked & ~replaced
            v = cw.victims(s, alive, rng)
            if len(v):
                leaked[v] = True
                leak_total += leak_cost[v].sum()
                n_victims += len(v)
        if idx >= prep.n and year >= 0:
            # nothing left to replace: the remaining leaks can only hit pipes outside every circle (none for iron)
            alive_any = nonpoly0 & ~leaked & ~replaced
            if not alive_any.any():
                break
    finished = idx >= prep.n
    feasible = finished and prep.covers_all_iron
    surplus = annual_budget * n_years - spend
    return {"feasible": feasible, "spend": spend, "leak_cost": leak_total, "score": surplus - leak_total if feasible else -np.inf,
            "n_primary": n_primary, "n_victims": n_victims, "last_year": int(exec_year.max())}


def monte_carlo_events(world: World, prep: Prepared, cw: ContagionWorld, n_draws: int = 100, seed: int = 0,
                       cost_ratios: dict | None = None, keep_draws: bool = False) -> dict:
    """Same seed -> same primary futures and the same uniform draws for every plan (paired comparisons)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_draws):
        t = sample_future(world, rng)
        cost = sample_costs(world, cost_ratios, rng) if cost_ratios is not None else world.leak_cost
        sub = np.random.default_rng(rng.integers(2**63))
        r = simulate_events(world, prep, cw, t, cost, sub)
        rows.append((r["score"], r["spend"], r["leak_cost"], r["n_primary"], r["n_victims"], r["last_year"]))
    a = np.array(rows)
    names = ["score", "spend", "leak_cost", "n_primary", "n_victims", "last_year"]
    out = {f"{k}_mean": float(a[:, i].mean()) for i, k in enumerate(names)}
    out.update({f"{k}_sd": float(a[:, i].std(ddof=1)) for i, k in enumerate(names)})
    out["score_p05"], out["score_p95"] = (float(x) for x in np.percentile(a[:, 0], [5, 95]))
    out["n_draws"] = n_draws
    if keep_draws:
        out["score_draws"] = a[:, 0]
    return out
