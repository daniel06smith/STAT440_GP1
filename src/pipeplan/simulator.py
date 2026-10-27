"""Local re-implementation of the Star Kingdom scoring rules.

We cannot see the professor's evaluator, so this module re-creates it from the
written rules. It gives three views of a plan (an ordered list of circles):

1. ``simulate``           -- event-exact: the rules applied to ONE concrete set of
                             future leaks (leak time + leak cost for every pipe).
2. ``monte_carlo``        -- ``simulate`` repeated over randomly drawn futures
                             (leak times drawn from the per-pipe hazard model).
3. ``expected_score``     -- fast approximation used inside the optimiser: it
                             assumes no pipe leaks before its own replacement when
                             working out *when* each circle runs, then adds the
                             exact expected leak cost for those replacement times.

Score = surplus - leak_cost
      = (25 * 10M - total replacement spend) - (cost of leaks before replacement)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .world import (
    ANNUAL_BUDGET, AREA_EXPONENT, BASE_FEE, LINEAR_RATE, N_YEARS, RATE_BY_MASK, World,
)

TOTAL_BUDGET = ANNUAL_BUDGET * N_YEARS


def fixed_cost(surface_rate: float, radius: float) -> float:
    """The part of a work-unit cost that does not depend on which pipes are non-poly:
    100 + (surface_rate * area) ** 0.85."""
    return BASE_FEE + (surface_rate * math.pi * radius * radius) ** AREA_EXPONENT


def work_unit_cost(surface_rate: float, radius: float, nonpoly_metres: float) -> float:
    """Full cost of one work unit, rounded to the cent (professor's formula)."""
    return round(fixed_cost(surface_rate, radius) + LINEAR_RATE * nonpoly_metres, 2)


@dataclass
class Prepared:
    """A plan with its pipe memberships worked out (do this once, simulate many times)."""

    circles: np.ndarray  # (n, 3): x, y, r  -- in execution order
    indptr: np.ndarray  # CSR layout of members: members of circle j are
    indices: np.ndarray  # indices[indptr[j]:indptr[j+1]]
    fixed: np.ndarray  # (n,) 100 + (s*A)^0.85
    surface_rate: np.ndarray  # (n,)
    uncovered_required: np.ndarray  # indices of required pipes in no circle

    @property
    def n(self) -> int:
        return len(self.circles)

    @property
    def covers_all_iron(self) -> bool:
        return len(self.uncovered_required) == 0

    def members(self, j: int) -> np.ndarray:
        return self.indices[self.indptr[j]: self.indptr[j + 1]]


def prepare(world: World, circles: np.ndarray) -> Prepared:
    circles = np.asarray(circles, dtype=float).reshape(-1, 3)
    mem = [world.members(cx, cy, r) for cx, cy, r in circles]
    indptr = np.concatenate([[0], np.cumsum([len(m) for m in mem])]).astype(np.int64)
    indices = np.concatenate(mem) if len(mem) else np.empty(0, dtype=np.int64)
    s = np.array([world.surface_rate(m) for m in mem])
    fixed = np.array([fixed_cost(si, r) for si, r in zip(s, circles[:, 2])])
    covered = np.zeros(world.n, dtype=bool)
    covered[indices] = True
    return Prepared(circles, indptr, indices, fixed, s, np.flatnonzero(world.required & ~covered))


# --------------------------------------------------------------------------
# 1. Event-exact simulation for one concrete future
# --------------------------------------------------------------------------
def simulate(world: World, prep: Prepared, leak_time: np.ndarray | None = None,
             leak_cost: np.ndarray | None = None, annual_budget: float = ANNUAL_BUDGET,
             n_years: int = N_YEARS) -> dict:
    """Apply the professor's evaluation rules to one concrete future.

    leak_time : (n_pipes,) years after 1 Jan 2027 at which each pipe WOULD leak if
                it were still non-poly (inf = never within the window).
    leak_cost : (n_pipes,) cost of that leak (default: the model's mean cost).

    Evaluation (project description):
      * the budget starts at 10M; at the start of each year 2027..2051 add 10M
        (unspent money carries over)
      * work units run strictly in order; a unit that would take the budget below
        zero waits for the next year (and blocks every later unit)
      * a unit with no non-polyurethane pipe left inside is skipped for free
      * unit cost = 100 + (surface_rate * area)^0.85 + 200 * (non-poly metres inside)
        where a pipe that already leaked/was replaced is polyurethane
    """
    if leak_time is None:
        leak_time = np.full(world.n, np.inf)
    if leak_cost is None:
        leak_cost = world.leak_cost
    length, nonpoly0 = world.length, world.nonpoly0
    replaced_at = np.full(world.n, np.inf)  # year in which a work unit replaced the pipe

    budget = 0.0
    spend = 0.0
    idx = 0
    n_exec = n_skip = 0
    exec_year = np.full(prep.n, -1, dtype=np.int64)
    for year in range(n_years):
        budget += annual_budget
        while idx < prep.n:
            m = prep.members(idx)
            # non-poly NOW: non-poly at start, not leaked yet, not replaced by an earlier unit
            alive = nonpoly0[m] & (leak_time[m] >= year) & np.isinf(replaced_at[m])
            if not alive.any():
                n_skip += 1  # nothing left to replace -> skipped, no cost
                idx += 1
                continue
            cost = round(prep.fixed[idx] + LINEAR_RATE * length[m][alive].sum(), 2)
            if cost > budget:
                break  # not enough money this year; wait for next year's budget
            budget -= cost
            spend += cost
            replaced_at[m[alive]] = year
            exec_year[idx] = year
            n_exec += 1
            idx += 1
        if idx >= prep.n:
            break

    finished = idx >= prep.n
    feasible = finished and prep.covers_all_iron
    # a leak counts if it happens inside the 25-year window and before the pipe is replaced
    hit = nonpoly0 & (leak_time < np.minimum(replaced_at, n_years))
    leak_total = float(leak_cost[hit].sum())
    surplus = annual_budget * n_years - spend
    return {
        "feasible": feasible,
        "finished_all_units": finished,
        "covers_all_iron": prep.covers_all_iron,
        "n_uncovered_iron": len(prep.uncovered_required),
        "spend": spend,
        "surplus": surplus,
        "leak_cost": leak_total,
        "n_leaks": int(hit.sum()),
        "score": surplus - leak_total if feasible else -np.inf,
        "last_year": int(exec_year.max()) if n_exec else -1,
        "n_executed": n_exec,
        "n_skipped": n_skip,
        "exec_year": exec_year,
    }


def sample_future(world: World, rng: np.random.Generator) -> np.ndarray:
    """Draw leak times: each non-poly pipe leaks after Exp(lam) years, so that
    P(leaks within y years) = 1 - (1 - p_annual)^y exactly like the risk model."""
    t = np.full(world.n, np.inf)
    a = world.lam > 0
    t[a] = rng.exponential(1.0 / world.lam[a])
    return t


def monte_carlo(world: World, prep: Prepared, n_draws: int = 100, seed: int = 0, **kw) -> dict:
    """Average ``simulate`` over randomly drawn futures. Leak cost is the model mean,
    so the reported spread reflects *when* leaks happen, not how big each one is
    (real leak sizes are heavy-tailed, so the real score is noisier than this)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_draws):
        r = simulate(world, prep, sample_future(world, rng), **kw)
        rows.append((r["score"], r["spend"], r["leak_cost"], r["last_year"], r["n_leaks"]))
    a = np.array(rows)
    names = ["score", "spend", "leak_cost", "last_year", "n_leaks"]
    out = {f"{k}_mean": a[:, i].mean() for i, k in enumerate(names)}
    out.update({f"{k}_sd": a[:, i].std(ddof=1) for i, k in enumerate(names)})
    out["n_draws"] = n_draws
    return out


# --------------------------------------------------------------------------
# 3. Fast approximation (no random leaks affecting the timeline)
# --------------------------------------------------------------------------
def expected_score(world: World, prep: Prepared, annual_budget: float = ANNUAL_BUDGET,
                   n_years: int = N_YEARS) -> dict:
    """Expected score assuming pipes only leak *after* the timeline is fixed.

    1. Walk the circles in order; each non-poly pipe is replaced by the first
       circle containing it, so a circle's linear cost counts only pipes not
       yet replaced.
    2. Because unspent money carries over, a circle whose running total spend is
       `cum` runs in year ``max(0, ceil(cum / 10M) - 1)``.
    3. A pipe replaced at the start of year y has been exposed for y years, so
       its expected leak cost is  K * (1 - (1 - p)^y);  a pipe that is never
       replaced is exposed for all 25 years.
    """
    replaced = np.zeros(world.n, dtype=bool)
    repl_year = np.full(world.n, float(n_years))
    cum = 0.0
    spend = 0.0
    year_of_circle = np.full(prep.n, -1, dtype=np.int64)
    new_len = np.zeros(prep.n)
    for j in range(prep.n):
        m = prep.members(j)
        new = m[world.nonpoly0[m] & ~replaced[m]]
        if len(new) == 0:
            continue  # skipped circle
        cost = round(prep.fixed[j] + LINEAR_RATE * world.length[new].sum(), 2)
        cum += cost
        spend += cost
        y = max(0, math.ceil(cum / annual_budget - 1e-12) - 1)
        year_of_circle[j] = y
        new_len[j] = world.length[new].sum()
        replaced[new] = True
        repl_year[new] = y
    finished = year_of_circle.max() < n_years
    exposure = np.minimum(repl_year, n_years)
    exp_leak = float((world.leak_cost * (1.0 - (1.0 - world.p_annual) ** exposure))[world.nonpoly0].sum())
    feasible = bool(finished and prep.covers_all_iron)
    return {
        "feasible": feasible,
        "spend": spend,
        "surplus": annual_budget * n_years - spend,
        "leak_cost": exp_leak,
        "score": annual_budget * n_years - spend - exp_leak if feasible else -np.inf,
        "last_year": int(year_of_circle.max()),
        "year_of_circle": year_of_circle,
        "repl_year": repl_year,
        "n_circles_used": int((year_of_circle >= 0).sum()),
    }
