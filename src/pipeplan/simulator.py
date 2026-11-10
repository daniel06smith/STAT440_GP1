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

PROJECT_YEARS = N_YEARS
TOTAL_BUDGET = ANNUAL_BUDGET * N_YEARS


# Two readings of the area term exist:
#   "original"  : (surface_rate * area) ** 0.85   <- the PDF we were given (printed 5 Oct)
#   "corrected" : surface_rate * area ** 0.85     <- what Wendy's notebook says the corrected
#                                                    project instructions state
# Our simulator reproduces Wendy's reported replacement cost to the cent under "corrected".
# The website's data page was checked on 28 Oct 2026 (by Qusai) and states the ORIGINAL
# wording, so "original" is the default and the one all submissions use. "corrected" is kept
# only so Wendy's cost numbers can be reproduced.
COST_FORMULA = "original"


def set_cost_formula(name: str) -> None:
    """Choose the area-term formula for everything in this process ('original'|'corrected')."""
    global COST_FORMULA
    if name not in ("original", "corrected"):
        raise ValueError("formula must be 'original' or 'corrected'")
    COST_FORMULA = name


def fixed_cost(surface_rate: float, radius: float) -> float:
    """The part of a work-unit cost that does not depend on which pipes are non-poly:
    100 + (surface_rate * area) ** 0.85   (original)   or   100 + surface_rate * area ** 0.85."""
    area = math.pi * radius * radius
    if COST_FORMULA == "corrected":
        return BASE_FEE + surface_rate * area**AREA_EXPONENT
    return BASE_FEE + (surface_rate * area) ** AREA_EXPONENT


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


def simulate_project(world: World, prep: Prepared,
                     leak_time: np.ndarray | None = None,
                     leak_cost: np.ndarray | None = None,
                     annual_budget: float = ANNUAL_BUDGET) -> dict:
    """Simulate using the complete 25-year project window.

    A backtest may score a shorter historical period, but that scoring horizon
    must not shorten the replacement project's budget window.
    """
    return simulate(
        world,
        prep,
        leak_time=leak_time,
        leak_cost=leak_cost,
        annual_budget=annual_budget,
        n_years=PROJECT_YEARS,
    )


def sample_future(world: World, rng: np.random.Generator) -> np.ndarray:
    """Draw leak times (years after 1 Jan 2027; inf = not within the window).

    Constant risk: each non-poly pipe leaks after Exp(lam) years, so that
    P(leaks within y years) = 1 - (1 - p_annual)^y exactly like the risk model.
    Year-by-year risk: the hazard is constant within each year, so draw E ~ Exp(1) and find
    where the cumulative hazard crosses E (inverse-transform sampling)."""
    t = np.full(world.n, np.inf)
    if not world.time_varying:
        a = world.lam > 0
        t[a] = rng.exponential(1.0 / world.lam[a])
        return t
    H = world.cum_hazard
    a = H[:, -1] > 0
    E = rng.exponential(1.0, size=int(a.sum()))
    Ha = H[a]
    y = (Ha[:, 1:] < E[:, None]).sum(axis=1)  # number of whole years survived
    inside = y < Ha.shape[1] - 1
    rows = np.flatnonzero(inside)
    yy = y[inside]
    lam_y = Ha[rows, yy + 1] - Ha[rows, yy]
    ta = np.full(len(E), np.inf)
    ta[rows] = yy + (E[rows] - Ha[rows, yy]) / lam_y
    t[a] = ta
    return t


def empirical_cost_ratios(root=None) -> dict[int, np.ndarray]:
    """Real 2019-2026 leak costs divided by their surface's mean cost, per surface code.

    Each pool has mean exactly 1, so multiplying a pipe's model mean cost by a draw from its
    surface's pool keeps the expected cost unchanged and adds the real heavy tail (the top 100
    of 14,113 leaks are ~40% of all leak $). Using surface-only pools slightly overstates the
    noise, since material/length explain a small part of the spread.
    """
    import pandas as pd

    from .world import REPO_ROOT, SURFACES

    root = REPO_ROOT if root is None else root
    pipes = pd.read_csv(f"{root}/data/raw/pipes.csv", usecols=["Pipe ID", "Surface"])
    leaks = pd.read_csv(f"{root}/data/raw/train.csv", usecols=["Pipe ID", "Cost"]).merge(pipes, on="Pipe ID")
    return {SURFACES.index(s): (g["Cost"] / g["Cost"].mean()).to_numpy() for s, g in leaks.groupby("Surface")}


def sample_costs(world: World, ratios: dict[int, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """One heavy-tailed draw of every pipe's leak cost: model mean x a real cost ratio."""
    code = np.log2(world.surf_bit).astype(np.int64)
    mult = np.ones(world.n)
    for c, pool in ratios.items():
        m = code == c
        mult[m] = rng.choice(pool, size=int(m.sum()))
    return world.leak_cost * mult


def monte_carlo(world: World, prep: Prepared, n_draws: int = 100, seed: int = 0,
                cost_ratios: dict[int, np.ndarray] | None = None, keep_draws: bool = False, **kw) -> dict:
    """Average ``simulate`` over randomly drawn futures.

    cost_ratios=None: every leak costs the model mean, so the spread only reflects *when* leaks
    happen and badly understates the real noise. Pass ``empirical_cost_ratios()`` to also draw how
    big each leak is from the real cost distribution (same expected score, realistic spread).
    The same seed gives the same futures for every plan (common random numbers), so per-draw
    differences between two plans (keep_draws=True -> out["score_draws"]) are paired.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_draws):
        t = sample_future(world, rng)
        cost = sample_costs(world, cost_ratios, rng) if cost_ratios is not None else None
        r = simulate(world, prep, t, leak_cost=cost, **kw)
        rows.append((r["score"], r["spend"], r["leak_cost"], r["last_year"], r["n_leaks"]))
    a = np.array(rows)
    names = ["score", "spend", "leak_cost", "last_year", "n_leaks"]
    out = {f"{k}_mean": a[:, i].mean() for i, k in enumerate(names)}
    out.update({f"{k}_sd": a[:, i].std(ddof=1) for i, k in enumerate(names)})
    out["score_p05"], out["score_p95"] = np.percentile(a[:, 0], [5, 95])
    out["n_draws"] = n_draws
    if keep_draws:
        out["score_draws"] = a[:, 0]
    return out


# --------------------------------------------------------------------------
# 3. Fast approximation (no random leaks affecting the timeline)
# --------------------------------------------------------------------------
def survival(world: World, idx: np.ndarray, years) -> np.ndarray:
    """P(pipe has NOT leaked before the start of project year `years`) (scalar or per-pipe years)."""
    y = np.clip(np.asarray(years, dtype=np.int64), 0, N_YEARS)
    if world.time_varying:
        return np.exp(-world.cum_hazard[idx, y])
    return (1.0 - world.p_annual[idx]) ** y


def expected_unit_cost(world: World, fixed: float, pipes: np.ndarray, year: int) -> float:
    """Expected cost of a work unit run at the start of `year`, counting leaks before then.

    A pipe that leaked first is polyurethane: no 200 $/m for it, and the unit is skipped for free
    when every one of its pipes leaked.  E = F * P(some pipe still non-poly) + 200 * sum len * S_p(year).
    """
    if not len(pipes):
        return 0.0
    S = survival(world, pipes, year)
    return float(fixed * -np.expm1(np.log1p(-np.minimum(S, 1 - 1e-15)).sum())
                 + LINEAR_RATE * (world.length[pipes] * S).sum())


def expected_score(world: World, prep: Prepared, annual_budget: float = ANNUAL_BUDGET,
                   n_years: int = N_YEARS, leak_aware: bool = False) -> dict:
    """Expected score assuming pipes only leak *after* the timeline is fixed.

    1. Walk the circles in order; each non-poly pipe is replaced by the first
       circle containing it, so a circle's linear cost counts only pipes not
       yet replaced.
    2. Because unspent money carries over, a circle whose running total spend is
       `cum` runs in year ``max(0, ceil(cum / 10M) - 1)``.
    3. A pipe replaced at the start of year y has been exposed for y years, so
       its expected leak cost is  K * P(leak in the first y years)  -- K * (1 - (1 - p)^y)
       for a constant risk, K * (1 - exp(-cum_hazard[y])) for a year-by-year risk;
       a pipe that is never replaced is exposed for all 25 years.

    leak_aware=True: a circle's cost is its EXPECTED cost at its year (``expected_unit_cost``: pipes
    that leaked first cost no 200 $/m, an all-leaked circle is skipped), and its year comes from the
    cumulative expected spend. This matches the Monte Carlo (the default over-counts spend by ~$30M
    for the batchwave plans). ``last_year_no_leaks`` is always the finishing year if NOTHING leaks,
    i.e. the worst case for the 25-year rule.
    """
    replaced = np.zeros(world.n, dtype=bool)
    repl_year = np.full(world.n, float(n_years))
    cum = 0.0
    cum_full = 0.0
    spend = 0.0
    year_of_circle = np.full(prep.n, -1, dtype=np.int64)
    new_len = np.zeros(prep.n)

    def year_at(total):
        return max(0, math.ceil(total / annual_budget - 1e-12) - 1)

    for j in range(prep.n):
        m = prep.members(j)
        new = m[world.nonpoly0[m] & ~replaced[m]]
        if len(new) == 0:
            continue  # skipped circle
        full = round(prep.fixed[j] + LINEAR_RATE * world.length[new].sum(), 2)
        cum_full += full
        if leak_aware:
            y0 = year_at(cum + 1e-6)
            cost = expected_unit_cost(world, prep.fixed[j], new, y0)
            y = year_at(cum + cost)
            if y != y0:
                cost = expected_unit_cost(world, prep.fixed[j], new, y)
        else:
            cost = full
        cum += cost
        spend += cost
        y = year_at(cum)
        year_of_circle[j] = y
        new_len[j] = world.length[new].sum()
        replaced[new] = True
        repl_year[new] = y
    last_year_no_leaks = year_at(cum_full)
    finished = year_of_circle.max() < n_years and last_year_no_leaks < n_years
    exposure = np.minimum(repl_year, n_years)
    if world.time_varying:
        p_leak = world.leak_prob_by(exposure.astype(np.int64))
    else:
        p_leak = 1.0 - (1.0 - world.p_annual) ** exposure
    exp_leak = float((world.leak_cost * p_leak)[world.nonpoly0].sum())
    feasible = bool(finished and prep.covers_all_iron)
    return {
        "feasible": feasible,
        "spend": spend,
        "surplus": annual_budget * n_years - spend,
        "leak_cost": exp_leak,
        "score": annual_budget * n_years - spend - exp_leak if feasible else -np.inf,
        "last_year": int(year_of_circle.max()),
        "last_year_no_leaks": int(last_year_no_leaks),
        "year_of_circle": year_of_circle,
        "repl_year": repl_year,
        "n_circles_used": int((year_of_circle >= 0).sum()),
    }
