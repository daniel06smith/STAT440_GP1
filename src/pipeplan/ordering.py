"""Order the circles: which work unit goes first?

Why order matters
-----------------
* Money arrives at 10M a year and units run strictly in the order we list them,
  so a unit's position decides the YEAR it is dug.
* A pipe that is still in the ground in year y has been exposed to y years of
  leak risk:   expected loss = leak_cost * (1 - (1 - p)^y).
* Total replacement spend is the same whatever the order (every pipe is replaced
  once), so the order only changes the expected leak bill.

Rule used: "Smith's rule" (weighted shortest processing time).
   Run circles in decreasing order of  (expected $ lost per year if left alone)
                                       / (cost to dig it).
This is the optimal order for a single machine when the "weight" of a job is
constant in time. Here the weight is NOT constant -- a pipe that survives
longer has the same hazard, but the probability it is still unreplaced shrinks
-- so we iterate: schedule -> estimate each circle's year -> recompute its
survival-adjusted loss rate  sum_p K_p * lam_p * (1-p_p)^year  -> re-sort.
With a year-by-year risk (World.time_varying) the weight is the expected $ lost by
delaying the circle one year from its scheduled year: sum_p K_p * P(p leaks during that year),
so a pipe whose batch spikes in 2035 is cheap to delay in 2027 but urgent by 2033.
"""
from __future__ import annotations

import numpy as np

from .simulator import Prepared, expected_score, prepare
from .world import ANNUAL_BUDGET, LINEAR_RATE, N_YEARS, World


def permute(prep: Prepared, order: np.ndarray) -> Prepared:
    """Same circles in a new execution order (members are re-packed)."""
    mem = [prep.members(j) for j in order]
    indptr = np.concatenate([[0], np.cumsum([len(m) for m in mem])]).astype(np.int64)
    return Prepared(
        prep.circles[order], indptr, np.concatenate(mem), prep.fixed[order],
        prep.surface_rate[order], prep.uncovered_required,
    )


def _first_cover(world: World, prep: Prepared):
    """For the current order: the pipes each circle actually replaces (first cover wins)."""
    replaced = np.zeros(world.n, dtype=bool)
    new_pipes, cost = [], np.zeros(prep.n)
    for j in range(prep.n):
        m = prep.members(j)
        new = m[world.nonpoly0[m] & ~replaced[m]]
        replaced[new] = True
        new_pipes.append(new)
        cost[j] = (prep.fixed[j] + LINEAR_RATE * world.length[new].sum()) if len(new) else 0.0
    return new_pipes, cost


def _years(cost: np.ndarray, annual_budget: float = ANNUAL_BUDGET) -> np.ndarray:
    cum = np.cumsum(cost)
    return np.maximum(0, np.ceil(cum / annual_budget - 1e-12) - 1)


def smith_order(world: World, prep: Prepared, iters: int = 8, verbose: bool = True):
    """Iterated Smith's rule. Returns (best Prepared in execution order, log).

    iteration 0 is plain Smith (loss rate measured at time 0); later iterations
    re-measure every circle's loss rate at the year it is currently scheduled for.
    """
    n = prep.n
    cur = prep
    year_guess = np.zeros(n)  # estimated execution year of each circle, in `cur` order
    best, best_score = cur, -np.inf
    log = []
    for it in range(iters + 1):
        new_pipes, cost = _first_cover(world, cur)
        rate = np.zeros(n)
        for j, pipes in enumerate(new_pipes):
            if not len(pipes):
                continue
            if world.time_varying:
                # expected $ lost by pushing this circle from its scheduled year to the next one
                rate[j] = world.delay_loss(pipes, int(year_guess[j])).sum()
            else:
                rate[j] = np.sum(world.loss_rate[pipes] * (1 - world.p_annual[pipes]) ** year_guess[j])
        dens = np.where(cost > 0, rate / np.maximum(cost, 1e-9), -1.0)
        order = np.argsort(-dens, kind="stable")
        yrs = _years(cost[order])
        year_guess = (0.5 * year_guess[order] + 0.5 * yrs) if it > 0 else yrs
        cur = permute(cur, order)
        sc = expected_score(world, cur)
        log.append({"iter": it, "score": sc["score"], "spend": sc["spend"],
                    "leak": sc["leak_cost"], "last_year": sc["last_year"]})
        if verbose:
            print(f"iter {it}: score ${sc['score']:,.0f}  spend ${sc['spend']:,.0f}  "
                  f"leak ${sc['leak_cost']:,.0f}  last year {sc['last_year']}")
        if sc["score"] > best_score:
            best, best_score = cur, sc["score"]
    return best, log


def lp_order(world: World, prep: Prepared, annual_budget: float = ANNUAL_BUDGET,
             n_years: int = N_YEARS, verbose: bool = True) -> Prepared:
    """Order circles by solving a year-assignment linear program (for year-by-year risk).

    Smith's rule ranks a circle by the loss of delaying it one year *at its scheduled year*. With a
    spiky hazard that is a trap: once a circle is scheduled after its pipes' spike, delaying it
    further looks cheap (the damage is done), so it never moves back in front of the spike.

    Instead choose each circle's year directly:
        minimise   sum_j sum_y x_jy * L_j(y)          L_j(y) = sum_{pipes p first replaced by j} K_p * P(p leaks before year y)
        subject to sum_y x_jy = 1                    (every circle runs once, within the window)
                   sum_{y' <= y} spend_y' <= B * (y + 1)   (money arrives at B per year, unspent money carries over)
    Circles cost ~$10k against a $10M yearly budget, and a basic LP solution splits at most one
    circle per budget constraint between years, so rounding each circle to its main year loses
    almost nothing. Within a year circles are sorted by urgency (next-year delay loss per dollar)
    so that if a year overflows, the least urgent circles are the ones pushed back.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

    new_pipes, cost = _first_cover(world, prep)
    live = np.flatnonzero(cost > 0)
    n, Y = len(live), n_years
    # L[j, y]: expected leak $ on circle j's pipes if it runs at the start of year y
    lens = np.array([len(new_pipes[j]) for j in live])
    flat = np.concatenate([new_pipes[j] for j in live])
    pl = world.leak_cost[flat, None] * -np.expm1(-world.cum_hazard[flat, :Y])
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    L = np.add.reduceat(pl, starts, axis=0)
    c = cost[live]

    nv = n * Y  # x_jy at column j*Y + y, then Y spend variables s_y
    cols = np.arange(nv)
    A_one = csr_matrix((np.ones(nv), (np.repeat(np.arange(n), Y), cols)), shape=(n, nv + Y))
    # s_y - sum_j c_j x_jy = 0
    A_s = hstack([coo_matrix((-np.repeat(c, Y), (np.tile(np.arange(Y), n), cols)), shape=(Y, nv)),
                  csr_matrix(np.eye(Y))])
    A_eq = vstack([A_one, A_s]).tocsr()
    b_eq = np.r_[np.ones(n), np.zeros(Y)]
    A_ub = hstack([csr_matrix((Y, nv)), csr_matrix(np.tril(np.ones((Y, Y))))]).tocsr()
    b_ub = annual_budget * np.arange(1, Y + 1)
    res = linprog(np.r_[L.ravel(), np.zeros(Y)], A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"ordering LP failed: {res.message}")
    X = res.x[:nv].reshape(n, Y)
    year = X.argmax(axis=1)
    urgency = (L[np.arange(n), np.minimum(year + 1, Y - 1)] - L[np.arange(n), year]) / np.maximum(c, 1e-9)
    order_live = live[np.lexsort((-urgency, year))]
    order = np.r_[order_live, np.flatnonzero(cost <= 0)]  # skipped circles last (dropped later)
    if verbose:
        n_split = int(((X > 1e-6).sum(axis=1) > 1).sum())
        print(f"ordering LP: {n} circles, LP expected leak ${res.fun:,.0f}, {n_split} circles split between years")
    return permute(prep, order)


def _required_margins(world: World, prep: Prepared, j: int):
    """(indices, margins) of the iron pipes inside circle j; margin = r - farthest end point."""
    m = prep.members(j)
    m = m[world.required[m]]
    cx, cy, r = prep.circles[j]
    d = np.maximum(np.hypot(world.x1[m] - cx, world.y1[m] - cy), np.hypot(world.x2[m] - cx, world.y2[m] - cy))
    return m, r - d


def drop_redundant(world: World, prep: Prepared, min_margin: float = 0.04) -> Prepared:
    """Remove circles that are skipped (every non-poly pipe inside was already
    replaced by an earlier circle). They cost nothing but clutter the file.

    Safety guard: a dropped circle is KEPT if some iron pipe would otherwise be
    covered only by circles that contain it by less than `min_margin` metres (a pipe
    can sit a few millimetres inside someone else's circle by accident). A kept
    guard circle is skipped for free by the real rules unless the pipe really was
    not replaced earlier, in which case it replaces it.
    """
    _, cost = _first_cover(world, prep)
    keep = cost > 0
    best = np.full(world.n, -np.inf)
    for j in np.flatnonzero(keep):
        m, mg = _required_margins(world, prep, j)
        best[m] = np.maximum(best[m], mg)
    for j in np.flatnonzero(~keep):
        m, mg = _required_margins(world, prep, j)
        weak = best[m] < min_margin
        if (weak & (mg >= min_margin)).any():
            keep[j] = True
            best[m] = np.maximum(best[m], mg)
    return permute(prep, np.flatnonzero(keep))


# ------------------------------------------------------------------ baselines
def baseline_orders(world: World, prep: Prepared, seed: int = 0) -> dict:
    """Naive orderings to compare against (same circles, different order)."""
    rng = np.random.default_rng(seed)
    n = prep.n
    out = {}
    out["random"] = rng.permutation(n)
    # centroid of the pipes each circle covers
    first = [prep.members(j) for j in range(n)]
    rate_all = np.array([world.loss_rate[m].sum() for m in first])
    cost_all = np.array([prep.fixed[j] + LINEAR_RATE * world.length[m][world.nonpoly0[m]].sum()
                         for j, m in enumerate(first)])
    out["by_x_coordinate"] = np.argsort(prep.circles[:, 0], kind="stable")
    age_like = np.array([world.p_annual[m].max() if len(m) else 0 for m in first])
    out["by_max_leak_probability"] = np.argsort(-age_like, kind="stable")
    out["by_total_loss_rate"] = np.argsort(-rate_all, kind="stable")
    out["by_loss_per_dollar (static Smith)"] = np.argsort(-(rate_all / np.maximum(cost_all, 1e-9)), kind="stable")
    return out
