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
"""
from __future__ import annotations

import numpy as np

from .simulator import Prepared, expected_score, prepare
from .world import ANNUAL_BUDGET, LINEAR_RATE, World


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
            if len(pipes):
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
