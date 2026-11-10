"""Circle construction as a weighted SET COVER, solved with an integer program.

Why
---
`construct.Builder` merges circles two at a time and stops when no single pairwise merge pays.
That greedy can get stuck: a group of pipes may be cheapest as one circle even though every
pairwise step towards it costs money, and a merge that was right early can be wrong later.

Here we instead
  1. generate many candidate circles (one per target pipe, the minimum circle of each pipe
     plus its 1..k nearest target pipes, and every circle of an existing plan, if given), and
  2. pick the cheapest set of candidates that covers every target pipe at least once
     (scipy's HiGHS MILP).

Candidate cost = 100 + area term (corrected or original formula, via simulator.fixed_cost, with
the surface rate of ALL pipes wholly inside) + the net cost of any non-target copper/brass pipe the
circle swallows (Builder.inc_net). The 200 $/m of the target pipes is the same for every cover, so it
is left out. Circles may overlap (a pipe in two chosen circles is paid for once; the second circle
still pays its fixed cost), so this is a relaxation of the Builder's partition and, when the
existing plan's circles are included as candidates, can never be worse in spend.

What it ignores: timing. A cover that bundles an urgent pipe with pipes that could wait forces
them to run together. `delay_weight` adds an optional penalty for that (see `_delay_penalty`).
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.spatial import cKDTree

from .geometry import min_enclosing_circle
from .simulator import fixed_cost
from .world import World

CUSHION = 0.05  # default m of spare radius beyond the farthest assigned end point (same as Builder.circles)
MIN_MARGIN = 0.04  # default m a target must be inside a circle by to count as covered (= drop_redundant)


def _final_circle(world: World, group: np.ndarray, cushion: float = CUSHION):
    """Minimum circle of the group's end points, centre rounded to the cent, radius rounded UP to the
    cent with `cushion` spare -- exactly what would be written to the submission file."""
    pts = np.column_stack([
        np.concatenate([world.x1[group], world.x2[group]]),
        np.concatenate([world.y1[group], world.y2[group]]),
    ])
    cx, cy, _ = min_enclosing_circle(pts)
    cx, cy = round(cx, 2), round(cy, 2)
    need = math.sqrt(((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2).max())
    return cx, cy, math.ceil((need + cushion) * 100) / 100


def generate_candidates(world: World, target: np.ndarray, k_max: int = 12, r_max: float = 80.0,
                        neigh_radius: float = 60.0, extra_circles: np.ndarray | None = None,
                        same_surface_k: int = 0, cushion: float = CUSHION, verbose: bool = True) -> np.ndarray:
    """Candidate circles (x, y, r): solo circle per target pipe, the circle of each target pipe plus
    its 1..k_max nearest target pipes (by mid point, within neigh_radius), the same with only
    target pipes of the SAME surface (1..same_surface_k; these avoid adding a second surface rate),
    and extra_circles."""
    t0 = time.time()
    seen: set = set()
    out = []
    tall = np.flatnonzero(target)
    pools = [(tall, k_max)]
    if same_surface_k:
        pools += [(tall[world.surf_bit[tall] == b], same_surface_k) for b in np.unique(world.surf_bit[tall])]
    for tidx, kk in pools:
        _grow(world, tidx, kk, r_max, neigh_radius, seen, out, t0, verbose, cushion)
    if extra_circles is not None:
        for c in np.asarray(extra_circles).reshape(-1, 3):
            key = tuple(round(float(v), 2) for v in c)
            if key not in seen:
                seen.add(key)
                out.append(key)
    if verbose:
        print(f"  {len(out)} distinct candidate circles ({time.time() - t0:.0f}s)")
    return np.array(out)


def _grow(world, tidx, k_max, r_max, neigh_radius, seen, out, t0, verbose, cushion):
    """Add the circle of each pipe in tidx plus its 1..k_max nearest pipes of tidx."""
    tree = cKDTree(np.column_stack([world.mx[tidx], world.my[tidx]]))
    k = min(k_max + 1, len(tidx))
    dist, nn = tree.query(np.column_stack([world.mx[tidx], world.my[tidx]]), k=k,
                          distance_upper_bound=neigh_radius)
    dist, nn = dist.reshape(len(tidx), -1), nn.reshape(len(tidx), -1)

    def add(c):
        key = (round(c[0], 2), round(c[1], 2), round(c[2], 2))
        if key not in seen:
            seen.add(key)
            out.append(key)

    for a in range(len(tidx)):
        group = [tidx[a]]
        add(_final_circle(world, np.array(group), cushion))
        last_r = None
        for d, b in zip(dist[a, 1:], nn[a, 1:]):
            if not np.isfinite(d):
                break
            group.append(tidx[b])
            c = _final_circle(world, np.array(group), cushion)
            if c[2] > r_max:
                break
            if c != last_r:
                add(c)
                last_r = c
        if verbose and a % 10000 == 0:
            print(f"  candidates: {a:6d}/{len(tidx)} pipes, {len(out):7d} circles ({time.time() - t0:.0f}s)")


def candidate_matrix(world: World, target: np.ndarray, circles: np.ndarray, inc_net: np.ndarray,
                     min_margin: float = MIN_MARGIN):
    """Coverage matrix (targets x candidates, CSC) and cost vector.

    A target only counts as covered if it is inside with >= min_margin metres to spare (the same
    safety rule as ordering.drop_redundant), so the plan never hinges on rounding in the checker.
    The cost uses every pipe wholly inside (no margin), as the real rules do."""
    tpos = np.full(world.n, -1, dtype=np.int64)
    tidx = np.flatnonzero(target)
    tpos[tidx] = np.arange(len(tidx))
    rows, cols, cost = [], [], np.empty(len(circles))
    for j, (cx, cy, r) in enumerate(circles):
        m = world.members(cx, cy, r)
        cost[j] = fixed_cost(world.surface_rate(m), r) + inc_net[m].sum()
        t = tpos[world.members(cx, cy, r - min_margin)]
        t = t[t >= 0]
        rows.append(t)
        cols.append(np.full(len(t), j, dtype=np.int64))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    A = sparse.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(tidx), len(circles)))
    return A, cost, tidx


def solve_cover(A, cost: np.ndarray, time_limit: float = 600.0, mip_gap: float = 1e-4,
                verbose: bool = True) -> np.ndarray:
    """Minimum-cost set cover (every row covered >= 1) -> boolean mask of chosen candidates."""
    # a candidate that covers nothing can only cost money
    useful = np.asarray(A.sum(axis=0)).ravel() > 0
    Au = A[:, useful].tocsr()
    res = milp(
        c=cost[useful],
        constraints=LinearConstraint(Au, lb=np.ones(Au.shape[0]), ub=np.inf),
        integrality=np.ones(Au.shape[1]),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit, "mip_rel_gap": mip_gap, "disp": verbose},
    )
    if res.x is None:
        raise RuntimeError(f"set cover failed: {res.message}")
    chosen = np.zeros(len(cost), dtype=bool)
    chosen[np.flatnonzero(useful)[res.x > 0.5]] = True
    if verbose:
        print(f"  MILP: {res.message}  objective ${res.fun:,.0f}  ({chosen.sum()} circles)")
    return chosen


def delay_penalty(world: World, A, tidx: np.ndarray, pref_year: np.ndarray, lam: float,
                  verbose: bool = True) -> np.ndarray:
    """Timing penalty ($) for every candidate: what bundling its pipes into one work unit costs.

    pref_year[i] = the project year (0 = 2027) pipe i would be replaced in a reference plan.
    A circle runs once, in some year t. Pipes it covers that wanted an EARLIER year are delayed
    (expected loss (K - 200 L) * P(leak between their year and t), the leak-aware delay cost, floored
    at 0); pipes that wanted a LATER year are paid for early, which costs `lam` $ per $ of their 200 $/m
    per year pulled forward (budget spent early delays someone else). The circle is credited with the
    best t among its pipes' preferred years. lam = 0 makes every penalty 0 (run at the earliest year).
    """
    t0 = time.time()
    y_all = np.clip(pref_year[tidx].astype(np.int64), 0, 25)
    F = world.leak_prob_by  # F(years, idx): P(leak before the start of that project year)
    gain = np.maximum(world.leak_cost[tidx] - 200.0 * world.length[tidx], 0.0)
    lin = 200.0 * world.length[tidx]
    pen = np.zeros(A.shape[1])
    for j in range(A.shape[1]):
        rows = A.indices[A.indptr[j]:A.indptr[j + 1]]
        if len(rows) < 2:
            continue
        y = y_all[rows]
        ts = np.unique(y)
        if len(ts) == 1:
            continue
        g, li, gi = gain[rows], lin[rows], tidx[rows]
        best = np.inf
        for t in ts:
            late = y < t
            cost = lam * (li[~late] * (y[~late] - t)).sum()
            if late.any():
                cost += (g[late] * (F(np.full(late.sum(), t), gi[late]) - F(y[late], gi[late]))).sum()
            best = min(best, cost)
        pen[j] = best
    if verbose:
        print(f"  delay penalty: {np.count_nonzero(pen)} candidates penalised, total ${pen.sum():,.0f} "
              f"({time.time() - t0:.0f}s)")
    return pen


def build_circles(world: World, params=None, k_max: int = 25, same_surface_k: int = 25, r_max: float = 150.0,
                  extra_circles: np.ndarray | None = None, time_limit: float = 900.0,
                  cushion: float = CUSHION, min_margin: float = MIN_MARGIN,
                  pref_year: np.ndarray | None = None, lam: float = 0.0,
                  verbose: bool = True) -> np.ndarray:
    """Whole set-cover construction: same target choice (iron + paying copper/brass) and the same
    'swallowed copper/brass' charge as construct.Builder, then candidates + MILP. Returns (x, y, r).

    pref_year + lam > 0 add `delay_penalty` to every candidate's cost (time-aware cover)."""
    from .construct import Builder, Params

    b = Builder(world, params or Params(verbose=False))
    cands = generate_candidates(world, b.target, k_max=k_max, same_surface_k=same_surface_k, r_max=r_max,
                                neigh_radius=r_max, extra_circles=extra_circles, cushion=cushion,
                                verbose=verbose)
    A, cost, tidx = candidate_matrix(world, b.target, cands, b.inc_net, min_margin=min_margin)
    if pref_year is not None and lam > 0:
        cost = cost + delay_penalty(world, A, tidx, pref_year, lam, verbose=verbose)
    chosen = solve_cover(A, cost, time_limit=time_limit, verbose=verbose)
    return cands[chosen]
