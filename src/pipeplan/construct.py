"""Circle construction: turn ~27,000 pipes into a few thousand work units.

Idea
----
Start with one tiny circle per pipe (the minimum circle through its two end
points). Then repeatedly MERGE nearby circles into one bigger circle whenever
that lowers the total expected cost. This is agglomerative ("bottom-up")
clustering, but with the project's real cost function as the merge criterion
instead of a distance.

Why merging can pay off
    * every work unit costs a flat $100 -> fewer units, fewer fees
    * the area term (surface_rate * area)^0.85 grows SUB-linearly in area,
      so one big circle is cheaper than many small ones covering the same pipes
Why merging can hurt
    * a bigger circle may swallow an expensive surface (water = $32, structure
      = $50 per m^2) and the surface rates of ALL pipes inside are summed
    * a merged circle is executed as one lump: if it bundles a pipe that is
      likely to leak soon with pipes that can wait, the urgent pipe is delayed.
      The "time-aware" penalty below prices this delay in dollars.

The linear cost (200 $/m of non-poly pipe) is the same however pipes are
grouped, so merging only trades the fixed cost  100 + (s*A)^0.85  against the
delay penalty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .geometry import convex_hull, min_enclosing_circle
from .simulator import fixed_cost
from .world import ANNUAL_BUDGET, LINEAR_RATE, World


@dataclass
class Params:
    r_max: float = 600.0  # largest circle radius we will consider (m)
    k_neigh: int = 8  # candidate merge partners per circle per round
    margin: float = 0.10  # safety margin added to every radius (m); final radius keeps >= margin/2 spare
    h_opt: float = 12.0  # years of avoided leaks credited to OPTIONAL pipes
    time_aware: bool = True  # include the delay penalty when merging
    min_gain: float = 1.0  # a merge must save at least this many dollars
    max_rounds: int = 100
    verbose: bool = True


class Cluster:
    """A group of target pipes plus the circle that covers them."""

    __slots__ = ("cid", "assigned", "pts", "cx", "cy", "r", "members", "F", "lin", "R", "C")

    def __init__(self, cid, assigned, pts, cx, cy, r, members, F, lin, R):
        self.cid = cid
        self.assigned = assigned  # np.ndarray of target pipe indices this circle must cover
        self.pts = pts  # convex-hull vertices of all their end points
        self.cx, self.cy, self.r = cx, cy, r
        self.members = members  # ALL pipes wholly inside the circle
        self.F = F  # fixed cost  100 + (s*A)^0.85
        self.lin = lin  # 200 * metres of assigned non-poly pipe
        self.R = R  # expected $/year lost while the assigned pipes are unreplaced
        self.C = F + lin  # cost to execute this circle


@dataclass
class Merge:
    cx: float
    cy: float
    r: float
    F: float
    members: np.ndarray
    absorbed: list
    gain_spend: float  # dollars of fixed cost saved by merging (before delay penalty)
    C_new: float


def choose_targets(world: World, params: Params) -> tuple[np.ndarray, np.ndarray]:
    """Decide which non-poly pipes the plan MUST (or chooses to) replace.

    * every iron pipe is a target (mandatory)
    * copper/brass pipes are optional. Replacing one costs 200 $/m plus its share
      of a circle, and saves its expected leak cost. We include an optional pipe
      if, replaced on its own, it would pay for itself within `h_opt` years:
          K * (1 - (1-p)^h_opt)  >  200*len + 100 + (s*pi*(len/2)^2)^0.85
    Returns (target mask, standalone-profitability mask for the optional pipes).
    """
    optional = world.nonpoly0 & ~world.required
    idx = np.flatnonzero(optional)
    benefit = world.leak_cost[idx] * (1 - (1 - world.p_annual[idx]) ** params.h_opt)
    cost = np.array([
        200 * world.length[i] + fixed_cost(world.surface_rate(np.array([i])), world.length[i] / 2)
        for i in idx
    ])
    target = world.required.copy()
    target[idx[benefit > cost]] = True
    return target, optional


class Builder:
    def __init__(self, world: World, params: Params | None = None):
        self.w = world
        self.p = params or Params()
        self.target, optional = choose_targets(world, self.p)
        self.n_optional_taken = int((self.target & optional).sum())
        self.n_optional = int(optional.sum())
        # net cost of replacing an optional pipe we did NOT target but a circle swallows anyway
        self.inc_net = np.zeros(world.n)
        inc = world.nonpoly0 & ~self.target
        self.inc_net[inc] = LINEAR_RATE * world.length[inc] - world.loss_rate[inc] * self.p.h_opt
        self.owner = np.full(world.n, -1, dtype=np.int64)  # cluster id owning each target pipe
        self.clusters: dict[int, Cluster] = {}
        self._next_id = 0
        self._cache: dict[tuple[int, int], Merge | None] = {}
        self.history: list[dict] = []

    # ---------------------------------------------------------------- init
    def initial_clusters(self):
        w, p = self.w, self.p
        for i in np.flatnonzero(self.target):
            pts = np.array([[w.x1[i], w.y1[i]], [w.x2[i], w.y2[i]]])
            cx, cy, r = min_enclosing_circle(pts)
            r += p.margin
            self._add(np.array([i]), pts, cx, cy, r)

    def _add(self, assigned, pts, cx, cy, r, members=None) -> Cluster:
        w = self.w
        if members is None:
            members = w.members(cx, cy, r)
        F = fixed_cost(w.surface_rate(members), r)
        lin = LINEAR_RATE * w.length[assigned][w.nonpoly0[assigned]].sum()
        R = w.loss_rate[assigned].sum()
        c = Cluster(self._next_id, assigned, pts, cx, cy, r, members, F, lin, R)
        self.clusters[c.cid] = c
        self.owner[assigned] = c.cid
        self._next_id += 1
        return c

    # ------------------------------------------------------------- merging
    def _evaluate(self, a: Cluster, b: Cluster) -> Merge | None:
        """Cost of replacing clusters a and b by their joint minimum circle."""
        key = (a.cid, b.cid)
        if key in self._cache:
            return self._cache[key]
        w, p = self.w, self.p
        pts = convex_hull(np.vstack([a.pts, b.pts]))
        cx, cy, r = min_enclosing_circle(pts)
        r += p.margin
        if r > p.r_max:
            self._cache[key] = None
            return None
        members = w.members(cx, cy, r)
        # other clusters whose target pipes ALL fall inside the new circle disappear (absorbed)
        tm = members[self.target[members]]
        owners, counts = np.unique(self.owner[tm], return_counts=True)
        absorbed = [
            int(o) for o, c in zip(owners, counts)
            if o != a.cid and o != b.cid and c == len(self.clusters[int(o)].assigned)
        ]
        F = fixed_cost(w.surface_rate(members), r)
        old_F = a.F + b.F + sum(self.clusters[o].F for o in absorbed)
        C_new = F + a.lin + b.lin + sum(self.clusters[o].lin for o in absorbed)
        # optional pipes swallowed by accident (or no longer swallowed) change the bill
        old_members = np.unique(np.concatenate([a.members, b.members] + [self.clusters[o].members for o in absorbed]))
        inc_old = self.inc_net[old_members].sum()
        inc_new = self.inc_net[members].sum()
        m = Merge(cx, cy, r, F, members, absorbed, (old_F - F) + (inc_old - inc_new), C_new)
        self._cache[key] = m
        return m

    def _apply(self, a: Cluster, b: Cluster, m: Merge):
        parts = [a] + [b] + [self.clusters[o] for o in m.absorbed]
        assigned = np.concatenate([c.assigned for c in parts])
        pts = convex_hull(np.vstack([c.pts for c in parts]))
        for c in parts:
            del self.clusters[c.cid]
        self._add(assigned, pts, m.cx, m.cy, m.r, members=m.members)

    def _schedule(self):
        """Provisional schedule: circles sorted by expected-loss-per-dollar (Smith's rule)."""
        ids = np.array(sorted(self.clusters))
        C = np.array([self.clusters[i].C for i in ids])
        R = np.array([self.clusters[i].R for i in ids])
        order = np.argsort(-(R / C), kind="stable")
        pos = np.empty(len(ids), dtype=np.int64)
        pos[order] = np.arange(len(ids))
        start = np.concatenate([[0.0], np.cumsum(C[order])])  # start[k] = money spent before slot k
        prefR = np.concatenate([[0.0], np.cumsum(R[order])])
        return ids, pos, start, prefR

    def delay_penalty(self, a, b, m, pos, start, prefR) -> float:
        """Dollar cost of the delay a merge causes, in a linear-time approximation.

        Expected leak loss ~ sum_j R_j * (money spent before circle j) / 10M, where R_j
        is the expected $/year lost on circle j's pipes. Merging A and B into one unit
        placed at either A's slot or B's slot moves: B's pipes earlier / A's pipes later,
        pushes the circles between them back, and pulls everything after both
        forward (the merge saves money). The cheaper placement is used.
        Returns the (possibly negative) increase in expected leak loss.
        """
        ia, ib = pos[self._idx[a.cid]], pos[self._idx[b.cid]]
        if ia > ib:
            a, b, ia, ib = b, a, ib, ia  # a = earlier slot
        C_A, C_B = a.C, b.C
        saved = C_A + C_B + sum(self.clusters[o].C for o in m.absorbed) - m.C_new
        M_A, M_B = start[ia], start[ib]
        Rmid = prefR[ib] - prefR[ia + 1]
        Rafter = prefR[-1] - prefR[ib + 1]
        d1 = -b.R * (M_B - M_A) + (m.C_new - C_A) * Rmid - saved * Rafter  # merged unit at A's slot
        d2 = a.R * (M_B - M_A) - C_A * (a.R + b.R + Rmid) - saved * Rafter  # merged unit at B's slot
        return min(d1, d2) / ANNUAL_BUDGET

    def round(self) -> int:
        cl, p = self.clusters, self.p
        n = len(cl)
        if n < 2:
            return 0
        ids, pos, start, prefR = self._schedule()
        self._idx = {int(c): i for i, c in enumerate(ids)}
        centres = np.array([[cl[i].cx, cl[i].cy] for i in ids])
        k = min(p.k_neigh + 1, n)
        dist, nn = cKDTree(centres).query(centres, k=k, distance_upper_bound=2 * p.r_max)
        cand = []
        for ai in range(n):
            for d, bi in zip(dist[ai], nn[ai]):
                if bi == n or bi == ai or not np.isfinite(d):
                    continue
                ca, cb = sorted((int(ids[ai]), int(ids[bi])))
                cand.append((ca, cb))
        cand = set(cand)
        scored = []
        for ca, cb in cand:
            a, b = cl[ca], cl[cb]
            m = self._evaluate(a, b)
            if m is None:
                continue
            gain = m.gain_spend
            if p.time_aware:
                gain -= self.delay_penalty(a, b, m, pos, start, prefR)
            if gain > p.min_gain:
                scored.append((gain, ca, cb))
        scored.sort(reverse=True)
        used: set[int] = set()
        n_merged = 0
        for gain, ca, cb in scored:
            if ca in used or cb in used:
                continue
            m = self._cache[(ca, cb)]
            if any(o in used for o in m.absorbed):
                continue
            self._apply(cl[ca], cl[cb], m)
            used.update([ca, cb, *m.absorbed])
            n_merged += 1
        # drop cache entries that mention dead clusters
        self._cache = {
            k_: v for k_, v in self._cache.items()
            if k_[0] in cl and k_[1] in cl and (v is None or all(o in cl for o in v.absorbed))
        }
        return n_merged

    def run(self) -> "Builder":
        if not self.clusters:
            self.initial_clusters()
        for rnd in range(1, self.p.max_rounds + 1):
            n_before = len(self.clusters)
            merged = self.round()
            F = sum(c.F for c in self.clusters.values())
            self.history.append({"round": rnd, "clusters": len(self.clusters), "merges": merged, "fixed_cost": F})
            if self.p.verbose:
                print(f"round {rnd:3d}: {n_before:6d} -> {len(self.clusters):6d} circles "
                      f"({merged} merges), total fixed cost ${F:,.0f}")
            if merged == 0:
                break
        return self

    # --------------------------------------------------------------- output
    def circles(self) -> np.ndarray:
        """Final circles (x, y, r) with coordinates rounded to the cent and a radius
        that provably still covers every assigned pipe after rounding."""
        w, p = self.w, self.p
        out = []
        for cid in sorted(self.clusters):
            c = self.clusters[cid]
            cx, cy = round(c.cx, 2), round(c.cy, 2)
            a = c.assigned
            need = np.sqrt(np.maximum(
                (w.x1[a] - cx) ** 2 + (w.y1[a] - cy) ** 2,
                (w.x2[a] - cx) ** 2 + (w.y2[a] - cy) ** 2,
            )).max()
            r = np.ceil((need + p.margin / 2) * 100) / 100  # round UP to the cent, plus a hair
            out.append((cx, cy, r))
        return np.array(out)

    def cluster_table(self):
        import pandas as pd

        rows = []
        for cid in sorted(self.clusters):
            c = self.clusters[cid]
            rows.append({
                "cluster": cid, "x": c.cx, "y": c.cy, "r": c.r, "n_targets": len(c.assigned),
                "n_members": len(c.members), "fixed_cost": c.F, "linear_cost": c.lin,
                "loss_rate": c.R, "density": c.R / c.C,
            })
        return pd.DataFrame(rows)
