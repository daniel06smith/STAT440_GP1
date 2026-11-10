"""How much circle (area + fee) spend is still on the table? -- headroom test for the set cover.

The set cover in pipeplan.setcover is exact OVER ITS CANDIDATE POOL, and every pool so far grew circles
the same way (a seed pipe + its k nearest target pipes by mid point). This script adds three candidate
families that way of growing cannot produce, mostly on water / structure (the two surfaces with ~$59M of
the ~$81M area term), and re-solves the full cover with the current best plan's circles included:

  F1 fixed-centre: centres on a grid near water/structure targets; the j nearest target pipes by their
     FARTHEST end point (j = 1..k), shrunk to the minimum circle of that group.
  F2 far-end order: each water/structure target + its nearest targets ordered by farthest end point
     (long pipes enter later than with mid-point order), all targets and same-surface only.
  F3 unions: the minimum circle of the targets of each current circle together with each of its
     nearest current circles (pairs) -- lets the MILP merge circles the k-nearest growth never paired.

Since the current plan's circles are candidates, the new cover can only be cheaper. The saving is the
measured headroom. If it is small, circle design is done.

    python scripts/spend_headroom.py --base x03_leakaware_setcover40_c1

Writes data/processed/spend_headroom.json (+ circles_headroom.csv, the new cover's circles unordered).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeplan import setcover, simulator  # noqa: E402
from pipeplan.construct import Builder, Params  # noqa: E402
from pipeplan.pipeline import read_submission  # noqa: E402
from pipeplan.world import SURFACES, load_world, read_risk_csv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="x03_leakaware_setcover40_c1")
ap.add_argument("--risk-csv", default=str(ROOT / "data/processed/worlds/leak_risk_all_x0.3.csv"))
ap.add_argument("--grid", type=float, default=6.0, help="F1 centre spacing (m)")
ap.add_argument("--k1", type=int, default=30)
ap.add_argument("--k2", type=int, default=50)
ap.add_argument("--k3", type=int, default=8)
ap.add_argument("--rmax", type=float, default=250.0)
ap.add_argument("--cushion", type=float, default=0.01)
ap.add_argument("--min-margin", type=float, default=0.005)
ap.add_argument("--time-limit", type=float, default=1800.0)
args = ap.parse_args()

t0 = time.time()
simulator.set_cost_formula("corrected")
world = load_world(risk=read_risk_csv(args.risk_csv, yearly=True))
b = Builder(world, Params(verbose=False))
target = b.target
tall = np.flatnonzero(target)
code = {s: i for i, s in enumerate(SURFACES)}
hard = np.isin(world.surf_bit, [1 << code["water"], 1 << code["structure"]]) & target
hidx = np.flatnonzero(hard)
print(f"{len(tall)} targets, {len(hidx)} on water/structure")

base = read_submission(ROOT / "submission" / f"submission_{args.base}.csv")
seen: set = set()
cands: list = []


def add(c):
    key = (round(c[0], 2), round(c[1], 2), round(c[2], 2))
    if key[2] <= args.rmax and key not in seen:
        seen.add(key)
        cands.append(key)


for c in base:
    add(tuple(c))
n_base = len(cands)



def far_dist(cx, cy, idx):
    return np.sqrt(np.maximum((world.x1[idx] - cx) ** 2 + (world.y1[idx] - cy) ** 2,
                              (world.x2[idx] - cx) ** 2 + (world.y2[idx] - cy) ** 2))


def grow(cx, cy, pool_tree, pool_idx, k):
    near = pool_tree.query_ball_point((cx, cy), args.rmax)
    if not near:
        return
    near = pool_idx[np.asarray(near)]
    order = near[np.argsort(far_dist(cx, cy, near))][:k]
    last = None
    for j in range(1, len(order) + 1):
        c = setcover._final_circle(world, order[:j], args.cushion)
        if c[2] > args.rmax:
            break
        if c != last:
            add(c)
            last = c


tree_all = cKDTree(np.column_stack([world.mx[tall], world.my[tall]]))

# F1: fixed-centre grid near water/structure targets
hx, hy = world.mx[hidx], world.my[hidx]
gx = np.arange(hx.min() - 10, hx.max() + 10, args.grid)
gy = np.arange(hy.min() - 10, hy.max() + 10, args.grid)
G = np.array(np.meshgrid(gx, gy)).reshape(2, -1).T
d, _ = cKDTree(np.column_stack([hx, hy])).query(G, distance_upper_bound=args.grid)
G = G[np.isfinite(d)]
print(f"F1: {len(G)} centres")
for i, (cx, cy) in enumerate(G):
    grow(cx, cy, tree_all, tall, args.k1)
    if i % 5000 == 0:
        print(f"  F1 {i}/{len(G)}  {len(cands)} candidates ({time.time() - t0:.0f}s)")
n_f1 = len(cands)

# F2: far-end ordering from each water/structure target, all targets and same surface only
for sb in np.unique(world.surf_bit[hidx]):
    same = tall[world.surf_bit[tall] == sb]
    tree_same = cKDTree(np.column_stack([world.mx[same], world.my[same]]))
    for a in hidx[world.surf_bit[hidx] == sb]:
        grow(world.mx[a], world.my[a], tree_all, tall, args.k2)
        grow(world.mx[a], world.my[a], tree_same, same, args.k2)
print(f"F2 done: {len(cands)} candidates ({time.time() - t0:.0f}s)")
n_f2 = len(cands)

# F3: pairwise unions of the base plan's circles
groups = []
for cx, cy, r in base:
    m = world.members(cx, cy, r - args.min_margin)
    groups.append(m[target[m]])
bt = cKDTree(base[:, :2])
_, nn = bt.query(base[:, :2], k=args.k3 + 1)
for i in range(len(base)):
    for j in nn[i, 1:]:
        if j < len(base) and len(groups[i]) and len(groups[j]):
            add(setcover._final_circle(world, np.concatenate([groups[i], groups[j]]), args.cushion))
print(f"F3 done: {len(cands)} candidates ({time.time() - t0:.0f}s)")

C = np.array(cands)
A, cost, tidx = setcover.candidate_matrix(world, target, C, b.inc_net, min_margin=args.min_margin)
print(f"matrix built ({time.time() - t0:.0f}s)")
base_cost = float(cost[:n_base].sum())
assert (np.asarray(A[:, :n_base].sum(axis=1)).ravel() > 0).all(), "base plan does not cover every target"
chosen = setcover.solve_cover(A, cost, time_limit=args.time_limit)
new_cost = float(cost[chosen].sum())

# where the saving comes from (by surface rate of the circle)
rate = np.array([world.surface_rate(world.members(*c)) for c in C])
by_rate = {}
for rt in np.unique(rate[:n_base].tolist() + rate[chosen].tolist()):
    o = cost[:n_base][rate[:n_base] == rt].sum()
    n = cost[chosen][rate[chosen] == rt].sum()
    if abs(o - n) > 1000:
        by_rate[str(rt)] = {"base": o, "new": n, "saving": o - n}

out = {
    "base": args.base, "n_candidates": len(C), "n_base": n_base,
    "n_F1": n_f1 - n_base, "n_F2": n_f2 - n_f1, "n_F3": len(C) - n_f2,
    "base_fixed_cost": base_cost, "new_fixed_cost": new_cost, "saving": base_cost - new_cost,
    "n_circles_base": n_base, "n_circles_new": int(chosen.sum()),
    "new_from_family": {
        "base": int(chosen[:n_base].sum()), "F1": int(chosen[n_base:n_f1].sum()),
        "F2": int(chosen[n_f1:n_f2].sum()), "F3": int(chosen[n_f2:].sum())},
    "saving_by_surface_rate": by_rate, "args": vars(args), "seconds": time.time() - t0,
}
(ROOT / "data/processed/spend_headroom.json").write_text(json.dumps(out, indent=2))
np.savetxt(ROOT / "data/processed/circles_headroom.csv", C[chosen], delimiter=",", fmt="%.2f",
           header="x,y,r", comments="")
print(json.dumps({k: v for k, v in out.items() if k != "args"}, indent=2))
