"""End-to-end: world -> circles -> order -> checks -> submission file.

Also holds the *independent* checks. They deliberately do not reuse
``World.members`` / ``simulator.prepare`` so that a bug in those cannot hide.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .construct import Builder, Params
from .ordering import drop_redundant, smith_order
from .simulator import Prepared, expected_score, prepare
from .world import LINEAR_RATE, World


def build_plan(world: World, params: Params | None = None, order_iters: int = 8, verbose: bool = True) -> dict:
    """Construct circles, order them, drop skipped ones. Returns everything needed downstream."""
    builder = Builder(world, params or Params(verbose=verbose)).run()
    prep0 = prepare(world, builder.circles())
    ordered, log = smith_order(world, prep0, iters=order_iters, verbose=verbose)
    final = drop_redundant(world, ordered)
    return {"builder": builder, "prep_unordered": prep0, "prep": final, "order_log": log}


# ------------------------------------------------------------------ outputs
def circle_table(world: World, prep: Prepared) -> pd.DataFrame:
    """One row per circle in execution order, with the numbers a human wants to see."""
    ex = expected_score(world, prep)
    replaced = np.zeros(world.n, dtype=bool)
    rows, cum = [], 0.0
    for j in range(prep.n):
        m = prep.members(j)
        new = m[world.nonpoly0[m] & ~replaced[m]]
        replaced[new] = True
        # a circle with nothing left to replace is skipped for free by the real rules
        skipped = len(new) == 0
        exec_cost = 0.0 if skipped else round(prep.fixed[j] + LINEAR_RATE * world.length[new].sum(), 2)
        cum += exec_cost
        rows.append({
            "order": j + 1,
            "x": prep.circles[j, 0], "y": prep.circles[j, 1], "r": prep.circles[j, 2],
            "year": np.nan if skipped else 2027 + int(ex["year_of_circle"][j]),
            "pipes_replaced": len(new),
            "iron_replaced": int(world.required[new].sum()),
            "metres_replaced": float(world.length[new].sum()),
            "surface_rate": prep.surface_rate[j],
            "fixed_cost": prep.fixed[j],
            "exec_cost": exec_cost,
            "cumulative_spend": cum,
            "loss_rate_replaced": float(world.loss_rate[new].sum()),
        })
    return pd.DataFrame(rows)


def write_submission(circles: np.ndarray, path: str | Path) -> None:
    """Professor's format, one line per work unit:   "(x, y)",r"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        for x, y, r in circles:
            f.write(f'"({x:.2f}, {y:.2f})",{r:.2f}\n')


_LINE = re.compile(r'^"\((-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\)",(\d+(?:\.\d+)?)$')


def read_submission(path: str | Path) -> np.ndarray:
    """Parse a submission file strictly (any malformed line raises)."""
    out = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        m = _LINE.match(line)
        if not m:
            raise ValueError(f"line {i} is not in the form \"(x, y)\",r : {line!r}")
        out.append([float(g) for g in m.groups()])
    return np.array(out)


def independent_coverage_check(world: World, circles: np.ndarray, chunk: int = 400) -> dict:
    """Brute-force check that every required (iron) pipe is wholly inside at least one circle.

    No KD-tree, no World.members: plain numpy distances for every pipe/circle pair.
    Also reports the *tightest* margin, i.e. how close the worst pipe is to falling
    outside its circle, to show the answer does not hinge on floating-point luck.
    """
    req = np.flatnonzero(world.required)
    px1, py1, px2, py2 = (a[req] for a in (world.x1, world.y1, world.x2, world.y2))
    best_margin = np.full(len(req), -np.inf)  # max over circles of (r - farthest endpoint distance)
    for s in range(0, len(circles), chunk):
        c = circles[s: s + chunk]
        d1 = np.hypot(px1[None, :] - c[:, 0:1], py1[None, :] - c[:, 1:2])
        d2 = np.hypot(px2[None, :] - c[:, 0:1], py2[None, :] - c[:, 1:2])
        margin = c[:, 2:3] - np.maximum(d1, d2)
        best_margin = np.maximum(best_margin, margin.max(axis=0))
    uncovered = req[best_margin < 0]
    return {
        "n_required": len(req),
        "n_uncovered": len(uncovered),
        "uncovered_ids": world.pipe_id[uncovered].tolist(),
        "min_margin_m": float(best_margin.min()),
    }


def surface_summary(world: World, prep: Prepared) -> pd.DataFrame:
    """Fixed cost and circle count grouped by the circle's surface-rate set."""
    rows = []
    for rate in np.unique(prep.surface_rate):
        sel = prep.surface_rate == rate
        rows.append({"surface_rate": rate, "circles": int(sel.sum()), "fixed_cost": float(prep.fixed[sel].sum())})
    return pd.DataFrame(rows).sort_values("fixed_cost", ascending=False)


__all__ = [
    "build_plan", "circle_table", "write_submission", "read_submission",
    "independent_coverage_check", "surface_summary",
]
