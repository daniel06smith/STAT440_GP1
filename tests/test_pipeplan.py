"""Unit tests for the scoring simulator and geometry helpers.

Run from the repo root with either of
    python tests/test_pipeplan.py
    python -m pytest tests            (if pytest is installed)

Every test builds a tiny hand-made network so the expected numbers can be
checked by hand against the professor's formula.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeplan.geometry import convex_hull, min_enclosing_circle  # noqa: E402
from pipeplan.simulator import (  # noqa: E402
    expected_score, monte_carlo, prepare, sample_future, simulate, work_unit_cost,
)
from pipeplan.world import build_world  # noqa: E402


def make_world(rows, risk=None, leaked=()):
    """rows: (id, material, surface, x1, y1, x2, y2)"""
    df = pd.DataFrame(rows, columns=["Pipe ID", "Material", "Surface", "GPS x1", "GPS y1", "GPS x2", "GPS y2"])
    if risk is None:
        ids = [r[0] for r in rows]
        risk = pd.DataFrame({"Pipe ID": ids, "p_annual": 0.1, "leak_cost": 1000.0})
    return build_world(df, set(leaked), risk)


def cost_by_hand(rate, r, metres):
    return round(100 + (rate * math.pi * r**2) ** 0.85 + 200 * metres, 2)


# ---------------------------------------------------------------- geometry
def test_min_enclosing_circle():
    cx, cy, r = min_enclosing_circle(np.array([[0, 0], [10, 0]]))
    assert (cx, cy, r) == (5, 0, 5)
    # equilateral-ish triangle: all three vertices on the circle
    pts = np.array([[0, 0], [10, 0], [5, 8.0]])
    cx, cy, r = min_enclosing_circle(pts)
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    assert np.allclose(d, r)
    # random cloud: circle contains all points and is no bigger than the bounding diagonal/2 * ... sanity
    rng = np.random.default_rng(1)
    cloud = rng.normal(size=(200, 2)) * 50
    cx, cy, r = min_enclosing_circle(convex_hull(cloud))
    assert (np.hypot(cloud[:, 0] - cx, cloud[:, 1] - cy) <= r + 1e-7).all()


# ---------------------------------------------------------- cost formula
def test_single_circle_cost_matches_formula():
    w = make_world([("A", "cast iron", "grassland", 0, 0, 10, 0)])
    prep = prepare(w, np.array([[5.0, 0.0, 5.02]]))
    assert prep.covers_all_iron
    res = simulate(w, prep)
    assert abs(res["spend"] - cost_by_hand(3, 5.02, 10)) < 0.01
    assert res["feasible"] and res["last_year"] == 0


def test_surface_rate_counts_every_pipe_inside_even_polyurethane():
    rows = [
        ("A", "cast iron", "grassland", 0, 0, 10, 0),
        ("B", "polyurethane", "structure", 0, 1, 10, 1),  # not replaced, but sets the rate
        ("C", "cast iron", "water", 0, 2, 10, 2),
    ]
    w = make_world(rows)
    prep = prepare(w, np.array([[5.0, 1.0, 5.2]]))
    assert prep.surface_rate[0] == 3 + 50 + 32
    res = simulate(w, prep)
    # only A and C are non-poly -> 20 m of linear cost
    assert abs(res["spend"] - cost_by_hand(85, 5.2, 20)) < 0.01


def test_pipe_must_be_wholly_inside():
    rows = [("A", "cast iron", "grassland", 0, 0, 10, 0), ("B", "cast iron", "grassland", 0, 0, 30, 0)]
    w = make_world(rows)
    inside = w.members(5.0, 0.0, 5.01)
    assert list(inside) == [0]
    prep = prepare(w, np.array([[5.0, 0.0, 5.01]]))
    assert list(prep.uncovered_required) == [1]  # iron pipe B not covered -> plan is rejected
    assert not simulate(w, prep)["feasible"]


def test_leaked_and_polyurethane_pipes_are_not_replaced_or_billed():
    rows = [
        ("A", "cast iron", "grassland", 0, 0, 10, 0),
        ("B", "cast iron", "grassland", 0, 1, 10, 1),  # leaked in 2019-2026 -> already poly
        ("C", "polyurethane", "grassland", 0, 2, 10, 2),
    ]
    w = make_world(rows, leaked={"B"})
    assert list(w.nonpoly0) == [True, False, False]
    assert list(w.required) == [True, False, False]
    prep = prepare(w, np.array([[5.0, 1.0, 5.2]]))
    assert abs(simulate(w, prep)["spend"] - cost_by_hand(3, 5.2, 10)) < 0.01


def test_missing_material_treated_as_required_iron():
    rows = [("A", None, "water", 0, 0, 10, 0)]
    w = make_world(rows)
    assert w.required[0] and w.material_missing[0]


# ------------------------------------------------------- budget & timing
def three_pipe_world():
    rows = [(f"P{i}", "cast iron", "grassland", 1000 * i, 0, 1000 * i + 10, 0) for i in range(3)]
    return make_world(rows)


def test_budget_carry_over_and_blocking():
    w = three_pipe_world()
    circles = np.array([[1000 * i + 5, 0, 5.02] for i in range(3)])
    prep = prepare(w, circles)
    c = cost_by_hand(3, 5.02, 10)  # ~2204.5 each
    # budget 3000/yr: unit1 in year 0 (leaves ~795), unit2 needs year 1, unit3 needs year 2
    res = simulate(w, prep, annual_budget=3000, n_years=5)
    assert list(res["exec_year"]) == [0, 1, 2]
    assert abs(res["spend"] - 3 * c) < 0.02
    # the closed-form shortcut must agree on the timeline
    ex = expected_score(w, prep, annual_budget=3000, n_years=5)
    assert list(ex["year_of_circle"]) == [0, 1, 2]
    # too little time -> rejected
    assert not simulate(w, prep, annual_budget=3000, n_years=2)["feasible"]


def test_a_unit_that_does_not_fit_blocks_later_cheaper_units():
    rows = [("BIG", "cast iron", "grassland", 0, 0, 30, 0), ("SMALL", "cast iron", "grassland", 1000, 0, 1003, 0)]
    w = make_world(rows)
    prep = prepare(w, np.array([[15.0, 0, 15.02], [1001.5, 0, 1.52]]))
    big = cost_by_hand(3, 15.02, 30)
    small = cost_by_hand(3, 1.52, 3)
    assert big > 6500 > small
    res = simulate(w, prep, annual_budget=6500, n_years=5)
    # BIG waits for year 1 (budget 13000 >= big), SMALL cannot jump the queue
    assert list(res["exec_year"]) == [1, 1]
    assert abs(res["spend"] - big - small) < 0.02


# ----------------------------------------------------------------- leaks
def test_leak_before_execution_costs_money_but_removes_linear_cost():
    w = three_pipe_world()
    circles = np.array([[1000 * i + 5, 0, 5.02] for i in range(3)])
    prep = prepare(w, circles)
    # pipe 1 leaks at t=0.5 years (cost 1000) with a 3000/yr budget: circle 2 would run in year 1
    t = np.array([np.inf, 0.5, np.inf])
    res = simulate(w, prep, t, annual_budget=3000, n_years=5)
    c = cost_by_hand(3, 5.02, 10)
    # circle for pipe1 is skipped?  No: pipe 1 leaked at 0.5 < year 1 => it is poly when
    # circle 2 runs, so circle 2 is skipped (nothing left) and costs nothing.
    assert res["n_skipped"] == 1 and res["n_executed"] == 2
    assert abs(res["spend"] - 2 * c) < 0.02
    assert res["leak_cost"] == 1000.0 and res["n_leaks"] == 1
    # surplus - leak
    assert abs(res["score"] - (3000 * 5 - 2 * c - 1000)) < 0.02


def test_leak_after_replacement_is_not_counted():
    w = three_pipe_world()
    prep = prepare(w, np.array([[1000 * i + 5, 0, 5.02] for i in range(3)]))
    t = np.array([0.0 + 3.0, np.inf, np.inf])  # pipe 0 would leak in year 3, but is replaced in year 0
    res = simulate(w, prep, t, annual_budget=1e7, n_years=5)
    assert res["leak_cost"] == 0.0


def test_expected_score_matches_monte_carlo_when_budget_is_loose():
    w = three_pipe_world()
    prep = prepare(w, np.array([[1000 * i + 5, 0, 5.02] for i in range(3)]))
    kw = dict(annual_budget=3000, n_years=6)
    ex = expected_score(w, prep, **kw)
    # replaced in years 0,1,2 with p=0.1/yr and cost 1000: E[leak] = 1000*(0 + (1-.9) + (1-.81))
    assert abs(ex["leak_cost"] - 1000 * (0 + 0.1 + 0.19)) < 1e-9
    mc = monte_carlo(w, prep, n_draws=4000, seed=3, **kw)
    # In the MC, a pipe that leaks first makes its circle cheaper, so spend is slightly lower
    assert abs(mc["leak_cost_mean"] - ex["leak_cost"]) < 25
    assert mc["spend_mean"] <= ex["spend"] + 1e-6


def test_sampled_leak_times_follow_the_annual_probability():
    w = three_pipe_world()
    rng = np.random.default_rng(0)
    t = np.array([sample_future(w, rng) for _ in range(20000)])
    assert abs((t[:, 0] < 1).mean() - 0.1) < 0.01
    assert abs((t[:, 0] < 2).mean() - 0.19) < 0.01


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\n{len(tests)} tests passed")
