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
from pipeplan import simulator  # noqa: E402
from pipeplan.simulator import (  # noqa: E402
    expected_score, monte_carlo, prepare, sample_future, simulate, work_unit_cost,
)
from pipeplan.world import build_world, window_to_annual  # noqa: E402


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


# ------------------------------------------- formula switch & risk-file handling
def test_cost_formula_switch():
    try:
        simulator.set_cost_formula("original")
        a = work_unit_cost(85, 20.0, 0.0)
        simulator.set_cost_formula("corrected")
        b = work_unit_cost(85, 20.0, 0.0)
    finally:
        simulator.set_cost_formula("original")
    area = math.pi * 20.0**2
    assert abs(a - round(100 + (85 * area) ** 0.85, 2)) < 0.005
    assert abs(b - round(100 + 85 * area**0.85, 2)) < 0.005
    assert b > a  # corrected reading is dearer whenever surface_rate > 1


def test_window_probability_to_yearly():
    p8 = 0.2260
    yearly = window_to_annual(p8, 8)
    assert abs(1 - (1 - yearly) ** 8 - p8) < 1e-12
    assert window_to_annual(0.0, 5) == 0.0


def test_zero_leak_probability_is_allowed_but_missing_is_not():
    rows = [("A", "cast iron", "grassland", 0, 0, 10, 0), ("B", "cast iron", "grassland", 50, 0, 60, 0)]
    ok = pd.DataFrame({"Pipe ID": ["A", "B"], "p_annual": [0.0, 0.1], "leak_cost": [500.0, 500.0]})
    w = make_world(rows, risk=ok)
    assert w.lam[0] == 0.0 and w.lam[1] > 0
    short = ok.iloc[:1]
    try:
        make_world(rows, risk=short)
    except ValueError:
        pass
    else:
        raise AssertionError("a missing risk estimate must raise")


# ------------------------------------------------------- year-by-year leak risk
def _yearly_risk(ids, p_by_year, cost=1000.0):
    """p_by_year: (len(ids), 25) array of P(leak in project year k | not leaked yet)."""
    df = pd.DataFrame({"Pipe ID": ids, "p_annual": p_by_year[:, :10].mean(axis=1), "leak_cost": cost})
    for k in range(25):
        df[f"py_{k}"] = p_by_year[:, k]
    return df


def test_constant_yearly_risk_reproduces_constant_model():
    rows = [(f"P{i}", "cast iron", "grassland", 1000 * i, 0, 1000 * i + 10, 0) for i in range(3)]
    ids = [r[0] for r in rows]
    w_const = make_world(rows)
    w_year = make_world(rows, risk=_yearly_risk(ids, np.full((3, 25), 0.1)))
    assert w_year.time_varying and not w_const.time_varying
    assert np.allclose(w_const.cum_hazard, w_year.cum_hazard)
    prep = prepare(w_const, np.array([[1000 * i + 5, 0, 5.02] for i in range(3)]))
    kw = dict(annual_budget=3000, n_years=6)
    assert abs(expected_score(w_const, prep, **kw)["leak_cost"] - expected_score(w_year, prep, **kw)["leak_cost"]) < 1e-9


def test_expected_score_uses_the_yearly_risk_of_the_replacement_year():
    # P0 never leaks; P1 is safe in year 0 and leaks with prob 0.9 in year 1.
    rows = [(f"P{i}", "cast iron", "grassland", 1000 * i, 0, 1000 * i + 10, 0) for i in range(2)]
    P = np.zeros((2, 25))
    P[1, 1] = 0.9
    w = make_world(rows, risk=_yearly_risk(["P0", "P1"], P))
    prep = prepare(w, np.array([[5, 0, 5.02], [1005, 0, 5.02]]))  # each circle costs ~2.2k
    # budget 2500/yr -> P1 runs at the start of year 1, before its risky year: no expected leak
    ex = expected_score(w, prep, annual_budget=2500, n_years=25)
    assert ex["repl_year"][1] == 1 and ex["leak_cost"] == 0.0
    # budget 1200/yr -> P1 runs at the start of year 3, after its risky year: 0.9 * 1000
    ex_late = expected_score(w, prep, annual_budget=1200, n_years=25)
    assert ex_late["repl_year"][1] == 3 and abs(ex_late["leak_cost"] - 900.0) < 1e-6


def test_sampled_leak_times_follow_the_yearly_probability():
    rows = [("A", "cast iron", "grassland", 0, 0, 10, 0)]
    P = np.full((1, 25), 0.02)
    P[0, 3] = 0.5  # a spike in project year 3
    w = make_world(rows, risk=_yearly_risk(["A"], P))
    rng = np.random.default_rng(0)
    t = np.array([sample_future(w, rng)[0] for _ in range(40000)])
    surv3 = 0.98 ** 3
    assert abs((t < 3).mean() - (1 - surv3)) < 0.01
    assert abs(((t >= 3) & (t < 4)).mean() - surv3 * 0.5) < 0.01
    assert abs((t < 25).mean() - (1 - 0.98 ** 24 * 0.5)) < 0.01


def test_ordering_puts_a_spiking_pipe_first():
    from pipeplan.ordering import smith_order
    rows = [(f"P{i}", "cast iron", "grassland", 1000 * i, 0, 1000 * i + 10, 0) for i in range(2)]
    P = np.full((2, 25), 0.01)
    P[1, 0] = 0.6  # P1 spikes right now, P0 never does
    w = make_world(rows, risk=_yearly_risk(["P0", "P1"], P))
    prep = prepare(w, np.array([[5, 0, 5.02], [1005, 0, 5.02]]))
    best, _ = smith_order(w, prep, iters=2, verbose=False)
    assert best.circles[0, 0] == 1005


def test_lp_order_replaces_each_pipe_before_its_spike():
    from pipeplan.ordering import lp_order
    ids = ["A", "B", "C"]
    rows = [(k, "cast iron", "grassland", 1000 * i, 0, 1000 * i + 10, 0) for i, k in enumerate(ids)]
    P = np.zeros((3, 25))
    P[0, 0], P[1, 1], P[2, 2] = 0.9, 0.9, 0.9  # A spikes in year 0, B in year 1, C in year 2
    w = make_world(rows, risk=_yearly_risk(ids, P))
    # listed in the worst order; budget pays for exactly one circle (~2.2k) per year
    prep = prepare(w, np.array([[2005, 0, 5.02], [1005, 0, 5.02], [5, 0, 5.02]]))
    assert abs(expected_score(w, prep, annual_budget=2300)["leak_cost"] - 900.0) < 1e-6  # A waits to year 2
    best = lp_order(w, prep, annual_budget=2300, verbose=False)
    assert list(best.circles[:, 0]) == [5, 1005, 2005]
    assert expected_score(w, best, annual_budget=2300)["leak_cost"] == 0.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\n{len(tests)} tests passed")
