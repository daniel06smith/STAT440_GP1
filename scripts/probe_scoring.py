"""Diagnostic submission: measure the evaluator's AREA-cost rule on the (fixed) public future.

    python scripts/probe_scoring.py                        # probe 1 -> submission/submission_probe_area.csv
    python scripts/probe_scoring.py --old-only             # probe 2 -> submission/submission_probe_oldpoly.csv

Why: the public score of every upload sits ~$16M above our corrected-formula prediction, and the final score
alone cannot tell whether the evaluator charges less for circles or whether leaks cost less than we model.

How: take the uploaded plan and enlarge some of its LAST circles (concentric, bigger radius) so that
  * every non-polyurethane pipe the bigger circle newly swallows is first-covered by an EARLIER circle (it is
    already polyurethane when the circle runs) -> nothing else about the plan changes;
  * no circle after them is pushed back in a way that matters (they are at the end of the order);
  * their own pipes are very unlikely to have ALL leaked by then (an all-leaked circle is skipped for free,
    which would hide the signal).
The public score then drops by (almost exactly) the extra area cost, which differs between the two readings
    original : (surface_rate * area)^0.85        corrected : surface_rate * area^0.85
by a factor surface_rate^0.15 (~1.8x for structure). Structure/water circles are preferred.

--old-only (probe 2, "are old pipes secretly polyurethane already?"): only circles whose single own pipe was
laid before --old-before (1954) are enlarged, each by about --per-circle dollars of corrected area cost. The spec
only says pipes that leaked DURING 2019-2026 were repaired, but the spike model puts the old batches' failure
waves in ~2011-2019, before the data starts. A circle whose only pipe is already polyurethane is SKIPPED (free),
so the score drop counts how many of those pipes are still iron. That status is a fixed fact about today's
network -- identical in the public and private futures -- so measuring it does not overfit the leaderboard.

Prints the predicted score change under both formulas (paired Monte Carlo over futures, so the spread shown is
how much the change itself depends on the unknown future) and writes data/processed/probe_area_formula.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeplan import simulator  # noqa: E402
from pipeplan.pipeline import independent_coverage_check, read_submission, write_submission  # noqa: E402
from pipeplan.simulator import empirical_cost_ratios, expected_score, monte_carlo, prepare, survival  # noqa: E402
from pipeplan.world import BASE_FEE, RATE_BY_MASK, load_world, read_risk_csv  # noqa: E402


def area_cost(rate: float, r: float, formula: str) -> float:
    a = math.pi * r * r
    return BASE_FEE + (rate * a) ** 0.85 if formula == "original" else BASE_FEE + rate * a ** 0.85


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="batchwave_corrected_leakaware")
    ap.add_argument("--risk", default="leak_risk_batchwave.csv")
    ap.add_argument("--target", type=float, default=6e6, help="extra area cost to add under the corrected reading ($)")
    ap.add_argument("--block", type=int, default=600, help="only circles among the last BLOCK executed are enlarged")
    ap.add_argument("--r-max", type=float, default=120.0)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--old-only", action="store_true", help="probe 2: single-pipe circles of pipes laid before --old-before")
    ap.add_argument("--old-before", type=float, default=1954.0)
    ap.add_argument("--per-circle", type=float, default=60e3, help="probe 2: extra corrected area cost per circle ($)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-p-all", type=float, default=None,
                    help="skip circles whose own pipes ALL leak before they run with more than this probability "
                         "(default 0.01; 0.4 with --old-only, where it is modelled per circle in the prediction)")
    args = ap.parse_args()

    simulator.set_cost_formula("corrected")
    w = load_world(risk=read_risk_csv(ROOT / "data/processed" / args.risk, yearly=True))
    base = read_submission(ROOT / "submission" / f"submission_{args.plan}.csv")
    prep = prepare(w, base)
    ex = expected_score(w, prep, leak_aware=True)
    year = ex["year_of_circle"]

    # first-cover owner of every non-poly pipe (position in the execution order)
    owner = np.full(w.n, -1, dtype=np.int64)
    own = []
    for j in range(prep.n):
        m = prep.members(j)
        new = m[w.nonpoly0[m] & (owner[m] < 0)]
        owner[new] = j
        own.append(new)

    import pandas as pd
    lay = pd.read_csv(ROOT / "data/raw/pipes.csv", parse_dates=["Lay date"]).set_index("Pipe ID")["Lay date"]
    lay = (lay.dt.year + (lay.dt.dayofyear - 0.5) / 365.25).reindex(w.pipe_id).to_numpy()
    start = prep.n - args.block
    cands = []
    for j in range(start, prep.n):
        new = own[j]
        if len(new) == 0 or year[j] < 0:
            continue
        if args.old_only and (len(new) != 1 or not (lay[new] < args.old_before).all()):
            continue
        # P(every own pipe leaked before the circle runs; a year later to be safe)
        p_all = float(np.prod(1 - survival(w, new, min(int(year[j]) + 1, 25))))
        if p_all > (args.max_p_all if args.max_p_all is not None else (0.4 if args.old_only else 0.01)):
            continue
        cx, cy, r = prep.circles[j]
        rate0 = RATE_BY_MASK[np.bitwise_or.reduce(w.surf_bit[prep.members(j)])]
        best = None
        for R in np.arange(math.ceil(r) + 2.0, args.r_max + 1e-9, 2.0):
            m = w.members(cx, cy, R)
            np_m = m[w.nonpoly0[m]]
            if ((owner[np_m] < 0) | (owner[np_m] > j)).any():
                break  # would swallow a pipe that is still non-poly when this circle runs
            rate = RATE_BY_MASK[np.bitwise_or.reduce(w.surf_bit[m])]
            d_corr = area_cost(rate, R, "corrected") - area_cost(rate0, r, "corrected")
            d_orig = area_cost(rate, R, "original") - area_cost(rate0, r, "original")
            best = (R, rate, d_corr, d_orig)
            if args.old_only and d_corr >= args.per_circle:
                break  # probe 2: equal-sized steps, so the drop counts circles
        if args.old_only and (best is None or best[2] < 0.8 * args.per_circle):
            continue
        if best and best[2] > 0:
            cands.append((j, *best, rate0, p_all))

    # take the circles with the largest gap between the two readings, until the corrected target is reached
    if not args.old_only:
        cands.sort(key=lambda c: -(c[3] - c[4]) / c[3])
    chosen, tot_c, tot_o = [], 0.0, 0.0
    for c in cands:
        if tot_c >= args.target:
            break
        chosen.append(c)
        tot_c += c[3]
        tot_o += c[4]
    if tot_c < 0.8 * args.target:
        raise SystemExit(f"only ${tot_c:,.0f} of extra corrected area cost available; lower --target or raise --block")

    probe = base.copy()
    for j, R, *_ in chosen:
        probe[j, 2] = math.ceil(R * 100) / 100
    out = ROOT / "submission" / (args.out or ("submission_probe_oldpoly.csv" if args.old_only else "submission_probe_area.csv"))
    write_submission(probe, out)

    # ---- checks on the written file
    circ = read_submission(out)
    cov = independent_coverage_check(w, circ)
    assert cov["n_uncovered"] == 0, "probe lost an iron pipe"
    prep_p = prepare(w, circ)
    res = {"plan": args.plan, "n_enlarged": len(chosen), "first_enlarged_position": int(min(c[0] for c in chosen)),
           "n_circles": int(prep.n), "static_extra_area_corrected": tot_c, "static_extra_area_original": tot_o,
           "surfaces_of_enlarged": {str(int(k)): int(v) for k, v in zip(*np.unique([c[5] for c in chosen], return_counts=True))}}
    ratios = empirical_cost_ratios(ROOT)
    for formula in ("corrected", "original"):
        simulator.set_cost_formula(formula)
        exb = expected_score(w, prepare(w, base), leak_aware=True)
        exp_ = expected_score(w, prepare(w, circ), leak_aware=True)
        assert exp_["last_year_no_leaks"] < 25, "probe would not finish in 25 years if nothing leaked"
        a = monte_carlo(w, prepare(w, base), n_draws=args.draws, seed=11, cost_ratios=ratios, keep_draws=True)
        b = monte_carlo(w, prepare(w, circ), n_draws=args.draws, seed=11, cost_ratios=ratios, keep_draws=True)
        dlt = b["score_draws"] - a["score_draws"]
        res[formula] = {"pred_change_mean": float(dlt.mean()), "pred_change_sd": float(dlt.std(ddof=1)),
                        "pred_change_p05": float(np.percentile(dlt, 5)), "pred_change_p95": float(np.percentile(dlt, 95)),
                        "last_year_no_leaks": int(exp_["last_year_no_leaks"]), "base_last_year_no_leaks": int(exb["last_year_no_leaks"])}
    simulator.set_cost_formula("original")
    if args.old_only:
        # P(circle is skipped | the pipe is iron): its pipe leaked (in 2027+) before the circle ran
        js = [c[0] for c in chosen]
        p_leak_first = np.array([1 - float(survival(w, own[j], int(year[j]))[0]) for j in js])
        dF = np.array([c[3] for c in chosen])
        res["old_probe"] = {
            "n_circles": len(js), "mean_dF_corrected": float(dF.mean()),
            "mean_p_pipe_leaked_before_its_circle": float(p_leak_first.mean()),
            "pred_change_if_none_secretly_poly": float(-(dF * (1 - p_leak_first)).sum()),
            "how_to_read": "change = -(sum dF over executed circles); share still iron ~= change / pred_change_if_none_secretly_poly",
        }
    name = "probe_oldpoly.json" if args.old_only else "probe_area_formula.json"
    (ROOT / "data/processed" / name).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nwrote {out}: {len(chosen)} of the last {args.block} circles enlarged; coverage OK")


if __name__ == "__main__":
    main()
