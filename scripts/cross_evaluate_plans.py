"""Score finished submission files under several leak-risk "worlds" (Monte Carlo over 2027-2051).

    python scripts/cross_evaluate_plans.py
    python scripts/cross_evaluate_plans.py --plans ensemble_w060 batchwave --draws 300

A plan built from one risk model should be judged under the others too: if it only wins under its own
model, the win is an artefact. Each world draws leak times from its risk file (year-by-year columns when
--yearly worlds are used) and uses that file's cost-if-leak. Output:
data/processed/cross_evaluation_batchwave.csv (one row per plan x world).

--heavy-tail also draws each leak's SIZE from the real 2019-2026 cost distribution (same expected score,
realistic spread; without it every leak costs its model mean and the sd is ~4-5x too small). It adds
mc_score_p05/p95 and, because every plan sees the same futures (common random numbers), the paired
difference to the first --plans entry: diff_vs_first_mean / diff_vs_first_sd ($).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeplan.pipeline import read_submission  # noqa: E402
import numpy as np  # noqa: E402

from pipeplan.simulator import empirical_cost_ratios, expected_score, monte_carlo, prepare  # noqa: E402
from pipeplan.world import load_world, read_risk_csv  # noqa: E402

WORLDS = {
    "w060 (constant p)": ("leak_risk_ensemble_w060.csv", False),
    "w060 (yearly p)": ("leak_risk_ensemble_w060.csv", True),
    "batchwave": ("leak_risk_batchwave.csv", True),
    "batchwave + 1971/1982 spikes": ("leak_risk_batchwave_extrap.csv", True),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plans", nargs="+", default=["ensemble_w060", "batchwave", "batchwave_extrap"])
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--heavy-tail", action="store_true", help="draw leak sizes from the real cost distribution")
    ap.add_argument("--worlds", nargs="+", default=list(WORLDS), help="subset of world names")
    args = ap.parse_args()
    if args.output is None:
        name = "cross_evaluation_heavytail.csv" if args.heavy_tail else "cross_evaluation_batchwave.csv"
        args.output = ROOT / "data/processed" / name
    ratios = empirical_cost_ratios(ROOT) if args.heavy_tail else None

    circles = {p: read_submission(ROOT / "submission" / f"submission_{p}.csv") for p in args.plans}
    rows = []
    for wname in args.worlds:
        fname, yearly = WORLDS[wname]
        first = None
        world = load_world(risk=read_risk_csv(ROOT / "data/processed" / fname, yearly=yearly))
        for plan, c in circles.items():
            prep = prepare(world, c)
            ex = expected_score(world, prep)
            mc = monte_carlo(world, prep, n_draws=args.draws, seed=args.seed, cost_ratios=ratios, keep_draws=True)
            draws = mc.pop("score_draws")
            first = draws if first is None else first
            extra = {"mc_score_p05": mc["score_p05"], "mc_score_p95": mc["score_p95"],
                     "diff_vs_first_mean": float(np.mean(draws - first)),
                     "diff_vs_first_sd": float(np.std(draws - first, ddof=1))}
            rows.append(extra | {"world": wname, "plan": plan, "mc_score": mc["score_mean"], "mc_score_sd": mc["score_sd"],
                         "mc_leak_cost": mc["leak_cost_mean"], "mc_spend": mc["spend_mean"],
                         "mc_n_leaks": mc["n_leaks_mean"], "expected_score": ex["score"],
                         "uncovered_iron": len(prep.uncovered_required)})
            print(f"{wname:30s} {plan:18s} MC score ${mc['score_mean']:>13,.0f} (sd {mc['score_sd']:>10,.0f})  "
                  f"leaks {mc['n_leaks_mean']:6.0f}  leak ${mc['leak_cost_mean']:>12,.0f}  spend ${mc['spend_mean']:>12,.0f}",
                  f"  vs first: {extra['diff_vs_first_mean']:>+12,.0f} (sd {extra['diff_vs_first_sd']:,.0f})", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    pd.set_option("display.width", 200)
    print("\nMonte Carlo mean score, $M (rows = world the plan is judged in):")
    print((out.pivot(index="world", columns="plan", values="mc_score") / 1e6).round(2).to_string())
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
