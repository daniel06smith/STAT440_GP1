"""Paired offline comparison of submissions across risk worlds (corrected formula, heavy-tailed leak costs).

    python scripts/compare_plans.py PLAN1 PLAN2 ... [--draws 100]     # PLAN = submission/submission_<PLAN>.csv

Every plan is scored on the SAME futures in each world (common random numbers), so "vs first" is a paired
difference with its standard error. Worlds: independent leaks (batchwave with the 1964-65 spike x0.35 as measured
by probe 3, the w060 ensemble) and triggered leaks (pipeplan.contagion_sim, primary hazard with spike x0.35).
This -- not the public leaderboard -- is how the final submission is chosen: the private future is a separate draw.
Output: data/processed/compare_plans.csv (world, plan, mean, sd, p05, p95, diff_vs_first, diff_se, win_rate).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeplan import simulator  # noqa: E402
from pipeplan.contagion_sim import ContagionWorld, load_trigger_model, monte_carlo_events  # noqa: E402
from pipeplan.pipeline import read_submission  # noqa: E402
from pipeplan.simulator import empirical_cost_ratios, monte_carlo, prepare  # noqa: E402
from pipeplan.world import load_world, read_risk_csv  # noqa: E402

WORLDS = {
    "triggered leaks, spike x0.35": ("leak_risk_batchwave_primary_s035.csv", True),
    "independent, spike x0.35": ("leak_risk_batchwave_s035.csv", False),
    "w060 ensemble": ("leak_risk_ensemble_w060.csv", False),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plans", nargs="+")
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--contagion-draws", type=int, default=40)
    ap.add_argument("--world-files", nargs="+", default=None,
                    help="independent-leak risk files (relative to data/processed) to use INSTEAD of the default worlds")
    ap.add_argument("--output", default="compare_plans.csv")
    args = ap.parse_args()
    worlds = WORLDS if not args.world_files else {Path(x).stem.replace("leak_risk_", ""): (x, False) for x in args.world_files}
    simulator.set_cost_formula("corrected")
    ratios = empirical_cost_ratios(ROOT)
    rows = []
    for wname, (fname, contagion) in worlds.items():
        w = load_world(risk=read_risk_csv(ROOT / "data/processed" / fname, yearly=True))
        cw = ContagionWorld(w, load_trigger_model(ROOT)) if contagion else None
        base = None
        for p in args.plans:
            prep = prepare(w, read_submission(ROOT / "submission" / f"submission_{p}.csv"))
            if contagion:
                mc = monte_carlo_events(w, prep, cw, n_draws=args.contagion_draws, seed=31, cost_ratios=ratios, keep_draws=True)
            else:
                mc = monte_carlo(w, prep, n_draws=args.draws, seed=31, cost_ratios=ratios, keep_draws=True)
            d = mc["score_draws"] / 1e6
            base = d if base is None else base
            diff = d - base
            rows.append(dict(world=wname, plan=p, mean=d.mean(), sd=d.std(ddof=1), p05=np.percentile(d, 5),
                             p95=np.percentile(d, 95), diff_vs_first=diff.mean(),
                             diff_se=diff.std(ddof=1) / np.sqrt(len(d)), win_rate=float(np.mean(diff > 0))))
            r = rows[-1]
            print(f"{wname:30s} {p:40s} ${r['mean']:6.2f}M  vs first {r['diff_vs_first']:+5.2f}M "
                  f"(se {r['diff_se']:.2f}, wins {r['win_rate']:.0%})", flush=True)
    simulator.set_cost_formula("original")
    pd.DataFrame(rows).to_csv(ROOT / "data/processed" / args.output, index=False)


if __name__ == "__main__":
    main()
