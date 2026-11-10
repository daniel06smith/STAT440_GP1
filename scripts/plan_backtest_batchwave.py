"""Rolling-origin PLAN-level backtest: w060 pipeline vs batch-wave model + time-aware planner.

    python scripts/plan_backtest_batchwave.py                      # all cutoffs, 2x2 design
    python scripts/plan_backtest_batchwave.py --cutoffs 2023 2024  # subset

For each cutoff c, using ONLY data up to the end of year c:
  risk "w060"      : the exact recipe behind leak_risk_ensemble_w060.csv -- LightGBM base+nbr with half the
                     out-of-sample level shift, blended 0.6/0.4 in hazard space with the hazard GLM + recal,
                     yearly probabilities for c+1..c+25 and p_leak_annual = their 10-year average.
  risk "batchwave" : scripts/batch_wave_hazard.py fitted at c.
  planner "const"  : the original planner (one constant yearly probability p_leak_annual per pipe).
  planner "yearly" : the time-aware planner (year-by-year probabilities).
The project is re-based so that "year 0" = c+1, every plan gets the full 25-year budget, and the plan is
scored against what really happened: the real leak dates (fractional years) and real costs of every leak in
c+1..2026. Leaks after 2026 are unknown, so later cutoffs score fewer years of leaks; spend is always the
whole plan. Compare rows within a cutoff, not across cutoffs.

Output: data/processed/plan_backtest_batchwave.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import batch_wave_hazard as bw  # noqa: E402
import leak_calibration as lc  # noqa: E402
from ensemble_risk import blend_probability_matrix  # noqa: E402
from lgbm_hazard import VARIANTS, add_geometry, apply_shift, fit_lgbm, lgbm_p, oos_level_shift  # noqa: E402
from pipeplan.construct import Params  # noqa: E402
from pipeplan.pipeline import build_plan  # noqa: E402
from pipeplan.simulator import N_YEARS, expected_score, simulate_project  # noqa: E402
from pipeplan.world import build_world  # noqa: E402

W060 = 0.6
LEVEL_SCALE = 0.5
VARIANT = "base+nbr"


def risk_w060(d: dict, c: int, at: np.ndarray, risk: dict) -> np.ndarray:
    years = range(c + 1, c + 1 + N_YEARS)
    cols = VARIANTS[VARIANT]
    P_l = np.column_stack([lgbm_p(fit_lgbm(d, c, cols), d, at, y, c) for y in years])
    P_l = apply_shift(P_l, oos_level_shift(d, c, cols) ** LEVEL_SCALE)
    glm = lc.recalibrate(lc.fit_hazard(d, c), d)
    P_h = np.column_stack([lc.hazard_p(glm, risk, y) for y in years])
    return blend_probability_matrix(P_l, P_h, W060)


def risk_frame(pid, P: np.ndarray, cost: np.ndarray) -> pd.DataFrame:
    p10 = 1 - np.prod(1 - P[:, :10], axis=1)
    df = pd.DataFrame({"Pipe ID": pid, "p_annual": 1 - (1 - p10) ** 0.1, "leak_cost": cost})
    for k in range(N_YEARS):
        df[f"py_{k}"] = P[:, k]
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoffs", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--output", type=Path, default=ROOT / "data/processed/plan_backtest_batchwave.csv")
    args = ap.parse_args()

    d = add_geometry(ROOT, bw.load(ROOT))
    pipes = pd.read_csv(ROOT / "data/raw/pipes.csv")
    rows = []
    for c in args.cutoffs:
        t0 = time.time()
        at_mask = ~(d["leak_year"] <= c)
        at = np.flatnonzero(at_mask)
        risk = lc.sub(d, at_mask)
        cost = lc.cost_hat(lc.fit_cost(d, c), risk)
        P = {
            "w060": risk_w060(d, c, at, risk),
            "batchwave": bw.yearly_p(bw.fit(d, c), d, at, range(c + 1, c + 1 + N_YEARS)),
        }
        leaked_ids = set(d["pid"][d["leak_year"] <= c])

        # what really happened after the cutoff
        future = (d["leak_dec"] >= c + 1) & (d["leak_dec"] < lc.LAST_YEAR + 1)
        for model, Pm in P.items():
            frame = risk_frame(risk["pid"], Pm, cost)
            for planner in ("const", "yearly"):
                f = frame if planner == "yearly" else frame[["Pipe ID", "p_annual", "leak_cost"]]
                world = build_world(pipes, leaked_ids, f)
                plan = build_plan(world, Params(verbose=False), verbose=False)
                prep = plan["prep"]
                pos = pd.Series(np.arange(world.n), index=world.pipe_id)
                leak_time = np.full(world.n, np.inf)
                leak_cost = world.leak_cost.copy()
                where = pos[d["pid"][future]].to_numpy()
                leak_time[where] = d["leak_dec"][future] - (c + 1)
                leak_cost[where] = d["cost"][future]
                res = simulate_project(world, prep, leak_time=leak_time, leak_cost=leak_cost)
                # same realised leaks, but each valued at its out-of-sample expected cost (removes the
                # heavy-tailed cost noise; both arms share the cost model, so this isolates timing)
                res_k = simulate_project(world, prep, leak_time=leak_time, leak_cost=world.leak_cost)
                ex = expected_score(world, prep)
                rows.append({
                    "cutoff": c, "scored_years": lc.LAST_YEAR - c, "risk_model": model, "planner": planner,
                    "score": res["score"], "spend": res["spend"], "leak_cost": res["leak_cost"],
                    "score_expected_cost": res_k["score"], "leak_cost_expected_cost": res_k["leak_cost"],
                    "n_leaks": res["n_leaks"], "actual_leaks_in_window": int(future.sum()),
                    "actual_leak_cost_in_window": float(d["cost"][future].sum()),
                    "feasible": res["feasible"], "last_year": res["last_year"], "n_circles": prep.n,
                    "model_expected_leak_cost": ex["leak_cost"],
                })
                print(f"{c} {model:9s} {planner:6s}: score ${res['score']:>14,.0f}  spend ${res['spend']:>13,.0f}  "
                      f"leaks {res['n_leaks']:5d} ${res['leak_cost']:>13,.0f}  ({time.time() - t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(args.output, index=False)

    out = pd.DataFrame(rows)
    base = out[(out.risk_model == "w060") & (out.planner == "const")].set_index("cutoff")["score"]
    out["vs_w060"] = out["score"] - out["cutoff"].map(base)
    base_k = out[(out.risk_model == "w060") & (out.planner == "const")].set_index("cutoff")["score_expected_cost"]
    out["vs_w060_expected_cost"] = out["score_expected_cost"] - out["cutoff"].map(base_k)
    out.to_csv(args.output, index=False)
    pd.set_option("display.width", 200)
    for col, label in (("vs_w060", "real leak costs"), ("vs_w060_expected_cost", "leaks valued at expected cost")):
        t = out.pivot_table(index=["risk_model", "planner"], columns="cutoff", values=col) / 1e6
        t["mean"] = t.mean(axis=1)
        print(f"\nscore minus the w060 pipeline (same cutoff), $M -- {label}:")
        print(t.round(2).to_string())
    print("\nleaks the plan failed to prevent:")
    print(out.pivot_table(index=["risk_model", "planner"], columns="cutoff", values="n_leaks").to_string())
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
