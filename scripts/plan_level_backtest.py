"""Rolling plan-level backtest for ensemble weights.

The historical ``horizon`` controls which later observed leaks are scored.
The replacement plan always receives the full 25-year project budget.

Example:
    python scripts/plan_level_backtest.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import leak_calibration as lc  # noqa: E402
from ensemble_risk import blend_probability  # noqa: E402
from lgbm_hazard import VARIANTS, add_geometry, fit_lgbm, lgbm_p  # noqa: E402
from pipeplan.construct import Params  # noqa: E402
from pipeplan.pipeline import build_plan  # noqa: E402
from pipeplan.simulator import PROJECT_YEARS, set_cost_formula, simulate_project  # noqa: E402
from pipeplan.world import build_world  # noqa: E402


def build_backtest_case(
    d: dict,
    cutoff: int,
    horizon: int,
    weight: float,
    variant: str,
    formula: str,
) -> dict:
    """Fit a cutoff-only risk model, build its plan, and score observed future leaks."""
    at_risk_mask = ~(d["leak_year"] <= cutoff)
    at_risk = np.flatnonzero(at_risk_mask)
    risk = lc.sub(d, at_risk_mask)

    hazard_fit = lc.recalibrate(lc.fit_hazard(d, cutoff), d)
    lgbm_fit = fit_lgbm(d, cutoff, VARIANTS[variant])
    p_lgbm = lgbm_p(lgbm_fit, d, at_risk, cutoff + 1, cutoff)
    p_hazard = lc.hazard_p(hazard_fit, risk, cutoff + 1)
    p = blend_probability(pd.Series(p_lgbm), pd.Series(p_hazard), weight)

    cost_fit = lc.fit_cost(d, cutoff)
    predicted_cost = lc.cost_hat(cost_fit, risk)
    risk_frame = pd.DataFrame({
        "Pipe ID": risk["pid"],
        "p_annual": p,
        "leak_cost": predicted_cost,
    })
    pipes = pd.read_csv(ROOT / "data/raw/pipes.csv")
    leaked_ids = set(d["pid"][d["leak_year"] <= cutoff])
    world = build_world(pipes, leaked_ids, risk_frame)

    set_cost_formula(formula)
    plan = build_plan(world, Params(verbose=False), verbose=False)

    leak_time = np.full(world.n, np.inf)
    leak_cost = world.leak_cost.copy()
    actual_year = d["leak_year"]
    actual_cost = d["cost"]
    pipe_positions = {pipe_id: i for i, pipe_id in enumerate(world.pipe_id)}
    for pipe_id, year, cost in zip(d["pid"], actual_year, actual_cost):
        if np.isnan(year) or year <= cutoff or year > cutoff + horizon:
            continue
        position = pipe_positions.get(pipe_id)
        if position is not None:
            leak_time[position] = year - (cutoff + 1)
            leak_cost[position] = cost

    result = simulate_project(
        world,
        plan["prep"],
        leak_time=leak_time,
        leak_cost=leak_cost,
    )
    return {
        "cutoff": cutoff,
        "horizon": horizon,
        "includes_2026": cutoff < 2026 <= cutoff + horizon,
        "weight": weight,
        "score": result["score"],
        "leak_cost": result["leak_cost"],
        "spend": result["spend"],
        "n_leaks": result["n_leaks"],
        "feasible": result["feasible"],
        "last_year": result["last_year"],
        "project_years": PROJECT_YEARS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["2021:3", "2022:3", "2023:3", "2024:2", "2025:1"],
        help="cutoff:horizon pairs; e.g. 2023:3 2024:2 2025:1",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.6, 0.75, 1.0],
    )
    parser.add_argument("--variant", default="base+nbr", choices=list(VARIANTS))
    parser.add_argument("--formula", choices=["original", "corrected"], default="corrected")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/plan_level_backtest.csv")
    args = parser.parse_args()

    d = add_geometry(ROOT, lc.load(ROOT))
    cases = []
    for spec in args.cases:
        try:
            cutoff_text, horizon_text = spec.split(":", 1)
            cases.append((int(cutoff_text), int(horizon_text)))
        except ValueError as exc:
            raise ValueError(f"invalid case {spec!r}; expected cutoff:horizon") from exc

    rows = []
    for cutoff, horizon in cases:
        for weight in args.weights:
            row = build_backtest_case(
                d, cutoff, horizon, weight, args.variant, args.formula
            )
            rows.append(row)
            print(
                f"{cutoff} weight={weight:.2f}: score=${row['score']:,.0f} "
                f"spend=${row['spend']:,.0f} feasible={row['feasible']}"
            )
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
