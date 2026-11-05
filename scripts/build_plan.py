"""Build a whole plan from scratch and write the outputs.

    python scripts/build_plan.py                       # v1: Cheryl's model, original cost formula

    # any other leak model (what the numbers in the probability column mean is set by --horizon-years)
    python scripts/build_plan.py --tag hazard --formula corrected \
        --risk-csv data/processed/priority_all_hazard.csv \
        --id-col id --p-col p --cost-col cost_if_leak --horizon-years 5

Outputs (TAG = --tag)
  submission/submission_TAG.csv          <- the file to upload to the course website
  data/processed/circles_TAG.csv         <- same circles, one row each, with costs/years/diagnostics
  data/processed/plan_TAG_summary.json   <- headline numbers (model-based expectation + Monte Carlo)

--horizon-years: what the probability column means.
   1 = already a yearly probability (Cheryl's leak_risk_cheryl.csv)
   8 = probability of leaking over 2019-2026 (Kenson's priority_all.csv / priority_all_stepwise.csv)
   5 = probability of leaking in the next 5 years (Kenson's priority_all_hazard.csv)
--formula: 'original' = (surface_rate x area)^0.85 - what the website's data page states
          (checked 28 Oct 2026); 'corrected' = surface_rate x area^0.85, the reading Wendy's
          notebook uses. Only 'original' should be used for real submissions.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeplan import simulator  # noqa: E402
from pipeplan.construct import Params  # noqa: E402
from pipeplan.pipeline import (  # noqa: E402
    build_plan, circle_table, independent_coverage_check, read_submission, write_submission,
)
from pipeplan.simulator import expected_score, monte_carlo, prepare  # noqa: E402
from pipeplan.world import load_world, read_risk_csv  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--tag", default="v1")
ap.add_argument("--formula", choices=["original", "corrected"], default="original")
ap.add_argument("--risk-csv", default=str(ROOT / "data/processed/leak_risk_cheryl.csv"))
ap.add_argument("--id-col", default="Pipe ID")
ap.add_argument("--p-col", default="p_leak_annual")
ap.add_argument("--cost-col", default="cost_if_leak")
ap.add_argument("--horizon-years", type=float, default=1.0)
ap.add_argument("--mc-draws", type=int, default=100)
args = ap.parse_args()

t0 = time.time()
simulator.set_cost_formula(args.formula)
risk = read_risk_csv(args.risk_csv, args.id_col, args.p_col, args.cost_col, args.horizon_years)
world = load_world(risk=risk)
res = build_plan(world, Params(verbose=True))
prep = res["prep"]

sub_path = ROOT / "submission" / f"submission_{args.tag}.csv"
write_submission(prep.circles, sub_path)

# --- round-trip: re-read the FILE and re-check it independently of everything above
circles_from_file = read_submission(sub_path)
cov = independent_coverage_check(world, circles_from_file)
prep_file = prepare(world, circles_from_file)
ex = expected_score(world, prep_file)
mc = monte_carlo(world, prep_file, n_draws=args.mc_draws, seed=1)

table = circle_table(world, prep_file)
table.to_csv(ROOT / f"data/processed/circles_{args.tag}.csv", index=False)

summary = {
    "tag": args.tag,
    "cost_formula": args.formula,
    "risk_file": str(Path(args.risk_csv).name),
    "risk_probability_horizon_years": args.horizon_years,
    "n_circles": int(len(circles_from_file)),
    "independent_coverage": {k: v for k, v in cov.items() if k != "uncovered_ids"},
    "expected": {k: ex[k] for k in ("feasible", "spend", "surplus", "leak_cost", "score", "last_year")},
    f"monte_carlo_{args.mc_draws}_draws": mc,
    "seconds": round(time.time() - t0, 1),
}
(ROOT / f"data/processed/plan_{args.tag}_summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps(summary, indent=2, default=float))
assert cov["n_uncovered"] == 0, "some iron pipe is not covered!"
