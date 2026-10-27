"""Build the whole plan from scratch and write the outputs.

    python scripts/build_plan.py

Outputs
  submission/submission_v1.csv          <- the file to upload to the course website
  data/processed/circles_v1.csv         <- same circles, one row each, with costs/years/diagnostics
  data/processed/plan_v1_summary.json   <- headline numbers (model-based expectation + Monte Carlo)
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeplan.construct import Params  # noqa: E402
from pipeplan.pipeline import (  # noqa: E402
    build_plan, circle_table, independent_coverage_check, read_submission, write_submission,
)
from pipeplan.simulator import expected_score, monte_carlo, prepare  # noqa: E402
from pipeplan.world import load_world  # noqa: E402

t0 = time.time()
world = load_world()
res = build_plan(world, Params(verbose=True))
prep = res["prep"]

sub_path = ROOT / "submission" / "submission_v1.csv"
write_submission(prep.circles, sub_path)

# --- round-trip: re-read the FILE and re-check it independently of everything above
circles_from_file = read_submission(sub_path)
cov = independent_coverage_check(world, circles_from_file)
prep_file = prepare(world, circles_from_file)
ex = expected_score(world, prep_file)
mc = monte_carlo(world, prep_file, n_draws=100, seed=1)

table = circle_table(world, prep_file)
table.to_csv(ROOT / "data/processed/circles_v1.csv", index=False)

summary = {
    "n_circles": int(len(circles_from_file)),
    "independent_coverage": {k: v for k, v in cov.items() if k != "uncovered_ids"},
    "expected": {k: ex[k] for k in ("feasible", "spend", "surplus", "leak_cost", "score", "last_year")},
    "monte_carlo_100_draws": mc,
    "seconds": round(time.time() - t0, 1),
}
(ROOT / "data/processed/plan_v1_summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps(summary, indent=2, default=float))
assert cov["n_uncovered"] == 0, "some iron pipe is not covered!"
