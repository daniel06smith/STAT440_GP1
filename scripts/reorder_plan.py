"""Re-order an existing plan's circles under a different leak-risk file (circles unchanged).

    python scripts/reorder_plan.py --base x03_leakaware_setcover40_c1 \
        --risk-csv data/processed/leak_risk_hedge_x03_s035_q50.csv --tag c1_hedge50

Used to separate the two parts of a plan: the circles (spend, the same in every world) and the
execution order (which depends on the assumed leak risk). Same ordering as build_plan.py
(--yearly --leak-aware, corrected formula): Smith's rule + LP year assignment, then drop_redundant.
Writes submission/submission_<tag>.csv after an independent coverage check of the written file.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeplan import simulator  # noqa: E402
from pipeplan.pipeline import independent_coverage_check, order_plan, read_submission, write_submission  # noqa: E402
from pipeplan.simulator import expected_score, prepare  # noqa: E402
from pipeplan.world import load_world, read_risk_csv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--risk-csv", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--min-margin", type=float, default=0.005)
args = ap.parse_args()

simulator.set_cost_formula("corrected")
world = load_world(risk=read_risk_csv(args.risk_csv, yearly=True))
circles = read_submission(ROOT / "submission" / f"submission_{args.base}.csv")
prep, _ = order_plan(world, prepare(world, circles), leak_aware=True, min_margin=args.min_margin, verbose=False)
path = ROOT / "submission" / f"submission_{args.tag}.csv"
write_submission(prep.circles, path)

back = read_submission(path)
cov = independent_coverage_check(world, back)
ex = expected_score(world, prepare(world, back), leak_aware=True)
assert cov["n_uncovered"] == 0, cov
assert ex["last_year_no_leaks"] < 25, "would not finish in 25 years if nothing leaked"
print(f"{args.tag}: {len(back)} circles (base {len(circles)}), uncovered {cov['n_uncovered']}, "
      f"min margin {cov['min_margin_m']:.4f} m, finish-if-no-leaks year {ex['last_year_no_leaks']}, "
      f"expected score under its own risk ${ex['score'] / 1e6:.2f}M")
