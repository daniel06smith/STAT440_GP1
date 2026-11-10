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
ap.add_argument("--yearly", action="store_true",
                help="use the year-by-year columns p_2027..p_2051 (time-aware planner) instead of one constant p")
ap.add_argument("--mc-draws", type=int, default=100)
ap.add_argument(
    "--mc-seed",
    type=int,
    default=None,
    help="Monte Carlo seed; omit for a fresh random seed on each run",
)
ap.add_argument("--h-opt", type=float, default=None,
                help="years of avoided leaks an optional copper/brass pipe must pay for itself in (default 12; ~0 = iron only)")
ap.add_argument("--no-time-aware", action="store_true", help="merge circles on spend only (no delay penalty)")
ap.add_argument("--k-neigh", type=int, default=None, help="merge partners per circle per round (default 8)")
ap.add_argument("--leak-aware", action="store_true",
                help="order circles with the expected cost that credits pipes leaking before their circle runs "
                     "(needs --yearly); see pipeplan/ordering.py")
ap.add_argument("--construct", choices=["merge", "setcover"], default="merge",
                help="circle construction: pairwise merging (construct.Builder, default) or set cover + MILP "
                     "(pipeplan.setcover; ignores timing when choosing circles, ordering is unchanged)")
ap.add_argument("--sc-k", type=int, default=25, help="setcover: nearest target pipes per candidate group")
ap.add_argument("--sc-same-k", type=int, default=25, help="setcover: same-surface nearest pipes per group (0 = off)")
ap.add_argument("--sc-rmax", type=float, default=150.0, help="setcover: largest candidate radius (m)")
ap.add_argument("--sc-seed-plans", nargs="*", default=[],
                help="setcover: submission tags whose circles are added as candidates (cover can only get cheaper)")
ap.add_argument("--sc-cushion", type=float, default=0.05,
                help="setcover: spare radius (m) beyond the farthest end point of a circle's own pipes")
ap.add_argument("--sc-min-margin", type=float, default=0.04,
                help="setcover: a pipe only counts as covered if inside by at least this (m); also used when "
                     "dropping redundant circles")
ap.add_argument("--sc-lam", type=float, default=0.0,
                help="setcover: timing penalty strength ($ per $ of 200 $/m pulled one year early); 0 = off")
ap.add_argument("--sc-pref-plan", default=None,
                help="setcover: submission tag whose (leak-aware expected) replacement years define each pipe's "
                     "preferred year for the timing penalty (needs --sc-lam > 0)")
ap.add_argument("--sc-opt-ref", default=None,
                help="setcover: choose copper/brass inside the MILP (prize-collecting) instead of the standalone "
                     "h_opt rule, valuing each at the year its area is dug in this reference submission tag "
                     "(needs --yearly)")
args = ap.parse_args()
if args.sc_opt_ref and not args.yearly:
    ap.error("--sc-opt-ref needs --yearly")
if args.leak_aware and not args.yearly:
    ap.error("--leak-aware needs --yearly")

t0 = time.time()
simulator.set_cost_formula(args.formula)
risk = read_risk_csv(args.risk_csv, args.id_col, args.p_col, args.cost_col, args.horizon_years, yearly=args.yearly)
world = load_world(risk=risk)
pkw = {}
if args.h_opt is not None:
    pkw["h_opt"] = args.h_opt
if args.no_time_aware:
    pkw["time_aware"] = False
if args.k_neigh is not None:
    pkw["k_neigh"] = args.k_neigh
if args.construct == "merge":
    res = build_plan(world, Params(verbose=True, **pkw), leak_aware=args.leak_aware)
    prep = res["prep"]
else:
    import numpy as np  # noqa: E402
    from pipeplan import setcover  # noqa: E402
    from pipeplan.pipeline import order_plan  # noqa: E402

    extra = [read_submission(ROOT / "submission" / f"submission_{t}.csv") for t in args.sc_seed_plans]
    pref = None
    if args.sc_lam > 0:
        if not args.sc_pref_plan:
            ap.error("--sc-lam needs --sc-pref-plan")
        ref = prepare(world, read_submission(ROOT / "submission" / f"submission_{args.sc_pref_plan}.csv"))
        pref = expected_score(world, ref, leak_aware=True)["repl_year"]
    opt_pref = None
    if args.sc_opt_ref:
        ref = prepare(world, read_submission(ROOT / "submission" / f"submission_{args.sc_opt_ref}.csv"))
        opt_pref = expected_score(world, ref, leak_aware=True)["repl_year"]
    circles = setcover.build_circles(world, Params(verbose=False, **pkw), k_max=args.sc_k,
                                     same_surface_k=args.sc_same_k, r_max=args.sc_rmax,
                                     extra_circles=np.vstack(extra) if extra else None,
                                     cushion=args.sc_cushion, min_margin=args.sc_min_margin,
                                     pref_year=pref, lam=args.sc_lam, opt_pref_year=opt_pref)
    prep, _ = order_plan(world, prepare(world, circles), leak_aware=args.leak_aware,
                         min_margin=args.sc_min_margin)
    pkw.update(construct="setcover", sc_k=args.sc_k, sc_same_k=args.sc_same_k, sc_rmax=args.sc_rmax,
               sc_seed_plans=args.sc_seed_plans, sc_cushion=args.sc_cushion,
               sc_min_margin=args.sc_min_margin, sc_lam=args.sc_lam, sc_pref_plan=args.sc_pref_plan,
               sc_opt_ref=args.sc_opt_ref)

sub_path = ROOT / "submission" / f"submission_{args.tag}.csv"
write_submission(prep.circles, sub_path)

# --- round-trip: re-read the FILE and re-check it independently of everything above
circles_from_file = read_submission(sub_path)
cov = independent_coverage_check(world, circles_from_file)
prep_file = prepare(world, circles_from_file)
ex = expected_score(world, prep_file, leak_aware=args.leak_aware)
mc = monte_carlo(world, prep_file, n_draws=args.mc_draws, seed=args.mc_seed)

table = circle_table(world, prep_file)
table.to_csv(ROOT / f"data/processed/circles_{args.tag}.csv", index=False)

summary = {
    "tag": args.tag,
    "cost_formula": args.formula,
    "risk_file": str(Path(args.risk_csv).name),
    "risk_probability_horizon_years": args.horizon_years,
    "yearly_risk": args.yearly,
    "leak_aware": args.leak_aware,
    "builder_params": pkw,
    "monte_carlo_seed": args.mc_seed,
    "n_circles": int(len(circles_from_file)),
    "independent_coverage": {k: v for k, v in cov.items() if k != "uncovered_ids"},
    "expected": {k: ex[k] for k in ("feasible", "spend", "surplus", "leak_cost", "score", "last_year",
                                    "last_year_no_leaks")},
    f"monte_carlo_{args.mc_draws}_draws": mc,
    "seconds": round(time.time() - t0, 1),
}
(ROOT / f"data/processed/plan_{args.tag}_summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps(summary, indent=2, default=float))
assert cov["n_uncovered"] == 0, "some iron pipe is not covered!"
assert ex["last_year_no_leaks"] < 25, "plan would not finish in 25 years if nothing leaked!"
