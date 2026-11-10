"""Diagnostic submission 4: is the evaluator's $200/m length charge what we think it is?

    python scripts/probe_length_charge.py      # -> submission/submission_probe_length.csv, data/processed/probe_length.json

Why: every probe so far reproduced the CHANGE in public score but every public score sits ~$14-16M above every
model (even after probe 3 shrank the 1964-65 spike). A cost that is about the same for every plan would do that;
the $200/m length charge (~$75-89M of every plan's spend) is the one part of the cost rule no probe has tested.

How: append, at the very END of the uploaded plan, single-pipe circles around non-polyurethane pipes the plan
leaves out (optional copper / brass it chose not to replace). Long pipes on cheap surfaces are preferred, so
nearly all of each added unit's cost is 200 $/m. Every other pipe inside an added circle is already polyurethane
by then (replaced earlier by the plan), so nothing else changes. Predicted public change (paired MC):
    -(fees + area + 200 x metres) for the added units that run   + (their leaks avoided after they run)
and, for reading, the same with the length charge removed. Fixed fact of the cost rule -> no overfitting.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeplan import simulator  # noqa: E402
from pipeplan.geometry import min_enclosing_circle  # noqa: E402
from pipeplan.ordering import _first_cover  # noqa: E402
from pipeplan.pipeline import independent_coverage_check, read_submission, write_submission  # noqa: E402
from pipeplan.simulator import empirical_cost_ratios, expected_score, fixed_cost, monte_carlo, prepare, survival  # noqa: E402
from pipeplan.world import LINEAR_RATE, RATE_BY_MASK, load_world, read_risk_csv  # noqa: E402

PLAN = "batchwave_corrected_leakaware"
TARGET_LINEAR = 3.0e6


def main() -> None:
    simulator.set_cost_formula("corrected")
    w = load_world(risk=read_risk_csv(ROOT / "data/processed/leak_risk_batchwave_s035.csv", yearly=True))
    base = read_submission(ROOT / "submission" / f"submission_{PLAN}.csv")
    prep = prepare(w, base)
    new, _ = _first_cover(w, prep)
    owner = np.full(w.n, -1)
    for j, pp in enumerate(new):
        owner[pp] = j
    free = np.flatnonzero(w.nonpoly0 & (owner < 0))          # non-poly pipes no circle covers
    assert not w.required[free].any()
    ex = expected_score(w, prep, leak_aware=True)
    end_year = int(ex["last_year_no_leaks"])                  # added units run no earlier than this
    cands = []
    for i in free:
        pts = np.array([[w.x1[i], w.y1[i]], [w.x2[i], w.y2[i]]])
        cx, cy, r = min_enclosing_circle(pts)
        r = math.ceil((r + 0.05) * 100) / 100
        cx, cy = round(cx, 2), round(cy, 2)
        m = w.members(cx, cy, r)
        others = m[m != i]
        if (w.nonpoly0[others] & (owner[others] < 0)).any():  # would also replace another left-out pipe
            continue
        if i not in m:
            continue
        rate = RATE_BY_MASK[np.bitwise_or.reduce(w.surf_bit[m])]
        F = fixed_cost(rate, r)
        lin = LINEAR_RATE * w.length[i]
        p_left = float(survival(w, np.array([i]), min(end_year + 1, 25))[0])  # P(still non-poly when it runs)
        cands.append((lin / (F + lin) * p_left, i, cx, cy, r, F, lin, p_left))
    cands.sort(reverse=True)
    chosen, tot = [], 0.0
    for c in cands:
        if tot >= TARGET_LINEAR:
            break
        chosen.append(c)
        tot += c[6]
    add = np.array([[c[2], c[3], c[4]] for c in chosen])
    probe = np.vstack([base, add])
    out = ROOT / "submission/submission_probe_length.csv"
    write_submission(probe, out)
    circ = read_submission(out)
    assert independent_coverage_check(w, circ)["n_uncovered"] == 0
    exp_ = expected_score(w, prepare(w, circ))
    assert exp_["last_year_no_leaks"] < 25

    F = np.array([c[5] for c in chosen]); L = np.array([c[6] for c in chosen]); P = np.array([c[7] for c in chosen])
    res = {"plan": PLAN, "n_added": len(chosen), "added_length_charge": float(L.sum()), "added_fees_area": float(F.sum()),
           "mean_p_still_nonpoly": float(P.mean()),
           "static_pred_if_200_per_m": float(-((F + L) * P).sum()), "static_pred_if_no_length_charge": float(-(F * P).sum()),
           "worlds": {}}
    ratios = empirical_cost_ratios(ROOT)
    for wname, fname in (("independent, spike x0.35", "leak_risk_batchwave_s035.csv"), ("w060 ensemble", "leak_risk_ensemble_w060.csv")):
        ww = load_world(risk=read_risk_csv(ROOT / "data/processed" / fname, yearly=True))
        a = monte_carlo(ww, prepare(ww, base), n_draws=200, seed=41, cost_ratios=ratios, keep_draws=True)
        b = monte_carlo(ww, prepare(ww, circ), n_draws=200, seed=41, cost_ratios=ratios, keep_draws=True)
        dlt = (b["score_draws"] - a["score_draws"]) / 1e6
        res["worlds"][wname] = {"pred_change_mean_M": float(dlt.mean()), "sd_M": float(dlt.std(ddof=1)),
                                "p05_M": float(np.percentile(dlt, 5)), "p95_M": float(np.percentile(dlt, 95))}
    simulator.set_cost_formula("original")
    (ROOT / "data/processed/probe_length.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"\nwrote {out}: {len(chosen)} single-pipe units appended after the plan's {prep.n} circles; coverage OK")


if __name__ == "__main__":
    main()
