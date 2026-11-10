"""Diagnostic submission 3: how much 2027 leak cost do the plan's first-year circles really prevent?

    python scripts/probe_swap_years.py          # -> submission/submission_probe_swap.csv, data/processed/probe_swap.json

Why: more than half of the predicted leak cost of the uploaded plan falls in 2027, mostly the age spike of the
1964-65 batches -- the model's biggest single forecast, and the most likely place for the ~$14M by which every
public score beats every model. The cost rule and today's network state are already verified (probes 1-2).

How: take the uploaded plan and swap the block of circles it runs at the start of 2027 (B0, fixed: no leak can
happen before 1 Jan 2027) with the block it would run at the start of 2028 (B1: the next ~$10M at full cost).
B0's pipes then sit through 2027; B1's are protected one year earlier. The public score change is mostly
    (2027 leak cost B1's pipes avoid)  -  (2027 leak cost B0's pipes now suffer)  +  (replacement spend saved on
    the pipes that leaked first)
and is predicted below under every risk world we have (paired Monte Carlo, heavy-tailed costs, corrected formula).
Use the observed change ONLY to judge batch-level facts (how large / early the 1964-65 spike is): which individual
pipes leak in the public 2027 is redrawn in the private future.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeplan import simulator  # noqa: E402
from pipeplan.contagion_sim import ContagionWorld, load_trigger_model, monte_carlo_events  # noqa: E402
from pipeplan.ordering import _first_cover, permute  # noqa: E402
from pipeplan.pipeline import independent_coverage_check, read_submission, write_submission  # noqa: E402
from pipeplan.simulator import empirical_cost_ratios, expected_score, monte_carlo, prepare, survival  # noqa: E402
from pipeplan.world import ANNUAL_BUDGET, load_world, read_risk_csv  # noqa: E402

PLAN = "batchwave_corrected_leakaware"
WORLDS = {  # name -> (risk file, contagion?)
    "batchwave": ("leak_risk_batchwave.csv", False),
    "batchwave + triggered leaks": ("leak_risk_batchwave_primary.csv", True),
    "batchwave_batch (1965 wrought later)": ("leak_risk_batchwave_batch.csv", False),
    "batchwave_extrap": ("leak_risk_batchwave_extrap.csv", False),
    "w060 ensemble": ("leak_risk_ensemble_w060.csv", False),
}


def main() -> None:
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    simulator.set_cost_formula("corrected")
    w0 = load_world(risk=read_risk_csv(ROOT / "data/processed/leak_risk_batchwave.csv", yearly=True))
    base_c = read_submission(ROOT / "submission" / f"submission_{PLAN}.csv")
    prep = prepare(w0, base_c)
    _, full = _first_cover(w0, prep)  # full cost of every circle in the uploaded order (no leaks)
    cum = np.cumsum(full)
    b0 = np.flatnonzero(cum <= ANNUAL_BUDGET)                       # exactly what runs at the start of 2027
    b1 = np.flatnonzero((cum > ANNUAL_BUDGET) & (cum <= 2 * ANNUAL_BUDGET))
    assert b0[-1] + 1 == b1[0]
    rest = np.arange(b1[-1] + 1, prep.n)
    order = np.r_[b1, b0, rest]
    probe = base_c[order]
    out = ROOT / "submission/submission_probe_swap.csv"
    write_submission(probe, out)
    circ = read_submission(out)
    cov = independent_coverage_check(w0, circ)
    assert cov["n_uncovered"] == 0
    assert expected_score(w0, prepare(w0, circ))["last_year_no_leaks"] < 25

    # what is in the two blocks
    pipes = pd.read_csv(ROOT / "data/raw/pipes.csv", parse_dates=["Lay date"]).set_index("Pipe ID").loc[w0.pipe_id]
    batch = (pipes["Lay date"].dt.year.fillna(0).astype(int).astype(str) + " " + pipes["Material"].fillna("?")).to_numpy()
    new, _ = _first_cover(w0, prep)
    info = {}
    for name, blk in (("B0 (now 2027 -> 2028)", b0), ("B1 (now 2028 -> 2027)", b1)):
        pp = np.concatenate([new[j] for j in blk])
        p27 = 1 - survival(w0, pp, 1)
        top = pd.Series(p27, index=batch[pp]).groupby(level=0).agg(["size", "sum"]).sort_values("sum", ascending=False).head(5)
        info[name] = {"circles": int(len(blk)), "pipes": int(len(pp)), "full_cost": float(full[blk].sum()),
                      "expected_2027_leaks_if_unreplaced (batchwave)": float(p27.sum()),
                      "expected_2027_leak_cost_if_unreplaced (batchwave)": float((p27 * w0.leak_cost[pp]).sum()),
                      "top_batches (pipes, expected 2027 leaks)": {k: [int(r["size"]), round(float(r["sum"]), 1)] for k, r in top.iterrows()}}
    print(json.dumps(info, indent=1))

    ratios = empirical_cost_ratios(ROOT)
    res = {"blocks": info, "worlds": {}}
    for wname, (fname, contagion) in WORLDS.items():
        w = load_world(risk=read_risk_csv(ROOT / "data/processed" / fname, yearly=True))
        pa, pb = prepare(w, base_c), prepare(w, circ)
        if contagion:
            cw = ContagionWorld(w, load_trigger_model(ROOT))
            a = monte_carlo_events(w, pa, cw, n_draws=max(draws // 3, 20), seed=21, cost_ratios=ratios, keep_draws=True)
            b = monte_carlo_events(w, pb, cw, n_draws=max(draws // 3, 20), seed=21, cost_ratios=ratios, keep_draws=True)
        else:
            a = monte_carlo(w, pa, n_draws=draws, seed=21, cost_ratios=ratios, keep_draws=True)
            b = monte_carlo(w, pb, n_draws=draws, seed=21, cost_ratios=ratios, keep_draws=True)
        dlt = (b["score_draws"] - a["score_draws"]) / 1e6
        res["worlds"][wname] = {"pred_change_mean_M": float(dlt.mean()), "sd_M": float(dlt.std(ddof=1)),
                                "p05_M": float(np.percentile(dlt, 5)), "p95_M": float(np.percentile(dlt, 95)),
                                "base_score_mean_M": float(a["score_mean"] / 1e6)}
        print(f"{wname:38s} predicted change {dlt.mean():+6.2f}M  (sd {dlt.std(ddof=1):.2f}, 90%: {np.percentile(dlt, 5):+.2f} to "
              f"{np.percentile(dlt, 95):+.2f})   base plan ${a['score_mean']/1e6:.1f}M", flush=True)
    simulator.set_cost_formula("original")
    (ROOT / "data/processed/probe_swap.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {out} (blocks of {len(b0)} and {len(b1)} circles swapped); coverage OK")


if __name__ == "__main__":
    main()
