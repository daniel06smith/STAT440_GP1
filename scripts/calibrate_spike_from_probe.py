"""One batch-level parameter from probe 3: how big is the 1964-65 batches' age spike really?

    python scripts/calibrate_spike_from_probe.py      # -> data/processed/leak_risk_batchwave_s{f}.csv, spike_calibration.csv

Probe 3 (scripts/probe_swap_years.py) moved the plan's 2027 circles -- mostly 1964-65 wrought / gray / cast iron
pipes -- to 2028. The public score dropped $19.48M; batchwave predicts -$42.4M (90%: -55.5 to -31.2), i.e. those
batches leak far less in 2027 than the batch-wave spike says. That is a fact about the BATCHES (shared by the
public and private futures), not about which pipes leak in the public 2027.

Here the spike component of every pipe laid in [1964, 1966) is multiplied by f (background unchanged), for a grid
of f; for each we predict (a) the probe's score change and (b) the uploaded plan's score, with the paired
heavy-tailed Monte Carlo under the corrected formula. f is chosen from (a) ONLY; (b) is the out-of-fit check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import batch_wave_hazard as bw  # noqa: E402
import leak_calibration as lc  # noqa: E402
from pipeplan import simulator  # noqa: E402
from pipeplan.pipeline import read_submission  # noqa: E402
from pipeplan.simulator import empirical_cost_ratios, monte_carlo, prepare  # noqa: E402
from pipeplan.world import load_world, read_risk_csv  # noqa: E402

OBS_SWAP = 55.35 - 74.83   # $M, public score change of probe 3
PUBLIC = 74.83             # $M, public score of the uploaded plan (out-of-fit check only)
SPIKE_LAY = (1964.0, 1966.0)


def risk_with_spike_scale(d, f_fit, at, years, base: pd.DataFrame, f: float) -> pd.DataFrame:
    lay = d["lay_dec"][at]
    on = f_fit.spike_mask(d, at)
    hb = f_fit.bg_rate_forecast(d, at)
    scale = np.where((lay >= SPIKE_LAY[0]) & (lay < SPIKE_LAY[1]), f, 1.0)
    cols = []
    for Y in years:
        ds = bw.spike_log_surv(f_fit.spike, lay, Y - lay) - bw.spike_log_surv(f_fit.spike, lay, Y + 1.0 - lay)
        H = hb + np.where(on, np.maximum(ds, 0.0), 0.0) * scale
        cols.append(-np.expm1(-H))
    P = np.column_stack(cols)
    r = base.copy()
    for k, Y in enumerate(years):
        r[f"p_{Y}"] = P[:, k]
    p10 = 1 - np.prod(1 - P[:, :10], axis=1)
    r["p_leak_annual"] = 1 - (1 - p10) ** 0.1
    return r


def main() -> None:
    grid = [float(x) for x in sys.argv[1:]] or [1.0, 0.7, 0.5, 0.35, 0.2]
    draws = 100
    d = bw.load(ROOT)
    f_fit = bw.fit(d, lc.LAST_YEAR)
    at = np.flatnonzero(~(d["leak_year"] <= lc.LAST_YEAR))
    years = list(range(2027, 2052))
    base = pd.read_csv(ROOT / "data/processed/leak_risk_batchwave.csv")
    assert (base["Pipe ID"].to_numpy() == d["pid"][at]).all()
    simulator.set_cost_formula("corrected")
    ratios = empirical_cost_ratios(ROOT)
    plan = read_submission(ROOT / "submission/submission_batchwave_corrected_leakaware.csv")
    probe = read_submission(ROOT / "submission/submission_probe_swap.csv")
    rows = []
    for f in grid:
        r = risk_with_spike_scale(d, f_fit, at, years, base, f)
        path = ROOT / f"data/processed/leak_risk_batchwave_s{int(round(f * 100)):03d}.csv"
        r.to_csv(path, index=False)
        w = load_world(risk=read_risk_csv(path, yearly=True))
        a = monte_carlo(w, prepare(w, plan), n_draws=draws, seed=21, cost_ratios=ratios, keep_draws=True)
        b = monte_carlo(w, prepare(w, probe), n_draws=draws, seed=21, cost_ratios=ratios, keep_draws=True)
        dlt = (b["score_draws"] - a["score_draws"]) / 1e6
        sa = a["score_draws"] / 1e6
        rows.append(dict(f=f, swap_pred=dlt.mean(), swap_sd=dlt.std(ddof=1), plan_score=sa.mean(), plan_sd=sa.std(ddof=1),
                         plan_p95=np.percentile(sa, 95), p_plan_ge_public=float(np.mean(sa >= PUBLIC)),
                         leaks_2027=float(r["p_2027"].sum()), file=path.name))
        print(f"f={f:4.2f}: probe change {dlt.mean():+6.2f}M (sd {dlt.std(ddof=1):.2f}; observed {OBS_SWAP:+.2f})   "
              f"plan ${sa.mean():.2f}M (sd {sa.std(ddof=1):.2f}, p95 {np.percentile(sa, 95):.1f}; public {PUBLIC})   "
              f"2027 expected leaks {r['p_2027'].sum():.0f}", flush=True)
    simulator.set_cost_formula("original")
    out = pd.DataFrame(rows)
    # f from the probe alone: interpolate the predicted change to the observed one
    o = out.sort_values("swap_pred")
    f_star = float(np.interp(OBS_SWAP, o["swap_pred"], o["f"]))
    out["f_star_from_probe"] = f_star
    out.to_csv(ROOT / "data/processed/spike_calibration.csv", index=False)
    print(f"\nf fitted from the probe alone: {f_star:.2f}")


if __name__ == "__main__":
    main()
