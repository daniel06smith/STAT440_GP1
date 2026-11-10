"""Plan that credits TRIGGERED leaks (cascading / clustered) -- option 1.

    python scripts/contagion_plan.py --start batchwave_s035_corrected_leakaware --spike-scale 0.35 --rounds 2

The planner itself treats pipes as independent; the trigger model enters through two schedule-dependent terms,
recomputed after every round from the current plan's replacement years R:
  victim exposure  v_i(y) = sum_s h_s(y) q_si 1{R_s > y}       added to pipe i's hazard (it can be dragged along)
  source value     E_i    = sum_j q_ij K_j 1{R_j > R_i}         added to pipe i's leak cost, weighted by the share
                                                                 of i's leaks that are its own primary leaks
with h the PRIMARY hazard (batch-wave fit with victims censored, --primary-only, 1964-65 spike x --spike-scale as
measured by probe 3) and q from data/processed/trigger_model.json.

Writes  data/processed/leak_risk_batchwave_primary_s{f}.csv   (primary hazard, for the contagion simulator)
        data/processed/leak_risk_contagion_r{k}.csv          (effective risk file of round k, leak_risk schema)
        submission/submission_contagion_r{k}.csv             (plan of round k, built --leak-aware --yearly corrected)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import batch_wave_hazard as bw  # noqa: E402
import leak_calibration as lc  # noqa: E402
from calibrate_spike_from_probe import risk_with_spike_scale  # noqa: E402
from pipeplan import contagion as ct, simulator  # noqa: E402
from pipeplan.contagion_sim import ContagionWorld, load_trigger_model  # noqa: E402
from pipeplan.pipeline import read_submission  # noqa: E402
from pipeplan.simulator import expected_score, prepare  # noqa: E402
from pipeplan.world import load_world, read_risk_csv  # noqa: E402

YEARS = list(range(2027, 2052))


def primary_risk(f: float) -> Path:
    out = ROOT / f"data/processed/leak_risk_batchwave_primary_s{int(round(f * 100)):03d}.csv"
    if out.exists():
        return out
    d = bw.load_victims(bw.load(ROOT), ROOT)
    fit = bw.fit(d, lc.LAST_YEAR)
    at = np.flatnonzero(~(d["leak_year"] <= lc.LAST_YEAR))
    base = pd.read_csv(ROOT / "data/processed/leak_risk_batchwave_primary.csv")
    assert (base["Pipe ID"].to_numpy() == d["pid"][at]).all()
    risk_with_spike_scale(d, fit, at, YEARS, base, f).to_csv(out, index=False)
    return out


def effective_risk(w, cw, H: np.ndarray, K: np.ndarray, R: np.ndarray, chunk: int = 400):
    """H: (n, 25) primary yearly hazards (0 for poly pipes), R: replacement year index per pipe (25 = never)."""
    idx = np.flatnonzero(w.nonpoly0)
    Hs, Rs, Ks = H[idx], R[idx], K[idx]
    alive_mask = (np.arange(25)[None, :] < Rs[:, None]).astype(np.float32)  # source unreplaced in year y
    HM = (Hs * alive_mask).astype(np.float32)
    V = np.zeros((len(idx), 25), np.float32)
    E = np.zeros(len(idx), np.float32)
    vf = cw.vf[idx].astype(np.float32)
    for a in range(0, len(idx), chunk):
        S = idx[a:a + chunk]
        Q = ct.prob_matrix(cw.tm, cw.A, cw.comp, S, idx, vf=vf)           # (chunk, n_idx): S -> j
        V += Q.T @ HM[a:a + chunk]                                       # victims' extra hazard
        later = Rs[None, :] > Rs[a:a + chunk, None]                      # j still non-poly when S could leak
        E[a:a + chunk] = (Q * later * Ks[None, :]).sum(1)
    return idx, V, E


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="batchwave_s035_corrected_leakaware")
    ap.add_argument("--spike-scale", type=float, default=0.35)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()
    simulator.set_cost_formula("corrected")
    prim = primary_risk(args.spike_scale)
    w = load_world(risk=read_risk_csv(prim, yearly=True))
    cw = ContagionWorld(w, load_trigger_model(ROOT))
    Hcum = w.cum_hazard
    H = np.diff(Hcum, axis=1)[:, :25]                                    # yearly primary hazard per pipe
    K = w.leak_cost.astype(np.float32)
    pr = pd.read_csv(prim).set_index("Pipe ID")
    plan = args.start
    for k in range(1, args.rounds + 1):
        prep = prepare(w, read_submission(ROOT / "submission" / f"submission_{plan}.csv"))
        R = np.minimum(expected_score(w, prep, leak_aware=True)["repl_year"], 25).astype(np.int64)
        idx, V, E = effective_risk(w, cw, H, K, R)
        Htot = H[idx] + V
        share_own = (H[idx].sum(1) + 1e-12) / (Htot.sum(1) + 1e-12)
        eff = pr.loc[w.pipe_id[idx]].copy()
        P = -np.expm1(-Htot)
        for j, Y in enumerate(YEARS):
            eff[f"p_{Y}"] = P[:, j]
        p10 = 1 - np.prod(1 - P[:, :10], axis=1)
        eff["p_leak_annual"] = 1 - (1 - p10) ** 0.1
        eff["cost_if_leak"] = K[idx] + share_own * E
        out = ROOT / f"data/processed/leak_risk_contagion_r{k}.csv"
        eff.reset_index().to_csv(out, index=False)
        print(f"round {k}: from plan {plan}: victim hazard adds {V.sum():.0f} expected leaks over 25 y "
              f"(primary {H[idx].sum():.0f}); source value adds ${(share_own * E).sum() / 1e6:.1f}M to leak costs "
              f"(median per pipe ${np.median(share_own * E):,.0f}, max ${(share_own * E).max():,.0f})", flush=True)
        tag = f"contagion_r{k}"
        subprocess.run([sys.executable, str(ROOT / "scripts/build_plan.py"), "--formula", "corrected", "--tag", tag,
                        "--risk-csv", str(out), "--yearly", "--leak-aware", "--mc-draws", "20", "--mc-seed", "1"],
                       check=True, stdout=subprocess.DEVNULL)
        plan = tag
        print(f"round {k}: built submission_{tag}.csv", flush=True)
    simulator.set_cost_formula("original")


if __name__ == "__main__":
    main()
