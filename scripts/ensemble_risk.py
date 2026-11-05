"""Blend two annual leak-risk files in hazard space.

Example:
    python scripts/ensemble_risk.py \
        --lgbm data/processed/leak_risk_lgbm_nbr.csv \
        --hazard data/processed/leak_risk_hazard_v2.csv \
        --output data/processed/leak_risk_ensemble.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def blend_probability(left: pd.Series, right: pd.Series, weight: float) -> np.ndarray:
    """Blend annual probabilities by averaging continuous hazards."""
    p_left = np.clip(left.to_numpy(float), 0.0, 1.0 - 1e-12)
    p_right = np.clip(right.to_numpy(float), 0.0, 1.0 - 1e-12)
    lam = weight * -np.log1p(-p_left) + (1.0 - weight) * -np.log1p(-p_right)
    return -np.expm1(-lam)


def blend_risk(lgbm_path: Path, hazard_path: Path, output_path: Path, weight: float) -> None:
    lgbm = pd.read_csv(lgbm_path)
    hazard = pd.read_csv(hazard_path)
    required = {"Pipe ID", "p_leak_annual", "cost_if_leak"}
    for name, frame in (("LightGBM", lgbm), ("hazard", hazard)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} risk file is missing columns: {sorted(missing)}")
        if frame["Pipe ID"].duplicated().any():
            raise ValueError(f"{name} risk file contains duplicate Pipe IDs")

    year_columns = sorted(
        set(lgbm.columns) & set(hazard.columns) & {"p_" + str(year) for year in range(2027, 2052)}
    )
    base_columns = ["Pipe ID", "p_leak_annual", "cost_if_leak"]
    merged = lgbm[base_columns + year_columns].merge(
        hazard[base_columns + year_columns],
        on="Pipe ID",
        how="outer",
        suffixes=("_lgbm", "_hazard"),
        validate="one_to_one",
    )
    if merged[["p_leak_annual_lgbm", "p_leak_annual_hazard"]].isna().any().any():
        raise ValueError("the two risk files do not contain the same Pipe IDs")

    out = pd.DataFrame({"Pipe ID": merged["Pipe ID"]})
    out["p_leak_annual"] = blend_probability(
        merged["p_leak_annual_lgbm"], merged["p_leak_annual_hazard"], weight
    )
    out["cost_if_leak"] = (
        weight * merged["cost_if_leak_lgbm"]
        + (1.0 - weight) * merged["cost_if_leak_hazard"]
    )

    for column in year_columns:
        out[column] = blend_probability(merged[f"{column}_lgbm"], merged[f"{column}_hazard"], weight)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"wrote {output_path} ({len(out)} pipes; lgbm weight={weight:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lgbm", type=Path, required=True)
    parser.add_argument("--hazard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lgbm-weight", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.lgbm_weight <= 1.0:
        parser.error("--lgbm-weight must be between 0 and 1")
    blend_risk(args.lgbm, args.hazard, args.output, args.lgbm_weight)


if __name__ == "__main__":
    main()
