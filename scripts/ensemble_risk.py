"""Blend two annual leak-risk files in hazard space.

Example:
    python scripts/ensemble_risk.py \
        --lgbm data/processed/leak_risk_lgbm_nbr.csv \
        --hazard data/processed/leak_risk_hazard_v2.csv \
        --output data/processed/leak_risk_ensemble.csv
        --select-weight
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from lgbm_hazard import (
    VARIANTS,
    add_geometry,
    fit_lgbm,
    lgbm_p,
)
import leak_calibration as lc


def blend_probability(left: pd.Series, right: pd.Series, weight: float) -> np.ndarray:
    """Blend annual probabilities by averaging continuous hazards."""
    p_left = np.clip(left.to_numpy(float), 0.0, 1.0 - 1e-12)
    p_right = np.clip(right.to_numpy(float), 0.0, 1.0 - 1e-12)
    lam = weight * -np.log1p(-p_left) + (1.0 - weight) * -np.log1p(-p_right)
    return -np.expm1(-lam)


def blend_probability_matrix(
    lgbm: np.ndarray,
    hazard: np.ndarray,
    weight: float,
) -> np.ndarray:
    lgbm = np.clip(lgbm, 0.0, 1.0 - 1e-12)
    hazard = np.clip(hazard, 0.0, 1.0 - 1e-12)

    lgbm_lambda = -np.log1p(-lgbm)
    hazard_lambda = -np.log1p(-hazard)

    ensemble_lambda = (
        weight * lgbm_lambda
        + (1.0 - weight) * hazard_lambda
    )

    return -np.expm1(-ensemble_lambda)


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability) - np.log1p(-probability)


def fit_logistic_calibrator(
    probabilities: np.ndarray,
    observed: np.ndarray,
    ridge: float = 1.0,
) -> tuple[float, float]:
    """Fit an intercept/slope calibration model on out-of-sample probabilities."""
    x = _logit(probabilities)
    y = np.asarray(observed, dtype=float)
    if len(x) == 0 or y.sum() == 0 or y.sum() == len(y):
        raise ValueError("calibration data must contain both leak outcomes")

    def objective(beta: np.ndarray) -> float:
        linear = np.clip(beta[0] + beta[1] * x, -30, 30)
        log_likelihood = np.sum(
            y * (-np.logaddexp(0.0, -linear))
            + (1.0 - y) * (-np.logaddexp(0.0, linear))
        )
        return float(-log_likelihood + 0.5 * ridge * beta[1] ** 2)

    initial = np.array([_logit(np.array([np.mean(y)]))[0], 1.0])
    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(-12.0, 2.0), (0.0, 3.0)],
    )
    if not fitted.success:
        raise RuntimeError(f"LightGBM calibration failed: {fitted.message}")
    return float(fitted.x[0]), float(fitted.x[1])


def apply_logistic_calibrator(
    probabilities: np.ndarray,
    calibrator: tuple[float, float],
) -> np.ndarray:
    intercept, slope = calibrator
    linear = np.clip(intercept + slope * _logit(probabilities), -30, 30)
    return 1.0 / (1.0 + np.exp(-linear))


def fit_oos_lgbm_calibrator(
    d: dict,
    cutoff: int,
    cols: list[str],
    window: int = 3,
) -> tuple[float, float]:
    """Fit calibration only from prior-year LightGBM forecasts."""
    forecast_probabilities = []
    observed_leaks = []
    first_origin = max(lc.FIRST_YEAR, cutoff - window)

    for origin in range(first_origin, cutoff):
        fitted = fit_lgbm(d, origin, cols)
        at_risk = np.flatnonzero(~(d["leak_year"] <= origin))
        forecast_probabilities.append(lgbm_p(fitted, d, at_risk, origin + 1, origin))
        observed_leaks.append((d["leak_year"][at_risk] == origin + 1).astype(float))

    if not forecast_probabilities:
        raise ValueError(f"no out-of-sample calibration years available before cutoff {cutoff}")

    return fit_logistic_calibrator(
        np.concatenate(forecast_probabilities),
        np.concatenate(observed_leaks),
    )


def choose_lgbm_weight(
    root: Path,
    variant: str = "base+nbr",
    cutoffs: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025),
    horizon: int = 1,
    calibration_window: int = 3,
    weights: tuple[float, ...] = tuple(np.arange(0, 1.001, 0.025)),
) -> pd.DataFrame:
    d = add_geometry(root, lc.load(root))
    results = []

    for cutoff in cutoffs:
        at_risk = np.flatnonzero(~(d["leak_year"] <= cutoff))
        risk = lc.sub(d, ~(d["leak_year"] <= cutoff))

        # Fit both models using only historical data through cutoff.
        hazard_fit = lc.recalibrate(lc.fit_hazard(d, cutoff), d)
        lgbm_fit = fit_lgbm(d, cutoff, VARIANTS[variant])
        lgbm_calibrator = fit_oos_lgbm_calibrator(
            d, cutoff, VARIANTS[variant], calibration_window
        )

        # Estimate leak cost using only historical leaks through cutoff.
        cost_fit = lc.fit_cost(d, cutoff)
        predicted_cost = lc.cost_hat(cost_fit, risk)

        p_lgbm = np.column_stack([
            apply_logistic_calibrator(
                lgbm_p(lgbm_fit, d, at_risk, cutoff + year, cutoff),
                lgbm_calibrator,
            )
            for year in range(1, horizon + 1)
        ])

        p_hazard = np.column_stack([
            lc.hazard_p(hazard_fit, risk, cutoff + year)
            for year in range(1, horizon + 1)
        ])

        actual = (
            (risk["leak_year"] > cutoff)
            & (risk["leak_year"] <= cutoff + horizon)
        ).astype(float)

        for weight in weights:
            p_ensemble = blend_probability_matrix(
                p_lgbm,
                p_hazard,
                weight,
            )

            cumulative_probability = 1 - np.prod(
                1 - p_ensemble,
                axis=1,
            )

            expected_loss = cumulative_probability * predicted_cost

            actual_cost = np.nan_to_num(risk["cost"], nan=0.0)
            predicted_total = expected_loss.sum()
            actual_total = actual_cost[actual.astype(bool)].sum()
            cost_error = abs(predicted_total - actual_total) / max(actual_total, 1.0)

            top_n = max(1, risk["n"] // 10)
            top = np.argsort(-expected_loss)[:top_n]
            top10_capture = actual_cost[
                actual.astype(bool) & np.isin(np.arange(risk["n"]), top)
            ].sum() / max(actual_total, 1.0)

            results.append({
                "cutoff": cutoff,
                "weight": weight,
                "cost_error": cost_error,
                "predicted_leak_cost": predicted_total,
                "actual_leak_cost": actual_total,
                "top10_cost_capture": top10_capture,
            })

    return pd.DataFrame(results)


def blend_risk(
    lgbm_path: Path,
    hazard_path: Path,
    output_path: Path,
    weight: float,
    lgbm_calibrator: tuple[float, float] | None = None,
) -> None:
    lgbm = pd.read_csv(lgbm_path)
    hazard = pd.read_csv(hazard_path)
    required = {"Pipe ID", "p_leak_annual", "cost_if_leak"}
    for name, frame in (("LightGBM", lgbm), ("hazard", hazard)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} risk file is missing columns: {sorted(missing)}")
        if frame["Pipe ID"].duplicated().any():
            raise ValueError(f"{name} risk file contains duplicate Pipe IDs")

    expected_year_columns = {f"p_{year}" for year in range(2027, 2052)}
    for name, frame in (("LightGBM", lgbm), ("hazard", hazard)):
        missing_years = expected_year_columns - set(frame.columns)
        if missing_years:
            raise ValueError(
                f"{name} risk file is missing yearly columns: {sorted(missing_years)}"
            )
    year_columns = sorted(expected_year_columns)
    if lgbm_calibrator is not None:
        lgbm = lgbm.copy()
        lgbm[year_columns] = apply_logistic_calibrator(
            lgbm[year_columns].to_numpy(float),
            lgbm_calibrator,
        )
        lgbm["p_leak_annual"] = 1.0 - (
            1.0 - lgbm[[f"p_{year}" for year in range(2027, 2037)]]
        ).prod(axis=1) ** 0.1
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
    parser.add_argument("--lgbm-weight", type=float, default=None)
    parser.add_argument("--select-weight", action="store_true")
    parser.add_argument("--variant", default="base+nbr", choices=list(VARIANTS))
    parser.add_argument("--calibration-window", type=int, default=3)
    args = parser.parse_args()

    if args.select_weight:
        root = Path(__file__).resolve().parents[1]
        validation = choose_lgbm_weight(
            root=root,
            variant=args.variant,
            calibration_window=args.calibration_window,
        )

        by_cutoff = (
            validation
            .sort_values(["cutoff", "top10_cost_capture", "cost_error", "weight"],
                         ascending=[True, False, True, True])
            .groupby("cutoff", as_index=False)
            .first()
        )
        print("Best validation weight by cutoff:")
        print(by_cutoff[
            ["cutoff", "weight", "top10_cost_capture", "cost_error"]
        ].to_string(index=False))

        summary = (
            validation
            .groupby("weight", as_index=False)
            .agg(
                mean_top10_cost_capture=("top10_cost_capture", "mean"),
                mean_cost_error=("cost_error", "mean"),
            )
        )
        best_capture = summary["mean_top10_cost_capture"].max()
        near_best = summary[
            summary["mean_top10_cost_capture"] >= best_capture - 0.005
        ].sort_values(["mean_cost_error", "weight"])
        weight = float(near_best.iloc[0]["weight"])
        d = add_geometry(root, lc.load(root))
        lgbm_calibrator = fit_oos_lgbm_calibrator(
            d, lc.LAST_YEAR, VARIANTS[args.variant], args.calibration_window
        )
        print("Aggregate validation results:")
        print(summary.sort_values(
            ["mean_top10_cost_capture", "mean_cost_error"],
            ascending=[False, True],
        ).head(10).to_string(index=False))
        print(f"Selected LightGBM weight: {weight:.2f}")
    elif args.lgbm_weight is not None:
        weight = args.lgbm_weight
        lgbm_calibrator = None
    else:
        weight = 0.5
        lgbm_calibrator = None

    if not 0.0 <= weight <= 1.0:
        parser.error("--lgbm-weight must be between 0 and 1")
    
    blend_risk(args.lgbm, args.hazard, args.output, weight, lgbm_calibrator)


if __name__ == "__main__":
    main()
