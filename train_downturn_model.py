from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

PANEL_FINAL_PATH = DATA_DIR / "worldbank_panel_final.csv"
PANEL_FALLBACK_PATH = DATA_DIR / "worldbank_panel.csv"


def load_panel() -> pd.DataFrame:
    if PANEL_FINAL_PATH.exists():
        return pd.read_csv(PANEL_FINAL_PATH)
    if PANEL_FALLBACK_PATH.exists():
        return pd.read_csv(PANEL_FALLBACK_PATH)
    raise FileNotFoundError(
        "Could not find input panel CSV. Expected one of:\n"
        f"- {PANEL_FINAL_PATH}\n"
        f"- {PANEL_FALLBACK_PATH}"
    )


def slope_3(values: np.ndarray) -> float:
    """
    Least-squares slope across three consecutive observations.
    """
    x = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.asarray(values, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def add_trend_feature(df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    def rolling_slope(series: pd.Series) -> pd.Series:
        return series.rolling(3).apply(lambda x: slope_3(np.array(x)), raw=False)

    df[target_col] = (
        df.groupby("country_code")[source_col]
        .transform(rolling_slope)
    )
    return df


def build_modelling_dataset(panel_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "country_code",
        "year",
        "unemployment",
        "inflation",
        "gdp_growth",
        "life_expectancy",
        "population_growth",
    ]

    # Optional but preferred for nicer dashboard labels
    keep_country_name = "country" in panel_df.columns

    missing = [c for c in required_cols if c not in panel_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in panel data: {missing}")

    cols = required_cols.copy()
    if keep_country_name:
        cols.insert(0, "country")

    df = panel_df[cols].copy()
    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)

    # Next-year target
    df["gdp_growth_next_year"] = df.groupby("country_code")["gdp_growth"].shift(-1)
    df["downturn_risk_next_year"] = (df["gdp_growth_next_year"] < 0).astype(int)

    base_features = [
        "unemployment",
        "inflation",
        "gdp_growth",
        "life_expectancy",
        "population_growth",
    ]

    # Lag features
    for col in base_features:
        df[f"{col}_lag1"] = df.groupby("country_code")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("country_code")[col].shift(2)

    # Delta features
    for col in base_features:
        df[f"{col}_change_1y"] = df[col] - df[f"{col}_lag1"]

    # Trend features over 3 years using least-squares slope
    for col in base_features:
        df = add_trend_feature(df, col, f"{col}_trend_3y")

    # Drop incomplete rows after lag/trend construction
    df = df.dropna().reset_index(drop=True)
    return df


def get_feature_columns() -> list[str]:
    return [
        # current levels
        "unemployment",
        "inflation",
        "gdp_growth",
        "life_expectancy",
        "population_growth",
        # lags
        "unemployment_lag1",
        "inflation_lag1",
        "gdp_growth_lag1",
        "life_expectancy_lag1",
        "population_growth_lag1",
        "unemployment_lag2",
        "inflation_lag2",
        "gdp_growth_lag2",
        "life_expectancy_lag2",
        "population_growth_lag2",
        # delta features
        "unemployment_change_1y",
        "inflation_change_1y",
        "gdp_growth_change_1y",
        "life_expectancy_change_1y",
        "population_growth_change_1y",
        # trend features
        "unemployment_trend_3y",
        "inflation_trend_3y",
        "gdp_growth_trend_3y",
        "life_expectancy_trend_3y",
        "population_growth_trend_3y",
    ]


def compute_optimal_threshold(y_true: pd.Series, y_score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    return float(thresholds[best_idx])


def evaluate_with_threshold(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if y_true.nunique() == 2 else np.nan,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    panel_df = load_panel()
    model_df = build_modelling_dataset(panel_df)
    feature_cols = get_feature_columns()

    X = model_df[feature_cols].copy()
    y = model_df["downturn_risk_next_year"].copy()

    meta_cols = ["country_code", "year"]
    if "country" in model_df.columns:
        meta_cols.insert(0, "country")

    meta = model_df[meta_cols].copy()
    eval_df = pd.concat([meta, X, y.rename("target")], axis=1)
    eval_df = eval_df.sort_values(["year", "country_code"]).reset_index(drop=True)

    X_sorted = eval_df[feature_cols]
    y_sorted = eval_df["target"]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    tscv = TimeSeriesSplit(n_splits=5)

    fold_rows = []
    all_scores = []
    all_targets = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_sorted), start=1):
        X_train, X_test = X_sorted.iloc[train_idx], X_sorted.iloc[test_idx]
        y_train, y_test = y_sorted.iloc[train_idx], y_sorted.iloc[test_idx]

        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]

        all_scores.extend(y_score.tolist())
        all_targets.extend(y_test.tolist())

        fold_metrics = evaluate_with_threshold(y_test, y_score, threshold=0.5)
        fold_rows.append(
            {
                "fold": fold,
                "accuracy": fold_metrics["accuracy"],
                "precision": fold_metrics["precision"],
                "recall": fold_metrics["recall"],
                "roc_auc": fold_metrics["roc_auc"],
                "confusion_matrix": json.dumps(fold_metrics["confusion_matrix"]),
            }
        )

    all_targets = pd.Series(all_targets)
    all_scores = np.array(all_scores)

    optimal_threshold = compute_optimal_threshold(all_targets, all_scores)
    overall_metrics = evaluate_with_threshold(
        all_targets,
        all_scores,
        threshold=optimal_threshold,
    )

    # Fit final model on all available data
    model.fit(X_sorted, y_sorted)

    # Save trained model
    joblib.dump(model, MODELS_DIR / "downturn_random_forest.joblib")

    # Save feature names
    pd.DataFrame({"feature": feature_cols}).to_csv(
        MODELS_DIR / "model_features.csv",
        index=False,
    )

    # Save CV metrics
    pd.DataFrame(fold_rows).to_csv(MODELS_DIR / "cv_metrics.csv", index=False)

    # Save empirical threshold
    pd.DataFrame(
        {
            "threshold_method": ["youden_j"],
            "optimal_threshold": [optimal_threshold],
            "accuracy": [overall_metrics["accuracy"]],
            "precision": [overall_metrics["precision"]],
            "recall": [overall_metrics["recall"]],
            "roc_auc": [overall_metrics["roc_auc"]],
            "confusion_matrix": [json.dumps(overall_metrics["confusion_matrix"])],
        }
    ).to_csv(MODELS_DIR / "optimal_threshold.csv", index=False)

    # Save latest country baselines for dashboard
    baseline_cols = ["country_code"]
    if "country" in model_df.columns:
        baseline_cols.insert(0, "country")
    baseline_cols.extend(feature_cols)

    country_baselines_latest = (
        model_df.sort_values(["country_code", "year"])
        .groupby("country_code", as_index=False)
        .tail(1)[baseline_cols]
        .reset_index(drop=True)
    )

    country_baselines_latest.to_csv(
        MODELS_DIR / "country_baselines_latest.csv",
        index=False,
    )

    print("Training complete.")
    print("Saved:")
    print("-", MODELS_DIR / "downturn_random_forest.joblib")
    print("-", MODELS_DIR / "model_features.csv")
    print("-", MODELS_DIR / "cv_metrics.csv")
    print("-", MODELS_DIR / "optimal_threshold.csv")
    print("-", MODELS_DIR / "country_baselines_latest.csv")
    print(f"Optimal threshold selected by Youden's J: {optimal_threshold:.4f}")


if __name__ == "__main__":
    main()