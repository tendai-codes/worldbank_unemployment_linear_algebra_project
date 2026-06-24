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

ENGINEERED_PANEL_PATH = DATA_DIR / "worldbank_panel_engineered_rust.csv"
RAW_PANEL_PATH = DATA_DIR / "worldbank_panel_final.csv"
RUST_RUN_HINT = (
    "Run the Rust feature engineering pipeline first:\n"
    "  cd rust_pipeline\n"
    "  cargo run -- ../data/worldbank_panel_final.csv ../data/worldbank_panel_engineered_rust.csv"
)


def load_engineered_panel() -> pd.DataFrame:
    """Load the Rust-engineered modelling panel."""
    if ENGINEERED_PANEL_PATH.exists():
        return pd.read_csv(ENGINEERED_PANEL_PATH)

    raise FileNotFoundError(
        "Rust-engineered panel not found. Expected:\n"
        f"- {ENGINEERED_PANEL_PATH}\n\n"
        f"Source panel exists: {RAW_PANEL_PATH.exists()}\n\n"
        f"{RUST_RUN_HINT}"
    )


def get_feature_columns() -> list[str]:
    """Return the exact model feature columns produced by the Rust pipeline."""
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
        # annual change features
        "unemployment_change_1y",
        "inflation_change_1y",
        "gdp_growth_change_1y",
        "life_expectancy_change_1y",
        "population_growth_change_1y",
        # 3-year least-squares trend features
        "unemployment_trend_3y",
        "inflation_trend_3y",
        "gdp_growth_trend_3y",
        "life_expectancy_trend_3y",
        "population_growth_trend_3y",
    ]


def build_modelling_dataset(engineered_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare the Rust-engineered panel for model training."""
    feature_cols = get_feature_columns()
    target_col = "downturn_risk_next_year"

    required_cols = [
        "country_code",
        "year",
        target_col,
        *feature_cols,
    ]

    keep_country_name = "country" in engineered_df.columns
    if keep_country_name:
        required_cols.insert(0, "country")

    missing = [col for col in required_cols if col not in engineered_df.columns]
    if missing:
        raise ValueError(
            "Missing required Rust-engineered columns: "
            f"{missing}\n\n{RUST_RUN_HINT}"
        )

    model_df = engineered_df[required_cols].copy()
    model_df = model_df.sort_values(["country_code", "year"]).reset_index(drop=True)

    numeric_cols = ["year", target_col, *feature_cols]
    for col in numeric_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    model_df = model_df.dropna(subset=[target_col, *feature_cols]).reset_index(drop=True)
    model_df[target_col] = model_df[target_col].astype(int)

    if model_df.empty:
        raise ValueError("No complete modelling rows remain after loading the Rust-engineered panel.")

    return model_df


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


def save_country_feature_matrix(model_df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Save a country-level feature matrix for dashboard similarity search."""
    feature_matrix = (
        model_df.groupby("country_code")[feature_cols]
        .mean()
        .dropna()
        .sort_index()
    )
    feature_matrix.to_csv(DATA_DIR / "feature_matrix.csv")


def main():
    engineered_df = load_engineered_panel()
    model_df = build_modelling_dataset(engineered_df)
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

    # Fit final model on all available Rust-engineered data.
    model.fit(X_sorted, y_sorted)

    joblib.dump(model, MODELS_DIR / "downturn_random_forest.joblib")

    pd.DataFrame({"feature": feature_cols}).to_csv(
        MODELS_DIR / "model_features.csv",
        index=False,
    )

    pd.DataFrame(fold_rows).to_csv(MODELS_DIR / "cv_metrics.csv", index=False)

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

    save_country_feature_matrix(model_df, feature_cols)

    print("Training complete using Rust-engineered features.")
    print("Input:", ENGINEERED_PANEL_PATH)
    print("Saved:")
    print("-", MODELS_DIR / "downturn_random_forest.joblib")
    print("-", MODELS_DIR / "model_features.csv")
    print("-", MODELS_DIR / "cv_metrics.csv")
    print("-", MODELS_DIR / "optimal_threshold.csv")
    print("-", MODELS_DIR / "country_baselines_latest.csv")
    print("-", DATA_DIR / "feature_matrix.csv")
    print(f"Optimal threshold selected by Youden's J: {optimal_threshold:.4f}")


if __name__ == "__main__":
    main()
