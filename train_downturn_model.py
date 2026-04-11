from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit


BASE_FEATURES = [
    "unemployment",
    "inflation",
    "gdp_growth",
    "life_expectancy",
    "population_growth",
]

MODEL_FEATURES = [
    "unemployment",
    "inflation",
    "gdp_growth",
    "life_expectancy",
    "population_growth",
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
    "unemployment_change_1y",
    "inflation_change_1y",
    "gdp_growth_change_1y",
]


def build_modelling_dataset(panel_df: pd.DataFrame) -> pd.DataFrame:
    required = {"country_code", "year", *BASE_FEATURES}
    missing = required.difference(panel_df.columns)
    if missing:
        raise ValueError(f"panel_df is missing required columns: {sorted(missing)}")

    df = panel_df[["country_code", "year", *BASE_FEATURES]].copy()
    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)

    df["gdp_growth_next_year"] = df.groupby("country_code")["gdp_growth"].shift(-1)
    df["downturn_risk_next_year"] = (df["gdp_growth_next_year"] < 0).astype(int)

    for col in BASE_FEATURES:
        df[f"{col}_lag1"] = df.groupby("country_code")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("country_code")[col].shift(2)

    df["unemployment_change_1y"] = df["unemployment"] - df["unemployment_lag1"]
    df["inflation_change_1y"] = df["inflation"] - df["inflation_lag1"]
    df["gdp_growth_change_1y"] = df["gdp_growth"] - df["gdp_growth_lag1"]

    df = df.dropna().reset_index(drop=True)
    return df


def evaluate_model(df: pd.DataFrame, model: RandomForestClassifier) -> pd.DataFrame:
    eval_df = df.sort_values(["year", "country_code"]).reset_index(drop=True)
    X = eval_df[MODEL_FEATURES]
    y = eval_df["downturn_risk_next_year"]

    tscv = TimeSeriesSplit(n_splits=5)
    rows: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        rows.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_score) if y_test.nunique() == 2 else np.nan,
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            }
        )

    return pd.DataFrame(rows)


def train_final_model(df: pd.DataFrame) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(df[MODEL_FEATURES], df["downturn_risk_next_year"])
    return model


def save_artifacts(
    model: RandomForestClassifier,
    panel_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "downturn_random_forest.joblib")
    pd.Series(MODEL_FEATURES, name="feature_name").to_csv(out_dir / "model_features.csv", index=False)
    metrics_df.to_csv(out_dir / "cv_metrics.csv", index=False)

    baseline_latest = panel_df.sort_values(["country_code", "year"]).groupby("country_code").tail(1)
    baseline_latest.to_csv(out_dir / "country_baselines_latest.csv", index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_candidates = [
        project_root / "data" / "worldbank_panel_final.csv",
        project_root / "data" / "worldbank_panel.csv",
    ]

    panel_path = next((p for p in data_candidates if p.exists()), None)
    if panel_path is None:
        raise FileNotFoundError(
            "Could not find a panel CSV. Save your cleaned panel as data/worldbank_panel_final.csv "
            "or data/worldbank_panel.csv and rerun."
        )

    panel_df = pd.read_csv(panel_path)
    modelling_df = build_modelling_dataset(panel_df)

    eval_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    metrics_df = evaluate_model(modelling_df, eval_model)
    final_model = train_final_model(modelling_df)

    save_artifacts(final_model, panel_df, metrics_df, project_root / "models")

    print("Saved model artifacts to:", project_root / "models")
    print("\nCross-validation summary:")
    print(metrics_df[["accuracy", "precision", "recall", "roc_auc"]].mean().round(3))


if __name__ == "__main__":
    main()
