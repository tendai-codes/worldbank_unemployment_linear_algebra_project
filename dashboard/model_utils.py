from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd


def load_model(path: str | Path):
    path = Path(path)
    return joblib.load(path)


def load_country_baselines(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path)


def load_model_features(path: str | Path) -> list[str]:
    path = Path(path)
    df = pd.read_csv(path)

    # Accept either a column named "feature" or a single-column CSV
    if "feature" in df.columns:
        return df["feature"].dropna().astype(str).tolist()

    if df.shape[1] == 1:
        return df.iloc[:, 0].dropna().astype(str).tolist()

    raise ValueError(
        f"Could not determine feature column in {path}. "
        "Expected a column named 'feature' or a single-column CSV."
    )


def load_cv_metrics(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path)