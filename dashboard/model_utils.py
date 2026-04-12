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

    if "feature" in df.columns:
        return df["feature"].dropna().astype(str).tolist()

    if df.shape[1] == 1:
        return df.iloc[:, 0].dropna().astype(str).tolist()

    raise ValueError(
        f"Could not determine feature column in {path}. "
        "Expected a column named 'feature' or a single-column CSV."
    )


def load_feature_matrix(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path, index_col=0)


def load_optimal_threshold(path: str | Path, default: float = 0.35) -> float:
    path = Path(path)
    if not path.exists():
        return default

    df = pd.read_csv(path)
    if df.empty or "optimal_threshold" not in df.columns:
        return default

    return float(df.loc[0, "optimal_threshold"])